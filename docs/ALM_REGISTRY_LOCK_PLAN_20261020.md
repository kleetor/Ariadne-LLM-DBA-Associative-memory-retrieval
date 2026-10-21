# ALM 空间构造锁重构计划

- 日期：2026-10-20
- 范围：`alm/engine.py` 的 user_id 空间注册表（仅此一处）
- 目标：把可能长达数分钟的 `MemorySpace` 构造（整图重建 / 文档区重嵌）移出全局
  `_registry_lock`，消除「一个用户冷启动，所有用户排队」的可用性问题
- 约束：**不改动** `alm/space.py`、HTTP 契约、落盘格式、MCP 链路、检索/回补行为

---

## 1. 现状链路分析

### 1.1 调用链

```
/add  ─┐
/search ─┴─> ALMEngine.add()/search()
              └─ _lease(user_id, create)
                   └─ with _registry_lock:                     ← 全局锁
                        _get_or_create_locked(user_id, create)
                          └─ MemorySpace.__init__()            ← 可能在这里 embedding
                        _in_use[user_id] += 1
                   yield space                                  ← 锁已释放
                   finally: with _registry_lock: _evict_locked()
```

`get_space()`（只读排查入口）同样在 `_registry_lock` 内构造。

### 1.2 慢点

`MemorySpace.__init__()` 中有三段与数据规模成正比的工作：

| 工作 | 触发条件 | 代价 |
| --- | --- | --- |
| `_reindex()`（[space.py#L613-L627](file:///c:/Users/makot/Desktop/Ariadne/alm/space.py#L613-L627)） | 图索引缺失 / 指纹不符 | 逐节点 embedding |
| 文档区全量重嵌（[space.py#L1554-L1562](file:///c:/Users/makot/Desktop/Ariadne/alm/space.py#L1554-L1562)） | `docs.index` 缺失 / 指纹不符 | 逐块 embedding |
| 补齐缺失尾部（[space.py#L1531-L1541](file:///c:/Users/makot/Desktop/Ariadne/alm/space.py#L1531-L1541)） | 索引落后于条文 | 差额 embedding |

此外还有 `_load_documents()` 的条文回放（`docs.log` 可达数 MB，纯磁盘 I/O）。

### 1.3 影响面

`_registry_lock` 是**进程级全局锁**。它被持有期间：

- 其它 user 的 `/add`、`/search` 全部阻塞（两者都走 `_lease`）；
- `space_count()`、`get_space()`、`close()` 一并阻塞；
- 生产 Add 并发 16–64，锁持续数十秒即可能触发调用方超时 / 5xx / 重试风暴。

这不是数据正确性问题，而是**可用性/尾延迟**问题；但对「升级后首次加载 + 存量文档空间」
几乎必现（见 §2.6）。

---

## 2. 必须保持的不变量

重构的改变点只有一个：**构造不持全局锁**。以下不变量必须逐条成立。

| 编号 | 不变量 | 保护手段 |
| --- | --- | --- |
| C1 | 同一 user_id 在任一时刻最多只有一个**对外可见**的实例 | `_constructing` 占位 + 注册表双检 |
| C2 | 同一实例被使用时不会被 LRU 换出 | `_in_use` 与「取到实例」在同一临界区递增 |
| C3 | 构造失败不污染注册表，后续可重试 | 异常路径清理占位并唤醒等待者 |
| C4 | 已有实例的请求不被冷构造阻塞 | 构造在 `_registry_lock` 之外 |
| C5 | 后台回补中的空间不被换出 | 保持 `bg_active` + `_evict_locked` 原逻辑不动 |
| C6 | `create=False` 且无图谱时返回 `None` | 保持 `graph.yaml` 存在性判定 |
| C7 | 锁序固定、无死锁 | 只允许 `Condition(_registry_lock)` 内做状态变更；构造时**不持有任何锁** |
| C8 | 冷启动并发不无限膨胀 | 构造并发上限（信号量，见 §3.3） |

---

## 3. 方案设计

### 3.1 选型对比

| 方案 | 优点 | 问题 | 结论 |
| --- | --- | --- | --- |
| A. 全局构造锁 + 锁外构造 | 改动最小 | 不同 user 的冷构造仍串行；启动风暴时新用户排队 | 备选 |
| B. per-user 锁字典 | 同 user 串行、不同 user 并行 | 字典随 user 数增长，需要引用计数清理，易泄漏 | 不选 |
| C. `Condition(_registry_lock)` + `_constructing` 集合 + 构造并发信号量 | 同 user 串行、不同 user 并行、无锁字典泄漏、资源有界 | 需要一次小幅重写（约 60 行） | **采用** |

### 3.2 状态与数据结构（仅 `ALMEngine.__init__`）

```python
# 正在构造的 user_id（仅在持有 _registry_lock 时读写）
self._constructing: set[str] = set()
# 与 _registry_lock 绑定：构造完成/失败后唤醒同一 user 的等待者
self._space_cv = threading.Condition(self._registry_lock)

# 构造并发上限：不同 user 可以并行冷启动，但同一时刻最多 N 个
# （避免 64 个新 user 同时重建大索引把内存/CPU 打满）
_MAX_PARALLEL_CONSTRUCTIONS = 4
self._construct_slots = threading.BoundedSemaphore(_MAX_PARALLEL_CONSTRUCTIONS)
```

> 说明：`_construct_slots` 只在**构造期间**持有；等待槽位时不持任何锁。

### 3.3 新入口：`_acquire_space(user_id, create, *, lease)`

替换 `_get_or_create_locked()`（后者删除，避免被误用）。

```python
def _acquire_space(self, user_id: str, create: bool, *, lease: bool) -> Optional[MemorySpace]:
    """取得（必要时锁外构造）注册表实例。

    lease=True 时在同一临界区内递增 _in_use（C2）；构造期间不持有任何锁（C4）。
    """
    with self._space_cv:
        while True:
            space = self._spaces.get(user_id)
            if space is not None:
                self._spaces.move_to_end(user_id)
                if lease:
                    self._in_use[user_id] = self._in_use.get(user_id, 0) + 1
                return space

            if user_id in self._constructing:
                # 同一 user 已有人在构造：等它结束，再统一按注册表状态判定
                self._space_cv.wait()
                continue

            if not create and not self._graph_path(user_id).exists():
                return None

            self._constructing.add(user_id)
            break

    # ---- 锁外构造（本项唯一的实质改变）----
    try:
        with self._construct_slots:
            space = MemorySpace(
                user_id=user_id,
                config=self.config,
                embeddings=self.embeddings,
                llm=self.llm,
                yaml_path=self._graph_path(user_id),
                stats=self.retrieval_stats,
                reranker=self.reranker,
                backfill_gate=self._backfill_gate,
            )
    except BaseException:
        with self._space_cv:
            self._constructing.discard(user_id)
            self._space_cv.notify_all()
        raise

    with self._space_cv:
        self._constructing.discard(user_id)
        existing = self._spaces.get(user_id)
        # 防御分支：正常路径下同一 user 的构造已被 _constructing 串行化，
        # 这里只保证任何未来新增路径都不会产生第二个对外实例（C1）。
        space = existing if existing is not None else space
        self._spaces[user_id] = space
        self._spaces.move_to_end(user_id)
        if lease:
            self._in_use[user_id] = self._in_use.get(user_id, 0) + 1
        self._space_cv.notify_all()
        return space
```

### 3.4 `_lease` 与 `get_space` 改造

```python
@contextmanager
def _lease(self, user_id: str, create: bool):
    space = self._acquire_space(user_id, create, lease=True)
    if space is None:
        yield None
        return
    try:
        yield space
    finally:
        with self._registry_lock:
            remaining = self._in_use.get(user_id, 1) - 1
            if remaining > 0:
                self._in_use[user_id] = remaining
            else:
                self._in_use.pop(user_id, None)
            self._evict_locked()

def get_space(self, user_id: str, create: bool = True) -> Optional[MemorySpace]:
    space = self._acquire_space(user_id, create, lease=False)
    with self._registry_lock:
        self._evict_locked()
    return space
```

`get_space` 的语义保持原样：返回对象**不享受换出保护**，仅供只读排查。

### 3.5 关键时序推演

**时序 1：用户 U 冷构造（2 分钟），用户 V 已有驻留空间**

```
T1: _lease(U) → _constructing={U} → 释放全局锁 → 构造中……
T2: _lease(V) → 注册表命中 V → 递增 _in_use(V) → 立刻返回  ✅ 不再等 U
T3: U 构造完成 → 加锁插入 → 递增 _in_use(U) → 返回
```

**时序 2：同一 user 两个并发首次请求**

```
T1: U 不在表、不在构造 → _constructing={U} → 构造
T2: U 不在表、在构造 → wait（释放锁）
T1: 插入 U + notify → T2 醒来 → 命中表 → 同一实例返回  ✅ 只构造一次
```

**时序 3：构造抛异常**

```
T1: 构造异常 → 清理 _constructing + notify → 异常上抛（/add 转 5xx）
T2: 醒来 → U 不在表、无占位 → 自己成为构造者重试  ✅ 不残留脏状态
```

**时序 4：`create=False` 且正在构造**

```
T1: _lease(U, create=True) 正在构造
T2: _lease(U, create=False) → 发现 U 在构造 → wait（而不是直接返回 None）
```

> 这是有意的：`create=True` 的实例在构造完成时已插入注册表，`create=False` 应等待并
> 命中它，而不是把「正在建立的空间」误判成「不存在」。

**时序 5：LRU 换出与换出竞态**

- 构造完成前实例不在注册表 → 不可能被换出；
- 插入与 `_in_use += 1` 在**同一个**临界区完成（C2），中间没有可被换出的窗口；
- 换出逻辑本身（`_evict_locked`）一行不改。

---

## 4. 改动边界（确保不涉及其他）

### 4.1 会修改的文件

| 文件 | 改动 |
| --- | --- |
| `alm/engine.py` | 新增 `_constructing` / `_space_cv` / `_construct_slots`；新增 `_acquire_space`；重写 `_lease`、`get_space`；删除 `_get_or_create_locked` |
| `tests/test_alm_engine_registry.py` | **新增**并发/失败/幂等构造测试（见 §5） |

### 4.2 明确不改

- `alm/space.py`：构造内容、索引指纹、回补、锁全部不动；
- `alm/server.py`、`alm/contract.py`：HTTP 契约与线程池调用不动；
- `alm/config.py`：不新增配置项（并发上限用模块常量）；
- `dba_pipeline/*`：MCP 链路完全不动；
- 落盘格式 / 目录布局 / 接口返回：完全不动；
- `_evict_locked` 与 `bg_active`：一行不改。

### 4.3 行为差异（预期内）

- 冷启动时，不同 user 的空间可以并行构造（最多 4 个）；
- 已有驻留 user 的请求不再被冷构造阻塞；
- 同一 user 的构造仍严格串行；
- 若构造与关闭（`close()`）并发，仍可能发生在关闭后插入注册表的情况——这是**既有**
  的关停竞态，本次不扩大也不修复。

---

## 5. 测试计划

### 5.1 新增 `tests/test_alm_engine_registry.py`

用 `object.__new__(ALMEngine)` + 假 `MemorySpace`（monkeypatch `alm.engine.MemorySpace`）
只测注册表逻辑，不触碰真实 embedding/LLM。

| 用例 | 断言 |
| --- | --- |
| `test_slow_construction_does_not_block_resident_user` | 慢构造 U（Event 阻塞）期间，`_lease(V)` 必须在 100ms 内返回；释放 Event 后 U 正常完成 |
| `test_same_user_concurrent_lease_constructs_once` | 两线程同时 `_lease(U, True)`，构造计数 == 1，两者拿到**同一对象** |
| `test_construction_failure_is_retryable` | 首次构造抛异常 → `_lease` 抛错、`_constructing` 为空；第二次成功 |
| `test_create_false_waits_for_inflight_construction` | U 构造中，`_lease(U, False)` 等待并命中最终实例，而非返回 `None` |
| `test_get_space_does_not_hold_registry_lock` | 慢构造期间 `space_count()`（只取全局锁）必须立刻返回 |
| `test_lease_evicts_only_after_release` | 租用期间 `_evict_locked` 不换出该空间；释放后按 LRU 换出 |

### 5.2 现有回归

```bash
python -m pytest tests -q          # 当前基线：263 passed
```

重点确认不破坏：

- `test_engine_eviction_skips_space_with_background_task`（直接调 `_evict_locked`）
- ALM e2e 全链路（`test_alm_pipeline_e2e.py`）
- `test_concurrency.py`

---

## 6. 验收标准

1. 人为把构造拖到 2s：驻留 user 的 `_lease` 延迟 < 100ms（改造前为 ~2s）。
2. 同一 user 并发 10 次首次访问：构造次数 == 1。
3. 构造失败后可立即重试成功，`_constructing` 不残留。
4. `pytest tests -q` 全绿（≥263）。
5. `git diff --stat` 只包含 `alm/engine.py` 与新测试文件。
6. 压测观测：`/add`、`/search` 的 p99 在冷启动阶段不再出现与「全局构造」同步的尖峰。

---

## 7. 风险与回滚

| 风险 | 概率 | 处置 |
| --- | --- | --- |
| 条件变量使用错误导致同 user 重复构造 | 低 | 用例 2 钉住；插入前双检 |
| `_in_use` 泄漏（异常路径未递减） | 低 | 递减仍在 `finally`，与原逻辑一致 |
| 并行构造导致瞬时内存升高 | 中 | `BoundedSemaphore(4)` 限流；必要时调小常量 |
| 关停期间构造插入注册表 | 低 | 既有竞态，本次不改变；进程退出即回收 |
| 行为回归 | 低 | 全量测试 + 单文件回滚 |

回滚方式：`git revert` 本次提交即可；无数据迁移、无格式变更。

---

## 8. 实施步骤（建议单次提交）

1. 在 `ALMEngine.__init__` 增加 `_constructing` / `_space_cv` / `_construct_slots`。
2. 新增 `_acquire_space()`；删除 `_get_or_create_locked()`。
3. 重写 `_lease()` 与 `get_space()`（仅替换取实例的入口，锁与换出逻辑不动）。
4. 新增 `tests/test_alm_engine_registry.py`（6 个用例）。
5. 运行 `pytest tests -q`，确认 263+ 全绿。
6. 代码审查重点：C1–C8 逐条对照；确认无任何文件落在 `alm/engine.py` 之外
   （测试文件除外）。

---

## 9. 后续可选（不在本次范围）

- 为 `/add` 增加构造耗时日志（开始/结束 + user 掩码），便于运行期观测冷启动成本；
- 若冷启动成为常态瓶颈，可把 `MemorySpace.__init__` 的索引重建改为后台线程（会改变
  首次检索的语义，需单独评估）；
- 为 `close()` 增加 `_closed` 标志，拒绝关停后的构造插入。
- **MCP 查询锁粒度**：当前 `query_memory` / `run_chain_trace` / `temporal_lookup` 在整段
  `retrieve_with_story`（含目的推断与 StoryRank 两次 LLM）期间持图锁，并发查询会串行。
  若后续吞吐成为瓶颈，可把锁收窄到纯图遍历段，或引入读写锁。
- **切图与 flush 的收尾竞态**：`switch_graph` 已在 `load_into` 前复查
  `is_flushing` / `buffer_size`，但仍存在「检查通过后、切图前」的微秒级窗口；
  彻底消除需要调度器支持「切图暂停」。

---

## 10. 实施记录（2026-10-20）

已按本计划完成实现，与设计无偏差。

**改动文件**

- `alm/engine.py`
  - 新增 `_MAX_PARALLEL_CONSTRUCTIONS = 4`；
  - `__init__` 新增 `_constructing` / `_space_cv` / `_construct_slots`；
  - 新增 `_acquire_space(user_id, create, *, lease)`，删除 `_get_or_create_locked()`；
  - `_lease()` / `get_space()` 改为走 `_acquire_space()`，租约与换出逻辑保持不变。
- `tests/test_alm_engine_registry.py`（新增，6 个用例）
  - 慢构造不阻塞驻留 user；
  - 同 user 并发只构造一次；
  - 构造失败可重试、占位清理；
  - `create=False` 等待在途构造；
  - `get_space` / `space_count` 不被构造阻塞；
  - 租用中不换出、释放后按 LRU 换出。

**验证结果**

- 新增用例：注册表 `6 passed`；回补断点回归 `1 passed`；evidence 补位回归 `1 passed`
- 全量回归：**`271 passed`**（原基线 263 + 新增 9）
- 本次改动范围：仅 `alm/engine.py` 与新增测试文件（外加本计划文档）。
  工作区里另有 2026-10-20 早前一轮的 ALM/MCP 缺陷修复，不属于本重构。

**复审记录（2026-10-20，改动后复查）**

| 发现 | 处理 |
| --- | --- |
| evidence 档在边被截断后返回条数少于 `top_k`（A4 修复的副作用） | 边过滤后用尾部节点补满预算，并加回归用例 |
| `dba_checkpoint` 整图序列化未持图锁（B2 并行化后更易撞上半张图） | `save_checkpoint` / `to_dict()` 包进 `_lock` |
| `switch_graph` 等待结束后仍可能被新 flush 插队 | `load_into` 前复查 `is_flushing` + `buffer_size` |
| `_acquire_space` 的 `graph_path` 在循环内赋值 | 移到循环外，避免后续改动引入未绑定变量 |

**部署状态（2026-10-20 确认）**

- 远端容器：`ariadne-alm`（`ariadne-alm:v0.2.0-alm`，镜像构建 2026-10-19 06:55 UTC），
  `Up 2 days (healthy)`；
- 代码标记：容器内 `hasattr(ALMEngine, "_acquire_space") == False` → **本重构尚未部署**；
- 数据卷：`almserver_alm-data` 为 3,026 个空间 / 8.0 GB，清理尚未执行；
- 资源：Mem 7.8 GiB（available 6.5 GiB），`ariadne-alm` 空闲占用 104 MiB → 4 个并行构造
  仍有余量；磁盘余量 7.5 GiB（81%），清理前无法做整卷备份；
- 结论：只做了状态确认，未部署、未清理。部署需重建镜像（`docker compose ... up -d --build`），
  并按回补治理文档第六节同步 `.env.alm` 与 nginx 超时。

