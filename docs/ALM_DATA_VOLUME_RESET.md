# /data 数据卷清理方案（下一轮评测前）

- 日期：2026-10-18
- 适用对象：生产服务器 `38.175.195.27` 上的 ALM 数据卷 `almserver_alm-data`
- 目标：清空上一轮全部空间，建立干净的评测基线；避免旧 `mode.yaml`（尤其 161 个僵尸空间）影响新一跑

---

## 一、现状

| 项 | 数值 |
| --- | --- |
| 数据卷 | `almserver_alm-data`（挂载到容器 `/data`） |
| 空间数 | 3,026 个目录 |
| 体积 | 约 8.0 GB |
| 宿主机磁盘余量 | 8.2 GB |

**关键约束**：磁盘余量不足以再复制一份数据卷（8.0 GB ≈ 全部剩余空间）。
因此本方案**不做整卷备份**，只保留元数据清单与少量抽样。

---

## 二、删除前的确认清单

执行前逐项确认，任一为「否」则停止：

1. 上一轮评测成绩已由平台出具，且**不需要再用旧数据复查**；
2. 旧数据的分析结论已归档（本仓库 `docs/ALM_BACKFILL_FIX_20261018.md`）；
3. `.env.alm` 已备份（部署流程会自动生成 `.env.alm.bak_deploy_*`）；
4. 当前没有正在进行的 Add / Search（平台侧已停止调用）。

> ⚠️ 删除数据卷**不可回滚**。如果上述任一项不满足，改用第四节「保留式清理」。

---

## 三、标准清理流程（停服 → 删除 → 重建 → 验收）

### 1. 停服（必须先停）

`engine.close()` 会把内存中驻留的空间重新落盘；运行期删文件会被它写回来。

```bash
cd /AlmServer
docker compose -p almserver -f docker-compose.alm.yml stop ariadne-alm
```

### 2. 记录删除前清单

```bash
docker run --rm -v almserver_alm-data:/data alpine \
  sh -c 'ls /data | wc -l; du -sh /data'
```

建议把输出保存到 `/AlmServer/backup/volume_manifest_$(date +%Y%m%d).txt`：

```bash
docker run --rm -v almserver_alm-data:/data alpine sh -c 'ls /data' \
  > /AlmServer/backup/volume_manifest_$(date +%Y%m%d).txt
```

### 3. 删除数据卷

```bash
docker volume rm almserver_alm-data
```

如果提示 `volume is in use`：说明还有容器挂载它（通常是旧的 `ariadne-alm` 或某个 `run --rm` 残留）。

```bash
docker ps -a --filter volume=almserver_alm-data
docker rm -f <残留容器>
docker volume rm almserver_alm-data
```

### 4. 重新启动（自动创建空卷）

```bash
cd /AlmServer
docker compose -p almserver -f docker-compose.alm.yml up -d
```

### 5. 验收

```bash
# 容器健康
docker ps --filter name=ariadne-alm --format '{{.Status}}'
curl -s http://127.0.0.1:8770/health

# 卷为空
docker run --rm -v almserver_alm-data:/data alpine sh -c 'ls -A /data | wc -l'   # 期望 0
```

再跑一次 smoke（会写入 1~2 个测试空间，**评测前需再次清空**）：

```bash
python -m alm.smoke --base-url http://127.0.0.1:8770 --api-key <KEY>
```

### 6. 评测前的最后一步

若第 5 步跑过 smoke，按第 1~4 步**再清一次**，确保评测从 `0` 个空间开始。

---

## 四、保留式清理（可选，磁盘不足时的折中）

如果只想清掉「旧赛道 + 僵尸空间」，但保留本轮新赛道数据，可按基线名单删除：

```bash
# 1) 备份名单（宿主机上已有 /tmp/alm_baseline_prefull_20261017.txt）
# 2) 停服
docker compose -p almserver -f docker-compose.alm.yml stop ariadne-alm

# 3) 按名单删除（在容器内执行，避免宿主机权限问题）
docker run --rm -v almserver_alm-data:/data -v /tmp:/hosttmp alpine sh -c '
  while read d; do [ -n "$d" ] && rm -rf "/data/$d"; done < /hosttmp/alm_baseline_prefull_20261017.txt
  ls /data | wc -l
'

# 4) 启动
docker compose -p almserver -f docker-compose.alm.yml up -d
```

> 该方式保留 1,518 个本轮新空间，但**它们已冻结了旧配置下的 `mode.yaml`**，
> 与新参数（1 题判定、10 分钟等待）不一致。若要做干净的对照，仍建议整卷重置。

---

## 五、磁盘回收

删除卷后，空间由 Docker 回收，但以下内容**不会自动释放**：

| 对象 | 处理方式 |
| --- | --- |
| 旧镜像 `v0.1.5-alm`（723MB） | 确认不再回滚后 `docker rmi ariadne-alm:v0.1.5-alm` |
| 回滚镜像 `v0.2.0-alm.prev_20261018`（725MB） | **保留到新版本验证通过**，之后再删 |
| 构建缓存 | `docker builder prune -f`（约可回收 0.8GB） |
| `alm/tools/alm_data_6AjBT.tar.gz`（本地 161MB） | 与本服务器无关；本地自行处理 |

预期回收：删除卷 ≈ 8.0GB + 构建缓存 ≈ 0.8GB → 磁盘余量回到 15GB 以上。

---

## 六、回滚说明

| 场景 | 操作 |
| --- | --- |
| 新镜像启动异常 | `docker tag ariadne-alm:v0.2.0-alm.prev_20261018 ariadne-alm:v0.2.0-alm`，然后 `docker compose ... up -d --force-recreate` |
| 新参数行为异常 | 恢复 `/AlmServer/.env.alm.bak_deploy_<时间戳>` 后 recreate |
| nginx 超时改动异常 | 恢复 `/www/server/panel/vhost/nginx/alm.ariadne.top.conf.bak_<时间戳>` 后 `nginx -t && nginx -s reload` |
| 数据卷已删除 | **不可回滚**——只能依赖外部备份（本方案未做） |

---

## 七、执行时机建议

1. 新镜像部署并 smoke 通过；
2. 确认平台已停止调用；
3. **评测开始前**执行第三、四节；
4. 清理后立即开始新一跑，避免空窗期内产生脏数据。
