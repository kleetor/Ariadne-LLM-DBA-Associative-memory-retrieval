/* SPDX-License-Identifier: AGPL-3.0-only */
/* 应用引导：装配 3D 图谱与各面板 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});
  const U = A.util;

  const data = { nodes: [], links: [] };
  let selectedId = null;
  let pollTimer = null;

  function setStatus(id, value) {
    const e = document.getElementById(id);
    if (e) e.textContent = value;
  }

  // 重载序号：轮询（15s）与用户 CRUD 可能同时在途，只应用最后一次的结果，
  // 否则先发的旧响应会覆盖后到的新数据（刚建的节点会「消失」到下一轮）。
  let reloadSeq = 0;
  // 3D 库加载成功前不去碰 A.graph：否则 setData/select 会在 null 上抛错
  let graphReady = false;

  async function reloadGraph(keepId) {
    const seq = ++reloadSeq;
    try {
      const payload = await A.api.getGraph();
      if (seq !== reloadSeq) return;
      data.nodes = payload.nodes || [];
      data.links = payload.links || [];
      updateTopbarStats();
      if (!graphReady) return;

      A.graph.setData({ nodes: data.nodes, links: data.links });
      if (keepId && data.nodes.some(function (n) { return n.id === keepId; })) {
        A.graph.select(keepId);
        // 用用户当前的跳数还原聚焦，而不是硬编码 1（否则 2 跳聚焦会被悄悄降级）
        A.graph.focus(keepId, A.graph.getFocusHops());
      } else {
        A.graph.select(null);
      }
    } catch (e) {
      if (seq !== reloadSeq) return;
      U.toast('加载图数据失败：' + e.message, 'error');
      setStatus('stat-sel', '-');
    }
  }

  function updateTopbarStats() {
    setStatus('topbar-nodes', data.nodes.length);
    setStatus('topbar-edges', data.links.length);
  }

  // 3D 库来自 unpkg CDN，取不到时 init() 会抛错。此处显式先检测一次，
  // 好给出可读原因，而不是让用户在控制台看到 "ForceGraph3D is not defined"。
  function libraryMissing() {
    return typeof ForceGraph3D === 'undefined' || typeof THREE === 'undefined';
  }

  function graphErrorMessage() {
    return '3D 渲染库未能加载。three.js 与 3d-force-graph 由 unpkg CDN 提供，'
      + '请检查网络或代理是否可访问 unpkg.com；离线环境请改用 `python -m '
      + 'dba_pipeline.viz.renderer` 生成的页面。';
  }

  function showGraphError(message) {
    const host = document.getElementById('graph-container');
    if (!host) return;
    host.textContent = '';
    host.appendChild(U.el('div', { class: 'graph-error' }, [
      U.el('div', { class: 'graph-error-title', text: '3D 图谱无法启动' }),
      U.el('div', { class: 'graph-error-msg', text: message }),
    ]));
  }

  function initGraph() {
    A.graph.init({
      container: document.getElementById('graph-container'),
      settings: A.store.settings,
      hooks: {
        onSelect: function (node) {
          selectedId = node ? node.id : null;
          setStatus('stat-sel', node ? node.id : '-');
          A.panels.renderInspector(node);
        },
        onStats: function (stats) {
          A.panels.updateStatusbar(stats);
          updateEmptyState(stats);
        },
        onHover: function (pos, node) { A.panels.showTooltip(pos, node); },
        onEdgeHover: function (pos, info) { A.panels.showEdgeTooltip(pos, info); },
        onContextMenu: function (node, evt) { A.panels.showContextMenu(node, evt); },
        onEdgeAction: function (action) { A.panels.handleEdgeAction(action); },
        onEdgeModeChange: function (mode) {
          if (!mode) A.panels.hideEdgeBanner();
          document.body.classList.toggle('edge-mode', !!mode);
        },
      },
    });
    A.graph.applySettings(A.store.settings);
    graphReady = true;
  }

  function startFps() {
    let frames = 0, last = performance.now();
    (function loop() {
      frames++;
      const now = performance.now();
      if (now - last >= 1000) {
        setStatus('stat-fps', frames);
        frames = 0;
        last = now;
      }
      requestAnimationFrame(loop);
    })();
  }

  // ---- 快捷键 ----
  // 设计口径（参考 WCAG 2.1.4 与常见 Web 应用做法）：
  //   1) 焦点在输入控件内一律不拦截，避免抢走打字；
  //   2) 单字符快捷键可整体关闭（设置 → 交互与快捷键），Ctrl 组合键不受影响；
  //   3) 不用浏览器保留组合（Ctrl+W/T/N/L 等），只用单字母与 Ctrl+Z；
  //   4) 帮助（?）始终可用，作为快捷键的可发现入口。

  function toggleAutoRotate() {
    const next = !A.store.settings.render.autoRotate;
    A.store.update('render', 'autoRotate', next);
    A.graph.applySettings(A.store.settings);
    const btn = document.getElementById('btn-autorotate');
    if (btn) btn.classList.toggle('active', next);
    U.toast(next ? '自动旋转：开' : '自动旋转：关', 'ok');
  }

  // factor > 1 拉远，< 1 拉近
  function zoomBy(factor) {
    const st = A.graph.cameraState();
    if (st) A.graph.setCameraDistance(st.distance * factor);
  }

  function focusSelected(hops) {
    if (!selectedId) { U.toast('先选中一个节点再聚焦', 'warn'); return; }
    A.graph.focus(selectedId, hops);
  }

  function deleteSelected() {
    if (!selectedId) { U.toast('先选中一个节点再删除', 'warn'); return; }
    if (!ensureEditable()) return;
    A.panels.commands.deleteNode({ id: selectedId });
  }

  // 离线导出包只读：编辑类动作统一给出同一句提示，避免落到一个注定失败的请求
  function ensureEditable() {
    if (A.api.isLive) return true;
    U.toast('离线 HTML 模式只读，编辑需要连接面板服务', 'warn');
    return false;
  }

  // 空状态：过滤后一个节点都不剩时，画布不该只剩一片空白
  function updateEmptyState(stats) {
    const box = document.getElementById('graph-empty');
    if (!box) return;
    if (stats.visibleNodes > 0) { box.hidden = true; return; }
    const title = document.getElementById('empty-title');
    const desc = document.getElementById('empty-desc');
    const f = A.graph.getFilter();
    if (!stats.totalNodes) {
      title.textContent = '当前图谱是空的';
      desc.textContent = '用 MCP 的 dba_add_conversation 写入对话，或用 dba_add_note 写入一条笔记。';
    } else if (f.showNotesOnly && !stats.notes) {
      // 判断「图里真的没有笔记」而不是「当前没显示笔记」：后者可能只是类型过滤挡掉了
      title.textContent = '当前图谱没有笔记';
      desc.textContent = '笔记是 channel=note 的普通节点，通过 MCP 的 dba_add_note 写入；也可以先切回全部节点。';
    } else {
      title.textContent = '当前过滤条件下没有节点';
      desc.textContent = '放宽节点类型 / 边类型过滤，或点击下方按钮恢复默认。';
    }
    box.hidden = false;
  }

  // 返回 true 表示这次按键已被消费（调用方应 preventDefault）
  function handleShortcut(e) {
    // Ctrl/Cmd+Z 撤销：不受单键开关限制
    if ((e.ctrlKey || e.metaKey) && String(e.key).toLowerCase() === 'z' && !e.shiftKey) {
      e.preventDefault();
      A.panels.undo();
      return true;
    }
    // 其余组合键交还给浏览器（Ctrl+K 等留给外部工具）
    if (e.ctrlKey || e.metaKey || e.altKey) return false;

    // 帮助：始终可用，否则用户没法知道快捷键被关掉了
    if (e.key === '?') { A.panels.commands.help(); return true; }

    // 模态框打开时其余快捷键不生效，避免误触到底层画布/编辑动作
    if (document.querySelector('.modal-overlay')) return false;

    const ui = A.store.settings.ui || {};
    if (ui.singleKeyShortcuts === false) return false;
    if (!graphReady) return false;

    const input = document.getElementById('search-input');

    switch (e.key) {
      // 视图切换
      case 'g': case 'G': A.panels.shell.closeDrawer(); return true;
      case 'o': case 'O': A.panels.shell.openDrawer('observability'); return true;
      case 'p': case 'P': A.panels.shell.openDrawer('params'); return true;
      case 's': case 'S': A.panels.shell.openDrawer('settings'); return true;

      // 画布
      case 'r': case 'R': A.graph.zoomFit(); return true;
      case 'f': case 'F': focusSelected(e.shiftKey ? 2 : 1); return true;
      case 'c': case 'C': A.graph.exitFocus(); A.graph.select(null); return true;
      case 'a': case 'A': toggleAutoRotate(); return true;
      case '+': case '=': zoomBy(0.8); return true;
      case '-': case '_': zoomBy(1.25); return true;
      case '1': A.graph.setCamera(0, 0, 800); return true;
      case '2': A.graph.setCamera(800, 0, 0); return true;
      case '3': A.graph.setCamera(0, 800, 0); return true;

      // 检索与编辑
      case '/':
        if (input) input.focus();
        return true;
      case 'm': case 'M': A.panels.commands.toggleSearchMode(); return true;
      case 'n': case 'N': if (ensureEditable()) A.panels.commands.newNode(); return true;
      case 'e': case 'E': if (ensureEditable()) A.panels.commands.newEdge(); return true;
      case 'Delete': case 'Backspace': deleteSelected(); return true;
      default: return false;
    }
  }

  function bindKeys() {
    document.addEventListener('keydown', function (e) {
      const target = e.target || {};
      const tag = (target.tagName || '').toLowerCase();
      // 输入控件与可编辑区域不拦截（含设置页的数字/文本输入）
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable) return;
      // 已有浮层时只放行 Esc（由 panels.js 统一处理）
      if (e.key === 'Escape') return;
      if (handleShortcut(e)) e.preventDefault();
    });
  }

  async function pollExternal() {
    if (!A.api.isLive || !A.api.isOnline()) return;
    try {
      const metrics = await A.api.getMetrics();
      const g = metrics.graph || {};
      if (g.nodes !== data.nodes.length || g.edges !== data.links.length) {
        await reloadGraph(selectedId);
      }
    } catch (e) { /* 静默：下一轮重试 */ }
  }

  async function bootstrap() {
    document.body.setAttribute('data-tab', 'graph');
    if (libraryMissing()) {
      showGraphError(graphErrorMessage());
    } else {
      try {
        initGraph();
      } catch (e) {
        // 3D 初始化失败（库异常 / WebGL 不可用）也要把面板拉起来：
        // 可观测、参数与数据操作仍然可用，不能因为 3D 挂了就整页静默空白。
        showGraphError(e.message || String(e));
      }
    }
    await reloadGraph(null);
    A.panels.init({ data: data, reloadGraph: reloadGraph });
    startFps();
    bindKeys();
    if (A.api.isLive) pollTimer = setInterval(function () { if (!document.hidden) pollExternal(); }, 15000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }
})();
