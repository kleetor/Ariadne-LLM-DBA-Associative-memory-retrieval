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
        onStats: function (stats) { A.panels.updateStatusbar(stats); },
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

  function bindKeys() {
    document.addEventListener('keydown', function (e) {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      // 撤销按钮的 title 写的就是 Ctrl+Z，这里补上对应绑定
      if ((e.ctrlKey || e.metaKey) && String(e.key).toLowerCase() === 'z') {
        e.preventDefault();
        A.panels.undo();
        return;
      }
      if (!graphReady) return;
      if (e.key === 'r' || e.key === 'R') A.graph.zoomFit();
      else if (e.key === '1') A.graph.setCamera(0, 0, 800);
      else if (e.key === '2') A.graph.setCamera(800, 0, 0);
      else if (e.key === '3') A.graph.setCamera(0, 800, 0);
      else if (e.key === '/') { e.preventDefault(); const s = document.getElementById('search-input'); if (s) s.focus(); }
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
