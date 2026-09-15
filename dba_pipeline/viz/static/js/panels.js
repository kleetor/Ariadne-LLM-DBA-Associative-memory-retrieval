/* SPDX-License-Identifier: AGPL-3.0-only */
/* 面板层：过滤器 / 检查器(CRUD) / 可观测性 / 设置 / 外壳 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});
  const C = A.const;
  const U = A.util;
  const el = U.el;
  const $ = U.$;

  let ctx = null;
  const undoStack = [];
  const MAX_UNDO = 50;

  // =========================================================
  // 外壳：导航 / 抽屉 / 状态栏
  // =========================================================

  const shell = {
    activeTab: 'graph',
    openDrawer(tab) {
      this.activeTab = tab;
      document.body.setAttribute('data-tab', tab);
      U.$$('#nav button').forEach(function (b) { b.classList.toggle('active', b.dataset.tab === tab); });
      const drawer = $('drawer');
      drawer.classList.add('open');
      U.$$('.view').forEach(function (v) { v.classList.toggle('active', v.dataset.view === tab); });
      $('drawer-title').textContent =
        tab === 'observability' ? '可观测性' : tab === 'params' ? '运行时参数' : '设置';
      // 打开时刷新一次，保证数据新鲜
      if (tab === 'observability') observability.refresh();
      else if (tab === 'params') paramsPanel.init();
      else if (tab === 'settings' && A.api.isLive) { settingsPanel.loadConfig(); mcpStatus.load(); }
    },
    closeDrawer() {
      $('drawer').classList.remove('open');
      this.activeTab = 'graph';
      document.body.setAttribute('data-tab', 'graph');
      U.$$('#nav button').forEach(function (b) { b.classList.toggle('active', b.dataset.tab === 'graph'); });
      U.$$('.view').forEach(function (v) { v.classList.remove('active'); });
    },
    init() {
      U.$$('#nav button').forEach(function (btn) {
        btn.addEventListener('click', function () {
          if (btn.dataset.tab === 'graph') shell.closeDrawer();
          else shell.openDrawer(btn.dataset.tab);
        });
      });
      $('drawer-close').addEventListener('click', function () { shell.closeDrawer(); });
      $('drawer-backdrop').addEventListener('click', function () { shell.closeDrawer(); });
      document.addEventListener('keydown', function (e) {
        if (e.key !== 'Escape') return;
        // 按「最上层浮层」顺序收：右键菜单 → 模态框 → 抽屉，都没开才清空搜索框
        if (closeTopOverlay()) return;
        hideContextMenu();
        const si = $('search-input');
        if (si) { si.value = ''; si.dispatchEvent(new Event('input')); }
      });
    },
  };

  // Esc 的关闭顺序。返回 true 表示这次按键已被某个浮层消费。
  function closeTopOverlay() {
    const menu = $('ctx-menu');
    if (menu && menu.classList.contains('open')) { hideContextMenu(); return true; }
    const overlay = document.querySelector('.modal-overlay');
    if (overlay) { overlay.remove(); return true; }
    const drawer = $('drawer');
    if (drawer && drawer.classList.contains('open')) { shell.closeDrawer(); return true; }
    return false;
  }

  function setStatus(id, value) {
    const e = $(id);
    if (e) e.textContent = value;
  }

  function updateStatusbar(stats) {
    setStatus('stat-nodes', stats.visibleNodes + (stats.visibleNodes !== stats.totalNodes ? ' / ' + stats.totalNodes : ''));
    setStatus('stat-edges', stats.visibleEdges + (stats.visibleEdges !== stats.totalEdges ? ' / ' + stats.totalEdges : ''));
    setStatus('stat-deprecated', stats.deprecated);
    setStatus('stat-forgotten', stats.forgotten);
    setStatus('stat-orphans', stats.orphans);
    setStatus('stat-undo', undoStack.length);
  }

  // =========================================================
  // 过滤器面板
  // =========================================================

  const filtersPanel = {
    init() {
      const host = $('panel-filters');
      host.innerHTML = '';
      host.appendChild(el('div', { class: 'panel-title' }, [
        el('span', { text: '图层过滤' }),
        el('button', { class: 'icon-btn', title: '折叠', onclick: function () { host.classList.toggle('collapsed'); } }, ['–']),
      ]));
      const body = el('div', { class: 'panel-body' });
      host.appendChild(body);

      // 节点类型
      body.appendChild(el('div', { class: 'group-label', text: '节点类型' }));
      const nodeWrap = el('div', { class: 'check-grid' });
      C.NODE_TYPES.forEach(function (t) {
        const input = el('input', { type: 'checkbox', checked: 'checked' });
        input.addEventListener('change', function () {
          const patch = {}; patch[t.key] = input.checked;
          A.graph.setFilter({ nodeTypes: Object.assign(A.graph.getFilter().nodeTypes, patch) });
        });
        nodeWrap.appendChild(el('label', { style: 'color:' + t.color }, [
          input,
          el('span', { class: 'swatch', style: 'background:' + t.color }),
          el('span', { class: 'type-cn', text: t.label }),
          el('span', { class: 'type-en', text: t.key }),
        ]));
      });
      body.appendChild(nodeWrap);

      // 边类型
      body.appendChild(el('div', { class: 'group-label', text: '边类型' }));
      const edgeWrap = el('div', { class: 'check-grid' });
      C.EDGE_TYPES.forEach(function (t) {
        const input = el('input', { type: 'checkbox', checked: 'checked' });
        input.addEventListener('change', function () {
          const patch = {}; patch[t.key] = input.checked;
          A.graph.setFilter({ edgeTypes: Object.assign(A.graph.getFilter().edgeTypes, patch) });
        });
        edgeWrap.appendChild(el('label', { style: 'color:' + t.color }, [
          input,
          el('span', { class: 'swatch line', style: 'background:' + t.color }),
          el('span', { class: 'type-cn', text: t.label }),
          el('span', { class: 'type-en', text: t.key }),
        ]));
      });
      body.appendChild(edgeWrap);

      // 开关
      body.appendChild(el('div', { class: 'divider' }));
      body.appendChild(toggle('显示废弃节点', true, function (v) { A.graph.setFilter({ showDeprecated: v }); }));
      body.appendChild(toggle('显示遗忘节点', false, function (v) { A.graph.setFilter({ showForgotten: v }); }));
      body.appendChild(toggle('隐藏孤立节点', false, function (v) { A.graph.setFilter({ hideOrphans: v }); }));
      body.appendChild(toggle('仅高连接度 (≥中位数)', false, function (v) { A.graph.setFilter({ highDegOnly: v }); }));

      // 快捷按钮
      body.appendChild(el('div', { class: 'divider' }));
      const row = el('div', { class: 'btn-row' });
      row.appendChild(el('button', { class: 'btn', text: '全图适配', onclick: function () { A.graph.zoomFit(); } }));
      row.appendChild(el('button', { class: 'btn', text: '退出聚焦', onclick: function () { A.graph.exitFocus(); A.graph.select(null); renderInspector(null); } }));
      row.appendChild(el('button', { class: 'btn warn', text: '释放固定节点', onclick: function () { A.graph.unfixAll(); U.toast('已释放全部固定节点'); } }));
      row.appendChild(el('button', { class: 'btn', text: '重置过滤', onclick: function () { resetFilters(); } }));
      body.appendChild(row);

      // 预设
      body.appendChild(el('div', { class: 'divider' }));
      body.appendChild(el('div', { class: 'group-label', text: '过滤预设' }));
      const presetRow = el('div', { class: 'preset-row' });
      const select = el('select', { class: 'input', id: 'preset-select' });
      presetRow.appendChild(select);
      presetRow.appendChild(el('button', { class: 'btn', text: '应用', onclick: function () { applySelectedPreset(select.value); } }));
      presetRow.appendChild(el('button', { class: 'btn', text: '保存', onclick: function () { saveCurrentPreset(); } }));
      presetRow.appendChild(el('button', { class: 'btn danger', text: '删除', onclick: function () { deleteSelectedPreset(select.value); } }));
      body.appendChild(presetRow);

      refreshPresetSelect();
      this.inputs = {
        nodeType: nodeWrap, edgeType: edgeWrap,
      };
    },
    sync(filter) {
      U.$$('input[type=checkbox]', this.inputs.nodeType).forEach(function (cb, i) {
        cb.checked = filter.nodeTypes[C.NODE_TYPES[i].key] !== false;
      });
      U.$$('input[type=checkbox]', this.inputs.edgeType).forEach(function (cb, i) {
        cb.checked = filter.edgeTypes[C.EDGE_TYPES[i].key] !== false;
      });
      const toggles = $('panel-filters').querySelectorAll('.toggle input');
      if (toggles[0]) toggles[0].checked = filter.showDeprecated;
      if (toggles[1]) toggles[1].checked = filter.showForgotten;
      if (toggles[2]) toggles[2].checked = filter.hideOrphans;
      if (toggles[3]) toggles[3].checked = filter.highDegOnly;
    },
  };

  function toggle(label, checked, onChange) {
    const input = el('input', { type: 'checkbox' });
    input.checked = checked;
    input.addEventListener('change', function () { onChange(input.checked); });
    return el('label', { class: 'toggle' }, [input, el('span', { text: label })]);
  }

  function resetFilters() {
    A.graph.setFilter({
      nodeTypes: {}, edgeTypes: {},
      showDeprecated: true, showForgotten: false, hideOrphans: false, highDegOnly: false,
    });
    // 空对象代表全部放行；重填后同步 UI
    const f = A.graph.getFilter();
    C.NODE_TYPES.forEach(function (t) { if (f.nodeTypes[t.key] === undefined) f.nodeTypes[t.key] = true; });
    C.EDGE_TYPES.forEach(function (t) { if (f.edgeTypes[t.key] === undefined) f.edgeTypes[t.key] = true; });
    A.graph.setFilter(f);
    filtersPanel.sync(A.graph.getFilter());
  }

  function refreshPresetSelect() {
    const select = $('preset-select');
    if (!select) return;
    const presets = A.store.listPresets();
    select.innerHTML = '<option value="">— 选择预设 —</option>' +
      presets.map(function (p) { return '<option value="' + p.id + '">' + U.escapeHtml(p.name) + '</option>'; }).join('');
  }

  function currentFilterState() {
    return JSON.parse(JSON.stringify(A.graph.getFilter()));
  }

  function saveCurrentPreset() {
    const name = window.prompt('预设名称', '预设 ' + new Date().toLocaleTimeString());
    if (!name) return;
    A.store.savePreset(name, currentFilterState());
    refreshPresetSelect();
    settingsPanel.refreshPresets();
    U.toast('已保存预设：' + name, 'ok');
  }

  function applySelectedPreset(id) {
    if (!id) { U.toast('请先选择预设', 'warn'); return; }
    const preset = A.store.listPresets().filter(function (p) { return p.id === id; })[0];
    if (!preset) return;
    A.graph.setFilter(preset.state);
    filtersPanel.sync(A.graph.getFilter());
    U.toast('已应用预设：' + preset.name, 'ok');
  }

  function deleteSelectedPreset(id) {
    if (!id) { U.toast('请先选择预设', 'warn'); return; }
    if (!window.confirm('删除该预设？')) return;
    A.store.deletePreset(id);
    refreshPresetSelect();
    settingsPanel.refreshPresets();
  }

  // =========================================================
  // 检查器（节点详情 + CRUD）
  // =========================================================

  const inspector = {
    editingId: null,
    render(node) {
      const host = $('panel-inspector');
      const offline = !A.api.isLive;
      if (!node) {
        this.editingId = null;
        host.innerHTML = '';
        host.appendChild(el('div', { class: 'panel-title' }, [el('span', { text: '节点检查器' })]));
        host.appendChild(el('div', { class: 'panel-body' }, [
          el('div', { class: 'hint', text: '点击 3D 图中的节点查看详情与编辑，或使用下方按钮新建。' }),
          offline
            ? el('div', { class: 'hint', text: '离线 HTML 模式：只读浏览，编辑功能不可用。' })
            : el('div', { class: 'btn-row' }, [
              el('button', { class: 'btn primary', text: '+ 新建节点', onclick: openNewNode }),
              el('button', { class: 'btn', text: '+ 新建边', onclick: openNewEdge }),
            ]),
        ]));
        return;
      }
      this.editingId = node.id;
      host.innerHTML = '';
      host.appendChild(el('div', { class: 'panel-title' }, [
        el('span', { text: '节点 ' + node.id }),
        el('button', { class: 'icon-btn', title: '复制 ID', onclick: function () { U.copy(node.id); } }, ['⧉']),
      ]));
      const body = el('div', { class: 'panel-body' });

      const color = C.NODE_COLORS[node.node_type] || '#64748b';
      body.appendChild(el('div', { class: 'node-meta' }, [
        el('span', { class: 'type-badge', style: 'background:' + color, text: node.node_type }),
        el('span', { class: 'muted', text: '入 ' + (node.in_degree || 0) + ' · 出 ' + (node.out_degree || 0) }),
        node.deprecated ? el('span', { class: 'tag warn', text: '已废弃' }) : null,
        node.forgotten ? el('span', { class: 'tag', text: '已遗忘' }) : null,
      ]));

      const typeSelect = el('select', { class: 'input' });
      C.NODE_TYPES.forEach(function (t) {
        const opt = el('option', { value: t.key, text: t.key + ' · ' + t.label });
        if (t.key === node.node_type) opt.selected = true;
        typeSelect.appendChild(opt);
      });
      const textarea = el('textarea', { class: 'input', rows: '4' });
      textarea.value = node.content || '';

      body.appendChild(el('label', { class: 'field' }, [el('span', { text: '类型' }), typeSelect]));
      body.appendChild(el('label', { class: 'field' }, [el('span', { text: '内容' }), textarea]));

      if (offline) {
        typeSelect.disabled = true;
        textarea.readOnly = true;
        body.appendChild(el('div', { class: 'hint', text: '离线 HTML 模式：只读浏览，编辑功能不可用。' }));
      } else {
        const actions = el('div', { class: 'btn-row' });
        actions.appendChild(el('button', {
          class: 'btn primary', text: '保存',
          onclick: function () { saveNode(node, typeSelect.value, textarea.value); },
        }));
        actions.appendChild(el('button', {
          class: 'btn', text: node.deprecated ? '恢复' : '废弃',
          onclick: function () { patchNode(node, { deprecated: !node.deprecated }, node.deprecated ? '恢复' : '废弃'); },
        }));
        actions.appendChild(el('button', {
          class: 'btn', text: node.forgotten ? '取消遗忘' : '遗忘',
          onclick: function () { patchNode(node, { forgotten: !node.forgotten }, node.forgotten ? '取消遗忘' : '遗忘'); },
        }));
        actions.appendChild(el('button', {
          class: 'btn danger', text: '删除',
          onclick: function () { deleteNode(node); },
        }));
        body.appendChild(actions);
      }

      // 快捷交互
      body.appendChild(el('div', { class: 'divider' }));
      const nav = el('div', { class: 'btn-row' });
      nav.appendChild(el('button', { class: 'btn', text: '聚焦 1 跳', onclick: function () { A.graph.focus(node.id, 1); } }));
      nav.appendChild(el('button', { class: 'btn', text: '聚焦 2 跳', onclick: function () { A.graph.focus(node.id, 2); } }));
      if (!offline) {
        nav.appendChild(el('button', { class: 'btn', text: '连接边', onclick: function () { startEdge('connect', node.id); } }));
        nav.appendChild(el('button', { class: 'btn', text: '删除边', onclick: function () { startEdge('disconnect', node.id); } }));
      }
      body.appendChild(nav);

      // 关联边
      const edges = edgesOf(node.id);
      body.appendChild(el('div', { class: 'group-label', text: '关联边（' + edges.length + '）' }));
      if (!edges.length) body.appendChild(el('div', { class: 'hint', text: '无关联边' }));
      edges.slice(0, 200).forEach(function (e) {
        const row = el('div', { class: 'edge-row' });
        row.appendChild(el('span', { class: 'arrow', text: e.direction === 'in' ? '←' : '→' }));
        row.appendChild(el('span', {
          class: 'type-badge small', style: 'background:' + (C.EDGE_COLORS[e.rel_type] || '#666'), text: e.rel_type,
        }));
        row.appendChild(el('span', { class: 'edge-peer', text: e.peerContent || e.peerId }));
        row.appendChild(el('button', {
          class: 'icon-btn danger', title: '删除此边',
          onclick: function () { removeEdge(e.source, e.target, e.rel_type); },
        }, ['×']));
        body.appendChild(row);
      });

      host.appendChild(body);
    },
  };

  function edgesOf(id) {
    const rows = [];
    (ctx.data.links || []).forEach(function (l) {
      const s = endId(l.source), t = endId(l.target);
      if (s === id) rows.push({ direction: 'out', rel_type: l.rel_type, peerId: t, peerContent: contentOf(t), source: s, target: t });
      if (t === id) rows.push({ direction: 'in', rel_type: l.rel_type, peerId: s, peerContent: contentOf(s), source: s, target: t });
    });
    return rows;
  }

  function contentOf(id) {
    const n = ctx.data.nodes.filter(function (x) { return x.id === id; })[0];
    return n ? n.content : id;
  }

  function endId(end) { return end && end.id !== undefined ? end.id : end; }

  function renderInspector(node) { inspector.render(node); }

  async function saveNode(node, type, content) {
    if (!content.trim()) { U.toast('内容不能为空', 'warn'); return; }
    const before = { node_type: node.node_type, content: node.content };
    try {
      await A.api.updateNode(node.id, { node_type: type, content: content });
      pushUndo('更新节点 ' + node.id, function () { return A.api.updateNode(node.id, before); });
      U.toast('已保存 ' + node.id, 'ok');
      await ctx.reloadGraph(node.id);
    } catch (e) { U.toast('保存失败：' + e.message, 'error'); }
  }

  async function patchNode(node, patch, label) {
    try {
      await A.api.updateNode(node.id, patch);
      pushUndo(label + ' ' + node.id, function () { return A.api.updateNode(node.id, inversePatch(patch)); });
      U.toast(label + '成功', 'ok');
      await ctx.reloadGraph(node.id);
    } catch (e) { U.toast(label + '失败：' + e.message, 'error'); }
  }

  function inversePatch(patch) {
    const inv = {};
    Object.keys(patch).forEach(function (k) { inv[k] = !patch[k]; });
    return inv;
  }

  async function removeEdge(source, target, relType) {
    if (!window.confirm('删除边 ' + source + ' → ' + target + '？')) return;
    try {
      await A.api.deleteEdge(source, target);
      pushUndo('删除边 ' + source + '→' + target, function () { return A.api.createEdge(source, target, relType); });
      U.toast('已删除边', 'ok');
      await ctx.reloadGraph(source);
    } catch (e) { U.toast('删除边失败：' + e.message, 'error'); }
  }

  async function deleteNode(node) {
    if (!window.confirm('确认删除节点 ' + node.id + '？关联边将一并删除。')) return;
    try {
      await A.api.deleteNode(node.id);
      U.toast('已删除 ' + node.id, 'ok');
      A.graph.select(null);
      await ctx.reloadGraph(null);
    } catch (e) { U.toast('删除失败：' + e.message, 'error'); }
  }

  function openNewNode() {
    openModal('新建节点', function (form) {
      const select = el('select', { class: 'input' });
      C.NODE_TYPES.forEach(function (t) { select.appendChild(el('option', { value: t.key, text: t.key + ' · ' + t.label })); });
      const textarea = el('textarea', { class: 'input', rows: '4', placeholder: '节点内容…' });
      form.appendChild(el('label', { class: 'field' }, [el('span', { text: '类型' }), select]));
      form.appendChild(el('label', { class: 'field' }, [el('span', { text: '内容' }), textarea]));
      return async function () {
        if (!textarea.value.trim()) { U.toast('请输入内容', 'warn'); return false; }
        const result = await A.api.createNode(select.value, textarea.value);
        pushUndo('新建节点 ' + result.id, function () { return A.api.deleteNode(result.id); });
        U.toast('已创建 ' + result.id, 'ok');
        await ctx.reloadGraph(result.id);
        return true;
      };
    });
  }

  function openNewEdge() {
    const nodes = ctx.data.nodes;
    if (nodes.length < 2) { U.toast('节点不足，无法建边', 'warn'); return; }
    openModal('新建边', function (form) {
      const sourceSel = nodeSelect(nodes, inspector.editingId);
      const targetSel = nodeSelect(nodes, null);
      const relSel = el('select', { class: 'input' });
      C.EDGE_TYPES.forEach(function (t) { relSel.appendChild(el('option', { value: t.key, text: t.key + ' · ' + t.label })); });
      form.appendChild(el('label', { class: 'field' }, [el('span', { text: '源节点' }), sourceSel]));
      form.appendChild(el('label', { class: 'field' }, [el('span', { text: '目标节点' }), targetSel]));
      form.appendChild(el('label', { class: 'field' }, [el('span', { text: '边类型' }), relSel]));
      return async function () {
        const s = sourceSel.value, t = targetSel.value, r = relSel.value;
        if (s === t) { U.toast('源与目标不能相同', 'warn'); return false; }
        await A.api.createEdge(s, t, r);
        pushUndo('新建边 ' + s + '→' + t, function () { return A.api.deleteEdge(s, t); });
        U.toast('已创建边', 'ok');
        await ctx.reloadGraph(s);
        return true;
      };
    });
  }

  function nodeSelect(nodes, selected) {
    const sel = el('select', { class: 'input' });
    nodes.forEach(function (n) {
      const opt = el('option', { value: n.id, text: n.id + ' · ' + String(n.content || '').slice(0, 34) });
      if (n.id === selected) opt.selected = true;
      sel.appendChild(opt);
    });
    return sel;
  }

  function openModal(title, build) {
    const overlay = el('div', { class: 'modal-overlay' });
    const box = el('div', { class: 'modal' });
    box.appendChild(el('h3', { text: title }));
    const form = el('div', { class: 'modal-form' });
    box.appendChild(form);
    const confirmFn = build(form);
    const actions = el('div', { class: 'btn-row end' });
    actions.appendChild(el('button', { class: 'btn', text: '取消', onclick: function () { overlay.remove(); } }));
    actions.appendChild(el('button', {
      class: 'btn primary', text: '确认',
      onclick: async function () {
        try {
          const ok = await confirmFn();
          if (ok !== false) overlay.remove();
        } catch (e) { U.toast(e.message, 'error'); }
      },
    }));
    box.appendChild(actions);
    overlay.appendChild(box);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
    const first = form.querySelector('input,textarea,select');
    if (first) first.focus();
  }

  // ---- 边模式 ----

  function startEdge(mode, sourceId) {
    const relType = mode === 'connect' ? 'causal' : null;
    A.graph.enterEdgeMode(mode, relType, sourceId);
    showEdgeBanner(mode, relType, sourceId);
  }

  function showEdgeBanner(mode, relType, sourceId) {
    const banner = $('edge-banner');
    banner.innerHTML = '';
    banner.classList.add('show', mode);
    banner.appendChild(el('span', {
      text: mode === 'connect'
        ? '连接模式：选择关系类型后点击目标节点（点击空白取消）'
        : '删除模式：点击与 ' + sourceId + ' 相连的目标节点（点击空白取消）',
    }));
    if (mode === 'connect') {
      const sel = el('select', { class: 'input small' });
      C.EDGE_TYPES.forEach(function (t) { sel.appendChild(el('option', { value: t.key, text: t.key })); });
      sel.addEventListener('change', function () {
        A.graph.enterEdgeMode('connect', sel.value, sourceId);
      });
      banner.appendChild(sel);
    }
    banner.appendChild(el('button', { class: 'btn small', text: '取消', onclick: function () { A.graph.exitEdgeMode(); } }));
  }

  function hideEdgeBanner() {
    const banner = $('edge-banner');
    banner.classList.remove('show');
    banner.innerHTML = '';
  }

  async function handleEdgeAction(action) {
    const { mode, relType, sourceId, targetId } = action;
    if (sourceId === targetId) { U.toast('不能连接自身', 'warn'); return; }
    if (mode === 'connect') {
      const exists = (ctx.data.links || []).some(function (l) {
        return endId(l.source) === sourceId && endId(l.target) === targetId;
      });
      if (exists) { U.toast('两点之间已存在边', 'warn'); A.graph.exitEdgeMode(); return; }
      try {
        await A.api.createEdge(sourceId, targetId, relType || 'causal');
        pushUndo('连接边 ' + sourceId + '→' + targetId, function () { return A.api.deleteEdge(sourceId, targetId); });
        U.toast('已连接', 'ok');
        A.graph.exitEdgeMode();
        await ctx.reloadGraph(sourceId);
      } catch (e) { U.toast('创建边失败：' + e.message, 'error'); }
    } else {
      const link = (ctx.data.links || []).filter(function (l) {
        const s = endId(l.source), t = endId(l.target);
        return (s === sourceId && t === targetId) || (s === targetId && t === sourceId);
      })[0];
      if (!link) { U.toast('两点之间没有边', 'warn'); A.graph.exitEdgeMode(); return; }
      const s = endId(link.source), t = endId(link.target);
      try {
        await A.api.deleteEdge(s, t);
        pushUndo('删除边 ' + s + '→' + t, function () { return A.api.createEdge(s, t, link.rel_type); });
        U.toast('已删除边', 'ok');
        A.graph.exitEdgeMode();
        await ctx.reloadGraph(sourceId);
      } catch (e) { U.toast('删除边失败：' + e.message, 'error'); }
    }
  }

  // ---- 撤销 ----

  function pushUndo(label, fn) {
    undoStack.push({ label: label, fn: fn });
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    setStatus('stat-undo', undoStack.length);
    const btn = $('btn-undo');
    if (btn) btn.disabled = undoStack.length === 0;
  }

  async function doUndo() {
    const action = undoStack.pop();
    setStatus('stat-undo', undoStack.length);
    const btn = $('btn-undo');
    if (btn) btn.disabled = undoStack.length === 0;
    if (!action) { U.toast('没有可撤销的操作', 'warn'); return; }
    try {
      await action.fn();
      U.toast('已撤销：' + action.label, 'ok');
      await ctx.reloadGraph(inspector.editingId);
    } catch (e) { U.toast('撤销失败：' + e.message, 'error'); }
  }

  // =========================================================
  // 搜索 / 右键菜单 / 悬浮提示
  // =========================================================

  function initSearch() {
    const input = $('search-input');
    const dropdown = $('search-dropdown');
    const run = U.debounce(function () {
      const q = input.value.trim();
      if (!q) { dropdown.classList.remove('open'); A.graph.clearSearch(); return; }
      const matched = A.graph.search(q);
      dropdown.innerHTML = matched.slice(0, 20).map(function (n) {
        return '<div class="search-item" tabindex="0" data-id="' + U.escapeHtml(n.id) + '">' +
          '<span class="dot" style="background:' + (C.NODE_COLORS[n.node_type] || '#888') + '"></span>' +
          '<span class="text">' + U.escapeHtml(n.content || '') + '</span>' +
          '<span class="type">' + U.escapeHtml(n.node_type) + '</span></div>';
      }).join('') || '<div class="search-empty">无匹配结果</div>';
      dropdown.classList.add('open');
      U.$$('.search-item', dropdown).forEach(function (item) {
        const choose = function () {
          const node = ctx.data.nodes.filter(function (x) { return x.id === item.dataset.id; })[0];
          if (node) {
            A.graph.select(node.id);
            A.graph.focus(node.id, 1);
          }
          input.value = '';
          dropdown.classList.remove('open');
          A.graph.clearSearch();
        };
        item.addEventListener('click', choose);
        // 结果项是 div，必须显式支持 Enter/Space，否则键盘用户选不中
        item.addEventListener('keydown', function (e) {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); }
        });
      });
    }, 160);
    input.addEventListener('input', run);
    document.addEventListener('click', function (e) {
      if (!dropdown.contains(e.target) && e.target !== input) dropdown.classList.remove('open');
    });
  }

  function initContextMenu() {
    const menu = $('ctx-menu');
    document.addEventListener('click', hideContextMenu);
    document.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  }

  function hideContextMenu() { const m = $('ctx-menu'); if (m) m.classList.remove('open'); }

  function showContextMenu(node, evt) {
    const menu = $('ctx-menu');
    const live = A.api.isLive;
    const items = [];
    if (node) {
      items.push(['聚焦 1 跳', function () { A.graph.focus(node.id, 1); }]);
      items.push(['聚焦 2 跳', function () { A.graph.focus(node.id, 2); }]);
      items.push(['sep']);
      if (live) {
        items.push(['从此节点连边', function () { startEdge('connect', node.id); }]);
        items.push(['删除与此节点相连的边', function () { startEdge('disconnect', node.id); }]);
        items.push(['sep']);
      }
      items.push(['隐藏 ' + node.node_type + ' 类型', function () {
        const nt = A.graph.getFilter().nodeTypes; nt[node.node_type] = false;
        A.graph.setFilter({ nodeTypes: nt }); filtersPanel.sync(A.graph.getFilter());
      }]);
      items.push(['复制节点 ID', function () { U.copy(node.id); }]);
      if (live) {
        items.push(['sep']);
        items.push([node.deprecated ? '恢复节点' : '废弃节点', function () { patchNode(node, { deprecated: !node.deprecated }, node.deprecated ? '恢复' : '废弃'); }]);
        items.push(['删除节点', function () { deleteNode(node); }]);
      }
    } else {
      items.push(['显示全部', function () { resetFilters(); A.graph.exitFocus(); }]);
      items.push(['仅显示 CAUSAL 链路', function () {
        const et = {}; C.EDGE_TYPES.forEach(function (t) { et[t.key] = t.key === 'causal'; });
        A.graph.setFilter({ edgeTypes: et }); filtersPanel.sync(A.graph.getFilter());
      }]);
      items.push(['隐藏废弃节点', function () { A.graph.setFilter({ showDeprecated: false }); filtersPanel.sync(A.graph.getFilter()); }]);
      items.push(['全图适配', function () { A.graph.zoomFit(); }]);
    }
    menu.innerHTML = items.map(function (it, i) {
      if (it[0] === 'sep') return '<div class="sep"></div>';
      return '<div class="menu-item" role="menuitem" tabindex="0" data-i="' + i + '">' +
        U.escapeHtml(it[0]) + '</div>';
    }).join('');
    const runItem = function (elm) {
      items[Number(elm.dataset.i)][1]();
      hideContextMenu();
    };
    U.$$('.menu-item', menu).forEach(function (elm) {
      elm.addEventListener('click', function () { runItem(elm); });
      // 菜单项是 div，必须显式支持 Enter/Space，否则键盘用户点不到
      elm.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); runItem(elm); }
      });
    });
    // 必须先展开再测量：`#ctx-menu` 默认 display:none，此时 offsetWidth/offsetHeight
    // 都是 0，钳位会失效（菜单在屏幕下半部会溢出视口下沿）。
    menu.classList.add('open');
    const w = menu.offsetWidth || 186;
    const h = menu.offsetHeight || 0;
    const x = Math.max(4, Math.min(evt.clientX, window.innerWidth - w - 8));
    const y = Math.max(4, Math.min(evt.clientY, window.innerHeight - h - 8));
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
  }

  function clip(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
  }

  function showTooltip(pos, node) {
    const tip = $('tooltip');
    if (!pos || !node) {
      // 节点悬浮结束：若当前显示的是连线提示，交给连线自己的 hover 结束事件收尾
      if (tip.dataset.mode !== 'edge') { tip.classList.remove('show'); tip.dataset.mode = ''; }
      return;
    }
    tip.dataset.mode = 'node';
    tip.innerHTML = '<b>[' + U.escapeHtml(node.id) + ']</b> ' +
      '<span style="color:' + (C.NODE_COLORS[node.node_type] || '#888') + '">' + U.escapeHtml(node.node_type) + '</span>' +
      '<div>' + U.escapeHtml(String(node.content || '').slice(0, 160)) + '</div>';
    tip.style.left = Math.min(pos.x + 14, window.innerWidth - 320) + 'px';
    tip.style.top = (pos.y - 10) + 'px';
    tip.classList.add('show');
  }

  // 连线悬浮提示：关系类型（中文 + 英文）+ 两端内容，用于精确辨认某条边
  function showEdgeTooltip(pos, info) {
    const tip = $('tooltip');
    if (!pos || !info) {
      if (tip.dataset.mode !== 'node') { tip.classList.remove('show'); tip.dataset.mode = ''; }
      return;
    }
    const color = C.EDGE_COLORS[info.relType] || '#888';
    const meta = C.EDGE_TYPES.filter(function (t) { return t.key === info.relType; })[0];
    const label = meta ? meta.label : info.relType;
    const arrow = C.BIDIRECTIONAL.indexOf(info.relType) >= 0 ? '⇄' : '→';
    tip.dataset.mode = 'edge';
    tip.innerHTML =
      '<span style="display:inline-block;width:11px;height:3px;border-radius:2px;' +
      'background:' + color + ';vertical-align:middle;margin-right:6px"></span>' +
      '<b style="color:' + color + '">' + U.escapeHtml(label) + '</b>' +
      '<span class="muted"> · ' + U.escapeHtml(info.relType) + '</span>' +
      '<div style="margin-top:3px">' +
      U.escapeHtml(clip(info.source.content || info.source.id, 42)) +
      ' <b style="color:' + color + '">' + arrow + '</b> ' +
      U.escapeHtml(clip(info.target.content || info.target.id, 42)) +
      '</div>';
    tip.style.left = Math.min(pos.x + 14, window.innerWidth - 320) + 'px';
    tip.style.top = (pos.y - 10) + 'px';
    tip.classList.add('show');
  }

  // =========================================================
  // 可观测性
  // =========================================================

  const observability = {
    tab: 'oplog',
    streamHandle: null,
    streamState: 'idle',
    paused: false,
    liveRows: [],
    metricsTimer: null,
    metricsData: null,
    // 调用测试：阶段时间线（SSE 推来）+ 最终轨迹 + 报错
    chainStages: [],
    chainResult: null,
    chainError: '',
    chainRunning: false,
    chainQuery: '',

    init() {
      const host = $('view-observability');
      host.innerHTML = '';
      host.appendChild(el('div', { class: 'metric-grid', id: 'metric-grid' }));

      const tabs = el('div', { class: 'subtabs' });
      [['oplog', '操作日志'], ['logs', '运行日志'], ['live', '实时流'], ['chain', '调用测试']].forEach(function (t) {
        const btn = el('button', { class: 'subtab', text: t[1], 'data-sub': t[0] });
        btn.addEventListener('click', function () { observability.switchTab(t[0]); });
        tabs.appendChild(btn);
      });
      tabs.appendChild(el('button', { class: 'btn small right', id: 'obs-refresh', text: '刷新', onclick: function () { observability.refresh(); } }));
      host.appendChild(tabs);

      const panel = el('div', { class: 'obs-panel', id: 'obs-panel' });
      host.appendChild(panel);

      this.switchTab('oplog');
      this.refreshMetrics();
      // 仅在抽屉停留在可观测页时轮询指标，避免后台无谓请求
      this.metricsTimer = setInterval(function () {
        if (shell.activeTab === 'observability' && !document.hidden) observability.refreshMetrics();
      }, 5000);
      this.connectStream();
      if (A.api.isLive) this.refresh();
    },

    switchTab(tab) {
      this.tab = tab;
      U.$$('.subtab', $('view-observability')).forEach(function (b) { b.classList.toggle('active', b.dataset.sub === tab); });
      this.renderPanel();
      if (tab === 'oplog' && A.api.isLive) this.loadOplog();
      if (tab === 'logs' && A.api.isLive) this.loadLogs();
    },

    renderPanel() {
      const host = $('obs-panel');
      if (!A.api.isLive) {
        host.innerHTML = '<div class="hint">离线 HTML 模式：无可观测数据。</div>';
        return;
      }
      if (this.tab === 'oplog') host.innerHTML = this.oplogHtml();
      else if (this.tab === 'logs') host.innerHTML = this.logsHtml();
      else if (this.tab === 'chain') host.innerHTML = this.chainHtml();
      else host.innerHTML = this.liveHtml();
      this.bindPanel();
    },

    chainHtml() {
      const stages = this.chainStages.length
        ? this.chainStages.map(chainStageHtml).join('')
        : '<div class="hint">还没有运行记录。输入一条信息点「运行」。</div>';
      const running = this.chainRunning
        ? '<span class="stream-state open">运行中…</span>' : '';
      return '' +
        '<div class="obs-toolbar">' +
        '<input class="input small grow" id="chain-q" value="' + U.escapeHtml(this.chainQuery || '') +
        '" placeholder="输入一条信息，跑真实链路：意图识别 → PAR → StoryRank">' +
        '<button class="btn small" id="chain-run"' + (this.chainRunning ? ' disabled' : '') + '>运行</button>' +
        '<button class="btn small" id="chain-clear">清除激活</button>' +
        running +
        '</div>' +
        '<div class="chain-note">阶段进度经 SSE 实时推送；链路跑完后 PAR 经过的节点在图谱里<b>保留颜色</b>，' +
        '其余节点变灰，StoryRank 采纳的节点额外金色高亮。<br>' +
        '首次运行要在面板进程内装配链路（加载 embedding 模型 + 向量对齐），会明显偏慢；' +
        '一次调用约 2 次 LLM + 2~3 次 embedding 请求。</div>' +
        '<div id="chain-stages">' + stages + '</div>' +
        '<div id="chain-result">' + this.chainResultHtml() + '</div>';
    },

    async runChain() {
      const input = $('chain-q');
      const query = input ? input.value.trim() : '';
      if (!query) return;
      if (this.chainRunning) return;
      this.chainRunning = true;
      this.chainQuery = query;
      this.chainStages = [];
      this.chainResult = null;
      this.chainError = '';
      A.graph.clearActivated();          // 新一轮开始先撤掉上一轮的激活
      if (this.tab === 'chain') this.renderPanel();
      try {
        const res = await A.api.runChainTest(query);
        this.chainResult = res;
        this.applyActivation(res.visited_ids, res.story_nodes);
      } catch (e) {
        // 409 = 已有一次在跑 / 链路装配失败
        this.chainError = e.message || String(e);
      } finally {
        this.chainRunning = false;
        if (this.tab === 'chain') this.renderPanel();
      }
    },

    applyActivation(visited, adopted) {
      if (!A.graph.setActivated) return;
      A.graph.setActivated(visited || [], { adopted: adopted || [] });
    },

    // SSE：channel="chain" 的阶段事件（装配 / 对齐 / 意图 / 每跳 / 寻峰 / 故事化）
    onChainEvent(evt) {
      if (!evt || !evt.stage) return;
      if (evt.stage === 'done') {
        const info = evt.info || {};
        this.applyActivation(info.visited_ids, info.story_nodes);
        // done 事件带了完整轨迹：POST 响应被代理/长连接拖住时靠它收尾，
        // 否则按钮会一直停在"运行中"、结果区永远空着
        if (info.result && !this.chainResult) {
          this.chainResult = info.result;
          this.chainRunning = false;
          if (this.tab === 'chain') this.renderPanel();
        }
      }
      this.chainStages.push({ stage: evt.stage, info: evt.info || {}, ts: evt.ts });
      if (this.chainStages.length > 80) this.chainStages.shift();
      if (this.tab !== 'chain') return;
      const host = $('chain-stages');
      if (host) {
        host.innerHTML = this.chainStages.map(chainStageHtml).join('');
        host.scrollTop = host.scrollHeight;
      }
    },

    chainResultHtml() {
      if (this.chainError) return '<div class="chain-error">运行失败：' + U.escapeHtml(this.chainError) + '</div>';
      const r = this.chainResult;
      if (!r) return '';
      const purpose = r.purpose || {};
      const adopted = new Set(r.story_nodes || []);
      const hops = r.hop_history || [];
      const pr = r.params || {};
      let html = '<div class="chain-summary">' +
        chainKv('意图 / 目的', (purpose.query_type || '-') + '　·　' + ((purpose.purposes || []).join('；') || '-')) +
        chainKv('用户状态', purpose.status || '-') +
        chainKv('PAR 经过节点', (r.visited_ids || []).length + ' 个（图上保留颜色）') +
        chainKv('StoryRank 采纳', (r.story_nodes || []).length + ' 个（金色高亮）') +
        chainKv('被丢弃', (r.discarded_nodes || []).length + ' 个') +
        chainKv('跳数 / 参数', hops.length + ' 跳　·　seed_k=' + (pr.seed_k || '-') +
          ' max_hops=' + (pr.max_hops || '-') + ' expand_k=' + (pr.expand_k || '默认')) +
        '</div>';

      const story = (r.stories || []).join('\n\n');
      html += '<div class="chain-section">故事</div><div class="chain-story">' +
        (story ? U.escapeHtml(story) : '<span class="muted">（未生成故事）</span>') + '</div>';

      html += '<div class="chain-section">PAR 轨迹（每跳候选，按组合得分降序）</div>';
      hops.forEach(function (hop) {
        const cands = hop.candidates || [];
        html += '<div class="chain-hop">' +
          '<div class="chain-hop-head">hop ' + hop.hop +
          '　均分 ' + fmtScore(hop.mean_score) +
          '　候选 ' + cands.length + '</div>' +
          '<table class="hop-table"><thead><tr>' +
          '<th>节点</th><th>内容</th><th>组合分</th><th>目的分</th><th>跳转权重</th><th>来自</th><th>关系</th>' +
          '</tr></thead><tbody>' +
          cands.map(function (c) {
            const act = adopted.has(c.id);
            return '<tr' + (act ? ' class="adopted"' : '') + '>' +
              '<td><a href="#" class="chain-node" data-id="' + U.escapeHtml(c.id) + '">' +
              U.escapeHtml(c.id) + '</a>' + (act ? ' ★' : '') + '</td>' +
              '<td class="chain-content">' + U.escapeHtml(clip(c.content, 42)) + '</td>' +
              '<td>' + fmtScore(c.combined_score) + '</td>' +
              '<td>' + fmtScore(c.purpose_score) + '</td>' +
              '<td>' + (c.jump_weight === undefined ? '-' : fmtScore(c.jump_weight)) + '</td>' +
              '<td>' + U.escapeHtml(c.from || '—') + '</td>' +
              '<td>' + U.escapeHtml((c.rel_type || '—') + (c.is_reverse ? ' ←' : '')) + '</td>' +
              '</tr>';
          }).join('') +
          '</tbody></table></div>';
      });
      return html;
    },

    oplogHtml() {
      return '' +
        '<div class="obs-toolbar">' +
        '<select class="input small" id="oplog-source"><option value="">全部来源</option><option value="llm">LLM</option><option value="dba">DBA</option><option value="system">system</option></select>' +
        '<input class="input small" id="oplog-q" placeholder="关键字（op / actor / 内容）">' +
        '<select class="input small" id="oplog-tail"><option value="100">100 条</option><option value="200" selected>200 条</option><option value="500">500 条</option></select>' +
        '<button class="btn small" id="oplog-apply">筛选</button>' +
        '<a class="btn small" href="' + A.api.exportOplogUrl() + '" download>导出</a>' +
        '</div><div class="log-list" id="oplog-list"><div class="hint">加载中…</div></div>';
    },

    logsHtml() {
      return '' +
        '<div class="obs-toolbar">' +
        '<select class="input small" id="logs-level"><option value="">全部级别</option><option value="DEBUG">DEBUG</option><option value="INFO">INFO</option><option value="WARNING">WARNING</option><option value="ERROR">ERROR</option></select>' +
        '<input class="input small" id="logs-q" placeholder="关键字">' +
        '<button class="btn small" id="logs-apply">筛选</button>' +
        '</div><div class="log-list" id="logs-list"><div class="hint">加载中…</div></div>';
    },

    liveHtml() {
      const stateText = this.streamState === 'open' ? '已连接' : this.streamState === 'closed' ? '连接中断（重连中）' : '未连接';
      return '' +
        '<div class="obs-toolbar">' +
        '<span class="stream-state ' + this.streamState + '">' + stateText + '</span>' +
        '<button class="btn small" id="live-pause">' + (this.paused ? '继续' : '暂停') + '</button>' +
        '<button class="btn small" id="live-clear">清空</button>' +
        '<span class="muted" id="live-count">' + this.liveRows.length + ' 条</span>' +
        '</div><div class="log-list live" id="live-list"></div>';
    },

    bindPanel() {
      const apply = $('oplog-apply');
      if (apply) apply.addEventListener('click', function () { observability.loadOplog(); });
      const logsApply = $('logs-apply');
      if (logsApply) logsApply.addEventListener('click', function () { observability.loadLogs(); });
      const pause = $('live-pause');
      if (pause) pause.addEventListener('click', function () { observability.paused = !observability.paused; observability.renderPanel(); });
      const clear = $('live-clear');
      if (clear) clear.addEventListener('click', function () { observability.liveRows = []; observability.renderLive(); });
      // 调用测试
      const chainRun = $('chain-run');
      if (chainRun) chainRun.addEventListener('click', function () { observability.runChain(); });
      const chainInput = $('chain-q');
      if (chainInput) {
        chainInput.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') { e.preventDefault(); observability.runChain(); }
        });
      }
      const chainClear = $('chain-clear');
      if (chainClear) {
        chainClear.addEventListener('click', function () {
          A.graph.clearActivated();
          observability.chainStages = [];
          observability.chainResult = null;
          observability.chainError = '';
          observability.renderPanel();
        });
      }
      // 轨迹表里的节点 id 可点：在图上选中它（不隐藏其他节点，避免盖掉激活视图）
      U.$$('#chain-result .chain-node').forEach(function (a) {
        a.addEventListener('click', function (e) {
          e.preventDefault();
          if (A.graph.select) A.graph.select(a.dataset.id);
        });
      });
      this.renderLive();
    },

    refresh() {
      this.refreshMetrics();
      if (this.tab === 'oplog') this.loadOplog();
      else if (this.tab === 'logs') this.loadLogs();
    },

    async loadOplog() {
      const list = $('oplog-list');
      if (!list) return;
      const params = {
        tail: ($('oplog-tail') || {}).value || 200,
        source: ($('oplog-source') || {}).value || '',
        q: ($('oplog-q') || {}).value || '',
      };
      list.innerHTML = '<div class="hint">加载中…</div>';
      try {
        const res = await A.api.getOplog(params);
        const rows = (res.ops || []).slice().reverse(); // 新→旧
        if (!rows.length) {
          const o = (observability.metricsData || {}).oplog || {};
          const msg = o.exists
            ? '暂无匹配的操作记录（可调整来源/关键字筛选）。'
            : '操作日志文件尚未生成：' + (o.path || '') +
              '。LLM 通过 MCP 调用工具、或在本面板执行新增/修改/删除后会自动写入。';
          list.innerHTML = '<div class="hint">' + U.escapeHtml(msg) + '</div>';
          return;
        }
        list.innerHTML = rows.map(oplogRowHtml).join('');
        U.$$('.log-row', list).forEach(function (row) {
          row.addEventListener('click', function () { row.classList.toggle('expanded'); });
        });
      } catch (e) { list.innerHTML = '<div class="hint error">加载失败：' + U.escapeHtml(e.message) + '</div>'; }
    },

    async loadLogs() {
      const list = $('logs-list');
      if (!list) return;
      const params = {
        level: ($('logs-level') || {}).value || '',
        q: ($('logs-q') || {}).value || '',
        tail: 300,
      };
      list.innerHTML = '<div class="hint">加载中…</div>';
      try {
        const res = await A.api.getLogs(params);
        const rows = (res.logs || []).slice().reverse();
        if (!rows.length) {
          const a = (observability.metricsData || {}).applog || {};
          const msg = '暂无运行日志。面板自身日志会即时入列；MCP 等服务的日志写入 ' +
            (a.path || '共享日志文件') + ' 后会自动出现（需先启动 MCP）。';
          list.innerHTML = '<div class="hint">' + U.escapeHtml(msg) + '</div>';
          return;
        }
        list.innerHTML = rows.map(logRowHtml).join('');
      } catch (e) { list.innerHTML = '<div class="hint error">加载失败：' + U.escapeHtml(e.message) + '</div>'; }
    },

    async refreshMetrics() {
      if (!A.api.isLive) {
        const grid = $('metric-grid');
        if (grid) grid.innerHTML = '<div class="hint">离线模式</div>';
        return;
      }
      if (!A.api.isOnline()) return;  // 后端不可达时暂停轮询（由健康检查探活恢复）
      try {
        const data = await A.api.getMetrics();
        this.metricsData = data;
        renderMetrics(data);
      } catch (e) { /* 静默 */ }
    },

    connectStream() {
      if (!A.api.isLive) return;
      this.streamHandle = A.api.connectStream(function (event) {
        // 调用测试的阶段事件不进日志列表，单独喂给「调用测试」面板（暂停日志也不该挡住它）
        if (event.channel === 'chain') {
          observability.onChainEvent(event);
          return;
        }
        if (observability.paused) return;
        observability.liveRows.push(event);
        if (observability.liveRows.length > 500) observability.liveRows.shift();
        if (observability.tab === 'live') observability.renderLive();
        const count = $('live-count');
        if (count) count.textContent = observability.liveRows.length + ' 条';
      }, function (st) {
        observability.streamState = st;
        if (observability.tab === 'live') observability.renderPanel();
      });
    },

    renderLive() {
      const list = $('live-list');
      if (!list) return;
      if (!this.liveRows.length) {
        list.innerHTML = '<div class="hint">已连接，等待实时事件（操作日志 / 运行日志）…<br>' +
          'MCP 未启动时不会有 LLM 侧事件；在本面板执行增删改或启动 MCP 即可看到推送。</div>';
        return;
      }
      list.innerHTML = this.liveRows.map(function (evt) {
        return evt.channel === 'oplog' ? oplogRowHtml(evt) : logRowHtml(evt);
      }).join('');
      list.scrollTop = list.scrollHeight;
    },
  };

  // ---- 调用测试：阶段时间线与结果渲染 ----

  const CHAIN_STAGE_LABELS = {
    init_start: '装配链路', init_done: '装配完成', align_start: '向量对齐',
    intent_start: '意图识别', intent_done: '意图识别完成', seeds: '种子记忆',
    hop: '关联扩展', peak_found: '寻到峰值', peak_collected: '峰值容忍带',
    core_nodes: '连通性粗筛', storyrank_start: 'StoryRank 故事化',
    storyrank_done: '故事化完成', done: '完成',
  };

  function fmtScore(v) {
    if (v === null || v === undefined || v === '') return '-';
    const n = Number(v);
    return isNaN(n) ? String(v) : n.toFixed(3);
  }

  function chainKv(key, value) {
    return '<div class="chain-kv"><span class="chain-k">' + U.escapeHtml(key) + '</span>' +
      '<span class="chain-v">' + U.escapeHtml(String(value)) + '</span></div>';
  }

  function chainStageDetail(evt) {
    const i = evt.info || {};
    switch (evt.stage) {
      case 'init_start':
        return i.yaml ? String(i.yaml) : '';
      case 'align_start':
        return i.note || '';
      case 'intent_start':
        return i.query ? '「' + clip(String(i.query), 30) + '」' : '';
      case 'intent_done': {
        const p = i.purpose || {};
        return (p.query_type || '-') + '　·　' + ((p.purposes || []).join('；') || '-');
      }
      case 'seeds':
        return '种子 ' + ((i.ids || []).length) + ' 个（文本 ' + (i.text_hits || 0) +
          ' / 目的 ' + (i.purpose_hits || 0) + '）　均分 ' + fmtScore(i.mean_score);
      case 'hop': {
        const parts = ['扩展 ' + (i.expanded === undefined ? '-' : i.expanded) +
          '　通过过滤 ' + (i.kept === undefined ? '-' : i.kept)];
        if (i.mean_score !== undefined) parts.push('均分 ' + fmtScore(i.mean_score));
        if (i.decision) parts.push('寻峰 ' + i.decision);
        if (i.stop) parts.push('停止：' + i.stop);
        if (i.top && i.top.length) {
          parts.push('Top ' + i.top.slice(0, 3).map(function (t) {
            return t.id + '(' + fmtScore(t.combined_score) + ')';
          }).join(' '));
        }
        return parts.join('　');
      }
      case 'peak_found':
        return 'hop ' + i.hop + '　均分 ' + fmtScore(i.mean_score) + (i.reason ? '　' + i.reason : '');
      case 'peak_collected':
        return '峰值容忍带 ' + (i.count || 0) + ' 个节点';
      case 'core_nodes':
        return '核心节点组 ' + (i.groups || 0) + ' 个　合并后 ' + (i.nodes || 0) + ' 个';
      case 'storyrank_start':
        return '交给 LLM 的路径：' + (i.nodes || 0) + ' 节点 / ' + (i.edges || 0) + ' 边';
      case 'storyrank_done':
        return '采纳 ' + (i.adopted || 0) + ' 个　丢弃 ' + (i.discarded || 0) + ' 个';
      case 'done':
        return '经过 ' + (i.visited || 0) + ' 个节点（图上已激活）';
      default:
        return '';
    }
  }

  function chainStageHtml(evt) {
    const label = CHAIN_STAGE_LABELS[evt.stage] || evt.stage;
    const hop = evt.info && evt.info.hop !== undefined ? ' #' + evt.info.hop : '';
    return '<div class="chain-stage stage-' + U.escapeHtml(evt.stage) + '">' +
      '<span class="chain-stage-ts">' + U.fmtTime(evt.ts) + '</span>' +
      '<span class="chain-stage-name">' + U.escapeHtml(label + hop) + '</span>' +
      '<span class="chain-stage-detail">' + U.escapeHtml(chainStageDetail(evt)) + '</span>' +
      '</div>';
  }

  function oplogRowHtml(row) {
    const sourceColor = row.source === 'llm' ? '#7c3aed' : row.source === 'dba' ? '#0d9488' : '#64748b';
    const actor = row.actor ? (row.actor.model || row.actor.user || row.actor.type || '') : '';
    return '<div class="log-row">' +
      '<div class="log-line">' +
      '<span class="log-ts">' + U.fmtTime(row.ts) + '</span>' +
      '<span class="type-badge small" style="background:' + sourceColor + '">' + U.escapeHtml(row.source || '-') + '</span>' +
      '<span class="log-op">' + U.escapeHtml(row.op || '-') + '</span>' +
      (row.tool ? '<span class="muted">' + U.escapeHtml(row.tool) + '</span>' : '') +
      '<span class="log-actor">' + U.escapeHtml(actor) + '</span>' +
      '</div>' +
      '<div class="log-detail"><pre>' + U.escapeHtml(JSON.stringify({ request: row.request, result: row.result }, null, 2)) + '</pre></div>' +
      '</div>';
  }

  // level 会落在 class 属性位，只允许这些取值（app.css 里有对应样式）
  const LEVEL_CLASSES = ['debug', 'info', 'warning', 'error', 'critical'];

  function logRowHtml(row) {
    const proc = row.proc ? '<span class="proc-badge">' + U.escapeHtml(row.proc) + '</span>' : '';
    // row.level 来自跨进程共享的日志文件，不能直接拼进 class 属性位
    const level = String(row.level || 'INFO');
    const levelKey = LEVEL_CLASSES.indexOf(level.toLowerCase()) >= 0 ? level.toLowerCase() : 'info';
    return '<div class="log-row static">' +
      '<div class="log-line">' +
      '<span class="log-ts">' + U.fmtTime(row.ts) + '</span>' +
      '<span class="level-badge ' + levelKey + '">' + U.escapeHtml(level) + '</span>' +
      proc +
      '<span class="log-logger">' + U.escapeHtml(row.logger || '') + '</span>' +
      '<span class="log-msg">' + U.escapeHtml(row.message || '') + '</span>' +
      '</div></div>';
  }

  function renderMetrics(data) {
    const grid = $('metric-grid');
    if (!grid) return;
    const g = data.graph || {};
    const o = data.oplog || {};
    const a = data.applog || {};
    const r = data.requests || {};
    const l = data.logs || {};
    const cards = [
      ['节点', g.nodes, ''],
      ['边', g.edges, ''],
      ['孤立节点', g.orphans, '无出入边'],
      ['废弃 / 遗忘', (g.deprecated || 0) + ' / ' + (g.forgotten || 0), ''],
      ['操作日志', o.lines, U.fmtBytes(o.size_bytes) + (o.exists ? '' : ' · 缺失')],
      ['服务日志', a.lines, U.fmtBytes(a.size_bytes) + (a.exists ? '' : ' · 缺失')],
      ['运行日志缓冲', l.buffered, l.subscribers + ' 订阅'],
      ['运行时长', U.fmtDuration(data.uptime_seconds), ''],
      ['请求总数', r.total, r.errors + ' 错误'],
    ];
    grid.innerHTML = cards.map(function (c) {
      return '<div class="metric-card"><div class="metric-label">' + c[0] + '</div>' +
        '<div class="metric-value">' + U.escapeHtml(String(c[1] === undefined || c[1] === null ? '-' : c[1])) + '</div>' +
        '<div class="metric-sub">' + U.escapeHtml(String(c[2] || '')) + '</div></div>';
    }).join('');
  }

  // =========================================================
  // 设置
  // =========================================================

  const settingsPanel = {
    configData: null,

    init() {
      const host = $('view-settings');
      host.innerHTML = '';
      this.render(host);
      if (A.api.isLive) this.loadConfig();
    },

    render(host) {
      host.innerHTML = '';
      const s = A.store.settings.render;

      // 1. 渲染与布局
      const renderSection = section('渲染与布局', '实时生效，保存在浏览器本地');
      renderSection.body.appendChild(selectField('性能档位', 'quality', [
        ['high', '高质量'], ['balanced', '均衡'], ['performance', '性能优先'],
      ], s.quality));
      renderSection.body.appendChild(checkField('网格底纹（56px）', 'grid', s.grid !== false));
      renderSection.body.appendChild(rangeField('节点尺寸', 'nodeScale', 0.4, 2, 0.1, s.nodeScale));
      renderSection.body.appendChild(rangeField('连线宽度', 'linkWidth', 0.2, 3, 0.1, s.linkWidth));
      renderSection.body.appendChild(rangeField('流向粒子', 'particles', 0, 3, 1, s.particles));
      renderSection.body.appendChild(rangeField('斥力强度', 'chargeStrength', -400, -5, 5, s.chargeStrength));
      renderSection.body.appendChild(rangeField('连线距离', 'linkDistance', 8, 80, 1, s.linkDistance));
      renderSection.body.appendChild(selectField('标签显示', 'labels', [
        ['none', '不显示'], ['focus', '聚焦时'], ['hover', '悬浮/选中'], ['all', '全部'],
      ], s.labels));
      renderSection.body.appendChild(checkField('显示方向箭头', 'showArrows', s.showArrows));
      renderSection.body.appendChild(checkField('自动旋转', 'autoRotate', s.autoRotate));
      renderSection.body.appendChild(checkField('管道式连线', 'showPipes', s.showPipes));
      const resetRow = el('div', { class: 'btn-row' });
      resetRow.appendChild(el('button', {
        class: 'btn', text: '恢复默认',
        onclick: function () { A.store.reset(); settingsPanel.render(host); A.graph.applySettings(A.store.settings); U.toast('已恢复默认设置', 'ok'); },
      }));
      renderSection.body.appendChild(resetRow);
      host.appendChild(renderSection.root);

      // 2. 过滤预设
      const presetSection = section('过滤预设', '把节点/边过滤组合保存为命名方案');
      const presetList = el('div', { class: 'preset-list', id: 'settings-preset-list' });
      presetSection.body.appendChild(presetList);
      const presetActions = el('div', { class: 'btn-row' });
      presetActions.appendChild(el('button', { class: 'btn', text: '保存当前过滤为预设', onclick: function () { saveCurrentPreset(); } }));
      presetActions.appendChild(el('button', { class: 'btn', text: '导出预设 JSON', onclick: function () { U.download('ariadne_filter_presets.json', JSON.stringify(A.store.listPresets(), null, 2), 'application/json'); } }));
      presetActions.appendChild(el('button', { class: 'btn', text: '导入预设', onclick: function () { importPresets(); } }));
      presetSection.body.appendChild(presetActions);
      host.appendChild(presetSection.root);
      this.refreshPresets();

      // 3. 服务端配置
      const configSection = section('服务端配置', '只读；密钥类环境变量已脱敏');
      configSection.body.appendChild(el('div', { class: 'config-box', id: 'config-box' }, [
        el('div', { class: 'hint', text: A.api.isLive ? '加载中…' : '离线 HTML 模式：无服务端配置。' }),
      ]));
      host.appendChild(configSection.root);

      // 4. MCP 连接（端点 / 鉴权 / 探活 / 客户端接入片段）
      const mcpSection = section('MCP 连接', 'MCP 与面板是两个进程；这里显示接入端点、鉴权方式与探活情况');
      const mcpActions = el('div', { class: 'btn-row' });
      mcpActions.appendChild(el('button', {
        class: 'btn', text: '重新探测',
        onclick: function () { mcpStatus.load(); U.toast('已重新探测', 'ok'); },
      }));
      mcpSection.body.appendChild(mcpActions);
      mcpSection.body.appendChild(el('div', { class: 'config-box', id: 'mcp-box' }, [
        el('div', { class: 'hint', text: A.api.isLive ? '加载中…' : '离线 HTML 模式：无法探测 MCP。' }),
      ]));
      mcpSection.body.appendChild(el('div', { class: 'snippets', id: 'mcp-snippets' }));
      host.appendChild(mcpSection.root);
      if (A.api.isLive) mcpStatus.load();

      // 5. 数据导入/导出
      const dataSection = section('数据导入 / 导出', '导入前会校验并自动备份现有 YAML（.bak）');
      if (A.api.isLive) {
        const row = el('div', { class: 'btn-row' });
        row.appendChild(el('a', { class: 'btn', href: A.api.exportYamlUrl(), download: '', text: '导出图谱 YAML' }));
        row.appendChild(el('a', { class: 'btn', href: A.api.exportOplogUrl(), download: '', text: '导出操作日志' }));
        dataSection.body.appendChild(row);
        const file = el('input', { type: 'file', accept: '.yaml,.yml', class: 'input' });
        file.addEventListener('change', function () {
          importYamlFile(file.files[0]);
          file.value = '';   // 清空后才能再次选中同一个文件（否则不会再触发 change）
        });
        dataSection.body.appendChild(el('label', { class: 'field' }, [el('span', { text: '导入图谱 YAML' }), file]));
      } else {
        dataSection.body.appendChild(el('div', { class: 'hint', text: '离线 HTML 模式：数据操作不可用。' }));
      }
      host.appendChild(dataSection.root);
    },

    refreshPresets() {
      const list = $('settings-preset-list');
      if (!list) return;
      const presets = A.store.listPresets();
      if (!presets.length) { list.innerHTML = '<div class="hint">暂无预设</div>'; return; }
      list.innerHTML = '';
      presets.forEach(function (p) {
        const row = el('div', { class: 'preset-item' });
        row.appendChild(el('span', { class: 'preset-name', text: p.name }));
        row.appendChild(el('span', { class: 'muted', text: U.fmtTime(p.savedAt) }));
        row.appendChild(el('button', {
          class: 'btn small', text: '应用',
          onclick: function () { A.graph.setFilter(p.state); filtersPanel.sync(A.graph.getFilter()); U.toast('已应用：' + p.name, 'ok'); },
        }));
        row.appendChild(el('button', {
          class: 'btn small danger', text: '删除',
          onclick: function () { A.store.deletePreset(p.id); settingsPanel.refreshPresets(); refreshPresetSelect(); },
        }));
        list.appendChild(row);
      });
    },

    async loadConfig() {
      const box = $('config-box');
      if (!box) return;
      try {
        const data = await A.api.getConfig();
        this.configData = data;
        const rows = [
          ['图谱路径', data.yaml_path],
          ['监听地址', data.host + ':' + data.port],
          ['操作日志', data.oplog_path],
          ['服务日志', data.log_file_path],
          ['鉴权', '开启（' + (data.current_user || data.auth_user) + '）'],
          ['凭据文件', data.auth_path],
          ['日志级别', data.log_level],
          ['版本', data.version + ' · Python ' + data.python],
        ];
        Object.keys(data.env || {}).forEach(function (k) { rows.push(['ENV · ' + k, data.env[k]]); });
        box.innerHTML = rows.map(function (r) {
          return '<div class="config-row"><span class="config-key">' + U.escapeHtml(r[0]) + '</span><span class="config-value">' + U.escapeHtml(String(r[1] || '-')) + '</span></div>';
        }).join('');
      } catch (e) {
        box.innerHTML = '<div class="hint error">加载配置失败：' + U.escapeHtml(e.message) + '</div>';
      }
    },
  };

  // ---- MCP 连接：顶栏徽章 + 设置页配置盒 ----
  // 面板与 MCP 是两个进程，状态由后端探活 + 共享操作日志统计得出（见 /api/mcp/status）。

  const mcpStatus = {
    data: null,

    async load() {
      if (!A.api.isLive) return null;
      try {
        this.data = await A.api.getMcpStatus();
      } catch (e) {
        this.data = { error: e.message || String(e) };
      }
      renderMcpPill(this.data);
      if (shell.activeTab === 'settings') renderMcpBox(this.data);
      return this.data;
    },
  };

  function renderMcpPill(data) {
    const pill = $('mcp-pill');
    if (!pill) return;
    // 只增删状态类，不做整串 className 赋值——否则会把 wireMcpPill 加的
    // pill-btn（手型光标与 hover 样式）一起抹掉。
    pill.classList.remove('ok', 'warn');
    if (!A.api.isLive) {
      pill.textContent = 'MCP —';
      return;
    }
    if (!data || data.error) {
      pill.textContent = 'MCP ?';
      pill.classList.add('warn');
      pill.title = 'MCP 状态未知：' + ((data && data.error) || '');
      return;
    }
    if (data.online) {
      pill.textContent = 'MCP 在线' + (data.latency_ms == null ? '' : ' · ' + data.latency_ms + 'ms');
      pill.classList.add('ok');
    } else {
      pill.textContent = 'MCP 离线';
      pill.classList.add('warn');
    }
    pill.title = 'MCP 端点 ' + data.endpoint + '（点击查看配置与状态）';
  }

  function renderMcpBox(data) {
    const box = $('mcp-box');
    if (!box) return;
    if (!data) { box.innerHTML = '<div class="hint">未探测，点「重新探测」。</div>'; return; }
    if (data.error) {
      box.innerHTML = '<div class="hint error">探测失败：' + U.escapeHtml(data.error) + '</div>';
      return;
    }
    const badge = data.online
      ? '<span class="mcp-badge ok">在线</span>' + (data.latency_ms == null ? '' : '（' + data.latency_ms + ' ms）')
      : '<span class="mcp-badge off">离线</span>' +
        (data.error ? '<span class="probe-error">' + U.escapeHtml(data.error) + '</span>' : '');
    const auth = 'HTTP Basic（用户 ' + (data.auth_user || '-') + '）' + (data.bearer_configured
      ? ' 或 Bearer Token（' + U.escapeHtml(data.bearer_hint || '已配置') + '）'
      : '；未配置 ARIADNE_MCP_TOKEN');
    const source = data.endpoint_source === 'ARIADNE_MCP_URL'
      ? 'ARIADNE_MCP_URL（显式配置）'
      : 'ARIADNE_PORT / ARIADNE_MCP_HOST（推导，默认 8765）';
    const rows = [
      ['接入端点', U.escapeHtml(data.endpoint)],
      ['端点来源', U.escapeHtml(source)],
      ['探活', badge],
      ['鉴权方式', auth],
      ['最近 MCP 操作', U.escapeHtml(data.last_op ? data.last_op + '　' + U.fmtTime(data.last_op_ts) : '暂无记录')],
      ['近 10 分钟 / 累计', (data.ops_10min || 0) + ' 次 / ' + (data.ops_total || 0) + ' 次'],
      ['操作日志', U.escapeHtml(data.oplog_path || '')],
      ['探测时间', U.escapeHtml(U.fmtTime(data.checked_at))],
    ];
    box.innerHTML = rows.map(function (r) {
      return '<div class="config-row"><span class="config-key">' + U.escapeHtml(r[0]) +
        '</span><span class="config-value">' + r[1] + '</span></div>';
    }).join('') + (data.must_change
      ? '<div class="hint warn">凭据仍是初始密码：MCP 会拒绝 Basic 接入，请先在面板改密。</div>'
      : '');
    renderMcpSnippets(data);
  }

  function renderMcpSnippets(data) {
    const host = $('mcp-snippets');
    if (!host || !data || !data.snippets) return;
    const items = [
      ['客户端接入 · Bearer（推荐）', 'bearer'],
      ['客户端接入 · Basic', 'basic'],
      ['本地 stdio（Claude Desktop / IDE）', 'stdio'],
    ];
    host.innerHTML = items.map(function (it) {
      return '<div class="snippet"><div class="snippet-head"><span>' + U.escapeHtml(it[0]) +
        '</span><button class="btn small" data-copy="' + it[1] + '">复制</button></div>' +
        '<pre>' + U.escapeHtml(data.snippets[it[1]] || '') + '</pre></div>';
    }).join('');
    U.$$('#mcp-snippets [data-copy]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const s = mcpStatus.data && mcpStatus.data.snippets;
        if (s && s[btn.dataset.copy]) U.copy(s[btn.dataset.copy]);
      });
    });
  }

  function wireMcpPill() {
    const pill = $('mcp-pill');
    if (!pill) return;
    pill.addEventListener('click', function () {
      shell.openDrawer('settings');
      // 抽屉有展开动画，稍后再滚动到 MCP 区块
      setTimeout(function () {
        const box = $('mcp-box');
        if (box && box.scrollIntoView) box.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }, 220);
    });
  }

  // =========================================================
  // 运行时参数（权重矩阵 / 检索规模 · 种子 / 目的回归）
  // 与 MCP 共用同一份 YAML：面板保存后，MCP 在下一次检索/写入时自动生效。
  // =========================================================

  const paramsPanel = {
    data: null,
    draft: null,

    init() {
      const host = $('view-params');
      if (!host) return;
      if (!A.api.isLive) {
        host.innerHTML = '<div class="hint">离线 HTML 模式：参数调整需要连接面板服务' +
          '（参数由面板与 MCP 共享读写，离线副本无法落盘）。</div>';
        return;
      }
      host.innerHTML = '<div class="hint">加载中…</div>';
      this.load();
    },

    async load() {
      const host = $('view-params');
      try {
        const data = await A.api.getParams();
        this.data = data;
        this.draft = JSON.parse(JSON.stringify(data.params));
        this.render(host);
      } catch (e) {
        host.innerHTML = '<div class="hint error">加载参数失败：' + U.escapeHtml(e.message) + '</div>';
      }
    },

    render(host) {
      host.innerHTML = '';
      const spec = this.data.spec;

      // 0. 来源
      const info = section('参数来源', '与 MCP 共用同一份文件，保存即生效，无需重启');
      info.body.appendChild(el('div', { class: 'config-box' }, [
        el('div', { class: 'config-row' }, [
          el('span', { class: 'config-key', text: '参数文件' }),
          el('span', {
            class: 'config-value',
            text: this.data.path + (this.data.exists ? '' : '（尚未创建，当前显示默认值；保存后创建）'),
          }),
        ]),
      ]));
      host.appendChild(info.root);

      // 1. 权重矩阵：论文式全矩阵（行=节点类型，列=关系类型，格内=正向/反向）
      const wSection = section('权重矩阵', '6 种节点类型 × 8 种关系；格内为「正向 / 反向」，权重 0 表示该方向不扩展');
      wSection.body.appendChild(el('div', { class: 'param-warning' }, [
        el('strong', { text: '重要提示：' }),
        el('span', {
          text: '权重矩阵决定联想扩张的走向与检索召回范围，改动会全局影响 MCP 的记忆检索结果，' +
            '并对所有会话即时生效。请确认影响面后再保存。',
        }),
      ]));
      wSection.body.appendChild(this.weightMatrix(spec));
      wSection.body.appendChild(el('div', { class: 'wmatrix-legend' }, [
        el('span', { text: '格内先后两个数字即' }),
        el('strong', { text: '正向 / 反向' }),
        el('span', { text: '：正向沿出边扩展（当前节点 → 边指向的目标），反向沿入边回溯' +
          '（指向当前节点的来源 → 当前节点）；列头小字即该方向实际取到的节点。' }),
      ]));
      const bidirectional = spec.relations
        .filter(function (r) { return r.bidirectional; })
        .map(function (r) { return r.label; })
        .join(' / ');
      if (bidirectional) {
        wSection.body.appendChild(el('div', { class: 'wmatrix-legend' }, [
          el('span', { text: bidirectional + ' 在图中会自动补一条反向边，两侧取到的是同一批邻居，' +
            '因此只标注对端对象并记作「双向」。' }),
        ]));
      }
      host.appendChild(wSection.root);

      // 2/3. 标量参数分组
      spec.groups.forEach((g) => {
        const sec = section(g.label, '');
        spec.fields.filter(function (f) { return f.group === g.key; }).forEach((f) => {
          sec.body.appendChild(this.scalarRow(f));
        }, this);
        host.appendChild(sec.root);
      });

      // 4. 操作
      const actions = el('div', { class: 'btn-row end param-actions' });
      actions.appendChild(el('button', { class: 'btn', text: '重新加载', onclick: () => this.load() }));
      actions.appendChild(el('button', { class: 'btn danger', text: '恢复默认', onclick: () => this.reset() }));
      actions.appendChild(el('button', { class: 'btn primary', text: '保存并生效', onclick: () => this.save() }));
      host.appendChild(actions);
    },

    // 6 行（节点类型）× 8 列（关系类型）的全矩阵，与论文表格同构
    weightMatrix(spec) {
      const table = el('div', { class: 'wmatrix' });
      const head = el('div', { class: 'wmatrix-row head' }, [
        el('span', { class: 'wmatrix-corner', text: '节点类型' }),
      ]);
      spec.relations.forEach((rel) => {
        head.appendChild(el('span', { class: 'wmatrix-col' }, [
          el('span', { class: 'wmatrix-col-key', text: rel.label }),
          el('span', {
            class: 'wmatrix-col-dir',
            text: rel.bidirectional
              ? rel.forward + '（双向）'
              : rel.forward + ' / ' + rel.reverse,
          }),
        ]));
      });
      table.appendChild(head);

      spec.node_types.forEach((nt) => {
        const rels = (this.draft.weights && this.draft.weights[nt.key]) || {};
        const row = el('div', { class: 'wmatrix-row' }, [
          el('span', { class: 'wmatrix-rowhead' }, [
            el('span', { text: nt.label }),
            nt.cn ? el('span', { class: 'type-en', text: nt.cn }) : null,
          ]),
        ]);
        spec.relations.forEach((rel) => {
          const pair = rels[rel.key] || [0, 0];
          row.appendChild(this.weightCell(nt, rel, pair[0], pair[1]));
        });
        table.appendChild(row);
      });
      return table;
    },

    // 一个格子 = 正向 / 反向，两个数字紧挨着以便像论文表格那样一眼对照
    weightCell(nt, rel, forward, reverse) {
      return el('span', { class: 'wmatrix-cell' }, [
        this.weightInput(nt, rel, 0, forward),
        el('span', { class: 'wmatrix-slash', text: '/' }),
        this.weightInput(nt, rel, 1, reverse),
      ]);
    },

    weightInput(nt, rel, index, value) {
      const input = el('input', { class: 'input tiny', type: 'number', min: 0, max: 1, step: 0.05 });
      input.value = value;
      const dirText = index === 0 ? '正向' : '反向';
      input.title = dirText + '（' + (index === 0 ? '沿出边 →' : '沿入边 ←') + '）：' +
        (index === 0 ? rel.forward : rel.reverse);
      input.addEventListener('input', () => {
        const label = nt.label + ' × ' + rel.label + ' ' + dirText;
        const v = this.parseNumeric(input, input.value, 0, 1, 'float', label);
        if (v === null) return;
        const weights = this.draft.weights || (this.draft.weights = {});
        const rels = weights[nt.key] || (weights[nt.key] = {});
        const pair = rels[rel.key] || [0, 0];
        pair[index] = v;
        rels[rel.key] = pair;
      });
      return input;
    },

    // 解析并校验一个数字输入：合法则清除标红并返回数值；非法则标红、记录原因并返回 null
    parseNumeric(input, raw, min, max, kind, label) {
      const text = String(raw === undefined || raw === null ? '' : raw).trim();
      let message = '';
      let value = NaN;
      if (text === '') {
        message = label + '：不能为空';
      } else {
        value = Number(text);
        if (!isFinite(value)) {
          message = label + '：必须是' + (kind === 'int' ? '整数' : '数值');
        } else if (kind === 'int' && !Number.isInteger(value)) {
          message = label + '：必须是整数';
        } else if (value < min || value > max) {
          message = label + '：需在 ' + min + ' ~ ' + max + ' 之间';
        }
      }
      if (message) {
        input.classList.add('invalid');
        input.setAttribute('data-invalid-msg', message);
        return null;
      }
      input.classList.remove('invalid');
      input.removeAttribute('data-invalid-msg');
      return value;
    },

    scalarRow(f) {
      const input = el('input', { class: 'input small', type: 'number', min: f.min, max: f.max, step: f.step });
      input.value = this.draft[f.group][f.key];
      input.addEventListener('input', () => {
        const v = this.parseNumeric(input, input.value, f.min, f.max, f.kind, f.label);
        if (v === null) return;
        this.draft[f.group][f.key] = v;
      });
      return el('div', { class: 'param-row' }, [
        el('span', { class: 'param-name', text: f.label }),
        input,
        el('span', { class: 'param-desc muted', text: f.desc + '（范围 ' + f.min + ' ~ ' + f.max + '）' }),
      ]);
    },

    // 收集当前所有未通过校验的输入（标红时写入了 data-invalid-msg）
    collectInvalid() {
      return Array.prototype.slice
        .call(document.querySelectorAll('#view-params input[data-invalid-msg]'))
        .map(function (input) { return input.getAttribute('data-invalid-msg'); });
    },

    showInvalid(items) {
      const overlay = el('div', { class: 'modal-overlay' });
      const box = el('div', { class: 'modal' });
      box.appendChild(el('h3', { text: '参数校验未通过' }));
      const form = el('div', { class: 'modal-form' });
      form.appendChild(el('div', {
        class: 'hint error',
        text: '以下 ' + items.length + ' 处输入不合法（已在表单中用红框标出），修正后才能保存：',
      }));
      const list = el('ul', { class: 'invalid-list' });
      items.forEach(function (message) { list.appendChild(el('li', { text: message })); });
      form.appendChild(list);
      box.appendChild(form);
      const actions = el('div', { class: 'btn-row end' });
      actions.appendChild(el('button', {
        class: 'btn primary', text: '知道了',
        onclick: function () { overlay.remove(); },
      }));
      box.appendChild(actions);
      overlay.appendChild(box);
      overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
      document.body.appendChild(overlay);
    },

    weightChangeCount() {
      const before = (this.data && this.data.params && this.data.params.weights) || {};
      const after = (this.draft && this.draft.weights) || {};
      let count = 0;
      Object.keys(after).forEach((nodeType) => {
        const relsBefore = before[nodeType] || {};
        const relsAfter = after[nodeType] || {};
        Object.keys(relsAfter).forEach((rel) => {
          const a = relsBefore[rel] || [0, 0];
          const b = relsAfter[rel] || [0, 0];
          if (Number(a[0]) !== Number(b[0]) || Number(a[1]) !== Number(b[1])) count += 1;
        });
      });
      return count;
    },

    // 权重矩阵改动属于高风险操作：保存前用红字二次确认
    confirmWeightSave(count) {
      return new Promise((resolve) => {
        const overlay = el('div', { class: 'modal-overlay' });
        const box = el('div', { class: 'modal' });
        box.appendChild(el('h3', { text: '确认调整权重矩阵' }));
        const form = el('div', { class: 'modal-form' });
        form.appendChild(el('div', { class: 'param-warning' }, [
          el('strong', { text: '重要提示：' }),
          el('span', {
            text: '权重矩阵决定联想扩张的走向与检索召回范围，改动会全局影响 MCP 的记忆检索结果，' +
              '并对所有会话即时生效。请确认影响面后再保存。',
          }),
        ]));
        form.appendChild(el('div', {
          class: 'hint',
          text: '本次共有 ' + count + ' 项权重发生变化。',
        }));
        box.appendChild(form);
        const actions = el('div', { class: 'btn-row end' });
        actions.appendChild(el('button', {
          class: 'btn', text: '取消',
          onclick: function () { overlay.remove(); resolve(false); },
        }));
        actions.appendChild(el('button', {
          class: 'btn danger', text: '确认保存',
          onclick: function () { overlay.remove(); resolve(true); },
        }));
        box.appendChild(actions);
        overlay.appendChild(box);
        overlay.addEventListener('click', function (e) {
          if (e.target === overlay) { overlay.remove(); resolve(false); }
        });
        document.body.appendChild(overlay);
      });
    },

    async save() {
      const invalid = this.collectInvalid();
      if (invalid.length) { this.showInvalid(invalid); return; }
      const changed = this.weightChangeCount();
      if (changed > 0 && !(await this.confirmWeightSave(changed))) return;
      try {
        const res = await A.api.saveParams(this.draft);
        this.applyServerState(res);
        this.render($('view-params'));
        U.toast('参数已保存，MCP 将在下一次检索/写入时生效', 'ok');
      } catch (e) {
        U.toast('保存失败：' + e.message, 'error');
      }
    },

    async reset() {
      try {
        const res = await A.api.resetParams();
        this.applyServerState(res);
        this.render($('view-params'));
        U.toast('已恢复默认参数', 'ok');
      } catch (e) {
        U.toast('恢复失败：' + e.message, 'error');
      }
    },

    // 用服务端返回的最新参数刷新本地状态。必须同时更新 data.params：
    // 它是 weightChangeCount() 的对比基准，只更新 draft 会导致
    // 保存成功后再次点击「保存并生效」仍弹「有 N 项权重发生变化」。
    applyServerState(res) {
      this.draft = JSON.parse(JSON.stringify(res.params));
      this.data.path = res.path;
      this.data.exists = true;
      this.data.mtime = res.mtime;
      this.data.params = JSON.parse(JSON.stringify(res.params));
    },
  };

  function section(title, subtitle) {
    const root = el('div', { class: 'settings-section' });
    root.appendChild(el('div', { class: 'settings-title' }, [
      el('span', { text: title }),
      subtitle ? el('span', { class: 'muted', text: subtitle }) : null,
    ]));
    const body = el('div', { class: 'settings-body' });
    root.appendChild(body);
    return { root: root, body: body };
  }

  function selectField(label, key, options, value) {
    const select = el('select', { class: 'input' });
    options.forEach(function (o) {
      const opt = el('option', { value: o[0], text: o[1] });
      if (o[0] === value) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', function () { applySetting(key, select.value); });
    return el('label', { class: 'field inline' }, [el('span', { text: label }), select]);
  }

  function rangeField(label, key, min, max, step, value) {
    const input = el('input', { type: 'range', min: min, max: max, step: step });
    input.value = value;
    const out = el('span', { class: 'range-value', text: value });
    input.addEventListener('input', function () { out.textContent = input.value; });
    input.addEventListener('change', function () { applySetting(key, Number(input.value)); });
    return el('label', { class: 'field inline' }, [el('span', { text: label }), input, out]);
  }

  function checkField(label, key, value) {
    const input = el('input', { type: 'checkbox' });
    input.checked = !!value;
    input.addEventListener('change', function () { applySetting(key, input.checked); });
    return el('label', { class: 'toggle' }, [input, el('span', { text: label })]);
  }

  function applySetting(key, value) {
    A.store.update('render', key, value);
    A.graph.applySettings(A.store.settings);
  }

  function importPresets() {
    const file = el('input', { type: 'file', accept: '.json' });
    file.addEventListener('change', function () {
      const f = file.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = function () {
        try {
          const presets = JSON.parse(reader.result);
          if (!Array.isArray(presets)) throw new Error('格式错误');
          presets.forEach(function (p) {
            if (p && p.name && p.state) A.store.savePreset(p.name, p.state);
          });
          refreshPresetSelect();
          settingsPanel.refreshPresets();
          U.toast('已导入 ' + presets.length + ' 个预设', 'ok');
        } catch (e) { U.toast('导入预设失败：' + e.message, 'error'); }
      };
      reader.readAsText(f);
    });
    file.click();
  }

  function importYamlFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async function () {
      if (!window.confirm('导入将覆盖当前图谱（自动备份为 .bak），确认继续？')) return;
      try {
        const result = await A.api.importYaml(reader.result);
        U.toast('导入成功：' + result.nodes + ' 节点 / ' + result.edges + ' 边', 'ok');
        A.graph.select(null);
        await ctx.reloadGraph(null);
      } catch (e) { U.toast('导入失败：' + e.message, 'error'); }
    };
    reader.readAsText(file);
  }

  // =========================================================
  // 初始化
  // =========================================================

  function init(context) {
    ctx = context;
    shell.init();
    filtersPanel.init();
    inspector.render(null);
    initSearch();
    initContextMenu();
    observability.init();
    settingsPanel.init();
    cameraPanel.init();
    const undoBtn = $('btn-undo');
    if (undoBtn) undoBtn.addEventListener('click', doUndo);
    if (undoBtn) undoBtn.disabled = true;

    wireTopbar();
    wireAuth();
    setInterval(refreshHealth, 15000);
    refreshHealth();
    // MCP 连接：顶栏徽章每 30s 探一次；设置页每次打开时也会刷新
    wireMcpPill();
    setInterval(function () { if (!document.hidden) mcpStatus.load(); }, 30000);
    mcpStatus.load();
  }

  // 相机悬浮窗：拖动滚动条控制缩放，并实时显示相机位置
  const cameraPanel = {
    init() {
      const slider = $('camera-zoom');
      if (!slider || !A.graph.onCameraChange) return;
      slider.addEventListener('input', function () {
        A.graph.setCameraDistance(A.graph.sliderToDistance(Number(slider.value)));
      });
      A.graph.onCameraChange(function (st) { cameraPanel.render(st); });
      this.render(A.graph.cameraState());
    },

    render(st) {
      if (!st) return;
      setStatus('cam-distance', st.distance.toFixed(0));
      setStatus('cam-x', st.position.x.toFixed(1));
      setStatus('cam-y', st.position.y.toFixed(1));
      setStatus('cam-z', st.position.z.toFixed(1));
      const slider = $('camera-zoom');
      // 用户正在拖动时不回写，否则会与输入打架
      if (slider && document.activeElement !== slider) {
        slider.value = A.graph.distanceToSlider(st.distance);
      }
    },
  };

  // 顶栏「当前用户 + 登出」：面板始终鉴权，故必定显示
  async function wireAuth() {
    if (!A.api.isLive) return;
    let cfg;
    try { cfg = await A.api.getConfig(); } catch (e) { return; }
    // 初始密码尚未修改：先去改密页
    if (cfg.must_change) { location.href = changeUrl(); return; }

    const pill = $('user-pill');
    if (pill) {
      pill.textContent = cfg.current_user || cfg.auth_user || '已登录';
      pill.title = '当前登录用户（点击修改密码）';
      pill.hidden = false;
      pill.addEventListener('click', function () { location.href = changeUrl(); });
    }
    const btn = $('btn-logout');
    if (btn) {
      btn.hidden = false;
      btn.addEventListener('click', async function () {
        try { await A.api.logout(); } catch (e) { /* 无论成功与否都回登录页 */ }
        location.href = '/login';
      });
    }
  }

  // 改密页地址（带上当前位置，改完回到这里）
  function changeUrl() {
    return '/login?mode=change&next=' + encodeURIComponent(location.pathname + location.search);
  }

  function wireTopbar() {
    const fit = $('btn-zoom-fit');
    if (fit) fit.addEventListener('click', function () { A.graph.zoomFit(); });
    const auto = $('btn-autorotate');
    if (auto) {
      auto.classList.toggle('active', !!A.store.settings.render.autoRotate);
      auto.addEventListener('click', function () {
        const next = !A.store.settings.render.autoRotate;
        applySetting('autoRotate', next);
        auto.classList.toggle('active', next);
        if (shell.activeTab === 'settings') settingsPanel.render($('view-settings'));
      });
    }
    const searchToggle = $('btn-search');
    if (searchToggle) searchToggle.addEventListener('click', function () { $('search-input').focus(); });
  }

  async function refreshHealth() {
    const pill = $('health-pill');
    if (!pill) return;
    if (!A.api.isLive) { pill.textContent = '离线'; pill.className = 'pill'; return; }
    try {
      const health = await A.api.getHealth();
      pill.textContent = health.status === 'ok' ? '服务正常' : '服务降级';
      pill.className = 'pill ' + (health.status === 'ok' ? 'ok' : 'warn');
    } catch (e) {
      pill.textContent = '服务不可用';
      pill.className = 'pill error';
    }
  }

  A.panels = {
    init: init,
    renderInspector: renderInspector,
    updateStatusbar: updateStatusbar,
    showTooltip: showTooltip,
    showEdgeTooltip: showEdgeTooltip,
    showContextMenu: showContextMenu,
    handleEdgeAction: handleEdgeAction,
    hideEdgeBanner: hideEdgeBanner,
    undo: doUndo,
    shell: shell,
  };
})();
