/* SPDX-License-Identifier: AGPL-3.0-only */
/* 3D 力导向图谱：几何、光照、连线、标签、聚焦/搜索/连边交互 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});
  const C = A.const;
  const U = A.util;

  const REFRESH_TICKS = 160;   // 布局收敛后自动缩放到位
  let settleTicks = REFRESH_TICKS;
  let zoomed = false;

  const state = {
    nodes: [],
    links: [],
    byId: new Map(),
    mutual: new Set(),
    filter: {
      nodeTypes: {}, edgeTypes: {},
      showDeprecated: true, showForgotten: false,
      hideOrphans: false, highDegOnly: false,
    },
    medianDegree: 1,
    settings: C.DEFAULT_SETTINGS,
    selectedId: null,
    focusId: null,
    focusSet: new Set(),
    focusHops: 1,          // 用户当前用的聚焦跳数，数据重载时据此还原而不是硬编码 1
    hoverId: null,
    fixed: new Set(),
    groups: new Map(),
    materials: new Map(),
    labels: new Map(),
    pins: new Map(),
    edgeMode: null,
    // 调用测试：本次链路"经过"的节点（激活）与 StoryRank 采纳的节点
    activated: new Set(),
    adopted: new Set(),
  };

  let graph = null;
  let hooks = {};
  let renderHandle = null;
  let containerEl = null;

  // ---- 初始化 ----

  function init(opts) {
    hooks = opts.hooks || {};
    state.settings = opts.settings || state.settings;
    containerEl = opts.container;

    graph = ForceGraph3D({ controlType: 'orbit' })(opts.container)
      // 透明画布：与官网一致，让页面 56px 网格底纹透出
      .backgroundColor('rgba(0,0,0,0)')
      .nodeVal(function (n) { return n.radius; })
      .nodeThreeObject(createNodeObject)
      .linkColor(function (l) { return C.EDGE_COLORS[l.rel_type] || '#64748b'; })
      .linkWidth(linkWidthFor)
      .linkOpacity(0.68)
      .linkCurvature(curvatureFor)
      .linkDirectionalParticles(function () { return state.settings.render.particles; })
      .linkDirectionalParticleWidth(1.0)
      .linkDirectionalParticleSpeed(function (l) { return particleSpeed(l.rel_type); })
      .linkDirectionalParticleColor(function (l) { return C.EDGE_COLORS[l.rel_type] || '#64748b'; })
      .linkDirectionalArrowLength(arrowFor)
      .linkDirectionalArrowRelPos(1)
      .linkDirectionalArrowColor(function (l) { return C.EDGE_COLORS[l.rel_type] || '#64748b'; })
      .onNodeClick(handleNodeClick)
      .onNodeRightClick(handleNodeRightClick)
      .onNodeHover(handleHover)
      .onLinkHover(handleLinkHover)
      .onNodeDragEnd(handleDragEnd)
      .onBackgroundClick(handleBackgroundClick)
      .onBackgroundRightClick(function (evt) { if (hooks.onContextMenu) hooks.onContextMenu(null, evt); })
      .onEngineStop(function () {
        if (!zoomed && state.nodes.length) { zoomFit(); zoomed = true; }
      })
      .onEngineTick(function () {
        if (settleTicks > 0) {
          settleTicks--;
          if (settleTicks === 0) { applyForces(); zoomFit(); zoomed = true; }
        }
      });

    addLights();
    window.addEventListener('resize', resize);
    startRenderLoop();
    resize();
  }

  function addLights() {
    // 光照总辐照控制在 ~1.0：超过 1 会把材质颜色裁切到接近白，
    // 在浅色底上表现为「粉彩褪色」。这里环境光 0.55 + 主光 0.40 + 轮廓光 0.15。
    const scene = graph.scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.4);
    key.position.set(1, 1, 1);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.15);
    rim.position.set(-1, -1, -0.6);
    scene.add(rim);
  }

  function resize() {
    if (!graph || !containerEl) return;
    graph.width(containerEl.clientWidth || window.innerWidth)
      .height(containerEl.clientHeight || window.innerHeight - 76);
  }

  // ---- 数据 ----

  function setData(data) {
    // 重载后节点是**全新**的对象，fx/fy/fz 与钉圈都不在了。先把当前钉住的位置
    // 记下来，重建后按 id 还原，否则每 15s 的轮询都会把用户钉住的节点散开。
    const pinned = [];
    state.fixed.forEach(function (id) {
      const prev = state.byId.get(id);
      if (prev && prev.x !== undefined) pinned.push({ id: id, x: prev.x, y: prev.y, z: prev.z });
    });

    state.nodes = (data && data.nodes) || [];
    state.links = (data && data.links) || [];
    state.byId = new Map(state.nodes.map(function (n) { return [n.id, n]; }));
    computeMutual();
    computeMedian();
    initFilterKeys();
    state.fixed.clear();
    // 聚焦集合指向的是**上一份**数据里的节点，必须一并清掉：否则 rebuild() 末尾
    // 的 applyVisibility() 会按旧 focusSet 过滤，数据重载后图谱只剩旧邻居可见。
    state.focusId = null;
    state.focusSet = new Set();
    state.hoverId = null;
    zoomed = false;
    settleTicks = REFRESH_TICKS;

    pinned.forEach(function (p) {
      const node = state.byId.get(p.id);
      if (node) { node.fx = p.x; node.fy = p.y; node.fz = p.z; }
    });
    rebuild();
    // 钉圈是挂在场景里的独立对象，rebuild() 已清空 pins，这里按固定集合补回
    pinned.forEach(function (p) {
      const node = state.byId.get(p.id);
      const group = state.groups.get(p.id);
      if (!node || !group || state.pins.has(p.id)) return;
      state.fixed.add(p.id);
      addPin(p.id, group, nodeRadius(node));
    });
  }

  function computeMutual() {
    const seen = new Set();
    state.mutual = new Set();
    state.links.forEach(function (l) {
      const s = endId(l.source), t = endId(l.target);
      const key = s < t ? s + '|' + t : t + '|' + s;
      if (seen.has(key)) state.mutual.add(key);
      seen.add(key);
    });
  }

  function computeMedian() {
    const degs = state.nodes.map(function (n) { return (n.in_degree || 0) + (n.out_degree || 0); }).sort(function (a, b) { return a - b; });
    state.medianDegree = degs[Math.floor(degs.length / 2)] || 1;
  }

  function initFilterKeys() {
    C.NODE_TYPES.forEach(function (t) {
      if (state.filter.nodeTypes[t.key] === undefined) state.filter.nodeTypes[t.key] = true;
    });
    C.EDGE_TYPES.forEach(function (t) {
      if (state.filter.edgeTypes[t.key] === undefined) state.filter.edgeTypes[t.key] = true;
    });
  }

  function visibleSets() {
    const f = state.filter;
    const nodes = state.nodes.filter(function (n) {
      if (f.nodeTypes[n.node_type] === false) return false;
      if (!f.showDeprecated && n.deprecated) return false;
      if (!f.showForgotten && n.forgotten) return false;
      if (f.hideOrphans && !(n.in_degree || 0) && !(n.out_degree || 0)) return false;
      if (f.highDegOnly && ((n.in_degree || 0) + (n.out_degree || 0)) < state.medianDegree) return false;
      return true;
    });
    const ids = new Set(nodes.map(function (n) { return n.id; }));
    const links = state.links.filter(function (l) {
      return ids.has(endId(l.source)) && ids.has(endId(l.target)) && f.edgeTypes[l.rel_type] !== false;
    });
    return { nodes: nodes, links: links };
  }

  function rebuild() {
    const sets = visibleSets();
    state.groups.clear();
    state.materials.clear();
    state.labels.clear();
    state.pins.clear();
    graph.graphData({ nodes: sets.nodes, links: sets.links });
    applyForces();
    graph.d3ReheatSimulation();
    paintSelection();
    applyVisibility();
    updateLabelVisibility();
    if (hooks.onStats) hooks.onStats(statsOf(sets));
  }

  function statsOf(sets) {
    return {
      visibleNodes: sets.nodes.length,
      visibleEdges: sets.links.length,
      totalNodes: state.nodes.length,
      totalEdges: state.links.length,
      deprecated: state.nodes.filter(function (n) { return n.deprecated; }).length,
      forgotten: state.nodes.filter(function (n) { return n.forgotten; }).length,
      orphans: state.nodes.filter(function (n) {
        return !n.deprecated && !n.forgotten && !(n.in_degree || 0) && !(n.out_degree || 0);
      }).length,
    };
  }

  // ---- 视觉配置 ----

  function segments() {
    const q = state.settings.render.quality;
    return q === 'high' ? 26 : q === 'performance' ? 8 : 16;
  }

  // 各类型的相对大小微调：让轮廓差异不只来自棱数（不改变整体观感比例）
  const SHAPE_SCALE = {
    STATUS: 1, REASON: 1, ACTION: 1.06, THING: 0.97, PERSON: 0.93, EMOTION: 0.95,
  };
  function nodeRadius(node) {
    const base = (node.radius || 4.5) * (state.settings.render.nodeScale || 1);
    return base * (SHAPE_SCALE[node.node_type] || 1);
  }

  // 多面体几何的法线是「逐面」的（同一三角形三个顶点共用一个法线），光照下就是一格
  // 一格的硬边，缺了 STATUS 那种由亮到暗的球面过渡（看起来更实的阴影边）。这里把法线
  // 换成由节点中心指向外的径向法线——顶点本就在以节点中心为原点的球面/凸面上，
  // 于是四种几何都得到同样的连续明暗。
  function smoothNormals(geometry) {
    const pos = geometry.getAttribute('position');
    const nrm = geometry.getAttribute('normal');
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.set(pos.getX(i), pos.getY(i), pos.getZ(i)).normalize();
      nrm.setXYZ(i, v.x, v.y, v.z);
    }
    nrm.needsUpdate = true;
    return geometry;
  }

  // 节点几何统一为「圆形轮廓」的球/多面球族：任何视角都不会看成薄片或尖角，
  // 类型辨识交给轮廓棱数 + 大小微差 + 颜色（配合图层过滤面板的色块图例）。
  // 球体的法线本来就是径向的，多面体则要显式抹平（见 smoothNormals）。
  function geometryFor(type, radius) {
    const seg = segments();
    switch (type) {
      case 'REASON': return smoothNormals(new THREE.IcosahedronGeometry(radius, 0));   // 二十面体
      case 'ACTION': return smoothNormals(new THREE.DodecahedronGeometry(radius, 0));  // 十二面体
      case 'THING': return smoothNormals(new THREE.OctahedronGeometry(radius, 1));     // 细分八面体
      case 'PERSON': return new THREE.SphereGeometry(radius, 8, 6);                    // 低面数球
      case 'EMOTION': return smoothNormals(new THREE.IcosahedronGeometry(radius, 1));  // 细分二十面体
      default: return new THREE.SphereGeometry(radius, seg, seg);                      // 光滑球
    }
  }

  function createNodeObject(node) {
    const group = new THREE.Group();
    const r = nodeRadius(node);
    const color = C.NODE_COLORS[node.node_type] || '#64748b';
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      // 低自发光：浅色底上保持「实心色块」质感，避免霓虹感
      roughness: 0.55, metalness: 0.05,
      emissive: new THREE.Color(color), emissiveIntensity: 0.16,
    });
    const mesh = new THREE.Mesh(geometryFor(node.node_type, r), material);
    // 雷射会递归命中子网格，需在其上回填 __data，节点悬浮提示才能解析出节点
    mesh.__data = node;
    group.add(mesh);
    registerMaterial(node.id, material);

    if (node.deprecated) {
      const wm = new THREE.MeshBasicMaterial({ color: 0x9aa7b4, wireframe: true, transparent: true, opacity: 0.55 });
      const wmesh = new THREE.Mesh(geometryFor(node.node_type, r * 1.06), wm);
      wmesh.__data = node;
      group.add(wmesh);
      registerMaterial(node.id, wm);
    }
    if (node.forgotten) {
      material.opacity = 0.35;
      material.transparent = true;
    }

    state.groups.set(node.id, group);
    if (state.fixed.has(node.id)) addPin(node.id, group, r);
    // 组件重建后，把缓存中的标签重新挂到新 group（否则旧 sprite 会随旧 group 一起丢失）
    const label = state.labels.get(node.id);
    if (label) group.add(label);
    // 新材质立即套用当前激活状态，避免重建（换质量档/过滤）后激活呈现丢失
    styleNodeMaterials(node.id, state.materials.get(node.id) || []);
    return group;
  }

  function registerMaterial(id, material) {
    const list = state.materials.get(id) || [];
    list.push(material);
    state.materials.set(id, list);
  }

  function addPin(id, group, radius) {
    // 复用已缓存的圆环：组件重建时避免重复创建导致旧环泄漏、且新旧各挂一半
    let ring = state.pins.get(id);
    if (!ring) {
      ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius * 1.9, 0.16, 6, 48),
        new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.65 })
      );
      ring.rotation.x = Math.PI / 2.6;
      state.pins.set(id, ring);
    }
    group.add(ring);
  }

  function removePin(id) {
    const ring = state.pins.get(id);
    if (!ring) return;
    if (ring.parent) ring.parent.remove(ring);
    ring.geometry.dispose();
    ring.material.dispose();
    state.pins.delete(id);
  }

  // ---- 标签 ----

  function ensureLabel(node) {
    if (!node) return null;
    let sprite = state.labels.get(node.id);
    if (!sprite) {
      sprite = makeLabelSprite(node.content || node.id);
      sprite.position.set(0, nodeRadius(node) + sprite.__height * 0.5 + 1.6, 0);
      state.labels.set(node.id, sprite);
    }
    // 节点对象会被 3d-force-graph 在可见性变化（切换聚焦跳数）时重建，缓存中的
    // sprite 会留在旧 group 上；仅命中缓存而不重新挂载，标签就再也不会显示。
    const group = state.groups.get(node.id);
    if (group && sprite.parent !== group) group.add(sprite);
    return sprite;
  }

  function makeLabelSprite(text) {
    const max = 22;
    const label = text.length > max ? text.slice(0, max) + '…' : text;
    const font = '600 40px "Segoe UI", "Microsoft YaHei", sans-serif';
    const canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    ctx.font = font;
    const width = Math.ceil(ctx.measureText(label).width) + 44;
    const height = 64;
    canvas.width = width;
    canvas.height = height;
    ctx = canvas.getContext('2d');
    ctx.font = font;
    // 浅色主题标签：白底深字，与官网卡片风格一致
    ctx.fillStyle = 'rgba(255,255,255,0.94)';
    roundRect(ctx, 2, 2, width - 4, height - 4, 14);
    ctx.fill();
    ctx.strokeStyle = 'rgba(15,23,42,0.12)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#1f2937';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, 22, height / 2 + 1);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture, transparent: true, depthWrite: false, depthTest: false,
    }));
    const scale = 0.13;
    sprite.scale.set(width * scale, height * scale, 1);
    sprite.__height = height * scale;
    return sprite;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function updateLabelVisibility() {
    const mode = state.settings.render.labels;
    const limitAll = state.nodes.length <= 400;
    state.groups.forEach(function (group, id) {
      let visible = false;
      if (mode === 'all') visible = limitAll;
      else if (mode === 'focus') visible = isFocused(id) || id === state.selectedId;
      else if (mode === 'hover') visible = id === state.hoverId || id === state.selectedId;
      let sprite = state.labels.get(id);
      if (visible && !sprite) sprite = ensureLabel(state.byId.get(id));
      if (sprite) sprite.visible = visible;
    });
  }

  // ---- 调用测试：节点「激活」呈现 ----

  // 未激活节点的统一灰（与废弃节点线框同色系，保持同一套中性色语言）
  const DIM_COLOR = '#b4bfca';

  // 单个节点的材质着色。两处调用：节点对象刚创建时、以及 paintSelection 收尾时。
  // 规则：激活＝保留类型本色（故事采纳的加金色自发光）；未激活＝灰化；
  //       选中/聚焦的节点豁免灰化，保证点击仍有反馈；未激活时还原本色。
  function styleNodeMaterials(id, mats) {
    const node = state.byId.get(id);
    if (!node) return;
    mats.forEach(function (m) {
      if (m.color && !m.__baseColor) m.__baseColor = m.color.clone();
    });
    const on = state.activated.size > 0;
    const act = on && state.activated.has(id);
    const exempt = id === state.selectedId || isFocused(id);
    mats.forEach(function (m) {
      if (!m.color) return;
      if (on && !act && !exempt) {
        m.color.set(DIM_COLOR);
        if (m.emissive) { m.emissive.set(DIM_COLOR); m.emissiveIntensity = 0.04; }
        return;
      }
      m.color.copy(m.__baseColor);
      if (!m.emissive || !on || exempt) return;   // 其余自发光交给 paintSelection
      if (state.adopted.has(id)) {
        m.emissive.set('#f59e0b');
        m.emissiveIntensity = 0.9;
      } else if (act) {
        m.emissive.copy(m.__baseColor);
        m.emissiveIntensity = 0.45;
      }
    });
  }

  function applyActivation() {
    state.materials.forEach(function (mats, id) { styleNodeMaterials(id, mats); });
  }

  function setActivated(ids, opts) {
    state.activated = new Set(ids || []);
    state.adopted = new Set((opts && opts.adopted) || []);
    applyActivation();
  }

  function clearActivated() {
    state.activated = new Set();
    state.adopted = new Set();
    applyActivation();
  }

  // ---- 高亮 / 选中 ----

  function paintSelection() {
    state.materials.forEach(function (mats, id) {
      const node = state.byId.get(id);
      if (!node) return;
      const base = C.NODE_COLORS[node.node_type] || '#64748b';
      const selected = id === state.selectedId;
      const focused = id === state.focusId;
      mats.forEach(function (m) {
        if (!m.emissive) return;
        if (selected || focused) {
          m.emissive.set(selected ? '#c2410c' : '#0f766e');
          m.emissiveIntensity = selected ? 0.95 : 0.7;
        } else {
          m.emissive.set(base);
          m.emissiveIntensity = 0.16;
        }
        if (!node.forgotten) { m.transparent = false; m.opacity = 1; }
      });
    });
    // 激活呈现优先级最低，放在最后收口：灰化 / 还原本色都在这统一做
    applyActivation();
  }

  function select(id) {
    state.selectedId = id;
    paintSelection();
    updateLabelVisibility();
    // 选中态的唯一出口：调用方（点击 / 搜索 / 数据重载）都走这里，不再各自
    // 手工同步 selectedId 与检查器，避免三份状态互相漂移。
    if (hooks.onSelect) hooks.onSelect(id ? (state.byId.get(id) || null) : null);
  }

  // ---- 聚焦 ----

  function neighbors(id, hops) {
    const visited = new Set([id]);
    let frontier = [id];
    for (let h = 0; h < hops; h++) {
      const next = [];
      frontier.forEach(function (nid) {
        state.links.forEach(function (l) {
          const s = endId(l.source), t = endId(l.target);
          if (s === nid && !visited.has(t)) { visited.add(t); next.push(t); }
          if (t === nid && !visited.has(s)) { visited.add(s); next.push(s); }
        });
      });
      frontier = next;
    }
    return visited;
  }

  function isFocused(id) { return id === state.focusId || state.focusSet.has(id); }

  function focus(id, hops) {
    if (!id) return;
    state.focusId = id;
    state.focusHops = hops || 1;
    state.focusSet = neighbors(id, state.focusHops);
    applyVisibility();
    paintSelection();
    updateLabelVisibility();
    const node = state.byId.get(id);
    if (node && node.x !== undefined) {
      graph.cameraPosition({ x: node.x, y: node.y, z: node.z + 130 }, node, 1200);
    }
  }

  function exitFocus() {
    state.focusId = null;
    state.focusSet = new Set();
    applyVisibility();
    paintSelection();
    updateLabelVisibility();
  }

  function applyVisibility() {
    if (!state.focusId) {
      graph.nodeVisibility(function () { return true; });
      graph.linkVisibility(function () { return true; });
      return;
    }
    const set = state.focusSet;
    graph.nodeVisibility(function (n) { return set.has(n.id); });
    graph.linkVisibility(function (l) { return set.has(endId(l.source)) && set.has(endId(l.target)); });
  }

  // ---- 搜索 ----

  function search(query) {
    const q = (query || '').trim().toLowerCase();
    const gd = graph.graphData();
    if (!q) {
      clearSearch();
      return [];
    }
    const matched = [];
    gd.nodes.forEach(function (n) {
      const hit = (n.content && n.content.toLowerCase().indexOf(q) >= 0) || String(n.id).toLowerCase().indexOf(q) >= 0;
      const mats = state.materials.get(n.id) || [];
      mats.forEach(function (m) {
        if (hit) {
          if (!n.forgotten) { m.transparent = false; m.opacity = 1; }
          if (m.emissive) { m.emissive.set('#a21caf'); m.emissiveIntensity = 0.9; }
        } else {
          m.transparent = true;
          m.opacity = 0.06;
        }
      });
      if (hit) matched.push(n);
    });
    return matched;
  }

  function clearSearch() {
    applyVisibility();
    paintSelection();
    state.materials.forEach(function (mats, id) {
      const node = state.byId.get(id);
      mats.forEach(function (m) {
        if (!node || !node.forgotten) { m.transparent = false; m.opacity = 1; }
      });
    });
  }

  // ---- 交互 ----

  function handleNodeClick(node) {
    if (state.edgeMode) {
      if (hooks.onEdgeAction) hooks.onEdgeAction(Object.assign({ targetId: node.id }, state.edgeMode));
      return;
    }
    select(node.id);
    focus(node.id, 1);
  }

  function handleNodeRightClick(node, evt) {
    select(node.id);
    if (hooks.onContextMenu) hooks.onContextMenu(node, evt || window.event);
  }

  function handleHover(node) {
    state.hoverId = node ? node.id : null;
    if (state.settings.render.labels === 'hover') updateLabelVisibility();
    if (node) {
      // 节点优先：收起可能同帧命中的连线提示
      clearTimeout(pendingEdgeTip);
      if (hooks.onEdgeHover) hooks.onEdgeHover(null, null);
    }
    if (hooks.onHover) {
      if (node && graph.graph2ScreenCoords) {
        const pos = graph.graph2ScreenCoords(node.x, node.y, node.z);
        hooks.onHover(pos, state.byId.get(node.id) || node);
      } else {
        hooks.onHover(null, null);
      }
    }
  }

  // 悬浮连线：把「关系类型 + 两端内容」交给 UI 展示，便于精确辨认关系。
  // 连线起点在节点球心，二者常同时命中，因此延迟一帧渲染并让位于节点悬浮。
  let pendingEdgeTip = 0;

  function handleLinkHover(link) {
    if (!hooks.onEdgeHover) return;
    clearTimeout(pendingEdgeTip);
    const s = link && typeof link.source === 'object' ? link.source : null;
    const t = link && typeof link.target === 'object' ? link.target : null;
    if (!s || !t || !graph.graph2ScreenCoords) { hooks.onEdgeHover(null, null); return; }
    const pos = graph.graph2ScreenCoords((s.x + t.x) / 2, (s.y + t.y) / 2, (s.z + t.z) / 2);
    const info = { relType: link.rel_type, source: s, target: t };
    pendingEdgeTip = setTimeout(function () {
      if (!state.hoverId) hooks.onEdgeHover(pos, info);  // 同帧内有节点悬浮则不显示
    }, 0);
  }

  function handleDragEnd(node) {
    node.fx = node.x; node.fy = node.y; node.fz = node.z;
    state.fixed.add(node.id);
    const group = state.groups.get(node.id);
    if (group && !state.pins.has(node.id)) addPin(node.id, group, nodeRadius(node));
  }

  function handleBackgroundClick() {
    state.edgeMode = null;
    exitFocus();
    select(null);
    if (hooks.onEdgeModeChange) hooks.onEdgeModeChange(null);
  }

  function unfixAll() {
    state.fixed.forEach(function (id) {
      const node = state.byId.get(id);
      if (node) { node.fx = null; node.fy = null; node.fz = null; }
      removePin(id);
    });
    state.fixed.clear();
  }

  function enterEdgeMode(mode, relType, sourceId) {
    state.edgeMode = { mode: mode, relType: relType, sourceId: sourceId };
    document.body.classList.add('edge-mode');
    if (hooks.onEdgeModeChange) hooks.onEdgeModeChange(state.edgeMode);
  }

  function exitEdgeMode() {
    state.edgeMode = null;
    document.body.classList.remove('edge-mode');
    if (hooks.onEdgeModeChange) hooks.onEdgeModeChange(null);
  }

  function zoomFit() { if (graph) graph.zoomToFit(700, 50); }

  // ---- 相机（悬浮窗滚动条控制缩放 + 位置读数）----

  // 缩放滚动条映射到的距离区间；用对数刻度，近处细腻、远处跨度大
  const CAM_MIN_DIST = 10;
  const CAM_MAX_DIST = 5000;
  const CAM_SLIDER_MAX = 1000;
  const cameraSubs = [];
  let lastCamera = { x: NaN, y: NaN, z: NaN };

  function cameraTarget() {
    const controls = graph && graph.controls && graph.controls();
    const t = controls && controls.target;
    return t ? { x: t.x, y: t.y, z: t.z } : { x: 0, y: 0, z: 0 };
  }

  function cameraState() {
    const cam = graph && graph.camera && graph.camera();
    if (!cam) return null;
    const p = cam.position;
    const t = cameraTarget();
    const dx = p.x - t.x, dy = p.y - t.y, dz = p.z - t.z;
    return {
      position: { x: p.x, y: p.y, z: p.z },
      target: t,
      distance: Math.sqrt(dx * dx + dy * dy + dz * dz),
    };
  }

  function onCameraChange(cb) {
    cameraSubs.push(cb);
    return function () {
      const i = cameraSubs.indexOf(cb);
      if (i >= 0) cameraSubs.splice(i, 1);
    };
  }

  // 由渲染循环驱动：只在相机真正移动时通知，避免依赖 OrbitControls 的事件
  function emitCameraIfMoved() {
    const cam = graph && graph.camera && graph.camera();
    if (!cam) return;
    const p = cam.position;
    if (Math.abs(p.x - lastCamera.x) < 1e-3 &&
        Math.abs(p.y - lastCamera.y) < 1e-3 &&
        Math.abs(p.z - lastCamera.z) < 1e-3) return;
    lastCamera = { x: p.x, y: p.y, z: p.z };
    const st = cameraState();
    for (let i = 0; i < cameraSubs.length; i++) cameraSubs[i](st);
  }

  // 沿当前视线方向把相机推到指定距离（保持朝向与视角不变）
  function setCameraDistance(distance) {
    const st = cameraState();
    if (!st) return;
    const d = Math.max(CAM_MIN_DIST, Number(distance) || CAM_MIN_DIST);
    const dx = st.position.x - st.target.x;
    const dy = st.position.y - st.target.y;
    const dz = st.position.z - st.target.z;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    const k = d / len;
    graph.cameraPosition({
      x: st.target.x + dx * k,
      y: st.target.y + dy * k,
      z: st.target.z + dz * k,
    }, st.target, 0);
  }

  function distanceToSlider(distance) {
    const t = Math.log(distance / CAM_MIN_DIST) / Math.log(CAM_MAX_DIST / CAM_MIN_DIST);
    return Math.round(Math.min(1, Math.max(0, t)) * CAM_SLIDER_MAX);
  }

  function sliderToDistance(value) {
    const t = Math.min(1, Math.max(0, Number(value) / CAM_SLIDER_MAX));
    return CAM_MIN_DIST * Math.pow(CAM_MAX_DIST / CAM_MIN_DIST, t);
  }

  // ---- 设置 ----

  function applyForces() {
    if (!graph) return;
    const r = state.settings.render;
    const distance = Number(r.linkDistance) || 30;
    const charge = graph.d3Force('charge');
    if (charge) {
      if (charge.strength) charge.strength(Number(r.chargeStrength) || -45);
      // 限制斥力作用半径：否则孤立节点会被推得极远，撑大包围盒，
      // 导致 zoomToFit 把镜头拉远、主体节点在屏幕上变得很小。
      if (charge.distanceMax) charge.distanceMax(Math.max(220, distance * 14));
      if (charge.distanceMin) charge.distanceMin(1);
    }
    const link = graph.d3Force('link');
    if (link && link.distance) link.distance(distance);
  }

  function applySettings(settings) {
    state.settings = settings;
    applyForces();
    // 页面 56px 网格底纹（与官网一致）由 body class 控制
    document.body.classList.toggle('no-grid', settings.render.grid === false);
    graph.linkDirectionalParticles(function () { return settings.render.particles; });
    graph.linkWidth(linkWidthFor);
    graph.linkDirectionalArrowLength(arrowFor);
    const controls = graph.controls && graph.controls();
    if (controls) { controls.autoRotate = !!settings.render.autoRotate; controls.autoRotateSpeed = 0.55; }
    // 质量/节点尺寸变化需要重建几何
    state.groups.clear();
    state.materials.clear();
    state.labels.clear();
    state.pins.clear();
    graph.nodeThreeObject(createNodeObject);
    paintSelection();
    applyVisibility();
    updateLabelVisibility();
  }

  // 线宽分 4 档：颜色之外再给一层「粗细」线索，便于快速区分关系类型
  function linkWidthFor(l) {
    const r = state.settings.render;
    if (r.showPipes === false) return 0.25;
    const base = l.rel_type === 'causal' ? 1.0
      : ['scenario', 'sequence', 'preference'].indexOf(l.rel_type) >= 0 ? 0.72
        : ['social', 'attribute'].indexOf(l.rel_type) >= 0 ? 0.5 : 0.36;
    return base * (r.linkWidth || 1);
  }

  function arrowFor(l) {
    if (state.settings.render.showArrows === false) return 0;
    if (C.BIDIRECTIONAL.indexOf(l.rel_type) >= 0) return 0;  // 双向关系不画箭头
    return l.rel_type === 'causal' ? 3.8 : 3.0;
  }

  function curvatureFor(l) {
    const s = endId(l.source), t = endId(l.target);
    const key = s < t ? s + '|' + t : t + '|' + s;
    if (C.BIDIRECTIONAL.indexOf(l.rel_type) >= 0) return 0.12;
    return state.mutual.has(key) ? 0.22 : 0;
  }

  function particleSpeed(type) {
    const map = { causal: 0.010, sequence: 0.007, preference: 0.006, temporal: 0.006, scenario: 0.004, social: 0.004, attribute: 0.003, taxonomic: 0.005 };
    return map[type] || 0.005;
  }

  function endId(end) { return end && end.id !== undefined ? end.id : end; }

  // ---- 固定环动画 ----

  function startRenderLoop() {
    function loop() {
      const t = performance.now() / 1600;
      state.pins.forEach(function (ring) { ring.rotation.z = t; });
      emitCameraIfMoved();
      renderHandle = requestAnimationFrame(loop);
    }
    loop();
  }

  // ---- 导出 ----

  A.graph = {
    init: init,
    setData: setData,
    rebuild: rebuild,
    setFilter: function (patch) { Object.assign(state.filter, patch); rebuild(); },
    getFilter: function () { return JSON.parse(JSON.stringify(state.filter)); },
    select: select,
    focus: focus,
    exitFocus: exitFocus,
    getFocusHops: function () { return state.focusHops; },
    search: search,
    clearSearch: clearSearch,
    setActivated: setActivated,
    clearActivated: clearActivated,
    unfixAll: unfixAll,
    enterEdgeMode: enterEdgeMode,
    exitEdgeMode: exitEdgeMode,
    zoomFit: zoomFit,
    setCamera: function (x, y, z) {
      if (graph) graph.cameraPosition({ x: x, y: y, z: z }, { x: 0, y: 0, z: 0 }, 1000);
    },
    cameraState: cameraState,
    onCameraChange: onCameraChange,
    setCameraDistance: setCameraDistance,
    distanceToSlider: distanceToSlider,
    sliderToDistance: sliderToDistance,
    applySettings: applySettings,
  };
})();
