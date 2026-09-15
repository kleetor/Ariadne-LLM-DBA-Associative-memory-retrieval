/* SPDX-License-Identifier: AGPL-3.0-only */
/* 常量 + 本地设置/过滤预设持久化 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});

  // 节点类型与配色（6 种角色）：三原色 + 三间色，色相均匀间隔 60°，
  // 同明度（L≈42-44%）、同饱和（S≈50-55%），既不在浅底上刺眼，
  // 又不会像紫/蓝/绿那样挤在色轮同一段。
  //   原色：蓝(240°) 红(0°) 绿(130°)
  //   间色：青=绿+蓝  品红=红+蓝  黄=红+绿
  const NODE_TYPES = [
    { key: 'STATUS', label: '状态', color: '#3266ae' },   // 原色·蓝
    { key: 'REASON', label: '原因', color: '#ae3232' },   // 原色·红
    { key: 'ACTION', label: '行为', color: '#309141' },   // 原色·绿
    { key: 'THING', label: '事物', color: '#339199' },    // 间色·青（绿+蓝）
    { key: 'PERSON', label: '人物', color: '#a6307f' },   // 间色·品红（红+蓝）
    { key: 'EMOTION', label: '情绪', color: '#a38733' },  // 间色·黄（红+绿）
  ];

  // 边类型与配色（8 种关系）：在色轮上均匀取 8 个可分辨色相（45° 步进），
  // 明度略高于节点、并以细线呈现，因此既能互相区分，又不会与实心节点球混淆。
  const EDGE_TYPES = [
    { key: 'causal', label: '因果', color: '#cf5a4e' },     // 红    0°
    { key: 'preference', label: '偏好', color: '#c8952f' }, // 琥珀  40°
    { key: 'scenario', label: '场景', color: '#56a45c' },   // 绿    130°
    { key: 'attribute', label: '属性', color: '#45a3ae' },  // 青    185°
    { key: 'sequence', label: '时序', color: '#5a83cc' },   // 蓝    220°
    { key: 'social', label: '社交', color: '#9070cd' },     // 紫    270°
    { key: 'taxonomic', label: '分类', color: '#bb569f' },  // 品红  320°
    { key: 'temporal', label: '时间', color: '#8b96a6' },   // 中性灰（时间轴）
  ];

  const NODE_COLORS = {};
  NODE_TYPES.forEach(function (t) { NODE_COLORS[t.key] = t.color; });
  const EDGE_COLORS = {};
  EDGE_TYPES.forEach(function (t) { EDGE_COLORS[t.key] = t.color; });
  const BIDIRECTIONAL = ['scenario', 'social', 'attribute'];

  // 渲染默认值对齐官网图谱（透明画布 + 56px 网格、粒子数 1、力参数 -160/36）
  const DEFAULT_SETTINGS = {
    render: {
      quality: 'balanced',     // high | balanced | performance
      nodeScale: 1.6,
      linkWidth: 1,
      particles: 1,
      chargeStrength: -90,
      linkDistance: 28,
      labels: 'focus',         // none | focus | hover | all
      showArrows: true,
      autoRotate: false,
      showPipes: true,
      grid: true,              // 页面/画布 56px 网格底纹
    },
  };

  const SETTINGS_KEY = 'ariadne.settings.v1';
  const PRESETS_KEY = 'ariadne.presets.v1';

  function deepMerge(base, patch) {
    const out = Array.isArray(base) ? base.slice() : Object.assign({}, base);
    Object.keys(patch || {}).forEach(function (k) {
      const v = patch[k];
      if (v && typeof v === 'object' && !Array.isArray(v) && base && typeof base[k] === 'object') {
        out[k] = deepMerge(base[k], v);
      } else {
        out[k] = v;
      }
    });
    return out;
  }

  // deepMerge 只做一层 Object.assign，嵌套对象（如 render）会是同一引用；
  // 因此凡是以 DEFAULT_SETTINGS 为底的地方都先深拷贝，避免「改设置」直接
  // 写到默认值上（那会让「恢复默认」恢复出被污染的默认值）。
  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  let settings = deepMerge(clone(DEFAULT_SETTINGS), readJson(SETTINGS_KEY) || {});

  function readJson(key) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

  function writeJson(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; }
    catch (e) { return false; }
  }

  function save() { writeJson(SETTINGS_KEY, settings); }

  function update(section, key, value) {
    settings[section][key] = value;
    save();
  }

  function reset() {
    settings = deepMerge(clone(DEFAULT_SETTINGS), {});
    save();
  }

  // ---- 过滤预设 ----

  function listPresets() {
    return readJson(PRESETS_KEY) || [];
  }

  function savePreset(name, state) {
    const presets = listPresets();
    const preset = { id: 'p' + Date.now().toString(36), name: name, savedAt: new Date().toISOString(), state: state };
    presets.push(preset);
    writeJson(PRESETS_KEY, presets);
    return preset;
  }

  function deletePreset(id) {
    writeJson(PRESETS_KEY, listPresets().filter(function (p) { return p.id !== id; }));
  }

  A.const = {
    NODE_TYPES: NODE_TYPES, EDGE_TYPES: EDGE_TYPES,
    NODE_COLORS: NODE_COLORS, EDGE_COLORS: EDGE_COLORS,
    BIDIRECTIONAL: BIDIRECTIONAL, DEFAULT_SETTINGS: clone(DEFAULT_SETTINGS),
  };

  A.store = {
    get settings() { return settings; },
    save: save,
    update: update,
    reset: reset,
    listPresets: listPresets,
    savePreset: savePreset,
    deletePreset: deletePreset,
  };
})();
