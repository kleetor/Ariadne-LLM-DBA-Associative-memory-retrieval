/* SPDX-License-Identifier: AGPL-3.0-only */
/* 后端 REST + SSE 访问层；离线 HTML 模式下只读降级 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});
  const boot = window.__ARIADNE_BOOT__ || { mode: 'live' };
  const isLive = boot.mode !== 'offline';
  // 去掉末尾的文件名/斜杠，得到不带尾斜杠的应用根
  const BASE = (window.location.origin + window.location.pathname).replace(/\/[^/]*$/, '');

  // 后端可达性：网络层失败置为 false，任一次成功请求恢复 true。
  // 各轮询据此暂停，避免服务停止后控制台被 ERR_CONNECTION_REFUSED 刷屏。
  let online = true;

  // 请求超时：默认为 30s，避免后端挂住时界面永远停在「加载中」。
  // 长任务（调用测试要跑完整条 LLM 链路）用 timeout: 0 自行关闭，
  // 它的完成信号走 SSE 的 channel="chain" 事件，不依赖这个 POST 的响应。
  const DEFAULT_TIMEOUT = 30000;

  function timeoutSignal(ms) {
    if (!ms) return { signal: undefined, cancel: function () {} };
    if (typeof AbortSignal !== 'undefined' && AbortSignal.timeout) {
      return { signal: AbortSignal.timeout(ms), cancel: function () {} };
    }
    const ctrl = new AbortController();
    const timer = setTimeout(function () { ctrl.abort(); }, ms);
    return { signal: ctrl.signal, cancel: function () { clearTimeout(timer); } };
  }

  async function request(method, path, body, opts) {
    const ms = opts && opts.timeout !== undefined ? opts.timeout : DEFAULT_TIMEOUT;
    const limit = timeoutSignal(ms);
    const fetchOpts = { method: method, headers: {} };
    if (limit.signal) fetchOpts.signal = limit.signal;
    if (body !== undefined) {
      fetchOpts.headers['Content-Type'] = 'application/json';
      fetchOpts.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(BASE + path, fetchOpts);
    } catch (e) {
      online = false;
      const name = e && e.name;
      if (name === 'TimeoutError' || name === 'AbortError') {
        throw new Error('请求超时（' + Math.round(ms / 1000) + ' 秒未响应）');
      }
      throw e;
    } finally {
      limit.cancel();
    }
    online = true;  // 能拿到响应（含 4xx/5xx）说明服务在线
    const text = await res.text();
    let payload = null;
    if (text) {
      try { payload = JSON.parse(text); } catch (e) { payload = { error: text }; }
    }
    if (!res.ok) {
      if (res.status === 401) toLogin();
      // 初始密码未改：服务端只放行改密相关接口，其余一律 403
      else if (res.status === 403 && payload && payload.must_change) toChangePassword();
      const message = (payload && (payload.error || payload.detail)) || res.statusText;
      throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
    }
    return payload;
  }

  // 会话失效：跳到登录页并带上回跳地址。
  // 已在登录页时不再跳，否则登录失败的 401 会变成重定向死循环。
  function toLogin() {
    if (jumpBlocked()) return;
    const next = encodeURIComponent(location.pathname + location.search);
    location.href = '/login?next=' + next;
  }

  function toChangePassword() {
    if (jumpBlocked()) return;
    const next = encodeURIComponent(location.pathname + location.search);
    location.href = '/login?mode=change&next=' + next;
  }

  function jumpBlocked() {
    return !isLive || location.pathname === '/login';
  }

  const api = {
    isLive: isLive,
    boot: boot,
    isOnline: function () { return online; },

    // ---- 图谱 ----
    getGraph: function () {
      if (!isLive && boot.data) return Promise.resolve(boot.data);
      return request('GET', '/api/graph');
    },
    createNode: function (nodeType, content) {
      return request('POST', '/api/nodes', { node_type: nodeType, content: content });
    },
    updateNode: function (id, patch) {
      return request('PUT', '/api/nodes/' + encodeURIComponent(id), patch);
    },
    deleteNode: function (id) {
      return request('DELETE', '/api/nodes/' + encodeURIComponent(id));
    },
    createEdge: function (source, target, relType) {
      return request('POST', '/api/edges', { source: source, target: target, rel_type: relType });
    },
    deleteEdge: function (source, target) {
      return request('DELETE', '/api/edges/' + encodeURIComponent(source) + '/' + encodeURIComponent(target));
    },

    // ---- 可观测性 ----
    getOplog: function (params) { return request('GET', '/api/oplog' + query(params)); },
    getLogs: function (params) { return request('GET', '/api/logs' + query(params)); },
    getMetrics: function () { return request('GET', '/api/metrics'); },
    getHealth: function () {
      // 健康检查在降级时返回 503，但仍需读取其 body 展示具体检查项
      return fetch(BASE + '/api/health', { signal: timeoutSignal(DEFAULT_TIMEOUT).signal })
        .then(function (res) {
          online = true;
          return res.json();
        }, function (e) {
          online = false;
          throw e;
        });
    },
    getConfig: function () { return request('GET', '/api/config'); },
    getMcpStatus: function () { return request('GET', '/api/mcp/status'); },
    login: function (user, password) { return request('POST', '/api/login', { user: user, password: password }); },
    logout: function () { return request('POST', '/api/logout'); },

    // ---- 运行时参数（权重矩阵 / 种子 / 目的回归）----
    getParams: function () { return request('GET', '/api/params'); },
    saveParams: function (patch) { return request('PUT', '/api/params', { params: patch }); },
    resetParams: function () { return request('POST', '/api/params/reset'); },

    // ---- 调用测试：跑一次真实链路（意图识别 → PAR → StoryRank）----
    // 阶段进度与完成信号都走 /api/stream 的 channel="chain" 事件，
    // 所以这里不做超时（真实链路耗时可达数分钟）
    runChainTest: function (query) {
      return request('POST', '/api/chain/test', { query: query }, { timeout: 0 });
    },

    exportYamlUrl: function () { return BASE + '/api/export/yaml'; },
    exportOplogUrl: function () { return BASE + '/api/export/oplog'; },
    importYaml: function (text) { return request('POST', '/api/import/yaml', { yaml: text }); },

    // ---- 实时流 ----
    connectStream: function (onEvent, onState) {
      if (!isLive || typeof EventSource === 'undefined') {
        if (onState) onState('offline');
        return { close: function () {} };
      }
      let source = null;
      let closed = false;
      let retry = 0;

      function open() {
        source = new EventSource(BASE + '/api/stream');
        source.onopen = function () { retry = 0; online = true; if (onState) onState('open'); };
        source.onmessage = function (evt) {
          if (!evt.data) return;
          try { onEvent(JSON.parse(evt.data)); } catch (e) {}
        };
        source.onerror = function () {
          if (closed) return;
          online = false;
          if (onState) onState('closed');
          try { source.close(); } catch (e) {}
          // 指数退避，最长 30s，减少服务停止时的重连噪声
          retry = Math.min(retry + 1, 30);
          setTimeout(open, Math.min(1000 * retry, 30000));
        };
      }
      open();

      return {
        close: function () {
          closed = true;
          if (source) source.close();
        },
      };
    },
  };

  function query(params) {
    if (!params) return '';
    const parts = Object.keys(params)
      .filter(function (k) { return params[k] !== undefined && params[k] !== null && params[k] !== ''; })
      .map(function (k) { return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]); });
    return parts.length ? '?' + parts.join('&') : '';
  }

  A.api = api;
})();
