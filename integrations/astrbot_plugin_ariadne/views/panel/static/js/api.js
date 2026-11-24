/* SPDX-License-Identifier: AGPL-3.0-only */
/* 后端 REST + SSE 访问层；离线 HTML 模式下只读降级 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});
  const boot = window.__ARIADNE_BOOT__ || { mode: 'live' };
  const isLive = boot.mode !== 'offline';
  // 去掉末尾的文件名/斜杠，得到不带尾斜杠的应用根
  const BASE = (window.location.origin + window.location.pathname).replace(/\/[^/]*$/, '');

  // AstrBot 插件视图的 iframe 没有 allow-same-origin，视图内既没有 Cookie 也没有
  // localStorage，直接 fetch 面板 API 一定 401。检测到宿主注入的 bridge 时，
  // 所有请求都交给父页面（带 Dashboard 登录态）转发给插件后端，再由插件用内部
  // Bearer 访问容器内的面板服务；直连面板时行为保持不变。
  const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
  const bridgeMode = !!(bridge && typeof bridge.apiPost === 'function');

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

  function splitPath(path) {
    const idx = path.indexOf('?');
    if (idx === -1) return { path: path, query: '' };
    return { path: path.slice(0, idx), query: path.slice(idx + 1) };
  }

  function raceTimeout(promise, ms) {
    if (!ms) return promise;
    return new Promise(function (resolve, reject) {
      const timer = setTimeout(function () {
        reject(new Error('请求超时（' + Math.round(ms / 1000) + ' 秒未响应）'));
      }, ms);
      promise.then(function (value) {
        clearTimeout(timer);
        resolve(value);
      }, function (error) {
        clearTimeout(timer);
        reject(error);
      });
    });
  }

  // bridge 模式：POST 到插件的 /proxy，由插件按 method/path/query/body 转发。
  // 不用 apiGet 是因为父页面把 params 交给 axios 序列化，传嵌套 body 不可控。
  async function bridgeRequest(method, path, body, ms) {
    const parts = splitPath(path);
    const payload = { method: method, path: parts.path };
    if (parts.query) payload.query = parts.query;
    if (body !== undefined) payload.body = body;
    let data;
    try {
      data = await raceTimeout(bridge.apiPost('proxy', payload), ms);
    } catch (error) {
      online = false;
      throw error instanceof Error ? error : new Error(String(error));
    }
    online = true;
    return data;
  }

  function downloadExport(path) {
    if (bridgeMode) {
      const parts = splitPath(path);
      const params = { path: parts.path };
      if (parts.query) params.query = parts.query;
      return bridge.download('download', params);
    }
    const link = document.createElement('a');
    link.href = BASE + path;
    link.download = '';
    document.body.appendChild(link);
    link.click();
    link.remove();
    return Promise.resolve();
  }

  function bridgeStream(onEvent, onState) {
    let closed = false;
    let retry = 0;
    let subscriptionId = null;
    let timer = null;

    function schedule() {
      retry = Math.min(retry + 1, 30);
      timer = setTimeout(open, Math.min(1000 * retry, 30000));
    }

    function open() {
      if (closed) return;
      bridge.subscribeSSE('stream', {
        onMessage: function (message) {
          if (closed || !message || message.hasData === false) return;
          online = true;
          try { onEvent(message.parsed); } catch (e) {}
        },
        onOpen: function () {
          retry = 0;
          online = true;
          if (onState) onState('open');
        },
        onError: function () {
          if (closed) return;
          online = false;
          if (onState) onState('closed');
          if (subscriptionId) {
            try { bridge.unsubscribeSSE(subscriptionId); } catch (e) {}
            subscriptionId = null;
          }
          schedule();
        },
      }, { path: '/api/stream' }).then(function (id) {
        if (closed) {
          bridge.unsubscribeSSE(id);
          return;
        }
        subscriptionId = id;
      }).catch(function () {
        if (!closed) schedule();
      });
    }

    open();
    return {
      close: function () {
        closed = true;
        if (timer) clearTimeout(timer);
        if (subscriptionId) bridge.unsubscribeSSE(subscriptionId);
      },
    };
  }

  async function request(method, path, body, opts) {
    const ms = opts && opts.timeout !== undefined ? opts.timeout : DEFAULT_TIMEOUT;
    if (bridgeMode) {
      return bridgeRequest(method, path, body, ms);
    }
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
    // bridge 模式下鉴权由 AstrBot Dashboard 负责，视图内没有登录页可跳
    return !isLive || bridgeMode || location.pathname === '/login';
  }

  const api = {
    isLive: isLive,
    bridgeMode: bridgeMode,
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

    // ---- 图谱库（面板内切换 / 管理；载入 MCP 是独立动作）----
    getGraphs: function () { return request('GET', '/api/graphs'); },
    switchGraph: function (name) { return request('POST', '/api/graphs/switch', { name: name }); },

    // ---- 知识搜索（关键词倒排，非向量）----
    knowledgeSearch: function (q, top_k) {
      if (!isLive) return Promise.resolve({ query: q, total: 0, items: [] });
      return request('GET', '/api/knowledge/search' + query({ q: q, top_k: top_k || 8 }));
    },
    createGraph: function (name, yaml) {
      return request('POST', '/api/graphs/create', { name: name, yaml: yaml });
    },
    deleteGraph: function (name) {
      return request('DELETE', '/api/graphs?name=' + encodeURIComponent(name));
    },
    activateMcpGraph: function (name) {
      return request('POST', '/api/graphs/activate-mcp', { name: name });
    },
    exportGraph: function (name) {
      return downloadExport('/api/export/yaml?name=' + encodeURIComponent(name));
    },
    exportOplog: function () { return downloadExport('/api/export/oplog'); },
    exportApplog: function () { return downloadExport('/api/export/applog'); },
    exportChainTests: function () { return downloadExport('/api/export/chaintests'); },

    // ---- 可观测性 ----
    getOplog: function (params) { return request('GET', '/api/oplog' + query(params)); },
    getLogs: function (params) { return request('GET', '/api/logs' + query(params)); },
    getMetrics: function () { return request('GET', '/api/metrics'); },
    getHealth: function () {
      // 健康检查在降级时返回 503，但仍需读取其 body 展示具体检查项。
      // 插件代理已把 /api/health 统一包成 200，因此 bridge 模式直接走 request()。
      if (bridgeMode) return request('GET', '/api/health');
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

    // ---- 实时流 ----
    connectStream: function (onEvent, onState) {
      if (!isLive) {
        if (onState) onState('offline');
        return { close: function () {} };
      }
      if (bridgeMode) {
        return bridgeStream(onEvent, onState);
      }
      if (typeof EventSource === 'undefined') {
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
