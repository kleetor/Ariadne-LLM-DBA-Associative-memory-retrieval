/* SPDX-License-Identifier: AGPL-3.0-only */
/* WebUI 通用工具：DOM、格式化、提示、下载 */
(function () {
  const A = (window.Ariadne = window.Ariadne || {});

  function $(id) { return document.getElementById(id); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k.indexOf('on') === 0 && typeof attrs[k] === 'function') node.addEventListener(k.slice(2), attrs[k]);
        else if (attrs[k] !== undefined && attrs[k] !== null) node.setAttribute(k, attrs[k]);
      });
    }
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  function escapeHtml(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function debounce(fn, wait) {
    let timer = null;
    return function () {
      const args = arguments, ctx = this;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(ctx, args); }, wait);
    };
  }

  function fmtTime(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    // 解析失败时原样回显便于排查，但必须转义：调用方都把它拼进 innerHTML
    if (isNaN(d.getTime())) return escapeHtml(String(iso));
    const pad = function (n) { return String(n).padStart(2, '0'); };
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' +
      pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  function fmtDuration(seconds) {
    if (seconds === undefined || seconds === null) return '-';
    let s = Math.max(0, Math.floor(seconds));
    const d = Math.floor(s / 86400); s -= d * 86400;
    const h = Math.floor(s / 3600); s -= h * 3600;
    const m = Math.floor(s / 60); s -= m * 60;
    if (d > 0) return d + 'd ' + h + 'h ' + m + 'm';
    if (h > 0) return h + 'h ' + m + 'm';
    if (m > 0) return m + 'm ' + s + 's';
    return s + 's';
  }

  function fmtBytes(bytes) {
    if (bytes === undefined || bytes === null) return '-';
    const units = ['B', 'KB', 'MB', 'GB'];
    let v = Number(bytes), i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return (i === 0 ? v : v.toFixed(1)) + ' ' + units[i];
  }

  // ---- 提示 ----

  function toast(message, kind) {
    let wrap = $('toast-wrap');
    if (!wrap) {
      wrap = el('div', { id: 'toast-wrap' });
      document.body.appendChild(wrap);
    }
    const item = el('div', { class: 'toast ' + (kind || 'info'), text: message });
    wrap.appendChild(item);
    setTimeout(function () { item.classList.add('show'); }, 10);
    setTimeout(function () {
      item.classList.remove('show');
      setTimeout(function () { if (item.parentNode) item.parentNode.removeChild(item); }, 300);
    }, 3600);
  }

  function copy(text) {
    const value = String(text === undefined || text === null ? '' : text);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(value).then(function () { toast('已复制到剪贴板', 'ok'); },
        function () { fallbackCopy(value); });
    } else {
      fallbackCopy(value);
    }
  }

  function fallbackCopy(value) {
    const ta = el('textarea', { style: 'position:fixed;opacity:0' });
    ta.value = value;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); toast('已复制到剪贴板', 'ok'); }
    catch (e) { toast('复制失败', 'error'); }
    document.body.removeChild(ta);
  }

  function download(filename, content, mime) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mime || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }

  A.util = {
    $: $, $$: $$, el: el, escapeHtml: escapeHtml, debounce: debounce,
    fmtTime: fmtTime, fmtDuration: fmtDuration, fmtBytes: fmtBytes,
    toast: toast, copy: copy, download: download,
  };
})();
