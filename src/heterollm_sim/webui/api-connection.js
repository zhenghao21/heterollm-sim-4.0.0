"use strict";

// API transport shared by every workbench request. Credentials never enter a
// URL, localStorage, a scenario export, or a diagnostic message.
(function (root) {
  const ADDRESS_KEY = "heterollm-lab:backend-address:v1";
  const isLoopback = (hostname) => ["localhost", "127.0.0.1", "::1", "[::1]"].includes(String(hostname).toLowerCase());

  function normalizeBackendUrl(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    let url;
    try { url = new URL(raw); } catch (_) { throw new Error("后端地址必须是完整的 HTTPS 地址。"); }
    if (url.username || url.password || raw.includes("?") || raw.includes("#")) {
      throw new Error("后端地址不能包含账号、密码、查询参数或片段；访问令牌请填入独立密码框。");
    }
    if (url.protocol !== "https:" && !(url.protocol === "http:" && isLoopback(url.hostname))) {
      throw new Error("远程后端必须使用 HTTPS；只有 localhost 或 127.0.0.1 开发连接允许 HTTP。");
    }
    return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
  }

  function createConnection(environment = root) {
    const page = environment.location || { origin: "", hostname: "" };
    const localPage = isLoopback(page.hostname);
    let backendUrl = "";
    let token = "";
    let active = false;
    let generation = 0;
    let configError = "";
    const pending = new Set();
    const listeners = new Set();
    const read = (storage, key) => { try { return storage?.getItem(key) || ""; } catch (_) { return ""; } };
    const write = (storage, key, value) => {
      try { if (value) storage?.setItem(key, value); else storage?.removeItem(key); } catch (_) { /* Private storage may be disabled. */ }
    };
    const sessionKey = (url) => `heterollm-lab:backend-session:v1:${encodeURIComponent(page.origin)}:${encodeURIComponent(url || page.origin)}`;
    const snapshot = () => ({ backendUrl, active, localPage, generation, configError, hasToken: Boolean(token) });
    const emit = () => { for (const listener of listeners) listener(snapshot()); };
    function abortPending() {
      generation += 1;
      for (const controller of pending) controller.abort();
      pending.clear();
    }
    function assertGeneration(expected) {
      if (generation !== expected) throw new DOMException("后端连接已切换，已忽略旧请求。", "AbortError");
    }
    function configure({ url = "", accessToken = "", remember = false } = {}) {
      const normalized = normalizeBackendUrl(url);
      if (!normalized && !localPage) throw new Error("网站工作台需要填写远程 HTTPS 后端，不能连接网站自身的数据接口。");
      const nextToken = String(accessToken).trim();
      if (/[\r\n]/.test(nextToken)) throw new Error("访问令牌格式无效。");
      if (normalized && !nextToken && !isLoopback(new URL(normalized).hostname)) throw new Error("远程连接需要访问令牌。");
      abortPending();
      // Forget the previous connection's credential when switching endpoints.
      if (backendUrl !== normalized) write(environment.sessionStorage, sessionKey(backendUrl), "");
      backendUrl = normalized;
      token = nextToken;
      active = true;
      configError = "";
      write(environment.localStorage, ADDRESS_KEY, normalized);
      write(environment.sessionStorage, sessionKey(normalized), remember ? token : "");
      emit();
      return snapshot();
    }
    function disconnect() {
      abortPending();
      write(environment.sessionStorage, sessionKey(backendUrl), "");
      token = "";
      active = false;
      emit();
    }
    async function initialize() {
      const stored = read(environment.localStorage, ADDRESS_KEY);
      if (stored) {
        try { backendUrl = normalizeBackendUrl(stored); } catch (error) { configError = error.message; }
      }
      if (!backendUrl && !localPage) {
        const controller = new AbortController();
        const timeout = environment.setTimeout(() => controller.abort(), 8000);
        try {
          const response = await environment.fetch(new URL("backend-config.json", page.href).href, {
            cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal,
          });
          if (response.ok && (response.headers.get("content-type") || "").includes("json")) {
            const config = await response.json();
            backendUrl = normalizeBackendUrl(config.backend_url || "");
          }
        } catch (_) { /* A static workbench without a default stays disconnected. */ }
        finally { environment.clearTimeout(timeout); }
      }
      // A cloud page always waits for an explicit connection, even when a
      // session credential exists. A loopback-served workbench keeps local use.
      active = localPage && !backendUrl;
      emit();
      return snapshot();
    }
    async function request(path, options = {}) {
      if (!active) {
        const error = new Error("尚未连接仿真后端，请先打开“后端连接”并验证地址和访问令牌。");
        error.code = "connection_required";
        throw error;
      }
      if (!String(path).startsWith("/") || String(path).startsWith("//")) throw new Error("API 路径无效。");
      const expected = generation;
      const controller = new AbortController();
      const onAbort = () => controller.abort();
      const callerSignal = options.signal;
      callerSignal?.addEventListener("abort", onAbort, { once: true });
      if (callerSignal?.aborted) controller.abort();
      pending.add(controller);
      const release = () => {
        pending.delete(controller);
        callerSignal?.removeEventListener("abort", onAbort);
      };
      const headers = new Headers(options.headers || {});
      if (options.body != null && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
      headers.set("Accept", "application/json");
      headers.delete("Authorization");
      if (token) headers.set("Authorization", `Bearer ${token}`);
      let response;
      try {
        response = await environment.fetch(`${backendUrl}/api${path}`, {
          ...options, headers, signal: controller.signal, credentials: "omit", redirect: "error",
        });
        assertGeneration(expected);
      } catch (error) {
        release();
        assertGeneration(expected);
        throw error;
      }
      const consume = async (method) => {
        try {
          const payload = await response[method]();
          assertGeneration(expected);
          return payload;
        } finally { release(); }
      };
      return {
        ok: response.ok, status: response.status, headers: response.headers,
        json: () => consume("json"), text: () => consume("text"),
      };
    }
    return Object.freeze({
      initialize, configure, disconnect, request, snapshot, assertGeneration,
      savedToken: (url) => read(environment.sessionStorage, sessionKey(normalizeBackendUrl(url))),
      subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    });
  }

  const api = Object.freeze({ createConnection, normalizeBackendUrl, isLoopback });
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SimulatorConnection = api;
})(globalThis);
