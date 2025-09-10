// keepalive.js
// This script keeps our port running so that we can always look for gestures
(() => {
  // only run in the top frame
  if (window.top !== window) return;

  // if an older copy exists on this page, clean it up first
  if (window.__handsOnKeepalive?.cleanup) {
    try { window.__handsOnKeepalive.cleanup(); } catch {}
  }

  const state = {
    port: null,
    heartbeatId: null,
    alive: false,
    cleanup() {
      this.alive = false;
      if (this.heartbeatId) { clearInterval(this.heartbeatId); this.heartbeatId = null; }
      try { this.port?.disconnect?.(); } catch {}
      this.port = null;
    }
  };
  window.__handsOnKeepalive = state;

  const PING_MS = 20000;

  function connect() {
    // if the extension was reloaded, runtime.id will be undefined in this page
    if (!chrome?.runtime?.id) {
      state.cleanup();
      // don’t spam logs here: context is gone. We’ll reconnect on next inject.
      return;
    }

    try {
      state.port = chrome.runtime.connect({ name: "keepalive" });
      // console.log("[Hands-On/content] Keepalive port opened");
    } catch {
      state.cleanup();
      setTimeout(connect, 1000);
      return;
    }

    state.alive = true;

    state.heartbeatId = setInterval(() => {
      // context/port can die at any time (SPA nav, extension reload)
      if (!state.alive || !chrome?.runtime?.id || !state.port) {
        state.cleanup();
        setTimeout(connect, 1000);
        return;
      }
      try {
        state.port.postMessage({ type: "ping" });
      } catch {
        // stale port — clean & reconnect (no console spam)
        state.cleanup();
        setTimeout(connect, 1000);
      }
    }, PING_MS);

    state.port.onDisconnect.addListener(() => {
      state.cleanup();
      setTimeout(connect, 1000);
    });
  }

  connect();

  // Reestablish visibility
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.heartbeatId && !state.port) {
      connect();
    }
  });

  // page is going away
  window.addEventListener("pagehide", () => state.cleanup());
})();