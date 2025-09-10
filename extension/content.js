// content.js
// This will get the gesture from the background worker and make it work on the actual video
(() => {
  // Small probe so you can check injection quickly
  window.__handsOnProbe = { loadedAt: Date.now(), href: location.href };

  // --- helpers ---
  function getAllVideos(root = document) {
    const vids = new Set(root.querySelectorAll('video'));
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) {
        for (const v of el.shadowRoot.querySelectorAll?.('video') || []) vids.add(v);
      }
    }
    return Array.from(vids);
  }

  function elementIsOnscreen(el) {
    const r = el.getBoundingClientRect();
    const vw = window.innerWidth || document.documentElement.clientWidth;
    const vh = window.innerHeight || document.documentElement.clientHeight;
    return r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 && r.left < vw && r.top < vh;
  }
  // Picks the videos that are playing
  function pickActiveVideo() {
    const vids = getAllVideos();
    if (!vids.length) return null;
    const onscreen = vids.filter(elementIsOnscreen);
    const pool = onscreen.length ? onscreen : vids;
    const playing = pool.filter(v => !v.paused && !v.ended && v.readyState > 2);
    const audible = playing.filter(v => v.volume > 0 && !v.muted);
    const byArea = (a, b) => (b.clientWidth * b.clientHeight) - (a.clientWidth * a.clientHeight);
    return (audible[0] ?? playing.sort(byArea)[0] ?? pool.sort(byArea)[0]) || null;
  }
  // Command that is sent to youtube which will actually play or pause the video
  async function togglePlayPause(v) {
    if (!v) return false;
    try {
      if (v.paused) { await v.play(); } else { v.pause(); }
      return true;
    } catch {
      try {
        if (v.paused) {
          const wasMuted = v.muted;
          v.muted = true; await v.play(); setTimeout(() => { try { v.muted = wasMuted; } catch {} }, 200);
        } else { v.pause(); }
        return true;
      } catch {
        try { v.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window })); return true; }
        catch { return false; }
      }
    }
  }
  // Comamnd sent to youtube to skip forward or backward
  function seekRelative(v, deltaSecs) {
    if (!v) return;
    try {
      const dur = v.duration;
      const isFiniteDur = Number.isFinite(dur);
      const target = (v.currentTime || 0) + deltaSecs;
      if (!isFiniteDur) v.currentTime = Math.max(0, target);
      else v.currentTime = Math.max(0, Math.min(dur, target));
    } catch {}
  }

  // Debounce bursts
  let lastAction = { key: '', ts: 0 };
  function shouldAct(key, ms = 120) {
    const now = Date.now();
    if (lastAction.key === key && (now - lastAction.ts) < ms) return false;
    lastAction = { key, ts: now };
    return true;
  }

  // Listens to the back ground worker
  chrome.runtime.onMessage.addListener(async (msg) => {
    if (msg?.kind !== 'hands-on/gesture') return;
        console.log('[Hands-On/content] gesture received:', msg.payload);
    const payload = msg.payload;
    const v = document.querySelector('video.html5-main-video') || pickActiveVideo();
    if (!v || !payload) return;

    if (payload.type === 'control' && payload.value === 'toggle_play_pause') {
      if (!shouldAct('pp')) return;
      await togglePlayPause(v);
    } else if (payload.type === 'swipe' && payload.value === 'left') {
      if (!shouldAct('seekL')) return;
      seekRelative(v, -10);
    } else if (payload.type === 'swipe' && payload.value === 'right') {
      if (!shouldAct('seekR')) return;
      seekRelative(v, 10);
    }
  });
})();
