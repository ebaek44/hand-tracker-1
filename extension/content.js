function getActiveVideo() {
  const vids = Array.from(document.querySelectorAll("video"));
  if (!vids.length) return null;

  const inView = (v) => {
    const r = v.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 &&
           r.top < (window.innerHeight || document.documentElement.clientHeight) &&
           r.left < (window.innerWidth  || document.documentElement.clientWidth);
  };

  const visible = vids.filter(inView);
  const pool = visible.length ? visible : vids;

  const playing = pool.filter(v => !v.paused && !v.ended && v.readyState > 2);
  const byArea = (a,b) => (b.clientWidth*b.clientHeight) - (a.clientWidth*a.clientHeight);
  return (playing.length ? playing : pool).sort(byArea)[0] || null;
}

function togglePlayPause(v){ if(!v) return; v.paused ? v.play().catch(()=>{}) : v.pause(); }
function seekRelative(v, s){
  if(!v) return;
  try {
    const t = Math.max(0, Math.min((isFinite(v.duration)? v.duration : 1e9), v.currentTime + s));
    v.currentTime = t;
  } catch {}
}

// Optional debounce in case of duplicate events
let lastAction = { key: "", ts: 0 };
function shouldAct(key, ms=120) {
  const now = Date.now();
  if (lastAction.key === key && (now - lastAction.ts) < ms) return false;
  lastAction = { key, ts: now }; return true;
}

chrome.storage.sync.get(
  { gestureUrl: "http://localhost:3000", gestureNs: "/gestures" },
  ({ gestureUrl, gestureNs }) => {
    try {
      const socket = io(gestureUrl + gestureNs, { transports: ["websocket"], reconnection: true });

      socket.on("gesture", (payload) => {
        const v = getActiveVideo(); if (!v || !payload) return;

        if (payload.type === "control" && payload.value === "toggle_play_pause") {
          if (!shouldAct("pp")) return;
          togglePlayPause(v);
        }
        if (payload.type === "swipe" && payload.value === "left")  {
          if (!shouldAct("seekL")) return;
          seekRelative(v, -10);
        }
        if (payload.type === "swipe" && payload.value === "right") {
          if (!shouldAct("seekR")) return;
          seekRelative(v,  10);
        }
      });
    } catch (e) {
      // Silent fail keeps content script safe if socket lib fails to load
      console.warn("[Hands-On] Socket init error:", e);
    }
  }
);
