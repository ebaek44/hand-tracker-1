// background.js
// This is our background worker that is polling for gestures sent from the backend using the localhost
// This will whisper to content.js which will execute the command
let polling = false;
let pollTimer = null;
let pollController = null;
let backoffMs = 1000;

// Function to start polling
async function startPolling() {
  if (polling) {
    console.log('[Hands-On/bg] Already polling');
    return;
  }

  // Read settings
  const settings = await chrome.storage.sync.get({
    gestureUrl: "http://localhost:3333",
    gestureNs: "/gestures",
  });
  const pollUrl = `${settings.gestureUrl}/poll-gestures`;
  const healthUrl = `${settings.gestureUrl}/health`;
  console.log('[Hands-On/bg] Starting HTTP polling to', pollUrl);

  // Quick health check (not needed but good for debugging)
  try {
    const r = await fetch(healthUrl, { method: 'GET', headers: { 'Accept': 'application/json' }, cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const health = await r.json();
    console.log('[Hands-On/bg] Server health check:', health);
  } catch (e) {
    console.warn('[Hands-On/bg] Server not reachable:', e?.message || e);
    scheduleRestart(); 
    return;
  }

  // Arm polling
  polling = true;
  backoffMs = 1000; // reset backoff on success
  pollController = new AbortController();

  const tick = async () => {
    if (!polling) return;

    // snapshot the controller to avoid the null .signal race
    const ctrl = pollController;
    if (!ctrl || ctrl.signal.aborted) return;

    try {
      const resp = await fetch(pollUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        cache: 'no-store',
        signal: ctrl.signal,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      if (data?.type && data.type !== 'none') {
        console.log('[Hands-On/bg] Received gesture:', data);
        broadcastGesture(data);
      }
    } catch (err) {
    // If expecting to stop then we can ignore the error
      if (!polling) return;                
      if (err?.name === 'AbortError') return;  p
      console.error('[Hands-On/bg] Poll error:', err?.message || err);

      // hard stop and schedule a restart with backoff
      stopPolling(/*abort=*/true);
      scheduleRestart();
    }
  };

  // kick once immediately so we don’t wait 150ms
  tick();
  pollTimer = setInterval(tick, 150);
}

function stopPolling(abort = true) {
  if (!polling && !pollTimer && !pollController) return;

  console.log('[Hands-On/bg] Stopping polling');
  polling = false;

  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }

  // abort controller after clearing the interval (prevents race)
  if (abort && pollController) {
    try { pollController.abort(); } catch {}
  }
  pollController = null;
}

function scheduleRestart() {
  // exponential backoff up to 30s
  const delay = Math.min(backoffMs, 30000);
  console.log(`[Hands-On/bg] Scheduling restart in ${Math.round(delay/1000)}s...`);
  setTimeout(() => {
    if (!polling) startPolling();
  }, delay);
  backoffMs = Math.min(delay * 2, 30000);
}
// broadcasts the gesture to the api
function broadcastGesture(payload) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    for (const tab of tabs) {
      if (!tab.id) continue;
      chrome.tabs.sendMessage(
        tab.id,
        { kind: 'hands-on/gesture', payload },
        () => { void chrome.runtime.lastError; }
      );
    }
    console.log(`[Hands-On/bg] Broadcasting gesture to ${tabs.length} tab(s)`);
  });
}
// lifecycle
chrome.runtime.onStartup.addListener(() => startPolling());
chrome.runtime.onInstalled.addListener(() => startPolling());
chrome.runtime.onSuspend.addListener(() => stopPolling(true));
chrome.runtime.onSuspendCanceled?.addListener(() => startPolling());
