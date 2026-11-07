// Event-driven broker only. The offscreen page owns camera and inference;
// Manifest V3 service-worker suspension cannot interrupt the tracking loop.
const offscreenUrl = chrome.runtime.getURL('offscreen.html');
let creation;
let transition = Promise.resolve();
let routing = false;
let routeGeneration = 0;

async function hasTracker() {
  return (await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'], documentUrls: [offscreenUrl] })).length > 0;
}
async function ensureTracker() {
  if (await hasTracker()) return;
  creation ??= chrome.offscreen.createDocument({
    url: 'offscreen.html', reasons: ['USER_MEDIA', 'WORKERS'],
    justification: 'Capture the user-approved camera and classify hand gestures locally in a worker.'
  }).finally(() => { creation = undefined; });
  await creation;
}
const trackerMessage = action => chrome.runtime.sendMessage({ target: 'tracker', action });

async function routeGesture(payload) {
  if (routing) return;
  routing = true;
  const current = routeGeneration;
  try {
    const window = await chrome.windows.getLastFocused();
    if (!window.focused) return;
    const [tab] = await chrome.tabs.query({ active: true, windowId: window.id });
    if (!tab?.id || !/^https?:/.test(tab.url ?? '')) return;
    // Injection is idempotent. Scan only on an actual gesture, never on a timer.
    await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, files: ['content.js'] });
    const candidates = await chrome.scripting.executeScript({
      target: { tabId: tab.id, allFrames: true },
      func: () => globalThis.__handsOnController?.probe() ?? null,
    });
    const best = candidates.filter(c => c.result).sort((a, b) => b.result.score - a.result.score)[0];
    if (!best) {
      await chrome.runtime.sendMessage({ target: 'tracker', action: 'feedback', feedback: { error: 'No visible, ready video found' } });
      return;
    }
    const [active] = await chrome.tabs.query({ active: true, windowId: window.id });
    const focused = await chrome.windows.getLastFocused();
    if (current !== routeGeneration || active?.id !== tab.id || focused.id !== window.id || !focused.focused) return;
    const result = await chrome.tabs.sendMessage(tab.id,
      { kind: 'hands-on/gesture', payload, token: best.result.token }, { frameId: best.frameId });
    await chrome.runtime.sendMessage({ target: 'tracker', action: 'feedback', feedback: result });
  } catch (error) {
    console.debug('[Hands On] Video unavailable:', error.message);
  } finally { routing = false; }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id || !sender.url?.startsWith(chrome.runtime.getURL(''))) return;
  if (message?.target !== 'background') return;
  const fromTracker = sender.url === offscreenUrl;
  if (message.action === 'gesture' && fromTracker) {
    void routeGesture(message.payload);
    return;
  }
  if (message.action === 'status' && fromTracker) {
    const active = ['starting', 'running'].includes(message.status.state);
    void chrome.action.setBadgeText({ text: active ? 'ON' : message.status.state === 'error' ? '!' : '' });
    void chrome.action.setBadgeBackgroundColor({ color: message.status.state === 'error' ? '#b54430' : '#226b53' });
    return;
  }
  const run = async () => {
    if (message.action === 'get-status') return await hasTracker() ? trackerMessage('get-status') : { state: 'stopped' };
    if (message.action === 'start') {
      routeGeneration++;
      await ensureTracker();
      return trackerMessage('start');
    }
    if (message.action === 'stop') {
      routeGeneration++;
      if (await hasTracker()) {
        await trackerMessage('stop');
        await chrome.offscreen.closeDocument();
      }
      await chrome.action.setBadgeText({ text: '' });
      return { state: 'stopped' };
    }
    throw new Error('Unknown action');
  };
  const operation = message.action === 'get-status' ? run() : (transition = transition.catch(() => {}).then(run));
  operation.then(respond, error => respond({ state: 'error', error: error.message }));
  return true;
});
chrome.runtime.onStartup.addListener(() => { void chrome.action.setBadgeText({ text: '' }); });
chrome.runtime.onInstalled.addListener(() => { void chrome.action.setBadgeText({ text: '' }); });
