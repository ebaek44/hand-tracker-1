// Event-driven broker only. The offscreen page owns camera and inference;
// Manifest V3 service-worker suspension cannot interrupt the tracking loop.
import { adSkippingEnabled, setAdSkipping } from './lib/ad-settings.js';
import { trustedSkip, detachAdDebugger } from './lib/trusted-click.js';
const offscreenUrl = chrome.runtime.getURL('offscreen.html');
let creation;
let transition = Promise.resolve();
let routing = null;
let routeGeneration = 0;
let feedbackSequence = 0;

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
  if (routing) return { ok: false, code: 'busy', error: 'Another control is being handled. Try again.' };
  const owner = {};
  routing = owner;
  const order = ++feedbackSequence;
  const current = routeGeneration;
  const skip = payload.type === 'control' && payload.value === 'skip_ad';
  let stage = 'tab';
  const finish = async result => {
    try {
      if (current === routeGeneration && order === feedbackSequence && await hasTracker()) {
        await chrome.runtime.sendMessage({ target: 'tracker', action: 'feedback', feedback: { ...result, action: payload.value } });
      }
    } catch (error) { console.debug('[Hands On] Status unavailable:', error.message); }
    return result;
  };
  try {
    if (skip && !await adSkippingEnabled()) return finish({ ok: false, code: 'disabled', error: 'Enable YouTube ad skipping in Options to use point up.' });
    const window = await chrome.windows.getLastFocused();
    if (!window.focused) return finish({ ok: false, code: 'not-focused', error: 'Bring the video’s Chrome window to the front and try again.' });
    const [tab] = await chrome.tabs.query({ active: true, windowId: window.id });
    if (!tab?.id || !/^https?:/.test(tab.url ?? '')) return finish({ ok: false, code: 'not-web-page', error: 'Open a video website in the active tab and try again.' });
    // Injection is idempotent. Scan only on an actual gesture, never on a timer.
    stage = 'injection';
    await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, files: skip ? ['youtube.js'] : ['content.js'] });
    stage = skip ? 'button-search' : 'video-search';
    const candidates = await chrome.scripting.executeScript({
      target: { tabId: tab.id, allFrames: true },
      func: skip => skip ? globalThis.__handsOnYouTube?.inspect() ?? null : globalThis.__handsOnController?.probe() ?? null,
      args: [skip],
    });
    const best = candidates.filter(c => skip ? c.result?.available : c.result).sort((a, b) => b.result.score - a.result.score)[0];
    if (!best) {
      return finish(skip ? candidates.some(c => c.result?.supported)
        ? { ok: false, code: 'unavailable', stage, error: 'Skip ad is not available yet.' }
        : { ok: false, code: 'unsupported', stage, error: 'Ad skipping is available on YouTube.' }
        : { ok: false, code: 'no-video', stage, error: 'No visible, ready video found' });
    }
    const [active] = await chrome.tabs.query({ active: true, windowId: window.id });
    const focused = await chrome.windows.getLastFocused();
    if (current !== routeGeneration || active?.id !== tab.id || focused.id !== window.id || !focused.focused) return finish({ ok: false, code: 'cancelled', error: 'The active video tab changed. Try again.' });
    stage = 'dispatch';
    const response = skip ? trustedSkip(tab, best.frameId, best.result.token, async () => {
      if (current !== routeGeneration || !await adSkippingEnabled()) return false;
      const [active] = await chrome.tabs.query({ active: true, windowId: window.id });
      const focused = await chrome.windows.getLastFocused();
      return active?.id === tab.id && active.url === tab.url && focused.id === window.id && focused.focused;
    }) : chrome.tabs.sendMessage(tab.id,
      { kind: 'hands-on/gesture', payload, token: best.result.token }, { frameId: best.frameId });
    // Skip confirmation may take a moment; do not block play/pause or seeking.
    routing = null;
    const result = await response;
    return finish(result ?? { ok: false, code: 'no-response', stage, error: 'The video page did not respond. Refresh it and try again.' });
  } catch (error) {
    console.debug('[Hands On] Video unavailable:', error.message);
    return finish({ ok: false, code: 'routing-error', stage, error: `Couldn’t reach the video page (${stage}). Refresh it and try again.` });
  } finally { if (routing === owner) routing = null; }
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
    if (message.action === 'get-ad-settings') return { enabled: Boolean(await adSkippingEnabled()) };
    if (message.action === 'set-ad-skipping') {
      routeGeneration++;
      if (!message.enabled) {
        const result = await setAdSkipping(false);
        await detachAdDebugger();
        return result;
      }
      return setAdSkipping(Boolean(message.enabled));
    }
    if (message.action === 'get-status') return await hasTracker() ? trackerMessage('get-status') : { state: 'stopped' };
    if (message.action === 'start') {
      routeGeneration++;
      await ensureTracker();
      return trackerMessage('start');
    }
    if (message.action === 'stop') {
      routeGeneration++;
      await detachAdDebugger();
      if (await hasTracker()) {
        await trackerMessage('stop');
        await chrome.offscreen.closeDocument();
      }
      await chrome.action.setBadgeText({ text: '' });
      return { state: 'stopped' };
    }
    throw new Error('Unknown action');
  };
  const independent = ['get-status', 'get-ad-settings'].includes(message.action);
  const operation = independent ? run() : (transition = transition.catch(() => {}).then(run));
  operation.then(respond, error => respond({ state: 'error', error: error.message }));
  return true;
});
chrome.runtime.onStartup.addListener(() => { void chrome.action.setBadgeText({ text: '' }); });
chrome.runtime.onInstalled.addListener(() => { void chrome.action.setBadgeText({ text: '' }); });
chrome.permissions.onRemoved.addListener(permissions => {
  if (!permissions.permissions?.includes('debugger')) return;
  routeGeneration++;
  void setAdSkipping(false);
  void detachAdDebugger();
});
