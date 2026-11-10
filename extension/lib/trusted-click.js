// Uses only the Input domain, only for the already-visible native Skip ad control.
// Chrome requires declaring debugger at installation; use is gated by the saved opt-in.
let active = null;

export async function detachAdDebugger() {
  const session = active;
  active = null;
  if (!session?.attached) return;
  if (session.pressed) {
    try {
      await chrome.debugger.sendCommand({ tabId: session.tabId }, 'Input.dispatchMouseEvent',
        { type: 'mouseReleased', x: 0, y: 0, button: 'left', buttons: 0, clickCount: 1 });
    } catch { /* The tab may already be gone. */ }
  }
  try { await chrome.debugger.detach({ tabId: session.tabId }); } catch { /* Already detached. */ }
}

export async function trustedSkip(tab, frameId, token, stillCurrent) {
  if (frameId !== 0) return { ok: false, code: 'embedded', error: 'Open the YouTube video in its own tab to skip ads.' };
  if (active) return { ok: false, code: 'busy', error: 'An ad-skip click is already being checked.' };
  listenForDebuggerDetach();
  const session = { tabId: tab.id, attached: false, pressed: false };
  active = session;
  const target = { tabId: tab.id };
  const inspect = async method => {
    const [execution] = await chrome.scripting.executeScript({
      target: { tabId: tab.id, frameIds: [0] },
      func: (method, token) => globalThis.__handsOnYouTube?.[method](token), args: [method, token],
    });
    return execution?.result;
  };
  const valid = async () => active === session && await stillCurrent();
  try {
    if (!await valid()) return { ok: false, code: 'cancelled', error: 'Skip cancelled because the active tab changed.' };
    await chrome.debugger.attach(target, '1.3');
    session.attached = true;
    if (!await valid()) return { ok: false, code: 'cancelled', error: 'Skip cancelled.' };
    // Attaching may show Chrome's banner and resize the viewport. Remeasure after it.
    let control = await inspect('prepare');
    if (!control?.ok) return control ?? { ok: false, code: 'unavailable', error: 'Skip ad is no longer available.' };
    await chrome.debugger.sendCommand(target, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x: control.x, y: control.y, button: 'none', buttons: 0 });
    if (!await valid()) return { ok: false, code: 'cancelled', error: 'Skip cancelled.' };
    control = await inspect('prepare');
    if (!control?.ok) return control ?? { ok: false, code: 'unavailable', error: 'Skip ad is no longer available.' };
    const observation = await inspect('beginVerification');
    if (!observation?.ok) return observation;
    if (!await valid()) return { ok: false, code: 'cancelled', error: 'Skip cancelled.' };
    session.pressed = true;
    await chrome.debugger.sendCommand(target, 'Input.dispatchMouseEvent', { type: 'mousePressed', x: control.x, y: control.y, button: 'left', buttons: 1, clickCount: 1 });
    if (!await valid()) return { ok: false, code: 'cancelled', error: 'Skip cancelled.' };
    await chrome.debugger.sendCommand(target, 'Input.dispatchMouseEvent', { type: 'mouseReleased', x: control.x, y: control.y, button: 'left', buttons: 0, clickCount: 1 });
    session.pressed = false;
    return await inspect('waitForResult') ?? { ok: false, code: 'unconfirmed', error: 'YouTube’s response was not confirmed.' };
  } catch (error) {
    console.debug('[Hands On] Ad click failed:', error.message);
    return { ok: false, code: 'browser-click-failed', error: 'Couldn’t control the Skip ad button. Close DevTools on the video tab and try again.' };
  } finally {
    if (active === session) await detachAdDebugger();
    else if (session.attached) { try { await chrome.debugger.detach(target); } catch { /* Already detached. */ } }
  }
}

let listening = false;
function listenForDebuggerDetach() {
  if (listening || !chrome.debugger?.onDetach) return;
  listening = true;
  chrome.debugger.onDetach.addListener((source, reason) => {
    if (source.tabId !== active?.tabId) return;
    active = null;
    if (reason === 'canceled_by_user') void chrome.storage.local.set({ youtubeAdSkipping: false });
  });
}
