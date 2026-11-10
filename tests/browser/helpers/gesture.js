// Exercise the authenticated production route; there is no manual/test action.
export async function pointUp(context, popup) {
  const before = await popup.evaluate(async () => (await chrome.runtime.sendMessage({ target: 'background', action: 'get-status' })).feedbackAt ?? 0);
  const cdp = await context.newCDPSession(popup);
  try {
    const { targetInfos } = await cdp.send('Target.getTargets');
    const offscreen = targetInfos.find(target => target.url.endsWith('/offscreen.html'));
    if (!offscreen) throw new Error('Tracking must be running');
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId: offscreen.targetId, flatten: false });
    const execution = new Promise((resolve, reject) => {
      const listener = event => {
        const message = JSON.parse(event.message);
        if (event.sessionId !== sessionId || message.id !== 1) return;
        cdp.off('Target.receivedMessageFromTarget', listener);
        if (message.error || message.result?.exceptionDetails) reject(new Error(JSON.stringify(message)));
        else resolve();
      };
      cdp.on('Target.receivedMessageFromTarget', listener);
    });
    await cdp.send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id: 1, method: 'Runtime.evaluate', params: {
      expression: `(async () => { const { HoldGestureEngine } = await import(chrome.runtime.getURL('lib/gestures.js')); const engine = new HoldGestureEngine(); let payload; for (let t = 0; t <= 600; t += 100) payload = engine.update('skip-ad', t) || payload; return chrome.runtime.sendMessage({target:'background',action:'gesture',payload}); })()`, awaitPromise: true,
    } }) });
    await execution;
    await cdp.send('Target.detachFromTarget', { sessionId });
  } finally { await cdp.detach(); }
  return before;
}
