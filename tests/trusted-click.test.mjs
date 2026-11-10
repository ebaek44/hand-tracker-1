import test from 'node:test';
import assert from 'node:assert/strict';
import { trustedSkip } from '../extension/lib/trusted-click.js';
import { adSkippingEnabled } from '../extension/lib/ad-settings.js';

function mock({ cancelAt, attachError } = {}) {
  const calls = [];
  let current = true;
  globalThis.chrome = {
    debugger: {
      onDetach: { addListener() {} },
      async attach() { calls.push({ action: 'attach' }); if (attachError) throw new Error(attachError); if (cancelAt === 'attach') current = false; },
      async detach() { calls.push({ action: 'detach' }); },
      async sendCommand(target, method, input) { calls.push(input); if (cancelAt === input.type) current = false; },
    },
    scripting: { async executeScript({ args: [method] }) {
      calls.push({ action: method });
      return [{ result: method === 'prepare' ? { ok: true, x: 50, y: 80 } : method === 'waitForResult' ? { ok: true, code: 'confirmed' } : { ok: true } }];
    } },
  };
  return { calls, current: async () => current };
}

test('trusted skip remeasures after attach, sends browser input, and detaches', async () => {
  const state = mock();
  assert.equal((await trustedSkip({ id: 1 }, 0, 1, state.current)).code, 'confirmed');
  assert.deepEqual(state.calls.filter(c => c.type).map(c => c.type), ['mouseMoved', 'mousePressed', 'mouseReleased']);
  assert.equal(state.calls.filter(c => c.action === 'prepare').length, 2);
  assert.equal(state.calls.at(-1).action, 'detach');
});

test('focus/permission loss before press cancels without clicking and detaches', async () => {
  for (const cancelAt of ['attach', 'mouseMoved']) {
    const state = mock({ cancelAt });
    assert.equal((await trustedSkip({ id: 1 }, 0, 1, state.current)).code, 'cancelled');
    assert.ok(!state.calls.some(c => c.type === 'mousePressed'));
    assert.equal(state.calls.at(-1).action, 'detach');
  }
});

test('cancellation during press releases away from the control before detaching', async () => {
  const state = mock({ cancelAt: 'mousePressed' });
  assert.equal((await trustedSkip({ id: 1 }, 0, 1, state.current)).code, 'cancelled');
  const releases = state.calls.filter(c => c.type === 'mouseReleased');
  assert.equal(releases.length, 1);
  assert.equal(releases[0].x, 0);
  assert.equal(releases[0].y, 0);
  assert.equal(state.calls.at(-1).action, 'detach');
});

test('attach failure is reported and does not leave the mode stuck busy', async () => {
  let state = mock({ attachError: 'already attached' });
  assert.equal((await trustedSkip({ id: 1 }, 0, 1, state.current)).code, 'browser-click-failed');
  state = mock();
  assert.equal((await trustedSkip({ id: 1 }, 0, 1, state.current)).code, 'confirmed');
});

test('embedded skip is rejected before browser input can target the wrong frame', async () => {
  const state = mock();
  assert.equal((await trustedSkip({ id: 1 }, 4, 1, state.current)).code, 'embedded');
  assert.equal(state.calls.length, 0);
});

test('ad skipping requires both the saved opt-in and Chrome permission', async () => {
  for (const [enabled, permission, expected] of [[false, false, false], [false, true, false], [true, false, false], [true, true, true]]) {
    globalThis.chrome = {
      storage: { local: { get: async () => ({ youtubeAdSkipping: enabled }) } },
      permissions: { contains: async () => permission },
    };
    assert.equal(await adSkippingEnabled(), expected);
  }
});
