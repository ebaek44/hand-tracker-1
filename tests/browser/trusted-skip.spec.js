import { test, expect, chromium } from '@playwright/test';
import { cp, readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pointUp } from './helpers/gesture.js';

test('opt-in browser click is trusted, targets the visible ad, and detaches', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'hands-on-trusted-'));
  const extension = path.join(directory, 'extension');
  await cp(fileURLToPath(new URL('../../extension/', import.meta.url)), extension, { recursive: true });
  const manifestFile = path.join(extension, 'manifest.json');
  const manifest = JSON.parse(await readFile(manifestFile, 'utf8'));
  expect(manifest.permissions).toContain('debugger');
  expect(manifest.optional_permissions ?? []).not.toContain('debugger');
  // Load the unchanged production manifest in an isolated profile.
  const context = await chromium.launchPersistentContext(path.join(directory, 'profile'), {
    channel: 'chromium', headless: true,
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  try {
    let [background] = context.serviceWorkers();
    background ??= await context.waitForEvent('serviceworker');
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${new URL(background.url()).host}/popup.html`);
    await expect(popup.locator('#ad-skipping')).not.toBeChecked();
    await popup.locator('.options summary').click();
    await popup.locator('#ad-skipping').check();
    await expect(popup.locator('#ad-option-status')).toContainText('Ad skipping is on');
    await popup.locator('body').screenshot({ path: '/private/tmp/hands-on-ad-option-on.png' });
    const page = await context.newPage();
    let markup = '';
    await context.route('https://www.youtube.com/**', route => route.fulfill({ contentType: 'text/html', body: markup }));
    const fixture = (disabled = false, requiresTrusted = true) => `
      <div id="movie_player" class="html5-video-player ad-showing">
        <button id="skip" class="ytp-skip-ad-button-modern" ${disabled ? 'disabled' : ''}>Skip ad</button>
      </div>
      <script>
        window.clicks = 0; window.trustedClicks = 0;
        document.getElementById('skip').onclick = event => {
          window.clicks++;
          if (${requiresTrusted} && !event.isTrusted) return;
          window.trustedClicks++;
          document.getElementById('movie_player').classList.remove('ad-showing');
          event.currentTarget.remove();
        };
      </script>`;
    const load = async html => {
      markup = html;
      await page.goto(`https://www.youtube.com/watch?v=trusted-${Date.now()}`);
      await page.bringToFront();
    };
    await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'start' }));
    await expect.poll(() => popup.evaluate(async () => (await chrome.runtime.sendMessage({ target: 'background', action: 'get-status' })).state)).toBe('running');
    const skip = async () => {
      const before = await pointUp(context, popup);
      await expect.poll(() => popup.evaluate(async () => (await chrome.runtime.sendMessage({ target: 'background', action: 'get-status' })).feedbackAt ?? 0)).toBeGreaterThan(before);
      const status = await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'get-status' }));
      return { ok: status.skipResult === 'confirmed', code: status.skipResult };
    };
    // Every request uses the real offscreen gesture engine -> broker -> trusted input.
    // No video prerequisite, and the current YouTube trust gate is reproduced.
    await load(fixture());
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed' });
    expect(await page.evaluate(() => window.trustedClicks)).toBe(1);
    const detached = () => background.evaluate(async () => {
      const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
      try { await chrome.debugger.sendCommand({ tabId: tab.id }, 'Page.getFrameTree', {}); return false; }
      catch { return true; }
    });
    // getTargets().attached also includes Playwright's own CDP connection;
    // sendCommand tests whether this extension still owns a debugger session.
    expect(await detached()).toBe(true);

    await load(fixture(true));
    expect(await skip()).toMatchObject({ ok: false, code: 'unavailable' });
    await page.locator('#skip').evaluate(button => { button.disabled = false; });
    await page.waitForTimeout(1500);
    expect(await page.evaluate(() => window.clicks)).toBe(0);
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed' });

    await popup.locator('#ad-skipping').uncheck();
    await expect(popup.locator('#ad-option-status')).toContainText('Ad skipping is off');
    await load(fixture());
    expect(await skip()).toMatchObject({ ok: false, code: 'disabled' });
    expect(await page.evaluate(() => window.clicks)).toBe(0);
    expect(await detached()).toBe(true);
    await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'stop' }));
  } finally { await context.close(); await rm(directory, { recursive: true, force: true }); }
});
