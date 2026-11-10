import { test, expect, chromium } from '@playwright/test';
import { mkdtemp, rm, access } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';

test('packaged camera inference, lifecycle, video routing, and training studio', async () => {
  const profile = await mkdtemp(path.join(tmpdir(), 'hands-on-test-'));
  const extension = fileURLToPath(new URL('../../extension/', import.meta.url));
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium', headless: true,
    ignoreDefaultArgs: ['--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding'],
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`,
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  try {
    let [background] = context.serviceWorkers();
    background ??= await context.waitForEvent('serviceworker');
    const id = new URL(background.url()).host;
    const popup = await context.newPage();
    const errors = [];
    context.on('weberror', e => errors.push(e.error().message));
    await popup.goto(`chrome-extension://${id}/popup.html`);
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    await expect(popup.locator('#training')).toBeHidden();
    await expect(popup.locator('#try-skip')).toHaveCount(0);
    await expect(popup.locator('#setup')).toHaveCount(0);
    await popup.locator('.options summary').click();
    await expect(popup.locator('#training-mode')).not.toBeChecked();
    await popup.locator('#training-mode').check();
    await expect(popup.locator('#training')).toBeVisible();
    await expect(popup.locator('#ad-skipping')).not.toBeChecked();
    expect(await popup.evaluate(() => chrome.permissions.contains({ permissions: ['debugger'] }))).toBe(true);
    await popup.locator('.options summary').click();
    await popup.locator('body').screenshot({ path: '/private/tmp/hands-on-popup.png' });
    // Run all inference offline; every JS/WASM/model asset must be packaged.
    await context.setOffline(true);
    await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'start' }));
    await expect(popup.locator('#state')).toHaveText('You’re in control', { timeout: 40000 });
    await expect.poll(() => popup.locator('#fps').textContent()).not.toBe('0');
    await expect.poll(() => popup.locator('#latency').textContent()).not.toBe('—');
    const first = await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'get-status' }));
    expect(first.hand).toBe('No hand detected');
    expect(first.fps).toBeLessThanOrEqual(16);
    console.log('Offline tracking:', first);
    await popup.locator('body').screenshot({ path: '/private/tmp/hands-on-running.png' });
    const targets = await background.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
    expect(targets).toHaveLength(1);
    // A seekable, local video fixture exercises the real broker and content scripts.
    await context.setOffline(false);
    const clip = await readFile(new URL('../fixtures/video.webm', import.meta.url));
    const server = createServer((request, response) => {
      if (request.url === '/clip.webm') {
        response.writeHead(200, { 'Content-Type': 'video/webm', 'Content-Length': clip.length, 'Accept-Ranges': 'bytes' }); response.end(clip);
      } else {
        response.writeHead(200, { 'Content-Type': 'text/html' });
        response.end(request.url === '/frame'
          ? '<video id="embedded" src="/clip.webm" preload="auto" style="width:320px;height:180px"></video>'
          : '<video id="main" src="/clip.webm" preload="auto" style="width:640px;height:360px"></video><video id="other" src="/clip.webm" preload="auto" style="width:100px;height:60px"></video><iframe src="/frame" width="330" height="200"></iframe>');
      }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const page = await context.newPage();
    const cdp = await context.newCDPSession(popup);
    const allTargets = await cdp.send('Target.getTargets');
    const offscreen = allTargets.targetInfos.find(target => target.url.endsWith('/offscreen.html'));
    expect(offscreen).toBeTruthy();
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId: offscreen.targetId, flatten: false });
    let commandId = 0;
    async function fromTracker(payload) {
      const id = ++commandId;
      const result = new Promise((resolve, reject) => {
        const listener = event => {
          const message = JSON.parse(event.message);
          if (event.sessionId !== sessionId || message.id !== id) return;
          cdp.off('Target.receivedMessageFromTarget', listener);
          if (message.error || message.result?.exceptionDetails) reject(new Error(JSON.stringify(message)));
          else resolve(message.result);
        };
        cdp.on('Target.receivedMessageFromTarget', listener);
      });
      await cdp.send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method: 'Runtime.evaluate', params: {
        expression: `chrome.runtime.sendMessage({target:'background',action:'gesture',payload:${JSON.stringify(payload)}})`, awaitPromise: true,
      } }) });
      return result;
    }
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}`);
      await page.bringToFront();
      await expect.poll(() => page.locator('#main').evaluate(v => v.readyState)).toBeGreaterThanOrEqual(2);
      await expect.poll(() => page.locator('#main').evaluate(v => v.seekable.length)).toBeGreaterThan(0);
      await page.locator('#main').evaluate(v => { v.currentTime = 20; });
      await fromTracker({ type: 'seek', value: 'right' });
      await expect.poll(() => page.locator('#main').evaluate(v => Math.round(v.currentTime))).toBe(30);
      expect(await page.locator('#other').evaluate(v => v.currentTime)).toBe(0);
      const embedded = page.frameLocator('iframe').locator('#embedded');
      expect(await embedded.evaluate(v => v.currentTime)).toBe(0);
      // Observe feedback to ensure the previous route has completed.
      await expect.poll(() => popup.evaluate(async () => (await chrome.runtime.sendMessage({ target: 'background', action: 'get-status' })).feedback)).toBe('Video controlled');
      await page.waitForTimeout(250);
      await fromTracker({ type: 'seek', value: 'left' });
      await expect.poll(() => page.locator('#main').evaluate(v => Math.round(v.currentTime))).toBe(20);
      await page.waitForTimeout(250);
      await fromTracker({ type: 'control', value: 'toggle_play_pause' });
      await expect.poll(() => page.locator('#main').evaluate(v => v.paused)).toBe(false);
      await page.waitForTimeout(250);
      await fromTracker({ type: 'control', value: 'toggle_play_pause' });
      await expect.poll(() => page.locator('#main').evaluate(v => v.paused)).toBe(true);
      // Embedded video is selected when the main videos are hidden.
      await page.locator('#main').evaluate(v => { v.style.display = 'none'; });
      await page.locator('#other').evaluate(v => { v.style.display = 'none'; });
      await page.waitForTimeout(250);
      await fromTracker({ type: 'seek', value: 'right' });
      await expect.poll(() => embedded.evaluate(v => Math.round(v.currentTime))).toBe(10);
      // Ad skipping is opt-in. Point-up cannot enable browser control.
      await fromTracker({ type: 'control', value: 'skip_ad' });
      await expect.poll(() => popup.evaluate(async () => (await chrome.runtime.sendMessage({ target: 'background', action: 'get-status' })).feedback)).toBe('Enable YouTube ad skipping in Options to use point up.');
      // Tracking must survive the MV3 broker going idle/restarting.
      await cdp.send('Target.detachFromTarget', { sessionId });
      await cdp.send('ServiceWorker.enable');
      await cdp.send('ServiceWorker.stopAllWorkers');
      const resumed = await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'get-status' }));
      expect(resumed.state).toBe('running');
      expect(resumed.inferenceMs).toBeGreaterThan(0);
      background = context.serviceWorkers().find(w => w.url().endsWith('/background.js'));
      console.log('Video routing: play, pause, ±10s, single-frame selection, embedded video, worker restart passed.');
    } finally {
      await page.close();
      await new Promise(resolve => { server.close(resolve); server.closeAllConnections(); });
      await cdp.detach();
    }
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    expect(await background.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }))).toHaveLength(0);
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('You’re in control', { timeout: 30000 });
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    await context.setOffline(false);
    const training = await context.newPage();
    await training.goto(`chrome-extension://${id}/training.html`);
    await expect(training.locator('#label option')).toHaveCount(6);
    await training.locator('#connect').click();
    await expect(training.locator('#connect')).toHaveText('Disconnect camera', { timeout: 30000 });
    await expect(training.locator('#record')).toBeEnabled();
    await training.locator('#record').click();
    await expect(training.locator('#guidance')).toContainText('0 samples collected', { timeout: 10000 });
    // Simulated camera has no hands; do not save bogus training rows.
    await training.locator('#connect').click();
    await training.screenshot({ path: '/private/tmp/hands-on-training.png', fullPage: true });
    await expect(training.locator('#connect')).toHaveText('Connect camera');
    const seed = JSON.parse(await readFile(new URL('../../extension/models/training-seed.json', import.meta.url), 'utf8'));
    const labels = [...seed.labels, { name: 'Point right', action: 'seek-right' }, { name: 'Point left', action: 'seek-left' }];
    const csv = (await readFile(new URL('../../hand_tracker/model/keypoint_classifier/keypoint.csv', import.meta.url), 'utf8')).trim().split(/\r?\n/).map(row => row.split(',').map(Number));
    const samples = labels.flatMap((_, label) => csv.filter(row => row[0] === label).slice(0, 16).map((row, i) => ({ label, features: row.slice(1), session: `pose-${label}-batch-${Math.floor(i / 5)}` })));
    const dataset = { schemaVersion: 1, labels, samples };
    const upload = { name: 'hands-on-dataset.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(dataset)) };
    await training.locator('#import').setInputFiles(upload);
    await expect(training.locator('#saved')).toContainText('4735 landmark samples');
    await training.locator('#import').setInputFiles(upload);
    await expect(training.locator('#saved')).toContainText('4735 landmark samples');
    await training.locator('summary').first().click();
    await training.locator('#name').fill('Custom pose');
    await training.locator('#action').selectOption('seek-right');
    await training.locator('#add').click();
    await expect(training.locator('#label option')).toHaveCount(7);
    await training.reload();
    await expect(training.locator('#label option')).toHaveCount(7);
    const downloadPromise = training.waitForEvent('download');
    await training.locator('#export').click();
    const download = await downloadPromise;
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
    expect(exported.labels[6]).toEqual({ name: 'Custom pose', action: 'seek-right' });
    expect(exported.samples).toHaveLength(4735);
    await training.locator('#label').selectOption('6');
    await training.locator('#remove').click();
    await expect(training.locator('#label option')).toHaveCount(6);
    await training.locator('#label').selectOption('0');
    await training.locator('#remove').click();
    await expect(training.locator('#error')).toContainText('Only empty poses');
    await training.setViewportSize({ width: 390, height: 844 });
    expect(await training.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await training.locator('#connect').click();
    await expect(training.locator('#connect')).toHaveText('Disconnect camera', { timeout: 30000 });
    await popup.locator('.options summary').click();
    await popup.locator('#training-mode').uncheck();
    await expect(popup.locator('#training')).toBeHidden();
    await expect(training).toHaveURL(`chrome-extension://${id}/popup.html`);
    await training.goto(`chrome-extension://${id}/training.html`);
    await expect(training).toHaveURL(`chrome-extension://${id}/popup.html`);
    await popup.locator('#training-mode').check();
    await training.goto(`chrome-extension://${id}/training.html`);
    await expect(training.locator('#saved')).toContainText('4735 landmark samples');
    expect(errors).toEqual([]);
  } finally { await context.close(); await rm(profile, { recursive: true, force: true }); }
});

test('user-only release omits training files and runs packaged tracking offline', async () => {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  execFileSync(process.execPath, ['scripts/build-release.mjs'], { cwd: root });
  const extension = path.join(root, 'dist/hands-on');
  for (const file of ['training.html', 'training.js', 'training.css', 'models/training-seed.json', 'models/training-report.json', 'lib/training-data.js', 'setup.html', 'setup.js']) {
    await expect(access(path.join(extension, file))).rejects.toThrow();
  }
  expect(JSON.parse(await readFile(path.join(extension, 'manifest.json'), 'utf8')).name).toBe('Hands On Video Controls');
  expect(await readFile(path.join(extension, 'popup.html'), 'utf8')).not.toMatch(/training|diagnostics|FPS|ms \/ frame/);
  const profile = await mkdtemp(path.join(tmpdir(), 'hands-on-release-'));
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium', headless: true,
    ignoreDefaultArgs: ['--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding'],
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  try {
    let [background] = context.serviceWorkers();
    background ??= await context.waitForEvent('serviceworker');
    const id = new URL(background.url()).host;
    const popup = await context.newPage();
    const errors = [];
    context.on('weberror', event => errors.push(event.error().message));
    await context.setOffline(true);
    await popup.goto(`chrome-extension://${id}/popup.html`);
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    await expect(popup.locator('#training')).toHaveCount(0);
    await expect(popup.locator('#diagnostics')).toHaveCount(0);
    await expect(popup.locator('#training-mode')).toHaveCount(0);
    await expect(popup.locator('#try-skip')).toHaveCount(0);
    await expect(popup.locator('#setup')).toHaveCount(0);
    expect(await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'test-skip' }))).toMatchObject({ error: 'Unknown action' });
    expect(await popup.locator('body').evaluate(body => body.getBoundingClientRect().height)).toBeLessThanOrEqual(600);
    await popup.locator('body').screenshot({ path: '/private/tmp/hands-on-release.png' });
    await popup.evaluate(() => chrome.runtime.sendMessage({ target: 'background', action: 'start' }));
    await expect(popup.locator('#state')).toHaveText('You’re in control', { timeout: 30000 });
    expect(await popup.locator('body').evaluate(body => body.getBoundingClientRect().height)).toBeLessThanOrEqual(600);
    await expect.poll(() => popup.evaluate(async () => (await chrome.runtime.sendMessage({ target: 'background', action: 'get-status' })).inferenceMs)).toBeGreaterThan(0);
    await popup.locator('body').screenshot({ path: '/private/tmp/hands-on-release-running.png' });
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    expect(await background.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }))).toHaveLength(0);
    expect(errors).toEqual([]);
  } finally { await context.close(); await rm(profile, { recursive: true, force: true }); }
});
