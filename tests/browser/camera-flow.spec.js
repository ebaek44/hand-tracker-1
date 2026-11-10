import { test, expect, chromium } from '@playwright/test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('Start tracking handles first permission, retry, and stopping without a setup page', async () => {
  const profile = await mkdtemp(path.join(tmpdir(), 'hands-on-permission-'));
  const extension = fileURLToPath(new URL('../../extension/', import.meta.url));
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium', headless: true,
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  try {
    let [background] = context.serviceWorkers();
    background ??= await context.waitForEvent('serviceworker');
    const origin = `chrome-extension://${new URL(background.url()).host}`;
    const popup = await context.newPage();
    await popup.goto(`${origin}/popup.html`);
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    const cdp = await context.newCDPSession(popup);
    await cdp.send('Browser.setPermission', { origin, permission: { name: 'camera' }, setting: 'prompt' });
    const opened = [];
    context.on('page', page => opened.push(page));
    const permissionPage = context.waitForEvent('page');
    await popup.locator('#toggle').click();
    const request = await permissionPage;
    await expect.poll(() => request.isClosed() ? true : request.url().endsWith('popup.html?camera=1')).toBe(true);
    await expect(popup.locator('#state')).toHaveText('You’re in control', { timeout: 30000 });
    await expect.poll(() => request.isClosed()).toBe(true);
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    expect(await background.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }))).toHaveLength(0);

    // Once granted, no extra window; a denied camera is explained inline.
    await cdp.send('Browser.setPermission', { origin, permission: { name: 'camera' }, setting: 'granted' });
    const count = opened.length;
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('You’re in control', { timeout: 30000 });
    expect(opened).toHaveLength(count);
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('Camera is off');
    await cdp.send('Browser.setPermission', { origin, permission: { name: 'camera' }, setting: 'denied' });
    await popup.locator('#toggle').click();
    await expect(popup.locator('#error')).toContainText('Camera access is blocked');
    await popup.waitForTimeout(1500);
    await expect(popup.locator('#error')).toBeVisible();
    expect(opened).toHaveLength(count);
    expect(await background.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }))).toHaveLength(0);
    await cdp.send('Browser.setPermission', { origin, permission: { name: 'camera' }, setting: 'granted' });
    await popup.locator('#toggle').click();
    await expect(popup.locator('#state')).toHaveText('You’re in control', { timeout: 30000 });
    await popup.locator('#toggle').click();
    await cdp.detach();
  } finally { await context.close(); await rm(profile, { recursive: true, force: true }); }
});
