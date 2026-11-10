import { test, expect, chromium } from '@playwright/test';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('selector guards and current YouTube synthetic-input rejection are reproducible', async () => {
  const extension = fileURLToPath(new URL('../../extension/', import.meta.url));
  const profile = await mkdtemp(path.join(tmpdir(), 'hands-on-skip-'));
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium', headless: true,
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  try {
    let [background] = context.serviceWorkers();
    background ??= await context.waitForEvent('serviceworker');
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${new URL(background.url()).host}/popup.html`);
    const clip = await readFile(new URL('../fixtures/video.webm', import.meta.url));
    const page = await context.newPage();
    let markup = '';
    await context.route('https://www.youtube.com/**', route => route.fulfill(route.request().url().endsWith('/clip.webm')
      ? { contentType: 'video/webm', body: clip } : { contentType: 'text/html', body: markup }));
    const fixture = (control, behavior = 'accept') => `
      <div id="movie_player" class="html5-video-player ad-interrupting">
        <video src="/clip.webm" preload="auto" style="width:640px;height:360px"></video>
        ${control}
      </div>
      <script>
        window.events = []; window.clicks = 0;
        const button = document.getElementById('skip');
        for (const name of ['pointerdown','mousedown','pointerup','mouseup','click']) {
          button.addEventListener(name, event => {
            window.events.push(name);
            if (name !== 'click') return;
            window.clicks++;
            window.lastTrusted = event.isTrusted;
            if (${JSON.stringify(behavior)} === 'trusted-only') {
              if (!event.isTrusted) return;
              document.getElementById('movie_player').classList.remove('ad-interrupting');
              button.remove(); return;
            }
            if (${JSON.stringify(behavior)} === 'remove-only') { button.remove(); return; }
            if (${JSON.stringify(behavior)} === 'accept' && window.events.join(',') === 'pointerdown,mousedown,pointerup,mouseup,click') {
              document.getElementById('movie_player').classList.remove('ad-interrupting');
              button.remove();
            }
          });
        }
      </script>`;
    async function load(html, hasVideo = true) {
      markup = html;
      await page.goto(`https://www.youtube.com/watch?v=fixture-${Date.now()}`);
      await page.bringToFront();
      if (hasVideo) await expect.poll(() => page.locator('video').first().evaluate(video => video.readyState)).toBeGreaterThanOrEqual(2);
    }
    const skip = () => background.evaluate(async () => {
      const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['youtube.js'] });
      const [execution] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: async () => {
        const match = globalThis.__handsOnYouTube.inspect();
        if (!match.available) return { ok: false, code: 'unavailable' };
        const control = globalThis.__handsOnYouTube.prepare(match.token);
        if (!control.ok) return control;
        globalThis.__handsOnYouTube.beginVerification(match.token);
        const button = document.elementFromPoint(control.x, control.y).closest('button, [role="button"]');
        // Reference implementation for the synthetic approach, confined to this
        // test: production now sends trusted input through the optional mode.
        for (const name of ['pointerdown','mousedown','pointerup','mouseup']) {
          const EventClass = name.startsWith('pointer') ? PointerEvent : MouseEvent;
          button.dispatchEvent(new EventClass(name, { bubbles: true, composed: true, button: 0, view: window }));
        }
        button.click();
        return globalThis.__handsOnYouTube.waitForResult(match.token);
      } });
      return execution.result;
    });

    // This layout and handler reject the old implementation: non-button modern
    // control, ad-interrupting state, and required pointer/mouse sequence.
    await load(fixture('<div id="skip" class="ytp-skip-ad-button-modern" role="button" tabindex="0" style="width:100px;height:35px">Skip ad</div>'));
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed', message: 'Ad skipped' });
    expect(await page.evaluate(() => window.events)).toEqual(['pointerdown','mousedown','pointerup','mouseup','click']);
    expect(await page.evaluate(() => window.clicks)).toBe(1);
    expect(await page.locator('video').evaluate(video => video.currentTime)).toBe(0);

    // Button discovery does not depend on a ready video or on picking the right
    // video among side players. The user's gesture directly starts a button scan.
    await load(fixture('<button id="skip" class="ytp-skip-ad-button">Skip ad</button>').replace(/<video[^>]*><\/video>/, ''), false);
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed' });
    await load('<video id="decoy" muted src="/clip.webm" style="position:absolute;right:0;top:0;width:120px;height:80px"></video>' + fixture('<button id="skip" class="ytp-skip-ad-button">Skip ad</button>'));
    await page.locator('#decoy').evaluate(video => video.play());
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed' });

    await load(fixture('<div class="ytp-ad-skip-button-slot"><button id="skip">Skip ad</button></div>'));
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed' });

    // A gesture before availability is rejected immediately; enabling the button
    // later must do nothing until another explicit gesture/request arrives.
    await load(fixture('<button id="skip" class="ytp-skip-ad-button" disabled>Skip ad</button>'));
    expect(await skip()).toMatchObject({ ok: false, code: 'unavailable' });
    await page.locator('#skip').evaluate(button => { button.disabled = false; });
    await page.waitForTimeout(1400);
    expect(await page.evaluate(() => window.clicks)).toBe(0);
    expect(await skip()).toMatchObject({ ok: true, code: 'confirmed' });

    // Click delivery alone and button removal alone are not proof of ad skipping.
    for (const behavior of ['ignore', 'remove-only']) {
      await load(fixture('<button id="skip" class="ytp-ad-skip-button-modern">Skip ad</button>', behavior));
      expect(await skip()).toMatchObject({ ok: false, code: 'unconfirmed' });
      expect(await page.evaluate(() => window.clicks)).toBe(1);
      await expect(page.locator('#movie_player')).toHaveClass(/ad-interrupting/);
    }

    await load(fixture('<button id="skip" class="ytp-skip-ad-button">Skip ad</button><div style="position:fixed;inset:0;background:white;z-index:10"></div>'));
    expect(await skip()).toMatchObject({ ok: false, code: 'covered' });
    expect(await page.evaluate(() => window.clicks)).toBe(0);

    // Observation after an ignored click is verification only; it never clicks a
    // replacement control that appears during that response window.
    await load(fixture('<button id="skip" class="ytp-skip-ad-button">Skip ad</button>', 'ignore'));
    const pending = skip();
    await expect.poll(() => page.evaluate(() => window.clicks)).toBe(1);
    await page.locator('#skip').evaluate(button => {
      const replacement = button.cloneNode(true);
      replacement.onclick = () => { window.clicks++; };
      button.replaceWith(replacement);
    });
    expect(await pending).toMatchObject({ ok: false, code: 'unconfirmed' });
    expect(await page.evaluate(() => window.clicks)).toBe(1);

    // Current YouTube's public player has this trust guard. Synthetic click()
    // reaches the exact button but is rejected; browser-level input is accepted.
    await load(fixture('<button id="skip" class="ytp-skip-ad-button">Skip ad</button>', 'trusted-only'));
    expect(await skip()).toMatchObject({ ok: false, code: 'unconfirmed' });
    expect(await page.evaluate(() => window.lastTrusted)).toBe(false);
    await page.locator('#skip').click();
    expect(await page.evaluate(() => window.lastTrusted)).toBe(true);
    await expect(page.locator('#movie_player')).not.toHaveClass(/ad-interrupting/);
  } finally { await context.close(); await rm(profile, { recursive: true, force: true }); }
});
