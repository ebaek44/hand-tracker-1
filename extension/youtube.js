// Button matching and visibility checks adapted from xhy0405/youtube-auto-skip.
// Copyright (c) 2026 xhhhhhhhy; MIT license: licenses/youtube-auto-skip.txt.
(() => {
  if (globalThis.__handsOnYouTube?.version === 5) return;
  const hosts = new Set(['youtube.com', 'www.youtube.com', 'm.youtube.com', 'www.youtube-nocookie.com', 'youtube-nocookie.com']);
  const selectors = [
    '.ytp-skip-ad-button', '.ytp-skip-ad-button-modern',
    '.ytp-ad-skip-button', '.ytp-ad-skip-button-modern',
    '.ytp-ad-skip-button-slot button', '.ytp-ad-skip-button-container button',
  ];
  let selected = null;
  let selectionToken = 0;
  let observed = null;
  const adActive = player => player.matches('.ad-showing, .ad-interrupting');
  const unavailable = () => ({ ok: false, code: 'unavailable', error: 'Skip ad is not available yet.' });

  function available(element) {
    if (!(element instanceof HTMLElement) || !element.isConnected || element.matches(':disabled') ||
        element.getAttribute('aria-disabled') === 'true' || element.closest('[inert]')) return false;
    if (element.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) === false) return false;
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 && rect.left < innerWidth && rect.top < innerHeight &&
      style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) > 0 && style.pointerEvents !== 'none';
  }

  function findButton(player) {
    for (const selector of selectors) {
      for (const match of player.querySelectorAll(selector)) {
        // Some layouts put the class on a child or a non-button clickable control.
        const button = match.closest('button, [role="button"]') || match;
        if (player.contains(button) && available(button)) return { button, selector };
      }
    }
    return null;
  }

  function inspect() {
    selected = null;
    if (!hosts.has(location.hostname)) return { supported: false, available: false };
    let bestScore = 0;
    for (const player of document.querySelectorAll('.html5-video-player, #movie_player')) {
      const found = findButton(player);
      if (!found) continue;
      const rect = found.button.getBoundingClientRect();
      const score = Math.max(1, rect.width * rect.height) + (adActive(player) ? 1e8 : 0);
      if (score > bestScore) { bestScore = score; selected = { player, found, url: location.href, source: player.querySelector('video')?.currentSrc || '' }; }
    }
    return { supported: true, available: Boolean(selected), score: bestScore, token: ++selectionToken };
  }

  function prepare(token) {
    if (token !== selectionToken || !selected || selected.url !== location.href || !selected.player.isConnected ||
        !available(selected.found.button) || document.visibilityState === 'hidden') return unavailable();
    const video = selected.player.querySelector('video');
    if ((video?.currentSrc || '') !== selected.source) return unavailable();
    const rect = selected.found.button.getBoundingClientRect();
    const x = (Math.max(0, rect.left) + Math.min(innerWidth, rect.right)) / 2;
    const y = (Math.max(0, rect.top) + Math.min(innerHeight, rect.bottom)) / 2;
    const hit = document.elementFromPoint(x, y);
    if (!hit || !(hit === selected.found.button || selected.found.button.contains(hit))) {
      return { ok: false, code: 'covered', error: 'The Skip ad button is covered. Try again when it is clear.' };
    }
    return { ok: true, x, y, selector: selected.found.selector };
  }

  function beginVerification(token) {
    const control = prepare(token);
    if (!control.ok) return control;
    const { player, found, url } = selected;
    const video = player.querySelector('video');
    const source = video?.currentSrc || '';
    const wasAd = adActive(player);
    const result = new Promise(resolve => {
      let done = false;
      let timer;
      const finish = value => {
        if (done) return;
        done = true;
        observer.disconnect(); clearTimeout(timer);
        video?.removeEventListener('loadedmetadata', check);
        document.removeEventListener('visibilitychange', check);
        window.removeEventListener('pagehide', cancel);
        resolve({ ...value, selector: found.selector });
      };
      const cancel = () => finish({ ok: false, code: 'cancelled', error: 'Skip cancelled because the video changed.' });
      const check = () => {
        if (location.href !== url || document.visibilityState === 'hidden' || !player.isConnected) { cancel(); return; }
        if (wasAd && !adActive(player) && !findButton(player)) finish({ ok: true, code: 'confirmed', message: 'Ad skipped' });
        else if (source && video?.isConnected && video.currentSrc && video.currentSrc !== source) finish({ ok: true, code: 'confirmed', message: 'Ad advanced' });
      };
      const observer = new MutationObserver(check);
      observer.observe(player, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'src'] });
      video?.addEventListener('loadedmetadata', check);
      document.addEventListener('visibilitychange', check);
      window.addEventListener('pagehide', cancel, { once: true });
      timer = setTimeout(() => finish({ ok: false, code: 'unconfirmed', error: 'The browser click was sent, but YouTube’s response was not confirmed.' }), 1800);
    });
    observed = { token, result };
    return { ok: true };
  }
  const waitForResult = token => observed?.token === token ? observed.result : Promise.resolve(unavailable());
  globalThis.__handsOnYouTube = { version: 5, inspect, prepare, beginVerification, waitForResult };
})();
