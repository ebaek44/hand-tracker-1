(() => {
  if (globalThis.__handsOnController) return;
  let selected;
  let selectionToken = 0;
  let lastAction = -Infinity;
  let busy = false;

  function visibleArea(video) {
    if (!video.isConnected || video.readyState < 1) return 0;
    if (video.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) === false) return 0;
    const style = getComputedStyle(video);
    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return 0;
    const rect = video.getBoundingClientRect();
    return Math.max(0, Math.min(innerWidth, rect.right) - Math.max(0, rect.left)) *
      Math.max(0, Math.min(innerHeight, rect.bottom) - Math.max(0, rect.top));
  }
  function collect(root, videos = []) {
    videos.push(...root.querySelectorAll('video'));
    // Walk open shadow roots once per command, not continuously per video frame.
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    while (walker.nextNode()) if (walker.currentNode.shadowRoot) collect(walker.currentNode.shadowRoot, videos);
    return videos;
  }
  function probe() {
    selected = null;
    let bestScore = 0;
    for (const video of collect(document)) {
      const area = visibleArea(video);
      if (!area) continue;
      const playing = !video.paused && !video.ended;
      const score = area + (playing ? 1e8 : 0) + (playing && !video.muted && video.volume > 0 ? 1e9 : 0);
      if (score > bestScore) { selected = video; bestScore = score; }
    }
    return selected ? { score: bestScore, token: ++selectionToken } : null;
  }
  async function act(message) {
    if (busy || performance.now() - lastAction < 200) return { ok: false, error: 'Gesture already handled' };
    const video = selected;
    if (message.token !== selectionToken || !video || !visibleArea(video)) return { ok: false, error: 'Video changed. Try again.' };
    const { type, value } = message.payload ?? {};
    if (!(type === 'control' && value === 'toggle_play_pause') && !(type === 'seek' && ['left', 'right'].includes(value))) {
      return { ok: false, error: 'Unknown gesture' };
    }
    busy = true;
    lastAction = performance.now();
    try {
      if (type === 'control') {
        if (video.paused) await video.play();
        else video.pause();
      } else {
        if (!video.seekable.length) throw new Error('This video is not seekable yet.');
        const target = video.currentTime + (value === 'left' ? -10 : 10);
        // Choose the seekable range nearest the requested position (also supports live DVR).
        let closest;
        let distance = Infinity;
        for (let i = 0; i < video.seekable.length; i++) {
          const clamped = Math.max(video.seekable.start(i), Math.min(video.seekable.end(i), target));
          if (Math.abs(target - clamped) < distance) { closest = clamped; distance = Math.abs(target - clamped); }
        }
        video.currentTime = closest;
      }
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error.name === 'NotAllowedError' ? 'Click play on the video once to enable playback.' : error.message };
    } finally { busy = false; }
  }
  globalThis.__handsOnController = { probe };
  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (sender.id !== chrome.runtime.id || message?.kind !== 'hands-on/gesture') return;
    void act(message).then(respond);
    return true;
  });
})();
