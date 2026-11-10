const video = document.getElementById('camera');
let status = { state: 'stopped' };
let stream;
let worker;
let timer;
let generation = 0;
let lastVideoTime = -1;
let lastStatusAt = 0;
let watchdog;
let averageMs = 35;
let lastFrameAt = 0;

function publish(patch) {
  status = { ...status, ...patch };
  void chrome.runtime.sendMessage({ target: 'background', action: 'status', status }).catch(() => {});
}
function cleanup() {
  generation++;
  clearTimeout(timer);
  clearTimeout(watchdog);
  worker?.terminate();
  worker = undefined;
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.srcObject = null;
}
function fail(error) {
  cleanup();
  const cameraError = ['NotAllowedError', 'PermissionDeniedError', 'SecurityError'].includes(error.name);
  publish({ state: 'error', needsPermission: cameraError, error: cameraError
    ? 'Allow Hands On in Chrome’s camera settings and your system privacy settings, then start tracking again.'
    : error.name === 'NotReadableError' ? 'The camera is busy. Close other camera apps and try again.'
    : error.name === 'NotFoundError' ? 'No camera found. Connect a camera and try again.'
    : error.message || 'Tracking stopped unexpectedly.' });
}

async function start() {
  if (['starting', 'running'].includes(status.state)) return status;
  cleanup();
  const current = generation;
  status = {};
  publish({ state: 'starting', hand: 'Waiting for camera', fps: 0 });
  watchdog = setTimeout(() => fail(new Error('Camera or model did not start. Check camera permission and reload the extension.')), 30000);
  try {
    const camera = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 15 } },
    });
    if (current !== generation) { camera.getTracks().forEach(track => track.stop()); return status; }
    stream = camera;
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (current === generation) fail(new Error('Camera access ended. Start tracking to reconnect.'));
    });
    video.srcObject = stream;
    await video.play();
    if (current !== generation) return status;
    worker = new Worker('tracking-worker.js');
    worker.onerror = event => {
      event.preventDefault();
      if (current === generation) fail(new Error(event.message || 'Tracking worker could not start.'));
    };
    worker.onmessage = ({ data }) => {
      if (current !== generation) return;
      if (data.type === 'error') { fail(new Error(data.error)); return; }
      if (data.type === 'ready') {
        clearTimeout(watchdog);
        lastVideoTime = -1;
        lastFrameAt = 0;
        averageMs = 35;
        publish({ state: 'running', hand: 'No hand detected' });
        void frame(current);
      } else if (data.type === 'result') {
        clearTimeout(watchdog);
        averageMs = averageMs * 0.8 + data.inferenceMs * 0.2;
        if (data.observation) Object.assign(status, data.observation);
        status.inferenceMs = Math.round(averageMs);
        if (data.event) {
          status.lastGesture = data.event.type === 'control' ? data.event.value === 'skip_ad' ? 'Skip ad' : 'Play / pause' : `Seek ${data.event.value === 'right' ? '+10' : '−10'} seconds`;
          void chrome.runtime.sendMessage({ target: 'background', action: 'gesture', payload: data.event });
        }
        if (performance.now() - lastStatusAt >= 1000 || data.event) {
          lastStatusAt = performance.now();
          publish({});
        }
        const interval = Math.max(1000 / 15, averageMs * 2);
        timer = setTimeout(() => { void frame(current); }, Math.max(0, Math.ceil(interval - (performance.now() - lastFrameAt))));
      }
    };
    worker.postMessage({ type: 'init' });
  } catch (error) { if (current === generation) fail(error); }
  return status;
}

async function frame(current) {
  if (current !== generation || status.state !== 'running') return;
  if (video.readyState < 2 || video.currentTime === lastVideoTime) {
    if (lastFrameAt && performance.now() - lastFrameAt > 10000) {
      fail(new Error('Camera stopped delivering frames. Start tracking to reconnect.'));
      return;
    }
    timer = setTimeout(() => { void frame(current); }, 70);
    return;
  }
  lastVideoTime = video.currentTime;
  try {
    const bitmap = await createImageBitmap(video);
    if (current !== generation) { bitmap.close(); return; }
    const now = performance.now();
    status.fps = lastFrameAt ? Math.round(1000 / (now - lastFrameAt)) : 0;
    lastFrameAt = now;
    watchdog = setTimeout(() => fail(new Error('Hand tracking stalled. Stop and start tracking to reconnect.')), 10000);
    worker.postMessage({ type: 'frame', bitmap, time: now }, [bitmap]);
  } catch (error) { if (current === generation) fail(error); }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id || message?.target !== 'tracker') return;
  if (message.action === 'get-status') respond(status);
  else if (message.action === 'start') { void start(); respond(status); }
  else if (message.action === 'stop') { cleanup(); publish({ state: 'stopped' }); respond(status); }
  else if (message.action === 'feedback') {
    status.feedback = message.feedback?.ok ? message.feedback.message || 'Video controlled' : message.feedback?.error || 'No video available';
    status.feedbackAction = message.feedback?.action;
    status.feedbackAt = Date.now();
    status.skipResult = message.feedback?.action === 'skip_ad' ? message.feedback.code : undefined;
    respond({ ok: true });
  }
});
