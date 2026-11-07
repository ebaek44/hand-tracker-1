import { sampleKey, withRetainedExamples } from './lib/training-data.js';
const $ = id => document.getElementById(id);
const defaultLabels = [
  { name: 'Open palm', action: 'toggle' }, { name: 'Closed hand', action: 'none' },
  { name: 'Point up', action: 'none' }, { name: 'Confirm / neutral', action: 'none' },
  { name: 'Point right', action: 'seek-right' }, { name: 'Point left', action: 'seek-left' },
];
let dataset = { schemaVersion: 1, labels: defaultLabels, samples: [] };
let camera, worker, timer, watchdog;
let connected = false;
let generation = 0;
let recording;
let lastFrame = -1;
let lastSent = 0;
const dbReady = new Promise((resolve, reject) => {
  const request = indexedDB.open('hands-on-training', 1);
  request.onupgradeneeded = () => request.result.createObjectStore('datasets');
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});
async function persist() {
  $('saved').textContent = 'Saving locally…';
  const db = await dbReady;
  await new Promise((resolve, reject) => {
    const transaction = db.transaction('datasets', 'readwrite');
    transaction.objectStore('datasets').put(dataset, 'current');
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
  });
  $('saved').textContent = `Saved locally · ${dataset.samples.length} landmark samples. Export a backup before removing this extension.`;
}
function error(message) { $('error').textContent = message; $('error').hidden = false; }
function render() {
  const value = $('label').value;
  $('label').replaceChildren(...dataset.labels.map((label, index) => new Option(label.name, index)));
  if (value && dataset.labels[Number(value)]) $('label').value = value;
  $('counts').replaceChildren(...dataset.labels.map((label, index) => {
    const row = document.createElement('div');
    const name = document.createElement('span'); name.textContent = label.name;
    const samples = dataset.samples.filter(sample => sample.label === index);
    const carried = samples.filter(s => s.session === 'legacy').length;
    const batches = new Set(samples.filter(s => s.session !== 'legacy').map(s => s.session)).size;
    const count = document.createElement('span'); count.textContent = `${carried} carried + ${samples.length - carried} new · ${batches} batches`;
    row.append(name, count); return row;
  }));
}
function stop(savePartial = true) {
  const partial = Boolean(recording);
  generation++;
  connected = false;
  recording = undefined;
  clearTimeout(timer); clearTimeout(watchdog);
  worker?.terminate(); worker = undefined;
  camera?.getTracks().forEach(track => track.stop()); camera = undefined;
  $('preview').srcObject = null;
  $('overlay').getContext('2d').clearRect(0, 0, 640, 480);
  $('connect').textContent = 'Connect camera';
  $('connect').disabled = false;
  $('record').disabled = true;
  $('record').textContent = 'Record a 3-second batch';
  $('label').disabled = false;
  $('camera-state').textContent = 'Camera is off';
  if (partial && savePartial) void persist().catch(e => error(e.message));
}
async function frame(current) {
  if (current !== generation || !worker) return;
  const video = $('preview');
  if (video.readyState < 2 || video.currentTime === lastFrame) {
    if (lastSent && performance.now() - lastSent > 10000) { stop(); error('Camera stopped delivering frames. Reconnect it.'); return; }
    timer = setTimeout(() => { void frame(current); }, 100); return;
  }
  lastFrame = video.currentTime;
  try {
    const bitmap = await createImageBitmap(video);
    if (current !== generation) { bitmap.close(); return; }
    lastSent = performance.now();
    watchdog = setTimeout(() => { stop(); error('Tracking stalled. Reconnect the camera.'); }, 10000);
    worker.postMessage({ type: 'frame', bitmap, time: lastSent }, [bitmap]);
  } catch (e) { stop(); error(e.message); }
}
$('connect').addEventListener('click', async () => {
  if (connected) { stop(); return; }
  $('connect').disabled = true;
  $('error').hidden = true;
  const current = ++generation;
  try {
    await chrome.runtime.sendMessage({ target: 'background', action: 'stop' });
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 15 } } });
    if (current !== generation) { stream.getTracks().forEach(t => t.stop()); return; }
    camera = stream;
    camera.getVideoTracks()[0].addEventListener('ended', () => { if (current === generation) { stop(); error('Camera disconnected.'); } });
    $('preview').srcObject = camera;
    await $('preview').play();
    lastFrame = -1;
    lastSent = 0;
    worker = new Worker('tracking-worker.js');
    watchdog = setTimeout(() => { stop(); error('Model did not start. Reload the extension.'); }, 30000);
    worker.onerror = e => { e.preventDefault(); stop(); error(e.message); };
    worker.onmessage = ({ data }) => {
      if (current !== generation) return;
      clearTimeout(watchdog);
      if (data.type === 'error') { stop(); error(data.error); return; }
      if (data.type === 'ready') {
        connected = true;
        $('connect').disabled = false;
        $('connect').textContent = 'Disconnect camera';
        $('record').disabled = false;
        $('camera-state').textContent = 'Camera connected · not recording';
        void frame(current); return;
      }
      const ctx = $('overlay').getContext('2d'); ctx.clearRect(0, 0, 640, 480);
      for (const [x, y] of data.landmarks ?? []) { ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = '#adfbb9'; ctx.fill(); }
      if (data.observation) $('prediction').textContent = `${data.observation.hand} · ${Math.round(data.observation.confidence * 100)}%`;
      if (recording) {
        if (data.features && dataset.samples.length < 10000) dataset.samples.push({ label: recording.label, features: data.features, session: recording.session });
        const remaining = recording.end - performance.now();
        $('record').textContent = `Recording… ${Math.max(0, remaining / 1000).toFixed(1)}s`;
        if (remaining <= 0 || dataset.samples.length >= 10000) {
          const added = dataset.samples.length - recording.start;
          recording = undefined;
          $('record').disabled = false;
          $('record').textContent = 'Record a 3-second batch';
          $('label').disabled = false;
          $('camera-state').textContent = 'Camera connected · not recording';
          $('guidance').textContent = `${added} samples collected. Change hand position and record another batch.`;
          render(); void persist().catch(e => error(e.message));
        }
      }
      timer = setTimeout(() => { void frame(current); }, Math.max(0, 100 - (performance.now() - lastSent)));
    };
    worker.postMessage({ type: 'init', collect: true });
  } catch (e) { stop(); error(e.message); }
});
$('record').addEventListener('click', () => {
  if (dataset.samples.length >= 10000) { error('Dataset limit reached. Export your dataset before collecting more.'); return; }
  recording = { label: Number($('label').value), end: performance.now() + 3000, session: crypto.randomUUID(), start: dataset.samples.length };
  $('record').disabled = true; $('label').disabled = true;
  $('camera-state').textContent = 'Recording landmark samples';
});
$('add').addEventListener('click', async () => {
  const name = $('name').value.trim();
  if (!name || dataset.labels.length >= 20) { error('Enter a pose name. Up to 20 poses are supported.'); return; }
  dataset.labels.push({ name, action: $('action').value });
  try { await persist(); } catch (e) { error(e.message); }
  render();
  $('label').value = String(dataset.labels.length - 1); $('name').value = '';
});
$('remove').addEventListener('click', async () => {
  const index = Number($('label').value);
  if (recording || dataset.labels.length <= 1 || dataset.samples.some(sample => sample.label === index)) {
    error('Only empty poses can be removed. Keep at least one pose.'); return;
  }
  dataset.labels.splice(index, 1);
  for (const sample of dataset.samples) if (sample.label > index) sample.label--;
  try { await persist(); } catch (e) { error(e.message); }
  render();
});
$('export').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(dataset)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'hands-on-dataset.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('import').addEventListener('change', async () => {
  try {
    const file = $('import').files[0];
    if (!file || file.size > 25e6) throw new Error('Choose a dataset smaller than 25 MB.');
    const data = JSON.parse(await file.text());
    if (data.schemaVersion !== 1 || !Array.isArray(data.labels) || !Array.isArray(data.samples) ||
        data.labels.length < 1 || data.labels.length > 20 || data.samples.length > 10000 ||
        data.labels.some(l => typeof l.name !== 'string' || l.name.length > 60 || !['none', 'toggle', 'seek-left', 'seek-right'].includes(l.action)) ||
        data.samples.some(s => !Number.isInteger(s.label) || s.label < 0 || s.label >= data.labels.length || typeof s.session !== 'string' ||
          !Array.isArray(s.features) || s.features.length !== 42 || s.features.some(n => typeof n !== 'number' || !Number.isFinite(n) || Math.abs(n) > 1.001))) {
      throw new Error('Invalid Hands On dataset.');
    }
    if (recording) throw new Error('Finish recording before importing.');
    if (JSON.stringify(data.labels) !== JSON.stringify(dataset.labels)) {
      if (dataset.samples.length) throw new Error('Pose lists differ. Import into a fresh studio dataset to avoid changing existing labels.');
      dataset.labels = data.labels;
    }
    const existing = new Set(dataset.samples.map(sampleKey));
    for (const sample of data.samples) {
      const key = sampleKey(sample);
      if (!existing.has(key) && dataset.samples.length < 10000) { dataset.samples.push(sample); existing.add(key); }
    }
    await persist(); render();
  } catch (e) { error(e.message); }
});
try {
  const db = await dbReady;
  const stored = await new Promise((resolve, reject) => {
    const request = db.transaction('datasets').objectStore('datasets').get('current');
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
  if (stored) dataset = stored;
} catch (e) { error(`Local saving unavailable: ${e.message}. Export before closing.`); }
try {
  const seed = await (await fetch('models/training-seed.json')).json();
  dataset = withRetainedExamples(dataset, seed);
  await persist();
} catch (e) { error(`Could not load/save retained examples: ${e.message}`); }
render();
window.addEventListener('pagehide', () => stop(false));
