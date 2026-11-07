const $ = id => document.getElementById(id);
const send = action => chrome.runtime.sendMessage({ target: 'background', action });
let currentState = 'stopped';
let pending = false;

function render(status) {
  currentState = status.state;
  const active = ['starting', 'running'].includes(currentState);
  $('state').textContent = { stopped: 'Camera is off', starting: 'Starting tracking…', running: 'Tracking is on', error: 'Tracking needs attention' }[currentState] ?? 'Camera is off';
  $('dot').className = `dot${active ? ' active' : currentState === 'error' ? ' failed' : ''}`;
  $('toggle').textContent = active ? 'Stop tracking' : 'Start tracking';
  $('toggle').classList.toggle('stop', active);
  $('toggle').disabled = pending;
  $('error').hidden = !status.error;
  $('error').textContent = status.error ?? '';
  $('metrics').hidden = currentState !== 'running';
  $('fps').textContent = status.fps ?? '—';
  $('latency').textContent = status.inferenceMs ?? '—';
  $('detail').textContent = currentState === 'running'
    ? `${status.hand ?? 'Looking for a hand'}${status.action?.startsWith('seek-') ? ' · Hold to seek' : ''}${status.feedback ? ` · ${status.feedback}` : ''}`
    : currentState === 'starting' ? 'Loading your camera and local models.' : 'Your camera stays on your device.';
}
async function refresh() {
  try { render(await send('get-status')); }
  catch (error) { render({ state: 'error', error: error.message }); }
}
const openSetup = () => chrome.tabs.create({ url: chrome.runtime.getURL('setup.html') });
$('setup').addEventListener('click', () => { void openSetup(); });
$('toggle').addEventListener('click', async () => {
  pending = true;
  $('toggle').disabled = true;
  try {
    if (['running', 'starting'].includes(currentState)) render(await send('stop'));
    else {
      const permission = await navigator.permissions.query({ name: 'camera' });
      if (permission.state !== 'granted') await openSetup();
      else render(await send('start'));
    }
  } catch (error) { render({ state: 'error', error: error.message }); }
  finally { pending = false; $('toggle').disabled = false; }
});
void refresh();
void fetch('models/pose-classifier.json').then(r => r.json()).then(model => {
  if (!['seek-left', 'seek-right'].every(action => model.labels.some(label => label.action === action))) {
    $('training-note').hidden = false;
    $('training-note').textContent = 'Left/right poses need training. Record fresh examples in Training studio to enable seeking.';
  }
}).catch(() => {});
setInterval(() => { if (!pending) void refresh(); }, 1000);

$('training').addEventListener('click', () => { void chrome.tabs.create({ url: chrome.runtime.getURL('training.html') }); });
