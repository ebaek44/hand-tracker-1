import { trainingEnabled } from './config.js';
import { trainingOptedIn } from './lib/training-settings.js';
const $ = id => document.getElementById(id);
const send = action => chrome.runtime.sendMessage({ target: 'background', action });
let currentState = 'stopped';
let pending = false;
let optionsPending = false;
let trainingOn = false;
let startupError = '';
let closing = false;
const permissionWindow = new URLSearchParams(location.search).has('camera');
$('app-version').textContent = chrome.runtime.getManifest().version;

function renderOptions(enabled) {
  if (optionsPending) return;
  $('ad-skipping').checked = enabled;
  $('ad-skipping').disabled = false;
  if ($('training-mode')) {
    $('training-mode').checked = trainingOn;
    $('training-mode').disabled = false;
    $('training').hidden = !trainingOn;
  }
  $('skip-hint').textContent = enabled ? 'When “Skip ad” appears' : 'Enable in Options';
}

function render(status) {
  if (startupError && status.state === 'stopped') status = { state: 'error', error: startupError };
  currentState = status.state;
  const active = ['starting', 'running'].includes(currentState);
  $('state').textContent = { stopped: 'Camera is off', starting: 'Getting ready…', running: 'You’re in control', error: 'Let’s reconnect' }[currentState] ?? 'Camera is off';
  $('dot').className = `dot${active ? ' active' : currentState === 'error' ? ' failed' : ''}`;
  $('toggle').textContent = active ? 'Stop tracking' : 'Start tracking';
  $('toggle').classList.toggle('stop', active);
  $('toggle').disabled = pending;
  $('error').hidden = !status.error;
  $('error').textContent = status.error ?? '';
  if ($('diagnostics')) $('diagnostics').hidden = !trainingOn || currentState !== 'running';
  if ($('fps')) $('fps').textContent = status.fps ?? '—';
  if ($('latency')) $('latency').textContent = status.inferenceMs ?? '—';
  const hints = { 'seek-left': 'Seeking back · lower your hand to stop', 'seek-right': 'Seeking forward · lower your hand to stop', toggle: 'Open palm · relax to use again', 'skip-ad': 'Point up · lower your hand to use again' };
  $('detail').textContent = currentState === 'running'
    ? status.action && hints[status.action] ? hints[status.action] : 'Show a gesture to control your video.'
    : currentState === 'starting' ? 'Connecting to your camera…' : 'Start when you’re ready to watch.';
  if (currentState === 'running' && status.feedbackAction === 'skip_ad' && Date.now() - status.feedbackAt < 6000) {
    $('detail').textContent = status.feedback;
  }
}
async function refresh() {
  try {
    const [status, settings, optedIn] = await Promise.all([send('get-status'), send('get-ad-settings'), trainingOptedIn()]);
    trainingOn = optedIn;
    render(status); renderOptions(settings.enabled);
    if (permissionWindow && status.state === 'running' && !closing) {
      closing = true;
      const window = await chrome.windows.getCurrent();
      if (window.type === 'popup') await chrome.windows.remove(window.id);
    }
  }
  catch { render({ state: 'error', error: 'Couldn’t connect. Close and reopen Hands On to try again.' }); }
}
$('ad-skipping').addEventListener('change', async () => {
  optionsPending = true;
  const enabled = $('ad-skipping').checked;
  $('ad-skipping').disabled = true;
  $('ad-option-status').hidden = false;
  try {
    const result = await chrome.runtime.sendMessage({ target: 'background', action: 'set-ad-skipping', enabled });
    if (!result.ok) throw new Error(result.error || 'Could not update the option.');
    $('ad-option-status').textContent = result.enabled ? 'Ad skipping is on. Point up when Skip ad appears.' : 'Ad skipping is off. Browser control is inactive.';
  } catch (error) { $('ad-option-status').textContent = error.message; }
  finally { optionsPending = false; $('ad-skipping').disabled = false; void refresh(); }
});
async function toggleTracking() {
  pending = true;
  startupError = '';
  $('toggle').disabled = true;
  try {
    if (['running', 'starting'].includes(currentState)) render(await send('stop'));
    else {
      const permission = await navigator.permissions.query({ name: 'camera' });
      if (permission.state === 'denied') throw new DOMException('Camera permission denied', 'NotAllowedError');
      if (permission.state !== 'granted' && !permissionWindow) {
        await chrome.windows.create({ url: chrome.runtime.getURL('popup.html?camera=1'), type: 'popup', width: 410, height: 660 });
        return;
      }
      if (permission.state !== 'granted') {
        $('detail').textContent = 'Allow the camera in Chrome’s permission prompt.';
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        stream.getTracks().forEach(track => track.stop());
      }
      render(await send('start'));
    }
  } catch (error) { startupError = error.name === 'NotAllowedError'
    ? 'Camera access is blocked. Allow Hands On in Chrome’s camera settings and your system privacy settings, then try again.'
    : error.name === 'NotFoundError' ? 'No camera found. Connect a camera and try again.'
    : 'Couldn’t start the camera. Close other camera apps and try again.';
    render({ state: 'error', error: startupError }); }
  finally { pending = false; $('toggle').disabled = false; }
}
$('toggle').addEventListener('click', () => { void toggleTracking(); });
if (trainingEnabled) {
  $('training-mode').addEventListener('change', async () => {
    optionsPending = true;
    $('training-mode').disabled = true;
    try { await chrome.storage.local.set({ trainingOptIn: $('training-mode').checked }); }
    catch { render({ state: 'error', error: 'Couldn’t save your training preference. Try again.' }); }
    finally { optionsPending = false; void refresh(); }
  });
  $('training').addEventListener('click', async () => {
    if (await trainingOptedIn()) await chrome.tabs.create({ url: chrome.runtime.getURL('training.html') });
  });
} else {
  $('training')?.closest('.authoring-tools')?.remove();
  $('diagnostics')?.remove();
}
void refresh().then(() => { if (permissionWindow && !['running', 'starting'].includes(currentState)) void toggleTracking(); });
setInterval(() => { if (!pending) void refresh(); }, 1000);
