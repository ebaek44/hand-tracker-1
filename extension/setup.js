const $ = id => document.getElementById(id);
const send = action => chrome.runtime.sendMessage({ target: 'background', action });
let pending = false;
async function refresh() {
  if (pending) return;
  try {
    const status = await send('get-status');
    const active = ['running', 'starting'].includes(status.state);
    $('state').textContent = status.state === 'running' ? 'Tracking is on' : status.state === 'starting' ? 'Starting tracking…' : 'Ready when you are';
    $('detail').textContent = active ? 'Return to your video. Open the extension any time to stop the camera.' : 'Chrome will ask for camera access. No microphone is needed.';
    $('allow').hidden = active;
    $('stop').hidden = !active;
    $('error').hidden = !status.error;
    $('error').textContent = status.error ?? '';
  } catch (error) { showError(error.message); }
}
function showError(message) { $('error').hidden = false; $('error').textContent = message; }
$('allow').addEventListener('click', async () => {
  pending = true;
  $('allow').disabled = true;
  $('error').hidden = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach(track => track.stop());
    const result = await send('start');
    if (result.error) throw new Error(result.error);
  } catch (error) {
    showError(error.name === 'NotAllowedError' ? 'Camera access was denied. Allow it in Chrome’s camera/site settings and check your system camera permissions, then try again.' : error.message);
  } finally { pending = false; $('allow').disabled = false; }
  // Do not clear a permission error with a stopped status immediately.
  if ($('error').hidden) void refresh();
});
$('stop').addEventListener('click', async () => { await send('stop'); void refresh(); });
void refresh();
setInterval(() => { if (!$('stop').hidden) void refresh(); }, 1000);
