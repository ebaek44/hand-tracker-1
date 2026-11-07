# Hands On Video Controls

Control browser videos with a camera and deliberate hand poses. **The Chrome extension now owns camera capture, MediaPipe landmark detection, gesture classification, and video control. No Python server, Flask API, HTTP polling, or helper application is needed to use it.**

## Try the local extension

1. Open `chrome://extensions` in Chrome 116 or newer and enable **Developer mode**.
2. Choose **Load unpacked** and select this project's `extension/` folder. If you loaded the old extension already, select **Reload**.
3. Open Hands On from Chrome's toolbar. Select **Start tracking**, then allow the camera in **Camera setup** when prompted. Camera setup may also require permission in your operating system.
4. Return to an ordinary HTTP/HTTPS video page and click play once if Chrome requires it. The setup tab can be closed; tracking runs in a hidden extension document.
5. Open the extension and select **Stop tracking** to release the camera and remove the worker/document. Tracking starts only when requested, not automatically on browser launch.

All JavaScript, WebAssembly, and model assets are included in `extension/`. Loading this folder needs no npm install, development server, or internet connection. Streaming sites themselves still need their usual connection. The root `extension.zip`, root `popup.html`, helper ZIPs, and old releases are **legacy artifacts**, not the new extension.

## Controls

| Pose | Behavior |
| --- | --- |
| Open palm | Hold for 0.4 seconds to toggle play/pause once. Relax briefly before toggling again. |
| Point right | Hold for 0.35 seconds to seek +10 seconds; continue holding to repeat about once per second. |
| Point left | Hold for 0.35 seconds to seek −10 seconds; continue holding to repeat about once per second. |
| Lower/relax hand, or point up | Stop seeking. |

The active tab in the focused browser window receives commands. The extension selects a single visible, ready video, favoring playing/audible video and then visible area, including accessible embedded players and open shadow roots. Playback respects browser autoplay restrictions; seek commands clamp to the available seekable ranges. Chrome's internal pages, the Chrome Web Store, inaccessible frames, and players without a usable HTML video element cannot be controlled. Closed shadow DOM and unusual site-specific players may need further adaptation.

The current local model was trained from the user's `hands-on-dataset.json`: 614 fresh right-point samples across 20 recordings, 519 fresh left-point samples across 17 recordings, and the retained palm/neutral examples. No archived directional rows were added. The model is installed in `extension/models/pose-classifier.json`; reload the unpacked extension to use it. Held-out directional recordings were correctly classified in recorded validation (122/122 right, 92/92 left); live hand/site accuracy still needs manual checking. See `extension/models/training-report.json`.

`npm run reset:training-baseline` restores the original three-class palm/neutral model and disables directional seeking until you install a newly trained model. The studio retains its recorded sessions independently.

## Train your own poses

Open **Training studio** from the extension. Collect labeled landmark batches, add poses and assign their actions, export the dataset, then train locally:

```bash
.venv/bin/python training/train.py --data /absolute/path/to/hands-on-dataset.json --install
```

Reload the extension at `chrome://extensions`. [Training instructions](training/README.md) cover environment setup, neutral examples, validation, the included data, and model export. Python/TensorFlow are training tools only. Runtime inference remains JavaScript + WebAssembly.

## How it works

```text
Camera (offscreen extension document)
  → one transferable frame at a time
  → dedicated worker: mirrored frame → MediaPipe landmarks → 42 normalized x/y features
  → small trained dense classifier → confidence gate → time-based gesture state
  → MV3 service worker, awakened only by events
  → one chosen frame/video in the active tab
```

- Requests a 640×480 camera stream capped at 15 FPS, tracks one hand, skips duplicate camera frames, and adapts processing intervals toward a 50% inference duty cycle. That is a scheduling target, not a total CPU-usage guarantee.
- Never queues inference: the next frame is scheduled only after the previous result. Detection and the small classifier run away from page UI work.
- Removes OpenCV debug copies/drawing, motion-history inference, the 90 Hz gesture thread, network polling, and per-page keepalive timers from the active runtime.
- A 0.80 confidence gate rejects uncertain poses. Fresh dwell is required after changes or processing gaps. Toggle uses a release latch; seeking repeats on a single shared one-second cooldown and cancels on a neutral/missing hand.
- Frames are not uploaded or recorded. Training saves landmark features only. The runtime CSP permits packaged scripts/WASM and same-extension connections only.
- Camera disconnect, initialization errors, and stalled inference appear in the popup. Stop terminates the worker and every media track. Camera/model ownership survives service-worker suspension.

MediaPipe's [web guidance](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js) describes synchronous inference and the use of workers. Chrome's [Offscreen API](https://developer.chrome.com/docs/extensions/reference/api/offscreen) supplies the hidden camera document.

## Development and verification

Use Node 20+ (verified locally with Node 24):

```bash
npm ci
npm run prepare:extension
npm run check
npm test
npx playwright install chromium
npm run test:browser
```

`prepare:extension` copies the pinned MediaPipe 0.10.32 runtime from npm and downloads the versioned landmark model only if missing, checking its committed SHA-256. Commit the prepared vendor/model assets when sharing the extension so users need no build step. Third-party attribution is in `extension/THIRD_PARTY_NOTICES.md`.

Tests check original TFLite and active classifier prediction parity, normalization, held-pose repeat/release/noise/gap behavior, real offline inference with a simulated camera, lifecycle cleanup/restart, actual HTML video play/pause/seeking, choosing a single frame, embedded video, MV3 worker restart, and studio dataset persistence/export. Simulated-camera tests do **not** establish live hand accuracy or YouTube/Vimeo compatibility; those require manual testing.

## Next interaction

Pinch-to-scrub is planned separately: capture a hand/time anchor on pinch, map horizontal movement to playback time, smooth small movements, and exit on release or lost tracking. It needs a visible scrub-on/off indicator that remains visible with the popup closed, such as a video border/overlay. This interaction is **not implemented yet**.

## Historical implementation

`server.py`, `hand_tracker/`, `handlers/`, training notebooks, and native packaging specs remain as reference/training history. They are outside the new extension's startup path. Their old setup instructions and binaries describe the original Python/Flask version.
