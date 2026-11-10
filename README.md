# Hands On Video Controls

Control browser videos with a camera and deliberate hand poses. **The Chrome extension now owns camera capture, MediaPipe landmark detection, gesture classification, and video control. No Python server, Flask API, HTTP polling, or helper application is needed to use it.**

## Working version and user version

- `extension/` is **Hands On Video Controls (Studio)**, with training tools available through a default-off **Options → Training studio** switch.
- Run `npm run build:release` to generate `dist/hands-on/`, **Hands On Video Controls**, with no training pages, datasets, reports, or diagnostics.
- The locally prepared shareable archive is `dist/hands-on-2.1.2.zip`. Nothing is hosted or published automatically.

[Sharing and Chrome Web Store options](PUBLISHING.md) cover unpacked testing, unlisted/public listings, and submission steps. Use one tracking instance at a time if both builds are loaded.

## Try the local extension

1. Open `chrome://extensions` in Chrome 116 or newer and enable **Developer mode**.
2. Choose **Load unpacked** and select this project's `extension/` folder. If you loaded the old extension already, select **Reload**.
3. Open Hands On from Chrome's toolbar. Select **Start tracking**. On first use, the same Hands On controls open temporarily in a small window so Chrome can show its native camera prompt. Allow camera access; the window closes once tracking starts. Your operating system may also ask for camera access.
4. Return to YouTube, Netflix, or Prime Video and click play once if Chrome requires it. Tracking runs in a hidden extension document; later starts need no extra window while camera permission remains granted.
5. Open the extension and select **Stop tracking** to release the camera and remove the worker/document. Tracking starts only when requested, not automatically on browser launch.

All JavaScript, WebAssembly, and model assets are included in `extension/`. Loading this folder needs no npm install, development server, or internet connection. Streaming sites themselves still need their usual connection. The root `extension.zip`, root `popup.html`, helper ZIPs, and old releases are **legacy artifacts**, not the new extension.

## Supported sites and permission scope

Access is limited to HTTPS pages on `www.youtube.com`, `youtube.com`, `m.youtube.com`, `www.netflix.com`, `netflix.com`, `www.primevideo.com`, `primevideo.com`, and `www.amazon.com`. Amazon use is additionally restricted in code to `/gp/video/` pages. Chrome ignores path restrictions for host permissions, so the Amazon manifest grant covers its host; ordinary shopping pages are refused before injection. No generic HTTP/HTTPS, wildcard host, CDN, localhost, or activeTab permission is requested. Regional Amazon storefronts other than the US are not included; use PrimeVideo.com where available.

Ad skipping is **YouTube only**. Other sites are refused before script injection, even while ad skipping is enabled. Named host grants allow gestures to follow supported tabs without clicking the extension on every site change; activeTab would require a new browser interaction on a different origin. The debugger permission remains required for the user-approved ad-skip implementation and may still receive scrutiny during review.

Automated fixtures verify routing/permissions on the named services; they do not prove compatibility with every live Netflix/Prime Video player or DRM variant. Live YouTube skipping was confirmed by the user. No media downloads, DRM bypass, or ad skipping for Netflix/Prime Video is implemented.

## Camera permission and device choice

Start tracking checks Chrome's camera permission for the extension. Already granted: start the offscreen camera directly. Not decided: briefly open the same controls in a normal extension window, request camera access without microphone access, stop that initial stream, then start the offscreen tracking stream. The temporary window closes when tracking is running. Denied: show an inline error; the user must allow the extension in Chrome's camera settings and, where applicable, system privacy settings before retrying.

Hands On does not enumerate devices or present its own picker. With multiple cameras, Chrome chooses its configured default because the request has no `deviceId`. Change it at `chrome://settings/content/camera` before starting; stop/restart after changing it. With no camera, show “No camera found”; an occupied/unreadable camera gets a busy/retry message. Closing the regular popup does not stop tracking; **Stop tracking** stops every media track and the worker.

## Controls

| Pose | Behavior |
| --- | --- |
| Open palm | Hold for 0.4 seconds to toggle play/pause once. Relax briefly before toggling again. |
| Point right | Hold for 0.35 seconds to seek +10 seconds; continue holding to repeat about once per second. |
| Point left | Hold for 0.35 seconds to seek −10 seconds; continue holding to repeat about once per second. |
| Point up | With YouTube ad skipping enabled in Options, hold briefly to press the visible Skip ad button. Relax before using again. |
| Lower/relax hand | Stop seeking. |

Only the active YouTube, Netflix, or Prime Video tab in the focused browser window receives commands. The extension selects a single visible, ready video, favoring playing/audible video and then visible area, including accessible embedded players and open shadow roots. Playback respects browser autoplay restrictions; seek commands clamp to the available seekable ranges. Chrome's internal pages, the Chrome Web Store, inaccessible frames, and players without a usable HTML video element cannot be controlled. Closed shadow DOM and unusual site-specific players may need further adaptation.

The current local model was trained from the user's `hands-on-dataset.json`: 614 fresh right-point samples across 20 recordings, 519 fresh left-point samples across 17 recordings, and the retained palm/neutral examples. No archived directional rows were added. The model is installed in `extension/models/pose-classifier.json`; reload the unpacked extension to use it. Held-out directional recordings were correctly classified in recorded validation (122/122 right, 92/92 left); live hand/site accuracy still needs manual checking. See `extension/models/training-report.json`.

`npm run reset:training-baseline` restores the original three-class palm/neutral model and disables directional seeking until you install a newly trained model. The studio retains its recorded sessions independently.

## Train your own poses (Studio only)

In the working extension, turn on **Options → Training studio**, then open **Training studio**. Turning the switch off stops an open Studio camera, closes the Studio view, and preserves saved recordings. The Store build contains no training tools. Collect labeled landmark batches, add poses and assign their actions, export the dataset, then train locally:

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

## YouTube ad skipping (optional)

Open **Options → YouTube ad skipping** in the popup to enable it. It is off by default. Then point up when the native Skip ad button is visible. There is no queued click if the button appears later. Switching the option off disables ad clicks and detaches any active session. Play/pause and seeking do not use the debugger.

The current public YouTube player explicitly rejects untrusted synthetic clicks, including `button.click()`. Since version **2.1.0**, Hands On uses browser-level mouse input instead. A test reproducing the player’s trust guard verifies that the extension’s click has `isTrusted === true`. See [investigation and primary source](docs/youtube-skip-investigation.md).

Point-up starts a direct, one-time scan for the visible Skip ad control in the active tab. It does not depend on which video is audible or ready. The selector/visibility logic is adapted from [xhy0405/youtube-auto-skip](https://github.com/xhy0405/youtube-auto-skip), with its MIT license included. After locating the control, the extension attaches to that tab, remeasures its coordinates, sends one browser-level click, verifies the response, and detaches. It cancels on tab/URL/focus changes or permission/option removal. It does not seek ads, bypass countdowns, or use a separate helper. Embedded players must be opened in their own YouTube tab for this action.

Chrome requires declaring the debugger permission at installation; [it cannot be an optional permission](https://developer.chrome.com/docs/extensions/reference/api/permissions#implement_optional_permissions). The option controls whether Hands On uses it. Chrome may show a debugging notice during the click. No additional page data or camera frames are uploaded. The popup shows skip results. There is no manual Skip ad test button or message route; only recognized gestures trigger ad clicks.

After updating, reload the unpacked extension, confirm version 2.1.2, and refresh YouTube. No model retraining is needed. Automated checks verify default-off behavior, trusted input, no queued clicks, cancellation and cleanup. The user confirmed live YouTube skipping works with this mode; site changes can still affect compatibility.
