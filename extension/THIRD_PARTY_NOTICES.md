# Third-party assets

- YouTube Skip ad button selectors and visibility checks in `youtube.js` are adapted from [xhy0405/youtube-auto-skip](https://github.com/xhy0405/youtube-auto-skip), Copyright (c) 2026 xhhhhhhhy, MIT. The implementation is scoped to a single user gesture and player, with optional browser-level input and response verification. Full license: `licenses/youtube-auto-skip.txt`.

- MediaPipe Tasks Vision 0.10.32 (`vendor/vision.js`, `vendor/wasm/*`) is published by Google under Apache-2.0. Source: https://github.com/google-ai-edge/mediapipe. The JavaScript bundle is wrapped to expose its CommonJS exports in a classic worker; its inference implementation is unchanged. License: `vendor/LICENSE`.
- `models/hand_landmarker.task` is the MediaPipe Hand Landmarker float16 version 1 model distributed at https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task. Its SHA-256 is recorded beside the model. MediaPipe model documentation: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker.
- The project's original gesture models/training lineage are retained in the source repository under `hand_tracker/`. The gesture classifier was trained locally from recorded landmarks. Original gesture-project license: `GESTURE-LICENSE` in the release bundle.
