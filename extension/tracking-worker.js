/* Classic worker: MediaPipe's WASM loader requires importScripts(). */
(async () => {
  let landmarker;
  let classifiers;
  let engine;
  let core;
  let collecting = false;
  let lastReport = -Infinity;
  const canvas = new OffscreenCanvas(640, 480);
  const context = canvas.getContext('2d', { alpha: false });
  self.onmessage = async ({ data }) => {
    try {
      if (data.type === 'init') {
        importScripts('vendor/vision.js');
        core = await import('./lib/gestures.js');
        classifiers = await (await fetch('models/pose-classifier.json')).json();
        engine = new core.HoldGestureEngine();
        collecting = Boolean(data.collect);
        const vision = await HandsOnVision.FilesetResolver.forVisionTasks(new URL('vendor/wasm/', self.location.href).href);
        landmarker = await HandsOnVision.HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: new URL('models/hand_landmarker.task', self.location.href).href, delegate: 'CPU' },
          runningMode: 'VIDEO', numHands: 1,
          minHandDetectionConfidence: 0.7, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.5,
        });
        self.postMessage({ type: 'ready' });
      } else if (data.type === 'frame') {
        const started = performance.now();
        const { bitmap, time } = data;
        try {
          // Match cv.flip(image, 1) used with the original trained classifiers.
          context.setTransform(-1, 0, 0, 1, canvas.width, 0);
          context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        } finally { bitmap.close(); }
        const result = landmarker.detectForVideo(canvas, time);
        const landmarks = result.landmarks[0];
        let handId = -1;
        let score = 0;
        let points;
        let features;
        if (landmarks) {
          points = landmarks.map(p => [Math.min(Math.trunc(p.x * 640), 639), Math.min(Math.trunc(p.y * 480), 479)]);
          features = core.normalizeLandmarks(points);
          const prediction = core.predict(classifiers, features);
          score = prediction.score;
          if (score >= 0.8) handId = prediction.id;
        }
        const action = handId < 0 ? 'none' : classifiers.labels[handId].action;
        const event = collecting ? null : engine.update(action, time);
        const report = time - lastReport >= 250 || event;
        if (report) lastReport = time;
        self.postMessage({
          type: 'result', event, inferenceMs: performance.now() - started,
          ...(collecting ? { features: features ? Array.from(features) : null, landmarks: points } : {}),
          ...(report ? { observation: {
            hand: handId < 0 ? (landmarks ? 'Uncertain pose' : 'No hand detected') : classifiers.labels[handId].name,
            confidence: score, action,
          } } : {}),
        });
      }
    } catch (error) { self.postMessage({ type: 'error', error: error.message }); }
  };
})();
