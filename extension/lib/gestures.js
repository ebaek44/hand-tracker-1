// Small exported dense networks run directly in JS, avoiding a second ML
// runtime beside MediaPipe. Parity fixtures cover original and newly trained weights.
export function predict(model, input) {
  let values = input;
  for (const layer of model.layers) {
    const next = new Float32Array(layer.bias.length);
    for (let o = 0; o < next.length; o++) {
      let sum = layer.bias[o];
      const weights = layer.weights[o];
      for (let i = 0; i < weights.length; i++) sum += values[i] * weights[i];
      next[o] = layer.activation === 'relu' ? Math.max(0, sum) : sum;
    }
    if (layer.activation === 'softmax') {
      const max = Math.max(...next);
      let total = 0;
      for (let i = 0; i < next.length; i++) total += (next[i] = Math.exp(next[i] - max));
      for (let i = 0; i < next.length; i++) next[i] /= total;
    }
    values = next;
  }
  let id = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[id]) id = i;
  return { id, score: values[id], scores: values };
}

export function normalizeLandmarks(points) {
  const features = new Float32Array(42);
  let scale = 0;
  for (let i = 0; i < 21; i++) {
    features[i * 2] = points[i][0] - points[0][0];
    features[i * 2 + 1] = points[i][1] - points[0][1];
    scale = Math.max(scale, Math.abs(features[i * 2]), Math.abs(features[i * 2 + 1]));
  }
  if (scale > 0) for (let i = 0; i < features.length; i++) features[i] /= scale;
  return features;
}

// A neutral release rearms play/pause. Directional poses repeat once per second;
// every change of direction must pass a fresh dwell, with a global seek cooldown.
export class HoldGestureEngine {
  constructor() {
    this.action = 'none';
    this.since = null;
    this.lastFrame = null;
    this.toggleLatched = false;
    this.releaseSince = null;
    this.lastToggle = -Infinity;
    this.lastSeek = -Infinity;
    this.skipLatched = false;
    this.skipReleaseSince = null;
  }
  update(action, time) {
    if (this.lastFrame !== null && time - this.lastFrame > 600) {
      this.action = 'none';
      this.since = null;
      this.releaseSince = null;
      this.skipReleaseSince = null;
    }
    this.lastFrame = time;
    if (action !== this.action || this.since === null) {
      this.action = action;
      this.since = time;
    }
    if (action !== 'toggle') {
      this.releaseSince ??= time;
      if (time - this.releaseSince >= 180) this.toggleLatched = false;
    } else this.releaseSince = null;
    if (action !== 'skip-ad') {
      this.skipReleaseSince ??= time;
      if (time - this.skipReleaseSince >= 180) this.skipLatched = false;
    } else this.skipReleaseSince = null;
    if (action === 'skip-ad' && !this.skipLatched && time - this.since >= 600) {
      this.skipLatched = true;
      return { type: 'control', value: 'skip_ad' };
    }
    if (action === 'toggle' && !this.toggleLatched && time - this.since >= 400 && time - this.lastToggle >= 1000) {
      this.toggleLatched = true;
      this.lastToggle = time;
      return { type: 'control', value: 'toggle_play_pause' };
    }
    if (['seek-left', 'seek-right'].includes(action) && time - this.since >= 350 && time - this.lastSeek >= 1000) {
      this.lastSeek = time;
      return { type: 'seek', value: action === 'seek-left' ? 'left' : 'right' };
    }
    return null;
  }
}
