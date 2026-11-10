import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { predict, normalizeLandmarks, HoldGestureEngine } from '../extension/lib/gestures.js';
import { withRetainedExamples, upgradePointUpAction } from '../extension/lib/training-data.js';
const json = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const original = json('../extension/models/classifiers.json');
const fixtures = json('./fixtures/classifier-parity.json');
for (const name of ['keypoint', 'point_history']) {
  test(`original ${name} predictions match TensorFlow Lite (${fixtures[name].length} cases)`, () => {
    for (const item of fixtures[name]) {
      const result = predict(original[name], item.input);
      item.scores.forEach((score, i) => assert.ok(Math.abs(result.scores[i] - score) < 1e-5));
      assert.equal(result.id, item.scores.indexOf(Math.max(...item.scores)));
    }
  });
}
test('active static classifier matches its exported reference predictions', () => {
  const model = json('../extension/models/pose-classifier.json');
  for (const item of json('./fixtures/pose-parity.json')) {
    const result = predict(model, item.input);
    item.scores.forEach((score, i) => assert.ok(Math.abs(result.scores[i] - score) < 1e-5));
  }
  assert.ok(model.labels.some(l => l.action === 'toggle'));
});
test('studio retains palm/neutral examples only and preserves fresh direction recordings', () => {
  const seed = json('../extension/models/training-seed.json');
  assert.equal(seed.samples.length, 4639);
  assert.ok(seed.samples.every(sample => sample.label < 4));
  const labels = [...seed.labels, { name: 'Point right', action: 'seek-right' }, { name: 'Point left', action: 'seek-left' }];
  const features = seed.samples[0].features;
  const fresh = { label: 4, features, session: 'new-user-recording' };
  const merged = withRetainedExamples({ labels, samples: [{ label: 4, features, session: 'legacy' }, fresh] }, seed);
  assert.equal(merged.samples.length, 4640);
  assert.ok(merged.samples.includes(fresh));
  assert.ok(!merged.samples.some(s => s.session === 'legacy' && s.label >= 4));
  assert.equal(withRetainedExamples(merged, seed).samples.length, 4640);
});
test('normalization is translation/scale invariant and finite for collapsed input', () => {
  const points = Array.from({ length: 21 }, (_, i) => [i * 2, i * 3]);
  assert.deepEqual(normalizeLandmarks(points), normalizeLandmarks(points.map(([x,y]) => [x * 2 + 30, y * 2 + 60])));
  assert.ok(normalizeLandmarks(Array.from({ length: 21 }, () => [1,1])).every(n => n === 0));
});
const sample = (engine, action, start, end, step = 50) => {
  const events = [];
  for (let time = start; time <= end; time += step) {
    const event = engine.update(action, time);
    if (event) events.push({ time, ...event });
  }
  return events;
};
test('palm toggles once even when held for many seconds', () => {
  const events = sample(new HoldGestureEngine(), 'toggle', 0, 5000);
  assert.deepEqual(events.map(e => e.time), [400]);
});
test('brief pose noise does not rearm toggle; sustained release does', () => {
  const engine = new HoldGestureEngine();
  sample(engine, 'toggle', 0, 1000);
  sample(engine, 'none', 1050, 1150);
  assert.equal(sample(engine, 'toggle', 1200, 1800).length, 0);
  sample(engine, 'none', 1850, 2100);
  assert.equal(sample(engine, 'toggle', 2150, 2600).length, 1);
});
test('direction holds seek once each second and stop immediately on release', () => {
  const engine = new HoldGestureEngine();
  const events = sample(engine, 'seek-right', 0, 4000);
  assert.deepEqual(events.map(e => e.time), [350,1350,2350,3350]);
  assert.ok(events.every(e => e.value === 'right'));
  assert.equal(sample(engine, 'none', 4050, 6000).length, 0);
});
test('quick transitions, direction reversal and frame gaps need a new dwell', () => {
  const engine = new HoldGestureEngine();
  assert.equal(sample(engine, 'seek-right', 0, 250).length, 0);
  assert.equal(sample(engine, 'seek-left', 300, 550).length, 0);
  assert.equal(engine.update('seek-left', 2000), null);
  assert.equal(sample(engine, 'seek-left', 2050, 2300).length, 0);
  assert.equal(engine.update('seek-left', 2350)?.value, 'left');
  assert.equal(sample(engine, 'seek-right', 2400, 3300).length, 0);
  assert.equal(engine.update('seek-right', 3350)?.value, 'right');
});
test('missing/uncertain hands never emit actions or carry stale dwell', () => {
  const engine = new HoldGestureEngine();
  sample(engine, 'toggle', 0, 300);
  sample(engine, 'none', 350, 1000);
  assert.equal(sample(engine, 'toggle', 1050, 1350).length, 0);
  assert.equal(engine.update('toggle', 1450)?.type, 'control');
});
test('skip-ad needs a deliberate hold, fires once, and rearms after release', () => {
  const engine = new HoldGestureEngine();
  assert.equal(sample(engine, 'skip-ad', 0, 550).length, 0);
  const events = sample(engine, 'skip-ad', 600, 5000);
  assert.deepEqual(events, [{ time: 600, type: 'control', value: 'skip_ad' }]);
  sample(engine, 'none', 5050, 5150);
  assert.equal(sample(engine, 'skip-ad', 5200, 6000).length, 0);
  sample(engine, 'none', 6050, 6300);
  assert.equal(sample(engine, 'skip-ad', 6350, 7000).length, 1);
});
test('old datasets gain the requested point-up action without changing recorded samples', () => {
  const dataset = { labels: [{ name: 'Point up', action: 'none' }, { name: 'Closed hand', action: 'none' }], samples: [{ label: 0, features: [1,2], session: 'recorded' }] };
  const updated = upgradePointUpAction(dataset);
  assert.equal(updated.labels[0].action, 'skip-ad');
  assert.equal(updated.labels[1].action, 'none');
  assert.equal(updated.samples, dataset.samples);
  assert.equal(dataset.labels[0].action, 'none');
});
