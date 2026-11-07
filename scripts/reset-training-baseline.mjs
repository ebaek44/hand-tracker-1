// Explicit reset: restore the original three-class classifier and preload only
// archived palm/neutral examples for future training. Never called during builds.
import { readFile, writeFile } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
const labels = [
  { name: 'Open palm', action: 'toggle' }, { name: 'Closed hand', action: 'none' },
  { name: 'Point up', action: 'none' }, { name: 'Confirm / neutral', action: 'none' },
];
const original = JSON.parse(await readFile(new URL('extension/models/classifiers.json', root), 'utf8')).keypoint;
const rows = (await readFile(new URL('hand_tracker/model/keypoint_classifier/keypoint.csv', root), 'utf8')).trim().split(/\r?\n/).map(row => row.split(',').map(Number));
const samples = rows.filter(row => row[0] < 4).map(row => ({ label: row[0], features: row.slice(1), session: 'legacy' }));
await writeFile(new URL('extension/models/training-seed.json', root), JSON.stringify({ schemaVersion: 1, labels, samples }) + '\n');
await writeFile(new URL('extension/models/pose-classifier.json', root), JSON.stringify({
  schemaVersion: 1, source: original.source, sha256: original.sha256,
  labels: labels.slice(0, original.labels.length), layers: original.layers,
}) + '\n');
const report = {
  source: 'Original three-class TensorFlow Lite classifier; exact weights preserved',
  activeClasses: labels.slice(0, original.labels.length),
  directionalTrainingIncluded: false,
  retainedExamples: Object.fromEntries(labels.map((label, index) => [label.name, samples.filter(s => s.label === index).length])),
  directions: 'Untrained. Collect new left/right examples in Training studio and train/install before using held-direction seeking.',
  limitations: 'Original training/validation split is unknown. Retained CSV counts describe studio seed data, not verified original training counts. Confirm samples are retained for the next training run but are not a fourth output of the original classifier.',
};
await writeFile(new URL('extension/models/training-report.json', root), JSON.stringify(report, null, 2) + '\n');
const parity = JSON.parse(await readFile(new URL('tests/fixtures/classifier-parity.json', root), 'utf8')).keypoint;
await writeFile(new URL('tests/fixtures/pose-parity.json', root), JSON.stringify(parity) + '\n');
console.log(`Restored original palm/neutral classifier; retained ${samples.length} examples; zero archived left/right samples in the active pipeline.`);
