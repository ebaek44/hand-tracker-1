export const sampleKey = sample => `${sample.label}:${sample.session}:${sample.features.join(',')}`;

// Add only the archived palm/neutral examples. Preserve fresh user recordings,
// including newly recorded directions, when upgrading an existing studio dataset.
export function withRetainedExamples(dataset, seed) {
  if (JSON.stringify(dataset.labels.slice(0, seed.labels.length)) !== JSON.stringify(seed.labels)) return dataset;
  const samples = dataset.samples.filter(s => !(s.session === 'legacy' && s.label >= seed.labels.length));
  const keys = new Set(samples.map(sampleKey));
  for (const sample of seed.samples) {
    const key = sampleKey(sample);
    if (!keys.has(key)) { samples.push(sample); keys.add(key); }
  }
  return { ...dataset, samples };
}
