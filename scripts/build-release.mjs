import { cp, mkdir, readFile, writeFile, rm, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const source = new URL('extension/', root);
const target = new URL('dist/hands-on/', root);
const model = JSON.parse(await readFile(new URL('models/pose-classifier.json', source), 'utf8'));
for (const action of ['toggle', 'seek-left', 'seek-right']) {
  if (!model.labels.some(label => label.action === action)) throw new Error(`Train and install ${action} before building a release.`);
}
// Explicit runtime allowlist: no studio, samples, reports, notebooks, or author notes.
const files = [
  'manifest.json', 'background.js', 'content.js', 'youtube.js', 'icon128.png',
  'popup.html', 'popup.js', 'styles.css',
  'offscreen.html', 'offscreen.js', 'tracking-worker.js', 'lib/gestures.js',
  'lib/sites.js', 'lib/ad-settings.js', 'lib/trusted-click.js', 'lib/training-settings.js',
  'models/pose-classifier.json', 'models/hand_landmarker.task',
  'models/hand_landmarker.sha256', 'THIRD_PARTY_NOTICES.md',
  'licenses/youtube-auto-skip.txt',
];
await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });
for (const file of files) {
  const destination = new URL(file, target);
  await mkdir(new URL('./', destination), { recursive: true });
  if (file.endsWith('.html')) {
    const html = (await readFile(new URL(file, source), 'utf8')).replace(/<!-- authoring:start -->[\s\S]*?<!-- authoring:end -->/g, '');
    await writeFile(destination, html);
  } else if (file === 'manifest.json') {
    const manifest = JSON.parse(await readFile(new URL(file, source), 'utf8'));
    manifest.name = 'Hands On Video Controls';
    await writeFile(destination, JSON.stringify(manifest, null, 2) + '\n');
  } else await cp(new URL(file, source), destination);
}
await writeFile(new URL('config.js', target), 'export const trainingEnabled = false;\n');
await cp(new URL('vendor/', source), new URL('vendor/', target), { recursive: true });
await cp(new URL('hand_tracker/LICENSE', root), new URL('GESTURE-LICENSE', target));
// Source README is maintainer documentation; retain runtime license files only.
for (const name of await readdir(new URL('vendor/', target))) {
  if (name.includes('README')) await rm(new URL(`vendor/${name}`, target));
}
console.log(`User-only extension prepared at ${fileURLToPath(target)}. Nothing published.`);
