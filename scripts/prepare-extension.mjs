import { mkdir, readFile, writeFile, copyFile, cp, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const root = new URL('../', import.meta.url);
const vendor = new URL('extension/vendor/', root);
const source = new URL('node_modules/@mediapipe/tasks-vision/', root);
await mkdir(vendor, { recursive: true });
// Adapt the published CommonJS build to a classic worker. MediaPipe's WASM
// loader uses importScripts(), which is unavailable in module workers.
const bundle = await readFile(new URL('vision_bundle.cjs', source), 'utf8');
await writeFile(new URL('vision.js', vendor), `(function(exports){\n${bundle}\n})(globalThis.HandsOnVision = {});\n`);
await cp(new URL('wasm/', source), new URL('wasm/', vendor), { recursive: true });
const model = new URL('extension/models/hand_landmarker.task', root);
await mkdir(new URL('extension/models/', root), { recursive: true });
try {
  await access(model);
} catch {
  const response = await fetch('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');
  if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
  await writeFile(model, new Uint8Array(await response.arrayBuffer()));
}
const hash = createHash('sha256').update(await readFile(model)).digest('hex');
// Verify the versioned model against the committed checksum.
const checksumFile = new URL('extension/models/hand_landmarker.sha256', root);
const previous = (await readFile(checksumFile, 'utf8')).trim();
if (previous !== hash) throw new Error('Hand landmark model checksum mismatch');
await copyFile(new URL('README.md', source), new URL('MEDIAPIPE-README.md', vendor));
console.log(`Extension prepared locally (MediaPipe 0.10.32; model SHA-256 ${hash}).`);
