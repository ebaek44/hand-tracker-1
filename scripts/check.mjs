import { readdir, readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const root = new URL('../', import.meta.url);
async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (['vendor', 'models', 'fixtures'].includes(entry.name)) continue;
    const url = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory);
    if (entry.isDirectory()) await walk(url);
    else if (/\.(m?js)$/.test(entry.name)) execFileSync(process.execPath, ['--check', url.pathname]);
  }
}
await walk(new URL('extension/', root));
await walk(new URL('scripts/', root));
const manifest = JSON.parse(await readFile(new URL('extension/manifest.json', root)));
if (manifest.host_permissions.some(p => p.includes('3333')) || manifest.permissions.includes('tabs')) throw new Error('Legacy permission remains');
if (manifest.optional_permissions?.includes('debugger')) throw new Error('Chrome does not support optional debugger permission');
const hash = createHash('sha256').update(await readFile(new URL('extension/models/hand_landmarker.task', root))).digest('hex');
if (hash !== (await readFile(new URL('extension/models/hand_landmarker.sha256', root), 'utf8')).trim()) throw new Error('Model checksum mismatch');
console.log('JavaScript syntax, manifest and bundled model checksum passed.');
