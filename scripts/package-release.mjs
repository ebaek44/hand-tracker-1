import { readFile, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const folder = new URL('../dist/hands-on/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', folder), 'utf8'));
if (!/^\d+(\.\d+){0,3}$/.test(manifest.version)) throw new Error('Invalid extension version');
const archive = fileURLToPath(new URL(`../hands-on-${manifest.version}.zip`, folder));
await rm(archive, { force: true });
try {
  execFileSync('zip', ['-qr', archive, '.'], { cwd: fileURLToPath(folder) });
} catch (error) {
  if (error.code === 'ENOENT') throw new Error('The zip command is required for packaging. You can also compress the contents of dist/hands-on manually.');
  throw error;
}
console.log(`Shareable archive: ${archive}`);
