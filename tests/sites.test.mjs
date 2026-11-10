import test from 'node:test';
import assert from 'node:assert/strict';
import { supportedSite, youtubeSite } from '../extension/lib/sites.js';

test('only the explicit HTTPS streaming hosts are supported', () => {
  for (const url of ['https://www.youtube.com/watch?v=x','https://m.youtube.com/watch?v=x','https://www.netflix.com/watch/1','https://www.primevideo.com/detail/1','https://www.amazon.com/gp/video/detail/1']) assert.equal(supportedSite(url), true, url);
  for (const url of ['http://www.youtube.com/watch','https://youtube.com.evil.test/','https://example.com/','https://www.amazon.com/cart','https://www.amazon.com/gp/videographer/','https://unlisted.netflix.com/','invalid']) assert.equal(supportedSite(url), false, url);
});
test('ad skipping is limited to YouTube', () => {
  assert.equal(youtubeSite('https://www.youtube.com/watch?v=x'), true);
  for (const url of ['https://www.netflix.com/watch/1','https://www.primevideo.com/','https://www.amazon.com/gp/video/detail/1']) assert.equal(youtubeSite(url), false);
});
