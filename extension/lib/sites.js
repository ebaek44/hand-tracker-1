// Keep this explicit list in sync with manifest host_permissions.
export const supportedHosts = new Set([
  'www.youtube.com', 'youtube.com', 'm.youtube.com',
  'www.netflix.com', 'netflix.com', 'www.primevideo.com', 'primevideo.com',
  'www.amazon.com',
]);
export function supportedSite(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || !supportedHosts.has(url.hostname)) return false;
    // Host permissions cannot restrict URL paths. Keep Amazon use to Prime Video.
    return url.hostname !== 'www.amazon.com' || /^\/gp\/video(?:\/|$)/.test(url.pathname);
  } catch { return false; }
}
export function youtubeSite(value) {
  try { return supportedSite(value) && ['www.youtube.com', 'youtube.com', 'm.youtube.com'].includes(new URL(value).hostname); }
  catch { return false; }
}
