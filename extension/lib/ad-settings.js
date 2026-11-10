export async function adSkippingEnabled() {
  const { youtubeAdSkipping = false } = await chrome.storage.local.get('youtubeAdSkipping');
  return youtubeAdSkipping && await chrome.permissions.contains({ permissions: ['debugger'] });
}

export async function setAdSkipping(enabled) {
  if (enabled && !await chrome.permissions.contains({ permissions: ['debugger'] })) {
    return { ok: false, error: 'Chrome’s permission is needed to enable ad skipping.' };
  }
  await chrome.storage.local.set({ youtubeAdSkipping: enabled });
  return { ok: true, enabled };
}
