import { trainingEnabled } from '../config.js';

export async function trainingOptedIn() {
  if (!trainingEnabled) return false;
  const { trainingOptIn = false } = await chrome.storage.local.get('trainingOptIn');
  return trainingOptIn === true;
}
