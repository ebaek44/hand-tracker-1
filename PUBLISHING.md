# Publish Hands On yourself

Hands On runs entirely on the user's computer. No hosted backend, Python installation, API key, or model training is required to use it. These instructions prepare a submission; they do not publish anything automatically.

## Upload this build

The current package is **`dist/hands-on-2.1.2.zip`**. It contains `manifest.json` at the ZIP root. Do not upload the repository, old root `extension.zip`, Studio folder, or old native-helper binaries.

| Folder | Chrome name | Use |
| --- | --- | --- |
| `extension/` | Hands On Video Controls (Studio) | Working source. Training is off by default; opt in/out with Options → Training studio. Saved recordings remain when disabled. |
| `dist/hands-on/` | Hands On Video Controls | Generated Store build. No training pages/data, training switch, diagnostics, camera setup page, or manual Skip ad test route. |

Edit source and generate a fresh Store ZIP with:

```bash
npm run check
npm test
npm run test:browser
npm run package:release
```

The builder copies your installed trained model; it never resets or retrains it. `dist/` is generated and ignored by Git. Use only one tracker at a time if both builds are installed.

## 1. Test the packaged extension

Extract the new ZIP into a permanent folder. In `chrome://extensions`, enable Developer mode and Load unpacked that folder. Disable the Studio extension while testing. Confirm version **2.1.2** in Options.

Start tracking and approve the native camera prompt. First permission uses a temporary window showing the same Hands On controls; it closes when tracking starts. There is no separate Camera access page or persistent link. Later starts use the existing grant. Verify play/pause, held ±10-second seeking, release, stopping/restarting and optional YouTube skipping on your actual camera. Ad skipping remains off until enabled in Options. Pinch scrubbing is not included.

Multiple cameras: Chrome's configured default is used; change it at `chrome://settings/content/camera` and stop/restart tracking. No camera: Hands On shows a missing-camera error. Denied access: allow Hands On in Chrome/system camera settings. A busy camera gets a retry message.

## 2. Set up your publisher account

Open the [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole), register, accept the developer terms yourself, and pay the one-time fee displayed by Google. Fill in your publisher name, verify your contact email, and enable Google account two-step verification. [Registration](https://developer.chrome.com/docs/webstore/register), [account setup](https://developer.chrome.com/docs/webstore/set-up-account), [two-step requirement](https://developer.chrome.com/docs/webstore/program-policies/policies).

## 3. Prepare the listing

- Name: **Hands On Video Controls**.
- Suggested short description: **Control browser videos with hand gestures. Camera processing stays on your device.**
- Suggested category: **Accessibility**, if available in your Dashboard.
- Store icon: `extension/icon128.png` (128 × 128).
- Capture at least one actual-use screenshot at 1280 × 800; up to five can demonstrate playback, seeking and the optional ad setting. Keep training tools out of screenshots of the user build.
- Prepare a 440 × 280 promotional tile if requested by the Dashboard. Google's image and listing pages differ on whether it is mandatory, so having one prepared avoids blocking submission.
- Supply your support contact/URL and a public privacy-policy URL. Host the text from `PRIVACY.md` on a public page you control, adding your real publisher/contact details. The ZIP's local Markdown file alone does not provide that URL. The simplest option is to commit and push `PRIVACY.md` to this public repository, then use `https://github.com/ebaek44/hand-tracker-1/blob/main/PRIVACY.md`. Confirm the policy is readable in a signed-out/incognito window. The policy includes the project's GitHub issue tracker as its contact.

See [listing fields](https://developer.chrome.com/docs/webstore/cws-dashboard-listing) and [image guidance](https://developer.chrome.com/docs/webstore/images).

Suggested full description:

> Hands On lets you control browser video with deliberate hand poses. Hold an open palm to play or pause. Point left or right to move backward or forward by 10 seconds, repeating while held. Lower your hand to stop seeking.
>
> Camera frames and hand recognition stay on your device. There is no account, server, or Python setup. Start and stop the camera from the extension.
>
> Optional YouTube ad skipping is off by default. Enable it in Options, then point up while YouTube's native Skip ad button is visible. It does not bypass countdowns or unskippable ads. This feature uses Chrome's debugger permission and may show a debugging notice during the click. Chrome requires this permission at installation, even while the option is off.
>
> Works with accessible HTML video players on YouTube, Netflix and Prime Video. Chrome's internal pages, the Chrome Web Store, some embedded players and unusual site-specific players cannot be controlled. Recognition depends on hand visibility and lighting.

## Scoped host-permission justification

Paste this into the Store's Host permission justification after uploading 2.1.2:

> Access is limited to the HTTPS domains used by YouTube, Netflix and Prime Video, including www.amazon.com for US Prime Video. It is needed to locate and control the video in the user's active supported tab after a hand gesture. Amazon use is additionally restricted in code to /gp/video/ pages; ordinary shopping pages and unrelated websites are refused before injection. Ad skipping works only on YouTube. No all-sites or CDN access is requested, and no browsing history or page content is uploaded.

The exact manifest lists the www/bare YouTube, Netflix and PrimeVideo domains, mobile YouTube, and www.amazon.com. Chrome ignores host-permission path restrictions, so Amazon's host grant cannot be narrowed to /gp/video/ in the manifest. The application enforces that restriction itself. Other regional Amazon storefronts are excluded. Scoped grants preserve switching between supported tabs without requiring a fresh extension click for each origin, as activeTab would. [Chrome host match-pattern rules](https://developer.chrome.com/docs/extensions/develop/concepts/match-patterns).

## 4. Upload and complete Privacy practices

In the Dashboard, select **Add new item**, upload `dist/hands-on-2.1.2.zip`, then fill Store listing, Privacy practices, Distribution and Test instructions. [Publishing workflow](https://developer.chrome.com/docs/webstore/publish).

Single-purpose text:

> Hands-free control of video playback in the user's active browser tab using locally recognized hand gestures.

Permission justifications, based on the actual implementation:

| Permission | Explanation |
| --- | --- |
| `offscreen` | Keeps the user-started camera and local inference running when the popup closes. Stop tracking releases the stream and closes this document. |
| `scripting` | Injects packaged video-control code into the active page and its accessible frames only when a gesture occurs. |
| Named HTTPS host access | Limited to YouTube, Netflix and Prime Video player domains, including www.amazon.com for US Prime Video. Finds and controls the active video only on those services. Amazon is further limited in code to /gp/video/ pages; ordinary shopping pages are refused. No all-sites, CDN or HTTP access is requested. |
| `storage` | Saves the user's local, default-off YouTube ad-skipping preference. |
| `debugger` | Only when the user enables ad skipping and points up: briefly attach to the active YouTube tab and send browser-level input to an already-visible native Skip ad button, then detach. No queued skips. Chrome disallows declaring this permission as optional. |

For remote code, explain that JavaScript, WASM and inference models are bundled; no executable code is downloaded at runtime. Complete data-use disclosures from the implementation, including local camera processing. Explain that frames/coordinates remain on-device in memory, are not recorded or uploaded, and the Store build contains no training-data collection. Do not equate “no upload” with “no access to user data.” Keep Dashboard answers consistent with the hosted policy. [Privacy fields](https://developer.chrome.com/docs/webstore/cws-dashboard-privacy), [local-data disclosure](https://developer.chrome.com/docs/webstore/program-policies/user-data-faq).

## 5. Add reviewer instructions and submit

Suggested reviewer instructions:

> No extension login is required. Use a public YouTube video for review without a streaming subscription. Netflix and Prime Video require the reviewer’s own service access. Use a desktop/laptop with a camera. Open a public YouTube video, open Hands On, select Start tracking and allow the camera. The first-use permission window closes when tracking starts. An open palm held briefly toggles play/pause once; relax before repeating. Point left/right to seek by 10 seconds repeatedly while held, and lower the hand to stop. Stop tracking releases the camera.
>
> YouTube ad skipping is an explicit default-off option. Enable Options → YouTube ad skipping. With a native Skip ad button already visible, hold point-up briefly. The extension sends one browser-level click, verifies the result and detaches. Missing/disabled buttons cause no future queued action. Chrome can show a debugging notice. Open embedded YouTube videos in their own YouTube tab for this feature. No separate helper, training step or test button is needed.

Choose **Public** for Store discovery or **Unlisted** for installation through a direct link. Both require review. Select your distribution regions, save all required fields, and submit for review. If offered, choose to publish manually after approval so you can control launch timing. [Distribution](https://developer.chrome.com/docs/webstore/cws-dashboard-distribution).

Google may ask for clarification about Amazon host access or debugger use; explain their narrow actual use above. Review approval is Google's decision. Do not change the description to conceal these capabilities.

## Future updates

Increase the manifest version, keep package versions in sync, rebuild/test the ZIP, and upload it to the **same Store item**. Chrome distributes published Store updates; people using unpacked ZIPs update manually.

Nothing has been registered, purchased, committed, pushed, or published by the local release scripts.
