# Hands On privacy policy

Last updated: November 10, 2026

Hands On uses hand gestures to control video playback in your active browser tab.

## Camera processing

Camera access begins only when you request tracking and approve Chrome’s camera permission. On first use, the Hands On controls open temporarily in a window for Chrome’s native permission prompt. After permission is granted, that initial camera stream is released and the tracking stream starts; the temporary window closes when tracking is ready. Stop tracking releases the tracking stream. Frames are processed locally in memory to detect hand landmarks and recognize gestures. Hands On does not record camera images or video, upload camera data, or transmit hand coordinates to a server. Recognition models and their runtime are packaged with the extension.

## Website access

Hands On requests HTTPS access only to the specified YouTube, Netflix, and Prime Video domains, including www.amazon.com for US Prime Video. It locates a video in a supported active tab and applies playback actions. Amazon use is additionally restricted in code to /gp/video/ pages; unrelated sites and ordinary Amazon shopping pages are refused before script injection. YouTube ad skipping is optional and off by default. Chrome requires its debugger permission at installation, even while ad skipping is off. If you enable ad skipping, Hands On temporarily attaches to the active YouTube tab and sends browser-level mouse input to its visible Skip ad button in response to a gesture. Chrome may display a debugging notice. The extension detaches after the operation; switching the option off also detaches and prevents further debugger use. It does not maintain a browsing-history log, upload page content, or use website access for advertising or analytics.

## Storage and sharing

The user version does not include training tools or store gesture recordings. It stores your ad-skipping preference locally on this device. Hands On does not use accounts, analytics, tracking cookies, or external application services. It does not sell or share your camera or browsing data.

## Your controls

Use Stop tracking to release the camera. Camera permission and site access can also be changed in Chrome's extension and privacy settings. Uninstalling the extension removes its local extension storage.

## Contact

For support or privacy questions, contact the developer through the [Hands On GitHub issue tracker](https://github.com/ebaek44/hand-tracker-1/issues).
