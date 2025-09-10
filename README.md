# Hands-On Video Controls

Control online videos with **hand gestures**. A native server (MediaPipe + OpenCV) detects gestures from your camera and a Chrome extension polls them and controls `<video>` elements (YouTube, Vimeo, etc.).

---

## Downloads

> Replace `<OWNER>` and `<REPO>` with your GitHub user/org and repo name.
> Use **Latest** links if you want the README to always point to your newest release,
> or **Tagged** links if you want a specific version.

**Device Specific Download:**

- [Server (macOS/Linux)](https://github.com/ebaek44/hand-tracker-1/releases/download/v1.0.0/gesture-server)
- [Server (Windows)](https://github.com/ebaek44/hand-tracker-1/releases/download/v1.0.0/gesture-server.exe)
- [Chrome extension (ZIP)](https://github.com/ebaek44/hand-tracker-1/releases/download/v1.0.0/extension.zip)

---

## Quickstart (demo)

### 1) Start the server (back-end)

- **Binary (recommended for demos):**
  - macOS/Linux: download and run `./gesture-server`
  - Windows: download and run `gesture-server.exe`
- **Or from source:**
  ```bash
  python3 server.py
  ```

### 2) Start the chrome extension (front-end)

- Go to chrome://extensions → enable Developer mode

- Click Load unpacked → select the extension/ folder
  (or drag-drop extension.zip from Releases)

### 3) Try It Out!!!

- **Open Youtube or another video streaming platform**
  - Open palm to control pause/play
  - Swipe left/right to scrub 10 sec backward/forward
