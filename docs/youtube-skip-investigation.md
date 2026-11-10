# YouTube Skip ad investigation — November 9, 2026

The upstream README and current public YouTube player source were read directly.

- [Upstream README](https://github.com/xhy0405/youtube-auto-skip/blob/main/README.md) documents the skip-button selectors and synthetic mouse sequence. It does not demonstrate reliable operation against every current player.
- [YouTube player asset inspected](https://www.youtube.com/s/player/a5c2aafe/player_ias.vflset/en_US/base.js), obtained from YouTube's public watch-page script reference. This is the bundle served to our anonymous request, not proof that every account receives the same experiment.
- That bundle still creates `ytp-skip-ad-button`, `ytp-ad-skip-button`, `ytp-ad-skip-button-modern`, and the skip-button-slot/container variants. Renaming the selectors was not the underlying browser-input limitation.
- Its `Mta` check returns a failing result for an event with `isTrusted === false`. Both inspected skip-button handlers invoke the abnormality path on that result and run their normal skip action only otherwise.
- [MDN confirms](https://developer.mozilla.org/en-US/docs/Web/API/Event/isTrusted) that `.click()` and dispatched events are untrusted.

Version 2.1.0 therefore uses default-off, user-enabled [Chrome debugger input](https://developer.chrome.com/docs/extensions/reference/api/debugger) for ad skipping. Automated Chromium tests verify the extension produces `isTrusted === true` and passes a handler reproducing that guard. The target must already be visible; nothing is queued. Live ads on the user's account remain a separate manual check.

Chrome’s [permissions documentation](https://developer.chrome.com/docs/extensions/reference/api/permissions#implement_optional_permissions) explicitly excludes `debugger` from optional permissions. It must be declared in the installation manifest. The popup opt-in gates its use; turning it off detaches any session, but cannot remove this required manifest permission.

November 10 follow-up: the user confirmed debugger-mode ad skipping works on live YouTube. Version 2.1.1 removes the Studio manual test button and route; ad skipping now enters only through recognized gestures with the default-off option enabled.
