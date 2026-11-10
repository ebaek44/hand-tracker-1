# Train a static gesture

The extension is self contained. Python is only needed when you choose to train a new classifier.

## 1. Collect in Chrome

Load the local `extension/` folder at `chrome://extensions`, open **Hands On Video Controls (Studio)**, and select **Training studio**. Connect the camera (this stops live video control while collecting). Choose a pose and hold it throughout a three-second batch. Relax and reposition between batches; do not switch poses during a batch. The mirrored preview shows the same landmarks the classifier receives.

The studio preloads 4,639 archived palm/neutral examples. **Left/right start with zero carried examples.** After an explicit baseline reset, the original three-class palm/neutral classifier is active until you install new directional training. The current local extension already has the model trained from the user’s exported recordings installed; reload it to use those directions.

Default labels and actions:

| Pose              | Action                                                         |
| ----------------- | -------------------------------------------------------------- |
| Open palm         | Toggle play/pause once after a 0.4-second hold                 |
| Closed hand       | None                                                           |
| Point up          | None                                                           |
| Confirm / neutral | None                                                           |
| Point right       | Seek +10 seconds after a 0.35-second hold; repeat every second |
| Point left        | Seek −10 seconds after a 0.35-second hold; repeat every second |

Use **Add another pose** to assign an additional pose to any of the available actions. Use **Remove selected empty pose** before collecting if you want a smaller pose list. These are model classes, not hard-coded landmark heuristics. Pinch/drag is a separate future interaction and is not implemented by adding a static label.

**Start with 10 three-second batches for Point right and 10 for Point left**, roughly 300 samples each at the studio’s 10 FPS target (fewer on slower machines). This is a starting collection plan, not a reliability guarantee. Change hand position, distance, and slight angle between batches, and include a second lighting condition. The palm/neutral classes already have examples; add 5–10 batches of your own relaxed/closed hand if ordinary movements cause false triggers.

Neutral means a visible hand that is not issuing a command: relaxed/curled fingers, a loose fist, and resting positions. Use Closed hand for relaxed/closed examples. Point up is now reserved for the YouTube Skip ad action; it is ignored when no supported Skip ad button is available. Do not label deliberate open-palm or left/right poses as neutral. A missing hand is handled directly and needs no training samples. Each listed class needs at least 10 examples to train, including any custom empty classes you added.

The studio stores **42 normalized x/y landmark features**, labels, and a recording ID in IndexedDB. It does not store camera images. Dataset state survives reloading the studio; removing the extension removes its local data. Export a JSON backup. Import merges matching label lists and skips duplicate rows. Export/import are available while the camera is off.

## 2. Set up a training environment

From the project folder, use the existing `.venv` if it works, or create one with Python 3.10 or 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`. No Flask, OpenCV, or Socket.IO is needed for this workflow.

## 3. Train and inspect

```bash
.venv/bin/python training/train.py --data /absolute/path/to/hands-on-dataset.json
```

The model is a small 42 → 32 → 16 → class-count dense network with dropout and class-balanced training. It stops early on validation loss and restores the best weights. Outputs go to ignored `training/output/`:

- `pose-classifier.keras`: retraining checkpoint.
- `pose-classifier.json`: weights, class names, and action mapping used by JavaScript.
- `report.json`: sample counts, validation accuracy, confusion matrix, split method, and warnings.
- `parity.json`: Keras predictions for checking the JavaScript implementation.

When each class has at least three recordings, validation holds out whole recordings. For classes with archived data or fewer recordings, it falls back to a stratified row split and reports `splitByRecording: false` overall. Directional classes with sufficient new recordings still use a recording-group split. Adjacent recorded frames are correlated, so row-split accuracy can be optimistic. Inspect individual classes, then test live with movements and people not used during collection. The report is validation, not an independent test-set benchmark.

## 4. Install locally

After reviewing the report, train with `--install` to update the bundled model and parity fixtures:

```bash
.venv/bin/python training/train.py --data ~/Downloads/hands-on-dataset.json --install
npm test
```

Reload the unpacked extension at `chrome://extensions`. Start tracking and try play/pause, both hold directions, release, neutral motion, and leaving the camera. Test on a paused video first. Stop immediately if false actions occur, collect more neutral/directional examples, and retrain.

To optionally retrain only the retained palm/neutral classes from the old CSV (no directional examples):

```bash
.venv/bin/python training/train.py --install
```

Studio exports already include the archived palm/neutral examples, so `--legacy-data` is normally unnecessary. If needed, it merges **only labels 0–3**, deduplicates identical rows, and requires the first four labels to match. The old **16 right / 62 left** rows remain in the untouched original CSV for history but are excluded from the seed and all legacy-data training paths. The original TFLite/Keras files and notebooks remain untouched.

`npm run reset:training-baseline` explicitly restores the original three-class classifier and refreshes the palm/neutral seed; it is never run automatically during builds. Fresh directional recordings in the studio are preserved. `scripts/export-classifiers.py` exports original dense weights and TensorFlow Lite parity cases for reference.

## Interaction tuning

`extension/lib/gestures.js` contains dwell, release, and repeat timing. `extension/tracking-worker.js` applies a 0.80 classifier confidence threshold. Timing can suppress brief accidental poses; it cannot fix missing training coverage or an overconfident classifier. Do not add overlapping cooldown locks to compensate for bad labels. Collect better examples and use the confusion matrix first.

## Hide training for users

Keep training in the `extension/` Studio build. `npm run build:release` creates `dist/hands-on/` without the studio, datasets, reports, training links, or diagnostic UI. The installed trained model is copied into that build. Use **Options → Training studio** to opt in or out in the working extension. It starts off; turning it off stops an open Studio camera and hides its view without deleting recordings. `trainingEnabled` in `extension/config.js` remains the build capability flag, and the release builder always disables it. See `PUBLISHING.md` for sharing and Store submission. Older stored/imported recordings automatically gain the requested Point up → Skip ad action mapping without changing their sample features or discarding recordings.
