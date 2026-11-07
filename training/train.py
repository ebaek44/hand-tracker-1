"""Train/export static poses. Python is needed only for training, never for use."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = [
    {"name": "Open palm", "action": "toggle"},
    {"name": "Closed hand", "action": "none"},
    {"name": "Point up", "action": "none"},
    {"name": "Confirm / neutral", "action": "none"},
    {"name": "Point right", "action": "seek-right"},
    {"name": "Point left", "action": "seek-left"},
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="JSON downloaded from the extension's Training studio")
    parser.add_argument("--legacy-data", action="store_true", help="Merge only old palm/neutral CSV samples (never old left/right examples)")
    parser.add_argument("--output", type=Path, default=ROOT / "training/output")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--install", action="store_true", help="Copy the trained JSON into extension/models/pose-classifier.json")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    labels = DEFAULT_LABELS[:4]
    samples = []
    if args.data:
        dataset = json.loads(args.data.read_text())
        if dataset.get("schemaVersion") != 1:
            raise ValueError("Unsupported dataset version")
        labels = dataset["labels"]
        samples = dataset["samples"]
    if args.legacy_data or not args.data:
        if labels[:4] != DEFAULT_LABELS[:4]:
            raise ValueError("Legacy palm/neutral data requires the original first four label names and actions")
        with (ROOT / "hand_tracker/model/keypoint_classifier/keypoint.csv").open() as source:
            samples += [{"label": int(row[0]), "features": list(map(float, row[1:])), "session": "legacy"} for row in csv.reader(source) if int(row[0]) < 4]
    # Studio exports already contain retained legacy rows. Do not weight them twice.
    samples = list({(s["label"], s.get("session", "legacy"), tuple(s["features"])): s for s in samples}.values())
    if not labels or any(label.get("action") not in ("none", "toggle", "seek-left", "seek-right") for label in labels):
        raise ValueError("Labels need a name and one supported action")
    x = np.asarray([s["features"] for s in samples], dtype=np.float32)
    y = np.asarray([s["label"] for s in samples], dtype=np.int32)
    if x.ndim != 2 or x.shape[1] != 42 or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.001:
        raise ValueError("Expected 42 finite, normalized landmark features per sample")
    if (y < 0).any() or (y >= len(labels)).any():
        raise ValueError("Sample label outside the label list")
    rng = np.random.default_rng(42)
    train, validation = [], []
    counts = []
    grouped = True
    for label in range(len(labels)):
        indices = np.flatnonzero(y == label)
        counts.append(len(indices))
        if len(indices) < 10:
            raise ValueError(f"{labels[label]['name']}: collect at least 10 samples (prefer 100+ across several recordings)")
        groups = sorted(set(samples[i].get("session", "legacy") for i in indices))
        if len(groups) >= 3 and "legacy" not in groups:
            # Hold out complete recording bursts, not neighboring camera frames.
            rng.shuffle(groups)
            held = set(groups[:max(1, round(len(groups) * .2))])
            validation.extend(i for i in indices if samples[i].get("session") in held)
            train.extend(i for i in indices if samples[i].get("session") not in held)
        else:
            grouped = False
            rng.shuffle(indices)
            cut = max(2, round(len(indices) * .2))
            validation.extend(indices[:cut])
            train.extend(indices[cut:])
    train = np.asarray(train)
    validation = np.asarray(validation)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((42,)), tf.keras.layers.Dropout(.15),
        tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Balance classes so the old abundant neutral examples do not drown out directions.
    weights = {i: len(train) / (len(labels) * max(1, np.sum(y[train] == i))) for i in range(len(labels))}
    model.fit(x[train], y[train], validation_data=(x[validation], y[validation]),
              epochs=args.epochs, batch_size=64, class_weight=weights, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=35, restore_best_weights=True)])
    scores = model.predict(x[validation], verbose=0)
    predicted = scores.argmax(axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, guess in zip(y[validation], predicted):
        confusion[actual, guess] += 1
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            layers.append({"weights": kernel.T.tolist(), "bias": bias.tolist(), "activation": layer.activation.__name__})
    output = {"schemaVersion": 1, "labels": labels, "layers": layers,
              "datasetSha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()}
    report = {"samplesPerLabel": dict(zip((l["name"] for l in labels), counts)),
              "validationAccuracy": float(np.mean(predicted == y[validation])),
              "validationSamples": len(validation), "splitByRecording": grouped,
              "confusionMatrix": confusion.tolist(),
              "limitations": "Recorded validation only; not a real-camera generalization guarantee. Collect varied lighting, hand positions, and neutral/transition examples.",
              "warnings": [f"{labels[i]['name']}: only {n} samples" for i, n in enumerate(counts) if n < 100]}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "pose-classifier.keras")
    text = json.dumps(output, separators=(",", ":")) + "\n"
    (args.output / "pose-classifier.json").write_text(text)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    cases = []
    for label in range(len(labels)):
        for index in np.flatnonzero(y[validation] == label)[:16]:
            cases.append({"input": x[validation[index]].tolist(), "scores": scores[index].tolist()})
    (args.output / "parity.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    if args.install:
        (ROOT / "extension/models/pose-classifier.json").write_text(text)
        (ROOT / "extension/models/training-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (ROOT / "tests/fixtures/pose-parity.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2))
    print("Installed. Reload the unpacked extension in chrome://extensions." if args.install else f"Staged model in {args.output}. Review report.json before using --install.")

if __name__ == "__main__":
    main()
