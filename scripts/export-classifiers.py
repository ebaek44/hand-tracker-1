"""Development only: export the actual TFLite dense weights, plus parity cases.

Run with the existing training environment: .venv/bin/python scripts/export-classifiers.py
No TensorFlow or Python is needed by the extension or its users.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
models = {}
fixtures = {}
for name in ("keypoint", "point_history"):
    source = ROOT / f"hand_tracker/model/{name}_classifier/{name}_classifier.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(source), num_threads=1)
    interpreter.allocate_tensors()
    layers = []
    for op in interpreter._get_ops_details():
        if op["op_name"] == "FULLY_CONNECTED":
            _, weights, bias = op["inputs"]
            layers.append({
                "weights": interpreter.get_tensor(weights).tolist(),
                "bias": interpreter.get_tensor(bias).tolist(),
                "activation": "relu",
            })
        elif op["op_name"] not in ("SOFTMAX", "DELEGATE"):
            raise ValueError(f"Unsupported operation: {op['op_name']}")
    layers[-1]["activation"] = "softmax"
    labels_file = source.with_name(f"{name}_classifier_label.csv")
    labels = [row[0] for row in csv.reader(labels_file.open(encoding="utf-8-sig"))]
    outputs = interpreter.get_output_details()[0]
    models[name] = {
        "source": str(source.relative_to(ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "labels": labels[:int(outputs["shape"][-1])],
        "layers": layers,
    }
    # Stratified recorded training features + deterministic synthetic inputs.
    data = np.loadtxt(source.with_name(f"{name}.csv"), delimiter=",", dtype=np.float32)
    inputs = []
    for label in np.unique(data[:, 0]):
        rows = data[data[:, 0] == label, 1:]
        inputs.extend(rows[::max(1, len(rows) // 12)][:12])
    rng = np.random.default_rng(2026)
    inputs.extend(rng.uniform(-1, 1, size=(20, interpreter.get_input_details()[0]["shape"][-1])).astype(np.float32))
    cases = []
    for features in inputs:
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], features[None, :])
        interpreter.invoke()
        cases.append({"input": features.tolist(), "scores": interpreter.get_tensor(outputs["index"])[0].tolist()})
    fixtures[name] = cases

(ROOT / "extension/models/classifiers.json").write_text(json.dumps(models, separators=(",", ":")) + "\n")
(ROOT / "tests/fixtures/classifier-parity.json").write_text(json.dumps(fixtures, separators=(",", ":")) + "\n")
print("Exported original TFLite weights and", sum(map(len, fixtures.values())), "parity cases.")
