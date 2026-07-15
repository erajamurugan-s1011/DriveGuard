import torch
import numpy as np
import os
import sys
import time
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.hybrid_cnn_quantum import HybridCNNQuantumModel, SUBSYSTEM_CONFIG

DATA_DIR = "data/processed"
CHECKPOINT_DIR = "models"
OUT_DIR = "evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

HELD_OUT_BATTERY = "B0018"

with open(os.path.join(CHECKPOINT_DIR, "label_encoders.pkl"), "rb") as f:
    LABEL_ENCODERS = pickle.load(f)
with open(os.path.join(CHECKPOINT_DIR, "scalers.pkl"), "rb") as f:
    SCALERS = pickle.load(f)


def load_val_split(name):
    X = np.load(os.path.join(DATA_DIR, name, "X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, name, "y.npy"))
    le = LABEL_ENCODERS[name]
    y = le.transform(y_raw)

    if name == "battery":
        sources = np.load(os.path.join(DATA_DIR, name, "source_files.npy"))
        val_mask = sources == HELD_OUT_BATTERY
        X_val, y_val = X[val_mask], y[val_mask]
        X_val = SCALERS["battery"].transform(X_val).astype(np.float32)
        return X_val, y_val

    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    else:
        X = X.transpose(0, 2, 1)
    X = X.astype(np.float32)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_val, y_val


def measure_latency(model, x_single, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(x_single)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(x_single)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per inference


report_lines = ["# DriveGuard — Evaluation Report\n"]
summary_rows = []

for name, cfg in SUBSYSTEM_CONFIG.items():
    print(f"\n{'=' * 50}\nEvaluating: {name}\n{'=' * 50}")

    model = HybridCNNQuantumModel(**cfg)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model_{name}.pt", map_location="cpu"))
    model.eval()

    X_val, y_val = load_val_split(name)
    le = LABEL_ENCODERS[name]
    class_names = list(le.classes_)

    x_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        preds = logits.argmax(dim=1).numpy()

    cm = confusion_matrix(y_val, preds)
    report_dict = classification_report(y_val, preds, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_val, preds, target_names=class_names, zero_division=0)
    print(report_text)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name.capitalize()} — Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, f"confusion_matrix_{name}.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"  Saved {cm_path}")

    if cfg["encoder_type"] == "cnn":
        single_input = x_tensor[0:1]
    else:
        single_input = x_tensor[0:1]
    latency_ms = measure_latency(model, single_input)
    print(f"  Avg inference latency: {latency_ms:.2f} ms (CPU, single sample, 50-run average)")

    accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    summary_rows.append((name, len(class_names), len(y_val), accuracy, macro_f1, latency_ms))

    report_lines.append(f"\n## {name.capitalize()}\n")
    report_lines.append(f"- Validation samples: {len(y_val)}")
    report_lines.append(f"- Classes: {', '.join(class_names)}")
    report_lines.append(f"- Accuracy: {accuracy:.4f}")
    report_lines.append(f"- Macro F1: {macro_f1:.4f}")
    report_lines.append(f"- Avg inference latency (CPU): {latency_ms:.2f} ms")
    report_lines.append(f"\n![Confusion Matrix]({os.path.basename(cm_path)})\n")
    report_lines.append("```")
    report_lines.append(report_text)
    report_lines.append("```")

report_lines.insert(1, "\n## Summary\n")
report_lines.insert(2, "| Subsystem | Classes | Val samples | Accuracy | Macro F1 | Latency (CPU, ms) |")
report_lines.insert(3, "|---|---|---|---|---|---|")
for i, (name, n_classes, n_val, acc, f1, lat) in enumerate(summary_rows):
    report_lines.insert(4 + i, f"| {name.capitalize()} | {n_classes} | {n_val} | {acc:.4f} | {f1:.4f} | {lat:.2f} |")

with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
    f.write("\n".join(report_lines))

print(f"\n{'=' * 50}\nFull report saved to {OUT_DIR}/REPORT.md\n{'=' * 50}")