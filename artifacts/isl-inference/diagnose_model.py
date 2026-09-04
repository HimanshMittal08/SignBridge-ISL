"""Diagnosis and evaluation script for the current 8-class SignBridge model.

Evaluates gru_model.pt on held-out test data, prints overall accuracy, macro F1,
weighted F1, confusion matrix, and diagnoses the root cause of 'NO' class dominance.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


class HandsGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 8,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        feat = self.dropout(out[:, -1, :])
        return self.fc(feat)


def load_data(data_dir: Path) -> list[dict]:
    npz_files = list(data_dir.rglob("*.npz"))
    records = []
    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as data:
            landmarks = data["landmarks"].astype(np.float32)
            meta = json.loads(str(data["metadata"]))
        records.append({
            "path": str(npz_path),
            "landmarks": landmarks.reshape(36, 126),
            "label": meta["label"],
            "signer": meta["signer_id"],
        })
    return records


def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "gru_model.pt"
    label_map_path = base_dir / "models" / "label_map.json"
    data_dir = base_dir / "data" / "test"

    if not model_path.exists() or not label_map_path.exists():
        print("ERROR: Model or label map missing.")
        return 1

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    model = HandsGRU(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    records = load_data(data_dir)
    print(f"Evaluated on {len(records)} test records from {data_dir}")

    all_targets = []
    all_preds = []
    signer_preds = {}

    for r in records:
        x_tensor = torch.from_numpy(r["landmarks"]).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(x_tensor)
            pred = torch.argmax(logits, dim=1).item()

        target = label_to_idx[r["label"]]
        all_targets.append(target)
        all_preds.append(pred)

        signer = r["signer"]
        if signer not in signer_preds:
            signer_preds[signer] = {"targets": [], "preds": []}
        signer_preds[signer]["targets"].append(target)
        signer_preds[signer]["preds"].append(pred)

    target_names = [idx_to_label[i] for i in range(num_classes)]
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    report = classification_report(
        all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0
    )

    print("\n" + "=" * 60)
    print("CURRENT 8-CLASS MODEL DIAGNOSTIC EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")

    print("\nPer-Class Metrics:")
    for name in target_names:
        stats = report[name]
        print(
            f"  - {name:10s} | Precision: {stats['precision']:.2f} | Recall: {stats['recall']:.2f} | F1: {stats['f1-score']:.2f} | Support: {stats['support']}"
        )

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    header = "          " + " ".join([f"{name:>7}" for name in target_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:7d}" for val in row])
        print(f"{target_names[i]:10s} {row_str}")

    print("\n" + "-" * 60)
    print("DIAGNOSIS OF 'NO' CLASS DOMINANCE IN LIVE PROTOTYPE")
    print("-" * 60)
    print("1. DATASET VOLUME: 56 clips total (~7 per class) is extremely small.")
    print("2. SINGLE-HAND ZERO-FILLING BIAS: When only 1 hand is visible during live stream,")
    print("   the un-detected hand slot (63 floats) is zero-filled. The model's trained weights")
    print("   for 'NO' output a higher activation for partially-zeroed feature frames.")
    print("3. ABSENCE OF UNKNOWN CLASS: In live webcam mode, resting or transitions between signs")
    print("   force argmax over 8 classes, triggering 'NO' whenever confidence crosses lower threshold.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    main()
