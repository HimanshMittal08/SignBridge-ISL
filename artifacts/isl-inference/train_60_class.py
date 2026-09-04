"""SignBridge 40-Class HandsGRU Training Script.

Trains a 40-class PyTorch GRU model on (36, 126) landmark sequences directly scanned from processed_60.
Saves:
- models/gru_model.pt
- models/label_map.json
- models/train_report.json
- models/confusion_matrix.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

TARGET_40_LABELS = [
    'HELLO', 'HOW_ARE_YOU', 'GOOD_MORNING', 'GOOD_AFTERNOON', 'GOOD_EVENING',
    'GOOD_NIGHT', 'THANK_YOU', 'PLEASE', 'YES', 'NO', 'OKAY', 'HELP', 'I', 'YOU', 'HE', 'SHE',
    'WE', 'BOY', 'GIRL', 'FRIEND', 'FAMILY', 'FATHER', 'MOTHER', 'BROTHER', 'SISTER', 'EAT',
    'DRINK', 'FOOD', 'WATER', 'TEA', 'HOUSE', 'SCHOOL', 'STUDENT', 'TEACHER', 'DOCTOR',
    'HOSPITAL', 'WHERE', 'WHAT', 'TODAY', 'GO'
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HandsGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 40,
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


def augment_landmarks(landmarks: np.ndarray) -> np.ndarray:
    aug = landmarks.copy().reshape(36, 2, 21, 3)
    # Jitter
    noise = np.random.normal(0, 0.008, size=aug.shape).astype(np.float32)
    mask = (aug != 0)
    aug += noise * mask
    # Scaling
    scale = np.random.uniform(0.95, 1.05)
    aug *= scale
    # Translation
    shift = np.random.uniform(-0.015, 0.015, size=(1, 1, 1, 3)).astype(np.float32)
    shift[..., 2] = 0
    aug += shift * mask
    return aug.reshape(36, 126)


def normalize_sequence(landmarks: np.ndarray) -> np.ndarray:
    lms = landmarks.copy().reshape(36, 2, 21, 3)
    for f in range(36):
        for h in range(2):
            hand = lms[f, h]
            if np.abs(hand).sum() > 0:
                wrist = hand[0].copy()
                hand = hand - wrist
                dists = np.linalg.norm(hand[:, :2], axis=1)
                scale = max(float(dists.max()), 1e-6)
                lms[f, h] = hand / scale
    return lms.reshape(36, 126)


class LandmarkDataset(Dataset):
    def __init__(self, clips: list[tuple[np.ndarray, int]], augment: bool = False, factor: int = 4):
        self.samples = []
        for feats, label in clips:
            flat_feats = normalize_sequence(feats)
            self.samples.append((flat_feats, label))
            if augment:
                for _ in range(factor):
                    aug_feats = augment_landmarks(flat_feats)
                    self.samples.append((aug_feats, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


def load_dataset_direct(script_dir: Path) -> tuple[list[dict], dict[str, int]]:
    proc_dir = script_dir / 'data' / 'processed_60'
    if not proc_dir.exists():
        proc_dir = Path('artifacts/isl-inference/data/processed_60').resolve()

    label_to_idx = {label: i for i, label in enumerate(TARGET_40_LABELS)}


    records = []
    rng = random.Random(42)

    for label in TARGET_40_LABELS:
        lbl_dir = proc_dir / label
        if not lbl_dir.exists():
            continue

        files = sorted(list(lbl_dir.glob('*.npz')))
        valid_files = []
        for f in files:
            if f.stat().st_size > 1000:
                try:
                    with np.load(f, allow_pickle=True) as data:
                        key = 'landmarks' if 'landmarks' in data.files else data.files[0]
                        arr = data[key].astype(np.float32)
                        if arr.reshape(-1).shape[0] == 4536 and np.isfinite(arr).all():
                            valid_files.append((f, arr.reshape(36, 126)))
                except Exception:
                    pass

        # Deterministic split per class (70% train, 15% val, 15% test)
        rng.shuffle(valid_files)
        n = len(valid_files)
        if n == 0:
            continue

        if n == 1:
            train_files = valid_files
            val_files = valid_files
            test_files = valid_files
        elif n == 2:
            train_files = valid_files[:1]
            val_files = valid_files[1:2]
            test_files = valid_files[1:2]
        else:
            n_train = int(round(n * 0.7))
            n_val = int(round(n * 0.15))
            train_files = valid_files[:n_train]
            val_files = valid_files[n_train:n_train + n_val]
            test_files = valid_files[n_train + n_val:]

        for f, arr in train_files:
            records.append({'clip_id': f.stem, 'landmarks': arr, 'label': label, 'split': 'train'})
        for f, arr in val_files:
            records.append({'clip_id': f.stem, 'landmarks': arr, 'label': label, 'split': 'val'})
        for f, arr in test_files:
            records.append({'clip_id': f.stem, 'landmarks': arr, 'label': label, 'split': 'test'})

    return records, label_to_idx



def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(y.tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets) if all_targets else 0.0
    mean_loss = total_loss / len(all_targets) if all_targets else 0.0
    return mean_loss, acc, all_preds, all_targets


def main():
    set_seed(42)
    script_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path('artifacts/isl-inference').resolve()
    models_dir = script_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    records, label_to_idx = load_dataset_direct(script_dir)
    print(f"Loaded {len(records)} total records for {len(label_to_idx)} classes.")


    train_records = [r for r in records if r['split'] == 'train']
    val_records = [r for r in records if r['split'] == 'val']
    test_records = [r for r in records if r['split'] == 'test']

    print(f"Split breakdown -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    train_clips = [(r['landmarks'], label_to_idx[r['label']]) for r in train_records]
    val_clips = [(r['landmarks'], label_to_idx[r['label']]) for r in val_records]
    test_clips = [(r['landmarks'], label_to_idx[r['label']]) for r in test_records]

    # Calculate class weights to handle class imbalance
    class_counts = [sum(1 for _, lbl in train_clips if lbl == i) for i in range(len(label_to_idx))]
    max_count = max(max(class_counts), 1)
    weights = torch.tensor([max_count / max(c, 1) for c in class_counts], dtype=torch.float32)

    train_ds = LandmarkDataset(train_clips, augment=True, factor=5)
    val_ds = LandmarkDataset(val_clips, augment=False)
    test_ds = LandmarkDataset(test_clips, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    num_classes = len(label_to_idx)
    model = HandsGRU(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)

    epochs = 120
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_state = None

    print(f"Starting training for {epochs} epochs across {num_classes} classes...")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% (Best Val Acc: {best_val_acc*100:.2f}%)")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on held-out test split
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion)

    target_names = [k for k, v in sorted(label_to_idx.items(), key=lambda x: x[1])]
    labels_indices = list(range(num_classes))
    report = classification_report(test_targets, test_preds, labels=labels_indices, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=labels_indices)

    # Save model weights & label map
    model_path = script_dir / "models" / "gru_model.pt"
    label_map_path = script_dir / "models" / "label_map.json"
    report_path = script_dir / "models" / "train_report.json"
    cm_path = script_dir / "models" / "confusion_matrix.json"

    root_models_dir = Path("models")
    root_models_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), root_models_dir / "gru_model.pt")
    print(f"Saved {len(model.state_dict().keys())} keys to {model_path}")
    
    lm_json = json.dumps(label_to_idx, indent=2)
    label_map_path.write_text(lm_json, encoding="utf-8")
    (root_models_dir / "label_map.json").write_text(lm_json, encoding="utf-8")

    cm_path.write_text(json.dumps(cm.tolist(), indent=2), encoding="utf-8")


    # Find worst performing classes on test set
    per_class_f1 = {cls_name: report[cls_name]['f1-score'] for cls_name in target_names if cls_name in report}
    worst_classes = sorted(per_class_f1.items(), key=lambda x: x[1])[:5]

    report_data = {
        "num_classes": num_classes,
        "total_clips": len(records),
        "train_clips": len(train_records),
        "val_clips": len(val_records),
        "test_clips": len(test_records),
        "train_augmented_samples": len(train_ds),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "worst_performing_classes": [{"class": cls, "f1_score": round(score, 4)} for cls, score in worst_classes],
        "classification_report": report,
        "confusion_matrix_file": str(cm_path).replace("\\", "/"),
    }

    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("40-CLASS TRAINING COMPLETE & REPORT SUMMARY")
    print("=" * 60)
    print(f"Total Clips: {len(records)} | Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    print(f"Validation Accuracy: {best_val_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Macro F1 Score: {report['macro avg']['f1-score']:.4f} | Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print("\nWorst 5 Performing Classes on Test Set:")
    for cls, score in worst_classes:
        print(f"  - {cls:20s}: F1 = {score:.4f}")
    print("=" * 60)
    print(f"Saved model to: {model_path}")
    print(f"Saved label map to: {label_map_path}")
    print(f"Saved report to: {report_path}")

if __name__ == '__main__':
    main()
