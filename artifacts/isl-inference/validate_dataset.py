"""Dataset quality validation script for SignBridge.

Validates that all .npz landmark clips meet tensor shape, value integrity,
and metadata completeness criteria.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def main() -> int:
    data_dir = Path(__file__).resolve().parent / "data" / "test"
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} does not exist.")
        return 1

    npz_files = list(data_dir.rglob("*.npz"))
    print(f"Scanning {len(npz_files)} .npz files in {data_dir}...")

    errors = []
    signers = set()
    labels = set()

    for path in npz_files:
        try:
            with np.load(path, allow_pickle=True) as data:
                if "landmarks" not in data or "metadata" not in data:
                    errors.append(f"{path.name}: Missing 'landmarks' or 'metadata' key")
                    continue

                landmarks = data["landmarks"]
                meta = json.loads(str(data["metadata"]))

                # Check shape: expect (36, 126) or (36, 2, 21, 3)
                flat_shape = landmarks.reshape(landmarks.shape[0], -1).shape
                if flat_shape != (36, 126):
                    errors.append(f"{path.name}: Expected shape (36, 126), got {flat_shape}")

                # Check NaN / Inf
                if np.isnan(landmarks).any() or np.isinf(landmarks).any():
                    errors.append(f"{path.name}: Contains NaN or Inf values")

                # Check empty
                if np.all(landmarks == 0):
                    errors.append(f"{path.name}: Sequence is all zeros")

                label = meta.get("label")
                signer = meta.get("signer_id")
                if label:
                    labels.add(label)
                if signer:
                    signers.add(signer)

        except Exception as e:
            errors.append(f"{path.name}: Failed to load file ({e})")

    print("\n" + "=" * 50)
    print("DATASET VALIDATION RESULTS")
    print("=" * 50)
    print(f"Total .npz clips checked: {len(npz_files)}")
    print(f"Unique classes found ({len(labels)}): {sorted(list(labels))}")
    print(f"Unique signers found ({len(signers)}): {sorted(list(signers))}")

    if errors:
        print(f"\nFound {len(errors)} validation errors:")
        for err in errors:
            print(f"  ERROR: {err}")
        return 1

    print("\n[OK] All dataset clips passed validation clean (shape=36x126, no NaNs, non-empty)!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
