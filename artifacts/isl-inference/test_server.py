"""Test script for SignBridge FastAPI server using real landmark data from data/test.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from server import app


def test_fastapi_server():
    with TestClient(app) as client:
        # 1. Test /health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        health_data = health_resp.json()
        print("Health response:", health_data)
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert health_data["num_classes"] == 8

        # 2. Test /predict with real .npz clip
        base_dir = Path(__file__).resolve().parent
        sample_npz = base_dir / "data" / "test" / "HELLO" / "USER001" / "isl500_hello__ISL500__00087__Hello__session101__clip021.npz"
        assert sample_npz.exists(), f"Sample npz file not found at {sample_npz}"

        with np.load(sample_npz, allow_pickle=True) as data:
            landmarks = data["landmarks"].astype(np.float32)  # Shape: (36, 2, 21, 3)
            meta = json.loads(str(data["metadata"]))

        # Flatten to (36, 126)
        sequence_flat = landmarks.reshape(36, 126).tolist()

        pred_resp = client.post("/predict", json={"landmarks": sequence_flat})
        assert pred_resp.status_code == 200, f"Prediction failed: {pred_resp.text}"
        pred_data = pred_resp.json()
        print("\nReal sample prediction response:")
        print(f"Ground Truth Label: {meta['label']}")
        print(f"Predicted Label   : {pred_data['label']}")
        print(f"Confidence        : {pred_data['confidence']}")
        print(f"Probabilities     : {pred_data['probabilities']}")

        assert "label" in pred_data
        assert "confidence" in pred_data
        assert "probabilities" in pred_data
        print("\nAll FastAPI tests passed successfully!")


if __name__ == "__main__":
    test_fastapi_server()
