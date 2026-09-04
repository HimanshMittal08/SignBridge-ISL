"""Regression and integration tests for FastAPI live ISL recognition server.

Tests:
1. GET /health -> verify status 200, model_loaded=True, labels count
2. POST /predict -> valid 36x126 landmark sequence -> status 200, return label, confidence, probabilities
3. POST /predict -> invalid sequence length -> status 400
"""

from __future__ import annotations

import unittest
import numpy as np
from fastapi.testclient import TestClient
from server import app, load_model


class TestServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_model()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["num_classes"], 8)
        self.assertIn("HELLO", data["labels"])

    def test_predict_valid_sequence(self):
        dummy_seq = np.random.normal(0, 0.1, (36, 126)).tolist()
        payload = {"landmarks": dummy_seq}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("label", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertIsInstance(data["label"], str)
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertEqual(len(data["probabilities"]), 8)

    def test_predict_invalid_sequence_length(self):
        invalid_seq = np.zeros((10, 126)).tolist()
        payload = {"landmarks": invalid_seq}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
