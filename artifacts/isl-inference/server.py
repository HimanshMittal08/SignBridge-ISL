"""FastAPI live ISL recognition server for SignBridge.

Serves PyTorch GRU model predictions on (36, 126) landmark sequences.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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


class PredictRequest(BaseModel):
    landmarks: List[List[float]] = Field(
        ...,
        description="36x126 list of floats representing normalized hand landmarks across 36 frames."
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


# Global model state
MODEL: HandsGRU | None = None
LABEL_MAP: dict[str, int] = {}
IDX_TO_LABEL: dict[int, str] = {}


def load_model():
    global MODEL, LABEL_MAP, IDX_TO_LABEL
    if MODEL is not None:
        return

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "gru_model.pt"
    label_map_path = base_dir / "models" / "label_map.json"

    if not model_path.exists():
        raise RuntimeError(f"Model checkpoint not found at {model_path}")
    if not label_map_path.exists():
        raise RuntimeError(f"Label map not found at {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        LABEL_MAP = json.load(f)

    IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
    num_classes = len(LABEL_MAP)

    model = HandsGRU(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    MODEL = model
    print(f"Successfully loaded GRU model from {model_path} with {num_classes} classes: {list(LABEL_MAP.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="SignBridge ISL Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if MODEL is None:
        try:
            load_model()
        except Exception:
            pass
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "num_classes": len(LABEL_MAP),
        "labels": list(LABEL_MAP.keys()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if MODEL is None:
        load_model()

    seq = payload.landmarks
    if len(seq) != 36:
        raise HTTPException(
            status_code=400,
            detail=f"Expected sequence length of 36 frames, got {len(seq)}"
        )

    for i, frame in enumerate(seq):
        if len(frame) != 126:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {i} expected 126 float values, got {len(frame)}"
            )

    try:
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 36, 126)
        with torch.no_grad():
            logits = MODEL(x_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].item())
        label = IDX_TO_LABEL[int(pred_idx)]

        probs_dict = {
            IDX_TO_LABEL[i]: round(float(probs[i].item()), 4)
            for i in range(len(IDX_TO_LABEL))
        }

        return PredictResponse(
            label=label,
            confidence=round(confidence, 4),
            probabilities=probs_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
