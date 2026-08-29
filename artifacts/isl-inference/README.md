# SignBridge hands-only ISL data pipeline

This directory is deliberately isolated from the React application.  It collects
real, labelled MediaPipe Hands landmark clips for a future, limited-vocabulary
temporal classifier.  It does not contain a model, trained weights, or a
prediction endpoint yet.

## Scope and guardrails

- Hands only: MediaPipe Hands with up to two hands.  No pose, face, or BlazePose
  dependency is used.
- Every recorded clip is 36 frames of two 21-point `x, y, z` hand slots.
- A missing hand is represented by zeros.  Handedness determines the slot, so
  left and right remain stable in the recorded feature tensor.
- Coordinates are wrist-relative and scaled by the wrist-to-middle-MCP distance.
- Labels are supplied by the recorder.  A label is not a supported SignBridge
  feature until clips have been collected, a model has been trained, and that
  model has been evaluated.

## Prerequisites

Use Python 3.10 or 3.11 in a virtual environment.  The supplied desktop host
does not currently include Python, so install it before running these commands.

```powershell
cd artifacts/isl-inference
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-collect.txt
```

## Record a clip

Record one labelled clip with a stable signer/session identifier:

```powershell
python collect_landmarks.py --label HELLO --signer signer-01 --session 2026-08-29-am
```

The recorder shows a 3-second countdown, then captures exactly 36 frames for
which at least one hand is detected.  Press `r` to record another clip with the
same label/signer/session, or `q` to quit.  Press `Esc` before capture to quit.

The result is saved beneath `data/raw/<LABEL>/<SIGNER>/` as an `.npz` clip and a
sidecar `.json` metadata file.  The feature array shape is `(36, 2, 21, 3)`.

## Collection protocol

Start with these candidate labels only after confirming the signer knows the
intended ISL form: `HELLO`, `THANK_YOU`, `HELP`, `WATER`, `ME`, and `YOU`.

Collect **at least 80 usable clips per label per signer**, with **at least 4
signers** (target: 1,920 clips total).  Record at least two sessions per signer
on different days or with changed lighting/clothing/background.  Vary distance,
speed, hand dominance, and natural rest-to-sign-to-rest movement; do not record
near-duplicate takes consecutively.

Keep each signer in a separate `--signer` group.  The future trainer will split
by signer, never individual frames or adjacent clips, so the test signer has no
clips in training.  Discard clips with tracking failure, an incorrect sign, or
an incomplete movement.  Do not synthesize or augment landmark clips until a
real baseline has been measured.

## Live FastAPI Inference Service

Start the live ISL inference server loading `models/gru_model.pt`:

```powershell
cd artifacts/isl-inference
.\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
```

The server provides:
- `GET /health`: Health status & loaded labels (`EAT`, `GO`, `HELLO`, `HELP`, `NO`, `PLEASE`, `WATER`, `YES`).
- `POST /predict`: Predicts sign label & confidence for a 36-frame `(36, 126)` landmark sequence.

Run server tests:
```powershell
.\.venv\Scripts\python.exe test_server.py
```

## Running the Complete Prototype (Windows)

1. **Terminal 1 — FastAPI Backend (Port 8000)**:
   ```powershell
   cd artifacts/isl-inference
   .\.venv\Scripts\python.exe -m uvicorn server:app --port 8000
   ```

2. **Terminal 2 — React Frontend (Vite)**:
   ```powershell
   cd artifacts/signbridge
   powershell -ExecutionPolicy Bypass -Command "pnpm run dev"
   ```

