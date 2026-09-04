# PROJECT_STATUS

## 1. CURRENT ARCHITECTURE
- **frontend path + stack**: `artifacts/signbridge` (React 18 + TypeScript + Vite + Tailwind CSS + MediaPipe Tasks Vision + Wouter)
- **backend path + stack**: `artifacts/isl-inference` (FastAPI + PyTorch GRU + Pydantic + Uvicorn)
- **model file path**: `artifacts/isl-inference/models/gru_model.pt`
- **inference flow in one line**: Web camera frames -> MediaPipe HandLandmarker (in browser) -> 36x126 landmark array POSTed to `/predict` -> PyTorch GRU model predicts sign label + confidence score.

## 2. CURRENT SIGN SUPPORT
- **exact number of signs currently supported by the trained model**: 60
- **exact sign labels**: `BANK`, `BOY`, `BROTHER`, `BUS`, `CAR`, `CITY`, `COLD`, `DOCTOR`, `DRINK`, `EAT`, `FAMILY`, `FATHER`, `FOOD`, `FRIEND`, `GIRL`, `GO`, `GOOD_AFTERNOON`, `GOOD_EVENING`, `GOOD_MORNING`, `GOOD_NIGHT`, `HAPPY`, `HE`, `HELLO`, `HELP`, `HOSPITAL`, `HOUSE`, `HOW_ARE_YOU`, `I`, `INDIA`, `LIBRARY`, `LOCATION`, `MARKET`, `MOTHER`, `NO`, `OFFICE`, `OKAY`, `PARK`, `PLEASE`, `POLICE`, `RESTAURANT`, `SCHOOL`, `SHE`, `SICK`, `SISTER`, `SIT`, `STORE_OR_SHOP`, `STUDENT`, `TEA`, `TEACHER`, `THANK_YOU`, `TIME`, `TODAY`, `TRAIN`, `TRAIN_STATION`, `WATER`, `WE`, `WHAT`, `WHERE`, `YES`, `YOU`
- **where label_map is located**: `artifacts/isl-inference/models/label_map.json` (and `backup_8class/label_map.json` for 8-class backup)
- **whether frontend vocabulary contains additional signs not supported by the model**: No. Both frontend vocabulary (`concepts` in `App.tsx`) and backend model currently match the same 60 sign labels.

## 3. DATASET STATUS
- **dataset/landmark directories that currently exist**:
  - `artifacts/isl-inference/data/processed_60` (contains 60 class subdirectories with `.npz` landmark files)
- **sample count per sign if cheaply available**: 5678 total samples across 60 signs (varies per class, tracked in `artifacts/isl-inference/data/dataset_manifest.json`)
- **total unique signs with usable training data**: 60
- **any partially processed datasets**: `artifacts/isl-inference/data/dataset_report.json` documents an earlier 8-class subset report; currently full 60-class dataset has been landmark-extracted into `processed_60`.

## 4. 60-SIGN EXPANSION STATUS
- **list/count of signs already prepared beyond the original 8**: 52 signs prepared (`BANK`, `BOY`, `BROTHER`, `BUS`, `CAR`, `CITY`, `COLD`, `DOCTOR`, `DRINK`, `FAMILY`, `FATHER`, `FOOD`, `FRIEND`, `GIRL`, `GOOD_AFTERNOON`, `GOOD_EVENING`, `GOOD_MORNING`, `GOOD_NIGHT`, `HAPPY`, `HE`, `HOSPITAL`, `HOUSE`, `HOW_ARE_YOU`, `I`, `INDIA`, `LIBRARY`, `LOCATION`, `MARKET`, `MOTHER`, `OFFICE`, `OKAY`, `PARK`, `POLICE`, `RESTAURANT`, `SCHOOL`, `SHE`, `SICK`, `SISTER`, `SIT`, `STORE_OR_SHOP`, `STUDENT`, `TEA`, `TEACHER`, `THANK_YOU`, `TIME`, `TODAY`, `TRAIN`, `TRAIN_STATION`, `WE`, `WHAT`, `WHERE`, `YOU`). Total prepared: 60.
- **list/count of signs still missing**: 0 missing from the 60-class target set.
- **any existing scripts/manifests/status files related to expansion**:
  - `artifacts/isl-inference/build_60_dataset.py`
  - `artifacts/isl-inference/download_60_dataset.py`
  - `artifacts/isl-inference/extract_60_landmarks.py`
  - `artifacts/isl-inference/process_60_dataset.py`
  - `artifacts/isl-inference/process_missing_60_npzs.py`
  - `artifacts/isl-inference/train_60_class.py`
  - `artifacts/isl-inference/data/dataset_manifest.json`

## 5. INTEGRATION STATUS
- **current frontend API endpoint**: `http://localhost:8000/predict` (and `http://localhost:8000/health` check)
- **current backend endpoints**: GET `/health`, POST `/predict`
- **whether the current model and frontend label set match**: Yes, both support the 60 sign labels.

## 6. SAFE NEXT STEP
- **ONE smallest next action only**: Run end-to-end verification of `/predict` inference by sending a test payload to `artifacts/isl-inference/server.py` or executing `artifacts/isl-inference/test_server.py`.
