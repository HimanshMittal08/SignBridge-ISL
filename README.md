# SignBridge (ISL) ????

**SignBridge** is a real-time, two-way Indian Sign Language (ISL) communication bridge designed to enable seamless bidirectional interaction between Deaf / Hard-of-Hearing individuals and non-signers.

---

## ?? Overview

SignBridge addresses the communication gap by providing real-time, two-way translation:
1. **Sign to Text / Speech (Direction 1)**: Captures hand gestures live via camera feed, processes frame sequence landmarks, and predicts the corresponding ISL sign concept using a deep learning classifier.
2. **Text / Speech to Sign (Direction 2)**: Accepts spoken or typed input, extracts key concepts, and visualizes the corresponding ISL sign via synchronized video playback.

This repository contains an **MVP research prototype** operating on a curated 40-concept ISL vocabulary dataset.

---

## ? Key Features

- **Real-Time Hand Landmark Tracking**: Uses MediaPipe Tasks Vision directly in the browser to extract 21 3D hand keypoints per hand across two hands (126 coordinates per frame).
- **Temporal Deep Learning Inference**: 36-frame temporal sequence modeling powered by a PyTorch GRU architecture (HandsGRU).
- **Bidirectional Communication**:
  - **Sign-to-Text**: Real-time sign classification with live probability distribution and confidence metrics.
  - **Speech-to-Text Integration**: Web Speech API integration for instant voice input capture.
  - **Text-to-Speech Output**: Integrated browser speech synthesis for audio readout of translated signs.
  - **Sign Video Renderer**: High-definition H.264 video playback for visual sign demonstration.
- **Modern Responsive UI**: Built with React, TypeScript, Vite, Tailwind CSS, and Lucide icons.

---

## ?? System Architecture

`
                                  DIRECTION 1: SIGN TO TEXT/SPEECH
+----------------------+    +---------------------------------+    +--------------------------------+
¦   Webcam Feed        ¦ -? ¦ MediaPipe Hand Landmarker       ¦ -? ¦ 36-Frame Landmark Sequence     ¦
¦  (User Gesturing)    ¦    ¦ (Browser - 126 coordinates/frame¦    ¦ (Shape: 36 x 126 array)        ¦
+----------------------+    +---------------------------------+    +--------------------------------+
                                                                                   ¦ POST /predict
                                                                                   ?
+----------------------+    +---------------------------------+    +--------------------------------+
¦ UI Display & Speech  ¦ ?- ¦ Concept & Confidence Score      ¦ ?- ¦ FastAPI Backend + PyTorch GRU  ¦
¦ (Web Speech Synthesis¦    ¦ (e.g. " THANK_YOU\, 94.2% conf) ¦ ¦ (40-Class HandsGRU Model) ¦
+----------------------+ +---------------------------------+ +--------------------------------+

 DIRECTION 2: SPEECH/TEXT TO SIGN
+----------------------+ +---------------------------------+ +--------------------------------+
¦ Voice / Text Input ¦ -? ¦ Concept Extraction Engine ¦ -? ¦ ISL Video Player Component ¦
¦ (Web Speech Recog.) ¦ ¦ (Matches vocabulary concepts) ¦ ¦ (H.264 ISL Sign Video Assets) ¦
+----------------------+ +---------------------------------+ +--------------------------------+
`

---

## ?? Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Wouter, Framer Motion
- **Vision Tracking**: @mediapipe/tasks-vision (HandLandmarker)
- **Backend Service**: Python 3.10+, FastAPI, Uvicorn, Pydantic
- **Machine Learning**: PyTorch ( orch.nn.GRU), NumPy
- **Speech Integration**: Web Speech API (SpeechRecognition & SpeechSynthesis)
- **Video Assets**: H.264 encoded MP4 sign demonstrations

---

## ?? Supported ISL Vocabulary (40 Concepts)

The active MVP model supports classification and playback across **40 ISL concepts**:

| Category | Concepts |
| :--- | :--- |
| **Greetings & Civilities** | HELLO, HOW_ARE_YOU, GOOD_MORNING, GOOD_AFTERNOON, GOOD_EVENING, GOOD_NIGHT, THANK_YOU, PLEASE, YES, NO, OKAY, HELP |
| **Pronouns & People** | I, YOU, HE, SHE, WE, BOY, GIRL, FRIEND, FAMILY, FATHER, MOTHER, BROTHER, SISTER |
| **Actions & Needs** | EAT, DRINK, FOOD, WATER, TEA, GO |
| **Places & Roles** | HOUSE, SCHOOL, STUDENT, TEACHER, DOCTOR, HOSPITAL |
| **Questions & Time** | WHERE, WHAT, TODAY |

---

## ?? Model & Evaluation

### Architecture
- **Model Type**: HandsGRU (2-layer Recurrent Neural Network with Dropout)
- **Input Dimension**: (36, 126) representing 36 temporal frames of wrist-normalized hand coordinates
- **Hidden Dimensions**: 128 units, 2 layers, dropout = 0.3
- **Output**: 40 classification logits with Softmax confidence probabilities
- **Active Runtime Checkpoint**: models/gru_model.pt & models/label_map.json

### Dataset & Held-Out Evaluation Benchmark
- **Dataset Notice**: Training datasets are intentionally **not** included in the public repository due to large size; raw and processed training datasets remain stored locally and ignored via .gitignore.
- **Test Benchmark**: Evaluation on the held-out test dataset (93 test clips across multiple signers):
 - **Accuracy**: **76.34%** (71 / 93 correct classifications)
 - **Macro F1 Score**: **0.6071**

> [!NOTE]
> **Prototype Status & Transparency**: The evaluation set is small and imbalanced across classes. SignBridge is an MVP research prototype. Accuracy rates in live deployment may vary depending on lighting, camera angle, and signer speed.

---

## ?? Project Structure

`
SignBridge-ISL/
+-- README.md # Repository Overview & Documentation
+-- package.json # Monorepo Workspace Configuration
+-- pnpm-workspace.yaml # pnpm Workspace Packages setup
+-- models/ # Deployment Model Assets
¦ +-- gru_model.pt # Trained 40-Class PyTorch GRU Checkpoint
¦ +-- label_map.json # Label-to-Index Mapping Dictionary
+-- artifacts/
¦ +-- signbridge/ # React + TypeScript Frontend Application
¦ ¦ +-- package.json
¦ ¦ +-- vite.config.ts
¦ ¦ +-- src/ # React UI Components & Hand Tracking Logic
¦ ¦ +-- public/signs/ # Browser-compatible H.264 ISL Sign Videos
¦ +-- isl-inference/ # FastAPI + PyTorch Backend Inference Server
¦ +-- server.py # FastAPI Live Server (/health, /predict)
¦ +-- models/ # Local Backend Model Directory
+-- scripts/ # Workspace Utility Scripts
`

---

## ?? Local Setup

### Prerequisites
- **Node.js** (v18 or higher) & **pnpm** installed
- **Python** (v3.10 or v3.11) with pip installed

---

### Step 1: Clone Repository
`ash
git clone https://github.com/HimanshMittal08/SignBridge-ISL.git
cd SignBridge-ISL
`

---

### Step 2: Backend Setup (FastAPI & PyTorch)
`powershell
cd artifacts/isl-inference

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install backend requirements
pip install fastapi uvicorn torch numpy pydantic

# Launch Inference Server on Port 8000
python -m uvicorn server:app --host 0.0.0.0 --port 8000
`
*The backend API will run at http://localhost:8000 (GET /health, POST /predict).*

---

### Step 3: Frontend Setup (React + Vite)
Open a second terminal window:

`powershell
# Navigate to frontend directory
cd artifacts/signbridge

# Install workspace dependencies
pnpm install

# Set environment variables and launch dev server
=5000
=/
pnpm run dev
`
*Access the web app in your browser at http://localhost:5000.*

---

## ? How It Works

1. **Feature Extraction**: Browser captures webcam video. MediaPipe Vision detects left and right hand keypoints (21 joints × 3D coordinates × 2 hands = 126 values per frame).
2. **Wrist Normalization**: Coordinates are normalized relative to the wrist position and scaled by hand span size to ensure invariance to distance and position.
3. **Temporal Buffer**: A sliding window collects 36 consecutive frames of landmarks (~1.2 seconds of gesturing).
4. **PyTorch Inference**: The sequence is submitted to POST /predict. The HandsGRU model computes probability scores over the 40 classes.
5. **UI Rendering**: High-confidence predictions update the real-time transcript and trigger text-to-speech output.

---

## ?? Limitations

- **Vocabulary Size**: Currently restricted to 40 primary ISL concepts.
- **Hands-Only Tracking**: Relies exclusively on hand landmarks; facial expressions and torso gestures are not currently modeled.
- **Lighting & Camera Dependence**: Performance varies under poor lighting or non-standard camera angles.

---

## ?? Future Improvements

- **Holistic Tracking**: Integrate MediaPipe Holistic (Pose + Face + Hands) to capture subtle non-manual signals.
- **Continuous Sign Translation**: Expand from isolated concept classification to continuous ISL sentence translation.
- **Expanded Dataset**: Scale training data across diverse signers, backgrounds, and regional ISL variations.
- **On-Device Inference**: Port PyTorch model to ONNX Web / TensorFlow.js for zero-latency in-browser inference.

---

## ?? Project Status

This repository is maintained as an open MVP research prototype for Indian Sign Language accessibility technology.
