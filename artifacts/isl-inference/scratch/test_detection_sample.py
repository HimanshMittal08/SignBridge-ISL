import json
import urllib.request
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

model_path = Path("scratch/hand_landmarker.task")
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

root = Path("scratch/raw_hf_dataset")
sample_videos = list(root.rglob("*.mp4"))[:5]

for vp in sample_videos:
    cap = cv2.VideoCapture(str(vp))
    total = 0
    detected_1p = 0
    detected_2h = 0
    hand_counts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        nhands = len(result.hand_landmarks) if result.hand_landmarks else 0
        hand_counts.append(nhands)
        if nhands >= 1:
            detected_1p += 1
        if nhands == 2:
            detected_2h += 1
            
    cap.release()
    pct = (detected_1p / total * 100.0) if total > 0 else 0
    print(f"Video: {vp.name} | Total: {total} | >=1 hand: {detected_1p} ({pct:.1f}%) | 2 hands: {detected_2h}")

detector.close()
