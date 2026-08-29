"""Record real, hands-only MediaPipe landmark clips for SignBridge.

No model inference happens in this utility.  It records only real webcam
observations and explicit label/signer metadata for later training.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np

FRAME_COUNT = 36
HAND_COUNT = 2
LANDMARK_COUNT = 21
COORDINATE_COUNT = 3
COUNTDOWN_SECONDS = 3


def safe_identifier(value: str, field_name: str) -> str:
    normalized = value.strip().upper() if field_name == "label" else value.strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        raise argparse.ArgumentTypeError(
            f"{field_name} must contain only letters, numbers, hyphens, or underscores"
        )
    return normalized


def normalized_hand(landmarks: object) -> np.ndarray:
    """Return a wrist-relative, scale-normalized (21, 3) hand tensor."""
    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32
    )
    wrist = points[0].copy()
    relative = points - wrist
    # Landmark 9 is the middle-finger MCP.  Its wrist distance is a stable
    # per-frame scale reference; a tiny floor handles degenerate detections.
    scale = max(float(np.linalg.norm(relative[9, :2])), 1e-6)
    return relative / scale


def frame_features(results: object) -> tuple[np.ndarray, int]:
    """Return left/right landmark slots and the number of detected hands."""
    features = np.zeros((HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return features, 0

    for landmarks, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        label = handedness.classification[0].label.lower()
        slot = 0 if label == "left" else 1
        features[slot] = normalized_hand(landmarks)
    return features, len(results.multi_hand_landmarks)


def draw_status(frame: np.ndarray, status: str, detail: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (20, 35, 37), thickness=-1)
    cv2.putText(frame, status, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (86, 224, 209), 2)
    cv2.putText(frame, detail, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (244, 243, 234), 1)


def write_clip(
    root: Path, label: str, signer: str, session: str, frames: list[np.ndarray]
) -> Path:
    output_dir = root / label / signer
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    clip_path = output_dir / f"{session}-{clip_id}.npz"
    metadata = {
        "schema_version": 1,
        "clip_id": clip_id,
        "label": label,
        "signer_id": signer,
        "session_id": session,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": FRAME_COUNT,
        "feature_shape": [HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT],
        "normalization": "wrist-relative; divide all xyz coordinates by wrist-to-middle-MCP xy distance",
        "hand_slots": ["left", "right"],
        "missing_hand": "zero-filled",
    }
    sequence = np.stack(frames).astype(np.float32)
    np.savez_compressed(clip_path, landmarks=sequence, metadata=json.dumps(metadata))
    clip_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return clip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record SignBridge hands-only landmark clips")
    parser.add_argument("--label", required=True, type=lambda value: safe_identifier(value, "label"))
    parser.add_argument("--signer", required=True, type=lambda value: safe_identifier(value, "signer"))
    parser.add_argument("--session", required=True, type=lambda value: safe_identifier(value, "session"))
    parser.add_argument("--camera", default=0, type=int, help="OpenCV camera index (default: 0)")
    parser.add_argument("--output-dir", default="data/raw", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    recording = False
    countdown_started: float | None = None
    frames: list[np.ndarray] = []
    last_saved = "Press R to begin a recording. Press Q or Esc to quit."

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera returned no frame")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            features, detected_count = frame_features(results)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

            now = time.monotonic()
            if countdown_started is not None:
                remaining = COUNTDOWN_SECONDS - (now - countdown_started)
                if remaining > 0:
                    draw_status(frame, f"Get ready: {int(np.ceil(remaining))}", f"{args.label} | signer {args.signer}")
                else:
                    countdown_started = None
                    recording = True
                    frames.clear()

            if recording:
                if detected_count:
                    frames.append(features)
                draw_status(
                    frame,
                    "RECORDING",
                    f"{len(frames)}/{FRAME_COUNT} detected frames | {detected_count} hand(s) in current frame",
                )
                if len(frames) == FRAME_COUNT:
                    saved = write_clip(args.output_dir, args.label, args.signer, args.session, frames)
                    last_saved = f"Saved: {saved}"
                    recording = False
                    frames.clear()
            elif countdown_started is None:
                draw_status(frame, "READY", f"{args.label} | {last_saved}")

            cv2.imshow("SignBridge hands-only landmark recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and not recording and countdown_started is None:
                countdown_started = time.monotonic()
                last_saved = "Countdown started. Hold the intended sign naturally."
    finally:
        hands.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
