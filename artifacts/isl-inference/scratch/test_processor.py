import json
import re
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

FRAME_COUNT = 36
HAND_COUNT = 2
LANDMARK_COUNT = 21
COORDINATE_COUNT = 3


def safe_identifier(value: str) -> str:
    normalized = value.strip().upper()
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", normalized)
    return sanitized if sanitized else "UNKNOWN"


def normalized_hand(landmarks: object) -> np.ndarray:
    """Return a wrist-relative, scale-normalized (21, 3) hand tensor."""
    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32
    )
    wrist = points[0].copy()
    relative = points - wrist
    # Landmark 9 is the middle-finger MCP. Its wrist distance is a stable
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


def process_video(video_path: Path, hands_processor: object):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    all_frame_features = []
    hand_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_processor.process(rgb)
        feats, count = frame_features(results)
        all_frame_features.append(feats)
        hand_counts.append(count)

    cap.release()

    total_frames = len(all_frame_features)
    if total_frames == 0:
        raw_sequence = np.zeros((1, HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)
        hand_counts = [0]
        total_frames = 1
    else:
        raw_sequence = np.stack(all_frame_features).astype(np.float32)

    frames_ge1 = sum(1 for c in hand_counts if c >= 1)
    frames_2 = sum(1 for c in hand_counts if c == 2)
    usable_pct = (frames_ge1 / total_frames) * 100.0 if total_frames > 0 else 0.0

    # Sample to exact 36 frames
    # Prefer frames with >= 1 hand if available, otherwise sample full video
    valid_indices = [i for i, c in enumerate(hand_counts) if c >= 1]
    if len(valid_indices) >= FRAME_COUNT:
        # Uniformly sample 36 frames from valid detected frames
        idx = np.linspace(0, len(valid_indices) - 1, FRAME_COUNT, dtype=int)
        sampled_indices = [valid_indices[i] for i in idx]
    elif len(valid_indices) > 0:
        # Uniformly interpolate valid detected frames to 36
        idx = np.linspace(0, len(valid_indices) - 1, FRAME_COUNT)
        sampled_indices = [valid_indices[int(round(i))] for i in idx]
    else:
        # Uniformly sample across all frames (which are 0-hand frames)
        idx = np.linspace(0, total_frames - 1, FRAME_COUNT)
        sampled_indices = [int(round(i)) for i in idx]

    sequence_36 = raw_sequence[sampled_indices]

    stats = {
        "total_frames": total_frames,
        "frames_ge1_hand": frames_ge1,
        "frames_2_hands": frames_2,
        "usable_pct": round(usable_pct, 2),
        "hand_counts_distribution": {
            "0_hands": sum(1 for c in hand_counts if c == 0),
            "1_hand": sum(1 for c in hand_counts if c == 1),
            "2_hands": sum(1 for c in hand_counts if c == 2),
        },
    }

    return sequence_36, stats


if __name__ == "__main__":
    test_video = list(Path(r"c:\Users\neetu\Documents\Codex\2026-08-29\continue-working-on-the-existing-signbridge\artifacts\isl-inference\scratch\raw_hf_dataset").rglob("*.mp4"))[0]
    print("Testing on video:", test_video)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,  # static_image_mode=True for accurate per-frame video analysis
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    seq, stats = process_video(test_video, hands)
    hands.close()
    print("Extracted shape:", seq.shape)
    print("Stats:", json.dumps(stats, indent=2))
