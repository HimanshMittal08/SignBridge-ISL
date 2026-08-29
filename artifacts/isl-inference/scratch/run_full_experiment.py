import json
import re
import urllib.request
from pathlib import Path
from uuid import uuid4
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

FRAME_COUNT = 36
HAND_COUNT = 2
LANDMARK_COUNT = 21
COORDINATE_COUNT = 3


def safe_identifier(value: str, default: str = "UNKNOWN") -> str:
    if not value:
        return default
    normalized = value.strip().upper()
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", normalized)
    return sanitized if sanitized else default


def normalized_hand(landmarks: list) -> np.ndarray:
    """Return a wrist-relative, scale-normalized (21, 3) hand tensor."""
    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
    )
    wrist = points[0].copy()
    relative = points - wrist
    # Landmark 9 is the middle-finger MCP. Its wrist distance is a stable
    # per-frame scale reference; a tiny floor handles degenerate detections.
    scale = max(float(np.linalg.norm(relative[9, :2])), 1e-6)
    return relative / scale


def frame_features(detection_result: object) -> tuple[np.ndarray, int]:
    """Return left/right landmark slots and the number of detected hands."""
    features = np.zeros((HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)
    if not detection_result.hand_landmarks or not detection_result.handedness:
        return features, 0

    for landmarks, handedness in zip(
        detection_result.hand_landmarks, detection_result.handedness
    ):
        label = handedness[0].category_name.lower()
        slot = 0 if label == "left" else 1
        features[slot] = normalized_hand(landmarks)
    return features, len(detection_result.hand_landmarks)


def process_video_file(video_path: Path, detector: object):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_feats = []
    hand_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        feats, count = frame_features(result)
        frame_feats.append(feats)
        hand_counts.append(count)

    cap.release()

    total_frames = len(frame_feats)
    if total_frames == 0:
        raw_sequence = np.zeros((1, HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)
        hand_counts = [0]
        total_frames = 1
    else:
        raw_sequence = np.stack(frame_feats).astype(np.float32)

    frames_ge1 = sum(1 for c in hand_counts if c >= 1)
    frames_2 = sum(1 for c in hand_counts if c == 2)
    usable_pct = (frames_ge1 / total_frames) * 100.0 if total_frames > 0 else 0.0

    valid_indices = [i for i, c in enumerate(hand_counts) if c >= 1]
    if len(valid_indices) >= FRAME_COUNT:
        idx = np.linspace(0, len(valid_indices) - 1, FRAME_COUNT, dtype=int)
        sampled_indices = [valid_indices[i] for i in idx]
    elif len(valid_indices) > 0:
        idx = np.linspace(0, len(valid_indices) - 1, FRAME_COUNT)
        sampled_indices = [valid_indices[int(round(i))] for i in idx]
    else:
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


def main():
    root = Path(r"c:\Users\neetu\Documents\Codex\2026-08-29\continue-working-on-the-existing-signbridge\artifacts\isl-inference\scratch\raw_hf_dataset")
    output_base = Path(r"c:\Users\neetu\Documents\Codex\2026-08-29\continue-working-on-the-existing-signbridge\artifacts\isl-inference\data\test")
    output_base.mkdir(parents=True, exist_ok=True)

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

    mp4_files = sorted(list(root.rglob("*.mp4")))
    print(f"Starting processing of {len(mp4_files)} videos...")

    all_results = []

    for idx, video_path in enumerate(mp4_files, start=1):
        sidecar_json_path = video_path.with_suffix(".json")
        orig_metadata = {}
        if sidecar_json_path.exists():
            try:
                orig_metadata = json.loads(sidecar_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error reading sidecar {sidecar_json_path}: {e}")

        word = orig_metadata.get("word") or orig_metadata.get("normalized_word") or video_path.parent.name
        label = safe_identifier(word)
        signer = safe_identifier(orig_metadata.get("signer"), "EXTERNAL_SIGNER")
        dataset_src = orig_metadata.get("dataset", "UNKNOWN_DATASET")

        seq_36, stats = process_video_file(video_path, detector)

        clip_id = f"test-{uuid4().hex[:8]}"
        signer_dir = output_base / label / signer
        signer_dir.mkdir(parents=True, exist_ok=True)

        clip_stem = f"{dataset_src.lower()}_{video_path.stem}"
        clip_path = signer_dir / f"{clip_stem}.npz"
        json_path = signer_dir / f"{clip_stem}.json"

        meta = {
            "schema_version": 1,
            "clip_id": clip_id,
            "label": label,
            "signer_id": signer,
            "source_dataset": dataset_src,
            "source_video_filename": video_path.name,
            "frame_count": FRAME_COUNT,
            "feature_shape": [HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT],
            "normalization": "wrist-relative; divide all xyz coordinates by wrist-to-middle-MCP xy distance",
            "hand_slots": ["left", "right"],
            "missing_hand": "zero-filled",
            "detection_stats": stats,
            "provenance": orig_metadata,
        }

        np.savez_compressed(clip_path, landmarks=seq_36, metadata=json.dumps(meta))
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        result_entry = {
            "index": idx,
            "video_filename": video_path.name,
            "label": label,
            "dataset_source": dataset_src,
            "signer": signer,
            "total_frames": stats["total_frames"],
            "frames_ge1_hand": stats["frames_ge1_hand"],
            "frames_2_hands": stats["frames_2_hands"],
            "usable_pct": stats["usable_pct"],
            "hand_distribution": stats["hand_counts_distribution"],
            "npz_rel_path": str(clip_path.relative_to(output_base)).replace("\\", "/"),
            "json_rel_path": str(json_path.relative_to(output_base)).replace("\\", "/"),
        }
        all_results.append(result_entry)
        print(f"[{idx}/{len(mp4_files)}] {video_path.name} -> {label}/{signer} | Usable: {stats['usable_pct']}% ({stats['frames_ge1_hand']}/{stats['total_frames']} frames)")

    detector.close()

    summary_file = output_base / "experiment_summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Experiment complete! Processed {len(all_results)} videos. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
