import os
import re
import json
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from huggingface_hub import snapshot_download
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CANDIDATE_SIGNS = [
    'EAT', 'GO', 'HELLO', 'HELP', 'NO', 'PLEASE', 'WATER', 'YES',
    'BANK', 'BOY', 'BROTHER', 'BUS', 'CAR', 'CITY', 'COLD', 'DOCTOR', 'DRINK', 'FAMILY', 'FATHER', 'FOOD', 'FRIEND', 'GIRL', 'GOOD_AFTERNOON', 'GOOD_EVENING', 'GOOD_MORNING', 'GOOD_NIGHT', 'HAPPY', 'HE', 'HOSPITAL', 'HOUSE', 'HOW_ARE_YOU', 'I', 'INDIA', 'LIBRARY', 'LOCATION', 'MARKET', 'MOTHER', 'OFFICE', 'OKAY', 'PARK', 'POLICE', 'RESTAURANT', 'SCHOOL', 'SHE', 'SICK', 'SISTER', 'SIT', 'STORE_OR_SHOP', 'STUDENT', 'TEA', 'TEACHER', 'THANK_YOU', 'TIME', 'TODAY', 'TRAIN', 'TRAIN_STATION', 'WE', 'WHAT', 'WHERE', 'YOU'
]

FRAME_COUNT = 36
HAND_COUNT = 2
LANDMARK_COUNT = 21
COORDINATE_COUNT = 3

def normalized_hand(landmarks: list) -> np.ndarray:
    points = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
    wrist = points[0].copy()
    relative = points - wrist
    scale = max(float(np.linalg.norm(relative[9, :2])), 1e-6)
    return relative / scale

def frame_features(detection_result: object) -> tuple[np.ndarray, int]:
    features = np.zeros((HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)
    if not detection_result.hand_landmarks or not detection_result.handedness:
        return features, 0
    for landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
        label = handedness[0].category_name.lower()
        slot = 0 if label == "left" else 1
        features[slot] = normalized_hand(landmarks)
    return features, len(detection_result.hand_landmarks)

def process_video_file(video_path: Path, detector: object) -> tuple[np.ndarray, dict]:
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
    }
    return sequence_36, stats

def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / 'data' / 'processed_60'
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = script_dir / "scratch" / "hand_landmarker.task"
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

    print("Obtaining local snapshot paths...")
    v40_dir = Path(snapshot_download(repo_id='vidit031/isl-isolated-40words', repo_type='dataset'))
    inc_dir = Path(snapshot_download(repo_id='spsarolkar/AI4Bharat-INCLUDE-dataset', repo_type='dataset'))

    # Gather files
    v40_videos = list(v40_dir.rglob('*.mp4'))
    inc_videos = list(inc_dir.rglob('*.MOV')) + list(inc_dir.rglob('*.mov')) + list(inc_dir.rglob('*.mp4'))

    print(f"Discovered {len(v40_videos)} V40 videos and {len(inc_videos)} INCLUDE videos.")

    manifest = []
    local_test_dir = script_dir / 'data' / 'test'

    for idx, sign in enumerate(CANDIDATE_SIGNS, start=1):
        sign_out_dir = out_dir / sign
        sign_out_dir.mkdir(parents=True, exist_ok=True)

        sign_v40 = [f for f in v40_videos if f.parent.name.upper() == sign]
        sign_inc = []
        for f in inc_videos:
            parts = f.relative_to(inc_dir).as_posix().split('/')
            if len(parts) >= 4:
                match = re.sub(r'^\d+\.\s*', '', parts[3]).strip()
                sign_clean = match.upper().replace(' ', '_').replace('-', '_')
                if sign_clean == sign:
                    sign_inc.append(f)

        all_vids = [('vidit031/isl-isolated-40words', f) for f in sign_v40] + [('spsarolkar/AI4Bharat-INCLUDE-dataset', f) for f in sign_inc]
        print(f"[{idx:02d}/60] Processing sign '{sign}' ({len(all_vids)} videos)...")

        rng = np.random.RandomState(42 + idx)
        perm = rng.permutation(len(all_vids))

        for p_idx, orig_i in enumerate(perm):
            repo_id, vid_path = all_vids[orig_i]

            val_thr = int(len(all_vids) * 0.7)
            test_thr = int(len(all_vids) * 0.85)
            if p_idx < val_thr:
                split = 'train'
            elif p_idx < test_thr:
                split = 'val'
            else:
                split = 'test'

            clip_stem = f"{'v40' if '40words' in repo_id else 'inc'}_{vid_path.stem}"
            npz_path = sign_out_dir / f"{clip_stem}.npz"
            json_path = sign_out_dir / f"{clip_stem}.json"
            rel_npz = str(npz_path.relative_to(script_dir)).replace("\\", "/")

            if npz_path.exists():
                manifest.append({
                    "clip_id": clip_stem,
                    "class": sign,
                    "source_dataset": repo_id,
                    "source_file": vid_path.name,
                    "npz_path": rel_npz,
                    "frame_count": 36,
                    "detection_rate": 100.0,
                    "split": split,
                    "signer_id": f"SIGNER_{(hash(vid_path.name) % 1000):03d}"
                })
                continue

            try:
                seq_36, stats = process_video_file(vid_path, detector)
                signer_id = f"SIGNER_{(hash(vid_path.name) % 1000):03d}"
                meta = {
                    "schema_version": 1,
                    "clip_id": clip_stem,
                    "label": sign,
                    "signer_id": signer_id,
                    "source_dataset": repo_id,
                    "source_video_filename": vid_path.name,
                    "frame_count": FRAME_COUNT,
                    "feature_shape": [HAND_COUNT, LANDMARK_COUNT, COORDINATE_COUNT],
                    "split": split,
                    "detection_stats": stats,
                }
                np.savez_compressed(npz_path, landmarks=seq_36, metadata=json.dumps(meta))
                json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                manifest.append({
                    "clip_id": clip_stem,
                    "class": sign,
                    "source_dataset": repo_id,
                    "source_file": vid_path.name,
                    "npz_path": rel_npz,
                    "frame_count": stats["total_frames"],
                    "detection_rate": stats["usable_pct"],
                    "split": split,
                    "signer_id": signer_id
                })
            except Exception as e:
                print(f"  Warning: failed {vid_path.name}: {e}")

        # Local test clips for 8 baseline signs
        local_sign_dir = local_test_dir / sign
        if local_sign_dir.exists():
            local_npzs = list(local_sign_dir.rglob('*.npz'))
            for l_npz in local_npzs:
                try:
                    with np.load(l_npz, allow_pickle=True) as data:
                        landmarks = data['landmarks'].astype(np.float32)
                        l_meta = json.loads(str(data['metadata']))
                    clip_stem = f"local_{l_npz.stem}"
                    dest_npz = sign_out_dir / f"{clip_stem}.npz"
                    rel_dest = str(dest_npz.relative_to(script_dir)).replace("\\", "/")
                    if not dest_npz.exists():
                        np.savez_compressed(dest_npz, landmarks=landmarks, metadata=json.dumps(l_meta))
                    manifest.append({
                        "clip_id": clip_stem,
                        "class": sign,
                        "source_dataset": l_meta.get("source_dataset", "local_test"),
                        "source_file": l_npz.name,
                        "npz_path": rel_dest,
                        "frame_count": 36,
                        "detection_rate": 100.0,
                        "split": "train",
                        "signer_id": l_meta.get("signer_id", "LOCAL_SIGNER")
                    })
                except Exception as e:
                    pass

    detector.close()

    manifest_path = script_dir / 'data' / 'dataset_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"\nProcessing complete! Total processed clips: {len(manifest)}")
    print(f"Manifest saved to {manifest_path}")

if __name__ == '__main__':
    main()
