import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

summary_file = Path(r"c:\Users\neetu\Documents\Codex\2026-08-29\continue-working-on-the-existing-signbridge\artifacts\isl-inference\data\test\experiment_summary.json")
data = json.loads(summary_file.read_text(encoding="utf-8"))

print("| # | Video Filename | Label | Source | Signer | Total Frames | >=1 Hand Frames | 2 Hands Frames | Usable % |")
print("|---|---|---|---|---|---|---|---|---|")
for d in data:
    fn = d.get("video_filename") or d.get("video")
    ds = d.get("dataset_source") or d.get("dataset", "UNKNOWN")
    print(f"| {d['index']} | `{fn}` | `{d['label']}` | {ds} | `{d['signer']}` | {d['total_frames']} | {d['frames_ge1_hand']} | {d['frames_2_hands']} | **{d['usable_pct']}%** |")
