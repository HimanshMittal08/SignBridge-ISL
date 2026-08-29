import sys
from pathlib import Path
from huggingface_hub import snapshot_download

target_dir = Path(r"c:\Users\neetu\Documents\Codex\2026-08-29\continue-working-on-the-existing-signbridge\artifacts\isl-inference\scratch\raw_hf_dataset")
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading vidit031/isl-isolated-8words to {target_dir}...")
local_dir = snapshot_download(
    repo_id="vidit031/isl-isolated-8words",
    repo_type="dataset",
    local_dir=target_dir,
)
print(f"Download complete: {local_dir}")
