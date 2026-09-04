import os
import re
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

CANDIDATE_SIGNS = [
    'EAT', 'GO', 'HELLO', 'HELP', 'NO', 'PLEASE', 'WATER', 'YES',
    'BANK', 'BOY', 'BROTHER', 'BUS', 'CAR', 'CITY', 'COLD', 'DOCTOR', 'DRINK', 'FAMILY', 'FATHER', 'FOOD', 'FRIEND', 'GIRL', 'GOOD_AFTERNOON', 'GOOD_EVENING', 'GOOD_MORNING', 'GOOD_NIGHT', 'HAPPY', 'HE', 'HOSPITAL', 'HOUSE', 'HOW_ARE_YOU', 'I', 'INDIA', 'LIBRARY', 'LOCATION', 'MARKET', 'MOTHER', 'OFFICE', 'OKAY', 'PARK', 'POLICE', 'RESTAURANT', 'SCHOOL', 'SHE', 'SICK', 'SISTER', 'SIT', 'STORE_OR_SHOP', 'STUDENT', 'TEA', 'TEACHER', 'THANK_YOU', 'TIME', 'TODAY', 'TRAIN', 'TRAIN_STATION', 'WE', 'WHAT', 'WHERE', 'YOU'
]

def download_dataset():
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / 'data' / 'raw_60'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    api = HfApi()
    print("Fetching file manifests from HuggingFace...")
    v40_files = api.list_repo_files('vidit031/isl-isolated-40words', repo_type='dataset')
    inc_files = api.list_repo_files('spsarolkar/AI4Bharat-INCLUDE-dataset', repo_type='dataset')
    
    download_summary = {}
    failed_downloads = []
    
    for idx, sign in enumerate(CANDIDATE_SIGNS, start=1):
        sign_dir = raw_dir / sign
        sign_dir.mkdir(parents=True, exist_ok=True)
        
        v40_matches = [f for f in v40_files if f.split('/')[0].upper() == sign and (f.endswith('.mp4') or f.endswith('.json'))]
        inc_matches = []
        for f in inc_files:
            if f.lower().endswith('.mov') or f.lower().endswith('.mp4') or f.lower().endswith('.json'):
                parts = f.split('/')
                if len(parts) >= 4:
                    match = re.sub(r'^\d+\.\s*', '', parts[3]).strip()
                    sign_clean = match.upper().replace(' ', '_').replace('-', '_')
                    if sign_clean == sign:
                        inc_matches.append(f)
                        
        print(f"[{idx:02d}/60] Downloading {sign}... (V40: {len([f for f in v40_matches if not f.endswith('.json')])}, INC: {len([f for f in inc_matches if not f.endswith('.json')])})")
        
        downloaded_count = 0
        
        # Download V40 files
        for rel_file in v40_matches:
            try:
                dest = sign_dir / f"v40_{Path(rel_file).name}"
                if not dest.exists() or dest.stat().st_size == 0:
                    downloaded_path = hf_hub_download(
                        repo_id='vidit031/isl-isolated-40words',
                        filename=rel_file,
                        repo_type='dataset'
                    )
                    shutil.copy(downloaded_path, dest)
                if dest.name.endswith('.mp4') or dest.name.endswith('.mov'):
                    downloaded_count += 1
            except Exception as e:
                failed_downloads.append({'sign': sign, 'repo': 'vidit031/isl-isolated-40words', 'file': rel_file, 'error': str(e)})
                
        # Download INCLUDE files
        for rel_file in inc_matches:
            try:
                dest = sign_dir / f"inc_{Path(rel_file).name}"
                if not dest.exists() or dest.stat().st_size == 0:
                    downloaded_path = hf_hub_download(
                        repo_id='spsarolkar/AI4Bharat-INCLUDE-dataset',
                        filename=rel_file,
                        repo_type='dataset'
                    )
                    shutil.copy(downloaded_path, dest)
                if dest.name.endswith('.mp4') or dest.name.endswith('.mov'):
                    downloaded_count += 1
            except Exception as e:
                failed_downloads.append({'sign': sign, 'repo': 'spsarolkar/AI4Bharat-INCLUDE-dataset', 'file': rel_file, 'error': str(e)})
                
        download_summary[sign] = downloaded_count
        
    summary_path = raw_dir / 'download_summary.json'
    summary_path.write_text(json.dumps({
        'total_classes': len(CANDIDATE_SIGNS),
        'downloaded_counts': download_summary,
        'failed_downloads': failed_downloads
    }, indent=2), encoding='utf-8')
    
    print("\nDownload finished!")
    print(f"Total videos downloaded/verified: {sum(download_summary.values())}")
    print(f"Failed downloads count: {len(failed_downloads)}")
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    download_dataset()
