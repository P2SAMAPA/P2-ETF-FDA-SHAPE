import json, tempfile, os
from datetime import datetime
from huggingface_hub import HfApi, upload_file
import config

def push_daily_result(results: dict):
    api = HfApi(token=config.HF_TOKEN)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"fda_{today}.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f, indent=2, default=str)
        temp_path = f.name
    try:
        upload_file(path_or_fileobj=temp_path, path_in_repo=filename,
                    repo_id=config.HF_OUTPUT_DATASET, repo_type="dataset", token=config.HF_TOKEN)
        print(f"Uploaded results to {config.HF_OUTPUT_DATASET}/{filename}")
    finally:
        os.unlink(temp_path)

def load_latest_result() -> dict:
    api = HfApi(token=config.HF_TOKEN)
    files = api.list_repo_files(repo_id=config.HF_OUTPUT_DATASET, repo_type="dataset")
    json_files = [f for f in files if f.startswith("fda_") and f.endswith(".json")]
    if not json_files:
        return {}
    json_files.sort(reverse=True)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=config.HF_OUTPUT_DATASET, filename=json_files[0],
                           repo_type="dataset", token=config.HF_TOKEN)
    with open(path, "r") as f:
        return json.load(f)
