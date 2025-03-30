from huggingface_hub import snapshot_download

snapshot_download(repo_id="kh4dien/gemma-3-4b-saes", allow_patterns="gemma-3-4b-step-final/*", local_dir="/workspace/gemma-3-4b-saes")
