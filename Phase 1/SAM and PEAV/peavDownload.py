from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/pe-av-small-16-frame",
    local_dir="pe-av-small-16-frame"
)