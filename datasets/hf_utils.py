"""HuggingFace download helpers — fetches shared .pt data from the Echo-ViLD dataset repo."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "preetsojitra/Echo-VilD"
# Cache lands at <repo_root>/hf_cache/ (two levels up from this file)
HF_CACHE = Path(__file__).parent.parent / "hf_cache"


def _download(allow_patterns: list[str]) -> Path:
    """Run a snapshot download restricted to the given glob patterns and return the local path."""
    token = os.environ.get("HF_TOKEN", None)
    local = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        cache_dir=str(HF_CACHE),
        token=token,
    )
    return Path(local)


def download_100_bbox_256D() -> Path:
    """Download bboxes + 256D RoIAlign features (one .pt per image)."""
    return _download(["Bboxes_and_256D/*"])


def download_image_embeddings(variant: str) -> Path:
    """Download PE-AV image embeddings for a specific variant (one .pt per image)."""
    return _download([f"image_embeddings/{variant}/*"])


def download_text_embeddings(variant: str) -> Path:
    """Download PE-AV text embeddings for a specific variant (single .pt file)."""
    return _download([f"Text_Embeddings/{variant}.pt"])
