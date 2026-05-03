"""HuggingFace download helpers — fetches shared .pt data from the Echo-ViLD dataset repo."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "preetsojitra/Echo-VilD"
LOCAL_CLONE_DIR = Path("./data")

# Maps config `image_emb_variant` names → HF folder names under Sam_Peav_Outputs/
IMAGE_EMB_VARIANTS = {
    "vanilla":                "all_baseline",
    "sam_nocontext":          "all_sam_nocontext",
    "sam_withcontext_equal":  "all_sam_withcontext_equal",
    "sam_withcontext_80_20":  "all_sam_withcontext_80_20",
}


def _download(allow_patterns: list[str]) -> Path:
    """Run a snapshot download restricted to the given glob patterns and return the local path."""
    if LOCAL_CLONE_DIR.exists():
        return LOCAL_CLONE_DIR
    # Fallback to HF (Will likely rate limit if too many files)
    print("Warning: Local clone not found. Falling back to HF API...")
    token = os.environ.get("HF_TOKEN", None)
    local = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        token=token,
        max_workers=2
    )
    return Path(local)


def download_100_bbox_256D() -> Path:
    """Download bboxes + 256D RoIAlign features (one .pt per image)."""
    return _download(["Bboxes_and_256D/*"])


def download_image_embeddings(variant: str) -> Path:
    """Download PE-AV image embeddings for a specific variant (one .pt per image).

    Also downloads the metadata needed to map variable-length teacher
    embeddings back to the original 300-proposal indices.
    """
    hf_folder = IMAGE_EMB_VARIANTS[variant]
    return _download([
        f"Sam_Peav_Outputs/{hf_folder}/*",
        f"Sam_Peav_Outputs/all_metadata/*",
    ])


def download_text_embeddings(variant: str) -> Path:
    """Download PE-AV text embeddings for a specific variant (single .pt file)."""
    return _download([f"Text_Embeddings_COCO/{variant}.pt"])
