"""
PyTorch Dataset class that loads M1 & M2's .pt files
"""
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

from datasets.hf_utils import download_100_bbox_256D, download_image_embeddings


# Raw COCO category_ids are non-contiguous (1..90 with gaps), but text embeddings
# are indexed by position in coco_class_descriptions.json. Build a lookup tensor
# that maps raw_id -> compact class index (1..80), reserving 0 for background.
_COCO_DESC_PATH = Path(__file__).parent.parent / "offline_prep" / "coco_class_descriptions.json"


def _build_coco_label_lut() -> torch.Tensor:
    with open(_COCO_DESC_PATH) as f:
        classes = json.load(f)
    max_raw_id = max(c["id"] for c in classes)
    lut = torch.zeros(max_raw_id + 1, dtype=torch.long)
    for i, c in enumerate(classes):
        lut[c["id"]] = i + 1
    return lut


class EchoViLDDataset(Dataset):
    """Joins RoIAlign features with PE-AV image embeddings.

    Each item is a tuple of:
        roi_feat    (300, 256)   — Mask R-CNN RoIAlign features (student input)
        labels      (300,)       — GT category ids; 0 = background
        teacher_emb (300, 1024)  — PE-AV image embeddings (distillation target)

    Pass mock_teacher=True to substitute random tensors for teacher_emb,
    allowing training/testing without data being available.
    """

    def __init__(
        self,
        image_emb_variant: str,
        mock_teacher: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        self.mock_teacher = mock_teacher
        self.dtype = dtype
        self._label_lut = _build_coco_label_lut()

        # Download bboxes + RoI features from HF
        bbox_feature_root = download_100_bbox_256D()
        self.roi_features_dir = bbox_feature_root / "Bboxes_and_256D"

        self.image_ids = sorted(p.stem for p in self.roi_features_dir.glob("*.pt"))
        assert len(self.image_ids) > 0, f"No .pt files found in {self.roi_features_dir}"

        if not mock_teacher:
            # Download PE-AV embeddings for the requested variant
            teacher_embed_root = download_image_embeddings(image_emb_variant)
            self.teacher_embeds_dir = teacher_embed_root / "image_embeddings" / image_emb_variant
        else:
            self.teacher_embeds_dir = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        roi_data = torch.load(self.roi_features_dir / f"{image_id}.pt", map_location="cpu", weights_only=True)
        roi_feat = roi_data["roi_features"].to(self.dtype)  # (300, 256)
        # Stored labels are raw COCO category_ids; remap to compact [1..80] (0 = bg)
        labels   = self._label_lut[roi_data["labels"].long()]  # (300,)

        if self.mock_teacher:
            teacher_emb = torch.randn(300, 1024, dtype=self.dtype)
        else:
            teacher_emb = torch.load(
                self.teacher_embeds_dir / f"{image_id}.pt", map_location="cpu", weights_only=True
            ).to(self.dtype)                            # (300, 1024)

        return roi_feat, labels, teacher_emb
