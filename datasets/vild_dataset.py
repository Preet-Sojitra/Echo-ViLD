"""
PyTorch Dataset class that loads M1 & M2's .pt files
"""
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

from datasets.hf_utils import download_100_bbox_256D, download_image_embeddings, IMAGE_EMB_VARIANTS

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
        labels      (300,)       — GT category ids; 0 = background, 1..80 = COCO classes
        teacher_emb (300, 1024)  — PE-AV teacher embeddings padded to 300 (zeros where missing)
        valid_mask  (300,)       — Boolean mask; True = teacher embedding exists for this proposal

    Pass mock_teacher=True to substitute random tensors for teacher_emb,
    allowing training/testing without data being available.
    """

    NUM_PROPOSALS = 300
    EMBED_DIM = 1024

    def __init__(
        self,
        image_emb_variant: str,
        mock_teacher: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        self.mock_teacher = mock_teacher
        self.dtype = dtype
        # self._label_lut = _build_coco_label_lut()

        # Download bboxes + RoI features from HF
        bbox_feature_root = download_100_bbox_256D()
        self.roi_features_dir = bbox_feature_root / "Bboxes_and_256D"

        if not mock_teacher:
            # Download PE-AV embeddings + metadata for the requested variant
            hf_folder = IMAGE_EMB_VARIANTS[image_emb_variant]
            teacher_embed_root = download_image_embeddings(image_emb_variant)
            self.teacher_embeds_dir = teacher_embed_root / "Sam_Peav_Outputs" / hf_folder
            self.metadata_dir = teacher_embed_root / "Sam_Peav_Outputs" / "all_metadata"

            # Only include images that have both student features AND teacher embeddings
            student_ids = {p.stem for p in self.roi_features_dir.glob("*.pt")}
            teacher_ids = {p.stem for p in self.teacher_embeds_dir.glob("*.pt")}
            common_ids = student_ids & teacher_ids
            self.image_ids = sorted(common_ids)
        else:
            self.teacher_embeds_dir = None
            self.metadata_dir = None
            self.image_ids = sorted(p.stem for p in self.roi_features_dir.glob("*.pt"))

        assert len(self.image_ids) > 0, (
            f"No matching .pt files found. "
            f"Student dir: {self.roi_features_dir}, Teacher dir: {self.teacher_embeds_dir}"
        )
        print(f"EchoViLDDataset: {len(self.image_ids)} images loaded "
              f"(variant={image_emb_variant}, mock={mock_teacher})")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        roi_data = torch.load(self.roi_features_dir / f"{image_id}.pt", map_location="cpu", weights_only=True)
        roi_feat = roi_data["roi_features"].to(self.dtype)  # (300, 256)

        # Remap raw COCO category_ids (1..90 with gaps) → compact (0=bg, 1..80)
        # raw_labels = roi_data["labels"].long()
        # labels = self._label_lut[raw_labels]  # (300,)
        labels = roi_data["labels"].long() # (300,)

        if self.mock_teacher:
            teacher_emb = torch.randn(self.NUM_PROPOSALS, self.EMBED_DIM, dtype=self.dtype)
            valid_mask = torch.ones(self.NUM_PROPOSALS, dtype=torch.bool)
        else:
            # Load variable-length teacher embeddings (N, 1024) where N <= 300
            raw_teacher = torch.load(
                self.teacher_embeds_dir / f"{image_id}.pt", map_location="cpu", weights_only=True
            ).to(self.dtype)

            # Load metadata to get det_idx → original proposal index mapping
            metadata = torch.load(
                self.metadata_dir / f"{image_id}.pt", map_location="cpu", weights_only=False
            )

            # Build padded teacher tensor and validity mask
            teacher_emb = torch.zeros(self.NUM_PROPOSALS, self.EMBED_DIM, dtype=self.dtype)
            valid_mask = torch.zeros(self.NUM_PROPOSALS, dtype=torch.bool)

            for i, meta in enumerate(metadata):
                det_idx = meta["det_idx"]
                if det_idx < self.NUM_PROPOSALS and i < raw_teacher.shape[0]:
                    teacher_emb[det_idx] = raw_teacher[i]
                    valid_mask[det_idx] = True

        return roi_feat, labels, teacher_emb, valid_mask
