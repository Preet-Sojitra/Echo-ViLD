import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from datasets.hf_utils import download_text_embeddings


class ClassificationHead(nn.Module):
    """Computes per-proposal logits via cosine similarity against PE-AV text embeddings.

    The classification matrix is [background + 80 COCO classes], where the background
    embedding is a learned parameter and the text embeddings are fixed buffers loaded
    from HuggingFace.
    """

    def __init__(
        self,
        text_emb_variant: str,
        num_classes: int = 80,
        embed_dim: int = 1024,
        temperature: float = 0.01,
    ):
        super().__init__()
        self.temperature = temperature

        # Learnable background token, initialized small so softmax starts near uniform
        self.bg_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Load and freeze text embeddings from HF  →  [num_classes, embed_dim]
        text_root = download_text_embeddings(text_emb_variant)
        text_path = text_root / "Text_Embeddings_COCO" / f"{text_emb_variant}.pt"
        text_embeddings = torch.load(text_path, map_location="cpu", weights_only=True).float()

        assert text_embeddings.shape == (num_classes, embed_dim), (
            f"Expected text_emb shape ({num_classes}, {embed_dim}), got {tuple(text_embeddings.shape)}"
        )
        self.register_buffer("text_emb", text_embeddings)

    def _text_matrix(self) -> torch.Tensor:
        """Build the [num_classes+1, D] classification matrix with background token first."""
        bg_normalized   = F.normalize(self.bg_embed, dim=-1)
        text_normalized = F.normalize(self.text_emb.to(bg_normalized.dtype), dim=-1)

        # Prepend background so index 0 = background, indices 1..C = COCO classes
        return torch.cat([bg_normalized, text_normalized], dim=0)

    def forward(self, proj_feat: torch.Tensor) -> torch.Tensor:
        """Return per-proposal logits.  proj_feat: [..., D]  →  logits: [..., C+1]"""
        class_matrix      = self._text_matrix()
        similarity_scores = proj_feat @ class_matrix.T
        return similarity_scores / self.temperature
