"""Smoke test — validates the full pipeline on random tensors. No data download required."""
import sys
from pathlib import Path

# Add repo root so absolute imports resolve without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.projection_head import ProjectionHead
from train.losses import splitting_loss


class _MockClassificationHead(nn.Module):
    """Stands in for ClassificationHead using random frozen text embeddings (no HF download)."""

    def __init__(self, num_classes=80, embed_dim=1024, temperature=0.01):
        super().__init__()
        self.temperature = temperature
        self.bg_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Random unit-norm text embeddings — never updated during training
        random_text = torch.randn(num_classes, embed_dim)
        self.register_buffer("text_emb", F.normalize(random_text, dim=-1))

    def forward(self, proj_feat):
        bg_normalized   = F.normalize(self.bg_embed, dim=-1)
        text_normalized = F.normalize(self.text_emb.to(bg_normalized.dtype), dim=-1)
        class_matrix    = torch.cat([bg_normalized, text_normalized], dim=0)  # [C+1, D]

        similarity_scores = proj_feat @ class_matrix.T
        return similarity_scores / self.temperature


def run_smoke_test(steps: int = 10, batch_size: int = 2):
    """Run `steps` training iterations on random data and assert output shapes are correct."""
    print("Running smoke test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    proj_head = ProjectionHead().to(device)
    cls_head  = _MockClassificationHead().to(device)
    params    = list(proj_head.parameters()) + [cls_head.bg_embed]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    for step in range(steps):
        roi_feat    = torch.randn(batch_size, 300, 256, device=device)
        labels      = torch.randint(0, 81, (batch_size, 300), device=device)
        teacher_emb = torch.randn(batch_size, 300, 1024, device=device)

        optimizer.zero_grad()
        proj_feat   = proj_head(roi_feat)
        logits      = cls_head(proj_feat)
        total_loss, cls_loss, distill_loss = splitting_loss(proj_feat, teacher_emb, logits, labels)
        total_loss.backward()
        optimizer.step()

        print(f"  step {step+1:2d}: loss={total_loss.item():.4f}  "
              f"cls={cls_loss.item():.4f}  distill={distill_loss.item():.4f}")

    # Shape assertions
    assert proj_feat.shape == (batch_size, 300, 1024), f"proj_feat shape wrong: {proj_feat.shape}"
    assert logits.shape    == (batch_size, 300, 81),   f"logits shape wrong: {logits.shape}"
    assert cls_head.bg_embed.requires_grad
    assert not cls_head.text_emb.requires_grad

    print("\nAll shape assertions passed.")
    print("Smoke test PASSED.")


if __name__ == "__main__":
    run_smoke_test()
