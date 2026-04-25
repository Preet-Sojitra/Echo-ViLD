"""Splitting loss: weighted sum of CE classification loss and L1 distillation loss."""
import torch
import torch.nn.functional as F


def splitting_loss(
    proj_feat: torch.Tensor,
    teacher_emb: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_distill: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total_loss = cls_loss + lambda_distill * distill_loss.

    Args:
        proj_feat:      Student projections   [B, N, D]
        teacher_emb:    PE-AV teacher embeds  [B, N, D]
        logits:         Class logits          [B, N, C+1]  (C classes + 1 background)
        labels:         GT category ids       [B, N]        (0 = background)
        lambda_distill: Weight on the distillation term.

    Returns:
        (total_loss, cls_loss, distill_loss)
    """
    batch_size, num_proposals, embed_dim = proj_feat.shape

    # L1 distance between student projections and PE-AV teacher embeddings
    distill_loss = F.l1_loss(proj_feat.float(), teacher_emb.float())

    # Flatten batch × proposals before cross-entropy  [B*N, C+1] vs [B*N]
    flat_logits = logits.float().reshape(batch_size * num_proposals, -1)
    flat_labels = labels.reshape(batch_size * num_proposals)
    cls_loss = F.cross_entropy(flat_logits, flat_labels)

    total_loss = cls_loss + lambda_distill * distill_loss
    return total_loss, cls_loss, distill_loss
