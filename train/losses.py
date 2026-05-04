import torch
import torch.nn.functional as F


def splitting_loss(
    proj_feat: torch.Tensor,
    teacher_emb: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_distill: float = 0.5,
    valid_mask: torch.Tensor | None = None,
):
    """Compute total_loss = cls_loss + lambda_distill * distill_loss.

    Args:
        proj_feat:      Student projections   [B, N, D]
        teacher_emb:    PE-AV teacher embeds  [B, N, D]  (padded; zeros where invalid)
        logits:         Class logits          [B, N, C+1]  (C classes + 1 background)
        labels:         GT category ids       [B, N]        (0 = background)
        lambda_distill: Weight on the distillation term.
        valid_mask:     Boolean mask          [B, N]  True = teacher embedding exists.
                        If None, all proposals are assumed valid.

    Returns:
        (total_loss, cls_loss, distill_loss)
    """
    batch_size, num_proposals, embed_dim = proj_feat.shape

    # Classification loss (CE) on ALL proposals
    # Every proposal has a label from IoU matching, so CE trains the
    # classification head (including the learnable background embedding) on all 300.
    flat_logits = logits.float().reshape(batch_size * num_proposals, -1)
    flat_labels = labels.reshape(batch_size * num_proposals)
    cls_loss = F.cross_entropy(flat_logits, flat_labels)

    # Distillation loss (L1) on VALID proposals only 
    # Only proposals that have a teacher embedding contribute to L1.
    if valid_mask is None:
        valid_mask = torch.ones(batch_size, num_proposals, dtype=torch.bool, device=proj_feat.device)

    valid_proj = proj_feat[valid_mask].float()       # (V, D)
    valid_teacher = teacher_emb[valid_mask].float()  # (V, D)

    if valid_proj.numel() == 0:
        # Edge case: no valid proposals in this batch
        distill_loss = torch.tensor(0.0, device=proj_feat.device, requires_grad=True)
    else:
        distill_loss = F.l1_loss(valid_proj, valid_teacher)

    total_loss = cls_loss + lambda_distill * distill_loss
    return total_loss, cls_loss, distill_loss
