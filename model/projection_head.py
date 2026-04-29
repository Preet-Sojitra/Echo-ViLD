"""
Neural Network Architecture
The MLP code (256D -> 1024D)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP that maps Mask R-CNN 256-D RoIAlign features into PE-AV's 1024-D space.

    Output is L2-normalized so it can be directly dot-producted with PE-AV embeddings.
    """

    def __init__(self, roi_dim: int = 256, hidden_dim: int = 512, embed_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(roi_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project RoI features and L2-normalize.  x: [..., 256]  →  out: [..., 1024]"""
        return F.normalize(self.net(x), dim=-1)
