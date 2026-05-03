"""
Main training loop, L1 + CE Loss, saves .pth weights in "weights" folder
"""
import argparse
import csv
import os
import sys
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

import yaml
from dotenv import load_dotenv

load_dotenv()

# Add repo root to sys.path so absolute imports (datasets.*, models.*, train.*) resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.vild_dataset import EchoViLDDataset
from model.projection_head import ProjectionHead
from model.classification_head import ClassificationHead
from train.losses import splitting_loss


def build_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """LR schedule: linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def upload_weights_to_hf(ckpt_dir: Path, model_id: str):
    """Upload best.pth and last.pth to the HuggingFace dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "preetsojitra/Echo-VilD"
    token = os.environ.get("HF_TOKEN", None)

    for name in ["best.pth", "last.pth"]:
        local_path = ckpt_dir / name
        if local_path.exists():
            remote_path = f"weights/{model_id}/{name}"
            print(f"Uploading {local_path} → {repo_id}/{remote_path}")
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  ✓ Uploaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to upload {name}: {e}")


def train(config: dict, mock_teacher: bool = False):
    """Run the full training loop defined by config.

    Args:
        config:       Dict loaded from a YAML config file.
        mock_teacher: If True, substitute random tensors for PE-AV teacher embeddings
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("amp", "fp16") == "fp16" and device.type == "cuda"

    torch.manual_seed(config.get("seed", 42))

    # --- Dataset & loaders ---
    dataset = EchoViLDDataset(
        image_emb_variant=config["image_emb_variant"],
        mock_teacher=mock_teacher,
    )
    val_size = max(1, int(0.1 * len(dataset)))
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

    # --- Models ---
    proj_head = ProjectionHead(
        roi_dim=config.get("roi_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        embed_dim=config.get("embed_dim", 1024),
    ).to(device)

    cls_head = ClassificationHead(
        text_emb_variant=config["text_emb_variant"],
        num_classes=config.get("num_classes", 80),
        embed_dim=config.get("embed_dim", 1024),
        temperature=config.get("temperature", 0.01),
    ).to(device)

    # Only proj_head params + learnable bg_embed are optimized; text_emb is a frozen buffer
    trainable_params = list(proj_head.parameters()) + [cls_head.bg_embed]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    total_steps = len(train_loader) * config.get("epochs", 10)
    # Suppress harmless PyTorch warning: LambdaLR.__init__ internally calls
    # step() before any optimizer.step(), triggering a false-positive.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=config.get("warmup_steps", 500),
            total_steps=total_steps,
        )
    scaler = GradScaler(device.type, enabled=use_amp)

    # --- Logging ---
    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.get("log_csv", str(ckpt_dir / "loss.csv")))
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "split", "loss", "cls_loss", "distill_loss"])

    lambda_distill = config.get("lambda_distill", 0.5)
    best_val_loss  = float("inf")

    # --- Training loop ---
    for epoch in range(1, config.get("epochs", 10) + 1):
        proj_head.train()
        cls_head.train()

        train_loss         = 0.0
        train_cls_loss     = 0.0
        train_distill_loss = 0.0

        for roi_feat, labels, teacher_emb, valid_mask in train_loader:
            roi_feat    = roi_feat.to(device).float()
            labels      = labels.to(device)
            teacher_emb = teacher_emb.to(device).float()
            valid_mask  = valid_mask.to(device)

            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                proj_feat_out = proj_head(roi_feat)
                logits        = cls_head(proj_feat_out)
                total_loss, cls_loss, distill_loss = splitting_loss(
                    proj_feat_out, teacher_emb, logits, labels,
                    lambda_distill, valid_mask,
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss         += total_loss.item()
            train_cls_loss     += cls_loss.item()
            train_distill_loss += distill_loss.item()

        num_train_batches = len(train_loader)
        csv_writer.writerow([
            epoch, "train",
            train_loss         / num_train_batches,
            train_cls_loss     / num_train_batches,
            train_distill_loss / num_train_batches,
        ])

        # --- Validation ---
        proj_head.eval()
        cls_head.eval()

        val_loss         = 0.0
        val_cls_loss     = 0.0
        val_distill_loss = 0.0

        with torch.no_grad():
            for roi_feat, labels, teacher_emb, valid_mask in val_loader:
                roi_feat    = roi_feat.to(device).float()
                labels      = labels.to(device)
                teacher_emb = teacher_emb.to(device).float()
                valid_mask  = valid_mask.to(device)

                with autocast(device.type, enabled=use_amp):
                    proj_feat_out = proj_head(roi_feat)
                    logits        = cls_head(proj_feat_out)
                    total_loss, cls_loss, distill_loss = splitting_loss(
                        proj_feat_out, teacher_emb, logits, labels,
                        lambda_distill, valid_mask,
                    )

                val_loss         += total_loss.item()
                val_cls_loss     += cls_loss.item()
                val_distill_loss += distill_loss.item()

        num_val_batches = len(val_loader)
        csv_writer.writerow([
            epoch, "val",
            val_loss         / num_val_batches,
            val_cls_loss     / num_val_batches,
            val_distill_loss / num_val_batches,
        ])
        log_file.flush()

        print(
            f"Epoch {epoch:3d} | "
            f"train {train_loss/num_train_batches:.4f} "
            f"(cls {train_cls_loss/num_train_batches:.4f} "
            f"dist {train_distill_loss/num_train_batches:.4f}) | "
            f"val {val_loss/num_val_batches:.4f} "
            f"(cls {val_cls_loss/num_val_batches:.4f} "
            f"dist {val_distill_loss/num_val_batches:.4f})"
        )

        # Save checkpoint — .pth extension per project convention
        checkpoint = {
            "proj":     proj_head.state_dict(),
            "bg_embed": cls_head.bg_embed.data,
            "epoch":    epoch,
        }
        torch.save(checkpoint, ckpt_dir / "last.pth")

        if val_loss / num_val_batches < best_val_loss:
            best_val_loss = val_loss / num_val_batches
            torch.save(checkpoint, ckpt_dir / "best.pth")
            print(f"  -> saved best.pth (val_loss={best_val_loss:.4f})")

    log_file.close()
    print(f"Done. Best checkpoint: {ckpt_dir / 'best.pth'}")

    # --- Upload trained weights to HuggingFace ---
    model_id = config.get("model_id", Path(config["ckpt_dir"]).name)
    upload_weights_to_hf(ckpt_dir, model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--mock-teacher", action="store_true",
                        help="Use random tensors as teacher embeddings")
    args = parser.parse_args()

    config = build_config(args.config)
    train(config, mock_teacher=args.mock_teacher)
