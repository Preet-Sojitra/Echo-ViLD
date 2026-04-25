import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

from utils.sam_peav_utils import (
    load_peav,
    load_sam,
    image_id_to_filename,
    crop_from_box,
    get_text_prompt,
    tensor_to_numpy,
    get_video_embed_from_image,
    get_sam_mask_on_crop,
    blackout_background,
    l2_normalize,
    pick_debug_indices,
    make_binary_mask_image,
    make_overlay,
    make_inspection_grid,
    load_detection_pt
)

load_dotenv()

def process_single_image(
    image_id: str,
    detections: dict,
    peav_model,
    peav_processor,
    sam_predictor,
    images_dir: Path,
    device: torch.device,
    debug_config: dict
):
    """
    For each image: do the crop, peav, sam, etc.
    """
    file_name = detections.get("file_name")
    boxes = tensor_to_numpy(detections.get("boxes", []))
    scores = tensor_to_numpy(detections.get("roi_scores", []))
    labels = detections.get("labels", None)

    if labels is None:
        print(f"[Warning] No labels found for image: {image_id}")
        labels = ["object"] * len(boxes)
    elif isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    image_filename = str(file_name) if file_name else image_id_to_filename(image_id)
    image_path = images_dir / image_filename

    if not image_path.exists():
        print(f"[Warning] Image not found: {image_path}")
        return None

    full_bgr = cv2.imread(str(image_path))
    if full_bgr is None:
        print(f"[Warning] Failed to read image: {image_path}")
        return None

    full_text_prompt = get_text_prompt(labels[0]) if len(labels) > 0 else "object"

    t0 = time.perf_counter()
    full_img_embed = get_video_embed_from_image(
        peav_model,
        peav_processor,
        full_bgr,
        full_text_prompt,
        device
    )
    print(f"{image_id} full_image_peav {time.perf_counter() - t0:.4f}s")

    baseline_embeds = []
    sam_nocontext_embeds = []
    sam_withcontext_equal_embeds = []
    sam_withcontext_80_20_embeds = []
    metadata = []

    debug_indices = set()
    if debug_config["enabled"]:
        debug_indices = pick_debug_indices(len(boxes), debug_config["max_detections"])

    for det_idx, pred_box in enumerate(boxes):
        pred_class = labels[det_idx] if det_idx < len(labels) else "object"
        pred_score = float(scores[det_idx]) if det_idx < len(scores) else 0.0

        if pred_box is None or pred_score < debug_config["min_pred_score"]:
            continue

        text_prompt = get_text_prompt(pred_class)
        crop_bgr, fixed_box = crop_from_box(full_bgr, pred_box)

        if crop_bgr.size == 0:
            continue

        t0 = time.perf_counter()
        baseline_emb = get_video_embed_from_image(
            peav_model,
            peav_processor,
            crop_bgr,
            text_prompt,
            device
        )
        print(f"{image_id} det_{det_idx} baseline_peav {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        crop_mask, sam_score = get_sam_mask_on_crop(sam_predictor, crop_bgr)
        masked_crop_bgr = blackout_background(crop_bgr, crop_mask, blackout_value=0)
        print(f"{image_id} det_{det_idx} sam_blackout {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        sam_nocontext_emb = get_video_embed_from_image(
            peav_model,
            peav_processor,
            masked_crop_bgr,
            text_prompt,
            device
        )
        print(f"{image_id} det_{det_idx} masked_peav {time.perf_counter() - t0:.4f}s")

        sam_withcontext_equal = l2_normalize(
            0.5 * sam_nocontext_emb + 0.5 * full_img_embed
        )
        sam_withcontext_80_20 = l2_normalize(
            0.8 * sam_nocontext_emb + 0.2 * full_img_embed
        )

        baseline_embeds.append(baseline_emb)
        sam_nocontext_embeds.append(sam_nocontext_emb)
        sam_withcontext_equal_embeds.append(sam_withcontext_equal)
        sam_withcontext_80_20_embeds.append(sam_withcontext_80_20)

        row = {
            "det_idx": det_idx,
            "pred_class": text_prompt,
            "pred_score": pred_score,
            "pred_box_original": np.asarray(pred_box).tolist(),
            "pred_box_clamped": list(fixed_box),
            "text_prompt": text_prompt,
            "sam_score": sam_score,
            "mask_area_ratio": float(crop_mask.mean()) if crop_mask.size > 0 else 0.0,
        }

        if debug_config["enabled"] and det_idx in debug_indices:
            base_name = f"{image_id}_{det_idx:04d}"
            
            masked_path = debug_config["mask_dir"] / f"{base_name}_masked.png"
            binary_mask_path = debug_config["binary_mask_dir"] / f"{base_name}_mask.png"
            overlay_path = debug_config["overlay_dir"] / f"{base_name}_overlay.png"
            grid_path = debug_config["grid_dir"] / f"{base_name}_grid.png"

            binary_mask = make_binary_mask_image(crop_mask)
            overlay = make_overlay(crop_bgr, crop_mask)
            grid = make_inspection_grid(crop_bgr, binary_mask, masked_crop_bgr, overlay)

            cv2.imwrite(str(masked_path), masked_crop_bgr)
            cv2.imwrite(str(binary_mask_path), binary_mask)
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(grid_path), grid)

            row["masked_crop_path"] = str(masked_path)
            row["binary_mask_path"] = str(binary_mask_path)
            row["overlay_path"] = str(overlay_path)
            row["inspection_grid_path"] = str(grid_path)

        metadata.append(row)

    if len(baseline_embeds) == 0:
        return None

    return {
        "baseline": torch.from_numpy(np.stack(baseline_embeds, axis=0)).float(),
        "sam_nocontext": torch.from_numpy(np.stack(sam_nocontext_embeds, axis=0)).float(),
        "sam_withcontext_equal": torch.from_numpy(np.stack(sam_withcontext_equal_embeds, axis=0)).float(),
        "sam_withcontext_80_20": torch.from_numpy(np.stack(sam_withcontext_80_20_embeds, axis=0)).float(),
        "metadata": metadata,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM and PEAV targets from bounded boxes.")
    parser.add_argument("--images_dir", required=True, help="Path to the COCO images directory (e.g. train2017)")
    parser.add_argument("--sam-checkpoint", required=True, help="Path to the SAM checkpoint file")
    parser.add_argument("--output_dir", default="./sam_peav_outputs", help="Output directory to save target PT files")
    parser.add_argument("--min_pred_score", type=float, default=0.0, help="Minimum prediction score for crops")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (limits samples and saves inspection grids)")
    parser.add_argument("--debug_samples", type=int, default=5, help="Number of files to process when in debug mode")
    parser.add_argument("--debug_max_detections", type=int, default=4, help="Max detections per image to visualize in debug mode")
    
    args = parser.parse_args()
    
    return args

def main():

    HF_INPUT_REPO = "preetsojitra/Echo-VilD"
    HF_INPUT_FOLDER = "Bboxes_and_256D"

    PEAV_MODEL_ID = "facebook/pe-av-small-16-frame"
    SAM_CHECKPOINT = args.sam_checkpoint
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # HF Dataset Download
    print(f"Downloading/Using dataset from {HF_INPUT_REPO}/{HF_INPUT_FOLDER}...")
    local_dir = snapshot_download(
        repo_id=HF_INPUT_REPO, 
        repo_type="dataset", 
        allow_patterns=f"{HF_INPUT_FOLDER}/*",
        token=os.getenv("HF_TOKEN")
    )
    input_pt_dir = Path(local_dir) / HF_INPUT_FOLDER
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    all_baseline = output_dir / "all_baseline"
    all_sam_nocontext = output_dir / "all_sam_nocontext"
    all_sam_withcontext_equal = output_dir / "all_sam_withcontext_equal"
    all_sam_withcontext_80_20 = output_dir / "all_sam_withcontext_80_20"
    all_metadata = output_dir / "all_metadata"

    for d in [all_baseline, all_sam_nocontext, all_sam_withcontext_equal, all_sam_withcontext_80_20, all_metadata]:
        d.mkdir(parents=True, exist_ok=True)

    debug_config = {
        "enabled": args.debug,
        "max_detections": args.debug_max_detections,
        "min_pred_score": args.min_pred_score,
    }

    if args.debug:
        debug_config["mask_dir"] = output_dir / "debug_masked_crops"
        debug_config["binary_mask_dir"] = output_dir / "debug_binary_masks"
        debug_config["overlay_dir"] = output_dir / "debug_overlays"
        debug_config["grid_dir"] = output_dir / "debug_inspection_grids"
        
        for p in ["mask_dir", "binary_mask_dir", "overlay_dir", "grid_dir"]:
            debug_config[p].mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    peav_model, peav_processor = load_peav(PEAV_MODEL_ID, device)
    sam_predictor = load_sam(SAM_CHECKPOINT, device)

    pt_files = sorted(input_pt_dir.glob("*.pt"))
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {input_pt_dir}")

    if args.debug:
        print(f"Running in debug mode. Limiting processing to {args.debug_samples} files.")
        pt_files = pt_files[:args.debug_samples]

    print(f"Processing {len(pt_files)} extracted MaskRCNN feature files...")

    for i, pt_path in enumerate(pt_files):
        print(f"[{i+1}/{len(pt_files)}] Processing {pt_path.name}")
        detections = load_detection_pt(pt_path)
        image_id = detections.get("img_id", pt_path.stem)

        result = process_single_image(
            image_id=image_id,
            detections=detections,
            peav_model=peav_model,
            peav_processor=peav_processor,
            sam_predictor=sam_predictor,
            images_dir=images_dir,
            device=device,
            debug_config=debug_config
        )

        if result is None:
            continue

        image_id = str(image_id)
        torch.save(result["baseline"], all_baseline / f"{image_id}.pt")
        torch.save(result["sam_nocontext"], all_sam_nocontext / f"{image_id}.pt")
        torch.save(result["sam_withcontext_equal"], all_sam_withcontext_equal / f"{image_id}.pt")
        torch.save(result["sam_withcontext_80_20"], all_sam_withcontext_80_20 / f"{image_id}.pt")
        torch.save(result["metadata"], all_metadata / f"{image_id}.pt")

    print(f"Finished processing! Outputs are available in: {output_dir}")

if __name__ == "__main__":
    main()
