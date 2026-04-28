import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download, HfApi
from dotenv import load_dotenv

from utils.sam_peav_utils import (
    load_peav,
    load_sam,
    image_id_to_filename,
    crop_from_box,
    get_text_prompt,
    tensor_to_numpy,
    get_video_embed_from_image,
    get_video_embed_batch_from_images,
    get_sam_masks_batched,
    blackout_background,
    l2_normalize,
    pick_debug_indices,
    make_binary_mask_image,
    make_overlay,
    make_inspection_grid,
    load_detection_pt
)

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM and PEAV targets from bounded boxes.")
    parser.add_argument("--images_dir", required=True, help="Path to the COCO images directory (e.g. train2017)")
    parser.add_argument("--sam_checkpoint", required=True, help="Path to the SAM checkpoint file")
    parser.add_argument("--output_dir", default="./sam_peav_outputs", help="Output directory to save target PT files")
    parser.add_argument("--min_pred_score", type=float, default=0.0, help="Minimum prediction score for crops")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (limits samples and saves inspection grids)")
    parser.add_argument("--debug_samples", type=int, default=5, help="Number of files to process when in debug mode")
    parser.add_argument("--debug_max_detections", type=int, default=4, help="Max detections per image to visualize in debug mode")
    parser.add_argument("--peav_batch_size", type=int, default=32, help="Batch size for PE-AV inferences")
    parser.add_argument("--sam_batch_size", type=int, default=8, help="Batch size for SAM ViT inferences")
    
    args = parser.parse_args()
    return args

def process_single_image(
    image_id: str,
    detections: dict,
    peav_model,
    peav_processor,
    sam_predictor,
    images_dir: Path,
    device: torch.device,
    debug_config: dict,
    peav_batch_size: int,
    sam_batch_size: int
):
    """
    For each image: do the crop, peav, sam, etc (Batched approach).
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

    valid_det_indices = []
    valid_pred_scores = []
    valid_text_prompts = []
    valid_original_boxes = []
    valid_clamped_boxes = []
    
    bgr_crops = []

    # 1. Collect all valid crops
    for det_idx, pred_box in enumerate(boxes):
        pred_class = labels[det_idx] if det_idx < len(labels) else "object"
        pred_score = float(scores[det_idx]) if det_idx < len(scores) else 0.0

        if pred_box is None or pred_score < debug_config["min_pred_score"]:
            continue

        text_prompt = get_text_prompt(pred_class)
        crop_bgr, fixed_box = crop_from_box(full_bgr, pred_box)

        if crop_bgr.size == 0:
            continue

        bgr_crops.append(crop_bgr)
        valid_text_prompts.append(text_prompt)
        valid_det_indices.append(det_idx)
        valid_pred_scores.append(pred_score)
        valid_original_boxes.append(pred_box)
        valid_clamped_boxes.append(fixed_box)

    if len(bgr_crops) == 0:
        return None

    debug_indices = set()
    if debug_config["enabled"]:
        debug_indices = pick_debug_indices(len(bgr_crops), debug_config["max_detections"])

    print(f"{image_id} collected {len(bgr_crops)} valid crops. Running batched inferences...")

    # 2. Batched Baseline PE-AV
    t0 = time.perf_counter()
    baseline_embeds = get_video_embed_batch_from_images(
        peav_model, peav_processor, bgr_crops, valid_text_prompts, device, peav_batch_size
    )
    print(f"{image_id} batched baseline_peav {time.perf_counter() - t0:.4f}s")

    # 3. Batched SAM (Option B - isolated crops)
    t0 = time.perf_counter()
    sam_masks, sam_scores = get_sam_masks_batched(sam_predictor.model, bgr_crops, device, sam_batch_size)
    print(f"{image_id} batched sam {time.perf_counter() - t0:.4f}s")
    
    # 4. Apply blackout to get masked crops
    masked_crops = []
    for crop_bgr, mask in zip(bgr_crops, sam_masks):
        masked_crops.append(blackout_background(crop_bgr, mask, blackout_value=0))

    # 5. Batched Masked PE-AV
    t0 = time.perf_counter()
    sam_nocontext_embeds = get_video_embed_batch_from_images(
        peav_model, peav_processor, masked_crops, valid_text_prompts, device, peav_batch_size
    )
    print(f"{image_id} batched masked_peav {time.perf_counter() - t0:.4f}s")

    # 6. Gather all outputs and calculate Contextual Embeddings
    sam_withcontext_equal_embeds = []
    sam_withcontext_80_20_embeds = []
    metadata = []

    for i in range(len(bgr_crops)):
        det_idx = valid_det_indices[i]
        sam_nocontext_emb = sam_nocontext_embeds[i]
        
        sam_withcontext_equal = l2_normalize(
            0.5 * sam_nocontext_emb + 0.5 * full_img_embed
        )
        sam_withcontext_80_20 = l2_normalize(
            0.8 * sam_nocontext_emb + 0.2 * full_img_embed
        )

        sam_withcontext_equal_embeds.append(sam_withcontext_equal)
        sam_withcontext_80_20_embeds.append(sam_withcontext_80_20)

        crop_mask = sam_masks[i]
        crop_bgr = bgr_crops[i]
        masked_crop_bgr = masked_crops[i]

        row = {
            "det_idx": det_idx,
            "pred_class": valid_text_prompts[i],
            "pred_score": valid_pred_scores[i],
            "pred_box_original": np.asarray(valid_original_boxes[i]).tolist(),
            "pred_box_clamped": list(valid_clamped_boxes[i]),
            "text_prompt": valid_text_prompts[i],
            "sam_score": sam_scores[i],
            "mask_area_ratio": float(crop_mask.mean()) if crop_mask.size > 0 else 0.0,
        }

        if debug_config["enabled"] and i in debug_indices:
            base_name = f"{image_id}_{det_idx:04d}"
            
            masked_path = debug_config["mask_dir"] / f"{base_name}_masked.png"
            binary_mask_path = debug_config["binary_mask_dir"] / f"{base_name}_mask.png"
            overlay_path = debug_config["overlay_dir"] / f"{base_name}_overlay.png"
            grid_path = debug_config["grid_dir"] / f"{base_name}_grid.png"

            binary_mask_img = make_binary_mask_image(crop_mask)
            overlay = make_overlay(crop_bgr, crop_mask)
            grid = make_inspection_grid(crop_bgr, binary_mask_img, masked_crop_bgr, overlay)

            cv2.imwrite(str(masked_path), masked_crop_bgr)
            cv2.imwrite(str(binary_mask_path), binary_mask_img)
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(grid_path), grid)

            row["masked_crop_path"] = str(masked_path)
            row["binary_mask_path"] = str(binary_mask_path)
            row["overlay_path"] = str(overlay_path)
            row["inspection_grid_path"] = str(grid_path)

        metadata.append(row)

    return {
        "baseline": torch.from_numpy(baseline_embeds).half(),
        "sam_nocontext": torch.from_numpy(sam_nocontext_embeds).half(),
        "sam_withcontext_equal": torch.from_numpy(np.stack(sam_withcontext_equal_embeds, axis=0)).half(),
        "sam_withcontext_80_20": torch.from_numpy(np.stack(sam_withcontext_80_20_embeds, axis=0)).half(),
        "metadata": metadata,
    }


def main():
    args = parse_args()

    HF_REPO = "preetsojitra/Echo-VilD"
    HF_INPUT_FOLDER = "Bboxes_and_256D"
    HF_OUTPUT_FOLDER = "Sam_Peav_Outputs"

    PEAV_MODEL_ID = "facebook/pe-av-small-16-frame"
    
    SAM_CHECKPOINT = args.sam_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # HF Dataset Download
    print(f"Downloading/Using dataset from {HF_REPO}/{HF_INPUT_FOLDER}...")
    local_dir = snapshot_download(
        repo_id=HF_REPO, 
        repo_type="dataset", 
        allow_patterns=f"{HF_INPUT_FOLDER}/*",
        token=os.getenv("HF_TOKEN")
    )
    input_pt_dir = Path(local_dir) / HF_INPUT_FOLDER
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    all_baseline = output_dir / "all_baseline" # this is peav on simple cropped images
    all_sam_nocontext = output_dir / "all_sam_nocontext" # this is peav on cropped and then bg removed using sam
    all_sam_withcontext_equal = output_dir / "all_sam_withcontext_equal" # this is peav on cropped and then bg removed using sam and then combined with full image using equal weights
    all_sam_withcontext_80_20 = output_dir / "all_sam_withcontext_80_20" # this is peav on cropped and then bg removed using sam and then combined with full image using 80-20 weights
    all_metadata = output_dir / "all_metadata"

    for d in [all_baseline, all_sam_nocontext, all_sam_withcontext_equal, all_sam_withcontext_80_20, all_metadata]:
        d.mkdir(parents=True, exist_ok=True)

    debug_config = {
        "enabled": args.debug,
        "max_detections": args.debug_max_detections,
        "min_pred_score": args.min_pred_score,
    }

    if args.debug:
        debug_config["mask_dir"] = output_dir / "debug_masked_crops" # this saves the masked crops. masked crops are the cropped images with background removed using sam
        debug_config["binary_mask_dir"] = output_dir / "debug_binary_masks" # this saves the binary masks. binary masks are the masks of the foreground objects
        debug_config["overlay_dir"] = output_dir / "debug_overlays" # this saves the overlay of the masked crops. overlay is the masked crop with the original crop
        debug_config["grid_dir"] = output_dir / "debug_inspection_grids" # this saves the grid of the masked crops. grid is the overlay of the masked crops and the binary mask
        
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
        detections = load_detection_pt(pt_path)
        image_id = str(detections.get("img_id", pt_path.stem))

        # Strict Preemption Check (Ensures Job resumes correctly if killed)
        expected_files = [
            all_baseline / f"{image_id}.pt",
            all_sam_nocontext / f"{image_id}.pt",
            all_sam_withcontext_equal / f"{image_id}.pt",
            all_sam_withcontext_80_20 / f"{image_id}.pt",
            all_metadata / f"{image_id}.pt"
        ]
        
        if all(p.exists() for p in expected_files):
            print(f"\n[{i+1}/{len(pt_files)}] Skipping {pt_path.name} - Already fully processed.")
            continue

        print(f"\n[{i+1}/{len(pt_files)}] Processing {pt_path.name}")

        try:    
            result = process_single_image(
                image_id=image_id,
                detections=detections,
                peav_model=peav_model,
                peav_processor=peav_processor,
                sam_predictor=sam_predictor,
                images_dir=images_dir,
                device=device,
                debug_config=debug_config,
                peav_batch_size=args.peav_batch_size,
                sam_batch_size=args.sam_batch_size
            )

            if result is None:
                continue

            torch.save(result["baseline"], all_baseline / f"{image_id}.pt")
            torch.save(result["sam_nocontext"], all_sam_nocontext / f"{image_id}.pt")
            torch.save(result["sam_withcontext_equal"], all_sam_withcontext_equal / f"{image_id}.pt")
            torch.save(result["sam_withcontext_80_20"], all_sam_withcontext_80_20 / f"{image_id}.pt")
            torch.save(result["metadata"], all_metadata / f"{image_id}.pt")
        except Exception as e:
            print(f"\n[ERROR] Failed processing image: {image_id}: {e}")
            continue

    print(f"\nFinished local processing! Outputs are available in: {output_dir}")

    # Synchronize back to HuggingFace
    if not args.debug:
        print(f"Uploading output directory to Hugging Face: {HF_REPO}/{HF_OUTPUT_FOLDER}")
        api = HfApi()
        try:
            api.upload_folder(
                folder_path=str(output_dir),
                path_in_repo=HF_OUTPUT_FOLDER,
                repo_id=HF_REPO,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN")
            )
            print("Successfully synced to Hugging Face!")
        except Exception as e:
            print(f"[ERROR] Failed to upload to Hugging Face: {e}")

if __name__ == "__main__":
    main()
