"""
Loads M1's .pt files, runs SAM/PE-AV, saves target .pt files

Code runs with the assumption that it is being run in google colab where Bboxes_and_256D and train2017 are directories in the Phase 1 directory in  google drive 
"""


from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from transformers import PeAudioVideoModel, PeAudioVideoProcessor

#project_root / "SAM and PEAV" / t.py
# SCRIPT_DIR = Path(__file__).resolve().parent

#SAM and PEAV = project root
PROJECT_ROOT = Path("/content/drive/MyDrive/Phase 1")

#results  
INPUT_PT_DIR = PROJECT_ROOT / "Bboxes_and_256D"

#images path 
IMAGES_DIR = PROJECT_ROOT / "train2017"

#PEAV path 
# PEAV_MODEL_PATH = Path("/content/pe-av-small-16-frame")
PEAV_MODEL_ID = "facebook/pe-av-small-16-frame"

#output folder for embedding 
##outputing inside colab for storage saving 
OUTPUT_DIR = Path("/content/sam_peav_outputs")

#debug folder 
DEBUG_MASK_DIR = OUTPUT_DIR / "debug_masked_crops"
DEBUG_BINARY_MASK_DIR = OUTPUT_DIR / "debug_binary_masks"
DEBUG_OVERLAY_DIR = OUTPUT_DIR / "debug_overlays"
DEBUG_GRID_DIR = OUTPUT_DIR / "debug_inspection_grids"

#SAM checkpoint path 
SAM_CHECKPOINT = Path("/content/sam_vit_l_0b3195.pth")

#saver for mask cropping dubugging
SAVE_DEBUG_MASKS = True
DEBUG_MAX_DETECTIONS_PER_IMAGE = 8
DEBUG_SAMPLE_SEED = 7

#detecgtion = skip those are that below 0
MIN_PRED_SCORE = 0.0


#use CPU if no GPU. I have NVIDIA GPU so I used this 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#load peav once 
def load_peav():
    model = PeAudioVideoModel.from_pretrained(
        PEAV_MODEL_ID,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    processor = PeAudioVideoProcessor.from_pretrained(str(PEAV_MODEL_ID))

    #olny doing interference 
    model.eval()

    return model, processor


#load sam model
def load_sam():
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT}")

    sam = sam_model_registry["vit_l"](checkpoint=str(SAM_CHECKPOINT)).to(DEVICE)
    predictor = SamPredictor(sam)

    return predictor



#helper functions 

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    #normalize vector to unit length for stable comparison 
    return vec / (np.linalg.norm(vec) + eps)


def image_id_to_filename(image_id: str) -> str:
    #change coco image to it's corresponding jpg name 
    return f"{int(image_id):012d}.jpg"


def clamp_box_xyxy(box, width: int, height: int):
    #clap coordinates bwfore cropping  
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(round(float(x1)), width - 1)))
    y1 = int(max(0, min(round(float(y1)), height - 1)))
    x2 = int(max(x1 + 1, min(round(float(x2)), width)))
    y2 = int(max(y1 + 1, min(round(float(y2)), height)))
    return x1, y1, x2, y2


def crop_from_box(image_bgr: np.ndarray, box_xyxy):
    #crop an image
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box_xyxy(box_xyxy, w, h)
    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def get_text_prompt(det_label) -> str:
    #get text from peav
    if isinstance(det_label, bytes):
        det_label = det_label.decode("utf-8", errors="ignore")
    if isinstance(det_label, str):
        det_label = det_label.strip()
        return det_label if det_label else "object"
    if isinstance(det_label, torch.Tensor):
        if det_label.numel() == 1:
            det_label = det_label.item()
    if isinstance(det_label, (int, np.integer)):
        return f"object_{int(det_label)}"
    if isinstance(det_label, (float, np.floating)):
        return f"object_{int(det_label)}"
    return "object"


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_video_embed_from_image(
    model,
    processor,
    image_bgr: np.ndarray,
    text_prompt: str
) -> np.ndarray:
    """
        Compute the Transformers PE-AV visual embedding.
            by creating a fake 16-frame video: it basically repeats the same image 16 times.
    """
    #OpenCV color converter 
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    #creating a fake 16-frame video: it basically repeats the same image 16 times.
    fake_video = [image_rgb for _ in range(16)]

    #using videos 
    inputs = processor(
        videos=[fake_video],
        text=[text_prompt],
        return_tensors="pt",
        padding=True
    )

    #mvoe rensor to device 
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    #run in inference mode
    with torch.inference_mode():
        outputs = model(**inputs)

    #pull out the visual embedding
    emb = outputs.video_embeds[0].float().detach().cpu().numpy()

    return l2_normalize(emb)


def get_sam_mask_on_crop(
    sam_predictor: SamPredictor,
    crop_bgr: np.ndarray
):
    #run SAM 
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(crop_rgb)

    h, w = crop_bgr.shape[:2]

    #SAM is run with full crop 
    input_box = np.array([0, 0, w - 1, h - 1], dtype=np.float32)

    masks, scores, _ = sam_predictor.predict(
        box=input_box,
        multimask_output=True
    )

    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])


def blackout_background(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    blackout_value: int = 0
) -> np.ndarray:
    #blackout background
    masked = crop_bgr.copy()
    masked[~crop_mask] = blackout_value
    return masked


def make_binary_mask_image(crop_mask: np.ndarray) -> np.ndarray:
    return (crop_mask.astype(np.uint8) * 255)


def make_overlay(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    overlay = crop_bgr.copy()
    green = np.zeros_like(crop_bgr)
    green[:, :, 1] = 255
    alpha = 0.35
    overlay[crop_mask] = cv2.addWeighted(crop_bgr[crop_mask], 1 - alpha, green[crop_mask], alpha, 0)
    return overlay


def make_inspection_grid(original_crop: np.ndarray, binary_mask: np.ndarray, masked_crop: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if binary_mask.ndim == 2:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    h = max(original_crop.shape[0], binary_mask.shape[0], masked_crop.shape[0], overlay.shape[0])

    def fit(img):
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    panels = [fit(original_crop), fit(binary_mask), fit(masked_crop), fit(overlay)]
    return np.concatenate(panels, axis=1)


def pick_debug_indices(num_items: int, max_items: int):
    if num_items <= max_items:
        return set(range(num_items))
    rng = random.Random(DEBUG_SAMPLE_SEED)
    return set(rng.sample(range(num_items), k=max_items))


def load_detection_pt(pt_path: Path):
    return torch.load(pt_path, map_location="cpu")


def process_single_image(
    image_id: str,
    detections: dict,
    peav_model,
    peav_processor,
    sam_predictor
):
    """
    for each image 
        do the crop,peave,same, ... 
    """
    file_name = detections.get("file_name")
    boxes = tensor_to_numpy(detections.get("boxes", []))
    scores = tensor_to_numpy(detections.get("roi_scores", []))
    labels = detections.get("labels", None)

    if labels is None:
        labels = ["object"] * len(boxes)
    elif isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    #Image ID --> coco filename 
    image_filename = str(file_name) if file_name else image_id_to_filename(image_id)
    image_path = IMAGES_DIR / image_filename

    #file != exists = skip
    if not image_path.exists():
        return

    #load the full image
    full_bgr = cv2.imread(str(image_path))
    if full_bgr is None:
        return

    full_text_prompt = get_text_prompt(labels[0]) if len(labels) > 0 else "object"

    t0 = time.perf_counter()
    full_img_embed = get_video_embed_from_image(
        peav_model,
        peav_processor,
        full_bgr,
        full_text_prompt
    )
    print(f"{image_id} full_image_peav {time.perf_counter() - t0:.4f}s")

    #lits for embeddings 
    baseline_embeds = []
    sam_nocontext_embeds = []
    sam_withcontext_equal_embeds = []
    sam_withcontext_80_20_embeds = []

    #for row mapping and debugging
    metadata = []

    debug_indices = pick_debug_indices(len(boxes), DEBUG_MAX_DETECTIONS_PER_IMAGE)

    #loop
    for det_idx, pred_box in enumerate(boxes):
        pred_class = labels[det_idx] if det_idx < len(labels) else "object"
        pred_score = float(scores[det_idx]) if det_idx < len(scores) else 0.0

        #skip low or invalid scores 
        if pred_box is None or pred_score < MIN_PRED_SCORE:
            continue

        #choose text prompt 
        text_prompt = get_text_prompt(pred_class)

        #crop only once 
        crop_bgr, fixed_box = crop_from_box(full_bgr, pred_box)

        #if empty crop = skip 
        if crop_bgr.size == 0:
            continue

        # Have bboxes (300,4) -> crop rectangular boxes -> peav 
        t0 = time.perf_counter()
        baseline_emb = get_video_embed_from_image(
            peav_model,
            peav_processor,
            crop_bgr,
            text_prompt
        )
        print(f"{image_id} det_{det_idx} baseline_peav {time.perf_counter() - t0:.4f}s")

        #Have bboxes -> crop -> apply SAM -> blackout bg -> peav -> save -> will be our sam_nocontext embeddings
        t0 = time.perf_counter()
        crop_mask, sam_score = get_sam_mask_on_crop(sam_predictor, crop_bgr)
        masked_crop_bgr = blackout_background(crop_bgr, crop_mask, blackout_value=0)
        print(f"{image_id} det_{det_idx} sam_blackout {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        sam_nocontext_emb = get_video_embed_from_image(
            peav_model,
            peav_processor,
            masked_crop_bgr,
            text_prompt
        )
        print(f"{image_id} det_{det_idx} masked_peav {time.perf_counter() - t0:.4f}s")

        #Have bboxes -> crop -> SAM -> blackout bg -> peav -> average with full img ->  will be our sam_withcontext embeddings + weights 
        sam_withcontext_equal = l2_normalize(
            0.5 * sam_nocontext_emb + 0.5 * full_img_embed
        )

        sam_withcontext_80_20 = l2_normalize(
            0.8 * sam_nocontext_emb + 0.2 * full_img_embed
        )

        #store results 
        baseline_embeds.append(baseline_emb)
        sam_nocontext_embeds.append(sam_nocontext_emb)
        sam_withcontext_equal_embeds.append(sam_withcontext_equal)
        sam_withcontext_80_20_embeds.append(sam_withcontext_80_20)

        #save data for the detections 
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

        #saving mask for debugging 
        if SAVE_DEBUG_MASKS and det_idx in debug_indices:
            base_name = f"{image_id}_{det_idx:04d}"

            masked_path = DEBUG_MASK_DIR / f"{base_name}_masked.png"
            binary_mask_path = DEBUG_BINARY_MASK_DIR / f"{base_name}_mask.png"
            overlay_path = DEBUG_OVERLAY_DIR / f"{base_name}_overlay.png"
            grid_path = DEBUG_GRID_DIR / f"{base_name}_grid.png"

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

    #no valid detections = stop 
    if len(baseline_embeds) == 0:
        return

    #convering embedded list to NumPy arrays
    baseline_embeds = np.stack(baseline_embeds, axis=0)
    sam_nocontext_embeds = np.stack(sam_nocontext_embeds, axis=0)
    sam_withcontext_equal_embeds = np.stack(sam_withcontext_equal_embeds, axis=0)
    sam_withcontext_80_20_embeds = np.stack(sam_withcontext_80_20_embeds, axis=0)

    #save embeddings 
    return {
        "baseline": torch.from_numpy(baseline_embeds).float(),
        "sam_nocontext": torch.from_numpy(sam_nocontext_embeds).float(),
        "sam_withcontext_equal": torch.from_numpy(sam_withcontext_equal_embeds).float(),
        "sam_withcontext_80_20": torch.from_numpy(sam_withcontext_80_20_embeds).float(),
        "metadata": metadata,
    }




def main():

    all_baseline = OUTPUT_DIR / "all_baseline"
    all_sam_nocontext = OUTPUT_DIR / "all_sam_nocontext"
    all_sam_withcontext_equal = OUTPUT_DIR / "all_sam_withcontext_equal"
    all_sam_withcontext_80_20 = OUTPUT_DIR / "all_sam_withcontext_80_20"
    all_metadata = OUTPUT_DIR / "all_metadata"

    #output folder creation 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_baseline.mkdir(parents=True, exist_ok=True)
    all_sam_nocontext.mkdir(parents=True, exist_ok=True)
    all_sam_withcontext_equal.mkdir(parents=True, exist_ok=True)
    all_sam_withcontext_80_20.mkdir(parents=True, exist_ok=True)
    all_metadata.mkdir(parents=True, exist_ok=True)

    if SAVE_DEBUG_MASKS:
        DEBUG_MASK_DIR.mkdir(parents=True, exist_ok=True)
        DEBUG_BINARY_MASK_DIR.mkdir(parents=True, exist_ok=True)
        DEBUG_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        DEBUG_GRID_DIR.mkdir(parents=True, exist_ok=True)

    peav_model, peav_processor = load_peav()
    sam_predictor = load_sam()

    pt_files = sorted(INPUT_PT_DIR.glob("*.pt"))
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {INPUT_PT_DIR}")

    #load each image in json 
    for pt_path in pt_files:
        detections = load_detection_pt(pt_path)

        image_id = detections.get("img_id", pt_path.stem)

        result = process_single_image(
            image_id=image_id,
            detections=detections,
            peav_model=peav_model,
            peav_processor=peav_processor,
            sam_predictor=sam_predictor
        )

        if result is None:
            continue

        image_id = str(image_id)

        torch.save(result["baseline"], all_baseline / f"{image_id}.pt")
        torch.save(result["sam_nocontext"], all_sam_nocontext / f"{image_id}.pt")
        torch.save(result["sam_withcontext_equal"], all_sam_withcontext_equal / f"{image_id}.pt")
        torch.save(result["sam_withcontext_80_20"], all_sam_withcontext_80_20 / f"{image_id}.pt")
        torch.save(result["metadata"], all_metadata / f"{image_id}.pt")


if __name__ == "__main__":
    main()
