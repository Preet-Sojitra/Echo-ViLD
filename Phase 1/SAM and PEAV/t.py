import json
from pathlib import Path

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
RESULTS_JSON_PATH = PROJECT_ROOT / "result" / "results.json"

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

#SAM checkpoint path 
SAM_CHECKPOINT = Path("/content/sam_vit_h_4b8939.pth")

#saver for mask cropping dubugging
SAVE_DEBUG_MASKS = True

#detecgtion = skip those are that below 0
MIN_PRED_SCORE = 0.0



if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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

    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT)).to(DEVICE)
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
    x1 = int(max(0, min(round(x1), width - 1)))
    y1 = int(max(0, min(round(y1), height - 1)))
    x2 = int(max(x1 + 1, min(round(x2), width)))
    y2 = int(max(y1 + 1, min(round(y2), height)))
    return x1, y1, x2, y2


def crop_from_box(image_bgr: np.ndarray, box_xyxy):
    #crop an image
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box_xyxy(box_xyxy, w, h)
    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def get_text_prompt(det: dict) -> str:
    #get text from peav
    return det.get("pred_class", "object")


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
    return masks[best_idx], float(scores[best_idx])


def blackout_background(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    blackout_value: int = 0
) -> np.ndarray:
    #blackout background
    masked = crop_bgr.copy()
    masked[~crop_mask] = blackout_value
    return masked


def save_embeddings_json(
    save_path: Path,
    baseline_embeds: np.ndarray,
    sam_nocontext_embeds: np.ndarray,
    sam_withcontext_equal_embeds: np.ndarray,
    sam_withcontext_80_20_embeds: np.ndarray,
    #full_image_embed: np.ndarray
):
    #conver to python list and save to JSON 
    data = {
        "baseline": baseline_embeds.tolist(),
        "sam_nocontext": sam_nocontext_embeds.tolist(),
        "sam_withcontext_equal": sam_withcontext_equal_embeds.tolist(),
        "sam_withcontext_80_20": sam_withcontext_80_20_embeds.tolist(),
        #"full_image_embed": full_image_embed.tolist(),
    }

    with open(save_path, "w") as f:
        json.dump(data, f)


def save_metadata_json(save_path: Path, metadata: list):
    #save data to Json 
    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=2)


def process_single_image(
    image_id: str,
    detections: list,
    peav_model,
    peav_processor,
    sam_predictor
):
    """
    for each image 
        do the crop,peave,same, ... 
    """
    #Image ID --> coco filename 
    image_filename = image_id_to_filename(image_id)
    image_path = IMAGES_DIR / image_filename

    #file != exists = skip
    if not image_path.exists():
        print(f"[WARN] Missing image: {image_path}")
        return

    #load the full image
    full_bgr = cv2.imread(str(image_path))
    if full_bgr is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    print(f"\nProcessing image {image_id} -> {image_filename} with {len(detections)} detections")

 
    #use the first detection's class as a placeholder full-image text prompt
    # full_text_prompt = get_text_prompt(detections[0]) if detections else "object"

    #compute full-image visual embedding once
    # full_img_embed = get_video_embed_from_image(
    #     peav_model,
    #     peav_processor,
    #     full_bgr,
    #     full_text_prompt
    # )

    #lits for embeddings 
    baseline_embeds = []
    sam_nocontext_embeds = []
    sam_withcontext_equal_embeds = []
    sam_withcontext_80_20_embeds = []

    #for row mapping and debugging
    metadata = []

    #loop
    for det_idx, det in enumerate(detections):
        print(f"  Detection {det_idx + 1}/{len(detections)}: start")
        pred_box = det.get("pred_box")
        pred_class = det.get("pred_class")
        pred_score = float(det.get("pred_score", 0.0))

        #skip low or invalid scores 
        if pred_box is None or pred_score < MIN_PRED_SCORE:
            print(f"  Detection {det_idx + 1}/{len(detections)}: skipped")
            continue

        #choose text prompt 
        text_prompt = get_text_prompt(det)

        #crop only once 
        crop_bgr, fixed_box = crop_from_box(full_bgr, pred_box)

        #if empty crop = skip 
        if crop_bgr.size == 0:
            print(f"  Detection {det_idx + 1}/{len(detections)}: empty crop, skipped")
            continue

        print(f"  Detection {det_idx + 1}/{len(detections)}: baseline PEAV")
        # Have bboxes (300,4) -> crop rectangular boxes -> peav 
        baseline_emb = get_video_embed_from_image(
            peav_model,
            peav_processor,
            crop_bgr,
            text_prompt
        )

        print(f"  Detection {det_idx + 1}/{len(detections)}: SAM")
        #Have bboxes -> crop -> apply SAM -> blackout bg -> peav -> save -> will be our sam_nocontext embeddings
        crop_mask, sam_score = get_sam_mask_on_crop(sam_predictor, crop_bgr)

        masked_crop_bgr = blackout_background(crop_bgr, crop_mask, blackout_value=0)

        print(f"  Detection {det_idx + 1}/{len(detections)}: masked PEAV")
        sam_nocontext_emb = get_video_embed_from_image(
            peav_model,
            peav_processor,
            masked_crop_bgr,
            text_prompt
        )

        print(f"  Detection {det_idx + 1}/{len(detections)}: done")
        #Have bboxes -> crop -> SAM -> blackout bg -> peav -> average with full img ->  will be our sam_withcontext embeddings + weights 
        sam_withcontext_equal = l2_normalize(
            0.5 * sam_nocontext_emb + 0.5 * baseline_emb
        )

        sam_withcontext_80_20 = l2_normalize(
            0.8 * sam_nocontext_emb + 0.2 * baseline_emb
        )

        #store results 
        baseline_embeds.append(baseline_emb)
        sam_nocontext_embeds.append(sam_nocontext_emb)
        sam_withcontext_equal_embeds.append(sam_withcontext_equal)
        sam_withcontext_80_20_embeds.append(sam_withcontext_80_20)

        #save data for the detections 
        row = {
            "det_idx": det_idx,
            "pred_class": pred_class,
            "pred_score": pred_score,
            "pred_box_original": pred_box,
            "pred_box_clamped": list(fixed_box),
            "text_prompt": text_prompt,
            "sam_score": sam_score,
            "mask_area_ratio": float(crop_mask.mean()) if crop_mask.size > 0 else 0.0,
        }

        #saving mask for debugging 
        if SAVE_DEBUG_MASKS:
            debug_path = DEBUG_MASK_DIR / f"{image_id}_{det_idx:04d}_masked.png"
            cv2.imwrite(str(debug_path), masked_crop_bgr)
            row["masked_crop_path"] = str(debug_path)

        metadata.append(row)

    #no valid detections = stop 
    if len(baseline_embeds) == 0:
        print(f"[WARN] No valid detections for image {image_id}")
        return

    #convering embedded list to NumPy arrays
    baseline_embeds = np.stack(baseline_embeds, axis=0)
    sam_nocontext_embeds = np.stack(sam_nocontext_embeds, axis=0)
    sam_withcontext_equal_embeds = np.stack(sam_withcontext_equal_embeds, axis=0)
    sam_withcontext_80_20_embeds = np.stack(sam_withcontext_80_20_embeds, axis=0)

    #save embeddings 
    return {
    "baseline": baseline_embeds.tolist(),
    "sam_nocontext": sam_nocontext_embeds.tolist(),
    "sam_withcontext_equal": sam_withcontext_equal_embeds.tolist(),
    "sam_withcontext_80_20": sam_withcontext_80_20_embeds.tolist(),
    "metadata": metadata,
    }

    # print(f"[DONE] {image_id}")
    # print("  baseline:", baseline_embeds.shape)
    # print("  sam_nocontext:", sam_nocontext_embeds.shape)
    # print("  sam_withcontext_equal:", sam_withcontext_equal_embeds.shape)
    # print("  sam_withcontext_80_20:", sam_withcontext_80_20_embeds.shape)




def main():

    all_baseline = {}
    all_sam_nocontext = {}
    all_sam_withcontext_equal = {}
    all_sam_withcontext_80_20 = {}
    all_metadata = {}

    #output folder creation 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_DEBUG_MASKS:
        DEBUG_MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RESULTS_JSON_PATH exists:", RESULTS_JSON_PATH.exists())
    print("IMAGES_DIR exists:", IMAGES_DIR.exists())
    print("OUTPUT_DIR exists:", OUTPUT_DIR.exists())
    #print("PEAV_MODEL_PATH exists:", PEAV_MODEL_PATH.exists())
    print("SAM_CHECKPOINT exists:", SAM_CHECKPOINT.exists())
    #load models 
    peav_model, peav_processor = load_peav()
    sam_predictor = load_sam()

    #load results 
    with open(RESULTS_JSON_PATH, "r") as f:
        predictions = json.load(f)

    #load each image in json 
    for image_id, detections in predictions.items():
        result = process_single_image(
        image_id=image_id,
        detections=detections,
        peav_model=peav_model,
        peav_processor=peav_processor,
        sam_predictor=sam_predictor
    )

        if result is None:
            continue

        all_baseline[image_id] = result["baseline"]
        all_sam_nocontext[image_id] = result["sam_nocontext"]
        all_sam_withcontext_equal[image_id] = result["sam_withcontext_equal"]
        all_sam_withcontext_80_20[image_id] = result["sam_withcontext_80_20"]
        all_metadata[image_id] = result["metadata"]

    with open(OUTPUT_DIR / "all_baseline.json", "w") as f:
        json.dump(all_baseline, f)

    with open(OUTPUT_DIR / "all_sam_nocontext.json", "w") as f:
        json.dump(all_sam_nocontext, f)

    with open(OUTPUT_DIR / "all_sam_withcontext_equal.json", "w") as f:
        json.dump(all_sam_withcontext_equal, f)

    with open(OUTPUT_DIR / "all_sam_withcontext_80_20.json", "w") as f:
        json.dump(all_sam_withcontext_80_20, f)

    with open(OUTPUT_DIR / "all_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)


if __name__ == "__main__":
    main()