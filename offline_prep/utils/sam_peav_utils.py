import random
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from transformers import PeAudioVideoModel, PeAudioVideoProcessor
import transformers

# Suppress "channel dimension is ambiguous" warning for very small crops (e.g. 4×4 px).
# We already pass input_data_format="channels_last" but the internal rescale pipeline
# still warns on tiny images.  The warning is harmless — PE-AV processes them correctly.
transformers.logging.set_verbosity_error()

def load_peav(model_id, device):
    model = PeAudioVideoModel.from_pretrained(
        model_id,
        low_cpu_mem_usage=True
    ).to(device)

    processor = PeAudioVideoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor

def load_sam(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    sam = sam_model_registry["vit_l"](checkpoint=str(checkpoint_path)).to(device)
    predictor = SamPredictor(sam)
    return predictor

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + eps)

def image_id_to_filename(image_id: str) -> str:
    return f"{int(image_id):012d}.jpg"

def clamp_box_xyxy(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(round(float(x1)), width - 1)))
    y1 = int(max(0, min(round(float(y1)), height - 1)))
    x2 = int(max(x1 + 1, min(round(float(x2)), width)))
    y2 = int(max(y1 + 1, min(round(float(y2)), height)))
    return x1, y1, x2, y2

def crop_from_box(image_bgr: np.ndarray, box_xyxy):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box_xyxy(box_xyxy, w, h)
    crop = image_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)

def get_text_prompt(det_label) -> str:
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
    text_prompt: str,
    device
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fake_video = [image_rgb for _ in range(16)]

    inputs = processor(
        videos=[fake_video],
        text=[text_prompt],
        return_tensors="pt",
        padding=True,
        input_data_format="channels_last"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    emb = outputs.video_embeds[0].float().detach().cpu().numpy()
    return l2_normalize(emb)

def get_sam_mask_on_crop(
    sam_predictor: SamPredictor,
    crop_bgr: np.ndarray
):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(crop_rgb)

    h, w = crop_bgr.shape[:2]
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

def pick_debug_indices(num_items: int, max_items: int, seed: int = 7):
    if num_items <= max_items:
        return set(range(num_items))
    rng = random.Random(seed)
    return set(rng.sample(range(num_items), k=max_items))

def load_detection_pt(pt_path: Path):
    return torch.load(pt_path, map_location="cpu")

def get_video_embed_batch_from_images(
    model,
    processor,
    images_bgr: list,
    text_prompts: list,
    device,
    batch_size: int = 16
) -> np.ndarray:
    all_embeds = []
    
    for i in range(0, len(images_bgr), batch_size):
        batch_bgr = images_bgr[i:i+batch_size]
        batch_texts = text_prompts[i:i+batch_size]

        batch_videos = []
        for img_bgr in batch_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            batch_videos.append([img_rgb for _ in range(16)])

        inputs = processor(
            videos=batch_videos,
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            input_data_format="channels_last"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        batch_embs = outputs.video_embeds.float().detach().cpu().numpy()
        for emb in batch_embs:
            all_embeds.append(l2_normalize(emb))
            
    if len(all_embeds) == 0:
        return np.array([])
    return np.stack(all_embeds, axis=0)

def get_sam_masks_batched(
    sam_model,
    bgr_crops: list,
    device,
    batch_size: int = 8
):
    from segment_anything.utils.transforms import ResizeLongestSide
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    all_masks = []
    all_scores = []
    
    for i in range(0, len(bgr_crops), batch_size):
        batch_crops = bgr_crops[i:i+batch_size]
        batched_inputs = []
        
        for crop in batch_crops:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = crop_rgb.shape[:2]
            
            input_image = transform.apply_image(crop_rgb)
            img_tensor = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous()
            
            box = np.array([[0, 0, orig_w - 1, orig_h - 1]])
            box_tensor = torch.as_tensor(transform.apply_boxes(box, (orig_h, orig_w)), device=device)
            
            batched_inputs.append({
                "image": sam_model.preprocess(img_tensor),
                "boxes": box_tensor,
                "original_size": (orig_h, orig_w)
            })
            
        with torch.inference_mode():
            batched_outputs = sam_model(batched_inputs, multimask_output=True)
            
        for out in batched_outputs:
            masks = out["masks"] # (1, 3, H, W)
            scores = out["iou_predictions"] # (1, 3)
            best_idx = int(torch.argmax(scores[0]))
            curr_mask = masks[0, best_idx].detach().cpu().numpy()
            curr_score = float(scores[0, best_idx])
            all_masks.append(curr_mask)
            all_scores.append(curr_score)
            
    return all_masks, all_scores
