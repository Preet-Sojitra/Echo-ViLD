import random
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from transformers import PeAudioVideoModel, PeAudioVideoProcessor

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
        padding=True
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
