"""
Echo-ViLD Gradio Demo — Interactive open-vocabulary + audio-query object detector.

Upload an image, provide a text query OR a .wav audio clip,
and see which regions the model detects.

Usage:
    python demo/app.py                           # default: vanilla model
    python demo/app.py --variant sam_80_20_llm    # specify model variant
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.projection_head import ProjectionHead
from eval.eval_dataset import extract_proposals_live


# ── Globals (loaded once at startup) ────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJ_HEAD = None
BG_EMBED = None
PEAV_MODEL = None
PEAV_PROCESSOR = None

# Model variant → checkpoint + text embed paths
VARIANTS = {
    "vanilla":         ("weights/vanilla/best.pth",         "vild_text_embeddings_80"),
    "sam_nocontext":   ("weights/sam_nocontext/best.pth",   "vild_text_embeddings_80"),
    "sam_eq":          ("weights/sam_eq/best.pth",          "vild_text_embeddings_80"),
    "sam_80_20":       ("weights/sam_80_20/best.pth",       "vild_text_embeddings_80"),
    "sam_80_20_llm":   ("weights/sam_80_20_llm/best.pth",   "llm_text_embeddings_80"),
}


def load_models(variant: str):
    """Load ProjectionHead + PE-AV (lazy, cached)."""
    global PROJ_HEAD, BG_EMBED, PEAV_MODEL, PEAV_PROCESSOR

    ckpt_path = Path(__file__).parent.parent / VARIANTS[variant][0]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    PROJ_HEAD = ProjectionHead().to(DEVICE)
    PROJ_HEAD.load_state_dict(ckpt["proj"])
    PROJ_HEAD.eval()
    BG_EMBED = ckpt["bg_embed"].to(DEVICE)

    if PEAV_MODEL is None:
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor
        print("Loading PE-AV model …")
        PEAV_MODEL = PeAudioVideoModel.from_pretrained("facebook/pe-av-large").to(DEVICE)
        PEAV_PROCESSOR = PeAudioVideoProcessor.from_pretrained("facebook/pe-av-large")
        PEAV_MODEL.eval()


def embed_text_query(text: str) -> torch.Tensor:
    """Embed a single text query via PE-AV → (1, 1024), L2-normalized."""
    dummy_video = torch.zeros(1, 1, 3, 224, 224)
    inputs = PEAV_PROCESSOR(
        text=[text], videos=list(dummy_video),
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = PEAV_MODEL(**inputs)
    emb = outputs.text_video_embeds[0:1].float()
    return F.normalize(emb, dim=-1)


def embed_audio_query(wav_path: str) -> torch.Tensor:
    """Embed a .wav file via PE-AV audio encoder → (1, 1024), L2-normalized."""
    import torchaudio

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    dummy_video = torch.zeros(1, 1, 3, 224, 224)
    inputs = PEAV_PROCESSOR(
        audios=[waveform.squeeze(0).numpy()],
        videos=list(dummy_video),
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = PEAV_MODEL(**inputs)
    emb = outputs.audio_embeds[0:1].float()
    return F.normalize(emb, dim=-1)


def draw_detections(pil_img: Image.Image, boxes, scores, top_k: int = 10, label: str = ""):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # Sort by score descending, keep top-K
    order = scores.argsort(descending=True)[:top_k]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    colors = [
        "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE",
        "#30B0C7", "#007AFF", "#5856D6", "#AF52DE", "#FF2D55",
    ]

    for rank, idx in enumerate(order):
        x1, y1, x2, y2 = boxes[idx].tolist()
        score = scores[idx].item()
        color = colors[rank % len(colors)]
        # Draw box
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
        # Draw label
        text = f"{label} {score:.2f}" if label else f"{score:.2f}"
        bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), text, fill="white", font=font)

    return pil_img


# ── Gradio inference function ───────────────────────────────────────────────

def detect(image: np.ndarray, text_query: str, audio_file: str, variant: str, top_k: int):
    """Main detection function called by Gradio."""
    if image is None:
        return None, "Please upload an image."

    # Load model for selected variant
    try:
        load_models(variant)
    except FileNotFoundError as e:
        return None, f"Model not found: {e}"

    # Save temp image for Mask R-CNN
    pil_img = Image.fromarray(image)
    tmp_path = Path(__file__).parent / "_tmp_input.jpg"
    pil_img.save(tmp_path)

    # Extract proposals
    boxes, roi_features, rpn_scores, _ = extract_proposals_live(tmp_path, DEVICE)
    tmp_path.unlink(missing_ok=True)

    if len(boxes) == 0:
        return np.array(pil_img), "No proposals detected."

    # Project through MLP
    with torch.no_grad():
        proj_feat = PROJ_HEAD(roi_features.to(DEVICE).unsqueeze(0))  # (1, N, 1024)
        proj_feat = proj_feat[0]  # (N, 1024)

    # Get query embedding
    query_label = ""
    if audio_file and Path(audio_file).exists():
        query_emb = embed_audio_query(audio_file)  # (1, 1024)
        query_label = "🔊 audio"
    elif text_query and text_query.strip():
        query_emb = embed_text_query(text_query.strip())  # (1, 1024)
        query_label = text_query.strip()
    else:
        return np.array(pil_img), "Please provide a text query or audio file."

    # Cosine similarity scores
    similarity = (proj_feat @ query_emb.T).squeeze(-1)  # (N,)

    # Draw results
    result_img = draw_detections(pil_img.copy(), boxes, similarity.cpu(), top_k=top_k, label=query_label)

    top_scores = similarity.cpu().sort(descending=True).values[:top_k]
    info = f"Top {top_k} detections (cosine similarity):\n"
    info += "\n".join([f"  #{i+1}: {s:.4f}" for i, s in enumerate(top_scores.tolist())])

    return np.array(result_img), info


# ── Gradio UI ───────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(
        title="Echo-ViLD: Audio-Visual Open-Vocabulary Detector",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🔊 Echo-ViLD: Segment-Guided Audio-Visual Distillation
            ### Zero-Shot Object Detection with Text and Audio Queries

            Upload an image and provide either a **text query** (e.g., "a cat") or an
            **audio clip** (.wav) to detect matching objects.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="numpy")
                text_input = gr.Textbox(label="Text Query", placeholder="e.g. 'a dog playing'")
                audio_input = gr.Audio(label="Audio Query (.wav)", type="filepath")
                variant_dropdown = gr.Dropdown(
                    choices=list(VARIANTS.keys()),
                    value="vanilla",
                    label="Model Variant",
                )
                top_k_slider = gr.Slider(1, 50, value=10, step=1, label="Top-K Detections")
                detect_btn = gr.Button("🔍 Detect", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Detections")
                output_info = gr.Textbox(label="Detection Info", lines=12)

        detect_btn.click(
            fn=detect,
            inputs=[image_input, text_input, audio_input, variant_dropdown, top_k_slider],
            outputs=[output_image, output_info],
        )

        gr.Markdown(
            """
            ---
            **How it works:**
            1. Mask R-CNN extracts region proposals from the image
            2. Our trained MLP projects proposals into PE-AV's shared embedding space
            3. Your query (text or audio) is embedded in the same space via PE-AV
            4. Regions are ranked by cosine similarity with your query
            """
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="vanilla", choices=list(VARIANTS.keys()))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    # Pre-load the default variant
    try:
        load_models(args.variant)
        print(f"Loaded variant: {args.variant}")
    except FileNotFoundError:
        print(f"Warning: {args.variant} checkpoint not found. Will load on first request.")

    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share)