import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import librosa

# Import PE-AV and your MLP
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.projection_head import ProjectionHead
from offline_prep.sam_peav_utils import load_peav, l2_normalize


def build_maskrcnn(device):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    return model


def get_audio_embeddings(audio_dir, categories, peav_model, peav_processor, device):
    audio_embeds = []
    
    print("Generating PE-AV Audio Embeddings...")
    for cat in categories:
        wav_path = os.path.join(audio_dir, f"{cat}.wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Missing audio file for category: {cat}")
            
        # Load audio (PE-AV expects 16kHz)
        waveform, sr = librosa.load(wav_path, sr=48000)
        
        inputs = peav_processor(
            audios=[waveform],
            sampling_rate=48000,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = peav_model(**inputs)
            
        emb = outputs.audio_embeds[0].float().cpu().numpy()
        emb_norm = l2_normalize(emb)
        audio_embeds.append(emb_norm)
        
    return torch.tensor(np.stack(audio_embeds)).to(device)


def calculate_iou(boxA, boxB):
    """Calculate IoU between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


@torch.no_grad()
def evaluate_audio(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load VPO-SS Categories
    vpo_data_dir = os.path.join(args.vpo_dir, "data")
    categories = sorted([d for d in os.listdir(vpo_data_dir) if os.path.isdir(os.path.join(vpo_data_dir, d))])
    print(f"Loaded {len(categories)} VPO-SS Categories: {categories}")

    # 2. Load Models
    peav_model, peav_processor = load_peav("facebook/pe-av-small-16-frame", device)
    maskrcnn = build_maskrcnn(device)

    proj_head = ProjectionHead(roi_dim=256, hidden_dim=512, embed_dim=1024).to(device)
    proj_head.eval()
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    proj_head.load_state_dict(checkpoint['proj'])
    bg_embed = checkpoint['bg_embed'].to(device) 

    # 3. Generate Audio Embeddings [22, 1024]
    audio_emb = get_audio_embeddings(args.audio_dir, categories, peav_model, peav_processor, device)
    
    # Normalize Audio and Background, then concatenate [23, 1024]
    bg_normalized = F.normalize(bg_embed, dim=-1)
    class_matrix = torch.cat([bg_normalized, audio_emb], dim=0)

    # Metrics Tracking
    total_images = 0
    correct_detections_iou50 = 0

    print("\nRunning Zero-Shot Acoustic Inference...")
    for cat_idx, cat in enumerate(categories):
        img_dir = os.path.join(vpo_data_dir, cat)
        mask_dir = os.path.join(args.vpo_dir, "mask", cat)
        
        for img_name in tqdm(os.listdir(img_dir), desc=f"Evaluating {cat}"):
            if not img_name.endswith(('.jpg', '.png')): continue
                
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))
            
            if not os.path.exists(mask_path): continue
                
            # A. Get Ground Truth Bounding Box from VPO-SS Mask
            gt_mask = np.array(Image.open(mask_path))
            rows = np.any(gt_mask, axis=1)
            cols = np.any(gt_mask, axis=0)
            if not np.any(rows) or not np.any(cols): continue # Empty mask
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            gt_box = [xmin, ymin, xmax, ymax]

            # B. Mask R-CNN Extraction
            img = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(img).to(device)
            maskrcnn.roi_heads.score_thresh = 0.0        
            maskrcnn.roi_heads.detections_per_img = 300

            outputs = maskrcnn([img_tensor])
            boxes = outputs[0]['boxes']     
            obj_scores = outputs[0]['scores'] 

            if len(boxes) == 0: continue

            features = maskrcnn.backbone(img_tensor.unsqueeze(0))
            box_features = maskrcnn.roi_heads.box_roi_pool(features, [boxes], [img_tensor.shape[-2:]])
            roi_feats_256d = box_features.mean(dim=[2, 3]).float()

            # C. Echo-ViLD MLP Projection
            proj_feat_out = proj_head(roi_feats_256d)

            # D. Zero-Shot Audio Classification (Dot Product)
            similarity_scores = proj_feat_out @ class_matrix.T
            logits = similarity_scores / args.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Remove background probability (index 0) -> [N, 22]
            class_probs = probs[:, 1:]
            
            # ViLD TRICK: Multiply by objectness score
            final_scores = class_probs * obj_scores.unsqueeze(1) 

            # E. Metric Evaluation
            # Get the highest scoring box for the CURRENT audio category
            target_class_col = cat_idx 
            best_box_idx = torch.argmax(final_scores[:, target_class_col]).item()
            best_pred_box = boxes[best_box_idx].cpu().numpy()

            # Calculate IoU
            iou = calculate_iou(best_pred_box, gt_box)
            if iou >= 0.5:
                correct_detections_iou50 += 1
                
            total_images += 1

    accuracy = (correct_detections_iou50 / total_images) * 100 if total_images > 0 else 0
    print("\n" + "="*40)
    print(f"Total Images Evaluated: {total_images}")
    print(f"Audio-Driven Detection Accuracy (IoU >= 0.5): {accuracy:.2f}%")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vpo_dir", required=True, help="Path to VPO-SS directory (contains data/ and mask/ folders)")
    parser.add_argument("--audio_dir", required=True, help="Path to folder containing 22 .wav files")
    parser.add_argument("--ckpt_path", required=True, help="Path to best.pth model weights")
    parser.add_argument("--temperature", type=float, default=0.01, help="Softmax temperature")
    args = parser.parse_args()

    evaluate_audio(args)