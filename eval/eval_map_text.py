"""
Evaluate trained Echo-ViLD models on the LVIS v1.0 Validation Set.
Calculates AP_r (Rare/Novel categories) to prove Zero-Shot capabilities.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from lvis import LVIS, LVISEval

# Import your MLP Projection Head
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.projection_head import ProjectionHead


def build_maskrcnn(device):
    """Load the exact same frozen Mask R-CNN used for feature extraction."""
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def evaluate_lvis(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load LVIS Dataset
    print(f"Loading LVIS annotations from {args.ann_file}...")
    lvis_api = LVIS(args.ann_file)
    img_ids = lvis_api.get_img_ids()
    
    # LVIS Category IDs (Should be 1203 categories)
    # We sort them to ensure they align exactly with your text embeddings tensor
    cat_ids = sorted(lvis_api.get_cat_ids())
    print(f"Total LVIS categories: {len(cat_ids)}")

    # 2. Load Models
    print("Loading Mask R-CNN...")
    maskrcnn = build_maskrcnn(device)

    print(f"Loading ProjectionHead from {args.ckpt_path}...")
    proj_head = ProjectionHead(roi_dim=256, hidden_dim=512, embed_dim=1024).to(device)
    proj_head.eval()
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    proj_head.load_state_dict(checkpoint['proj'])
    
    # The learnable background embedding [1, 1024]
    bg_embed = checkpoint['bg_embed'].to(device) 

    # 3. Load LVIS Text Embeddings [1203, 1024]
    print(f"Loading LVIS text embeddings from {args.text_emb_path}...")
    text_emb = torch.load(args.text_emb_path, map_location=device).float()
    
    assert text_emb.shape[0] == len(cat_ids), f"Text embed rows ({text_emb.shape[0]}) != LVIS categories ({len(cat_ids)})"

    # Normalize Text and Background, then concatenate [1204, 1024]
    bg_normalized = F.normalize(bg_embed, dim=-1)
    text_normalized = F.normalize(text_emb, dim=-1)
    class_matrix = torch.cat([bg_normalized, text_normalized], dim=0)

    results = []

    print("Running Zero-Shot Inference on LVIS Validation Set...")
    for img_id in tqdm(img_ids):
        img_info = lvis_api.load_imgs([img_id])[0]
        # LVIS uses the coco2017 image names
        img_path = os.path.join(args.coco_img_dir, img_info['coco_url'].split('/')[-1])
        
        if not os.path.exists(img_path):
            continue

        # --- A. Mask R-CNN Extraction ---
        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img).to(device)

        maskrcnn.roi_heads.score_thresh = 0.0        
        maskrcnn.roi_heads.detections_per_img = 300

        outputs = maskrcnn([img_tensor])
        boxes = outputs[0]['boxes']     # [N, 4] (x1, y1, x2, y2)
        obj_scores = outputs[0]['scores'] # [N] (Objectness score)

        if len(boxes) == 0:
            continue

        features = maskrcnn.backbone(img_tensor.unsqueeze(0))
        box_features = maskrcnn.roi_heads.box_roi_pool(features, [boxes], [img_tensor.shape[-2:]])
        roi_feats_256d = box_features.mean(dim=[2, 3]).float() # [N, 256]

        # --- B. Echo-ViLD MLP Projection ---
        proj_feat_out = proj_head(roi_feats_256d) # [N, 1024] (Already L2 Normalized)

        # --- C. Zero-Shot Classification (Dot Product) ---
        # Cosine similarity -> [N, 1204]
        similarity_scores = proj_feat_out @ class_matrix.T
        logits = similarity_scores / args.temperature

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Remove background probability (index 0) -> [N, 1203]
        class_probs = probs[:, 1:]

        # ViLD TRICK: Multiply class probabilities by Mask R-CNN objectness score
        final_scores = class_probs * obj_scores.unsqueeze(1) # [N, 1203]

        # --- D. Format for LVIS Evaluation ---
        boxes_cpu = boxes.cpu().numpy()
        scores_cpu = final_scores.cpu().numpy()

        for i in range(len(boxes_cpu)):
            x1, y1, x2, y2 = boxes_cpu[i]
            w, h = x2 - x1, y2 - y1 # LVIS requires [x, y, w, h] format
            
            # Get the top scoring classes for this box
            # To save memory, we only keep scores > 0.0001
            for class_idx in range(len(cat_ids)):
                score = float(scores_cpu[i, class_idx])
                if score > 0.0001:
                    results.append({
                        "image_id": img_id,
                        "category_id": cat_ids[class_idx],
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": score
                    })

    # Save raw predictions to disk
    os.makedirs(args.output_dir, exist_ok=True)
    res_file = os.path.join(args.output_dir, "lvis_predictions.json")
    print(f"\nSaving {len(results)} predictions to {res_file}...")
    with open(res_file, "w") as f:
        json.dump(results, f)

    # --- E. Calculate mAP ---
    print("\nStarting Official LVIS Evaluation...")
    lvis_eval = LVISEval(args.ann_file, res_file, "bbox")
    lvis_eval.run()
    lvis_eval.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_img_dir", required=True, help="Path to val2017 images folder")
    parser.add_argument("--ann_file", required=True, help="Path to lvis_v1_val.json")
    parser.add_argument("--ckpt_path", required=True, help="Path to best.pth model weights")
    parser.add_argument("--text_emb_path", required=True, help="Path to lvis text embeddings .pt file")
    parser.add_argument("--output_dir", default="./eval_results", help="Directory to save JSON predictions")
    parser.add_argument("--temperature", type=float, default=0.01, help="Softmax temperature")
    args = parser.parse_args()

    evaluate_lvis(args)