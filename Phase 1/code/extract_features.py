"""
extract_features.py
--------------------
Member-1 script: Extract region proposals and ROIAlign features from
a pretrained Mask RCNN (torchvision) on COCO images.

Output per image (saved as .pt files):
  {
    "img_id"         : int,
    "file_name"      : str,
    "boxes"          : Tensor (N, 4)   - [x1, y1, x2, y2] in pixel coords
    "roi_scores"     : Tensor (N,)     - RPN objectness scores
    "roi_features"   : Tensor (N, 256) - raw ROIAlign 256D features  ← student input
    "labels"         : Tensor (N,)     - COCO class id (0 = background)
  }

Usage:
    python extract_features.py \
        --coco_img_dir   /path/to/coco/train2017 \
        --ann_file       /path/to/annotations/instances_train2017.json \
        --output_dir     ./features \
        --max_proposals  300 \
        --max_images     5000   # set -1 for all
"""

import os
import json
import argparse
from collections import defaultdict

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes


# ──────────────────────────────────────────────
# 1.  Build the model and register the ROIAlign hook
# ──────────────────────────────────────────────

def build_model(device):
    """
    Load pretrained Mask RCNN. No hooks needed!
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    return model


# ──────────────────────────────────────────────
# 2.  Load COCO annotations
# ──────────────────────────────────────────────

def load_coco_annotations(ann_file):
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # img_id → file_name
    images = {img['id']: img['file_name'] for img in coco['images']}

    # img_id → list of {'bbox': [x,y,w,h], 'category_id': int}
    gt_annotations = defaultdict(list)
    for ann in coco['annotations']:
        gt_annotations[ann['image_id']].append({
            'bbox': ann['bbox'],           # [x, y, w, h]
            'category_id': ann['category_id']
        })

    # category_id → name
    class_map = {cat['id']: cat['name'] for cat in coco['categories']}

    return images, gt_annotations, class_map


# ──────────────────────────────────────────────
# 3.  Assign GT labels to proposals via IoU
# ──────────────────────────────────────────────

def assign_labels(pred_boxes, gt_boxes_coco, gt_cat_ids, iou_threshold=0.5):
    """
    Match each predicted box to the best-overlapping GT box.
    Returns a label tensor:  0 = background, else COCO category_id.

    pred_boxes   : Tensor (N, 4)  [x1,y1,x2,y2]
    gt_boxes_coco: list of [x,y,w,h]
    gt_cat_ids   : list of int
    """
    N = pred_boxes.shape[0]
    labels = torch.zeros(N, dtype=torch.long)

    if len(gt_boxes_coco) == 0:
        return labels

    # Convert GT from [x,y,w,h] → [x1,y1,x2,y2]
    gt_xyxy = torch.tensor(
        [[x, y, x + w, y + h] for x, y, w, h in gt_boxes_coco],
        dtype=torch.float32
    )

    # IoU matrix: (N_pred, N_gt)
    iou_matrix = box_iou(pred_boxes.cpu(), gt_xyxy)   # torchvision utility

    # For each pred box, find best matching GT
    best_iou, best_gt_idx = iou_matrix.max(dim=1)

    for i in range(N):
        if best_iou[i] >= iou_threshold:
            labels[i] = gt_cat_ids[best_gt_idx[i]]
        # else stays 0 (background)

    return labels


# ──────────────────────────────────────────────
# 4.  Extract features for one image
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_one_image(img_path, model, device, max_proposals=300):
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    # 1. Force model to return lots of boxes (do not filter them out)
    model.roi_heads.score_thresh = 0.0        
    model.roi_heads.detections_per_img = max_proposals

    # 2. Get the bounding boxes from the model
    outputs = model([img_tensor])
    boxes = outputs[0]['boxes']     # (N, 4)
    scores = outputs[0]['scores']          # (N,)

    # 3. Get the FPN feature maps from the backbone
    features = model.backbone(img_tensor.unsqueeze(0))

    # 4. EXPLICITLY extract the 256D features for OUR final boxes!
    # box_roi_pool requires a list of boxes per image
    box_features = model.roi_heads.box_roi_pool(features, [boxes], [img_tensor.shape[-2:]])
    
    # box_features shape is (N, 256, 7, 7). Average pool the 7x7 spatial dimensions:
    roi_feats_256d = box_features.mean(dim=[2, 3])

    return boxes.cpu(), scores.cpu(), roi_feats_256d.cpu()

def visualize_boxes(img_path, boxes, labels, class_map, save_path):
    # Load image as uint8 tensor (required for drawing)
    img_tensor = torchvision.io.read_image(img_path)
    
    # Optional: Filter out background (label 0) just to see the real objects
    # Or keep them all if you want to see the 300 messy proposals!
    keep_idx = labels > 0 
    real_boxes = boxes[keep_idx]
    real_labels = labels[keep_idx]
    
    # Convert category IDs to text names
    string_labels = [class_map.get(l.item(), "Unknown") for l in real_labels]
    
    # Draw boxes
    drawn_img = draw_bounding_boxes(img_tensor, real_boxes, labels=string_labels, colors="red", width=3)
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(drawn_img.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(save_path)
    print("Saved visualization to debug_boxes.png")

# ──────────────────────────────────────────────
# 5.  Main loop
# ──────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_model(device)
    images, gt_annotations, class_map = load_coco_annotations(args.ann_file)

    img_ids = list(images.keys())
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]

    print(f"Processing {len(img_ids)} images → saving to {args.output_dir}")

    failed = []
    
    # Flag to ensure we only draw the first image
    has_drawn_debug_image = False 

    for img_id in tqdm(img_ids):
        out_path = os.path.join(args.output_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            continue  # resume-friendly

        img_path = os.path.join(args.coco_img_dir, images[img_id])
        if not os.path.exists(img_path):
            failed.append(img_id)
            continue

        try:
            # Removed hook_store here!
            boxes, scores, roi_feats = extract_one_image(
                img_path, model, device, args.max_proposals
            )

            # Assign GT labels via IoU matching
            ann_list   = gt_annotations[img_id]
            gt_bboxes  = [a['bbox'] for a in ann_list]
            gt_cat_ids = [a['category_id'] for a in ann_list]
            labels = assign_labels(boxes, gt_bboxes, gt_cat_ids, iou_threshold=0.5)


            visualize_boxes(img_path, boxes, labels, class_map, save_path=f"/home/dal696598/scratch/debug_{img_id}.png")


            torch.save({
                'img_id'       : img_id,
                'file_name'    : images[img_id],
                'boxes'        : boxes,         # (N, 4)
                'roi_scores'   : scores,        # (N,)
                'roi_features' : roi_feats,     # (N, 256)  ← student MLP input
                'labels'       : labels,        # (N,)  0=bg, else COCO cat id
            }, out_path)

        except Exception as e:
            print(f"\n[ERROR] img_id={img_id}: {e}")
            failed.append(img_id)

    print(f"\nDone. Failed: {len(failed)} images.")
    if failed:
        print("Failed img_ids:", failed[:20])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_img_dir',  required=True,
                        help='Path to COCO train2017 image folder')
    parser.add_argument('--ann_file',      required=True,
                        help='Path to instances_train2017.json')
    parser.add_argument('--output_dir',    default='./features',
                        help='Where to save .pt files')
    parser.add_argument('--max_proposals', type=int, default=300,
                        help='Max region proposals per image')
    parser.add_argument('--max_images',    type=int, default=5000,
                        help='Max images to process (-1 = all)')
    args = parser.parse_args()
    main(args)