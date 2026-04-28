"""
Extract region proposals and ROIAlign features from a pretrained Mask RCNN (torchvision) on COCO images
"""

import os
import json
import argparse
from collections import defaultdict

from dotenv import load_dotenv
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
from PIL import Image
from tqdm import tqdm
from huggingface_hub import HfApi

load_dotenv()

def build_model(device):
    """
    Load pretrained Mask RCNN.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    return model

def load_coco_annotations(ann_file):
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # img_id -> file_name
    images = {img['id']: img['file_name'] for img in coco['images']}

    # img_id -> list of {'bbox': [x,y,w,h], 'category_id': int}
    gt_annotations = defaultdict(list)
    for ann in coco['annotations']:
        gt_annotations[ann['image_id']].append({
            'bbox': ann['bbox'],           # [x, y, w, h]
            'category_id': ann['category_id']
        })

    # category_id -> name
    class_map = {cat['id']: cat['name'] for cat in coco['categories']}

    return coco, images, gt_annotations, class_map


def assign_labels(pred_boxes, gt_boxes_coco, gt_cat_ids, iou_threshold=0.5):
    """
    Match each predicted box to the best-overlapping GT box.
    Returns a label tensor:  0 = background, else COCO category_id.

    pred_boxes   : Tensor (N, 4)  [x1,y1,x2,y2]
    gt_boxes_coco: list of [x,y,w,h]
    gt_cat_ids   : list of int (raw COCO category_ids, non-contiguous)
    """
    N = pred_boxes.shape[0]
    labels = torch.zeros(N, dtype=torch.long)

    if len(gt_boxes_coco) == 0:
        return labels

    # Convert GT from [x,y,w,h] -> [x1,y1,x2,y2]
    gt_xyxy = torch.tensor(
        [[x, y, x + w, y + h] for x, y, w, h in gt_boxes_coco],
        dtype=torch.float32
    )

    # IoU matrix: (N_pred, N_gt)
    iou_matrix = box_iou(pred_boxes.cpu(), gt_xyxy)

    # For each pred box, find best matching GT
    best_iou, best_gt_idx = iou_matrix.max(dim=1)

    for i in range(N):
        if best_iou[i] >= iou_threshold:
            labels[i] = gt_cat_ids[best_gt_idx[i]]
        # else stays 0 (background)

    return labels


def build_coco_id_remap(coco_data):
    """Map raw COCO category_id -> compact class index (1..80); 0 stays background."""
    # Extract the 80 active IDs and sort them
    valid_ids = sorted([cat['id'] for cat in coco_data['categories']])
    # Map them to 1 through 80 (0 is reserved for background)
    return {raw_id: i + 1 for i, raw_id in enumerate(valid_ids)}

@torch.no_grad()
def extract_one_image(img_path, model, device, max_proposals=300):
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    model.roi_heads.score_thresh = 0.0        
    model.roi_heads.detections_per_img = max_proposals

    # get bounding boxes from the model
    outputs = model([img_tensor])
    boxes = outputs[0]['boxes']     # (N, 4)
    scores = outputs[0]['scores']          # (N,)

    # get the FPN feature maps from the backbone
    features = model.backbone(img_tensor.unsqueeze(0))

    box_features = model.roi_heads.box_roi_pool(features, [boxes], [img_tensor.shape[-2:]])
    
    # box_features shape is (N, 256, 7, 7). Average pool the 7x7 spatial dimensions:
    roi_feats_256d = box_features.mean(dim=[2, 3]).half() # convert to fp16

    return boxes.cpu(), scores.cpu(), roi_feats_256d.cpu()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_model(device)
    coco_data, images, gt_annotations, class_map = load_coco_annotations(args.ann_file)

    id_remap = build_coco_id_remap(coco_data)

    img_ids = list(images.keys())
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]

    print(f"Processing {len(img_ids)} images -> saving to {args.output_dir}")

    failed = []

    for img_id in tqdm(img_ids):
        out_path = os.path.join(args.output_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            continue 

        img_path = os.path.join(args.coco_img_dir, images[img_id])
        if not os.path.exists(img_path):
            failed.append(img_id)
            continue

        try:
            boxes, scores, roi_feats = extract_one_image(
                img_path, model, device, args.max_proposals
            )

            # Assign GT labels via IoU matching
            ann_list   = gt_annotations[img_id]
            gt_bboxes  = [a['bbox'] for a in ann_list]
            gt_cat_ids = [id_remap[a['category_id']] for a in ann_list]
            labels = assign_labels(boxes, gt_bboxes, gt_cat_ids, iou_threshold=0.5)

            torch.save({
                'img_id'       : img_id,
                'file_name'    : images[img_id],
                'boxes'        : boxes,         # (N, 4)
                'roi_scores'   : scores,        # (N,)
                'roi_features' : roi_feats,     # (N, 256) <- student MLP input
                'labels'       : labels,        # (N,)  0=bg, 1..80 compact class index
            }, out_path)

        except Exception as e:
            print(f"\n[ERROR] img_id={img_id}: {e}")
            failed.append(img_id)

    print(f"\nDone. Failed: {len(failed)} images.")
    if failed:
        print("Failed img_ids:", failed[:20])

    # upload to HuggingFace
    api = HfApi()
    api.upload_folder(
        folder_path=args.output_dir,
        path_in_repo='Bboxes_and_256D',
        repo_id='preetsojitra/Echo-VilD',
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
)


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
    parser.add_argument('--class_desc',    default='offline_prep/coco_class_descriptions.json',
                        help='JSON defining the 80-class ordering (maps raw COCO ids -> 1..80)')
    args = parser.parse_args()

    main(args)