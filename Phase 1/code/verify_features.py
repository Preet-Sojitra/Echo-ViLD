"""
verify_features.py
-------------------
Sanity check: loads the saved .pt feature files and verifies that
Mask RCNN's region proposals actually cover the COCO ground truth boxes.

Reports:
  - Proposal recall  @ IoU 0.5  (do proposals cover GT objects?)
  - Label assignment accuracy   (does IoU matching assign correct class?)
  - Feature shape sanity        (are we getting 256D ROIAlign features?)
  - Per-category recall         (which classes are easy/hard to propose)

Usage:
    python verify_features.py \
        --features_dir  ./features \
        --ann_file      /path/to/annotations/instances_train2017.json \
        --num_images    100
"""

import os
import json
import argparse
from collections import defaultdict

import torch
from torchvision.ops import box_iou
from tqdm import tqdm


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def xywh_to_xyxy(boxes_xywh):
    """Convert list of [x,y,w,h] → Tensor [x1,y1,x2,y2]"""
    result = []
    for x, y, w, h in boxes_xywh:
        result.append([x, y, x + w, y + h])
    return torch.tensor(result, dtype=torch.float32)


def proposal_recall(pred_boxes, gt_boxes_xyxy, iou_threshold=0.5):
    """
    What fraction of GT boxes are covered by at least one proposal?
    pred_boxes    : Tensor (N, 4)
    gt_boxes_xyxy : Tensor (M, 4)
    Returns float in [0, 1]
    """
    if len(gt_boxes_xyxy) == 0:
        return 1.0  # nothing to cover

    iou_matrix = box_iou(pred_boxes, gt_boxes_xyxy)   # (N, M)
    # For each GT box, max IoU across all proposals
    max_iou_per_gt, _ = iou_matrix.max(dim=0)          # (M,)
    covered = (max_iou_per_gt >= iou_threshold).sum().item()
    return covered / len(gt_boxes_xyxy)


# ──────────────────────────────────────────────
# Main verification
# ──────────────────────────────────────────────

def main(args):
    # Load annotations
    with open(args.ann_file, 'r') as f:
        coco = json.load(f)

    gt_annotations = defaultdict(list)
    for ann in coco['annotations']:
        gt_annotations[ann['image_id']].append({
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        })

    class_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # Find saved feature files
    pt_files = sorted([
        f for f in os.listdir(args.features_dir) if f.endswith('.pt')
    ])
    if args.num_images > 0:
        pt_files = pt_files[:args.num_images]

    print(f"\nVerifying {len(pt_files)} feature files...\n")

    # ── Accumulators ──
    all_recalls        = []
    label_correct      = 0
    label_total        = 0
    feature_shapes_ok  = True
    per_cat_covered    = defaultdict(int)
    per_cat_total      = defaultdict(int)

    errors = []

    for fname in tqdm(pt_files):
        fpath = os.path.join(args.features_dir, fname)
        data  = torch.load(fpath, map_location='cpu')

        img_id      = data['img_id']
        boxes       = data['boxes']        # (N, 4)
        roi_scores  = data['roi_scores']   # (N,)
        roi_feats   = data['roi_features'] # (N, 256)
        labels      = data['labels']       # (N,)  assigned labels

        # ── 1. Feature shape check ──
        if roi_feats.shape[1] != 256:
            print(f"[SHAPE ERROR] {fname}: roi_features shape = {roi_feats.shape}, expected (N, 256)")
            feature_shapes_ok = False

        if roi_feats.shape[0] != boxes.shape[0]:
            print(f"[MISMATCH ERROR] {fname}: boxes={boxes.shape[0]}, features={roi_feats.shape[0]}")
            errors.append(fname)
            continue

        # ── 2. Get GT for this image ──
        ann_list   = gt_annotations.get(img_id, [])
        if len(ann_list) == 0:
            continue

        gt_bboxes  = [a['bbox']        for a in ann_list]
        gt_cat_ids = [a['category_id'] for a in ann_list]
        gt_xyxy    = xywh_to_xyxy(gt_bboxes)

        # ── 3. Proposal recall ──
        recall = proposal_recall(boxes, gt_xyxy, iou_threshold=0.5)
        all_recalls.append(recall)

        # ── 4. Per-category recall ──
        iou_matrix = box_iou(boxes, gt_xyxy)           # (N_pred, N_gt)
        max_iou_per_gt, _ = iou_matrix.max(dim=0)       # (N_gt,)
        for j, cat_id in enumerate(gt_cat_ids):
            per_cat_total[cat_id] += 1
            if max_iou_per_gt[j] >= 0.5:
                per_cat_covered[cat_id] += 1

        # ── 5. Label assignment accuracy ──
        # For each proposal that was assigned a non-background label,
        # check if the label matches the best-overlapping GT class.
        if len(gt_xyxy) > 0:
            iou_matrix2 = box_iou(boxes, gt_xyxy)      # (N, M)
            best_iou, best_gt_idx = iou_matrix2.max(dim=1)

            for i in range(len(boxes)):
                assigned_label = labels[i].item()
                if best_iou[i] >= 0.5:
                    true_label = gt_cat_ids[best_gt_idx[i]]
                    label_total += 1
                    if assigned_label == true_label:
                        label_correct += 1

    # ──────────────────────────────────────────────
    # Print Report
    # ──────────────────────────────────────────────

    print("\n" + "="*60)
    print("  VERIFICATION REPORT")
    print("="*60)

    # Feature shape
    print(f"\n[1] ROIAlign Feature Shape (256D check)")
    print(f"    {'✅ All feature files have correct 256D shape' if feature_shapes_ok else '❌ Shape errors found (see above)'}")

    # Proposal recall
    if all_recalls:
        avg_recall = sum(all_recalls) / len(all_recalls)
        perfect    = sum(1 for r in all_recalls if r == 1.0)
        zero       = sum(1 for r in all_recalls if r == 0.0)
        print(f"\n[2] Proposal Recall @ IoU 0.5")
        print(f"    Mean recall      : {avg_recall:.3f}  ({avg_recall*100:.1f}%)")
        print(f"    Images with 100% : {perfect} / {len(all_recalls)}")
        print(f"    Images with 0%   : {zero} / {len(all_recalls)}")
        print(f"    Interpretation   :")
        if avg_recall >= 0.8:
            print(f"    ✅ GOOD — proposals cover most GT boxes")
        elif avg_recall >= 0.5:
            print(f"    ⚠️  MEDIOCRE — some GT boxes are missed by proposals")
        else:
            print(f"    ❌ POOR — proposals are missing many GT boxes")

    # Label accuracy
    if label_total > 0:
        acc = label_correct / label_total
        print(f"\n[3] IoU-based Label Assignment Accuracy")
        print(f"    Correct / Total  : {label_correct} / {label_total}")
        print(f"    Accuracy         : {acc:.3f}  ({acc*100:.1f}%)")
        print(f"    Interpretation   :")
        if acc >= 0.95:
            print(f"    ✅ Labels are being assigned correctly")
        else:
            print(f"    ⚠️  Some label mismatches — check IoU threshold or box format")

    # Per-category recall (top/bottom 10)
    print(f"\n[4] Per-Category Recall @ IoU 0.5")
    cat_recalls = {}
    for cat_id in per_cat_total:
        cat_recalls[cat_id] = per_cat_covered[cat_id] / per_cat_total[cat_id]

    sorted_cats = sorted(cat_recalls.items(), key=lambda x: x[1], reverse=True)

    print(f"\n    Top 10 easiest categories (high recall):")
    for cat_id, rec in sorted_cats[:10]:
        name = class_map.get(cat_id, str(cat_id))
        total = per_cat_total[cat_id]
        print(f"      {name:<25} recall={rec:.2f}  ({per_cat_covered[cat_id]}/{total})")

    print(f"\n    Bottom 10 hardest categories (low recall):")
    for cat_id, rec in sorted_cats[-10:]:
        name = class_map.get(cat_id, str(cat_id))
        total = per_cat_total[cat_id]
        print(f"      {name:<25} recall={rec:.2f}  ({per_cat_covered[cat_id]}/{total})")

    # Errors
    if errors:
        print(f"\n[5] Files with errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    {e}")
    else:
        print(f"\n[5] No file errors ✅")

    print("\n" + "="*60)
    print("  QUICK CHECKLIST FOR MEMBER-1")
    print("="*60)
    print(f"  {'✅' if feature_shapes_ok else '❌'} ROIAlign features are 256D")
    print(f"  {'✅' if all_recalls and sum(all_recalls)/len(all_recalls) >= 0.7 else '❌'} Proposal recall ≥ 70%")
    print(f"  {'✅' if label_total > 0 and label_correct/label_total >= 0.95 else '❌'} Label assignment accuracy ≥ 95%")
    print(f"  {'✅' if not errors else '❌'} No file errors")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--ann_file',     required=True,
                        help='Path to instances_train2017.json')
    parser.add_argument('--num_images',   type=int, default=100,
                        help='Number of images to verify (-1 = all)')
    args = parser.parse_args()
    main(args)