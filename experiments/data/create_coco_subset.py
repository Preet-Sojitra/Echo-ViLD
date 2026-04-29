import os
import json
import random
import shutil
import gc
from pathlib import Path

def create_all_subsets(source_root, subset_sizes):
    source_root = Path(source_root).resolve()
    splits = ['train2017', 'val2017']

    for split in splits:
        print(f"\n{'='*20} Processing Split: {split} {'='*20}")
        
        # 1. Load the big JSON only ONCE for this split
        src_ann_path = source_root / 'annotations' / f'instances_{split}.json'
        if not src_ann_path.exists():
            print(f"Skipping {split}: File not found.")
            continue

        print(f"Loading {src_ann_path.name} into memory...")
        with open(src_ann_path, 'r') as f:
            full_data = json.load(f)

        # 2. Extract references to save time/memory during subsetting
        all_images = full_data['images']
        all_annotations = full_data['annotations']
        categories = full_data.get('categories', [])
        info = full_data.get('info', {})
        licenses = full_data.get('licenses', [])

        # 3. Create each subset size using the data already in memory
        for size in subset_sizes:
            dest_root = source_root.parent / f"coco_subset_{size}"
            dest_img_dir = dest_root / split
            dest_ann_dir = dest_root / 'annotations'
            
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_ann_dir.mkdir(parents=True, exist_ok=True)

            print(f"  --> Creating {dest_root.name} ({split})")

            # Randomly sample images
            actual_size = min(size, len(all_images))
            subset_images = random.sample(all_images, actual_size)
            subset_ids = {img['id'] for img in subset_images}

            # Filter annotations
            subset_anns = [ann for ann in all_annotations if ann['image_id'] in subset_ids]

            # Build JSON structure
            subset_output = {
                "info": info,
                "licenses": licenses,
                "images": subset_images,
                "annotations": subset_anns,
                "categories": categories
            }

            # Save subset JSON
            with open(dest_ann_dir / f'instances_{split}.json', 'w') as f:
                json.dump(subset_output, f)

            # Copy Images
            src_img_dir = source_root / split
            for img in subset_images:
                shutil.copy(src_img_dir / img['file_name'], dest_img_dir / img['file_name'])
            
            # Help GC by removing the reference to the dictionary we just built
            del subset_output
            print(f"      Done with {size} images.")

        # 4. CRITICAL: Clear the big data before moving to the next split
        print(f"Clearing memory for {split}...")
        del full_data, all_images, all_annotations
        gc.collect() 

if __name__ == "__main__":
    # Path to your full coco dataset folder
    full_coco_path = '/home/dal696598/scratch/coco_data_full'
    
    # Define your subset sizes
    target_sizes = [5, 100, 3000, 5000]

    create_all_subsets(full_coco_path, target_sizes)
    print("\nAll subsets completed successfully!")