import json
from pathlib import Path

def inspect_coco_json(file_path):
    print(f"\n{'='*20} Inspecting: {Path(file_path).name} {'='*20}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 1. Show Top Level Keys
    print(f"Top-level keys: {list(data.keys())}")
    
    # 2. Show count of items in each list
    for key in ['images', 'annotations', 'categories', 'licenses']:
        if key in data:
            print(f"Total {key}: {len(data[key])}")
    
    # 3. Show a sample of the first item in each major list
    if 'images' in data and len(data['images']) > 0:
        print(f"\nSample Image Entry:\n{json.dumps(data['images'][0], indent=2)}")
        
    if 'annotations' in data and len(data['annotations']) > 0:
        print(f"\nSample Annotation Entry:\n{json.dumps(data['annotations'][0], indent=2)}")

    if 'categories' in data and len(data['categories']) > 0:
        # Categories are usually the same across COCO subsets
        print(f"\nFirst 2 Categories: {data['categories'][:2]}")

    # Explicitly clear memory after inspection
    del data

# Example usage:
inspect_coco_json('/home/dal696598/scratch/coco_subset_100/annotations/instances_train2017.json')