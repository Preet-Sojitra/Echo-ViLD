import json

with open("lvis_v1_val.json", "r") as f:
    lvis = json.load(f)

print(f"Total Images: {len(lvis['images'])}")
print(f"Total Annotations: {len(lvis['annotations'])}")
print(f"Total Categories: {len(lvis['categories'])}")

# Count Rare vs Common vs Frequent
rare = [c['name'] for c in lvis['categories'] if c['frequency'] == 'r']
print(f"Number of Rare (Novel) Categories: {len(rare)}")
print(f"Sample Rare Categories: {rare[:10]}")