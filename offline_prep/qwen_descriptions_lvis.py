import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_lvis_categories(ann_file):
    with open(ann_file, 'r') as f:
        lvis = json.load(f)

    # category_id → name (replace underscores with spaces for the LLM)
    class_map = {cat['id']: cat['name'].replace('_', ' ') for cat in lvis['categories']}
    
    # Sort by ID to ensure strict 1 to 1203 ordering!
    class_map = dict(sorted(class_map.items()))

    return class_map

SYSTEM_PROMPT = """You are a vision-language model trainer. 
Your job is to write rich, visually descriptive paragraphs about objects.
These descriptions will be encoded into embeddings for object detection.
Focus only on visual appearance: shape, color, texture, size, typical context.
Do NOT mention the category name repeatedly. Be specific and concrete.
Write exactly ONE paragraph of 3-5 sentences. No bullet points. No headers."""

def build_prompt(category_name):
    return (
        f"Write a rich visual description of '{category_name}' as it appears in a real photograph. "
        f"Describe its typical appearance, shape, color, texture, and the context it is usually seen in. "
        f"This will be used as a text embedding for training an object detector."
    )


def load_model():
    MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_description(model, tokenizer, category_name):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(category_name)},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    description = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    description = ' '.join(description.split())
    return description


def main(args):
    LVIS_CATEGORIES = load_lvis_categories(args.ann_file)

    model, tokenizer = load_model()

    results = []

    print(f"\nGenerating descriptions for {len(LVIS_CATEGORIES)} LVIS categories...\n")
    for cat_id, cat_name in tqdm(LVIS_CATEGORIES.items()):
        description = generate_description(model, tokenizer, cat_name)
        results.append({
            'id':          cat_id,
            'name':        cat_name,
            'description': description,
        })
        tqdm.write(f"  [{cat_id:2d}] {cat_name:<20} → {description[:80]}...")

    # ── Save .json (full metadata) ──
    json_path = f"{args.output_dir}/lvis_descriptions.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n Saved:")
    print(f"   {json_path} ← full metadata with category ids")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', required=True,
                        help='Path to the annotations file')
    parser.add_argument('--output_dir', required=True,
                        help='Where to save output files')
    args = parser.parse_args()
    main(args)