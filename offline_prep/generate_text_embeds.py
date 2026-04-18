"""
Generate text embedding using original ViLD's 63 prompt template strategy and using the LLM based rich description.
"""
import os
import json

from dotenv import load_dotenv
import torch
from transformers import PeAudioVideoModel, PeAudioVideoProcessor
from tqdm import tqdm
from huggingface_hub import HfApi

load_dotenv()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading PE-AV Model...")
model = PeAudioVideoModel.from_pretrained("facebook/pe-av-large").to(device)
processor = PeAudioVideoProcessor.from_pretrained("facebook/pe-av-large")
model.eval()

# The 63 ViLD Templates
VILD_TEMPLATES = [
    'There is {article} {category} in the scene.'
    'There is the {category} in the scene.'
    'a photo of {article} {category} in the scene.'
    'a photo of the {category} in the scene.'
    'a photo of one {category} in the scene.'
    'itap of {article} {category}.'
    'itap of my {category}.'
    'itap of the {category}.'
    'a photo of {article} {category}.'
    'a photo of my {category}.'
    'a photo of the {category}.'
    'a photo of one {category}.'
    'a photo of many {category}.'
    'a good photo of {article} {category}.'
    'a good photo of the {category}.'
    'a bad photo of {article} {category}.'
    'a bad photo of the {category}.'
    'a photo of a nice {category}.'
    'a photo of the nice {category}.'
    'a photo of a cool {category}.'
    'a photo of the cool {category}.'
    'a photo of a weird {category}.'
    'a photo of the weird {category}.'
    'a photo of a small {category}.'
    'a photo of the small {category}.'
    'a photo of a large {category}.'
    'a photo of the large {category}.'
    'a photo of a clean {category}.'
    'a photo of the clean {category}.'
    'a photo of a dirty {category}.'
    'a photo of the dirty {category}.'
    'a bright photo of {article} {category}.'
    'a bright photo of the {category}.'
    'a dark photo of {article} {category}.'
    'a dark photo of the {category}.'
    'a photo of a hard to see {category}.'
    'a photo of the hard to see {category}.'
    'a low resolution photo of {article} {category}.'
    'a low resolution photo of the {category}.'
    'a cropped photo of {article} {category}.'
    'a cropped photo of the {category}.'
    'a close-up photo of {article} {category}.'
    'a close-up photo of the {category}.'
    'a jpeg corrupted photo of {article} {category}.'
    'a jpeg corrupted photo of the {category}.'
    'a blurry photo of {article} {category}.'
    'a blurry photo of the {category}.'
    'a pixelated photo of {article} {category}.'
    'a pixelated photo of the {category}.'
    'a black and white photo of the {category}.'
    'a black and white photo of {article} {category}.'
    'a plastic {category}.'
    'the plastic {category}.'
    'a toy {category}.'
    'the toy {category}.'
    'a plushie {category}.'
    'the plushie {category}.'
    'a cartoon {category}.'
    'the cartoon {category}.'
    'an embroidered {category}.'
    'the embroidered {category}.'
    'a painting of the {category}.'
    'a painting of a {category}.'
]

def get_article(category_name):
    """Returns 'an' if category starts with vowel, else 'a'."""
    if category_name[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    return 'a'

def extract_text_embedding(text_list):
    """Passes a list of strings through PE-AV and returns L2-normalized embeddings."""

    inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device)
    
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        outputs = model(**inputs)
        
    embeds = outputs.text_video_embeds 
    
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds

with open('coco_class_descriptions.json', 'r') as f:
    coco_classes = json.load(f)

vild_final_embeds = []
llm_final_embeds = []

print("Generating Embeddings for 80 COCO Classes...")

DEBUG = True

if DEBUG:
    coco_classes = coco_classes[:1]

for cls in tqdm(coco_classes):
    category = cls['name']
    article = get_article(category)
    
    # ========================================
    # approach 1: the vild ensemble strategy
    # ========================================
    # generate 63 sentences for this 1 category
    prompts = [template.format(article=article, category=category) for template in VILD_TEMPLATES]
    
    # get 63 embeddings [63, 1024]
    vild_63_embeds = extract_text_embedding(prompts)
    
    # average them together (mean pooling) -> [1, 1024]
    vild_mean_embed = vild_63_embeds.mean(dim=0, keepdim=True)
    
    # re-normalize after averaging
    vild_mean_embed = vild_mean_embed / vild_mean_embed.norm(dim=-1, keepdim=True)
    vild_final_embeds.append(vild_mean_embed.cpu())
    
    # ========================================
    # approach 2: the llm strategy
    # ========================================
    # get the single 1024-d embedding for the rich paragraph
    llm_embed = extract_text_embedding([cls['description']])
    llm_final_embeds.append(llm_embed.cpu())

# stack them into shape [80, 1024]
vild_tensor = torch.cat(vild_final_embeds, dim=0)
llm_tensor = torch.cat(llm_final_embeds, dim=0)

torch.save(vild_tensor, 'vild_text_embeddings_80.pt')
torch.save(llm_tensor, 'llm_text_embeddings_80.pt')

print(f"Saved ViLD Embeddings: {vild_tensor.shape}")
print(f"Saved LLM Embeddings: {llm_tensor.shape}")

# upload to huggingface
api = HfApi()
api.upload_file(
    path_or_fileobj="vild_text_embeddings_80.pt",
    path_in_repo="Text_Embeddings/vild_text_embeddings_80.pt",
    repo_id="preetsojitra/Echo-ViLD",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN"),
)
api.upload_file(
    path_or_fileobj="llm_text_embeddings_80.pt",
    path_in_repo="Text_Embeddings/llm_text_embeddings_80.pt",
    repo_id="preetsojitra/Echo-ViLD",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN"),
)