import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import librosa
from torchvision.transforms import functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.projection_head import ProjectionHead
from transformers import PeAudioVideoModel, PeAudioVideoProcessor



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + eps)

print("Loading Models for Demo...")
peav_model = PeAudioVideoModel.from_pretrained(
        "facebook/pe-av-small-16-frame",
        low_cpu_mem_usage=True
    ).to(device)
peav_processor = PeAudioVideoProcessor.from_pretrained("facebook/pe-av-small-16-frame")
peav_model.eval()

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
maskrcnn = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()

proj_head = ProjectionHead(roi_dim=256, hidden_dim=512, embed_dim=1024).to(device).eval()

checkpoint = torch.load("data/Echo-ViLD/weights/sam_80_20_llm/best.pth", map_location=device)
proj_head.load_state_dict(checkpoint['proj'])

print("Models loaded successfully!")

def process_inference(image, text_query, audio_file):
    if image is None:
        return None
    
    # 1. Mask R-CNN Feature Extraction
    img_tensor = TF.to_tensor(image).to(device)
    maskrcnn.roi_heads.score_thresh = 0.0        
    maskrcnn.roi_heads.detections_per_img = 100 
    
    with torch.no_grad():
        outputs = maskrcnn([img_tensor])
        boxes = outputs[0]['boxes']     
        obj_scores = outputs[0]['scores'] 
        
        if len(boxes) == 0:
            return image
            
        features = maskrcnn.backbone(img_tensor.unsqueeze(0))
        box_features = maskrcnn.roi_heads.box_roi_pool(features, [boxes], [img_tensor.shape[-2:]])
        roi_feats_256d = box_features.mean(dim=[2, 3]).float()
        
        # 2. MLP Projection
        proj_feat_out = proj_head(roi_feats_256d) # [N, 1024]
        
        # 3. Process Query (Audio OR Text)
        
         # Create a tiny dummy video (16 frames of 2x2 black pixels) to satisfy PE-AV's 2-input rule
        dummy_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        dummy_video = [dummy_frame for _ in range(16)]
        if audio_file is not None:
            # Audio Inference
            waveform, sr = librosa.load(audio_file, sr=48000)
            inputs = peav_processor(audio=[waveform], videos=[dummy_video], sampling_rate=48000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = peav_model(**inputs)
            query_emb = outputs.audio_embeds[0].float().cpu().numpy()
        else:
            # Text Inference
            text_query = text_query if text_query else "object"
            inputs = peav_processor(text=[text_query], videos=[dummy_video], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = peav_model(**inputs)
            query_emb = outputs.text_video_embeds[0].float().cpu().numpy()
            
        query_emb = torch.tensor(l2_normalize(query_emb)).to(device) # [1024]
        
        # 4. Dot Product & Finding Best Box
        similarity_scores = proj_feat_out @ query_emb.T # [N]
        valid_mask = obj_scores > 0.3
        
        # Fallback just in case no box is > 0.3
        if not valid_mask.any():
            valid_mask = obj_scores > 0.05
            
        # Set similarity of invalid boxes to -100 so they never win the argmax
        similarity_scores[~valid_mask] = -100.0 
        
        # Find the box with the highest PE-AV similarity!
        best_idx = torch.argmax(similarity_scores).item()
        best_box = boxes[best_idx].cpu().numpy()
        
    # 5. Draw Box on Image
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    x1, y1, x2, y2 = best_box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    return result_img

iface = gr.Interface(
    fn=process_inference,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Text Query (Optional)"),
        gr.Audio(type="filepath", label="Audio Query (Optional, overrides text)")
    ],
    outputs=gr.Image(type="pil", label="Detection Result"),
    title="Echo-ViLD: Zero-Shot Tri-Modal Object Detection",
    description="Upload an image. Then, either type what you want to find, or upload a sound of the object!"
)

if __name__ == "__main__":
    iface.launch()