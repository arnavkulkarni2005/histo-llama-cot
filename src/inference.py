import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from PIL import Image
from model import HistoLlama
from config import *

def run_inference(image_path, checkpoint_path):
    print("Loading Models for Inference...")
    
    # Load PLIP (Vision)
    processor = AutoProcessor.from_pretrained(VISION_MODEL)
    vision_model = AutoModel.from_pretrained(VISION_MODEL).to(DEVICE)
    
    # Load HistoLlama (Text)
    model = HistoLlama(use_lora=True)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # 1. Process Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_features = vision_model.get_image_features(**inputs).unsqueeze(1) # [1, 1, 512]

    # 2. Prepare Prompt
    prompt = "<|user|>\n<image>\nAnalyze this histology slide. </s>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # 3. Generate
    print("Generating Diagnosis...")
    with torch.no_grad():
        # Project image
        visual_tokens = model.projector(img_features.to(torch.float32))
        
        # Get prompt embeddings
        text_embeds = model.llm.get_input_embeddings()(inputs.input_ids)
        
        # Combine
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        
        # Generate using the LLM directly
        output_ids = model.llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=100,
            do_sample=False,
            temperature=None
        )
        
    # Decode
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n--- MODEL OUTPUT ---")
    print(result)

if __name__ == "__main__":
    # Example Usage: Change paths as needed
    test_image = "/home/kulkarni/projects/biomed/data/raw_slides/NCT-CRC-HE-100K/TUM/TUM-AANHEQVF.tif" # Update this
    checkpoint = "checkpoints/histollama_epoch_1.pth"       # Update this
    
    # Check if files exist before running
    if os.path.exists(checkpoint) and os.path.exists(test_image):
        run_inference(test_image, checkpoint)
    else:
        print("Please update the test_image path in src/inference.py to a real .tif file.")