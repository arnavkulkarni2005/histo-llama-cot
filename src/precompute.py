import torch
import os
import glob
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from config import DATA_DIR, EMBEDDING_DIR, VISION_MODEL, DEVICE, SUBSET_SIZE

def precompute_embeddings():
    print(f"Loading PLIP ({VISION_MODEL})...")
    processor = AutoProcessor.from_pretrained(VISION_MODEL)
    model = AutoModel.from_pretrained(VISION_MODEL).to(DEVICE)
    model.eval()

    all_files = glob.glob(os.path.join(DATA_DIR, "*", "*.tif"))
    if SUBSET_SIZE:
        import json
        from config import INSTRUCTION_FILE
        if os.path.exists(INSTRUCTION_FILE):
            with open(INSTRUCTION_FILE, 'r') as f:
                data = json.load(f)
            target_filenames = {item['id'] for item in data}
            all_files = [f for f in all_files if os.path.basename(f) in target_filenames]

    print(f"Pre-computing embeddings for {len(all_files)} images...")
    
    with torch.no_grad():
        for img_path in tqdm(all_files):
            filename = os.path.basename(img_path)
            save_path = os.path.join(EMBEDDING_DIR, filename + ".pt")
            
            if os.path.exists(save_path):
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                features = model.get_image_features(**inputs)
                
                torch.save(features.cpu(), save_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    precompute_embeddings()