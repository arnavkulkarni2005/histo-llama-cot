import os
import json
import random
import glob
from config import DATA_DIR, INSTRUCTION_FILE, SUBSET_SIZE

# Mappings: Class -> (Observation, Conclusion)
BIO_FEATURES = {
    "ADI": ("clear, empty-appearing cells with peripheral nuclei and honeycomb-like pattern", "Adipose Tissue (ADI)"),
    "BACK": ("bright white or light gray area with no cellular tissue visible", "Background (BACK)"),
    "DEB": ("amorphous necrotic material and fragmented tissue without clear structure", "Debris (DEB)"),
    "LYM": ("dense clusters of small, round, dark blue nuclei with little to no cytoplasm", "Lymphocytes (LYM)"),
    "MUC": ("pale blue or gray amorphous substance with floating extracellular material", "Mucus (MUC)"),
    "MUS": ("long, spindle-shaped cells organized in parallel bundles with eosinophilic fibers", "Smooth Muscle (MUS)"),
    "NORM": ("regular crypt structures with organized glandular layout and goblet cells", "Normal Mucosa (NORM)"),
    "STR": ("loose connective tissue containing fibroblasts and collagen fibers", "Stroma (STR)"),
    "TUM": ("complex irregular glandular structures with dark enlarged nuclei and loss of polarity", "Colorectal Adenocarcinoma (TUM)")
}

def generate_cot_data():
    data = []
    print(f"Scanning {DATA_DIR} for .tif files...")
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*", "*.tif"))
    
    if not all_files:
        print("Error: No .tif files found! Check your path.")
        return

    random.shuffle(all_files)
    if SUBSET_SIZE:
        all_files = all_files[:SUBSET_SIZE]
        print(f"Subsetting to {SUBSET_SIZE} images for speed.")

    for img_path in all_files:
        class_name = os.path.basename(os.path.dirname(img_path)) 
        filename = os.path.basename(img_path)
        
        if class_name in BIO_FEATURES:
            obs, conc = BIO_FEATURES[class_name]
            
            conversations = [
                {"role": "user", "content": "<image>\nAnalyze this histology slide."},
                {"role": "assistant", "content": f"Observation: The image shows {obs}. Conclusion: This is {conc}."}
            ]
            
            data.append({
                "id": filename,
                "class": class_name,
                "conversations": conversations
            })

    with open(INSTRUCTION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} instructions to {INSTRUCTION_FILE}")

if __name__ == "__main__":
    generate_cot_data()