import torch
import json
import os
import glob
import random
import logging
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from PIL import Image
from model import HistoLlama
from config import *

# ---------------------------------------------------------
# 1. Biological Features Mapping (From our generation script)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Logging Setup
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("results/evaluation_debug.log"),
        logging.StreamHandler()
    ]
)

# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
def plot_confusion_matrix_heatmap(y_true, y_pred, classes, save_dir="results"):
    logging.info("Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('HistoLlama-CoT Confusion Matrix (Unseen Test Data)', fontsize=16)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    
    save_path = os.path.join(save_dir, "confusion_matrix_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved Confusion Matrix Heatmap to {save_path}")
    plt.close()

def visualize_qualitative_prediction(image, original_path, predicted_text, true_class, save_dir="results"):
    logging.debug(f"Generating qualitative figure for {original_path}")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"True Class: {true_class}", fontsize=12, fontweight='bold')
    
    wrapped_text = textwrap.fill(predicted_text, width=65)
    plt.figtext(0.5, 0.02, wrapped_text, wrap=True, horizontalalignment='center', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) 
    
    base_name = os.path.basename(original_path)
    save_path = os.path.join(save_dir, f"fig_{base_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# 4. Core Evaluation Pipeline
# ---------------------------------------------------------
def get_unseen_test_set(num_samples=1000):
    logging.info(f"Identifying unseen test images to prevent data leakage...")
    
    # 1. Get the 10,000 IDs used for training
    with open(INSTRUCTION_FILE, 'r') as f:
        train_data = json.load(f)
    train_ids = set([item['id'] for item in train_data])
    logging.debug(f"Loaded {len(train_ids)} training IDs to exclude.")

    # 2. Get all images in the raw dataset directory
    all_files = glob.glob(os.path.join(DATA_DIR, "*", "*.tif"))
    
    # 3. Filter out the training images
    unseen_files = [f for f in all_files if os.path.basename(f) not in train_ids]
    logging.info(f"Found {len(unseen_files)} strictly unseen images available for testing.")
    
    if len(unseen_files) < num_samples:
        logging.warning("Not enough unseen files to meet the requested sample size. Using all available unseen files.")
        num_samples = len(unseen_files)
        
    random.seed(42) # Reproducibility
    test_files = random.sample(unseen_files, num_samples)
    return test_files

def run_evaluation(checkpoint_path, num_test_samples=1000):
    logging.info(f"Starting Evaluation Pipeline with checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load Models
    logging.debug("Loading PLIP Vision Model...")
    processor = AutoProcessor.from_pretrained(VISION_MODEL)
    vision_model = AutoModel.from_pretrained(VISION_MODEL).to(DEVICE)
    vision_model.eval()
    
    logging.debug("Loading HistoLlama LLM and LoRA weights...")
    model = HistoLlama(use_lora=True)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Get the strictly unseen test files
    test_files = get_unseen_test_set(num_test_samples)
    
    y_true = []
    y_pred = []
    classes = list(BIO_FEATURES.keys())

    logging.info(f"Beginning inference over {len(test_files)} unseen test samples...")
    for i, img_path in enumerate(tqdm(test_files)):
        # Extract true class from directory structure (e.g., .../TUM/TUM-xxx.tif -> TUM)
        true_class = os.path.basename(os.path.dirname(img_path))
        y_true.append(true_class)
        
        # 1. Process Image
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_features = vision_model.get_image_features(**inputs).unsqueeze(1)

        # 2. Prepare Prompt
        prompt = "<|user|>\n<image>\nAnalyze this histology slide. </s>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # 3. Generate CoT Text
        with torch.no_grad():
            visual_tokens = model.projector(img_features.to(torch.float32))
            text_embeds = model.llm.get_input_embeddings()(inputs.input_ids)
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            
            output_ids = model.llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=100,
                do_sample=False
            )
            
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        clean_result = result.replace("</s>", "").split("<|assistant|>\n")[-1].strip()
        
        # 4. Extract Predicted Class based on generated reasoning
        predicted_class = "UNKNOWN"
        for cls_name in classes:
            if BIO_FEATURES[cls_name][1] in clean_result:
                predicted_class = cls_name
                break
        y_pred.append(predicted_class)
        
        # Generate qualitative figures for the first 5 samples for our report
        if i < 5:
            visualize_qualitative_prediction(image, img_path, clean_result, true_class)

    # ---------------------------------------------------------
    # 5. Compute and Log Metrics
    # ---------------------------------------------------------
    logging.info("Computing final statistical metrics...")
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*50)
    print(" HistoLlama-CoT EVALUATION RESULTS (UNSEEN DATA)")
    print("="*50)
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(report)
    
    logging.info(f"Final Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{report}")
    
    # Generate Heatmap
    plot_confusion_matrix_heatmap(y_true, y_pred, classes)
    logging.info("Evaluation Complete. Check the 'results/' directory for paper assets.")

if __name__ == "__main__":

    CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "histollama_epoch_3.pth") 

    run_evaluation(CHECKPOINT_PATH, num_test_samples=1000)