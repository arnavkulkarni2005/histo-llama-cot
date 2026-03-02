import os
import torch

# Paths
BASE_DIR = "/home/kulkarni/projects/biomed"
DATA_DIR = os.path.join(BASE_DIR, "data/raw_slides/NCT-CRC-HE-100K")
EMBEDDING_DIR = os.path.join(BASE_DIR, "data/embeddings")
INSTRUCTION_FILE = os.path.join(BASE_DIR, "data/instructions/train_cot.json")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")

# Model Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISION_MODEL = "vinid/plip"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training Settings
BATCH_SIZE = 16 # Increase to 32/64 if VRAM allows
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 128
SUBSET_SIZE = 10000  # Train on 10k images for speed (Set to None for full dataset)

# Ensure directories exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INSTRUCTION_FILE), exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)