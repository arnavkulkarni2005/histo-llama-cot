import os
import torch

BASE_DIR = "/home/kulkarni/projects/biomed"
DATA_DIR = os.path.join(BASE_DIR, "data/raw_slides/NCT-CRC-HE-100K")
EMBEDDING_DIR = os.path.join(BASE_DIR, "data/embeddings")
INSTRUCTION_FILE = os.path.join(BASE_DIR, "data/instructions/train_cot.json")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISION_MODEL = "vinid/plip"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 4e-5
MAX_SEQ_LEN = 128
SUBSET_SIZE = 10000  # Trained on 10k images (10% of full dataset)

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INSTRUCTION_FILE), exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)