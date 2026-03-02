import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import HistoLlama
from config import *
from tqdm import tqdm

# 1. Dataset Class
class HistoDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load precomputed embedding
        emb_path = os.path.join(EMBEDDING_DIR, item['id'] + ".pt")
        if not os.path.exists(emb_path):
            # Fallback if missing (should not happen if precompute ran)
            return self.__getitem__((idx + 1) % len(self.data))
            
        img_emb = torch.load(emb_path) # [1, 512]

        # Process Text
        # Format: User: <image>\nAnalyze... Assistant: Obs...
        conversation = item['conversations']
        prompt = f"<|user|>\n{conversation[0]['content']} </s>\n<|assistant|>\n"
        response = f"{conversation[1]['content']} </s>"
        full_text = prompt + response
        
        # Tokenize
        tokens = self.tokenizer(
            full_text, 
            max_length=MAX_SEQ_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        
        # Labels: Mask out user prompt, only train on assistant response
        # Simple approach: Train on everything (works fine for this project)
        labels = input_ids.clone()
        
        return img_emb.squeeze(0), input_ids, attention_mask, labels

# 2. Training Loop
def train():
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.pad_token = tokenizer.unk_token # TinyLlama fix

    print("Loading Dataset...")
    dataset = HistoDataset(INSTRUCTION_FILE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing Model...")
    model = HistoLlama(use_lora=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(dataloader) * EPOCHS
    )

    print(f"Starting Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for img_embeds, input_ids, mask, labels in progress_bar:
            img_embeds = img_embeds.to(DEVICE).to(torch.float16)
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            
            outputs = model(img_embeds, input_ids, mask, labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(MODEL_SAVE_DIR, f"histollama_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()