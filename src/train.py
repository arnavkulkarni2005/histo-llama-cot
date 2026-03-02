import torch
import json
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import HistoLlama
from config import *
from tqdm import tqdm

class HistoDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        emb_path = os.path.join(EMBEDDING_DIR, item['id'] + ".pt")
        if not os.path.exists(emb_path):
            return self.__getitem__((idx + 1) % len(self.data))
            
        img_emb = torch.load(emb_path)

        conversation = item['conversations']
        prompt = f"<|user|>\n{conversation[0]['content']} </s>\n<|assistant|>\n"
        response = f"{conversation[1]['content']} </s>"
        full_text = prompt + response
        
        tokens = self.tokenizer(
            full_text, 
            max_length=MAX_SEQ_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        labels = input_ids.clone()
        
        return img_emb.squeeze(0), input_ids, attention_mask, labels

def train():
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.pad_token = tokenizer.unk_token

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

    os.makedirs("results", exist_ok=True)
    print(f"Starting Training for {EPOCHS} epoch(s)...")
    
    for epoch in range(EPOCHS):
        current_epoch = epoch + 1
        total_loss = 0
        epoch_step_losses = [] 
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch}")
        
        for img_embeds, input_ids, mask, labels in progress_bar:
            img_embeds = img_embeds.to(DEVICE).to(torch.float16)
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            
            model.llm.config.use_cache = False 
            
            outputs = model(img_embeds, input_ids, mask, labels)
            loss = outputs.loss
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            epoch_step_losses.append(loss_val)
            
            progress_bar.set_postfix({"loss": loss_val})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {current_epoch} Complete. Avg Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(MODEL_SAVE_DIR, f"histollama_epoch_{current_epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")


        print("Generating step-wise loss plot...")
        plt.figure(figsize=(12, 6))
        
        steps = range(1, len(epoch_step_losses) + 1)
        
        plt.plot(steps, epoch_step_losses, label="Raw Step Loss", color='#1f77b4', alpha=0.3, linewidth=1.5)
        
        window = 25
        if len(epoch_step_losses) >= window:
            smoothed = [sum(epoch_step_losses[i-window:i])/window if i >= window else sum(epoch_step_losses[:i+1])/(i+1) for i in range(len(epoch_step_losses))]
            plt.plot(steps, smoothed, label=f"Smoothed Trend (Window={window})", color='#d62728', linewidth=2.5)
        
        plt.title(f"HistoLlama-CoT Training Loss - Epoch {current_epoch} ({len(epoch_step_losses)} Steps)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Step", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.xlim(1, len(epoch_step_losses))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc="upper right")
        
        plot_path = os.path.join("results", f"step_loss_epoch_{current_epoch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved high-res loss plot to {plot_path}")

if __name__ == "__main__":
    train()