import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import LLM_MODEL, DEVICE

class HistoLlama(nn.Module):
    def __init__(self, use_lora=True):
        super().__init__()
        
        print("Loading TinyLlama...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, 
            torch_dtype=torch.float32,
            device_map=DEVICE
        )
        
        # Enable Gradient Checkpointing for memory saving
        self.llm.gradient_checkpointing_enable()
        
        # Apply LoRA
        if use_lora:
            peft_config = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, peft_config)
            print("LoRA applied.")

        # Projection Layer: PLIP (512 dim) -> TinyLlama (2048 dim)
        self.projector = nn.Linear(512, 2048).to(DEVICE).to(torch.float32)

    def forward(self, image_embeds, input_ids, attention_mask, labels=None):
        # 1. Project image embeddings to LLM space
        
        # FIX: Ensure image_embeds is 3D [Batch, 1, 512]
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(1)

        # image_embeds shape: [Batch, 1, 512] -> [Batch, 1, 2048]
        image_visual_tokens = self.projector(image_embeds.to(DEVICE).to(torch.float32))
        
        # 2. Get text embeddings from LLM
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 3. Concatenate: [Image Token] + [Text Tokens]
        combined_embeds = torch.cat([image_visual_tokens, inputs_embeds], dim=1)
        
        # 4. Adjust Attention Mask (add 1 for the image token)
        ones = torch.ones((attention_mask.shape[0], 1), device=DEVICE)
        combined_mask = torch.cat([ones, attention_mask], dim=1)

        # 5. Forward Pass
        # We shift labels because we added 1 token at the start
        if labels is not None:
            dummy_labels = torch.full((labels.shape[0], 1), -100, device=DEVICE)
            combined_labels = torch.cat([dummy_labels, labels], dim=1)
            
            return self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=combined_labels
            )
        else:
            return self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask)