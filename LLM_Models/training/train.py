import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm  # for progress bars

sys.path.append(os.path.abspath(".."))

from tokenizer.tokenizer import get_tokenizer
from dataset.qa_dataset import QADataset
from model.transformer_block import Transformer

# 1) Hyperparameters
DATA_PATH    =  r"C:\Users\Admin\Desktop\LLM_QA\data\processed\processed_aquaponics_dataset.json"
BATCH_SIZE   = 8
NUM_EPOCHS   = 10
LEARNING_RATE = 3e-5
MAX_LEN      = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH    = "data/processed/qa_transformer.pt"

# 2) Prepare tokenizer, dataset, dataloader
tokenizer = get_tokenizer()
train_dataset = QADataset(DATA_PATH, tokenizer, max_length=MAX_LEN)
train_loader  = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda samples: tokenizer.pad(
        {
            "input_ids": [s["input_ids"] for s in samples],
            "attention_mask": [s["attention_mask"] for s in samples],
            "start_positions": torch.stack([s["start_positions"] for s in samples]),
            "end_positions":   torch.stack([s["end_positions"]   for s in samples])
        },
        return_tensors="pt"
    )
)

# 3) Initialize model, optimizer, and loss
model = Transformer(vocab_size=tokenizer.vocab_size).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn   = CrossEntropyLoss()

# 4) Training loop
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in progress:
        # Move tensors to device
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        start_labels   = batch["start_positions"].to(DEVICE)
        end_labels     = batch["end_positions"].to(DEVICE)

        optimizer.zero_grad()
        
        # Forward pass
        start_logits, end_logits = model(input_ids, attention_mask)
        # [batch_size, seq_len] each, so squeeze extra dims if needed
        if start_logits.dim() > 2:
            start_logits = start_logits.squeeze(-1)
            end_logits   = end_logits.squeeze(-1)
        
        # Compute losses
        loss_start = loss_fn(start_logits, start_labels)
        loss_end   = loss_fn(end_logits,   end_labels)
        loss       = (loss_start + loss_end) / 2
        
        # Backward + optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress.set_postfix(loss=epoch_loss / (progress.n + 1))
    
    print(f"Epoch {epoch+1} finished — Avg Loss: {epoch_loss/len(train_loader):.4f}")

# 5) Save the fine-tuned model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
