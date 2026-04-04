import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
import argparse
import time

from model.architecture import NovaMind
from tokenizer.tokenizer import NovaMindTokenizer
from tokenizer.special_tokens import PAD_TOKEN_ID

class SFTDataset(Dataset):
    """
    Specialized dataset for Supervised Fine-Tuning (SFT).
    Processes chat-formatted data (<|user|> / <|assistant|>) and applies target masking
    so the model only trains on assistant responses.
    """
    def __init__(self, data_dir, tokenizer, context_length=512):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples = []

        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"[Dataset] Warning: Data dir {data_dir} not found.")
            return

        # Load all .txt files
        text_files = list(data_path.glob("*.txt"))
        print(f"[Dataset] Loading {len(text_files)} files from {data_dir}...")

        for f in text_files:
            try:
                content = f.read_text(encoding="utf-8")
                # Split into conversation blocks on <|user|>
                # Each block starts with <|user|> and presumably ends before the next <|user|>
                blocks = content.split("<|user|>")
                for block in blocks:
                    block = block.strip()
                    if not block or "<|assistant|>" not in block:
                        continue
                    
                    # Re-add tag for proper tokenization
                    full_text = "<|user|>" + block
                    self.examples.append(full_text)
            except Exception as e:
                print(f"  Error reading {f}: {e}")

        print(f"[Dataset] Total conversation blocks: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # 1. Tokenize full sequence
        tokens = self.tokenizer.encode(text)
        
        # 2. Prepare inputs and targets (standard language modeling)
        # Shifted by 1 in the trainer or here? We'll do it in the trainer/loss usually,
        # but let's provide standard input_ids and target_ids where targets = inputs.
        # We will mask the target_ids for user turns.
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        target_ids = input_ids.clone()
        
        # 3. Mask target_ids for the user turn
        # Target: Mask indices from start until the first token of the assistant's response.
        # We find the token IDs for "<|assistant|>"
        assistant_tag = "<|assistant|>"
        text_parts = text.split(assistant_tag)
        user_part = text_parts[0] + assistant_tag
        
        user_tokens = self.tokenizer.encode(user_part)
        user_len = len(user_tokens)
        
        # Set targets for user tokens to PAD_TOKEN_ID (ignored by loss)
        target_ids[:user_len] = PAD_TOKEN_ID
        
        # 4. Truncate/Pad to context_length
        if len(input_ids) > self.context_length:
            input_ids = input_ids[:self.context_length]
            target_ids = target_ids[:self.context_length]
        else:
            padding_len = self.context_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((padding_len,), PAD_TOKEN_ID, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.full((padding_len,), PAD_TOKEN_ID, dtype=torch.long)])
            
        return input_ids, target_ids

def sft_train(model_path, tokenizer_path, data_dir, epochs=3, lr=1e-4, batch_size=4, device="auto"):
    """Main Supervised Fine-Tuning loop."""
    
    # 1. Initialization
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[SFT] Loading model from {model_path}...")
    model = NovaMind.load(model_path, device=device)
    tokenizer = NovaMindTokenizer.load(tokenizer_path)
    
    # 2. Data Loading
    dataset = SFTDataset(data_dir, tokenizer, context_length=model.config.context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Simple linear warmup script
    warmup_steps = 100
    total_steps = len(dataloader) * epochs
    
    def get_lr_multiplier(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    # 4. Training Loop
    print("=" * 45)
    print("      NovaMind SFT Refinement Run        ")
    print("=" * 45)
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size}")
    
    checkpoint_dir = Path("sft_weights")
    checkpoint_dir.mkdir(exist_ok=True)
    
    global_step = 0
    model.train()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            # CrossEntropyLoss automatically ignores pad_token_id if we specify it
            logits, _ = model(input_ids)
            
            # Compute loss manually to use the masked targets
            # Shift logits and targets: model predicts NEXT token
            # Logits: (batch, seq_len, vocab_size)
            # Targets: (batch, seq_len)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
            loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
            
            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save Epoch Checkpoint
        epoch_dir = checkpoint_dir / f"epoch_{epoch+1}"
        model.save(epoch_dir)
        print(f"  ✓ Epoch {epoch+1} complete in {time.time() - epoch_start:.0f}s. Checkpoint saved.")

    # 5. Final Save
    final_dir = checkpoint_dir / "final"
    model.save(final_dir)
    # Save tokenizer as well in that directory for convenience
    tokenizer.save(final_dir)
    
    print(f"\n[SFT] Training complete. Final model saved to {final_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NovaMind Supervised Fine-Tuning")
    parser.add_argument("--model_path", type=str, default="weights/final_model", help="Path to base model")
    parser.add_argument("--tokenizer_path", type=str, default="weights/tokenizer", help="Path to tokenizer")
    parser.add_argument("--data_dir", type=str, default="personal_data", help="Directory with .txt SFT data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    
    args = parser.parse_args()
    
    sft_train(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device
    )
