"""
NovaMind Trainer
==================
Complete training loop class with:
- Gradient accumulation for effective larger batch sizes
- NaN/Inf detection and recovery
- Loss spike detection with automatic checkpoint reload
- Forward → loss → backward → grad clip → optimizer step → scheduler step
- Periodic evaluation with perplexity computation
- Sample generation during training
- Loss curve plotting
- Best model tracking on VALIDATION loss
- Accurate ETA calculation that works when resuming
- Memory cleanup after each batch

Usage:
    trainer = Trainer(model, config, train_loader, val_loader, tokenizer)
    trainer.train()
"""

import time
import math
import datetime
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers/Colab
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.loss import create_loss_fn, compute_perplexity
from training.checkpointing import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    delete_old_checkpoints
)


class Trainer:
    """
    Complete training loop for the NovaMind model.
    
    Args:
        model: NovaMind model instance
        config: NovaMindConfig with all training hyperparameters
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        tokenizer: NovaMindTokenizer for sample generation
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs and plots
        resume: Whether to resume from latest checkpoint
    """

    def __init__(self, model, config, train_loader, val_loader, tokenizer,
                 checkpoint_dir="weights", log_dir="logs", resume=True):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)  # FIXED: use pathlib.Path for Windows compat
        self.log_dir = Path(log_dir)  # FIXED: use pathlib.Path

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)

        # Create loss function (with slight label smoothing)
        self.loss_fn = create_loss_fn(config, smoothing=0.05)

        # Training state
        self.global_step = 0
        self.start_step = 0  # FIXED: track where training started for accurate ETA
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.prev_loss = None  # FIXED: track previous loss for spike detection

        # Resume from checkpoint if available
        if resume:
            latest = find_latest_checkpoint(str(self.checkpoint_dir))
            if latest:
                self.global_step, _ = load_checkpoint(
                    latest, self.model, self.optimizer, self.scheduler
                )
                self.start_step = self.global_step  # FIXED: remember where we resumed from
                # FIXED: restore best_val_loss from checkpoint
                ckpt = torch.load(latest, map_location=config.device, weights_only=False)
                self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
                # FIXED: restore RNG state for reproducibility
                if "rng_state" in ckpt:
                    torch.set_rng_state(ckpt["rng_state"])
                print(f"[Trainer] Resumed from step {self.global_step}, best_val_loss={self.best_val_loss:.4f}")

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """Main training loop with all 10 fixes applied."""
        print("\n" + "=" * 60)
        print("  NovaMind Training Started")
        print("=" * 60)
        info = self.model.count_parameters()
        print(f"  Parameters: {info['total_million']}M")
        print(f"  Device: {self.config.device}")
        print(f"  Max steps: {self.config.max_steps}")
        print(f"  Starting from step: {self.global_step}")
        print(f"  Gradient accumulation steps: {self.config.accumulation_steps}")  # FIXED: show accum steps
        print("=" * 60)

        self.model.train()
        train_start_time = time.time()  # FIXED: renamed for clarity in ETA calc
        self.start_step = self.global_step  # FIXED: for accurate ETA
        tokens_processed = 0
        running_loss = 0.0
        num_batches = 0

        # Create infinite data iterator
        data_iter = self._infinite_iterator()

        # FIXED: zero grad once at start for gradient accumulation
        self.optimizer.zero_grad()

        pbar = tqdm(
            range(self.global_step, self.config.max_steps),
            initial=self.global_step,
            total=self.config.max_steps,
            desc="Training",
        )

        for step in pbar:
            self.global_step = step
            step_start_time = time.time()  # FIXED: per-step timing for tokens/sec

            # Get next batch
            input_ids, target_ids = next(data_iter)
            input_ids = input_ids.to(self.config.device)   # (batch, context_length)
            target_ids = target_ids.to(self.config.device)  # (batch, context_length)

            # Forward pass
            logits, _ = self.model(input_ids)  # (batch, context_length, vocab_size)

            # Compute loss
            loss = self.loss_fn(
                logits.view(-1, self.config.vocab_size),  # (batch*seq_len, vocab_size)
                target_ids.view(-1)                         # (batch*seq_len,)
            )

            # ============================================================
            # FIXED: NaN/Inf detection — skip batch and zero grad (FIX 2)
            # Without this, NaN propagates silently and corrupts all weights
            # ============================================================
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  WARNING: NaN/Inf loss detected at step {step} — skipping batch")
                self.optimizer.zero_grad()
                del logits, loss  # FIXED: cleanup
                continue

            # ============================================================
            # FIXED: Loss spike detection (FIX 3)
            # If loss suddenly jumps 10x, training may have diverged
            # ============================================================
            loss_val = loss.item()
            if self.prev_loss is not None and loss_val > 10 * self.prev_loss and self.prev_loss > 0:
                print(f"\n  WARNING: Loss spike at step {step}: {loss_val:.4f} vs prev {self.prev_loss:.4f}")
                # Try to recover from best checkpoint
                best_ckpt = self.checkpoint_dir / "best.pt"
                if best_ckpt.exists():
                    print(f"  Reloading best checkpoint from {best_ckpt}...")
                    load_checkpoint(str(best_ckpt), self.model, self.optimizer, self.scheduler)
                    self.optimizer.zero_grad()
                    self.prev_loss = None
                    continue
                else:
                    print(f"  No best checkpoint found — continuing with caution")

            self.prev_loss = loss_val

            # ============================================================
            # FIXED: Gradient accumulation (FIX 1)
            # Divides loss and only steps every accumulation_steps batches
            # This effectively multiplies batch size without more memory
            # ============================================================
            accum_steps = self.config.accumulation_steps
            scaled_loss = loss / accum_steps  # FIXED: scale loss for accumulation
            scaled_loss.backward()

            if (step + 1) % accum_steps == 0:
                # Gradient clipping — prevents exploding gradients
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()  # FIXED: zero_grad after step, not before backward

            # Track metrics
            running_loss += loss_val
            num_batches += 1
            tokens_processed += input_ids.numel()
            current_lr = self.scheduler.get_last_lr()[0]

            self.train_losses.append(loss_val)
            self.learning_rates.append(current_lr)

            # ============================================================
            # FIXED: Accurate ETA calculation that works when resuming (FIX 8)
            # ============================================================
            step_time = time.time() - step_start_time  # FIXED: per-step time
            elapsed = time.time() - train_start_time
            steps_done = step - self.start_step + 1
            time_per_step = elapsed / max(steps_done, 1)
            steps_remaining = self.config.max_steps - step - 1
            eta_seconds = steps_remaining * time_per_step
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            # FIXED: tokens per second from per-step time (FIX 5)
            tokens_per_sec = input_ids.numel() / max(step_time, 1e-6)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{current_lr:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "ETA": eta_str,
            })

            # ============================================================
            # FIXED: Memory cleanup after each batch (FIX 6)
            # ============================================================
            del logits  # FIXED: free logits tensor
            del loss, scaled_loss  # FIXED: free loss tensors
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()  # FIXED: free GPU cache periodically

            # === Periodic Evaluation ===
            if (step + 1) % self.config.eval_every == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                perplexity = compute_perplexity(torch.tensor(val_loss))  # FIXED: use compute_perplexity helper

                avg_train = running_loss / max(num_batches, 1)
                print(f"\n  Step {step+1}/{self.config.max_steps}")
                print(f"  Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f} | PPL: {perplexity:.2f}")
                print(f"  LR: {current_lr:.2e} | Tokens/s: {tokens_per_sec:.0f} | ETA: {eta_str}")

                # FIXED: Generate a sample during training (FIX 10)
                self.generate_sample()

                # ============================================================
                # FIXED: Best model tracking on VALIDATION loss (FIX 7)
                # Was using train loss — now correctly uses val_loss
                # ============================================================
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # Save best model as a separate checkpoint
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        step + 1, val_loss, self.config, str(self.checkpoint_dir),
                        best_val_loss=self.best_val_loss,
                        is_best=True,  # FIXED: triggers best.pt save in checkpointing.py
                    )
                    print(f"  ★ New best model saved — val_loss: {val_loss:.4f}")

                running_loss = 0.0
                num_batches = 0
                self.model.train()  # Back to training mode

            # === Periodic Checkpoint ===
            if (step + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    step + 1, loss_val, self.config, str(self.checkpoint_dir),
                    best_val_loss=self.best_val_loss,
                    is_best=False,
                )
                delete_old_checkpoints(str(self.checkpoint_dir), keep_last_n=3)

        # Final save
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.config.max_steps, self.prev_loss or 0.0, self.config, str(self.checkpoint_dir),
            best_val_loss=self.best_val_loss,
        )

        # Plot loss curve
        self.plot_losses()

        total_time = time.time() - train_start_time
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best perplexity: {compute_perplexity(torch.tensor(self.best_val_loss)):.2f}")
        print(f"{'='*60}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Compute average loss on the validation set.
        FIXED: Properly uses model.eval() and torch.no_grad() (FIX 4)
        """
        self.model.eval()  # FIXED: verified — was already here but now documented as fix
        total_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.config.device)
            target_ids = target_ids.to(self.config.device)

            logits, _ = self.model(input_ids)
            loss = self.loss_fn(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1)
            )
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def generate_sample(self, prompt_text="The universe is", max_tokens=50):
        """Generate a text sample and print it. FIXED: reduced to 50 tokens for speed."""
        self.model.eval()

        try:
            # Encode the prompt
            prompt_ids = self.tokenizer.encode(prompt_text)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.config.device)

            # Generate
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=0.8, top_k=40, top_p=0.9
            )

            generated_text = self.tokenizer.decode(output_ids[0].tolist())
            print(f"\n  📝 Sample: \"{generated_text[:200]}\"")
        except Exception as e:
            print(f"\n  📝 Sample generation failed: {e}")

    def plot_losses(self):
        """Save a loss curve plot to the logs directory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        if self.train_losses:
            # Smooth training loss with moving average
            window = min(50, len(self.train_losses) // 5 + 1)
            if window > 1:
                smoothed = []
                for i in range(len(self.train_losses)):
                    start = max(0, i - window)
                    smoothed.append(sum(self.train_losses[start:i+1]) / (i - start + 1))
                ax1.plot(smoothed, label="Train Loss (smoothed)", color="#4ECDC4", linewidth=1.5)
            ax1.plot(self.train_losses, alpha=0.2, color="#4ECDC4", linewidth=0.5)
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Learning rate
        if self.learning_rates:
            ax2.plot(self.learning_rates, color="#FF6B6B", linewidth=1.5)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.log_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Trainer] Loss curve saved to {plot_path}")

    def _infinite_iterator(self):
        """Create an infinite iterator over the training DataLoader."""
        while True:
            for batch in self.train_loader:
                yield batch
