import glob
import json
import math
import multiprocessing
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ── Max performance settings ──────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

PROJECT_DIR = "/kaggle/working/NOVA"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

for pkg in ["model", "data", "tokenizer", "training"]:
    p = os.path.join(PROJECT_DIR, pkg)
    i = os.path.join(p, "__init__.py")
    if os.path.isdir(p) and not os.path.exists(i):
        open(i, "w").close()

from model.architecture import NovaMind  # noqa: E402
from model.config import NovaMindConfig  # noqa: E402

# ── Use kaggle_config — optimized for 2xT4 ───────────────────────────────────
config = NovaMindConfig.kaggle_config()
config.max_steps = 30000
config.save_every = 1000
config.eval_every = 500
config.gradient_checkpointing = True
config.device = "cuda"
config.batch_size = 16  # increased from 4 — T4s can handle it
config.accumulation_steps = 4  # effective batch = 16 * 4 * 2 GPUs = 128

CHECKPOINT_STEP = 8000
HF_REPO = "iPurushottam/NOAV_LLM"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NUM_CPU = multiprocessing.cpu_count()

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda:" + str(rank))
torch.cuda.set_device(device)
is_master = rank == 0


def log(msg):
    if is_master:
        print(msg, flush=True)


log("=" * 60)
log("NovaMind v2 — High Performance Training")
log("GPUs      : " + str(world_size))
log("CPU cores : " + str(NUM_CPU))
log("Batch/GPU : " + str(config.batch_size))
log("Eff batch : " + str(config.batch_size * config.accumulation_steps * world_size))
log("=" * 60)


# ── Fast tokenizer ────────────────────────────────────────────────────────────
class FastTokenizer:
    def __init__(self, tok_dir):
        with open(os.path.join(tok_dir, "vocab.json")) as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        txt_path = os.path.join(tok_dir, "merges.txt")
        json_path = os.path.join(tok_dir, "merges.json")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                lines = f.read().split("\n")[1:]
            self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(lines) if m.strip()}
        elif os.path.exists(json_path):
            with open(json_path) as f:
                merges = json.load(f)
            self.bpe_ranks = {tuple(m): i for i, m in enumerate(merges)}
        else:
            raise FileNotFoundError("No merges file in " + tok_dir)
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+")

    def encode(self, text):
        tokens = []
        for word in self.pat.findall(text):
            chars = list(word.encode("utf-8").decode("latin-1"))
            while len(chars) >= 2:
                pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
                best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
                if best not in self.bpe_ranks:
                    break
                a, b = best
                new, i = [], 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new.append(a + b)
                        i += 2
                    else:
                        new.append(chars[i])
                        i += 1
                chars = new
            tokens.extend(self.encoder.get(c, 0) for c in chars)
        return tokens


# ── Global tokenizer for multiprocessing ─────────────────────────────────────
_TOK = None
_TOK_DIR = None


def _init_worker(tok_dir):
    global _TOK, _TOK_DIR
    _TOK_DIR = tok_dir
    _TOK = FastTokenizer(tok_dir)


def _tokenize_file(path):
    global _TOK
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return _TOK.encode(f.read())
    except Exception:
        return []


class NovaStreamDataset(Dataset):
    def __init__(self, data_dir, tok_dir, context_len, split="train", val_fraction=0.05):
        self.context_len = context_len
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not all_files:
            raise RuntimeError("No .txt files in " + data_dir)
        random.seed(42)
        random.shuffle(all_files)
        split_idx = max(1, int(len(all_files) * (1 - val_fraction)))
        self.files = all_files[:split_idx] if split == "train" else all_files[split_idx:]

        cache_path = "/kaggle/working/nova_tokens_" + split + ".pt"

        # Only rank 0 tokenizes to avoid disk I/O contention
        if rank == 0:
            t0 = time.time()
            print(
                "Tokenizing "
                + str(len(self.files))
                + " files ["
                + split
                + "] using "
                + str(NUM_CPU)
                + " CPUs...",
                flush=True,
            )
            # Use all CPU cores with process pool for true parallel tokenization
            workers = min(NUM_CPU, len(self.files))
            if workers > 0:
                with ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_worker, initargs=(tok_dir,)
                ) as ex:
                    results = list(ex.map(_tokenize_file, self.files, chunksize=4))
            else:
                results = [_tokenize_file(f) for f in self.files]

            self.token_ids = []
            for r in results:
                self.token_ids.extend(r)
            elapsed = time.time() - t0
            print(
                "Tokens ["
                + split
                + "]: "
                + str(len(self.token_ids))
                + " in "
                + str(round(elapsed, 1))
                + "s",
                flush=True,
            )

            torch.save(self.token_ids, cache_path)

        # Sync all ranks — rank 1 waits here
        dist.barrier()

        if rank != 0:
            self.token_ids = torch.load(cache_path, weights_only=False)

        if len(self.token_ids) == 0 and rank == 0:
            raise RuntimeError("0 tokens — check tokenizer path!")

    def __len__(self):
        return max(0, len(self.token_ids) - self.context_len - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.context_len + 1]
        if len(chunk) < self.context_len + 1:
            chunk = chunk + [0] * (self.context_len + 1 - len(chunk))
        x = torch.tensor(chunk[: self.context_len], dtype=torch.long)
        y = torch.tensor(chunk[1 : self.context_len + 1], dtype=torch.long)
        return {"input_ids": x, "labels": y}


# ── Locate data + tokenizer ───────────────────────────────────────────────────
DATA_DIR = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if any(f.endswith(".txt") for f in files):
        DATA_DIR = root
        break
assert DATA_DIR, "No .txt files found!"

TOK_DIR = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "vocab.json" in files:
        TOK_DIR = root
        break
if TOK_DIR is None:
    TOK_DIR = os.path.join(PROJECT_DIR, "tokenizer_data")

log("Data : " + DATA_DIR)
log("Tok  : " + TOK_DIR)

train_ds = NovaStreamDataset(DATA_DIR, TOK_DIR, config.context_length, split="train")
val_ds = NovaStreamDataset(DATA_DIR, TOK_DIR, config.context_length, split="val")

# ── High performance DataLoaders ──────────────────────────────────────────────
train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(
    train_ds,
    batch_size=config.batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

log("Train: " + str(len(train_ds)) + " | Val: " + str(len(val_ds)))

# ── Model ─────────────────────────────────────────────────────────────────────
model = NovaMind(config).to(device)

# Print param count on master
if is_master:
    info = model.count_parameters()
    log(
        "Parameters: "
        + str(info["total_million"])
        + "M total | "
        + str(info["trainable_million"])
        + "M trainable"
    )

model = DDP(model, device_ids=[rank], find_unused_parameters=False)

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_path = "weights/checkpoints/step_" + str(CHECKPOINT_STEP) + ".pt"
assert os.path.exists(ckpt_path), "Checkpoint not found: " + ckpt_path
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.module.load_state_dict(ckpt["model_state_dict"])
log("Loaded checkpoint from step " + str(ckpt["step"]))

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=True,  # fused AdamW — faster on CUDA
)


def get_lr(current_step):
    if current_step < config.warmup_steps:
        return config.learning_rate * current_step / max(1, config.warmup_steps)
    progress = (current_step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.1 + 0.5 * (
        config.learning_rate - config.learning_rate * 0.1
    ) * (1.0 + math.cos(math.pi * progress))


optimizer.load_state_dict(ckpt["optimizer_state_dict"])

start_step = ckpt["step"]
best_val_loss = ckpt["best_val_loss"]
log("Resumed from step " + str(start_step) + " | best_val=" + str(round(best_val_loss, 4)))

# ── AMP scaler ────────────────────────────────────────────────────────────────
scaler = torch.amp.GradScaler("cuda")


def upload_hf(path, current_step):
    if not HF_TOKEN:
        return
    try:
        HfApi().upload_file(
            path_or_fileobj=path,
            path_in_repo="checkpoints/step_" + str(current_step) + ".pt",
            repo_id=HF_REPO,
            token=HF_TOKEN,
        )
        log("Uploaded step_" + str(current_step) + ".pt")
    except Exception as e:
        log("HF upload failed: " + str(e))


def evaluate(eval_model, loader, eval_device, max_b):
    eval_model.eval()
    tl = torch.tensor(0.0, device=eval_device)
    tc = torch.tensor(0, device=eval_device)
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= max_b:
                break
            x_eval = b["input_ids"].to(eval_device, non_blocking=True)
            y_eval = b["labels"].to(eval_device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                _logits_eval, loss_eval = eval_model(x_eval, targets=y_eval)
            tl += loss_eval.detach()
            tc += 1
    dist.all_reduce(tl, op=dist.ReduceOp.SUM)
    dist.all_reduce(tc, op=dist.ReduceOp.SUM)
    eval_model.train()
    return (tl / tc).item()


# ── Training loop ─────────────────────────────────────────────────────────────
data_iter = iter(train_loader)
step = start_step
running_loss = 0.0
t_start = time.time()
model.train()
optimizer.zero_grad()

log("Training starts now...")

while step < config.max_steps:
    try:
        batch = next(data_iter)
    except StopIteration:
        train_sampler.set_epoch(step)
        data_iter = iter(train_loader)
        batch = next(data_iter)

    x = batch["input_ids"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)

    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    with torch.amp.autocast("cuda"):
        _logits, loss = model(x, targets=y)
        loss = loss / config.accumulation_steps

    scaler.scale(loss).backward()
    running_loss += loss.item() * config.accumulation_steps

    if (step + 1) % config.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if is_master and step % 50 == 0:
        elapsed = time.time() - t_start
        steps_so_far = max(1, step - start_step)
        sps = steps_so_far / elapsed  # steps per second
        eta_secs = (config.max_steps - step) / max(sps, 1e-6)
        eta_hrs = round(eta_secs / 3600, 1)
        avg = running_loss / 50 if step > start_step else running_loss
        print(
            "Step "
            + str(step)
            + "/"
            + str(config.max_steps)
            + " | Loss: "
            + str(round(avg, 4))
            + " | LR: "
            + str(round(lr, 7))
            + " | "
            + str(round(sps, 2))
            + " steps/s"
            + " | ETA: "
            + str(eta_hrs)
            + "h",
            flush=True,
        )
        running_loss = 0.0

    if step % config.eval_every == 0 and step > start_step:
        vl = evaluate(model, val_loader, device, 50)
        ppl = math.exp(min(vl, 20))
        log("=" * 55)
        log(
            "EVAL step="
            + str(step)
            + " | loss="
            + str(round(vl, 4))
            + " | ppl="
            + str(round(ppl, 2))
        )
        log("=" * 55)
        if vl < best_val_loss:
            best_val_loss = vl
            log("New best: " + str(round(best_val_loss, 4)))

    if is_master and step % config.save_every == 0 and step > start_step:
        os.makedirs("weights/checkpoints", exist_ok=True)
        sp = "weights/checkpoints/step_" + str(step) + ".pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item() * config.accumulation_steps,
                "best_val_loss": best_val_loss,
                "config": config,
            },
            sp,
        )
        log("Saved " + sp)
        upload_hf(sp, step)

    step += 1

dist.destroy_process_group()
log("Training complete!")
