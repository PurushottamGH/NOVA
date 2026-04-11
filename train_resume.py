import os, sys, math, importlib, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor
import glob, random, json, re

PROJECT_DIR = "/kaggle/working/NOVA"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# ensure __init__.py in all packages
for pkg in ["model", "data", "tokenizer", "training"]:
    p = os.path.join(PROJECT_DIR, pkg)
    i = os.path.join(p, "__init__.py")
    if os.path.isdir(p) and not os.path.exists(i):
        open(i, "w").close()

# auto-find NovaMind across all model/ files
def find_novamind():
    model_dir = os.path.join(PROJECT_DIR, "model")
    priority  = ["nova", "architecture", "model", "transformer", "core"]
    files = [f[:-3] for f in os.listdir(model_dir)
             if f.endswith(".py") and f != "__init__.py"]
    ordered = sorted(files, key=lambda x: priority.index(x) if x in priority else 99)
    for name in ordered:
        try:
            mod = importlib.import_module("model." + name)
            if hasattr(mod, "NovaMind"):
                print("NovaMind found in model/" + name + ".py", flush=True)
                return mod.NovaMind
        except Exception:
            continue
    raise ImportError("NovaMind not found in any model/ file")

NovaMind = find_novamind()

config = {
    "vocab_size"        : 32000,
    "d_model"           : 1024,
    "n_heads"           : 16,
    "n_layers"          : 24,
    "d_ffn"             : 4096,
    "context_length"    : 512,
    "batch_size"        : 8,
    "learning_rate"     : 3e-4,
    "weight_decay"      : 0.1,
    "max_steps"         : 50000,
    "accumulation_steps": 4,
    "grad_clip"         : 1.0,
    "save_every"        : 1000,
    "eval_every"        : 500,
    "eval_batches"      : 50,
}

CHECKPOINT_STEP = 8000
HF_REPO  = "iPurushottam/NOAV_LLM"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device("cuda:" + str(rank))
torch.cuda.set_device(device)
is_master  = (rank == 0)

def log(msg):
    if is_master:
        print(msg, flush=True)

log("DDP ready — " + str(world_size) + " GPUs")

class FastTokenizer:
    def __init__(self, tok_dir):
        with open(os.path.join(tok_dir, "vocab.json"), "r") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        txt_path  = os.path.join(tok_dir, "merges.txt")
        json_path = os.path.join(tok_dir, "merges.json")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.read().split("\n")[1:]
            self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(lines) if m.strip()}
        elif os.path.exists(json_path):
            with open(json_path) as f:
                merges = json.load(f)
            self.bpe_ranks = {tuple(m): i for i, m in enumerate(merges)}
        else:
            raise FileNotFoundError("No merges file found in " + tok_dir)
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+")

    def encode(self, text):
        tokens = []
        for word in self.pat.findall(text):
            chars = list(word.encode("utf-8").decode("latin-1"))
            while len(chars) >= 2:
                pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                best  = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
                if best not in self.bpe_ranks:
                    break
                a, b = best
                new, i = [], 0
                while i < len(chars):
                    if i < len(chars)-1 and chars[i] == a and chars[i+1] == b:
                        new.append(a + b)
                        i += 2
                    else:
                        new.append(chars[i])
                        i += 1
                chars = new
            tokens.extend(self.encoder.get(c, 0) for c in chars)
        return tokens

class NovaStreamDataset(Dataset):
    def __init__(self, data_dir, tok_dir, context_len, split="train", val_fraction=0.05):
        self.context_len = context_len
        tok = FastTokenizer(tok_dir)
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        random.seed(42)
        random.shuffle(all_files)
        split_idx  = max(1, int(len(all_files) * (1 - val_fraction)))
        self.files = all_files[:split_idx] if split == "train" else all_files[split_idx:]
        if is_master:
            print("Tokenizing " + str(len(self.files)) + " files [" + split + "]...", flush=True)
        def tok_file(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return tok.encode(f.read())
            except Exception:
                return []
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(tok_file, self.files))
        self.token_ids = []
        for r in results:
            self.token_ids.extend(r)
        if is_master:
            print("Tokens [" + split + "]: " + str(len(self.token_ids)), flush=True)

    def __len__(self):
        return max(0, len(self.token_ids) - self.context_len - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.context_len + 1]
        if len(chunk) < self.context_len + 1:
            chunk = chunk + [0] * (self.context_len + 1 - len(chunk))
        x = torch.tensor(chunk[:self.context_len],    dtype=torch.long)
        y = torch.tensor(chunk[1:self.context_len+1], dtype=torch.long)
        return {"input_ids": x, "labels": y}

# locate data
DATA_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if any(f.endswith(".txt") for f in files):
        DATA_DIR = root
        break
assert DATA_DIR, "No .txt files found — attach nova-llmv1 dataset!"

# locate tokenizer — walks all subdirs, finds vocab.json
TOK_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "vocab.json" in files:
        TOK_DIR = root
        break
if TOK_DIR is None:
    TOK_DIR = os.path.join(PROJECT_DIR, "tokenizer_data")

log("Data : " + DATA_DIR)
log("Tok  : " + TOK_DIR)

train_ds = NovaStreamDataset(DATA_DIR, TOK_DIR, config["context_length"], split="train")
val_ds   = NovaStreamDataset(DATA_DIR, TOK_DIR, config["context_length"], split="val")

train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
train_loader  = DataLoader(train_ds, batch_size=config["batch_size"],
                           sampler=train_sampler, num_workers=4,
                           pin_memory=True, prefetch_factor=2)
val_loader    = DataLoader(val_ds, batch_size=config["batch_size"],
                           shuffle=False, num_workers=2, pin_memory=True)

log("Train: " + str(len(train_ds)) + " | Val: " + str(len(val_ds)))

model = NovaMind(config).to(device)
model = DDP(model, device_ids=[rank], find_unused_parameters=False)

ckpt_path = "weights/checkpoints/step_" + str(CHECKPOINT_STEP) + ".pt"
assert os.path.exists(ckpt_path), "Checkpoint not found: " + ckpt_path
ckpt = torch.load(ckpt_path, map_location=device)
model.module.load_state_dict(ckpt["model_state_dict"])
log("Loaded step " + str(ckpt["step"]))

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
    weight_decay=config["weight_decay"], betas=(0.9, 0.95), eps=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["max_steps"], eta_min=config["learning_rate"] * 0.1)
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
scheduler.load_state_dict(ckpt["scheduler_state_dict"])

start_step    = ckpt["step"]
best_val_loss = ckpt["best_val_loss"]
log("Resumed step " + str(start_step) + " | best_val=" + str(round(best_val_loss, 4)))

scaler = torch.amp.GradScaler("cuda")

def upload_hf(path, step):
    if not HF_TOKEN:
        return
    try:
        HfApi().upload_file(
            path_or_fileobj=path,
            path_in_repo="checkpoints/step_" + str(step) + ".pt",
            repo_id=HF_REPO,
            token=HF_TOKEN)
        log("Uploaded step_" + str(step) + ".pt")
    except Exception as e:
        log("Upload failed: " + str(e))

def evaluate(model, loader, device, max_b):
    model.eval()
    tl = torch.tensor(0.0, device=device)
    tc = torch.tensor(0,   device=device)
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= max_b:
                break
            x = b["input_ids"].to(device, non_blocking=True)
            y = b["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = model(x, labels=y)
                if isinstance(loss, tuple):
                    loss = loss[0]
            tl += loss.detach()
            tc += 1
    dist.all_reduce(tl, op=dist.ReduceOp.SUM)
    dist.all_reduce(tc, op=dist.ReduceOp.SUM)
    model.train()
    return (tl / tc).item()

data_iter    = iter(train_loader)
step         = start_step
running_loss = 0.0
model.train()
optimizer.zero_grad()

while step < config["max_steps"]:
    try:
        batch = next(data_iter)
    except StopIteration:
        train_sampler.set_epoch(step)
        data_iter = iter(train_loader)
        batch = next(data_iter)

    x = batch["input_ids"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda"):
        loss = model(x, labels=y)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss / config["accumulation_steps"]

    scaler.scale(loss).backward()
    running_loss += loss.item() * config["accumulation_steps"]

    if (step + 1) % config["accumulation_steps"] == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    if is_master and step % 100 == 0:
        avg = running_loss / 100 if step > start_step else running_loss
        print("Step " + str(step) + "/" + str(config["max_steps"]) +
              " | Loss: " + str(round(avg, 4)) +
              " | LR: " + str(round(scheduler.get_last_lr()[0], 8)), flush=True)
        running_loss = 0.0

    if step % config["eval_every"] == 0 and step > start_step:
        vl  = evaluate(model, val_loader, device, config["eval_batches"])
        ppl = math.exp(min(vl, 20))
        log("=" * 50)
        log("EVAL " + str(step) + " | loss=" + str(round(vl,4)) + " | ppl=" + str(round(ppl,2)))
        log("=" * 50)
        if vl < best_val_loss:
            best_val_loss = vl
            log("New best: " + str(round(best_val_loss, 4)))

    if is_master and step % config["save_every"] == 0 and step > start_step:
        os.makedirs("weights/checkpoints", exist_ok=True)
        sp = "weights/checkpoints/step_" + str(step) + ".pt"
        torch.save({
            "step"                : step,
            "model_state_dict"    : model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss"                : loss.item() * config["accumulation_steps"],
            "best_val_loss"       : best_val_loss,
            "config"              : config,
        }, sp)
        log("Saved " + sp)
        upload_hf(sp, step)

    step += 1

dist.destroy_process_group()
log("Training complete!")
