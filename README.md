# 🧠 NovaMind — Build Your Own LLM from Scratch

**A complete, production-ready custom Large Language Model built entirely from scratch using pure PyTorch.**

NovaMind is the personal AI brain of Nova — an intelligent assistant built by Purushottam. Every single line of code, every weight, every architectural decision is handcrafted. No HuggingFace transformers library. Pure math, pure PyTorch.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NovaMind Architecture                     │
│                 Decoder-Only Transformer                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Token IDs: [BOS, t1, t2, ..., tn]                   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Token Embed   │  (vocab_size=8000, embed_dim=256)       │
│  │ + Pos Encode  │  (sinusoidal, context_length=512)       │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────┐  ×6 layers       │
│  │  Transformer Decoder Block           │                   │
│  │  ┌────────────────────────────────┐  │                   │
│  │  │ LayerNorm                      │  │                   │
│  │  │ Multi-Head Causal Attention    │  │  8 heads          │
│  │  │ (Q, K, V projections)         │  │  head_dim=32      │
│  │  │ Scaled Dot-Product + Mask     │  │                   │
│  │  │ + Residual Connection         │  │                   │
│  │  ├────────────────────────────────┤  │                   │
│  │  │ LayerNorm                      │  │                   │
│  │  │ Feed-Forward Network           │  │  1024 inner dim   │
│  │  │ (Linear → GELU → Linear)      │  │                   │
│  │  │ + Residual Connection         │  │                   │
│  │  └────────────────────────────────┘  │                   │
│  └──────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ Final LN     │                                          │
│  │ LM Head      │  (embed_dim → vocab_size)                │
│  │ (weight tied)│                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  Output Logits: (batch, seq_len, vocab_size)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Model Specifications

| Parameter | Value |
|---|---|
| Parameters | ~4.7M |
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Layers | 6 |
| Context Length | 512 tokens |
| Vocabulary Size | 8,000 (BPE) |
| Feed-Forward Dim | 1,024 |
| Activation | GELU |
| Architecture | Pre-Norm Decoder-Only |

**~4.7M parameters** — small enough to train on free Google Colab GPUs, large enough to learn meaningful language patterns.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/purushottam/novamind.git
cd novamind
pip install -e .
```

### 2. Collect Training Data

```bash
python -m data.collector
```

Downloads books from Project Gutenberg, Wikipedia articles, and arXiv abstracts automatically.

### 3. Clean the Data

```bash
python -m data.cleaner personal_data
```

### 4. Train the Model

```bash
python -m training.train --data_dir personal_data --max_steps 5000
```

### 5. Chat with Nova

```python
from inference.chat import NovaChatEngine

engine = NovaChatEngine("weights/final_model", "weights/tokenizer")
response = engine.chat("Tell me about black holes")
print(response)
```

---

## 📁 Project Structure

```
novamind/
├── model/
│   ├── config.py             # All hyperparameters in one place
│   ├── attention.py          # Multi-head causal self-attention
│   ├── positional.py         # Sinusoidal positional encoding
│   ├── feedforward.py        # FFN with GELU activation
│   ├── block.py              # Transformer decoder block (pre-norm)
│   ├── architecture.py       # Full NovaMind model class
│   └── utils.py              # Weight init, model summary, FLOPs
├── tokenizer/
│   ├── bpe.py                # Byte Pair Encoding from scratch
│   ├── special_tokens.py     # PAD, BOS, EOS, UNK definitions
│   └── tokenizer.py          # Main tokenizer with save/load
├── data/
│   ├── collector.py          # Download training text
│   ├── cleaner.py            # Text normalization
│   ├── dataset.py            # PyTorch Dataset with sliding window
│   └── dataloader.py         # DataLoader factory
├── training/
│   ├── optimizer.py          # AdamW with param groups
│   ├── scheduler.py          # Cosine LR with warmup
│   ├── loss.py               # Cross-entropy + label smoothing
│   ├── trainer.py            # Full training loop
│   ├── checkpointing.py      # Save/load/resume checkpoints
│   └── train.py              # Entry point
├── inference/
│   ├── sampler.py            # Sampling strategies
│   ├── generate.py           # Streaming generation
│   ├── chat.py               # Chat engine (Groq API replacement)
│   └── evaluate.py           # Perplexity evaluation
├── personal_data/            # Your training text files
├── notebooks/
│   └── colab_train.ipynb     # Google Colab training notebook
├── weights/                  # Saved model checkpoints
├── logs/                     # Training logs and plots
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🎓 Training on Google Colab (GPU 🚀)

To train NovaMind using a free T4 GPU on Google Colab:

1. **Open the Notebook**: Locate `notebooks/colab_train.ipynb` and open it in Google Colab.
2. **Setup GPU**: Go to `Runtime` → `Change runtime type` → `T4 GPU`.
3. **One-Click Setup**:
   - The notebook now automatically clones your repository:
     `!git clone https://github.com/PurushottamGH/NOVA.git /content/novamind`
   - It also handles project path setup and dependency installation.
4. **Upload Your Data (Optional)**:
   - If you want to use your local `personal_data/`, run `python scripts/zip_data.py` locally.
   - Upload the resulting `personal_data.zip` to Colab's file browser.
   - The notebook handles the extraction!
5. **Run All**: Hit `Runtime` → `Run all` and watch NovaMind learn!

---

## 🔌 Integrating into Nova FastAPI Backend

Replace the Groq API call in your FastAPI backend with NovaMind:

```python
# === Before (Groq API) ===
from groq import Groq
client = Groq(api_key="your-key")

@app.post("/chat")
async def chat(request: ChatRequest):
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": request.message}],
        model="llama-3.3-70b-versatile",
    )
    return {"response": completion.choices[0].message.content}


# === After (NovaMind) ===
from inference.chat import NovaChatEngine

engine = NovaChatEngine("weights/final_model", "weights/tokenizer")

@app.post("/chat")
async def chat(request: ChatRequest):
    response = engine.chat(request.message, request.history)
    return {"response": response}

# Streaming endpoint
@app.get("/stream")
async def stream(message: str):
    async def generate():
        for token in engine.stream_chat(message):
            yield f"data: {token}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 📚 Adding Your Own Training Data

1. Create `.txt` files with the content you want Nova to learn
2. Place them in `personal_data/`
3. Run `python -m data.cleaner personal_data` to clean
4. Re-train: `python -m training.train --resume`

### Best data sources:
- Your own notes, essays, blog posts
- Technical documentation
- Books and articles in your areas of interest
- Formatted conversations: `User: ...\nNova: ...`

---

## 📈 Improving the Model Over Time

### More Data
The single biggest improvement comes from more training data. Aim for:
- 10M+ characters for basic language understanding
- 50M+ characters for domain-specific knowledge
- 100M+ characters for general competence

### Longer Training
- Increase `max_steps` in config (try 10000, 20000, 50000)
- Monitor loss curve — stop when validation loss plateaus

### Scale Up Architecture
Edit `model/config.py`:
```python
# Larger model (~20M params, needs GPU)
config = NovaMindConfig(
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    feedforward_dim=2048,
    vocab_size=16000,
    context_length=1024,
)
```

### Fine-Tuning
After pre-training on general text, fine-tune on conversations:
1. Prepare conversation data in `User: ...\nNova: ...` format
2. Train for fewer steps with lower learning rate

---

## 🧮 Understanding Parameter Count

For the default config (~4.7M parameters):

| Component | Parameters |
|---|---|
| Token Embedding | 8,000 × 256 = 2,048,000 |
| Per Attention Block | 4 × 256² = 262,144 |
| Per FFN Block | 2 × 256 × 1024 = 524,288 |
| Per Decoder Block | ~786,432 |
| All 6 Blocks | ~4,718,592 |
| LayerNorm + Other | ~2,000 |
| LM Head (tied) | 0 (shared with embedding) |
| **Total** | **~4.7M** |

---

## 🛠️ Tech Stack

- **PyTorch** — Deep learning framework (the only ML dependency)
- **Python** — Core language
- **NumPy** — Numerical computing
- **Matplotlib** — Loss curve visualization
- **tqdm** — Progress bars
- **requests** — Data collection

No HuggingFace. No TensorFlow. No shortcuts. Pure understanding.

---

## 📜 License

MIT License — use freely, learn deeply, build always.

---

**Built with 💜 by Purushottam — from Chennai to the stars 🌟**
