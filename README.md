# 🧠 NovaMind: High-Performance Agentic LLM From Scratch

<div align="center">

![NovaMind Banner](https://img.shields.io/badge/NovaMind-85M_--_300M-blueviolet?style=for-the-badge&logo=pytorch)
![Build Status](https://img.shields.io/badge/Status-Harden--Proof-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A state-of-the-art, decoder-only Transformer architecture built for specialized agentic intelligence.**

[Architecture](#-architecture) • [Focused Data](#-focused-data-pipeline) • [SFT Pipeline](#-sft-refinement) • [Deployment](#-cloud-training)

</div>

---

## ⚡ The Vision
NovaMind is not just another language model; it is a **hardened agentic brain**. Designed to transcend raw text generation, NovaMind integrates a sophisticated tool-routing layer, a persistent memory engine (ChromaDB), and specialized expertise in **Mathematics**, **Python Programming**, and **3D Automation (Blender)**.

## 🏗️ Technical Architecture
NovaMind utilizes a pre-norm, multi-head causal attention architecture optimized for low-latency inference and high-stability training.

| Configuration | Parameters | Hidden Dim | Heads | Layers | Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **85M (Fast)** | 85,248,000 | 768 | 12 | 12 | 1024 |
| **300M (Power)**| 298,421,760 | 1024 | 16 | 24 | 2048 |

### Key Innovations:
*   **Target Masking SFT**: Loss is only computed on assistant responses, ensuring the model learns *behavior*, not just repetition.
*   **Tokenizer Synchronization**: Custom BPE tokenizer with specialized boundary markers for perfect code/math reconstruction.
*   **Tool Router**: Integrated logic for real-time code execution, web search, and Blender API orchestration.

---

## 📊 Focused Data Pipeline
Intelligence is a function of data quality. NovaMind survives on a multi-stage **Focused Collector** strategy:

1.  **Reasoning (GSM8K/Orca)**: 12k+ high-quality mathematical reasoning chains.
2.  **Automation (Blender API)**: Full documentation scraping + 100+ production-grade Python/BPY scripts.
3.  **Knowledge (StackOverflow)**: Specialized Q&A across `pytorch`, `pandas`, `numpy`, and `python`.
4.  **Core SFT**: 200+ hand-crafted agentic instruction pairs.

---

## 🏋️ SFT Refinement
The Supervised Fine-Tuning (SFT) phase transforms a base pre-trained model into a helpful assistant.

```bash
python -m training.sft_train \
    --model_path weights/final_model \
    --data_dir personal_data \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-4
```

### Features:
*   **Cosine Schedule with Warmup** for stable gradient descent.
*   **AdamW Optimizer** with weight decay for better generalization.
*   **Automatic Device Scaling** (Multi-GPU T4 support).

---

## 🚀 Cloud Training (Kaggle Workflow)
NovaMind is designed for professional background training on Kaggle (2xT4 GPUs).

1.  **Pack**: Use `scripts/prepare_kaggle.py` to create a Linux-compatible archive.
2.  **Upload**: Upload `NovaMind_Kaggle_SFT_Final_v2.zip` to Kaggle as a Dataset.
3.  **Train**: Use the "Save & Run All" feature to train for up to 12 hours in the background.
4.  **Deploy**: Download `final_sft_model.pt` and swap it into your production environment.

---

## 📁 Project Structure
```text
NOVA/
├── model/           # Core Transformer architecture
├── tokenizer/       # Custom BPE with special token support
├── data/            # Multi-source Focused Collector logic
├── nova_modules/    # Agentic Layers (ToolRouter, Memory, Search)
├── training/        # SFT & Pre-training infrastructure
├── weights/         # Model checkpoints
└── scripts/         # Deployment & Hardening utilities
```

---

## 📜 License
MIT License. Built for the future of open-source agentic intelligence.

**Crafted with precision by Purushottam — Chennai 🌟**
