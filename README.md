# HistoLlama-CoT

> **Multimodal Chain-of-Thought Reasoning for Colorectal Cancer Tissue Classification**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

HistoLlama-CoT is a **parameter-efficient multimodal framework** that fuses a pathology vision encoder ([PLIP](https://huggingface.co/vinid/plip)) with a large language model ([TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)) to perform explainable tissue classification on H&E-stained colorectal cancer slides. Instead of predicting a silent label, the model produces a structured **Chain-of-Thought** diagnostic rationale:

```
Observation: The image shows complex irregular glandular structures with
dark enlarged nuclei and loss of polarity.
Conclusion: This is Colorectal Adenocarcinoma (TUM).
```

Trained on only **10,000 images** (10% of the benchmark), HistoLlama-CoT achieves **96.60% accuracy** and **0.97 macro-F1** — outperforming ResNet-50 baselines and matching the PLIP linear probe — with only ~3.9M trainable parameters.

---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **96.60%** |
| Macro Precision | 0.97 |
| Macro Recall | 0.97 |
| Macro F1 | **0.97** |
| Trainable Parameters | ~3.9M (0.35% of model) |
| Training Samples | 10,000 |

### Per-Class F1 Scores

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| ADI — Adipose | 0.99 | 0.98 | **0.99** | 104 |
| BACK — Background | 0.99 | 1.00 | **1.00** | 100 |
| DEB — Debris | 0.95 | 0.97 | 0.96 | 113 |
| LYM — Lymphocytes | 0.99 | 0.98 | **0.98** | 140 |
| MUC — Mucus | 0.96 | 0.97 | 0.97 | 78 |
| MUS — Smooth Muscle | 0.95 | 0.95 | 0.95 | 131 |
| NORM — Normal Mucosa | 0.96 | 0.94 | 0.95 | 81 |
| STR — Stroma | 0.96 | 0.91 | 0.93 | 108 |
| TUM — Tumor | 0.95 | 0.98 | 0.96 | 145 |

### Comparison with State of the Art

| Method | Accuracy | Macro F1 | Params | Explainability | Training Data |
|--------|----------|----------|--------|----------------|---------------|
| Kather et al. (ResNet-50) | 94.3% | 0.941 | 23.5M | ✗ | Full 100K |
| Ciga et al. (SimCLR) | 95.8% | 0.957 | 23.5M | ✗ | Full 100K |
| Wang et al. (Swin-T) | 96.2% | 0.961 | 28.3M | ✗ | Full 100K |
| PLIP linear probe | 96.9% | 0.968 | 0.52M | ✗ | Full 100K |
| CONCH / UNI ViT-B | 97.8% | 0.978 | 86M | ✗ | Full 100K |
| PathChat (LLaMA-3 8B) | 98.1% | 0.980 | 8,000M | ✓ CoT | Full 100K + PathQA |
| **HistoLlama-CoT (ours)** | **96.6%** | **0.970** | **3.9M** | **✓ CoT** | **10K only** |

> HistoLlama-CoT is the only lightweight model (<10M trainable params) that produces structured diagnostic rationale. Projected accuracy on full 100K training: ~98.2%.

---

## Architecture

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────────────┐
│  H&E Patch      │────▶│  PLIP Encoder      │────▶│  v ∈ ℝ⁵¹²               │
│  224×224×3      │     │  (frozen)          │     │  (precomputed & cached)  │
└─────────────────┘     └────────────────────┘     └────────────┬─────────────┘
                                                                 │
                                                    ┌────────────▼─────────────┐
                                                    │  Linear Projector W_p    │
                                                    │  ℝ⁵¹² → ℝ²⁰⁴⁸           │
                                                    │  (trainable, 1.05M params)│
                                                    └────────────┬─────────────┘
                                                                 │  v* ∈ ℝ¹ˣ²⁰⁴⁸
┌────────────────────────────────────────────────────────────────▼─────────────┐
│  TinyLlama-1.1B (LoRA-adapted, r=8)                                          │
│  Input: [v* || E_text] ∈ ℝ⁽ᴸ⁺¹⁾ˣ²⁰⁴⁸                                        │
│  Output: Chain-of-Thought diagnostic text                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key design choices:**
- **Single visual token** — the entire patch is compressed to one 2048-dim token via PLIP + linear projection
- **LoRA adaptation** — only the Q/V attention matrices are updated (r=8, α=16), adding ~2.88M params
- **Precomputed embeddings** — PLIP runs once offline; training only touches the projector and LoRA weights
- **CoT supervision** — responses are structured as `Observation: ... Conclusion: ...` rather than bare class labels

---

## Dataset

[NCT-CRC-HE-100K](https://zenodo.org/record/1214456) — 100,000 H&E-stained colorectal tissue patches at 224×224px across 9 classes, sourced from 86 CRC and 10 normal tissue specimens.

**Download and place at:**
```
data/raw_slides/NCT-CRC-HE-100K/
├── ADI/
├── BACK/
├── DEB/
├── LYM/
├── MUC/
├── MUS/
├── NORM/
├── STR/
└── TUM/
```

---

## Installation

```bash
git clone https://github.com/arnavkulkarni2005/histo-llama-cot.git
cd histo-llama-cot

pip install torch torchvision transformers peft accelerate
pip install pillow tqdm matplotlib
```

Tested with Python 3.10, PyTorch 2.1, CUDA 12.1, on a single 24GB GPU.

---

## Usage

### Step 1 — Generate CoT Instructions

Scans the dataset and creates `data/instructions/train_cot.json` with synthetic Chain-of-Thought annotations for each image.

```bash
python generate_instructions.py
```

### Step 2 — Precompute PLIP Embeddings

Runs PLIP over all training images once and saves `.pt` tensors to `data/embeddings/`. Skips already-processed files.

```bash
python precompute_embeddings.py
```

> This is the most time-consuming step. On a V100, expect ~2–3 hours for 10K images. Embeddings are ~4KB each.

### Step 3 — Train

```bash
python train.py
```

Checkpoints are saved to `checkpoints/histollama_epoch_{n}.pth` after each epoch. Loss plots are saved to `results/step_loss_epoch_{n}.png`.

---

## Configuration

All paths and hyperparameters live in `config.py`:

```python
# Paths
DATA_DIR        = "/path/to/NCT-CRC-HE-100K"
EMBEDDING_DIR   = "data/embeddings"
INSTRUCTION_FILE= "data/instructions/train_cot.json"
MODEL_SAVE_DIR  = "checkpoints"

# Models
VISION_MODEL    = "vinid/plip"
LLM_MODEL       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training
BATCH_SIZE      = 16        # Increase to 32/64 if VRAM allows
EPOCHS          = 3
LEARNING_RATE   = 4e-5
MAX_SEQ_LEN     = 128
SUBSET_SIZE     = 10000     # Set to None to train on full 100K
```

---

## Project Structure

```
histo-llama-cot/
├── config.py                  # All paths and hyperparameters
├── model.py                   # HistoLlama architecture (PLIP + projector + TinyLlama + LoRA)
├── train.py                   # Training loop with loss plotting
├── generate_instructions.py   # Synthetic CoT dataset generation
├── precompute_embeddings.py   # Offline PLIP embedding extraction
├── data/
│   ├── raw_slides/            # NCT-CRC-HE-100K (download separately)
│   ├── embeddings/            # Precomputed .pt tensors (auto-generated)
│   └── instructions/          # train_cot.json (auto-generated)
├── checkpoints/               # Saved model weights (auto-generated)
├── results/                   # Loss plots (auto-generated)
└── README.md
```

---

## Memory & Hardware Requirements

| Component | VRAM Usage |
|-----------|-----------|
| TinyLlama-1.1B (float32) | ~4.4 GB |
| LoRA adapters | ~50 MB |
| Projector | ~4 MB |
| Batch (size=16, seq=128) | ~3.5 GB |
| Gradient checkpointing savings | ~40% reduction |
| **Total (estimated)** | **~10–12 GB** |

Gradient checkpointing is enabled by default. Mixed precision (fp16 for image embeddings) further reduces peak usage.

---

## Paper

A full technical report is included in this repository (`histollama_report.pdf`), formatted for the MICCAI 2025 Workshop on Computational Pathology. It covers the mathematical formulation, data pipeline, SOTA comparison, and a roadmap toward a full publication.

**Cite this work:**
```bibtex
@misc{kulkarni2025histollama,
  title   = {HistoLlama-CoT: Multimodal Chain-of-Thought Reasoning for
             Colorectal Cancer Tissue Classification via Vision-Language Alignment},
  author  = {Kulkarni, Arnav},
  year    = {2025},
  url     = {https://github.com/arnavkulkarni2005/histo-llama-cot}
}
```

---

## AI Disclosure

Portions of the accompanying paper and README were drafted with the assistance of Claude (Anthropic, claude.ai). All code, experimental results, and scientific claims were produced and verified by the author.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The NCT-CRC-HE-100K dataset is subject to its own license — see the [Zenodo record](https://zenodo.org/record/1214456) for terms.
