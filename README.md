# Implementing Attention Is All You Need

A from-scratch PyTorch implementation of the original Transformer architecture introduced in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — no pre-built attention or transformer layers, every component built from the ground up.

Demonstrated end-to-end with an English → Bengali machine translation model.

<p align="left">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-unlicensed-lightgrey.svg" alt="License">
</p>

---

## Overview

This repository is a faithful, unmodified reimplementation of the Transformer architecture as described in the original paper — built to deeply understand the architecture that underlies modern LLMs, from the matrix math up rather than from a library call.

The architecture is fully decoupled from the training pipeline: the model (`model.ipynb`) is standalone and reusable, and the training notebook (`train.ipynb`) simply imports it. This means **you can train the architecture on your own dataset** without writing any model code — just point the config at your data and write your training loop, or reuse the one provided.

## Architecture

Every module is implemented from scratch in PyTorch, matching the paper's specification exactly:

- **Input Embeddings** — token embeddings scaled by `√d_model`
- **Positional Encoding** — sinusoidal (sin/cos) positional encoding, non-trainable
- **Multi-Head Attention** — scaled dot-product attention, implemented head-splitting and all
- **Masked Multi-Head Attention** — causal self-attention for the decoder
- **Cross-Attention** — decoder attends over encoder output
- **Position-wise Feed-Forward Network** — `Linear → ReLU → Dropout → Linear`
- **Residual Connections + Layer Normalization** — post-LN, as in the original paper
- **Encoder** — stack of N encoder blocks (self-attention + feed-forward)
- **Decoder** — stack of N decoder blocks (masked self-attention + cross-attention + feed-forward)
- **Output Projection** — linear layer to vocabulary size

No architectural modifications were made — this is the paper's model, not a variant.

## Project Structure

```
AttentionIsAllYouNeed/
├── model.ipynb          # Transformer architecture — standalone, reusable
├── train.ipynb          # Training pipeline (imports model.ipynb)
├── docs/
│   └── en_bn.csv         # English–Bengali sentence pairs (dataset)
├── tokenizers/           # Generated tokenizer files (created on first run)
├── weights/              # Saved model checkpoints (created on first run)
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/unthinkingFool/AttentionIsAllYouNeed.git
cd AttentionIsAllYouNeed
```

### 2. Install dependencies

```bash
pip install torch pandas tokenizers
```

### 3. Run

Open `train.ipynb` and run all cells. It will:
1. Load the dataset and build tokenizers
2. Build the Transformer using the architecture from `model.ipynb`
3. Train the model and save checkpoints to `weights/`
4. Run a few example translations from the validation set

## Dataset

The demo task is English → Bengali translation, using an [English–Bengali parallel corpus from Kaggle](#) *(add your specific dataset link here)*, stored at `docs/en_bn.csv` with `en_text` and `bn_text` columns.

## Configuration

All hyperparameters live in a single `config` dictionary in `train.ipynb`:

| Parameter | Value | Description |
|---|---|---|
| `seq_len` | 64 | Max sequence length |
| `d_model` | 512 | Model / embedding dimension |
| `num_encoder_layers` | 6 | Encoder blocks |
| `num_decoder_layers` | 6 | Decoder blocks |
| `num_heads` | 8 | Attention heads |
| `d_ff` | 2048 | Feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 20 | Training epochs |
| `lr` | 1e-4 | Learning rate |

## Usage: Translate a Custom Sentence

```python
translate(model, "Thank you very much for your help.", tokenizer_src, tokenizer_tgt, config["seq_len"], device)
```

## Roadmap

This version prioritized getting the architecture exactly right — the training pipeline is intentionally minimal. Next up:

- [ ] Hyperparameter tuning
- [ ] Tokenizer & embedding improvements (subword tokenization)
- [ ] Model fine-tuning
- [ ] LLM evaluation
- [ ] Deeper architecture optimization
- [ ] Implementing more foundational papers
- [ ] Agentic AI systems

## Acknowledgements

- [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- Special thanks to **Umar Jamil** for the extraordinary explanation of the paper, which was invaluable wherever I got stuck.

## Author

**Swapnil Das**
[LinkedIn](https://www.linkedin.com/in/swapnil-das-603824236) · [GitHub](https://github.com/unthinkingFool)

## License

This project is currently unlicensed — all rights reserved by default. Open an issue or reach out if you'd like to use this code and need explicit permission.
