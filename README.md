# GPT-2 Inference Engine

A from-scratch C++ inference engine for [OpenAI GPT-2](https://github.com/openai/gpt-2) (small, 124M parameters). Weights are exported from Hugging Face as flat text files and loaded at runtime — no PyTorch or CUDA required for inference.

This project is a learning-oriented implementation: matrix ops, layer norm, multi-head attention, MLP blocks, and a full transformer layer are written in plain C++ with `std::vector`.

## Overview

```
Hugging Face GPT-2  →  modelLoader (Python)  →  weights/*.txt  →  inference (C++)
```

| Component | Role |
|-----------|------|
| `modelLoader/` | Downloads `gpt2` via `transformers`, flattens each parameter to a space-separated text file |
| `weights/` | Exported model weights (one file per tensor) |
| `inference/` | C++ forward pass: attention, MLP, residual connections, weight loading |

## Model Spec (GPT-2 Small)

Defined in `inference/constants.hh`:

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 50,257 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Transformer layers | 12 |
| Context length | 1,024 |
| MLP hidden size | 3,072 |

Activation: GELU (GPT-2 "New GELU" approximation).

## Project Structure

```
gpt2InferenceEngine/
├── modelLoader/
│   ├── main.py          # Export weights from Hugging Face
│   ├── pyproject.toml
│   └── uv.lock
├── inference/
│   ├── constants.hh     # Model hyperparameters
│   └── main.cc          # Core ops, transformer block, weight loading
├── weights/             # Flattened weight files (generated)
└── README.md
```

Each weight file is named after the Hugging Face parameter key, e.g. `transformer.h.0.attn.c_attn.weight.txt`. Values are space-separated floats on a single line.

## Prerequisites

**Weight export (Python)**

- Python 3.14+ (see `modelLoader/.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

**Inference (C++)**

- A C++17 compiler (g++ or clang++)
- Standard library only — no external C++ dependencies

## Exporting Weights

From the repo root:

```bash
cd modelLoader
uv sync          # install torch + transformers
uv run main.py   # writes files to ../weights/
```

Or with pip:

```bash
cd modelLoader
pip install torch transformers
python main.py
```

The script loads `GPT2LMHeadModel.from_pretrained("gpt2")` and writes one `.txt` file per parameter into `weights/`. Re-run this step if you need to refresh weights or export a different checkpoint.

## Building Inference

There is no build system yet. Compile manually from the `inference/` directory (weights are loaded via relative paths):

```bash
cd inference
g++ -std=c++17 -O2 -o gpt2 main.cc
./gpt2
```

Run from `inference/` so the `../weights/` paths resolve correctly.

> **Note:** `main.cc` includes `global.hh`, which is not yet in the repo. Add an empty placeholder (`#pragma once`) until shared declarations land there. There is also no `main()` entry point yet — the binary will not run end-to-end until the full pipeline is wired up.

## Implementation Details

### Core operations (`inference/main.cc`)

- **LayerNorm** — mean/variance normalization with learned scale and shift
- **Multi-head attention** — Q/K/V projections, scaled dot-product attention, output projection
- **MLP** — two linear layers with GELU on the hidden layer
- **Transformer block** — pre-norm attention sub-layer, residual add, pre-norm MLP sub-layer, residual add

### Weight loading

`LoadTransformerWeights()` reads per-layer files from `weights/`:

- Layer norm: `ln_1`, `ln_2` (weight + bias)
- Attention: `attn.c_attn` (combined Q/K/V weight), `attn.c_proj` (output projection)
- MLP: `mlp.c_fc`, `mlp.c_proj` (weight + bias)

Embedding (`transformer.wte`, `transformer.wpe`), final layer norm (`transformer.ln_f`), and language-model head weights are also present in `weights/` for the full forward pass.

## Status

This repo is under active development. Implemented so far:

- [x] Python weight exporter
- [x] Core math utilities (matmul, softmax, GELU, layer norm)
- [x] Single transformer block forward pass
- [x] Per-layer weight file loading (partial)
- [ ] Attention Q/K/V bias handling
- [ ] Full 12-layer stack (`GPT()`)
- [ ] Token embedding + positional encoding
- [ ] LM head and token sampling
- [ ] Build system (CMake/Makefile)
- [ ] End-to-end text generation

## Contributing

Issues and pull requests are welcome. Useful next steps include finishing weight loading, wiring up the full model graph, adding a minimal tokenizer, and validating outputs against Hugging Face reference runs.
