# GPT-2 Inference Engine

A from-scratch C++ inference engine for [OpenAI GPT-2](https://github.com/openai/gpt-2) (small, 124M parameters).

No PyTorch. No CUDA. No ML frameworks. Just plain C++ and the standard library.

---

## What it does

You type a prompt. It generates text. That's it.

```
Enter text: The quick brown fox
 jumped over the lazy dog and ran into the forest,
```

Under the hood, every step is implemented by hand:

- **BPE tokenizer** — splits your input into tokens using the GPT-2 merge rules, loaded from `tokenizer.json`
- **Token + positional embeddings** — looks up `wte` and `wpe`, adds them together
- **12 transformer blocks** — each with pre-norm multi-head attention and a GELU MLP
- **LM head** — projects the final hidden state against the vocabulary, picks the most likely next token
- **Autoregressive loop** — appends each generated token and feeds it back in for the next one

---

## How it's built

Weights are exported from Hugging Face as flat `.txt` files (one file per tensor), then loaded at runtime by the C++ binary. No binary model format, no serialization library.

```
Hugging Face GPT-2
      ↓  (Python, once)
modelLoader/main.py
      ↓
weights/*.txt          ← one file per parameter, space-separated floats
      ↓  (C++, every run)
inference/main.cc
      ↓
interactive text generation
```

---

## Project structure

```
gpt2InferenceEngine/
├── inference/
│   ├── main.cc          # everything: ops, tokenizer, transformer, main loop
│   ├── constants.hh     # model hyperparameters
│   ├── include/
│   │   └── json.hpp     # single-header JSON (for tokenizer.json)
│   └── tokenizer/
│       └── tokenizer.json
├── modelLoader/
│   ├── main.py          # exports weights from Hugging Face
│   ├── pyproject.toml
│   └── uv.lock
├── weights/             # generated — one .txt per tensor
└── README.md
```

---

## Model spec (GPT-2 small)

| Hyperparameter    | Value  |
|-------------------|--------|
| Vocabulary size   | 50,257 |
| Embedding dim     | 768    |
| Attention heads   | 12     |
| Head dim          | 64     |
| Transformer layers| 12     |
| Context length    | 1,024  |
| MLP hidden size   | 3,072  |
| Activation        | GELU (new) |

---

## Getting started

### 1. Export weights (once)

You need Python and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
cd modelLoader
uv sync
uv run main.py
```

This downloads `gpt2` from Hugging Face and writes one `.txt` file per parameter into `../weights/`. Takes a minute or two.

<details>
<summary>Using pip instead of uv</summary>

```bash
cd modelLoader
pip install torch transformers
python main.py
```

</details>

### 2. Build the inference engine

Requires a C++20 compiler (g++ or clang++). No other dependencies.

```bash
cd inference
g++ -std=c++20 -O2 -o gpt2 main.cc
```

### 3. Run

```bash
# run from the inference/ directory so ../weights/ resolves correctly
./gpt2
```

You'll see weight and vocab loading messages, then an interactive prompt:

```
loading weights ...
weights loaded ...
loading vocab ...
vocab loaded...

Enter text: Once upon a time
```

It generates 20 tokens per prompt, then loops back for another input.

---

## Implementation notes

All ops live in `inference/main.cc` and use `std::vector<double>` with flat 1D layouts for matrices.

| Function | What it does |
|---|---|
| `LayerNorm` | Mean/variance normalize, then scale + shift |
| `SoftMax` | Numerically stable (max subtraction) |
| `MatMul` | Naive triple loop |
| `Transpose` | In-place reshape for weight files |
| `Attention` | Single-head: Q/K/V projections → scaled dot-product → causal mask → softmax → weighted V |
| `MultiHeadAttention` | Runs 12 heads, concatenates, projects through `c_proj` |
| `MLP` | `c_fc` (768→3072) + GELU → `c_proj` (3072→768) |
| `Transformer` | Pre-norm attn block + residual, pre-norm MLP block + residual |
| `GPT` | Stacks all 12 transformer blocks, final layer norm, dot against `wte` for logits |
| `Tokenize` | Regex chunking + BPE merge loop (GPT-2 rules from `tokenizer.json`) |
| `LoadWeights` | Reads all 12 layers + embeddings + final LN from `../weights/` |
| `LoadVocab` | Parses `tokenizer.json` for vocab and merge table |

Weight files follow the Hugging Face naming convention, e.g. `transformer.h.0.attn.c_attn.weight.txt`. The Q/K/V weights are interleaved in a single `c_attn` file and split at load time.

---

## Prerequisites

| Tool | Version |
|---|---|
| g++ / clang++ | C++20 support |
| Python | 3.12+ |
| uv (optional) | any recent |

No CUDA, no BLAS, no Boost, nothing else.
