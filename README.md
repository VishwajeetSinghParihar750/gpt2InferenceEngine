# GPT-2 Inference Engine

A from-scratch GPT-2 (small, 124M) inference engine — no PyTorch, no TensorFlow.
The model runs on **CUDA**, with the transformer ops written as custom GPU kernels.

You type a prompt. It generates text.

```
Enter text: The quick brown fox
 jumped over the lazy dog and ran into the forest,
```

---

## What you need

| Tool | Why |
|------|-----|
| **NVIDIA GPU + drivers** | Runs inference on CUDA |
| **nvcc** (CUDA toolkit) | Compiles `main.cu` |
| **Python 3.12+** | One-time weight export from Hugging Face |
| **uv** (optional) | Easier Python deps — or use pip |

---

## Quick start

### 1. Export weights (once)

```bash
cd modelLoader
uv sync
uv run main.py
```

This downloads `gpt2` from Hugging Face and writes one `.txt` file per tensor into `../weights/`.

<details>
<summary>Using pip instead of uv</summary>

```bash
cd modelLoader
pip install torch transformers
python main.py
```

</details>

### 2. Build

```bash
cd inference
nvcc -std=c++20 -O3 --extended-lambda --expt-relaxed-constexpr -o gpt2 main.cu
```

### 3. Run

Run from the `inference/` folder so `../weights/` resolves correctly:

```bash
./gpt2
```

```
loading weights ...
weights loaded ...
loading vocab ...
vocab loaded...

Enter text: Once upon a time
```

It generates 20 tokens per prompt, then asks for another.

---

## How it works

```
Hugging Face GPT-2
      ↓  (Python, once)
modelLoader/main.py
      ↓
weights/*.txt          ← one file per parameter
      ↓  (CUDA, every run)
inference/main.cu + model.cuh + ops.cuh
      ↓
interactive text generation
```

Pipeline, all hand-written:

1. **BPE tokenizer** — GPT-2 merge rules from `tokenizer.json`
2. **Embeddings** — token (`wte`) + position (`wpe`)
3. **12 transformer blocks** — pre-norm attention + GELU MLP (on GPU)
4. **LM head** — final layer norm, project to vocab, greedy next token
5. **Loop** — append token and generate the next one

---

## Project layout

```
gpt2InferenceEngine/
├── inference/
│   ├── main.cu          # CLI loop
│   ├── constants.hh     # model hyperparameters
│   ├── buffer.cuh       # CudaBuffer (device memory RAII)
│   ├── ops.cuh          # all CUDA kernels / helpers
│   ├── tokenizer.cuh    # BPE encode / decode
│   ├── model.cuh        # weights, Transformer, Gpt2 generate
│   └── include/
│       └── json.hpp     # tokenizer.json parsing
├── modelLoader/
│   └── main.py          # export HF weights → ../weights/
├── weights/             # generated (gitignored)
└── README.md
```

**Where to edit**

| Change… | Open… |
|---|---|
| Generation / blocks / weight load | `model.cuh` |
| BPE tokenization | `tokenizer.cuh` |
| GPU math kernels | `ops.cuh` |
| Device buffer API | `buffer.cuh` |
| Hyperparameters | `constants.hh` |

---

## Model (GPT-2 small)

| | |
|---|---|
| Vocabulary | 50,257 |
| Embedding dim | 768 |
| Heads | 12 (head dim 64) |
| Layers | 12 |
| Context | 1,024 |
| MLP hidden | 3,072 |
| Activation | GELU (tanh approx) |
| Precision | `double` on GPU |

---

## CUDA ops (`ops.cuh`)

| Helper | Role |
|---|---|
| `MatMul` | Tiled matrix multiply |
| `Transpose` | Matrix transpose |
| `LayerNorm` | Mean/var normalize + scale/shift |
| `SoftMaxInPlace` / `SoftMaxRows` | Stable softmax |
| `causalMask` | Mask future tokens in attention |
| `packHead` | Write one attention head into the concat buffer |
| `ForwardPass` | Linear layer (+ GELU for MLP) |
| `vectorCombine` / `vectorMap` / `vectorReduction` | Elementwise ops and reductions |

Weights use Hugging Face names, e.g. `transformer.h.0.attn.c_attn.weight.txt`. Q/K/V are interleaved in `c_attn` and split when loading.

---

## Notes

- Build and run from `inference/` so paths to `../weights/` and the tokenizer stay correct.
- First launch loads all weight `.txt` files — that can take a minute; generation is separate from load time.
- This is a learning / from-scratch engine, not a production serving stack.
