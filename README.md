[![Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](TinyStoriesSLM.ipynb)
[![PyPI](https://img.shields.io/pypi/v/torch.svg)](https://pypi.org/project/torch/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

# Building a Small Language Model (SLM) from Scratch

A comprehensive collection of Jupyter Notebooks demonstrating how to build and train compact small language models ("SLMs") from scratch. This repository covers experiments with both English (TinyStories) and Arabic (AraStories, Arabic Stories Corpus) datasets, including data preparation, BPE tokenization, efficient binary storage, GPU memory locking, Transformer architecture, training configuration, and sample text generation.

---

## 🚀 Highlights

- **End-to-end pipeline**  
  From raw text to a fully trained model—all within organized notebooks.
- **Multiple Datasets**  
  Experiments with TinyStories (English), AraStories (Arabic synthetic), and Arabic Stories Corpus.
- **Efficient Tokenization**  
  Uses `tiktoken` (GPT-2 BPE), custom byte-level BPE, and AraGPT2 tokenizers.
- **Disk-backed Dataset**  
  Saves token IDs in `.bin` files for fast reloads.
- **Memory Locking**  
  Demonstrates reserving GPU memory to avoid fragmentation.
- **Custom Transformer**  
  Minimal PyTorch GPT-style decoder models with multi-head attention and feed-forward blocks.
- **Model Scaling Experiments**  
  Enhanced models with increased depth, width, and context length.
- **Cross-lingual Translation Workflows**  
  Experiments with dataset translation and inference-time translation approaches.

---

## 📖 Table of Contents

1. [Introduction](#introduction)  
2. [Project Structure](#project-structure)  
3. [Datasets](#datasets)  
4. [Tokenization & Binarization](#tokenization--binarization)  
5. [Model Architectures](#model-architectures)  
6. [Training Scheme](#training-scheme)  
7. [Experimental Results](#experimental-results)  
8. [Prerequisites](#prerequisites)  
9. [Setup & Installation](#setup--installation)  
10. [Notebook Walk-through](#notebook-walk-through)  
11. [Training Configuration](#training-configuration)  
12. [Sample Generation](#sample-generation)  
13. [Important Notes](#important-notes)  
14. [Contributing](#contributing)  
15. [References](#references)  
16. [License](#license)

---

## Introduction

Building Large Language Models (LLMs) from scratch can be resource-intensive. This repository shows how to create **Small Language Models (SLMs)** using lightweight datasets, minimalist code, and standard hardware (e.g., a single GPU).

The project includes three experimental workflows:

| Workflow | Description |
|----------|-------------|
| **Normal Workflow** | Baseline approach using native or synthetic Arabic datasets (AraStories, arbml) for direct SLM training |
| **Dataset Translation Workflow** | Pre-training augmentation where English source data is machine-translated to create synthetic Arabic training corpus |
| **Inference Translation Workflow** | Cross-lingual approach using a robust English-trained model with translation layers applied to input prompts and output generations |

---

## Project Structure

### Main Notebooks

| Notebook | Description |
|----------|-------------|
| `TinyStoriesSLM.ipynb` | English TinyStories baseline model |
| `arastories_model_tinystories_specs.ipynb` | AraStories baseline model |
| `arastories_model_tinystories_specs_enhanced.ipynb` | AraStories enhanced model (scaled architecture) |

### Other Experiments (`other_expirements/`)

| Notebook | Description |
|----------|-------------|
| `ArabicTranslatedPortionTinyStories.ipynb` | Dataset translation workflow (TinyStories → Arabic) |
| `TinyStoriesSLM_ArabicTranslationAfterEnglishGeneration.ipynb` | Inference translation workflow |
| `arbmlArabicStoriesCorpusSLM.ipynb` | Arabic Stories Corpus experiment |

---

## Datasets

### Dataset Inventory

| Dataset / Source | Description | Effective Size Used |
|------------------|-------------|---------------------|
| **TinyStories** | Synthetic English children stories | Train: 2,119,719 / Val: 21,990 |
| **AraStories (GitHub)** | Synthetic Arabic stories (MSA + dialects) | Train: 2,696 / Val: 300 |
| **Arabic Stories Corpus** | Public Arabic stories dataset | Train: 539 / Val: 60 |

> **Important Clarification**: Although the AraStories paper describes a large translated dataset (hundreds of thousands of stories), the public AraStories GitHub repository releases only the synthetic subset. All AraStories experiments in this repository use only the synthetic data.

---

## Tokenization & Binarization

| Notebook Group | Tokenizer | Vocabulary | Serialization | Rationale |
|----------------|-----------|------------|---------------|-----------|
| TinyStoriesSLM | GPT-2 BPE via tiktoken | ~50k | Memmapped `.bin` | Efficient I/O and deterministic reuse |
| AraStories (baseline + enhanced) | Byte-level BPE trained to 32k | 32,000 | Token IDs in `.bin` shards | Robust to Arabic orthography |
| Other experiments | aragpt2-base tokenizer | HF tokenizer | Memmapped bins per split | Arabic-centric tokenizer |
| Translation-assisted | N/A | N/A | Text-to-text outputs | Translation applied pre/post training |

---

## Model Architectures

All models are decoder-only GPT-style Transformers trained with next-token prediction. The enhanced AraStories notebook modifies only model capacity, not the training algorithm.

### Model Configurations

| Notebook | Layers | Heads | Embedding | Context | Approx. Params | Notes |
|----------|--------|-------|-----------|---------|----------------|-------|
| TinyStoriesSLM | 6 | 6 | 384 | 128 | ~30.0M | Baseline |
| AraStories baseline | 6 | 6 | 384 | 128 | ~23.0M | Baseline |
| AraStories enhanced | 24 | 16 | 1024 | 512 | ~335.4M | Scaled (depth + width + context) |
| arbmlArabicStoriesCorpus | 12 | 12 | 768 | 256 | - | Larger architecture |

> **Interpretation**: The enhanced AraStories run is a pure capacity-scaling experiment. It increases representational power (depth, width) and long-range conditioning (context length). Observed improvements are attributed to model scaling, given a fixed dataset and training scheme.

---

## Training Scheme

The training scheme is conceptually identical across the main notebooks:

| Component | Approach | Notes for Enhanced Model |
|-----------|----------|--------------------------|
| **Objective** | Autoregressive next-token prediction with cross-entropy loss | Same |
| **Perplexity** | Computed as `ppl = exp(loss)` | Same |
| **Optimizer** | AdamW-style training loop | Larger model requires tighter stability control |
| **Evaluation** | Periodic validation using fixed evaluation iterations | Longer warmup often beneficial |
| **Checkpointing** | Save best model based on validation loss | Same |
| **Early Stopping** | Triggered by sustained degradation in validation loss/perplexity | Prevents overfitting on small synthetic data |
| **Compute Controls** | Gradient accumulation and mixed precision | More critical at ~335M parameters |

---

## Experimental Results

### Main Experiments

| Notebook | Dataset (effective) | Train Loss / PPL | Val Loss / PPL | Notes |
|----------|---------------------|------------------|----------------|-------|
| TinyStoriesSLM | TinyStories 2,119,719 / 21,990 | 5.8987 / 364.57 | 5.8987 / 364.57 | Train and validation closely match due to large dataset |
| AraStories baseline | Synthetic AraStories 2,696 / 300 | 6.4243 / 616.68 | 6.5233 / 680.85 | Trained on publicly released synthetic subset |
| AraStories enhanced | Same split 2,696 / 300 | 3.9565 / 52.28 | 5.3753 / 216.01 | Strong gains from architectural scaling |

**Key Outcome**: AraStories enhanced vs baseline shows validation perplexity drop from **~681 to ~216**.

### Other Experiments

| Notebook | Dataset (effective) | Tokenizer | Translation Usage | Model (L/H/D/C) | Train Loss / PPL | Val Loss / PPL |
|----------|---------------------|-----------|-------------------|-----------------|------------------|----------------|
| ArabicTranslatedPortionTinyStories | TinyStories 0.1% (~2.1k / ~22) | aragpt2-base | Dataset translated before training | 6/6/384/128 | 0.0389 / 1.04 | 0.1005 / 1.11 |
| TinyStoriesSLM_ArabicTranslation | Full TinyStories | GPT-2 BPE | Translate outputs and prompts | 6/6/384/128 | 5.8987 / 364.57 | 5.8987 / 364.57 |
| arbmlArabicStoriesCorpusSLM | Arabic Stories Corpus 539 / 60 | aragpt2-base | None | 12/12/768/256 | 5.3321 / 206.87 | 6.5606 / 706.71 |

---

## Prerequisites

- Python 3.8+  
- GPU with CUDA (highly recommended—experiments run on NVIDIA RTX 3080 Ti)  
- [PyTorch](https://pytorch.org/)  
- [Hugging Face `datasets`](https://github.com/huggingface/datasets)  
- [`tiktoken`](https://github.com/openai/tiktoken)  
- [`transformers`](https://github.com/huggingface/transformers) (for AraGPT2 tokenizer)
- `numpy`, `matplotlib`, `tqdm`

---

## Setup & Installation

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision \
            datasets \
            tiktoken \
            transformers \
            numpy \
            matplotlib \
            tqdm
```

---

## Notebook Walk-through

### Data Loading

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
```

### Tokenization

Implements Byte Pair Encoding via `tiktoken` (for English) or custom byte-level BPE (for Arabic).  
Converts text → token IDs → binary `.bin` files.

### Dataset Storage

Saves training/validation tokens on disk for fast reload.

### Memory Locking

Reserves GPU memory (`torch.cuda.set_per_process_memory_fraction` or similar) to prevent fragmentation.

### Model Definition

Lightweight GPT-style Transformer with configurable layers, heads, and embedding size.

### Training Loop

- **Optimizer**: AdamW  
- **LR Schedulers**: LinearLR, CosineAnnealingLR, SequentialLR  
- Gradient clipping, periodic evaluation, and loss logging.

### Evaluation & Generation

- Samples new stories to verify qualitative performance.  
- Plots loss curves with Matplotlib.

---

## Training Configuration

```python
@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
```

**Example hyperparameters (TinyStoriesSLM):**

| Parameter  | Value  |
|------------|--------|
| block_size | 128    |
| vocab_size | 50,000 |
| n_layer    | 6      |
| n_head     | 6      |
| n_embd     | 384    |
| dropout    | 0.1    |

---

## Sample Generation

After training, run:

```python
prompt = "Once upon a time"
generated = model.generate(prompt, max_new_tokens=100)
print(generated)
```

Expect TinyStories-style outputs (short, coherent sentences).

---

## Important Notes

> **⚠️ Execution Environment**  
> The notebooks in this repository include executed code cell outputs. However, they may not run successfully on Google Colab. All experiments were executed and tested locally using an **NVIDIA RTX 3080 Ti GPU**, which is faster and more reliable than the Colab-provided T4 GPU.
>
> Due to the number of notebooks involved, Colab free GPU credits were frequently exhausted; therefore, all experiments were continued and completed locally without Colab validation.

### Practical Interpretation

- **AraStories**: Main experiments are trained exclusively on the synthetic subset released publicly, not on the large translated dataset described in the paper.
- **Enhanced AraStories gains**: Improvements arise from model-capacity scaling, not data augmentation or training-scheme changes. The training algorithm is kept fixed while scaling the GPT decoder capacity (depth, width, and context), yielding large improvements in perplexity on the synthetic AraStories split.

---

## Contributing

1. Fork this repository  
2. Create a new branch (`git checkout -b feature/xyz`)  
3. Commit your changes (`git commit -m 'Add xyz feature'`)  
4. Push to your branch (`git push origin feature/xyz`)  
5. Open a Pull Request  

All contributions—bug reports, documentation fixes, new features—are welcome!

---

## References

1. engomarwasfy. (2023). [Building-a-Small-Language-Model-SLM](https://github.com/engomarwasfy/Building-a-Small-Language-Model-SLM-). GitHub repository.
2. Li, Y., & Eldan, R. (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://openreview.net/pdf?id=yiPtWSrBrN) OpenReview.
3. ChaitanyaK77. (2023). [Building-a-Small-Language-Model-SLM](https://github.com/ChaitanyaK77/Building-a-Small-Language-Model-SLM-). GitHub repository.
4. Eldan, R. (2023). [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Hugging Face Datasets.
5. El-Shangiti, A. O., Alwajih, F., & Abdul-Mageed, M. (2024). Arabic Automatic Story Generation with Large Language Models. ArabicNLP 2024.
6. UBC-NLP. (2024). [AraStories repository](https://github.com/UBC-NLP/arastories). GitHub.
7. OpenAI. (2023). [tiktoken: Fast BPE tokenization for GPT models](https://github.com/openai/tiktoken). GitHub.
8. AUB MIND Lab. (2023). [aubmindlab/aragpt2-base](https://huggingface.co/aubmindlab/aragpt2-base). Hugging Face Models.
9. Helsinki-NLP. (2023). [opus-mt-en-ar](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar). Hugging Face Models.
10. arbml. (2023). [Arabic Stories Corpus](https://huggingface.co/datasets/arbml/Arabic_Stories_Corpus). Hugging Face Datasets.

---

## License

Released under the MIT License.
