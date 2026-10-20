---
title: Getting Started
---

<!-- generated-by: gsd-doc-writer -->

# Getting Started

The 15-minute path from a clean machine to a fine-tuned DNA language model: install, load a pre-trained model, run one inference, run one fine-tune. Everything below runs on CPU; add a GPU extra when you want speed (one command, linked below — not duplicated here).

For the complete install matrix (CUDA lanes, NPU/XPU, Mamba, plain-pip GPU quirks) see the [Installation Guide](../getting_started/installation.md); for runnable examples of every workflow see the [Quick Start](../getting_started/quick_start.md).

## Prerequisites

- **Python >= 3.11** (3.12 recommended)
- **pip** — enough for the PyPI install lane; or **uv** — recommended, used by the project itself
- **Git** — only for the from-source install lane
- **GPU** — optional. CPU works for every step in this guide
- Platforms: Linux, Windows, macOS; accelerators: NVIDIA CUDA, Apple MPS, Intel XPU, Huawei Ascend NPU

## Step 1 — Install

### Option A: pip (released package)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install dnallm
```

### Option B: uv from source (recommended — unlocks all extras)

```bash
git clone https://github.com/zhangtaolab/DNALLM.git
cd DNALLM
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install uv
uv pip install -e '.[full]'        # dev + test + notebook + mcp extras
```

### GPU (optional, one command)

After either lane, add a hardware extra with uv:

```bash
uv pip install -e '.[full,cuda130]'   # CUDA 13.0, driver >= 580
```

Pick your lane by GPU generation in the [GPU Support Matrix](../getting_started/gpu_support.md). Note: the CUDA extras' wheel-index pins are uv-only — under plain `pip` a GPU torch build must be installed last from the PyTorch index; see [GPU Support with plain pip](../getting_started/installation.md#gpu-support-with-plain-pip) and the [Install Honesty Matrix](../getting_started/install_honesty.md).

### Verify

```bash
python -c "import dnallm; print(dnallm.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

More verification commands (core modules, CUDA, Mamba, XPU): [Installation — Verification](../getting_started/installation.md#verification).

## Step 2 — Load a model, run one inference

DNALLM is configuration-driven: `load_config` parses a YAML file, `load_model_and_tokenizer` fetches a pre-trained model from Hugging Face or ModelScope (first run downloads the weights), and `DNAInference` runs the prediction.

Save this as `inference_config.yaml` (repo users: the identical file already exists at `./example/notebooks/inference/inference_config.yaml`):

```yaml
task:
  task_type: binary
  num_labels: 2
  threshold: 0.5
inference:
  batch_size: 16
  device: auto        # cpu / cuda / mps / auto
  max_length: 512
  num_workers: 4
  output_dir: ./results
  use_fp16: false
```

Then:

```python
from dnallm import load_config, load_model_and_tokenizer, DNAInference

configs = load_config("inference_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnagpt-BPE-promoter",  # 130+ models available
    task_config=configs["task"],
    source="huggingface",  # or "modelscope" / "local"
)

engine = DNAInference(config=configs, model=model, tokenizer=tokenizer)

sequence = (
    "TCACATCCGGGTGAAACCTCGAGTTCCTATAACCTGCCGACAGGTGGCGGGTCTTATAAAACTGATCACTACAATTCCCAATGGAAAAA"
)
result = engine.infer(sequence)
print(result)
```

The result is a dict keyed by sequence index — `label` is the predicted class and `scores` the per-class probabilities (a binary task without explicit `label_names` defaults to `negative` / `positive`):

```python
{
    0: {
        "sequence": "TCACATCCG...",
        "label": "positive",
        "scores": {"negative": 0.03, "positive": 0.97},
    }
}
```

## Step 3 — Fine-tune a model once

Run this from the repository root — it uses the repo's example config and a tiny bundled CSV (`sequence,label` columns). Swap in your own CSV and model when ready.

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

configs = load_config("./example/notebooks/finetune_binary/finetune_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=configs["task"],
    source="modelscope",  # or "huggingface"
)

dataset = DNADataset.load_local_data(
    file_paths="./tests/test_data/binary_classification/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)
dataset.encode_sequences()

trainer = DNATrainer(model=model, config=configs, datasets=dataset)
trainer.train()  # checkpoints and logs land in ./outputs
```

The config's `finetune` section is a Hugging Face `TrainingArguments` subset (3 epochs, batch size 16, lr 2e-5 by default in the example). Pip-only installs (no repo clone) can fetch the example configs from the [GitHub repo](https://github.com/zhangtaolab/DNALLM) or generate one interactively with `dnallm-model-config-generator`. Every YAML section is documented in the [Configuration Reference](../configuration/configuration.md).

## Common setup issues

- **`torch.cuda.is_available()` is False after a plain `pip install`** — you most likely have the CPU wheel. On Windows the default PyPI torch wheel is CPU-only, and `pip` ignores the CUDA index pins in `pyproject.toml`. Fix: [GPU Support with plain pip](../getting_started/installation.md#gpu-support-with-plain-pip).
- **Resolver conflict when combining hardware extras** — `cpu`, `cuda124`–`cuda130` are mutually exclusive by design; pick exactly one (`mamba` is the only hardware extra that combines with a CUDA group). See [Dependency Groups](../getting_started/installation.md#dependency-groups).
- **Mamba extras fail to build** — `mamba-ssm` compiles from source and needs `nvcc` matching your PyTorch CUDA version. See [Mamba Installation Failure](../getting_started/installation.md#mamba-installation-failure).
- **First model load is slow / needs network** — weights download from Hugging Face (or ModelScope via `source="modelscope"`) on first use and are cached afterwards.

Full troubleshooting list: [Installation — Troubleshooting](../getting_started/installation.md#troubleshooting) and the [FAQ](../faq/index.md).

## Where to go next

- [Quick Start](../getting_started/quick_start.md) — the full tour: mutagenesis, benchmarking, notebooks, CLI
- [Configuration Reference](../configuration/configuration.md) — every YAML section (`task`, `finetune`, `lora`, `ia3`, `inference`, `vep`, `sweep`, `benchmark`)
- [Architecture Overview](../architecture/overview.md) — how a config flows through model loading, training, and inference
- Task guides: [Fine-tuning](../user_guide/fine_tuning/getting_started.md) · [Inference](../user_guide/inference/getting_started.md) · [CLI](../user_guide/cli/usage.md)
- Serve predictions to AI assistants: `dnallm-mcp-server --transport streamable-http` — [MCP Server Guide](../user_guide/mcp/startserver.md)
- Contributing code: [CONTRIBUTING.md](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md) · Full docs site: https://zhangtaolab.github.io/DNALLM/
