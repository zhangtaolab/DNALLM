# Guide to Mamba and State-Space Models (SSMs)

This guide provides a detailed walkthrough for using models based on the Mamba architecture and other State-Space Models (SSMs) like Caduceus within the DNALLM framework. These models are highly effective for capturing long-range dependencies in DNA sequences while maintaining computational efficiency.

**Related Documents**:
- [Installation Guide](../../getting_started/installation.md)
- [Model Selection Guide](../model_selection.md)

## 1. Introduction to Mamba and SSMs

**Mamba** is a modern sequence modeling architecture based on Structured State-Space Models (SSMs). Unlike traditional Transformers which have quadratic complexity with respect to sequence length, Mamba's complexity scales linearly. This makes it exceptionally well-suited for modeling very long DNA sequences.

**Key Advantages**:
- **Efficiency**: Linear scaling allows for faster processing and lower memory usage on long sequences compared to Transformers.
- **Long-Range Dependencies**: The state-space mechanism is designed to effectively capture relationships between distant parts of a sequence.

**Variants in DNALLM**:
- **Plant DNAMamba**: A Mamba model pre-trained on plant genomes.
- **Caduceus**: A bi-directional model that incorporates S4 layers (a precursor to Mamba), enabling it to model long DNA sequences with single-nucleotide resolution.

## 2. Installation

To use Mamba-based models, you need to install specific dependencies. The native Mamba implementation requires a CUDA-enabled GPU.

### Native Mamba Installation (Recommended for NVIDIA GPUs)

After completing the [base installation](../../getting_started/installation.md), run the following command to install the necessary packages, including `mamba-ssm` and `causal-conv1d`.

```bash
# Activate your virtual environment first
# e.g., source .venv/bin/activate

uv pip install -e '.[mamba]' --no-cache-dir --no-build-isolation
```

If you encounter network or compilation issues, you can use the provided helper script:

```bash
sh scripts/install_mamba.sh
```

### NPU accelerate for Mamba-2 models (for Huawei Ascend NPU)

Native Mamba-series models require NVIDIA CUDA to accelerate the training and inference. Huawei provides a specific framework named [MindSpeed](https://gitcode.com/Ascend/MindSpeed-LLM) to accelerate the training of Mamba-2/3 architechture (Mamba-1 model is not supported currently).

To utilize this function, several packages need to be installed first. Here we provide a tutorial for installing these packages (Offical tutorial see [here](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/training/install_guide.md)).

```bash
# 0. Activate your DNALLM environment
# for conda environment
conda activate dnallm
# for uv or python virtual environment
# source .venv/bin/activate

# 1. Install the MindSpeed-Core library
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout master  # checkout commit from MindSpeed master
pip install -r requirements.txt 
pip install -e .
cd ..

# 2. Install the MindSpeed-LLM and Nvidia Megatron-LM library (for LLM training)
git clone https://gitcode.com/ascend/MindSpeed-LLM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout master
pip install -r requirements.txt
cd ..
```

To fine-tuning a Mamba-2 model, pretrained model weight should be converted from huggingface format to Megatron-Mcore format first (official tutorial see [here](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/training/quick_start.md)).
```bash
# Activate the CANN dependencies
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Download mamba-2-based model
git clone https://huggingface.co/zhangtaolab/plant-dnamamba2-BPE

# Convert weight format (the model parameters need to be confirmed by users manually)
python MindSpeed-LLM/convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --load-dir plant-dnamamba2-BPE \
    --save-dir plant-dnamamba2-BPE_mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --hidden-size 1024 \
    --mamba-state-dim 128 \
    --mamba-head-dim 64 \
    --mamba-num-groups 1 \
    --model-type-hf 'mamba2'
```

After conversion and data processing, start fine-tuning
```bash
#!/bin/bash

# ================= Environment =================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2

# ================= Hardware for single NPU =================
NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=1

# ================= model and data path =================
# Recommend absolute path
CKPT_LOAD_DIR="/root/autodl-tmp/mamba2_130M_spv3"
CKPT_SAVE_DIR="/root/autodl-tmp/mamba2_130M_spv3_finetuned_V2"
TOKENIZER_PATH="/root/autodl-tmp/mamba2_130M_spv3"
TRAIN_CSV="/root/autodl-tmp/datasets/train.csv"
DEV_CSV="/root/autodl-tmp/datasets/dev.csv"
TEST_CSV="/root/autodl-tmp/datasets/test.csv"

# ================= Hyper-parameters =================
TP=1
PP=1
MBS=32
GBS=32

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ================= Mamba2 parameters =================
MAMBA_ARGS="
    --spec mindspeed_llm.tasks.models.spec.mamba_spec layer_spec \
    --reuse-fp32-param \
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 24 \
    --num_labels 3 \
    --problem_type single_label_classification \
    --group-query-attention \
    --num-query-groups 24 \
    --mamba-num-groups 1 \
    --mamba-chunk-size 256 \
    --mamba-state-dim 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-head-dim 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --hidden-size 768 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-attention-heads 24 \
    --make-vocab-size-divisible-by 1 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type none \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --no-save-rng \
    --no-save-optim
"

CKPT_ARGS="
    --enable-hf2mg-convert \
    --model-type-hf mamba2
"

EXTRA_ARGS="
    --train_csv $TRAIN_CSV \
    --dev_csv $DEV_CSV \
    --test_csv $TEST_CSV \
    --tensorboard_dir ${CKPT_SAVE_DIR}/tensorboard_logs
"

# ================= Run =================
mkdir -p logs
mkdir -p ${CKPT_SAVE_DIR}

python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_mamba2.py \
    $MAMBA_ARGS \
    $OUTPUT_ARGS \
    $CKPT_ARGS \
    $EXTRA_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/finetune_mamba2_130M_single_card.log
```

### Caduceus Models

Caduceus models are built into the DNALLM framework and do not require a separate installation beyond the base dependencies.

## 3. Usage and Application Scenarios

### Using Plant DNAMamba

Plant DNAMamba is a causal language model (CLM), making it ideal for sequence scoring and generation tasks.

**Example: Scoring a sequence with Plant DNAMamba**

This example demonstrates how to perform zero-shot mutation analysis by scoring sequence likelihood.

```python
from dnallm import load_config, Mutagenesis, load_model_and_tokenizer

# 1. Load a configuration for a generation task
configs = load_config("path/to/your/generation_config.yaml")

# 2. Load the Plant DNAMamba model
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnamamba-BPE",
    task_config=configs["task"],
    source="modelscope",
)

# 3. Perform in-silico mutagenesis
mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)
sequence = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA..."  # A long sequence
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# The evaluate() method will use the CLM scoring mechanism
predictions = mut_analyzer.evaluate()

mut_analyzer.plot(predictions, save_path="./results/dnamamba_mut_effects.pdf")
```

### Using Caduceus Models

Caduceus models are bi-directional (MLM-style) and excel at classification tasks, especially on long sequences where standard BERT models might struggle.

**Example: Fine-tuning PlantCAD2 for classification**
```python
from dnallm import (
    load_config,
    load_model_and_tokenizer,
    DNADataset,
    DNATrainer,
)

# 1. Load a config for a classification task
configs = load_config("path/to/your/finetune_config.yaml")

# 2. Load the PlantCAD2 model
# Note: The model ID might be a mirror like 'lgq12697/PlantCAD2-Small-l24-d0768'
model, tokenizer = load_model_and_tokenizer(
    "kuleshov-group/PlantCAD2-Small-l24-d0768",
    task_config=configs["task"],
    source="huggingface",
)

# 3. Load your dataset and initialize the trainer
# ... (code for loading DNADataset)

trainer = DNATrainer(model=model, config=configs, datasets=my_datasets)
trainer.train()
```

## 4. Troubleshooting

### Problem: `ImportError: No module named 'mamba_ssm'` or `causal_conv1d`
- **Solution**: You have not installed the Mamba-specific dependencies. Please run `uv pip install -e '.[mamba]'` as described in the installation section.

### Problem: Compilation errors during Mamba installation.
- **Cause**: The native Mamba packages require a C++ compiler and the CUDA toolkit to be properly installed and configured on your system.
- **Solution**:
    1. Ensure you have `gxx` and `clang` installed. On conda environments, you can run `conda install -c conda-forge gxx clang`.
    2. Verify that your NVIDIA driver version and CUDA toolkit version are compatible with the PyTorch and Mamba versions being installed.
    3. If issues persist, try using the `sh scripts/install_mamba.sh` script, which can help resolve some common path and environment issues.