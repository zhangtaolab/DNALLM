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
    --num-layers 48 \
    --hidden-size 1024 \
    --mamba-state-dim 128 \
    --mamba-head-dim 64 \
    --mamba-num-groups 1 \
    --model-type-hf 'mamba2'

# Download fine-tuning data
git clone https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters
```

After conversion and data processing, start fine-tuning
```bash
python -m torch.distributed.launch scripts/finetune_mamba2_megatron.py \
    --load plant-dnamamba2-BPE_mg \
    --save plant-dnamamba2-BPE_promoter \
    --tokenizer-name-or-path plant-dnamamba2-BPE \
    --train_csv plant-multi-species-core-promoters/train.csv \
    --test_csv plant-multi-species-core-promoters/test.csv \
    --dev_csv plant-multi-species-core-promoters/dev.csv \
    --num_labels 2 \
    --problem_type single_label_classification \
    --micro-batch-size 12 \
    --global-batch-size 12 \
    --epochs 3 \
    --lr 2e-5 \
    --no-enable-hf2mg-convert
```

To use the fine-tuned model, run the following script:
```bash
python -m torch.distributed.launch scripts/infer_mamba2_megatron.py \
    --load plant-dnamamba2-BPE_promoter \
    --tokenizer-name-or-path plant-dnamamba2-BPE \
    --input-file plant-multi-species-core-promoters/test.csv \
    --output-file inference_test.csv \
    --num_labels 2 \
    --problem_type single_label_classification \
    --micro-batch-size 16 \
    --global-batch-size 16
```

Output contents are looked like this:
```text
id,text,sp,label,probabilities,predictions
0,TTGTCGAACCATTGAATCATAGCCGAACCGATGAGGAAGATGATCAAAATCATAAAATTACGAGTCGTGAGATACACAAACTATGTGGAGTAGACCATGATAGTTTGGTCAAAAAAAGTAGACCATGATAGCCACGCCGAAACGGGATGGACCCGAGAGACCATTAATCTAAGCGTCGTTGCATCTACCGTCAGGCGCCGCCATAAAAAACACACAAAAACATTAAAAAAAAGGTACTAAAACGACGTCAGATGTTGATCCGTGGTTACTCAGCTCCTGATCGCATACGTTTTTTTTTTT,bd30,1,"[0.0038909912109375, 0.99609375]",1.0
1,ATCTTGCGACACATGTATAGAACATTATAGCAAAAACTAATTACACAGTTTATCTGTAAATCATGAGACGAATCTTTTAAGCCTAATTACTTCATGATTGAACAATATTTGTTAAATAAAAATAAGAATGCTACTGTGCACAAAAATTTTTCGTGCAGGTACTAAACAAGGCCAGCGCAAATGGCCTATACTTGCTCATAAAGGATGCTTCAAGTAGGAGTACCGTACTATACAGTTAGTACAGTAGTAGTGGTATAGATGGCCATGCAGCCCGAGGCACGACGGCCCGGCCCACGGTAC,broomcorn,0,"[0.99609375, 0.005645751953125]",0.0
2,TCATGTACATCCGTATACAGTTGATAATGCAATTTTTAAAAAGTCTTATATTTAGAAACAGAGGAAGTGATATTTATTGTTGGCAAGGACTAATATAGTTTTTCTTAACAACAAGTATTCTTCTTTTGAAATTACTTGTCATAAAAACAAATATAAATGGATGTATCTAAACTAAAATATACTTCCATAATATATGTCTTTTTTAGAGATTTCACTAAATGGCTACATACGGATGTATATAGATATATTTTAAAGTATAGATTCATTTATTTTGTTCCGTATGTAGTCCCCTAGTAAAAT,barley,0,"[1.0, 0.00014400482177734375]",0.0
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