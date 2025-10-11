# Sequence Mutation Analysis with DNALLM

This tutorial provides a comprehensive guide to performing *in silico* mutagenesis analysis using the `dnallm.Mutagenesis` class. This powerful tool allows you to systematically introduce mutations into a DNA sequence and evaluate their impact on a model's predictions, providing deep insights into sequence-function relationships and model interpretability.

We will cover:
- **Zero-shot Inference**: Using pre-trained models without fine-tuning.
- **Mutation Analysis with Base Models**: Scoring mutations with Masked Language Models (MLM) and Causal Language Models (CLM).
- **Mutation Analysis with Fine-tuned Models**: Analyzing mutation effects on classification and regression tasks.
- **Specialized Models**: Handling unique models like EVO for scoring.

**Prerequisites**:
- Familiarity with Basic Inference.
- An understanding of Advanced Inference concepts is helpful.

## 1. Zero-shot Inference Overview

Zero-shot inference is the ability of a large language model to perform tasks it was not explicitly trained for. In genomics, this means we can use a base, pre-trained model (like a CLM or MLM) to infer the "fitness" or "naturalness" of a DNA sequence without fine-tuning it on a specific labeled dataset.

The core idea is that a model trained on a vast corpus of genomic data has learned the underlying "grammar" of DNA. Sequences that are more "grammatical" or "likely" according to the model are often more biologically functional. The `Mutagenesis` class leverages this by measuring how much a mutation perturbs a sequence's likelihood score.

## 2. In Silico Mutagenesis Overview

*In silico* mutagenesis is the computational equivalent of saturation mutagenesis in the lab. Instead of physically creating and testing every possible mutation, we generate them computationally and use a trained model to predict their effects.

The `dnallm.Mutagenesis` class automates this process:
1.  **Mutation Generation**: It takes a reference sequence and creates a dataset of mutated sequences, including single nucleotide substitutions, deletions, and insertions.
2.  **Model-based Scoring**: It runs inference on the original and all mutated sequences.
3.  **Effect Calculation**: It compares the prediction scores of mutated sequences to the original to quantify the impact of each mutation.
4.  **Visualization**: It provides tools to plot the results as intuitive heatmaps and effect plots.

## 3. Mutation Analysis with Base Models (CLM & MLM)

Using pre-trained base models is a powerful zero-shot approach to identify functionally important regions in a sequence. The `Mutagenesis` class supports two primary scoring algorithms for this purpose.

### Scoring with Masked Language Models (MLM)

For MLMs (e.g., DNABERT), we use a **Pseudo-Log-Likelihood (PLL)** score. The `mlm_evaluate()` method calculates this by:
1. Iterating through each token in the sequence.
2. Masking one token at a time.
3. Asking the model to predict the original token.
4. Summing the log-probabilities of the correct predictions.

A higher PLL score indicates the sequence is more "expected" by the model. A mutation that causes a large drop in PLL is likely deleterious.

**Usage**:
Set `task_type: "mask"` in your configuration.

```python
from dnallm import load_config, load_model_and_tokenizer, Mutagenesis

# Use a config with task_type: "mask"
configs = load_config("config_mlm.yaml") 

model, tokenizer = load_model_and_tokenizer(
    "InstaDeepAI/nucleotide-transformer-500m-human-ref", 
    task_config=configs['task']
)

mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)

sequence = "GATTACA..." # Your sequence of interest
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# The evaluate() method will automatically use mlm_evaluate()
predictions = mut_analyzer.evaluate()

mut_analyzer.plot(predictions, save_path="./results/mlm_mut_effects.pdf")
```

### Scoring with Causal Language Models (CLM)

For CLMs (e.g., DNAGPT, Evo), we calculate the **log-probability** of the entire sequence. The `clm_evaluate()` method does this by:
1. Processing the sequence token by token.
2. At each position, calculating the log-probability of the correct next token given the preceding context.
3. Summing these log-probabilities.

This score represents how likely the model thinks the sequence is, from start to finish.

**Usage**:
Set `task_type: "generation"` in your configuration.

```python
# Use a config with task_type: "generation"
configs = load_config("config_clm.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnagpt-BPE-promoter", 
    task_config=configs['task']
)

mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)

sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# The evaluate() method will automatically use clm_evaluate()
predictions = mut_analyzer.evaluate()

mut_analyzer.plot(predictions, save_path="./results/clm_mut_effects.pdf")
```

## 4. Mutation Analysis with Fine-tuned Models

When you have a model fine-tuned for a specific task (e.g., predicting promoter strength), you can measure how mutations affect its output. This is a direct way to map sequence positions to functional outcomes.

The `evaluate()` method's `strategy` parameter is key here. It defines how the final "mutation effect score" is calculated from the model's output (which can be multi-dimensional for multi-class or multi-label tasks).

### Score Normalization Strategies

The effect of a mutation is measured as the log2 fold change (`logfc`) between the mutated sequence's score and the original sequence's score. The `strategy` parameter determines which value from the `logfc` array to use as the final score for plotting.

- `strategy="last"` (default): Uses the `logfc` of the last class. Ideal for binary or regression tasks.
- `strategy="first"`: Uses the `logfc` of the first class.
- `strategy="mean"`: Averages the `logfc` across all classes.
- `strategy="sum"`: Sums the `logfc` across all classes.
- `strategy=<int>`: Uses the `logfc` at a specific class index.
- `strategy="max"`: Uses the `logfc` of the class that had the highest raw score in the original prediction.

**Example: Regression Model**

Let's analyze a model that predicts promoter strength (a regression task).

```python
from dnallm import load_config, load_model_and_tokenizer, Mutagenesis

# Config for a regression task
configs = load_config("inference_config.yaml")

model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast",
    task_config=configs['task'],
    source="modelscope"
)

mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)

sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# For regression, 'last' or 'mean' are good choices.
predictions = mut_analyzer.evaluate(strategy="mean")

mut_analyzer.plot(predictions, save_path="./results/finetuned_mut_effects.pdf")
```

## 5. Special Models (e.g., EVO)

The `Mutagenesis` class has built-in support for specialized generative models like **Evo-1** and **Evo-2**. These models have their own optimized `scoring` methods.

When an Evo model is detected, `mutagenesis.evaluate()` automatically calls `inference_engine.scoring()` instead of the standard `batch_infer()`. The `strategy` parameter is passed to the `reduce_method` of the scoring function, typically with `"mean"` or `"sum"` being the most relevant options.

```python
from dnallm import load_config, load_model_and_tokenizer, Mutagenesis

# Config for a generation task
configs = load_config("inference_evo_config.yaml")

# Load an Evo model
model, tokenizer = load_model_and_tokenizer(
    "lgq12697/evo2_1b_base", 
    task_config=configs['task'], 
    source="modelscope"
)

mut_analyzer = Mutagenesis(model=model, tokenizer=tokenizer, config=configs)

sequence = "GATTACAGATTACAGATTACA"
mut_analyzer.mutate_sequence(sequence, replace_mut=True)

# 'mean' or 'sum' are the most effective strategies for Evo models
predictions = mut_analyzer.evaluate(strategy="mean")

mut_analyzer.plot(predictions, save_path="./results/evo_mut_effects.pdf")
```

## 6. Troubleshooting

### Problem: Slow Performance
- **Cause**: Mutagenesis generates many sequences (`len(seq) * 3` for substitutions).
- **Solution**:
    - Ensure you are using a GPU (`device: cuda`).
    - Increase `inference.batch_size` in your config to the largest value that fits in VRAM.
    - For very long sequences, consider analyzing only a specific region of interest by passing a subsequence to `mutate_sequence()`.

### Problem: Out-of-Memory (OOM) Errors
- **Solution**: **Reduce `batch_size`**. This is the most common fix. Start with a small value like 4 or 8 and increase it gradually.

### Problem: Unexpected or Flat Scores
- **Check Model & Task Type**: Ensure the `task_type` in your config matches the model you are using. Using a `regression` config with a base MLM model will not produce meaningful results.
- **Check Sequence Length**: If your sequence is much longer than the model's `max_length`, it will be truncated, and mutations outside the context window will have no effect.
- **Model Sensitivity**: Some models may not be sensitive to single-nucleotide changes. This is an insight in itself! You might need a model with higher resolution or one fine-tuned on a relevant task.