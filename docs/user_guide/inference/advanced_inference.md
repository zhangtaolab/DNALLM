# Advanced Inference with DNALLM

This tutorial is designed for users who are already familiar with [Basic Inference](./basic_inference.md) and want to explore the more powerful and flexible features of the `DNAInference` engine. We will cover advanced topics such as batch processing, extracting internal model states, custom inference workflows, and advanced result handling.

## 1. Advanced Features Overview

The `DNAInference` engine is more than just a tool for getting predictions. It provides a suite of advanced capabilities for in-depth model analysis and integration into complex pipelines:

- **Direct Batch Inference**: Process large datasets efficiently using `batch_infer()` for fine-grained control.
- **Model Introspection**: Extract hidden states and attention weights to understand *how* the model makes its decisions.
- **Custom Workflows**: Build tailored inference pipelines by directly using `DNADataset` and `DataLoader`.
- **Sequence Generation & Scoring**: Use generative models like Evo for tasks beyond classification, such as creating new DNA sequences or scoring their likelihood.
- **Advanced Configuration**: Fine-tune every aspect of the inference process for specific hardware and data characteristics.

## 2. Deep Dive into Batch Inference

While `infer()` is a convenient wrapper, `batch_infer()` is the workhorse method that gives you direct access to the model's raw outputs. This is useful when you need to implement custom post-processing or integrate with other ML frameworks.

The `batch_infer()` method returns three items:
1.  `all_logits`: A raw tensor of model outputs before any activation function (like Softmax or Sigmoid) is applied.
2.  `predictions`: A formatted dictionary of predictions (this is `None` if `do_pred=False`).
3.  `embeddings`: A dictionary containing hidden states and/or attention weights if requested.

```python
from dnallm import load_config, load_model_and_tokenizer, DNAInference

configs = load_config("inference_config.yaml")
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnagpt-BPE-promoter", 
    task_config=configs['task'], 
    source="modelscope"
)

inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs
)

# 1. Generate a dataset and dataloader
sequences = ["GATTACA...", "CGCGCGC..."]
_, dataloader = inference_engine.generate_dataset(
    sequences, 
    batch_size=configs['inference'].batch_size
)

# 2. Run batch inference, requesting hidden states and attentions
all_logits, _, embeddings = inference_engine.batch_infer(
    dataloader,
    do_pred=False,  # We'll process logits ourselves
    output_hidden_states=True,
    output_attentions=True
)

# 3. Now you have the raw materials
print("Logits shape:", all_logits.shape)
# >> Logits shape: torch.Size([2, 2])

print("Hidden states available:", "hidden_states" in embeddings)
# >> Hidden states available: True

print("Attention weights available:", "attentions" in embeddings)
# >> Attention weights available: True
```

### When to use `batch_infer()`:
- When you need raw logits for custom analysis (e.g., temperature scaling, ensemble methods).
- When you only need embeddings and don't want the overhead of formatting predictions.
- When integrating into a larger pipeline that has its own post-processing logic.

## 3. Customizing the Inference Workflow

You can gain full control over the data loading and preprocessing pipeline by creating the `DNADataset` and `DataLoader` yourself. This is useful for complex datasets or when you need to apply custom transformations.

```python
from torch.utils.data import DataLoader
from dnallm.datahandling.data import DNADataset

# Assume inference_engine is already initialized

# 1. Load data from a file into a DNADataset object
dna_dataset = DNADataset.load_local_data(
    "data/my_sequences.fasta",
    tokenizer=inference_engine.tokenizer,
    max_length=inference_engine.pred_config.max_length
)

# 2. Apply encoding
# This step tokenizes the sequences and prepares them for the model
dna_dataset.encode_sequences(
    task=inference_engine.task_config.task_type,
    remove_unused_columns=True
)

# 3. Create a custom DataLoader
custom_dataloader = DataLoader(
    dna_dataset,
    batch_size=32,  # Override config batch size
    shuffle=False,  # Inference should not be shuffled
    num_workers=inference_engine.pred_config.num_workers
)

# 4. Run inference with the custom dataloader
logits, predictions, _ = inference_engine.batch_infer(custom_dataloader)

print(f"Processed {len(predictions)} sequences.")
```

## 4. LoRA-Model Inference

The `DNAInference` engine seamlessly supports inference with models fine-tuned using LoRA adapters. This allows you to switch between different "personalities" of a base model without loading a completely new one.

To use a LoRA adapter, simply provide the path or hub ID to the `lora_adapter` argument during initialization.

```python
# 1. Load the base model
base_model_name = "lgq12697/PlantCAD2-Small-l24-d0768"
model, tokenizer = load_model_and_tokenizer(base_model_name, task_config=configs['task'], source="modelscope")

# 2. Specify the LoRA adapter
lora_adapter_id = "lgq12697/cross_species_acr_train_on_arabidopsis_plantcad2_small"

# 3. Initialize the engine with the adapter
# The engine will automatically download and apply the LoRA weights
lora_inference_engine = DNAInference(
    model=model,
    tokenizer=tokenizer,
    config=configs,
    lora_adapter=lora_adapter_id
)

# Now, all inference calls will use the LoRA-adapted model
results = lora_inference_engine.infer(sequences=["GATTACA..."])
print(results)
```

This is extremely powerful for comparing a base model's performance against its fine-tuned variants or for deploying multiple specialized models efficiently.

## 5. Advanced Result Post-Processing

The `DNAInference` engine includes powerful visualization tools for model interpretability, which require `output_attentions=True` or `output_hidden_states=True`.

### Visualizing Attention

`plot_attentions()` helps you see which parts of a sequence the model focused on.

```python
# Run inference first with output_attentions=True
inference_engine.infer(sequences=sequences, output_attentions=True)

# Plot the attention map for the first sequence, last layer, and last head
attention_figure = inference_engine.plot_attentions(
    seq_idx=0,
    layer=-1,
    head=-1,
    save_path="./results/attention_map.png"
)
```

### Visualizing Embeddings

`plot_hidden_states()` uses dimensionality reduction (t-SNE, PCA, UMAP) to visualize the sequence embeddings from each layer, which can reveal how the model separates different classes.

```python
# Run inference first with output_hidden_states=True
inference_engine.infer(file_path="data/labeled_data.csv", output_hidden_states=True, evaluate=True)

# Plot the embeddings using t-SNE
embedding_figure = inference_engine.plot_hidden_states(
    reducer="t-SNE",
    save_path="./results/embedding_plot.png"
)
```

---

Next, learn how to squeeze every drop of performance out of your hardware in the Performance Optimization guide.

---

## Next Steps

- [Performance Optimization](performance_optimization.md) - Optimize inference performance
- [Visualization](visualization.md) - Learn about result visualization
- [Mutagenesis Analysis](mutagenesis_analysis.md) - Analyze mutation effects
- [Inference Troubleshooting](../../faq/inference_troubleshooting.md) - Common inference issues and solutions