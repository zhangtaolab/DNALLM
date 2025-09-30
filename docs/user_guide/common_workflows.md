# Common Workflows in DNALLM

DNALLM is designed to streamline common tasks in computational genomics. This guide covers three primary workflows: fine-tuning a model, performing inference, and benchmarking multiple models.

## 1. Fine-tuning a Model

Fine-tuning adapts a pre-trained language model to a specific downstream task, such as classifying promoter sequences.

### Workflow Steps

1.  **Prepare a Configuration File**: Define the model, dataset, and training parameters in a `.yaml` file.
2.  **Load Data**: Use the `DNADataset` class to load and preprocess your training data.
3.  **Load Model**: Load a pre-trained model and tokenizer.
4.  **Initialize Trainer**: Create a `DNATrainer` instance with your configuration, model, and data.
5.  **Start Training**: Call the `train()` method.

### Example

This example fine-tunes `plant-dnabert-BPE` for a binary classification task.

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.datahandling import DNADataset
from dnallm.finetune import DNATrainer

# 1. Load configuration from a file
configs = load_config("./example/notebooks/finetune_binary/finetune_config.yaml")

# 2. Load model and tokenizer
model_name = "zhangtaolab/plant-dnabert-BPE"
model, tokenizer = load_model_and_tokenizer(
    model_name,
    task_config=configs["task"],
    source="huggingface"
)

# 3. Prepare dataset
dataset = DNADataset.load_local_data(
    file_paths="./tests/test_data/binary_classification/train.csv",
    seq_col="sequence",
    label_col="label",
    tokenizer=tokenizer,
)
dataset.encode_sequences() # Tokenize the sequences

# 4. Initialize the trainer
trainer = DNATrainer(
    config=configs,
    model=model,
    datasets=dataset
)

# 5. Start the fine-tuning process
trainer.train()
```

## 2. *In-silico* Mutagenesis Analysis

This workflow systematically introduces mutations into a sequence and evaluates their impact on the model's prediction, which is useful for identifying important nucleotides.

### Workflow Steps

1.  **Load a Fine-tuned Model**: Use a model that has been trained for a specific task (e.g., predicting promoter strength).
2.  **Initialize `Mutagenesis`**: Create an instance of the `Mutagenesis` analyzer.
3.  **Generate Mutations**: Use `mutate_sequence()` to create all possible single-nucleotide substitutions.
4.  **Evaluate Effects**: Run inference on all mutated sequences.
5.  **Visualize Results**: Plot the mutation effects to create a saliency map.

### Example

```python
from dnallm import load_config, load_model_and_tokenizer
from dnallm.inference import Mutagenesis

# 1. Load configuration and a fine-tuned model
configs = load_config("./example/notebooks/in_silico_mutagenesis/inference_config.yaml")
model_name = "zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast"
model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs["task"])

# 2. Initialize the mutagenesis analyzer
mutagenesis = Mutagenesis(config=configs, model=model, tokenizer=tokenizer)

# 3. Generate and evaluate mutations for a sequence
sequence = "AATATATTTAATCGGTGTATAATTTCTGTGAAGATCCTCGATACTTCATATAAGAGATTTTGAGAGAGAGAGAGAACCAATTTTCGAATGGGTGAGTTGGCAAAGTATTCACTTTTCAGAACATAATTGGGAAACTAGTCACTTTACTATTCAAAATTTGCAAAGTAGTC"
mutagenesis.mutate_sequence(sequence, replace_mut=True)
predictions = mutagenesis.evaluate(strategy="mean")

# 4. Plot and save the results
plot = mutagenesis.plot(predictions, save_path="mutation_effects.pdf")
print("Mutation analysis complete. Plot saved to mutation_effects.pdf")
```

For more workflows, such as benchmarking and embedding extraction, explore the Tutorials section.