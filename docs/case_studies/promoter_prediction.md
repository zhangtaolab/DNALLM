# Case Study: Promoter Prediction with DNALLM

This case study demonstrates how to use DNALLM to train a model for promoter prediction, a fundamental task in genomics. We will frame this as a binary classification problem: distinguishing promoter sequences from non-promoter sequences.

## 1. Case Background

A promoter is a region of DNA that initiates the transcription of a particular gene. Identifying promoters is crucial for understanding gene regulation and function. In this example, we will fine-tune a pre-trained DNA foundation model to classify sequences as either "promoter" (positive class) or "non-promoter" (negative class).

## 2. Code

This section provides a complete Python script to perform fine-tuning and inference for promoter prediction. The workflow consists of:
1.  Loading a configuration file.
2.  Loading a pre-trained model and tokenizer (e.g., Plant-DNABERT).
3.  Loading and processing a dataset of DNA sequences with binary labels.
4.  Initializing and running the `DNATrainer` to fine-tune the model.
5.  Running inference on the test set to evaluate performance.

### Setup

First, ensure you have a YAML configuration file (`finetune_config.yaml`) and your dataset.

**`finetune_config.yaml`:**
```yaml
# task configuration
task:
  task_type: "sequence_classification"
  num_labels: 2
  label_map:
    0: "non-promoter"
    1: "promoter"

# training configuration
training:
  output_dir: "./outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  warmup_steps: 500
  weight_decay: 0.01
  logging_dir: "./logs"
  logging_steps: 10
  evaluation_strategy: "steps"
  save_steps: 100
  eval_steps: 100
  load_best_model_at_end: True
  metric_for_best_model: "f1"
  greater_is_better: True
```

**Dataset:**
The model expects a dataset (e.g., from Hugging Face, ModelScope, or a local CSV/TSV file) with at least two columns: one for the DNA sequences and one for the corresponding labels (0 or 1).

### Python Script

```python title="finetune_binary.py"
import os
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

# --- 1. Load Configuration ---
# Load settings from the YAML file
configs = load_config("./finetune_config.yaml")

# --- 2. Load Model and Tokenizer ---
# Specify the pre-trained model to use. You can choose from ModelScope or Hugging Face.
# For this example, we use a plant-specific BERT model.
model_name = "zhangtaolab/plant-dnabert-BPE" 

# Load the model for sequence classification and its corresponding tokenizer
# The `task_config` provides the model with the number of labels.
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs['task'], 
    source="modelscope" # or "huggingface"
)

# --- 3. Load and Prepare Dataset ---
# Load a dataset from ModelScope Hub. 
# Replace with your dataset if needed.
data_name = "zhangtaolab/plant-multi-species-core-promoters"

# Create a DNADataset object
# `seq_col` and `label_col` specify the column names for sequences and labels.
datasets = DNADataset.from_modelscope(
    data_name, 
    seq_col="sequence", 
    label_col="label", 
    tokenizer=tokenizer, 
    max_length=512
)

# Tokenize the sequences
datasets.encode_sequences()

# For demonstration purposes, we'll use a small sample of the data.
# Remove the next line to train on the full dataset.
sampled_datasets = datasets.sampling(0.05, overwrite=True)

# --- 4. Fine-tune the Model ---
# Initialize the trainer with the model, configs, and datasets
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets
)

# Start the training process
print("Starting model fine-tuning...")
train_metrics = trainer.train()
print("Training finished.")
print("Training Metrics:", train_metrics)

# --- 5. Evaluate the Model ---
# Run inference on the test set to get performance metrics
print("Evaluating model on the test set...")
test_metrics = trainer.infer()
print("Evaluation finished.")
print("Test Metrics:", test_metrics)

```

## 3. Expected Results

After running the script, the training process will output logs showing the loss at each step. Upon completion, the `train_metrics` and `test_metrics` dictionaries will be printed.

The `test_metrics` dictionary will contain key performance indicators for binary classification, such as:
-   **`test_accuracy`**: The proportion of correctly classified sequences.
-   **`test_precision`**: The ability of the model to avoid false positives.
-   **`test_recall`**: The ability of the model to find all true positive sequences.
-   **`test_f1`**: The harmonic mean of precision and recall, providing a single score to balance them.
-   **`test_AUROC`**: The Area Under the Receiver Operating Characteristic Curve, which measures the model's ability to distinguish between classes.
-   **`test_AUPRC`**: The Area Under the Precision-Recall Curve, which is especially useful for imbalanced datasets.

Example output:
```
{
    'test_loss': 0.547,
    'test_accuracy': 0.740,
    'test_precision': 0.721,
    'test_recall': 0.808,
    'test_f1': 0.762,
    'test_mcc': 0.482,
    'test_AUROC': 0.821,
    'test_AUPRC': 0.810,
    ...
}
```
The fine-tuned model and training checkpoints will be saved in the directory specified by `output_dir` in the config file.

## 4. Tuning Strategies

To improve model performance, consider the following strategies:

-   **Learning Rate**: The default learning rate is `5e-5`. If the model is not converging, you can try adjusting it in the `training` section of the config file (e.g., `learning_rate: 3e-5`). A good starting point for fine-tuning is often between `1e-5` and `5e-5`.
-   **Batch Size**: `per_device_train_batch_size` can be increased if you have more GPU memory. Larger batch sizes can lead to more stable training.
-   **Epochs**: The `num_train_epochs` determines how many times the model sees the entire training dataset. If the model is underfitting, increase the number of epochs. If it is overfitting (validation loss increases), consider reducing it or using early stopping.
-   **Model Choice**: DNALLM supports various models. A larger or more domain-specific model (e.g., one pre-trained on plant genomes) might yield better results for this task.
-   **Max Sequence Length**: Adjust `max_length` during dataset loading based on the typical length of your promoter sequences. Longer sequences require more memory.

## 5. Troubleshooting

-   **`CUDA out of memory`**: This is a common issue. The primary solution is to decrease `per_device_train_batch_size` and `per_device_eval_batch_size` in your `finetune_config.yaml`. You can also reduce `max_length` if your sequences are very long.
-   **Slow Training**: Training large models on large datasets takes time. To speed up development, use the `.sampling()` method on your dataset to test your pipeline on a smaller subset first. For actual training, using a GPU is highly recommended.
-   **Low Performance (Low F1/Accuracy)**:
    -   Ensure your data is clean and correctly labeled.
    -   Try tuning the hyperparameters mentioned above (learning rate, epochs, etc.).
    -   Consider if the pre-trained model is a good fit for your specific type of data (e.g., human, plant, bacteria). You may need a different base model.
    -   Check for severe class imbalance in your dataset. If one class is much rarer than the other, metrics like `AUPRC` are more informative than `accuracy`. You may need to use techniques like over-sampling or using class weights.
