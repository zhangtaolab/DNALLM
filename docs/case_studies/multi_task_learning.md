# Case Study: Multi-Task Learning for DNA Sequences

This case study demonstrates how to train a single DNALLM model to perform multiple classification tasks simultaneously. This approach, known as multi-task learning, can improve model performance and efficiency.

## 1. Case Background

In genomics, a single DNA sequence can often be associated with multiple properties or functions. For example, a regulatory element might be active in different tissues (leaf, root) or under different conditions (stress, normal). Instead of training separate models for each property, we can train one model to predict all of them at once.

Multi-task learning has several benefits:
-   **Improved Generalization**: By learning related tasks together, the model can leverage shared representations, which can lead to better performance on all tasks, especially when data for some tasks is scarce.
-   **Efficiency**: Training and deploying a single model is more computationally and operationally efficient than managing multiple models.

In this example, we will fine-tune a model to predict multiple binary labels for a given DNA sequence. This is framed as a **multi-label classification** problem.

## 2. Code

This section provides a script for fine-tuning a model on a multi-label dataset.

### Setup

First, prepare your dataset and configuration file.

**Dataset:**
The dataset should be a local file (e.g., CSV or TSV) where one column contains the DNA sequences and another column contains the labels. For multi-label tasks, the labels for a single sequence should be consolidated into a single string, separated by a specific character (e.g., a comma).

*Example `multi_label_data.tsv`:*

```
sequence	labels
CACGGTCA...	label1,label3
AGTCGCTA...	label2
GCGATATA...	label1,label2,label4
```

**`multi_labels_config.yaml`:**
In the configuration, you must define the `task_type` as `sequence_classification` and provide a `label_map` that includes all possible labels across all tasks.

```yaml
# task configuration
task:
  task_type: "sequence_classification"
  problem_type: "multi_label_classification" # Specify multi-label problem
  num_labels: 4 # Total number of unique labels
  label_map:
    0: "label1"
    1: "label2"
    2: "label3"
    3: "label4"

# training configuration
training:
  output_dir: "./outputs_multi_task"
  num_train_epochs: 5
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  warmup_steps: 100
  weight_decay: 0.01
  logging_dir: "./logs_multi_task"
  logging_steps: 20
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: True
  metric_for_best_model: "f1_macro"
  greater_is_better: True
```

### Python Script

```python title="finetune_multi_label.py"
from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

# --- 1. Load Configuration ---
configs = load_config("./multi_labels_config.yaml")

# --- 2. Load Model and Tokenizer ---
# We use Plant-DNAGPT as an example. Any sequence classification model can be used.
model_name = "zhangtaolab/plant-dnagpt-BPE"

# The `problem_type` in the config tells the model to handle multi-label outputs.
model, tokenizer = load_model_and_tokenizer(
    model_name, 
    task_config=configs['task'], 
    source="modelscope"
)

# --- 3. Load and Prepare Dataset ---
# Load a local dataset. 
# `multi_label_sep` tells the loader how to split the labels in the 'labels' column.
datasets = DNADataset.load_local_data(
    "./maize_test.tsv", 
    seq_col="sequence", 
    label_col="labels", 
    multi_label_sep=",", 
    tokenizer=tokenizer, 
    max_length=256,
    config=configs
)

# Tokenize the sequences
datasets.encode_sequences()

# Split the data into training, validation, and test sets
datasets.split_data(train_size=0.8, test_size=0.1, validation_size=0.1)

# --- 4. Fine-tune the Model ---
# Initialize the trainer
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=datasets
)

# Start the training process
print("Starting multi-task model fine-tuning...")
trainer.train()
print("Training finished.")

# --- 5. Evaluate the Model ---
# Run inference on the test set
# For multi-label tasks, metrics are typically computed for each label and averaged.
print("Evaluating model on the test set...")
test_metrics = trainer.infer()
print("Test Metrics:", test_metrics)

```

## 3. Expected Results

During training, the model's loss will be a sum of the losses from all individual labels (typically Binary Cross-Entropy with Logits). 

After evaluation, the `test_metrics` dictionary will contain metrics that are averaged across all labels, such as:
-   **`test_f1_macro`**: The F1 score calculated independently for each label and then averaged. It treats all labels equally.
-   **`test_f1_micro`**: The F1 score calculated globally by counting the total true positives, false negatives, and false positives across all labels.
-   **`test_accuracy`**: The subset accuracy, which is a strict metric that considers a prediction correct only if all labels for a given sequence are correctly predicted.

Additionally, the output might include per-label metrics, allowing you to see how well the model performs on each individual task.

## 4. Tuning Strategies

-   **Balancing Tasks**: In multi-task learning, some tasks can dominate others. If your model performs well on common labels but poorly on rare ones, you might need to re-balance your dataset, for instance by over-sampling sequences with rare labels.
-   **Model Architecture**: While a standard classification head works well, more complex architectures could be explored. For example, using separate prediction heads for different groups of related tasks might improve performance if the tasks are not closely related.
-   **Hyperparameter Tuning**: Standard hyperparameters like `learning_rate`, `weight_decay`, and `num_train_epochs` are just as important here. It may require more careful tuning to find a set of parameters that works well for all tasks simultaneously.

## 5. Troubleshooting

-   **Negative Transfer**: Sometimes, training on multiple tasks can hurt performance on a specific task compared to training a model for that task alone. This is known as negative transfer. If this occurs:
    -   Verify that the tasks are actually related. Forcing a model to learn unrelated tasks can be detrimental.
    -   Consider grouping tasks. Instead of one model for all tasks, you could train one model for a subset of highly related tasks.
    -   Adjusting the learning rate or using a more sophisticated optimization scheme can sometimes help.
-   **Incorrect Label Preparation**: Ensure your `label_map` in the config file is complete and that the `multi_label_sep` character in `load_local_data` matches what is used in your data file. Any mismatch will lead to incorrect label parsing and poor model performance.
-   **`CUDA out of memory`**: As with other tasks, this can be resolved by reducing `per_device_train_batch_size` in the configuration file.
