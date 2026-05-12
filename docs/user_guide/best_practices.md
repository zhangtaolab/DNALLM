# Best Practices for DNALLM

To get the most out of DNALLM, follow these best practices for data handling, model selection, and training.

## 1. Data Preparation

-   **Start with High-Quality Data**: The principle of "garbage in, garbage out" is especially true for deep learning. Use sequences from trusted sources like NCBI or Ensembl.

-   **Perform Quality Control**: Always clean your data before training.
    -   Use the `DNADataset.validate_sequences()` method to filter out sequences that are too short, too long, or contain invalid characters.
    -   Check for and handle class imbalance in classification tasks. You can oversample the minority class or use weighted loss functions.

-   **Use Efficient Formats**: For large datasets, prefer high-performance formats like **Parquet** or **Arrow** over CSV. They are significantly faster to load and process.

    ```python
    # Load it quickly
    from dnallm.datahandling import DNADataset

    dna_ds = DNADataset.from_huggingface(
        "plantcad/maize-allele-frequency-raw-data"
    )
    ```

-   **Leverage Data Augmentation**: Increase the diversity of your training data to improve model generalization.
    -   For most DNA tasks, adding the **reverse complement** is a safe and effective augmentation strategy.
    -   Use `dna_ds.augment_reverse_complement()` to double your dataset size.

## 2. Model Selection

-   **Match the Model to the Task**:
    -   **Classification/Feature Extraction**: Use encoder-only models like **DNABERT** or **Nucleotide Transformer (ESM-based)**. They are excellent at understanding sequence context.
    -   **Sequence Generation**: Use decoder-only models like **DNAGPT (GPT-based)** or **Evo (Hyena-based)**. They are designed to predict the next token.
    -   **Long Sequences (>5kb)**: For very long sequences, consider architectures designed for efficiency, such as **Caduceus** or **HyenaDNA**. Standard transformers can be too slow and memory-intensive.

-   **Start with a Pre-trained Model**: Never train from scratch unless you have a massive dataset (billions of sequences). Fine-tuning a model pre-trained on a large biological corpus (like DNABERT2 or Nucleotide Transformer) will yield better results much faster.

-   **Check the Tokenizer**: Ensure the model's tokenizer is appropriate for your data. Most DNA models use a k-mer based or BPE tokenizer. Using a model trained on English text with its original tokenizer will not work for DNA.

## 3. Training and Fine-tuning

-   **Use Mixed-Precision Training**: Enable `fp16` (or `bf16` on newer GPUs) in your training configuration. This can speed up training by 2-3x and significantly reduce memory usage with minimal impact on accuracy.

    ```yaml
    # In your config.yaml
    finetune:
      fp16: true
    ```

-   **Optimize Memory Usage**: If you encounter `CUDA out of memory` errors:
    -   **Gradient Accumulation**: This is the most effective technique. It simulates a larger batch size without using more memory. Set `gradient_accumulation_steps` to 2, 4, 8, or higher.
    -   **Reduce Batch Size**: Lower `per_device_train_batch_size`.

-   **Log and Monitor Training**: Use logging tools like Weights & Biases (`wandb`) or TensorBoard to track your training progress. This helps you spot issues like overfitting or unstable training early. Set the `report_to` parameter in your training arguments under the `finetune` section within the `config.yaml`.

    ```yaml
    finetune:
      report_to: "tensorboard"
    ```

-   **Start Small**: Before launching a multi-day training run on your full dataset, test your entire pipeline on a small subset (e.g., 1% of the data) for one or two epochs. This ensures there are no bugs in your code or configuration. You can subsample your dataset with the `DNADataset.sampling` method.