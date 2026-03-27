# Fine-tuning Strategies and Best Practices

Fine-tuning is the process of adapting a pre-trained foundation model to a specific downstream task. While powerful, the success of fine-tuning depends heavily on the right strategies for data preparation, hyperparameter selection, and training approach.

## 1. Data Selection and Preprocessing

The quality of your fine-tuning data is the single most important factor for success.

### Data Quality over Quantity
- **Accurate Labels**: Ensure your labels are as accurate as possible. A smaller, high-quality dataset will produce a better model than a large, noisy one.
- **Remove Duplicates and Contaminants**: Clean your dataset to remove duplicate sequences or any sequences that don't belong (e.g., adapter sequences, poor quality reads).

### Data Balancing
- **The Problem**: If your dataset is highly imbalanced (e.g., 99% negative examples and 1% positive examples), the model may learn to simply predict the majority class.
- **Strategies**:
    - **Downsampling**: Randomly remove samples from the majority class. This is simple but discards data.
    - **Upsampling**: Duplicate samples from the minority class. This can lead to overfitting on the minority class.
    - **Weighted Loss**: A better approach, supported by DNALLM. You can configure the trainer to give more weight to errors on the minority class, forcing the model to pay more attention to it.

### Data Splitting
- **The Rule**: Always split your data into three sets: **training**, **validation**, and **testing**.
    - **Training Set**: Used to update the model's weights.
    - **Validation Set**: Used during training to monitor performance on unseen data, tune hyperparameters, and decide when to stop training (early stopping).
    - **Test Set**: Held out until the very end. It is used only once to get a final, unbiased estimate of the model's performance.

## 2. Hyperparameter Tuning

Hyperparameters are the "dials" you can turn to control the training process. Finding the right combination is key.

- **Learning Rate (`learning_rate`)**: The most critical hyperparameter. It controls how much the model's weights are updated in each step.
    - **Too high**: The model may diverge and fail to learn.
    - **Too low**: Training will be very slow and may get stuck in a suboptimal solution.
    - **Recommendation**: Start with a small value typical for fine-tuning, such as `1e-5` or `2e-5`. Use a learning rate scheduler (`lr_scheduler_type`) like `cosine` or `linear` to gradually decrease the learning rate during training.

- **Batch Size (`per_device_train_batch_size`)**: The number of samples processed in each training step.
    - **Constraint**: Limited by your GPU memory. Find the largest size that doesn't cause an out-of-memory error.
    - **Effect**: Larger batch sizes can lead to more stable training but may generalize slightly worse. Common values are 8, 16, 32, or 64.

- **Number of Epochs (`num_train_epochs`)**: An epoch is one full pass through the training dataset.
    - **Too few**: The model will be underfit.
    - **Too many**: The model will overfit to the training data.
    - **Recommendation**: Use **early stopping**. Monitor performance on the validation set and stop training when the validation metric (e.g., `eval_loss` or `eval_accuracy`) stops improving. DNALLM's `DNATrainer` handles this automatically.

- **Weight Decay (`weight_decay`)**: A regularization technique that helps prevent overfitting by penalizing large weights. A common value is `0.01`.

## 3. Fine-tuning Strategies

### Full Fine-tuning
- **What it is**: All the weights of the pre-trained model are updated during training.
- **Pros**: Can achieve the highest possible performance.
- **Cons**: Requires the most memory and computational resources. Can be prone to "catastrophic forgetting," where the model loses some of its general pre-trained knowledge.

### Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods freeze most of the pre-trained model's weights and only train a small number of new parameters. This is much more efficient.

- **LoRA (Low-Rank Adaptation)**: The most popular PEFT method, fully supported in DNALLM.
    - **How it works**: LoRA injects small, trainable "adapter" matrices into the layers of the Transformer. Only these small matrices are trained, representing the "update" to the original weights.
    - **Pros**:
        - **Drastically reduces memory**: Allows you to fine-tune very large models on consumer GPUs.
        - **Faster training**: Fewer parameters to update.
        - **Portable**: The trained LoRA adapters are tiny (a few megabytes), making it easy to store and share many different "specialized" versions of a single base model.
    - **Configuration**: In DNALLM, you can enable LoRA in your training configuration file. See the Advanced Fine-tuning Techniques tutorial.

**Recommendation**: For most applications, starting with **LoRA** is highly recommended due to its efficiency and strong performance.

---

**Next**: Learn how to measure the success of your fine-tuning with Evaluation Metrics.