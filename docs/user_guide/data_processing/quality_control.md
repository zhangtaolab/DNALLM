# Data Quality Control

"Garbage in, garbage out." This saying is especially true for training deep learning models. Ensuring your DNA data is clean, consistent, and free of errors is a crucial step before training. This guide outlines common quality control checks.

## 1. Common Data Issues and Solutions

### Invalid Characters

**Problem**: Your DNA sequences might contain characters other than `A`, `C`, `G`, `T`, and `N` (for unknown nucleotides). These can come from parsing errors or malformed source files. Most tokenizers will fail or produce incorrect tokens if they encounter unexpected characters like `U`, `R`, `Y`, or punctuation.

**Solution**: Sanitize your sequences to ensure they only contain valid characters.

The `dnallm.datahandling.data.clean_sequence` function is designed for this purpose.

```python
from dnallm.datahandling.data import clean_sequence

# Apply this function to every sequence in your dataset before saving.
# For a DNADataset object:
# dna_ds.validate_sequences(valid_chars="ACGTN")
```

### Inconsistent Sequence Lengths

**Problem**: For some biological tasks, all sequences are expected to be the same length (e.g., classifying 150bp promoter regions). Drastic variations in length might indicate data collection errors.

**Solution**: Analyze the distribution of sequence lengths in your dataset.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_dataset.csv")
lengths = df["sequence"].str.len()

print(lengths.describe())

# Plot a histogram to visualize the distribution
lengths.hist(bins=50)
plt.title("Distribution of Sequence Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()
```
Based on the distribution, you can decide whether to filter out sequences that are too short or too long, or to pad/truncate them during tokenization.

### Label Imbalance

**Problem**: In classification tasks, having a severe imbalance between classes (e.g., 99% negative examples and 1% positive examples) can cause the model to simply predict the majority class every time.

**Solution**:
-   **Check Class Distribution**: Use `df['label'].value_counts()` to see the number of samples per class.
-   **Resampling**:
    -   **Oversampling**: Randomly duplicate samples from the minority class.
    -   **Undersampling**: Randomly remove samples from the majority class.
-   **Weighted Loss**: During training, you can assign a higher weight to the minority class in the loss function. The DNALLM `finetune` command can handle this if class weights are provided.