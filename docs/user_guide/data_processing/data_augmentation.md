# Data Augmentation for DNA Sequences

Data augmentation is a powerful technique to increase the diversity of your training data without collecting new samples. By applying realistic transformations to your existing sequences, you can help the model generalize better and prevent overfitting.

## 1. Why Augment DNA Sequences?

In biology, certain transformations result in a sequence that is functionally equivalent or very similar to the original.

-   **Biological Equivalence**: The reverse complement of a DNA strand carries the same genetic information.
-   **Robustness to Noise**: Small mutations or sequencing errors should not drastically change a model's prediction for robust tasks.
-   **Increased Data Size**: Augmentation artificially expands your dataset, which is especially useful when you have limited labeled data.

## 2. Common Augmentation Methods

Here are some common methods for augmenting DNA sequences, which can be implemented with simple Python functions.

### Reverse Complement

This is the most common and biologically sound augmentation method. The model should learn that a sequence and its reverse complement are often functionally identical.

#### How to Operate

The `dnallm.datahandling.data` module provides an efficient `reverse_complement` function.

```python
from dnallm.datahandling.data import reverse_complement

# Example
original_seq = "ATGC"
augmented_seq = reverse_complement(original_seq)
print(f"Original:   {original_seq}")
print(f"Augmented:  {augmented_seq}") # Output: GCAT
```

When training, you can randomly choose to replace a sequence with its reverse complement in each training batch.

### Random Mutations

Introducing random point mutations (substitutions, insertions, or deletions) can make the model more robust to natural variations and sequencing errors.

#### How to Operate

The `dnallm.datahandling.data` module includes a `random_mutation` function for this purpose.

```python
from dnallm.datahandling.data import random_mutation

# Example
original_seq = "GATTACAGATTACA"
# The function returns the mutated sequence and the number of mutations
augmented_seq, num_mutations = random_mutation(original_seq, num_mutations=2)

print(f"Original:   {original_seq}")
print(f"Augmented:  {augmented_seq}") # e.g., GATCACAGATTACA
print(f"Mutations:  {num_mutations}")
```

**Caution**: Use a low `mutation_rate`. High rates can destroy the biological signal in the sequence, turning it into noise.

---

## Next Steps

- [Data Preparation](data_preparation.md) - Learn about data collection and organization
- [Format Conversion](format_conversion.md) - Convert between different data formats
- [Quality Control](quality_control.md) - Ensure data quality and consistency
- [Data Processing Troubleshooting](../../faq/data_processing_troubleshooting.md) - Common data processing issues and solutions