# Supported Data Formats and Conversion

The `DNADataset` class in DNALLM is highly flexible and can load data from a wide variety of formats. This guide covers the most common formats and provides examples for loading and converting them.

## 1. Supported Data Formats

The `DNADataset.load_local_data()` method can handle:

-   **Tabular Files**: `csv`, `tsv`
-   **Structured Files**: `json`, `jsonl`
-   **High-Performance Formats**: `arrow`, `parquet`
-   **Raw Sequence Files**: `fasta`, `txt`
-   **In-Memory Objects**: Python `dict` or `list` of dictionaries.

For security and compatibility reasons, loading directly from `pickle` files is not supported, but you can easily convert them.

## 2. Loading Standard Formats

For most file-based formats, you can use the `DNADataset.load_local_data()` class method. The key is to specify the column names for your sequences and labels if they differ from the defaults (`sequence` and `label`).

### CSV / TSV

```python
from dnallm.datahandling.data import DNADataset

# Assuming 'my_data.csv' has columns 'dna_string' and 'target'
dna_ds = DNADataset.load_local_data(
    "my_data.csv",
    seq_col="dna_string",
    label_col="target"
)
print(dna_ds)
```

**JSONL Format:**
Create a file named `train.jsonl`. Each line is a JSON object.

```json
// file: my_dataset/train.jsonl
{"sequence": "GATTACAGATTACAGATTACAGATTACA", "label": 1}
{"sequence": "CGCGCGCGCGCGCGCGCGCGCGCGCGCG", "label": 0}
{"sequence": "AAATTTCCGGGAAATTTCCGGGAAATTT", "label": 1}
```

### For Pre-training

A simple text file where each line is a sequence is sufficient.

```text
# file: my_corpus.txt
GATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACA...
CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC...
```

## 3. Conversion Example: FASTA to CSV

Often, you will have your sequences in a FASTA file and your labels in a separate file. The `dnallm.datahandling.data` module provides a `fasta_to_df` utility to easily parse FASTA files into a pandas DataFrame, which you can then merge with your labels.

Let's assume you have `sequences.fa` and `labels.csv` (with a `name` column matching the FASTA headers and a `label` column).

```python
import pandas as pd
from dnallm.datahandling.data import fasta_to_df

# Example usage
fasta_path = "sequences.fa"
labels_path = "labels.csv"
output_path = "train_dataset.csv"

# 1. Load sequences from FASTA into a DataFrame
# The 'name' column will contain the sequence headers from the FASTA file.
seq_df = fasta_to_df(fasta_path) # Columns: 'name', 'sequence'

# 2. Load labels and merge with sequences based on the name
label_df = pd.read_csv(labels_path)
merged_df = pd.merge(seq_df, label_df, on="name")

# 3. Save the final dataset to a CSV file
merged_df[["sequence", "label"]].to_csv(output_path, index=False)
print("Conversion complete!")
```

## 4. Loading Other Formats (Arrow, Parquet, Pickle)

The DNALLM `DNADataset` class can directly load data from several high-performance formats like Apache Arrow and Parquet. This is often more efficient than using CSV, especially for large datasets.

### Loading Arrow or Parquet Files

If your data is already in Arrow or Parquet format with `sequence` and `label` columns, you can load it directly.

```python
from dnallm.datahandling.data import DNADataset

# Load from a Parquet file
dna_ds_from_parquet = DNADataset.load_local_data("my_dataset.parquet")

# Load from an Arrow file
dna_ds_from_arrow = DNADataset.load_local_data("my_dataset.arrow")

print(dna_ds_from_parquet)
```

### Converting from Pickle to a Supported Format

While `DNADataset` doesn't load Pickle files directly for security and compatibility reasons, you can easily convert them using `pandas`.

Let's say you have a `data.pkl` file containing a list of dictionaries or a pandas DataFrame.

```python
import pandas as pd

# 1. Load the data from the Pickle file
data = pd.read_pickle("my_dataset.pkl")

# 2. Convert to a pandas DataFrame if it's not already
df = pd.DataFrame(data)

# 3. Save to a supported format like CSV or Parquet
df[["sequence", "label"]].to_csv("converted_dataset.csv", index=False)
# Or for better performance:
# df[["sequence", "label"]].to_parquet("converted_dataset.parquet")
```
```