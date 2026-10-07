<!-- generated-by: gsd-doc-writer -->
# Supported Data Formats and Conversion

The `DNADataset` class in DNALLM is highly flexible and can load data from a wide variety of formats. This guide covers the most common formats and provides examples for loading and converting them.

## 1. Supported Data Formats

The `DNADataset.load_local_data()` method can handle:

-   **Tabular Files**: `csv`, `tsv`
-   **Structured Files**: `json`
-   **High-Performance Formats**: `arrow`, `parquet`, `pkl`, `pickle`
-   **Raw Sequence Files**: `fasta`, `txt`
-   **Multiple Files**: a `dict` maps split names to file paths (e.g., `{"train": "train.csv", "test": "test.csv"}`) for pre-split datasets; a `list` of file paths loads several files of the same type as a single dataset.

Note that `.jsonl` is **not** a supported extension — `load_local_data()` raises `ValueError: Unsupported file type: jsonl`. Convert JSON Lines data to `json`, `csv`, or `parquet` before loading.

## 2. Loading Standard Formats

For most file-based formats, you can use the `DNADataset.load_local_data()` class method. The key is to specify the column names for your sequences and labels if they differ from the defaults (`sequence` and `labels`).

### CSV / TSV

```python
from dnallm.datahandling.data import DNADataset

# Assuming 'my_data.csv' has columns 'dna_string' and 'target'
dna_ds = DNADataset.load_local_data("my_data.csv", seq_col="dna_string", label_col="target")
print(dna_ds)
```

### JSON

Create a `json` file containing an array of records:

```json
// file: my_dataset/train.json
[
    {"sequence": "GATTACAGATTACAGATTACAGATTACA", "labels": 1},
    {"sequence": "CGCGCGCGCGCGCGCGCGCGCGCGCGCG", "labels": 0},
    {"sequence": "AAATTTCCGGGAAATTTCCGGGAAATTT", "labels": 1}
]
```

`.jsonl` (JSON Lines) files are not supported — the `.jsonl` extension raises `ValueError: Unsupported file type: jsonl`. Convert the file to `json`, `csv`, or `parquet` first.

### For Pre-training

For `.txt` files, each line must contain a sequence **and** its label, separated by whitespace (or a custom `sep`). Lines holding only a sequence are dropped, so a bare one-sequence-per-line file loads as an empty dataset — add a label column (a placeholder value is fine) or use `csv`/`json`/`parquet` instead.

```text
GATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACA 0
CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC 0
```

## 3. Conversion Example: FASTA to CSV

Often, you will have your sequences in a FASTA file and your labels in a separate file. `DNADataset.load_local_data()` parses FASTA files directly (`.fa`, `.fna`, `.fas`, `.fasta`): each record's label is read from the FASTA header — the last `fasta_sep`-separated field if the header contains `fasta_sep` (default `|`), otherwise the entire header.

Let's assume you have `sequences.fa` and `labels.csv` (with a `name` column matching the FASTA headers and a `label` column).
```text
>seq1
GATTACAGATTACAGATTACAGATTACAGATTACA
>seq2
CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG
```

```python
import pandas as pd
from dnallm.datahandling.data import DNADataset

# 1. Load the FASTA file. With no '|' in the headers, the full header
#    (e.g. 'seq1') is kept as the record id in the 'labels' column.
dna_ds = DNADataset.load_local_data("sequences.fa")
seq_df = dna_ds.dataset.to_pandas()  # Columns: 'sequence', 'labels'

# 2. Load external labels and merge on the FASTA header id
label_df = pd.read_csv("labels.csv")  # Columns: 'name', 'label'
merged_df = seq_df.merge(label_df, left_on="labels", right_on="name")

# 3. Save the final dataset as a CSV file
(
    merged_df[["sequence", "label"]]
    .rename(columns={"label": "labels"})
    .to_csv("train_dataset.csv", index=False)
)
```

If the labels are already embedded in the FASTA headers, skip the merge: put each label behind a `fasta_sep` in the header (e.g. `>seq1|1`) and the loader uses the last field as the label.

## 4. Loading Other Formats (Arrow, Parquet, Pickle)

The DNALLM `DNADataset` class can directly load data from several high-performance formats like Apache Arrow and Parquet. This is often more efficient than using CSV, especially for large datasets.

### Loading Arrow or Parquet Files

```python
from dnallm.datahandling.data import DNADataset

# Load from a Parquet file
dna_ds_from_parquet = DNADataset.load_local_data("my_dataset.parquet")

# Load from an Arrow file
dna_ds_from_arrow = DNADataset.load_local_data("my_dataset.arrow")

print(dna_ds_from_parquet)
```

### Loading Pickle Files

`DNADataset` loads `.pkl` / `.pickle` files directly (a single file only — passing a list of pickle files raises `ValueError`). The file must contain a dictionary of columns, e.g. `{"sequence": [...], "labels": [...]}`:

```python
import pickle  # ruff: ignore[suspicious-pickle-import]
from dnallm.datahandling.data import DNADataset

# 1. Serialize your dataset as a dict of columns
data = {"sequence": ["GATTACAGATTACAGATTACAGATTACA", "CGCGCGCGCGCGCGCGCGCGCGCGCGCG"], "labels": [1, 0]}
with open("my_dataset.pkl", "wb") as f:
    pickle.dump(data, f)

# 2. Load it directly
dna_ds = DNADataset.load_local_data("my_dataset.pkl")
print(dna_ds)
```

If your data lives in a pandas DataFrame, convert it to a dict of columns first: `pickle.dump({"sequence": df["sequence"].tolist(), "labels": df["labels"].tolist()}, f)`.

---

## Next Steps

- [Data Preparation](data_preparation.md) - Learn about data collection and organization
- [Data Augmentation](data_augmentation.md) - Learn about data augmentation techniques
- [Quality Control](quality_control.md) - Ensure data quality and consistency
- [Data Processing Troubleshooting](../../faq/data_processing_troubleshooting.md) - Common data processing issues and solutions
