"""DNA Dataset handling and processing utilities.

This module provides comprehensive tools for loading, processing, and managing
DNA sequence datasets. It supports various file formats, data augmentation
techniques, and statistical analysis.
"""

import os
import random
from collections.abc import Callable
from typing import Any
import pandas as pd
import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    concatenate_datasets,
)
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

from ..utils.sequence import (
    check_sequence,
    calc_gc_content,
    reverse_complement,
    random_generate_sequences,
)


class DNADataset:
    """A comprehensive wrapper for DNA sequence datasets with advanced
    processing capabilities.

    This class provides methods for loading DNA datasets from various sources
    (local files, Hugging Face Hub, ModelScope), encoding sequences with
    tokenizers, data augmentation, statistical analysis, and more.

    Attributes:
        dataset: The underlying Hugging Face Dataset or DatasetDict
        tokenizer: Tokenizer for sequence encoding
        max_length: Maximum sequence length for tokenization
        sep: Separator for multi-label data
        multi_label_sep: Separator for multi-label sequences
        data_type: Type of the dataset (classification, regression, etc.)
        stats: Cached dataset statistics
        stats_for_plot: Cached statistics for plotting
    """

    def __init__(
        self,
        ds: Dataset | DatasetDict,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 512,
    ) -> None:
        """Initialize a DNADataset.

        Args:
            ds: A Hugging Face Dataset containing at least 'sequence' and
                'label' fields
            tokenizer: A Hugging Face tokenizer for encoding sequences
            max_length: Maximum length for tokenization
        """
        if ds is None:
            raise TypeError("Dataset cannot be None")

        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.dataset = ds
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep: str | None = None
        self.multi_label_sep: str | None = None
        self.data_type: str | None = None
        self.stats: dict | None = None
        self.stats_for_plot: pd.DataFrame | None = None
        self.__data_type__()  # Determine the data type of the dataset

    @classmethod
    def load_local_data(
        cls,
        file_paths: str | list | dict,
        seq_col: str = "sequence",
        label_col: str = "labels",
        sep: str | None = None,
        fasta_sep: str = "|",
        multi_label_sep: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 512,
    ) -> "DNADataset":
        """Load DNA sequence datasets from one or multiple local files.

        Supports input formats: csv, tsv, json, parquet, arrow, dict, fasta,
        txt, pkl, pickle.

        Args:
            file_paths: Single dataset: Provide one file path
                (e.g., "data.csv").
                Pre-split datasets: Provide a dict like
                {"train": "train.csv", "test": "test.csv"}
            seq_col: Column name for DNA sequences
            label_col: Column name for labels
            sep: Delimiter for CSV, TSV, or TXT
            fasta_sep: Delimiter for FASTA files
            multi_label_sep: Delimiter for multi-label sequences
            tokenizer: A tokenizer for sequence encoding
            max_length: Max token length

        Returns:
            An instance wrapping a Dataset or DatasetDict

        Raises:
            ValueError: If file type is not supported
        """
        # Set separators
        cls.sep = sep
        cls.multi_label_sep = multi_label_sep
        # Check if input is a list or dict
        if isinstance(
            file_paths, dict
        ):  # Handling multiple files (pre-split datasets)
            ds_dict = {}
            for split, path in file_paths.items():
                ds_dict[split] = cls._load_single_data(
                    path, seq_col, label_col, sep, fasta_sep, multi_label_sep
                )
            dataset = DatasetDict(ds_dict)
        else:  # Handling a single file
            dataset = cls._load_single_data(
                file_paths, seq_col, label_col, sep, fasta_sep, multi_label_sep
            )
        dataset.stats = None  # Initialize stats as None

        return cls(dataset, tokenizer=tokenizer, max_length=max_length)

    @classmethod
    def _load_single_data(
        cls,
        file_path: str | list,
        seq_col: str = "sequence",
        label_col: str = "labels",
        sep: str | None = None,
        fasta_sep: str = "|",
        multi_label_sep: str | None = None,
    ) -> Dataset:
        """Load DNA data (sequences and labels) from a local file.

        Supported file types:
            - For structured formats (CSV, TSV, JSON, Parquet, Arrow, dict),
              uses load_dataset from datasets.
          - For FASTA and TXT, uses custom parsing.

        Args:
            file_path: For most file types, a path (or pattern) to the file(s).
                For 'dict', a dictionary.
            seq_col: Name of the column containing the DNA sequence
            label_col: Name of the column containing the label
            sep: Delimiter for CSV, TSV, or TXT files
            fasta_sep: Delimiter for FASTA files
            multi_label_sep: Delimiter for multi-label sequences

        Returns:
            A Hugging Face Dataset with 'sequence' and 'labels' columns

        Raises:
            ValueError: If file type is not supported
        """
        file_path, file_type = cls._normalize_file_path(file_path)
        sep = cls._determine_separator(file_type, sep)
        file_type = cls._check_header_and_type(
            file_path, file_type, seq_col, label_col, sep
        )

        # Load dataset based on file type
        if file_type in ["csv", "tsv", "json", "parquet", "arrow"]:
            ds = cls._load_structured_data(file_path, file_type, sep)
        elif file_type in ["pkl", "pickle", "dict"]:
            if isinstance(file_path, list):
                raise ValueError(
                    "Dictionary/pickle files must be single files, not lists"
                )
            ds = cls._load_dict_data(file_path)
        elif file_type in ["fa", "fna", "fas", "fasta"]:
            if isinstance(file_path, list):
                raise ValueError("FASTA files must be single files, not lists")
            ds = cls._load_fasta_data(file_path, fasta_sep)
        elif file_type == "txt":
            if isinstance(file_path, list):
                raise ValueError("TXT files must be single files, not lists")
            ds = cls._load_txt_data(file_path, seq_col, label_col, sep)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Rename columns if needed
        ds = cls._rename_columns(ds, seq_col, label_col)

        # Format labels
        ds = cls._format_labels(ds, multi_label_sep)

        return ds

    @classmethod
    def _normalize_file_path(
        cls, file_path: str | list
    ) -> tuple[str | list, str]:
        """Normalize file path and determine file type."""
        if isinstance(file_path, list):
            file_path = [os.path.expanduser(fpath) for fpath in file_path]
            file_type = os.path.basename(file_path[0]).split(".")[-1].lower()
        else:
            file_path = os.path.expanduser(file_path)
            file_type = os.path.basename(file_path).split(".")[-1].lower()
        return file_path, file_type

    @classmethod
    def _determine_separator(
        cls, file_type: str, sep: str | None
    ) -> str | None:
        """Determine the appropriate separator for the file type."""
        if file_type == "csv":
            return sep if sep else ","
        elif file_type == "tsv":
            return sep if sep else "\t"
        return sep

    @classmethod
    def _check_header_and_type(
        cls,
        file_path: str | list,
        file_type: str,
        seq_col: str,
        label_col: str,
        sep: str | None,
    ) -> str:
        """Check if file has header and determine final file type."""
        if file_type in ["csv", "tsv", "txt"] and isinstance(file_path, str):
            with open(file_path) as f:
                header = f.readline().strip()
                if not header or (
                    seq_col not in header and label_col not in header
                ):
                    return "txt"
        return file_type

    @classmethod
    def _load_structured_data(
        cls, file_path: str | list, file_type: str, sep: str | None
    ) -> Dataset:
        """Load structured data files (CSV, TSV, JSON, Parquet, Arrow)."""
        if file_type in ["csv", "tsv"]:
            ds = load_dataset(
                "csv", data_files=file_path, split="train", delimiter=sep
            )
        elif file_type == "json":
            ds = load_dataset("json", data_files=file_path, split="train")
        elif file_type in ["parquet", "arrow"]:
            ds = load_dataset(file_type, data_files=file_path, split="train")
        return ds

    @classmethod
    def _load_dict_data(cls, file_path: str) -> Dataset:
        """Load dictionary/pickle data files."""
        import pickle  # noqa: S403

        data = pickle.load(open(file_path, "rb"))  # noqa: S301
        return Dataset.from_dict(data)

    @classmethod
    def _load_fasta_data(cls, file_path: str, fasta_sep: str) -> Dataset:
        """Load FASTA data files."""
        sequences, labels = [], []
        with open(file_path) as f:
            seq = ""
            lab = None
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq and lab is not None:
                        sequences.append(seq)
                        labels.append(lab)
                    header = line[1:].strip()
                    lab = (
                        header.split(fasta_sep)[-1]
                        if fasta_sep in header
                        else header
                    )
                    seq = ""
                else:
                    seq += line.strip()
            if seq and lab is not None:
                sequences.append(seq)
                labels.append(lab)
        return Dataset.from_dict({"sequence": sequences, "labels": labels})

    @classmethod
    def _load_txt_data(
        cls, file_path: str, seq_col: str, label_col: str, sep: str | None
    ) -> Dataset:
        """Load TXT data files."""
        sequences, labels = [], []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i == 0 and seq_col in line and label_col in line:
                    return load_dataset(
                        "csv",
                        data_files=file_path,
                        split="train",
                        delimiter=sep,
                    )
                record = (
                    line.strip().split(sep) if sep else line.strip().split()
                )
                if len(record) >= 2:
                    sequences.append(record[0])
                    labels.append(record[1])
        return Dataset.from_dict({"sequence": sequences, "labels": labels})

    @classmethod
    def _rename_columns(
        cls, ds: Dataset, seq_col: str, label_col: str
    ) -> Dataset:
        """Rename columns to standard names if needed."""
        if seq_col != "sequence" and seq_col in ds.column_names:
            if "sequence" not in ds.features:
                ds = ds.rename_column(seq_col, "sequence")
        if label_col != "labels" and label_col in ds.column_names:
            if "labels" not in ds.features:
                ds = ds.rename_column(label_col, "labels")
        return ds

    @classmethod
    def _format_labels(
        cls, ds: Dataset, multi_label_sep: str | None
    ) -> Dataset:
        """Format labels to appropriate types."""

        def format_labels(example):
            labels = example["labels"]
            if isinstance(labels, str):
                try:
                    if (
                        multi_label_sep is not None
                        and multi_label_sep in labels
                    ):
                        example["labels"] = [
                            float(x) for x in labels.split(multi_label_sep)
                        ]
                    else:
                        try:
                            example["labels"] = float(labels)
                        except ValueError:
                            example["labels"] = labels
                except (ValueError, TypeError):
                    example["labels"] = labels
            return example

        if "labels" in ds.column_names:
            ds = ds.map(format_labels, desc="Format labels")
        return ds

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        seq_col: str = "sequence",
        label_col: str = "labels",
        data_dir: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 512,
    ) -> "DNADataset":
        """Load a dataset from the Hugging Face Hub.

        Args:
            dataset_name: Name of the dataset
            seq_col: Column name for the DNA sequence
            label_col: Column name for the label
            data_dir: Data directory in a dataset
            tokenizer: Tokenizer for sequence encoding
            max_length: Max token length

        Returns:
            An instance wrapping a datasets.Dataset
        """
        if data_dir:
            ds = load_dataset(dataset_name, data_dir=data_dir)
        else:
            ds = load_dataset(dataset_name)
        # Rename columns if necessary
        if seq_col != "sequence":
            ds = ds.rename_column(seq_col, "sequence")
        if label_col != "labels":
            ds = ds.rename_column(label_col, "labels")
        return cls(ds, tokenizer=tokenizer, max_length=max_length)

    @classmethod
    def from_modelscope(
        cls,
        dataset_name: str,
        seq_col: str = "sequence",
        label_col: str = "labels",
        data_dir: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 512,
    ) -> "DNADataset":
        """Load a dataset from the ModelScope.

        Args:
            dataset_name: Name of the dataset
            seq_col: Column name for the DNA sequence
            label_col: Column name for the label
            data_dir: Data directory in a dataset
            tokenizer: Tokenizer for sequence encoding
            max_length: Max token length

        Returns:
            An instance wrapping a datasets.Dataset
        """
        from modelscope import MsDataset

        if data_dir:
            ds = MsDataset.load(dataset_name, data_dir=data_dir)
        else:
            ds = MsDataset.load(dataset_name)
        # Rename columns if necessary
        if seq_col != "sequence":
            ds = ds.rename_column(seq_col, "sequence")
        if label_col != "labels":
            ds = ds.rename_column(label_col, "labels")
        return cls(ds, tokenizer=tokenizer, max_length=max_length)

    def encode_sequences(
        self,
        padding: str = "max_length",
        return_tensors: str = "pt",
        remove_unused_columns: bool = False,
        uppercase: bool = False,
        lowercase: bool = False,
        task: str | None = "SequenceClassification",
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Encode all sequences using the provided tokenizer.

        The dataset is mapped to include tokenized fields along with the
        label, making it directly usable with Hugging Face Trainer.

        Args:
            padding: Padding strategy for sequences. Can be 'max_length' or
                'longest'. Use 'longest' to pad to the length of the longest
                sequence in case of memory outage
            return_tensors: Returned tensor types, can be 'pt', 'tf', 'np', or
                'jax'
            remove_unused_columns: Whether to remove the original 'sequence'
                and 'label' columns
            uppercase: Whether to convert sequences to uppercase
            lowercase: Whether to convert sequences to lowercase
            task: Task type for the tokenizer. If not provided, defaults to
                'SequenceClassification'
            tokenizer: Tokenizer to use for encoding. If not provided, uses
                the instance's tokenizer

        Raises:
            ValueError: If tokenizer is not provided
        """
        if not self.tokenizer:
            if tokenizer:
                self.tokenizer = tokenizer
            else:
                raise ValueError("Tokenizer is required")

        # Get tokenizer configuration
        tokenizer_config = self._get_tokenizer_config()

        # Judge the task type and apply appropriate tokenization
        if task is None:
            task = "sequenceclassification"
        task = task.lower()

        if task in ["tokenclassification", "token", "ner"]:
            self._apply_token_classification_tokenization(
                tokenizer_config, padding, uppercase, lowercase
            )
        else:
            self._apply_sequence_classification_tokenization(
                tokenizer_config, padding, uppercase, lowercase
            )

        # Post-process dataset
        self._post_process_encoded_dataset(
            remove_unused_columns, return_tensors
        )

    def _get_tokenizer_config(self) -> dict:
        """Get tokenizer configuration."""
        if not self.tokenizer:
            raise ValueError("Tokenizer is required")
        sp_token_map = (
            self.tokenizer.special_tokens_map
            if hasattr(self.tokenizer, "special_tokens_map")
            else {}
        )
        for tok_id in ["pad_id", "eos_id", "bos_id", "cls_id", "sep_id"]:
            if hasattr(self.tokenizer, tok_id):
                sp_token_map[tok_id] = getattr(self.tokenizer, tok_id)
        for token in [
            "pad_token",
            "eos_token",
            "bos_token",
            "cls_token",
            "sep_token",
        ]:
            if not sp_token_map.get(token) and hasattr(self.tokenizer, token):
                sp_token_map[token] = getattr(self.tokenizer, token)
        if "pad_id" not in sp_token_map:
            if hasattr(self.tokenizer, "pad_token_id"):
                if self.tokenizer.pad_token_id is not None:
                    sp_token_map["pad_id"] = self.tokenizer.pad_token_id
            if not sp_token_map.get("pad_id"):
                if hasattr(self.tokenizer, "eos_token_id"):
                    sp_token_map["pad_id"] = self.tokenizer.eos_token_id
        if not sp_token_map.get("pad_token"):
            if hasattr(self.tokenizer, "decode"):
                sp_token_map["pad_token"] = self.tokenizer.decode(
                    sp_token_map["pad_id"]
                )
            elif hasattr(self.tokenizer, "convert_ids_to_tokens"):
                pad_token = self.tokenizer.convert_ids_to_tokens(
                    sp_token_map["pad_id"]
                )
                sp_token_map["pad_token"] = pad_token
            elif hasattr(self.tokenizer, "decode_token"):
                sp_token_map["pad_token"] = self.tokenizer.decode_token(
                    sp_token_map["pad_id"]
                )
        return {
            "pad_token": sp_token_map.get("pad_token"),
            "pad_id": self.tokenizer.encode(sp_token_map.get("pad_token", ""))[
                -1
            ]
            if sp_token_map.get("pad_token")
            else None,
            "cls_token": sp_token_map.get("cls_token"),
            "sep_token": sp_token_map.get("sep_token"),
            "eos_token": sp_token_map.get("eos_token"),
            "max_length": self.max_length,
        }

    def _apply_sequence_classification_tokenization(
        self, config: dict, padding: str, uppercase: bool, lowercase: bool
    ) -> None:
        """Apply sequence classification tokenization."""

        def tokenize_for_sequence_classification(example):
            sequences = example["sequence"]
            if uppercase:
                sequences = [x.upper() for x in sequences]
            if lowercase:
                sequences = [x.lower() for x in sequences]
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not initialized")
            return self.tokenizer(
                sequences,
                truncation=True,
                padding=padding,
                max_length=config["max_length"],
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = self.dataset.map(
            tokenize_for_sequence_classification,
            batched=True,
            desc="Encoding inputs",
        )

    def _apply_token_classification_tokenization(
        self, config: dict, padding: str, uppercase: bool, lowercase: bool
    ) -> None:
        """Apply token classification tokenization."""

        def tokenize_for_token_classification(examples):
            return self._process_token_classification_batch(examples, config)

        self.dataset = self.dataset.map(
            tokenize_for_token_classification,
            batched=True,
            desc="Encoding inputs",
        )

    def _process_token_classification_batch(
        self, examples: dict, config: dict
    ) -> BatchEncoding:
        """Process a batch for token classification."""
        tokenized_examples: dict = {
            "sequence": [],
            "input_ids": [],
            "attention_mask": [],
        }
        if "labels" in examples:
            tokenized_examples["labels"] = []

        input_seqs = examples["sequence"]
        if isinstance(input_seqs, str):
            input_seqs = input_seqs.split(self.multi_label_sep)

        for i, example_tokens in enumerate(input_seqs):
            processed = self._process_single_token_sequence(
                example_tokens, examples, i, config
            )
            for key, value in processed.items():
                if key in tokenized_examples:
                    tokenized_examples[key].append(value)

        from transformers.tokenization_utils_base import BatchEncoding

        return BatchEncoding(tokenized_examples)

    def _process_single_token_sequence(
        self, example_tokens: list, examples: dict, i: int, config: dict
    ) -> dict:
        """Process a single token sequence for token classification."""
        if not self.tokenizer:
            raise ValueError("Tokenizer is required")
        all_ids = list(
            self.tokenizer.encode(example_tokens, is_split_into_words=True)
        )
        example_ner_tags = (
            examples["labels"][i]
            if "labels" in examples
            else [0] * len(example_tokens)
        )

        pad_len = config["max_length"] - len(all_ids)

        if pad_len >= 0:
            return self._pad_sequence(
                all_ids, example_tokens, example_ner_tags, pad_len, config
            )
        else:
            return self._truncate_sequence(
                all_ids, example_tokens, example_ner_tags, config
            )

    def _pad_sequence(
        self,
        all_ids: list,
        example_tokens: list,
        example_ner_tags: list,
        pad_len: int,
        config: dict,
    ) -> dict:
        """Pad a sequence to max_length."""
        all_masks = [1] * len(all_ids) + [0] * pad_len
        all_ids = all_ids + [config["pad_id"]] * pad_len

        if config["cls_token"]:
            example_tokens, example_ner_tags = self._add_special_tokens(
                example_tokens, example_ner_tags, pad_len, config
            )
        else:
            example_tokens = example_tokens + [config["pad_token"]] * pad_len
            example_ner_tags = example_ner_tags + [-100] * pad_len

        return {
            "sequence": example_tokens,
            "input_ids": all_ids,
            "attention_mask": all_masks,
            "labels": example_ner_tags,
        }

    def _truncate_sequence(
        self,
        all_ids: list,
        example_tokens: list,
        example_ner_tags: list,
        config: dict,
    ) -> dict:
        """Truncate a sequence to max_length."""
        all_ids = all_ids[: config["max_length"]]
        all_masks = [1] * config["max_length"]

        if config["cls_token"]:
            example_tokens, example_ner_tags = (
                self._add_special_tokens_truncated(
                    example_tokens, example_ner_tags, config
                )
            )
        else:
            example_tokens = example_tokens[: config["max_length"]]
            example_ner_tags = example_ner_tags[: config["max_length"]]

        return {
            "sequence": example_tokens,
            "input_ids": all_ids,
            "attention_mask": all_masks,
            "labels": example_ner_tags,
        }

    def _add_special_tokens(
        self,
        example_tokens: list,
        example_ner_tags: list,
        pad_len: int,
        config: dict,
    ) -> tuple:
        """Add special tokens for padding."""
        if config["sep_token"]:
            example_tokens = (
                [config["cls_token"]]
                + example_tokens
                + [config["sep_token"]]
                + [config["pad_token"]] * pad_len
            )
            example_ner_tags = (
                [-100] + example_ner_tags + [-100] * (pad_len + 1)
            )
        elif config["eos_token"]:
            example_tokens = (
                [config["cls_token"]]
                + example_tokens
                + [config["eos_token"]]
                + [config["pad_token"]] * pad_len
            )
            example_ner_tags = (
                [-100] + example_ner_tags + [-100] * (pad_len + 1)
            )
        else:
            example_tokens = (
                [config["cls_token"]]
                + example_tokens
                + [config["pad_token"]] * pad_len
            )
            example_ner_tags = [-100] + example_ner_tags + [-100] * pad_len
        return example_tokens, example_ner_tags

    def _add_special_tokens_truncated(
        self, example_tokens: list, example_ner_tags: list, config: dict
    ) -> tuple:
        """Add special tokens for truncation."""
        if config["sep_token"]:
            example_tokens = [
                config["cls_token"],
                *example_tokens[: config["max_length"] - 2],
                config["sep_token"],
            ]
            example_ner_tags = [
                -100,
                *example_ner_tags[: config["max_length"] - 2],
                -100,
            ]
        else:
            example_tokens = [
                config["cls_token"],
                *example_tokens[: config["max_length"] - 1],
            ]
            example_ner_tags = [
                -100,
                *example_ner_tags[: config["max_length"] - 1],
            ]
        return example_tokens, example_ner_tags

    def _post_process_encoded_dataset(
        self, remove_unused_columns: bool, return_tensors: str
    ) -> None:
        """Post-process the encoded dataset."""
        if remove_unused_columns:
            self._remove_unused_columns()

        # Set tensor format
        format_map = {
            "tf": "tensorflow",
            "jax": "jax",
            "np": "numpy",
            "pt": "torch",
        }
        self.dataset.set_format(type=format_map.get(return_tensors, None))
        self.dataset._is_encoded = True

    def _remove_unused_columns(self) -> None:
        """Remove unused columns from the dataset."""
        used_cols = ["labels", "input_ids", "attention_mask"]
        if isinstance(self.dataset, DatasetDict):
            for dt in self.dataset:
                unused_cols = [
                    f for f in self.dataset[dt].features if f not in used_cols
                ]
                self.dataset[dt] = self.dataset[dt].remove_columns(unused_cols)
        else:
            unused_cols = [
                f for f in self.dataset.features if f not in used_cols
            ]
            self.dataset = self.dataset.remove_columns(unused_cols)

    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Split the dataset into train, test, and validation sets.

        Args:
            test_size: Proportion of the dataset to include in the test
                split
            val_size: Proportion of the dataset to include in the validation
                split
            seed: Random seed for reproducibility
        """
        # check if the dataset is already a DatasetDict
        if isinstance(self.dataset, DatasetDict):
            raise ValueError(
                "Dataset is already a DatasetDict, no need to split"
            )
        # First, split off test+validation from training data
        split_result = self.dataset.train_test_split(
            test_size=test_size + val_size, seed=seed
        )
        train_ds = split_result["train"]
        temp_ds = split_result["test"]
        # Further split temp_ds into test and validation sets
        if val_size > 0:
            rel_val_size = val_size / (test_size + val_size)
            temp_split = temp_ds.train_test_split(
                test_size=rel_val_size, seed=seed
            )
            test_ds = temp_split["train"]
            val_ds = temp_split["test"]
            self.dataset = DatasetDict({
                "train": train_ds,
                "test": test_ds,
                "val": val_ds,
            })
        else:
            self.dataset = DatasetDict({"train": train_ds, "test": temp_ds})

    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle the dataset.

        Args:
            seed: Random seed for reproducibility
        """
        self.dataset.shuffle(seed=seed)

    def validate_sequences(
        self,
        minl: int = 20,
        maxl: int = 6000,
        gc: tuple = (0, 1),
        valid_chars: str = "ACGTN",
    ) -> None:
        """Filter the dataset to keep sequences containing valid DNA bases or
        allowed length.

        Args:
            minl: Minimum length of the sequences
            maxl: Maximum length of the sequences
            gc: GC content range between 0 and 1
            valid_chars: Allowed characters in the sequences
        """
        self.dataset = self.dataset.filter(
            lambda example: check_sequence(
                example["sequence"], minl, maxl, gc, valid_chars
            )
        )

    def random_generate(
        self,
        minl: int,
        maxl: int = 0,
        samples: int = 1,
        gc: tuple = (0, 1),
        n_ratio: float = 0.0,
        padding_size: int = 0,
        seed: int | None = None,
        label_func: Callable | None = None,
        append: bool = False,
    ) -> None:
        """Replace the current dataset with randomly generated DNA sequences.

        Args:
            minl: Minimum length of the sequences
            maxl: Maximum length of the sequences, default is the same as minl
            samples: Number of sequences to generate, default 1
            gc: GC content range, default (0,1)
            n_ratio: Include N base in the generated sequence, default 0.0
            padding_size: Padding size for sequence length, default 0
            seed: Random seed, default None
            label_func: A function that generates a label from a sequence
            append: Append the random generated data to the existing dataset
                or use the data as a dataset
        """

        def process(
            minl, maxl, number, gc, n_ratio, padding_size, seed, label_func
        ):
            sequences = random_generate_sequences(
                minl=minl,
                maxl=maxl,
                samples=number,
                gc=gc,
                n_ratio=n_ratio,
                padding_size=padding_size,
                seed=seed,
            )
            labels = []
            for seq in sequences:
                labels.append(label_func(seq) if label_func else 0)
            random_ds = Dataset.from_dict({
                "sequence": sequences,
                "labels": labels,
            })
            return random_ds

        if append:
            if isinstance(self.dataset, DatasetDict):
                for dt in self.dataset:
                    total_length = sum(
                        len(self.dataset[split])
                        for split in self.dataset.keys()
                    )
                    number = round(
                        samples * len(self.dataset[dt]) / total_length
                    )
                    random_ds = process(
                        minl,
                        maxl,
                        number,
                        gc,
                        n_ratio,
                        padding_size,
                        seed,
                        label_func,
                    )
                    self.dataset[dt] = concatenate_datasets([
                        self.dataset[dt],
                        random_ds,
                    ])
            else:
                random_ds = process(
                    minl,
                    maxl,
                    samples,
                    gc,
                    n_ratio,
                    padding_size,
                    seed,
                    label_func,
                )
                self.dataset = concatenate_datasets([self.dataset, random_ds])
        else:
            self.dataset = process(
                minl,
                maxl,
                samples,
                gc,
                n_ratio,
                padding_size,
                seed,
                label_func,
            )

    def process_missing_data(self) -> None:
        """Filter out samples with missing or empty sequences or labels."""

        def non_missing(example):
            return (
                example["sequence"]
                and example["labels"] is not None
                and example["sequence"].strip() != ""
            )

        self.dataset = self.dataset.filter(non_missing)

    def raw_reverse_complement(
        self, ratio: float = 0.5, seed: int | None = None
    ) -> None:
        """Do reverse complement of sequences in the dataset.

        Args:
            ratio: Ratio of sequences to reverse complement
            seed: Random seed for reproducibility
        """

        def process(ds, ratio, seed):
            random.seed(seed)
            number = len(ds["sequence"])
            idxlist = set(random.sample(range(number), int(number * ratio)))

            def concat_fn(example, idx):
                rc = reverse_complement(example["sequence"])
                if idx in idxlist:
                    example["sequence"] = rc
                return example

            # Create a dataset with random reverse complement.
            ds.map(concat_fn, with_indices=True, desc="Reverse complementary")
            return ds

        if isinstance(self.dataset, DatasetDict):
            for dt in self.dataset:
                self.dataset[dt] = process(self.dataset[dt], ratio, seed)
        else:
            self.dataset = process(self.dataset, ratio, seed)

    def augment_reverse_complement(
        self, reverse: bool = True, complement: bool = True
    ) -> None:
        """Augment the dataset by adding reverse complement sequences.

        This method doubles the dataset size.

        Args:
            reverse: Whether to do reverse
            complement: Whether to do complement
        """

        def process(ds, reverse, complement):
            # Create a dataset with an extra field for the reverse complement.
            def add_rc(example):
                example["rc_sequence"] = reverse_complement(
                    example["sequence"], reverse=reverse, complement=complement
                )
                return example

            ds_with_rc = ds.map(add_rc, desc="Reverse complementary")
            # Build a new dataset where the reverse complement becomes the
            # 'sequence'
            rc_ds = ds_with_rc.map(
                lambda ex: {
                    "sequence": ex["rc_sequence"],
                    "labels": ex["labels"],
                },
                desc="Data augment",
            )
            ds = concatenate_datasets([ds, rc_ds])
            ds.remove_columns(["rc_sequence"])
            return ds

        if isinstance(self.dataset, DatasetDict):
            for dt in self.dataset:
                self.dataset[dt] = process(
                    self.dataset[dt], reverse, complement
                )
        else:
            self.dataset = process(self.dataset, reverse, complement)

    def concat_reverse_complement(
        self, reverse: bool = True, complement: bool = True, sep: str = ""
    ) -> None:
        """Augment each sample by concatenating the sequence with its reverse
        complement.

        Args:
            reverse: Whether to do reverse
            complement: Whether to do complement
            sep: Separator between the original and reverse complement
                sequences
        """

        def process(ds, reverse, complement, sep):
            def concat_fn(example):
                rc = reverse_complement(
                    example["sequence"], reverse=reverse, complement=complement
                )
                example["sequence"] = example["sequence"] + sep + rc
                return example

            ds = ds.map(concat_fn, desc="Data augment")
            return ds

        if isinstance(self.dataset, DatasetDict):
            for dt in self.dataset:
                self.dataset[dt] = process(
                    self.dataset[dt], reverse, complement, sep
                )
        else:
            self.dataset = process(self.dataset, reverse, complement, sep)

    def sampling(
        self,
        ratio: float = 1.0,
        seed: int | None = None,
        overwrite: bool = False,
    ) -> "DNADataset":
        """Randomly sample a fraction of the dataset.

        Args:
            ratio: Fraction of the dataset to sample. Default is 1.0
                (no sampling)
            seed: Random seed for reproducibility
            overwrite: Whether to overwrite the original dataset with the
                sampled one

        Returns:
            A DNADataset object with sampled data
        """
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")

        random.seed(seed)
        dataset = self.dataset
        if isinstance(dataset, DatasetDict):
            for dt in dataset.keys():
                random_idx = random.sample(
                    range(len(dataset[dt])), int(len(dataset[dt]) * ratio)
                )
                dataset[dt] = dataset[dt].select(random_idx)
        else:
            random_idx = random.sample(
                range(len(dataset)), int(len(dataset) * ratio)
            )
            dataset = dataset.select(random_idx)

        if overwrite:
            self.dataset = dataset
            return self
        else:
            # Create a new DNADataset object with the sampled data
            return DNADataset(dataset, self.tokenizer, self.max_length)

    def head(
        self, head: int = 10, show: bool = False
    ) -> dict[Any, Any] | None:
        """Fetch the head n data from the dataset.

        Args:
            head: Number of samples to fetch
            show: Whether to print the data or return it

        Returns:
            A dictionary containing the first n samples if show=False,
            otherwise None
        """
        import pprint

        def format_convert(data):
            df: dict[Any, Any] = {}
            length = len(data["sequence"])
            for i in range(length):
                df[i] = {}
                for key in data.keys():
                    df[i][key] = data[key][i]
            return df

        dataset = self.dataset
        if isinstance(dataset, DatasetDict):
            df = {}
            for dt in dataset.keys():
                data = dataset[dt][:head]
                if show:
                    print(f"Dataset: {dt}")
                    pprint.pp(format_convert(data))
                else:
                    df[dt] = data
            return df if not show else None
        else:
            data = dataset[:head]
            if show:
                pprint.pp(format_convert(data))
                return None
            else:
                return dict(data)

    def show(self, head: int = 10) -> None:
        """Display the dataset.

        Args:
            head: Number of samples to display
        """
        self.head(head=head, show=True)

    def iter_batches(self, batch_size: int) -> Any:
        """Generator that yields batches of examples from the dataset.

        Args:
            batch_size: Size of each batch

        Yields:
            A batch of examples

        Raises:
            ValueError: If dataset is a DatasetDict
        """
        if isinstance(self.dataset, DatasetDict):
            raise ValueError(
                "Dataset is a DatasetDict Object, please use "
                "`DNADataset.dataset[datatype].iter_batches(batch_size)` "
                "instead."
            )
        else:
            for i in range(0, len(self.dataset), batch_size):
                yield self.dataset[i : i + batch_size]

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            Length of the dataset or total length for DatasetDict
        """
        if isinstance(self.dataset, DatasetDict):
            # Return total length across all splits
            return sum(len(self.dataset[dt]) for dt in self.dataset)
        else:
            return len(self.dataset)

    def get_split_lengths(self) -> dict | None:
        """Get lengths of individual splits for DatasetDict.

        Returns:
            Dictionary of split names and their lengths, or None for single
            dataset
        """
        if isinstance(self.dataset, DatasetDict):
            return {dt: len(self.dataset[dt]) for dt in self.dataset}
        else:
            return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            The item at the specified index

        Raises:
            ValueError: If dataset is a DatasetDict
        """
        if isinstance(self.dataset, DatasetDict):
            raise ValueError(
                "Dataset is a DatasetDict Object, please use "
                "`DNADataset.dataset[datatype].__getitem__(idx)` "
                "instead."
            )
        else:
            return self.dataset[idx]  # type: ignore[no-any-return]

    def __data_type__(self) -> None:
        """Get the data type of the dataset (classification, regression, etc.).

        This method analyzes the labels to determine if the dataset is for:
        - classification (integer or string labels)
        - regression (float labels)
        - multi-label (multiple labels per sample)
        - multi-regression (multiple float values per sample)
        """
        labels = self._extract_labels()
        if labels is None:
            self.data_type = "unknown"
            return

        if not self._is_valid_labels(labels):
            self.data_type = "unknown"
            return

        first_label = self._get_first_label(labels)
        if first_label is None:
            self.data_type = "unknown"
            return

        self.data_type = self._determine_data_type(first_label)

    def _extract_labels(self) -> list[Any] | None:
        """Extract labels from dataset."""
        if isinstance(self.dataset, DatasetDict):
            keys = list(self.dataset.keys())
            if not keys:
                raise ValueError("DatasetDict is empty.")
            if "labels" in self.dataset[keys[0]].column_names:
                return list(self.dataset[keys[0]]["labels"])
        else:
            if "labels" in self.dataset.column_names:
                return list(self.dataset["labels"])
        return None

    def _is_valid_labels(self, labels: list) -> bool:
        """Check if labels are valid and non-empty."""
        try:
            if hasattr(labels, "__len__"):
                return len(labels) > 0
            return False
        except (TypeError, AttributeError):
            return False

    def _get_first_label(self, labels: list) -> Any:
        """Get the first label from the labels list."""
        if not hasattr(labels, "__getitem__"):
            return None
        try:
            return labels[0]
        except IndexError:
            return None

    def _determine_data_type(self, first_label: Any) -> str:
        """Determine data type based on first label."""
        if isinstance(first_label, str):
            return self._determine_string_label_type(first_label)
        elif isinstance(first_label, int):
            return "classification"
        else:
            return "regression"

    def _determine_string_label_type(self, first_label: str) -> str:
        """Determine data type for string labels."""
        if self.multi_label_sep is not None and self.multi_label_sep in str(
            first_label
        ):
            multi_labels = str(first_label).split(self.multi_label_sep)
            return (
                "multi_regression" if "." in multi_labels[0] else "multi_label"
            )
        else:
            return (
                "regression" if "." in str(first_label) else "classification"
            )

    def statistics(self) -> dict:
        """Get statistics of the dataset.

        Includes number of samples, sequence length (min, max, average,
        median), label distribution, GC content (by labels), nucleotide
        composition (by labels).

        Returns:
            A dictionary containing statistics of the dataset

        Raises:
            ValueError: If statistics have not been computed yet
        """

        def prepare_dataframe(dataset) -> pd.DataFrame:
            """Convert a datasets.Dataset to pandas DataFrame if needed.

            If the input is already a pandas DataFrame, return a copy.
            """
            # avoid importing datasets at top-level to keep dependency optional
            try:
                from datasets import Dataset

                is_dataset = isinstance(dataset, Dataset)
            except Exception:
                is_dataset = False

            df: pd.DataFrame
            if is_dataset:
                df = dataset.to_pandas()
            elif isinstance(dataset, pd.DataFrame):
                df = dataset.copy()
            else:
                raise ValueError(
                    "prepare_dataframe expects a datasets.Dataset or "
                    "pandas.DataFrame"
                )
            return df

        def compute_basic_stats(
            df: pd.DataFrame, seq_col: str = "sequence"
        ) -> dict:
            """Compute number of samples and sequence length statistics."""
            seqs = df[seq_col].fillna("").astype(str)
            lens = seqs.str.len()
            return {
                "n_samples": len(lens),
                "min_len": int(lens.min()) if len(lens) > 0 else 0,
                "max_len": int(lens.max()) if len(lens) > 0 else 0,
                "mean_len": float(lens.mean())
                if len(lens) > 0
                else float("nan"),
                "median_len": float(lens.median())
                if len(lens) > 0
                else float("nan"),
            }

        stats = {}
        seq_col = "sequence"
        # label_col = "labels"  # Not used in current implementation
        if isinstance(self.dataset, DatasetDict):
            self.stats_for_plot = {}
            for split_name, split_ds in self.dataset.items():
                df = prepare_dataframe(split_ds)
                data_type = self.data_type
                basic = compute_basic_stats(df, seq_col)
                stats[split_name] = {"data_type": data_type, **basic}
                self.stats_for_plot[split_name] = df
        else:
            df = prepare_dataframe(self.dataset)
            data_type = self.data_type
            basic = compute_basic_stats(df, seq_col)
            stats["full"] = {"data_type": data_type, **basic}
            self.stats_for_plot = df

        self.stats = stats  # Store stats in the instance for later use

        return stats

    def plot_statistics(self, save_path: str | None = None) -> None:
        """Plot statistics of the dataset.

        Includes sequence length distribution (histogram),
        GC content distribution (box plot) for each sequence.
        If dataset is a DatasetDict, length plots and GC content plots from
        different datasets will be concatenated into a single chart,
        respectively. Sequence length distribution is shown as a histogram,
        with min and max lengths for its' limit.

        Args:
            save_path: Path to save the plots. If None, plots will be shown
                interactively

        Raises:
            ValueError: If statistics have not been computed yet
        """
        import altair as alt

        alt.data_transformers.enable("vegafusion")

        if self.stats is None or self.stats_for_plot is None:
            raise ValueError(
                "Statistics have not been computed yet. Please call "
                "`statistics()` method first."
            )

        task_type = self.data_type or "unknown"
        if isinstance(self.stats_for_plot, dict):
            df_list = {}
            for split_name, df in self.stats_for_plot.items():
                df_list[split_name] = df.copy()
            final = self._create_final_chart(df_list, task_type)
        else:
            df = self.stats_for_plot.copy()
            final = self._create_final_chart(df, task_type)
        self._display_or_save_chart(final, save_path)

    def _create_final_chart(
        self, df: pd.DataFrame | dict[str, pd.DataFrame], task_type: str
    ) -> Any:
        """Create the final chart based on dataset type."""
        import altair as alt

        seq_col = "sequence"
        label_col = "labels"
        split_charts = []

        if isinstance(self.stats, dict):
            for split_name, _ in self.stats.items():
                stats = df[split_name] if isinstance(df, dict) else df
                chart = self._per_split_charts(
                    stats, task_type, seq_col, label_col
                ).properties(title=split_name)
                split_charts.append(chart)
            return alt.hconcat(*split_charts).properties(
                title="Dataset splits"
            )
        else:
            return self._per_split_charts(
                df, task_type, seq_col, label_col
            ).properties(title="Full dataset")

    def _display_or_save_chart(
        self, final: Any, save_path: str | None
    ) -> None:
        """Display or save the final chart."""
        if save_path:
            final.save(save_path)
        else:
            final.show()
            print("Successfully plotted dataset statistics.")

    def _parse_multi_labels(self, series: pd.Series) -> pd.DataFrame:
        """Split semicolon-separated labels in a Series into a dataframe of
        columns."""
        rows = []
        maxlen = 0
        for v in series.fillna(""):
            if v == "":
                parts = []
            else:
                parts = [p.strip() for p in str(v).split(";")]
            rows.append(parts)
            if len(parts) > maxlen:
                maxlen = len(parts)
        cols = [f"label_{i}" for i in range(maxlen)]
        parsed = [r + [""] * (maxlen - len(r)) for r in rows]
        df = pd.DataFrame(parsed, columns=cols)

        # try convert numeric types
        for c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("", np.nan))
        return df

    def _classification_plots(
        self, df: pd.DataFrame, label_col: str, seq_col: str
    ) -> tuple:
        """Build histogram of seq lengths colorized by label and GC boxplot
        grouped by label."""
        import altair as alt

        df = df.copy()
        df["label_str"] = df[label_col].astype(str)
        df["seq_len"] = df[seq_col].fillna("").astype(str).str.len()
        df["gc"] = df[seq_col].fillna("").astype(str).map(calc_gc_content)

        hist = (
            alt.Chart(df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "seq_len:Q",
                    bin=alt.Bin(maxbins=60),
                    title="Sequence length",
                ),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color(
                    "label_str:N", legend=alt.Legend(title="Labels")
                ),
                tooltip=[alt.Tooltip("seq_len:Q"), alt.Tooltip("count():Q")],
            )
            .properties(width=300, height=240)
        )

        box = (
            alt.Chart(df)
            .mark_boxplot(size=20)
            .encode(
                x=alt.X("label_str:N", title="Label"),
                y=alt.Y("gc:Q", title="GC content"),
                color=alt.Color("label_str:N", legend=None),
                tooltip=[
                    alt.Tooltip("label_str:N"),
                    alt.Tooltip("gc:Q"),
                ],
            )
            .properties(width=300, height=240)
        )
        return hist, box

    def _regression_plots(
        self, df: pd.DataFrame, label_col: str, seq_col: str
    ) -> tuple:
        """Build histogram of seq lengths (ungrouped) and GC scatter (GC vs
        label value)."""
        import altair as alt

        df = df.copy()
        df["seq_len"] = df[seq_col].fillna("").astype(str).str.len()
        df["gc"] = df[seq_col].fillna("").astype(str).map(calc_gc_content)

        hist = (
            alt.Chart(df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "seq_len:Q",
                    bin=alt.Bin(maxbins=60),
                    title="Sequence length",
                ),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("seq_len:Q"), alt.Tooltip("count():Q")],
            )
            .properties(width=300, height=240)
        )

        df["label_val"] = pd.to_numeric(df[label_col], errors="coerce")
        scatter = (
            alt.Chart(df)
            .mark_point()
            .encode(
                x=alt.X("gc:Q", title="GC content"),
                y=alt.Y("label_val:Q", title="Label value"),
                tooltip=[
                    alt.Tooltip("seq_len:Q"),
                    alt.Tooltip("gc:Q"),
                    alt.Tooltip("label_val:Q"),
                ],
            )
            .properties(width=300, height=240)
        )
        return hist, scatter

    def _per_split_charts(
        self, df: pd.DataFrame, data_type: str, seq_col: str, label_col: str
    ) -> Any:
        """Return a combined Altair chart for a single split based on
        data_type."""
        import altair as alt

        if data_type == "classification":
            hist, box = self._classification_plots(df, label_col, seq_col)
            return alt.hconcat(hist, box).properties(
                title="Classification stats"
            )

        if data_type == "regression":
            hist, scatter = self._regression_plots(df, label_col, seq_col)
            return alt.hconcat(hist, scatter).properties(
                title="Regression stats"
            )

        if data_type in ("multi-classification", "multi-regression"):
            return self._create_multi_target_charts(
                df, data_type, seq_col, label_col
            )

        raise ValueError(f"Unknown data_type: {data_type}")

    def _create_multi_target_charts(
        self, df: pd.DataFrame, data_type: str, seq_col: str, label_col: str
    ) -> Any:
        """Create charts for multi-target datasets."""
        import altair as alt

        lbls_df = self._parse_multi_labels(df[label_col])
        per_subcharts = []

        for c in lbls_df.columns:
            subdf = df.copy()
            subdf[c] = lbls_df[c]

            if data_type == "multi-classification":
                pair = self._create_multi_classification_chart(
                    subdf, c, seq_col
                )
            else:  # multi-regression
                pair = self._create_multi_regression_chart(subdf, c, seq_col)
            per_subcharts.append(pair)

        return alt.hconcat(*per_subcharts).properties(
            title="Multi-target stats"
        )

    def _create_multi_classification_chart(
        self, subdf: pd.DataFrame, c: str, seq_col: str
    ) -> Any:
        """Create chart for multi-classification sublabel."""
        import altair as alt

        subdf_for_plot = subdf.copy()
        subdf_for_plot["labels_for_plot"] = (
            subdf[c].astype("Int64").astype(str)
        )
        subdf_for_plot["seq_len"] = (
            subdf_for_plot[seq_col].fillna("").astype(str).str.len()
        )
        subdf_for_plot["gc"] = (
            subdf_for_plot[seq_col].fillna("").astype(str).map(calc_gc_content)
        )

        hist = (
            alt.Chart(subdf_for_plot)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "seq_len:Q",
                    bin=alt.Bin(maxbins=50),
                    title="Sequence length",
                ),
                y="count():Q",
                color=alt.Color("labels_for_plot:N", title=f"{c}"),
            )
            .properties(width=260, height=200)
        )

        box = (
            alt.Chart(subdf_for_plot)
            .mark_boxplot(size=20)
            .encode(
                x=alt.X("labels_for_plot:N", title=f"{c}"),
                y=alt.Y("gc:Q", title="GC content"),
                color=alt.Color("labels_for_plot:N", legend=None),
            )
            .properties(width=260, height=200)
        )
        return alt.vconcat(hist, box).properties(title=f"Sub-label {c}")

    def _create_multi_regression_chart(
        self, subdf: pd.DataFrame, c: str, seq_col: str
    ) -> Any:
        """Create chart for multi-regression subtarget."""
        import altair as alt

        subdf_for_plot = subdf.copy()
        subdf_for_plot["label_val"] = pd.to_numeric(subdf[c], errors="coerce")
        subdf_for_plot["seq_len"] = (
            subdf_for_plot[seq_col].fillna("").astype(str).str.len()
        )
        subdf_for_plot["gc"] = (
            subdf_for_plot[seq_col].fillna("").astype(str).map(calc_gc_content)
        )

        hist = (
            alt.Chart(subdf_for_plot)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "seq_len:Q",
                    bin=alt.Bin(maxbins=50),
                    title="Sequence length",
                ),
                y="count():Q",
            )
            .properties(width=260, height=200)
        )

        scatter = (
            alt.Chart(subdf_for_plot)
            .mark_point()
            .encode(
                x=alt.X("gc:Q", title="GC content"),
                y=alt.Y("label_val:Q", title="Label value"),
                tooltip=[
                    alt.Tooltip("seq_len:Q"),
                    alt.Tooltip("gc:Q"),
                    alt.Tooltip("label_val:Q"),
                ],
            )
            .properties(width=260, height=200)
        )
        return alt.vconcat(hist, scatter).properties(title=f"Sub-target {c}")


def show_preset_dataset() -> dict:
    """Show all preset datasets available in Hugging Face or ModelScope.

    Returns:
        A dictionary containing dataset names and their descriptions
    """
    from .dataset_auto import PRESET_DATASETS

    return PRESET_DATASETS


def load_preset_dataset(
    dataset_name: str, task: str | None = None
) -> "DNADataset":
    """Load a preset dataset from Hugging Face or ModelScope.

    Args:
        dataset_name: Name of the dataset
        task: Task directory in a dataset

    Returns:
        An instance wrapping a datasets.Dataset

    Raises:
        ValueError: If dataset is not found in preset datasets
    """
    from .dataset_auto import PRESET_DATASETS

    ds_info = _get_dataset_info(dataset_name, PRESET_DATASETS)
    ds = _load_dataset_from_modelscope(ds_info, task)
    ds = _standardize_column_names(ds)
    return _create_dna_dataset(ds, ds_info)


def _get_dataset_info(
    dataset_name: str, preset_datasets: dict
) -> dict[Any, Any]:
    """Get dataset information from preset datasets."""
    if dataset_name not in preset_datasets:
        raise ValueError(
            f"Dataset {dataset_name} not found in preset datasets."
        )
    return dict(preset_datasets[dataset_name])


def _load_dataset_from_modelscope(ds_info: dict, task: str | None) -> Any:
    """Load dataset from ModelScope."""
    from modelscope import MsDataset

    actual_dataset_name = ds_info["name"]
    print(f"Loading dataset: {actual_dataset_name} ...")
    print(task)
    if task and task in ds_info["tasks"]:
        return MsDataset.load(actual_dataset_name, data_dir=task)
    else:
        return MsDataset.load(actual_dataset_name)


def _standardize_column_names(ds: Any) -> Any:
    """Standardize column names in the dataset."""
    seq_cols = ["s", "seq", "sequence", "sequences"]
    label_cols = ["l", "label", "labels", "target", "targets"]
    seq_col = "sequence"
    label_col = "labels"

    if isinstance(ds, DatasetDict):
        return _standardize_datasetdict_columns(
            ds, seq_cols, label_cols, seq_col, label_col
        )
    else:
        return _standardize_single_dataset_columns(
            ds, seq_cols, label_cols, seq_col, label_col
        )


def _standardize_datasetdict_columns(
    ds: Any, seq_cols: list, label_cols: list, seq_col: str, label_col: str
) -> Any:
    """Standardize columns for DatasetDict."""
    for dt in ds:
        seq_col, label_col = _find_column_names(ds[dt], seq_cols, label_cols)
        if seq_col != "sequence":
            ds[dt] = ds[dt].rename_column(seq_col, "sequence")
        if label_col != "labels":
            ds[dt] = ds[dt].rename_column(label_col, "labels")
    return ds


def _standardize_single_dataset_columns(
    ds: Any, seq_cols: list, label_cols: list, seq_col: str, label_col: str
) -> Any:
    """Standardize columns for single dataset."""
    seq_col, label_col = _find_column_names(ds, seq_cols, label_cols)
    if seq_col != "sequence":
        ds = ds.rename_column(seq_col, "sequence")
    if label_col != "labels":
        ds = ds.rename_column(label_col, "labels")
    return ds


def _find_column_names(
    dataset: Any, seq_cols: list, label_cols: list
) -> tuple[str, str]:
    """Find appropriate column names for sequence and labels."""
    seq_col = "sequence"
    label_col = "labels"

    for s in seq_cols:
        if s in dataset.column_names:
            seq_col = s
            break
    for label_name in label_cols:
        if label_name in dataset.column_names:
            label_col = label_name
            break
    return seq_col, label_col


def _create_dna_dataset(ds: Any, ds_info: dict) -> "DNADataset":
    """Create DNADataset instance with proper configuration."""
    dna_ds = DNADataset(ds, tokenizer=None, max_length=1024)
    dna_ds.sep = str(ds_info.get("separator", ","))
    dna_ds.multi_label_sep = str(ds_info.get("multi_separator", ";"))
    return dna_ds


# Example usage:
"""
from dnallm import DNADataset
from dnallm.datahandling import show_preset_dataset, load_preset_dataset

# Show available preset datasets
show_preset_dataset()

# Load a specific dataset
ds = load_preset_dataset(
    "nucleotide_transformer_downstream_tasks", task="enhancers_types"
)

# Get dataset statistics
ds.statistics()

# Plot dataset statistics
ds.plot_statistics()
"""
