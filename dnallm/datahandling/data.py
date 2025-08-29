"""DNA Dataset handling and processing utilities.

This module provides comprehensive tools for loading, processing, and managing DNA sequence datasets.
It supports various file formats, data augmentation techniques, and statistical analysis.
"""

import os
import random
from typing import Union, Optional, Callable
from scipy import stats
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from ..utils.sequence import check_sequence, calc_gc_content, reverse_complement, random_generate_sequences


class DNADataset:
    """A comprehensive wrapper for DNA sequence datasets with advanced processing capabilities.
    
    This class provides methods for loading DNA datasets from various sources (local files,
    Hugging Face Hub, ModelScope), encoding sequences with tokenizers, data augmentation,
    statistical analysis, and more.
    
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
    
    def __init__(self, ds: Union[Dataset, DatasetDict], tokenizer: Optional[PreTrainedTokenizerBase] = None, max_length: int = 512) -> None:
        """Initialize a DNADataset.
        
        Args:
            ds: A Hugging Face Dataset containing at least 'sequence' and 'label' fields
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
        self.sep = None
        self.multi_label_sep = None
        self.data_type = None
        self.stats = None
        self.stats_for_plot = None
        self.__data_type__()  # Determine the data type of the dataset

    @classmethod
    def load_local_data(cls, file_paths: Union[str, list, dict], seq_col: str = "sequence", label_col: str = "labels",
                        sep: Optional[str] = None, fasta_sep: str = "|",
                        multi_label_sep: Optional[str] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None, max_length: int = 512) -> 'DNADataset':
        """Load DNA sequence datasets from one or multiple local files.
        
        Supports input formats: csv, tsv, json, parquet, arrow, dict, fasta, txt.
        
        Args:
            file_paths: Single dataset: Provide one file path (e.g., "data.csv").
                       Pre-split datasets: Provide a dict like {"train": "train.csv", "test": "test.csv"}
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
        if isinstance(file_paths, dict):  # Handling multiple files (pre-split datasets)
            ds_dict = {}
            for split, path in file_paths.items():
                ds_dict[split] = cls._load_single_data(path, seq_col, label_col, sep, fasta_sep, multi_label_sep)
            dataset = DatasetDict(ds_dict)
        else:  # Handling a single file
            dataset = cls._load_single_data(file_paths, seq_col, label_col, sep, fasta_sep, multi_label_sep)
        dataset.stats = None  # Initialize stats as None

        return cls(dataset, tokenizer=tokenizer, max_length=max_length)

    @classmethod
    def _load_single_data(cls, file_path: Union[str, list], seq_col: str = "sequence", label_col: str = "labels",
                          sep: Optional[str] = None, fasta_sep: str = "|",
                          multi_label_sep: Optional[str] = None) -> Dataset:
        """Load DNA data (sequences and labels) from a local file.

        Supported file types: 
          - For structured formats (CSV, TSV, JSON, Parquet, Arrow, dict), uses load_dataset from datasets.
          - For FASTA and TXT, uses custom parsing.

        Args:
            file_path: For most file types, a path (or pattern) to the file(s). For 'dict', a dictionary.
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
        if isinstance(file_path, list):
            file_path = [os.path.expanduser(fpath) for fpath in file_path]
            file_type = os.path.basename(file_path[0]).split(".")[-1].lower()
        else:
            file_path = os.path.expanduser(file_path)
            file_type = os.path.basename(file_path).split(".")[-1].lower()
        # Define data type
        default_types = ["csv", "tsv", "json", "parquet", "arrow"]
        dict_types = ["pkl", "pickle", "dict"]
        fasta_types = ["fa", "fna", "fas", "fasta"]
        # Check if the file contains a header
        if file_type in ["csv", "tsv", "txt"]:
            if file_type == "csv":
                sep = sep if sep else ","
            with open(file_path, "r") as f:
                header = f.readline().strip()
                if not header or (seq_col not in header and label_col not in header):
                    file_type = "txt"  # Treat as TXT if no header found
        # For structured formats that load via datasets.load_dataset
        if file_type in default_types:
            if file_type in ["csv", "tsv"]:
                sep = sep or ("," if file_type == "csv" else "\t")
                ds = load_dataset("csv", data_files=file_path, split="train", delimiter=sep)
            elif file_type == "json":
                ds = load_dataset("json", data_files=file_path, split="train")
            elif file_type in ["parquet", "arrow"]:
                ds = load_dataset(file_type, data_files=file_path, split="train")
            # Rename columns if needed
            if seq_col != "sequence":
                ds = ds.rename_column(seq_col, "sequence")
            if label_col != "labels":
                ds = ds.rename_column(label_col, "labels")
        elif file_type in dict_types:
            # Here, file_path is assumed to be a dictionary.
            import pickle
            data = pickle.load(open(file_path, 'rb'))
            ds = Dataset.from_dict(data)
            if seq_col != "sequence" or label_col != "labels":
                if seq_col in ds.column_names:
                    if "sequence" not in ds.features:
                        ds = ds.rename_column(seq_col, "sequence")
                if label_col in ds.column_names:
                    if "labels" not in ds.features:
                        ds = ds.rename_column(label_col, "labels")
        elif file_type in fasta_types:
            sequences, labels = [], []
            with open(file_path, "r") as f:
                seq = ""
                lab = None
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq and lab is not None:
                            sequences.append(seq)
                            labels.append(lab)
                        # Extract label from header, handle different formats
                        header = line[1:].strip()
                        if fasta_sep in header:
                            lab = header.split(fasta_sep)[-1]
                        else:
                            # If no separator, try to extract label from end of header
                            lab = header
                        seq = ""
                    else:
                        seq += line.strip()
                if seq and lab is not None:
                    sequences.append(seq)
                    labels.append(lab)
            ds = Dataset.from_dict({"sequence": sequences, "labels": labels})
        elif file_type == "txt":
            # Assume each line contains a sequence and a label separated by whitespace or a custom sep.
            sequences, labels = [], []
            with open(file_path, "r") as f:
                for i,line in enumerate(f):
                    if i == 0:
                        # Contain header, use load_dataset with csv method
                        if seq_col in line and label_col in line:
                            ds = load_dataset("csv", data_files=file_path, split="train", delimiter=sep)
                            break
                    record = line.strip().split(sep) if sep else line.strip().split()
                    if len(record) >= 2:
                        sequences.append(record[0])
                        labels.append(record[1])
                    else:
                        continue
            ds = Dataset.from_dict({"sequence": sequences, "labels": labels})
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Convert string labels to appropriate types, with error handling
        def format_labels(example):
            labels = example['labels']
            if isinstance(labels, str):
                try:
                    if multi_label_sep and multi_label_sep in labels:
                        example['labels'] = [float(x) for x in labels.split(multi_label_sep)]
                    else:
                        # Try to convert to float first, then int if that fails
                        try:
                            example['labels'] = float(labels)
                        except ValueError:
                            # If it's not a number, keep as string
                            example['labels'] = labels
                except (ValueError, TypeError):
                    # If conversion fails, keep original value
                    example['labels'] = labels
            return example
        
        if 'labels' in ds.column_names:
            ds = ds.map(format_labels, desc="Format labels")
        
        # Return processed dataset
        return ds

    @classmethod
    def from_huggingface(cls, dataset_name: str,
                         seq_col: str = "sequence", label_col: str = "labels",
                         data_dir: Optional[str] = None,
                         tokenizer: Optional[PreTrainedTokenizerBase] = None, max_length: int = 512) -> 'DNADataset':
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
    def from_modelscope(cls, dataset_name: str,
                        seq_col: str = "sequence", label_col: str = "labels",
                        data_dir: Optional[str] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None, max_length: int = 512) -> 'DNADataset':
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

    def encode_sequences(self, padding: str = "max_length", return_tensors: str = "pt",
                         remove_unused_columns: bool = False,
                         uppercase: bool = False, lowercase: bool = False,
                         task: Optional[str] = 'SequenceClassification') -> None:
        """Encode all sequences using the provided tokenizer.
        
        The dataset is mapped to include tokenized fields along with the label,
        making it directly usable with Hugging Face Trainer.
        
        Args:
            padding: Padding strategy for sequences. Can be 'max_length' or 'longest'.
                    Use 'longest' to pad to the length of the longest sequence in case of memory outage
            return_tensors: Returned tensor types, can be 'pt', 'tf', 'np', or 'jax'
            remove_unused_columns: Whether to remove the original 'sequence' and 'label' columns
            uppercase: Whether to convert sequences to uppercase
            lowercase: Whether to convert sequences to lowercase
            task: Task type for the tokenizer. If not provided, defaults to 'SequenceClassification'
            
        Raises:
            ValueError: If tokenizer is not provided
        """
        if self.tokenizer:
            sp_token_map = self.tokenizer.special_tokens_map
            pad_token = sp_token_map['pad_token'] if 'pad_token' in sp_token_map else None
            pad_id = self.tokenizer.encode(pad_token)[-1] if pad_token else None
            cls_token = sp_token_map['cls_token'] if 'cls_token' in sp_token_map else None
            sep_token = sp_token_map['sep_token'] if 'sep_token' in sp_token_map else None
            eos_token = sp_token_map['eos_token'] if 'eos_token' in sp_token_map else None
            max_length = self.max_length
        else:
            raise ValueError("Tokenizer is required")
        def tokenize_for_sequence_classification(example):
            sequences = example["sequence"]
            if uppercase:
                sequences = [x.upper() for x in sequences]
            if lowercase:
                sequences = [x.lower() for x in sequences]
            tokenized = self.tokenizer(
                sequences,
                truncation=True,
                padding=padding,
                max_length=max_length
            )
            return tokenized
        def tokenize_for_token_classification(examples):
            
            tokenized_examples = {'sequence': [],
                                  'input_ids': [],
                                  # 'token_type_ids': [],
                                  'attention_mask': []}
            if 'labels' in examples:
                tokenized_examples['labels'] = []
            input_seqs = examples['sequence']
            if isinstance(input_seqs, str):
                input_seqs = input_seqs.split(self.multi_label_sep)
            for i, example_tokens in enumerate(input_seqs):
                all_ids = [x for x in self.tokenizer.encode(example_tokens, is_split_into_words=True)]
                if 'labels' in examples:
                    example_ner_tags = examples['labels'][i]
                else:
                    example_ner_tags = [0] * len(example_tokens)
                pad_len = max_length - len(all_ids)
                if pad_len >= 0:
                    all_masks = [1] * len(all_ids) + [0] * pad_len
                    all_ids = all_ids + [pad_id] * pad_len
                    if cls_token:
                        if sep_token:
                            example_tokens = [cls_token] + example_tokens + [sep_token] + [pad_token] * pad_len
                            example_ner_tags = [-100] + example_ner_tags + [-100] * (pad_len + 1)
                        elif eos_token:
                            example_tokens = [cls_token] + example_tokens + [eos_token] + [pad_token] * pad_len
                            example_ner_tags = [-100] + example_ner_tags + [-100] * (pad_len + 1)
                        else:
                            example_tokens = [cls_token] + example_tokens + [pad_token] * pad_len
                            example_ner_tags = [-100] + example_ner_tags + [-100] * pad_len
                    else:
                        example_tokens = example_tokens + [pad_token] * pad_len
                        example_ner_tags = example_ner_tags + [-100] * pad_len
                elif pad_len < 0:
                    all_ids = all_ids[:max_length]
                    all_masks = [1] * (max_length)
                    if cls_token:
                        if sep_token:
                            example_tokens = [cls_token] + example_tokens[:max_length - 2] + [sep_token]
                            example_ner_tags = [-100] + example_ner_tags[:max_length - 2] + [-100]
                        else:
                            example_tokens = [cls_token] + example_tokens[:max_length - 1]
                            example_ner_tags = [-100] + example_ner_tags[:max_length - 1]
                    else:
                        example_tokens = example_tokens[:max_length]
                        example_ner_tags = example_ner_tags[:max_length]
                tokenized_examples['sequence'].append(example_tokens)
                tokenized_examples['input_ids'].append(all_ids)
                # tokenized_examples['token_type_ids'].append([0] * max_length)
                tokenized_examples['attention_mask'].append(all_masks)
                if 'labels' in examples:
                    tokenized_examples['labels'].append(example_ner_tags)
            return BatchEncoding(tokenized_examples)
        # Judge the task type
        task = task.lower()
        if task in ['sequenceclassification', 'binary', 'multiclass', 'multilabel', 'regression']:
            self.dataset = self.dataset.map(tokenize_for_sequence_classification, batched=True, desc="Encoding inputs")
        elif task in ['tokenclassification', 'token', 'ner']:
            from transformers.tokenization_utils_base import BatchEncoding
            self.dataset = self.dataset.map(tokenize_for_token_classification, batched=True, desc="Encoding inputs")
        elif task in ['maskedlm', 'mlm', 'mask', 'embedding']:
            self.dataset = self.dataset.map(tokenize_for_sequence_classification, batched=True, desc="Encoding inputs")
        elif task in ['causallm', 'clm', 'causal', 'generation', 'embedding']:
            self.dataset = self.dataset.map(tokenize_for_sequence_classification, batched=True)
        else:
            self.dataset = self.dataset.map(tokenize_for_sequence_classification, batched=True, desc="Encoding inputs")
        if remove_unused_columns:
            used_cols = ['labels', 'input_ids', 'attention_mask']
            if isinstance(self.dataset, DatasetDict):
                for dt in self.dataset:
                    unused_cols = [f for f in self.dataset[dt].features if f not in used_cols]
                    self.dataset[dt] = self.dataset[dt].remove_columns(unused_cols)
            else:
                unused_cols = [f for f in self.dataset.features if f not in used_cols]
                self.dataset = self.dataset.remove_columns(unused_cols)
        if return_tensors == "tf":
            self.dataset.set_format(type="tensorflow")
        elif return_tensors == "jax":
            self.dataset.set_format(type="jax")
        elif return_tensors == "np":
            self.dataset.set_format(type="numpy")
        else:
            self.dataset.set_format(type="torch")
        self.dataset._is_encoded = True  # Mark the dataset as encoded

    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, seed: Optional[int] = None) -> None:
        """Split the dataset into train, test, and validation sets.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            val_size: Proportion of the dataset to include in the validation split
            seed: Random seed for reproducibility
        """
        # First, split off test+validation from training data
        split_result = self.dataset.train_test_split(test_size=test_size + val_size, seed=seed)
        train_ds = split_result['train']
        temp_ds = split_result['test']
        # Further split temp_ds into test and validation sets
        if val_size > 0:
            rel_val_size = val_size / (test_size + val_size)
            temp_split = temp_ds.train_test_split(test_size=rel_val_size, seed=seed)
            test_ds = temp_split['train']
            val_ds = temp_split['test']
            self.dataset = DatasetDict({'train': train_ds, 'test': test_ds, 'val': val_ds})
        else:
            self.dataset = DatasetDict({'train': train_ds, 'test': temp_ds})
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.dataset.shuffle(seed=seed)

    def validate_sequences(self, minl: int = 20, maxl: int = 6000, gc: tuple = (0, 1), valid_chars: str = "ACGTN") -> None:
        """Filter the dataset to keep sequences containing valid DNA bases or allowed length.
        
        Args:
            minl: Minimum length of the sequences
            maxl: Maximum length of the sequences
            gc: GC content range between 0 and 1
            valid_chars: Allowed characters in the sequences
        """
        self.dataset = self.dataset.filter(
            lambda example: check_sequence(example["sequence"], minl, maxl, gc, valid_chars)
        )

    def random_generate(self, minl: int, maxl: int = 0, samples: int = 1,
                              gc: tuple = (0, 1), N_ratio: float = 0.0,
                              padding_size: int = 0, seed: Optional[int] = None,
                              label_func: Optional[Callable] = None, append: bool = False) -> None:
        """Replace the current dataset with randomly generated DNA sequences.

        Args:
            minl: Minimum length of the sequences
            maxl: Maximum length of the sequences, default is the same as minl
            samples: Number of sequences to generate, default 1
            gc: GC content range, default (0,1)
            N_ratio: Include N base in the generated sequence, default 0.0
            padding_size: Padding size for sequence length, default 0
            seed: Random seed, default None
            label_func: A function that generates a label from a sequence
            append: Append the random generated data to the existing dataset or use the data as a dataset
        """
        def process(minl, maxl, number, gc, N_ratio, padding_size, seed, label_func):
            sequences = random_generate_sequences(minl=minl, maxl=maxl, samples=number,
                                                gc=gc, N_ratio=N_ratio,
                                                padding_size=padding_size, seed=seed)
            labels = []
            for seq in sequences:
                labels.append(label_func(seq) if label_func else 0)
            random_ds = Dataset.from_dict({"sequence": sequences, "labels": labels})
            return random_ds
        if append:
            if isinstance(self.dataset, DatasetDict):
                for dt in self.dataset:
                    number = round(samples * len(self.dataset[dt]) / sum(self.__len__().values()))
                    random_ds = process(minl, maxl, number, gc, N_ratio, padding_size, seed, label_func)
                    self.dataset[dt] = concatenate_datasets([self.dataset[dt], random_ds])
            else:
                random_ds = process(minl, maxl, samples, gc, N_ratio, padding_size, seed, label_func)
                self.dataset = concatenate_datasets([self.dataset, random_ds])
        else:
            self.dataset = process(minl, maxl, samples, gc, N_ratio, padding_size, seed, label_func)

    def process_missing_data(self) -> None:
        """Filter out samples with missing or empty sequences or labels."""
        def non_missing(example):
            return example["sequence"] and example["labels"] is not None and example["sequence"].strip() != ""
        self.dataset = self.dataset.filter(non_missing)

    def raw_reverse_complement(self, ratio: float = 0.5, seed: Optional[int] = None) -> None:
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

    def augment_reverse_complement(self, reverse: bool = True, complement: bool = True) -> None:
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
            # Build a new dataset where the reverse complement becomes the 'sequence'
            rc_ds = ds_with_rc.map(lambda ex: {"sequence": ex["rc_sequence"], "labels": ex["labels"]}, desc="Data augment")
            ds = concatenate_datasets([ds, rc_ds])
            ds.remove_columns(["rc_sequence"])
            return ds
        if isinstance(self.dataset, DatasetDict):
            for dt in self.dataset:
                self.dataset[dt] = process(self.dataset[dt], reverse, complement)
        else:
            self.dataset = process(self.dataset, reverse, complement)

    def concat_reverse_complement(self, reverse: bool = True, complement: bool = True, sep: str = "") -> None:
        """Augment each sample by concatenating the sequence with its reverse complement.
        
        Args:
            reverse: Whether to do reverse
            complement: Whether to do complement
            sep: Separator between the original and reverse complement sequences
        """
        def process(ds, reverse, complement, sep):
            def concat_fn(example):
                rc = reverse_complement(example["sequence"], reverse=reverse, complement=complement)
                example["sequence"] = example["sequence"] + sep + rc
                return example
            ds = ds.map(concat_fn, desc="Data augment")
            return ds
        if isinstance(self.dataset, DatasetDict):
            for dt in self.dataset:
                self.dataset[dt] = process(self.dataset[dt], reverse, complement, sep)
        else:
            self.dataset = process(self.dataset, reverse, complement, sep)
    
    def sampling(self, ratio: float = 1.0, seed: Optional[int] = None, overwrite: bool = False) -> 'DNADataset':
        """Randomly sample a fraction of the dataset.

        Args:
            ratio: Fraction of the dataset to sample. Default is 1.0 (no sampling)
            seed: Random seed for reproducibility
            overwrite: Whether to overwrite the original dataset with the sampled one

        Returns:
            A DNADataset object with sampled data
        """
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")
        
        dataset = self.dataset
        if isinstance(dataset, DatasetDict):
            for dt in dataset.keys():
                random.seed(seed)
                random_idx = random.sample(range(len(dataset[dt])), int(len(dataset[dt]) * ratio))
                dataset[dt] = dataset[dt].select(random_idx)
        else:
            random_idx = random.sample(range(len(dataset)), int(len(dataset) * ratio))
            dataset = dataset.select(random_idx)
        
        if overwrite:
            self.dataset = dataset
            return self
        else:
            # Create a new DNADataset object with the sampled data
            return DNADataset(dataset, self.tokenizer, self.max_length)
    
    def head(self, head: int = 10, show: bool = False) -> Optional[dict]:
        """Fetch the head n data from the dataset.
        
        Args:
            head: Number of samples to fetch
            show: Whether to print the data or return it

        Returns:
            A dictionary containing the first n samples if show=False, otherwise None
        """
        import pprint
        def format_convert(data):
            df = {}
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
                    return df
        else:
            data = dataset[dt][:head]
            if show:
                pprint.pp(format_convert(data))
            else:
                return data
    
    def show(self, head: int = 10) -> None:
        """Display the dataset.
        
        Args:
            head: Number of samples to display
        """
        self.head(head=head, show=True)            

    def iter_batches(self, batch_size: int):
        """Generator that yields batches of examples from the dataset.
        
        Args:
            batch_size: Size of each batch

        Yields:
            A batch of examples
            
        Raises:
            ValueError: If dataset is a DatasetDict
        """
        if isinstance(self.dataset, DatasetDict):
            raise ValueError("Dataset is a DatasetDict Object, please use `DNADataset.dataset[datatype].iter_batches(batch_size)` instead.")
        else:
            for i in range(0, len(self.dataset), batch_size):
                yield self.dataset[i: i + batch_size]

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
    
    def get_split_lengths(self) -> Optional[dict]:
        """Get lengths of individual splits for DatasetDict.
        
        Returns:
            Dictionary of split names and their lengths, or None for single dataset
        """
        if isinstance(self.dataset, DatasetDict):
            return {dt: len(self.dataset[dt]) for dt in self.dataset}
        else:
            return None

    def __getitem__(self, idx: int):
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            The item at the specified index
            
        Raises:
            ValueError: If dataset is a DatasetDict
        """
        if isinstance(self.dataset, DatasetDict):
            raise ValueError("Dataset is a DatasetDict Object, please use `DNADataset.dataset[datatype].__getitem__(idx)` instead.")
        else:
            return self.dataset[idx]

    def __data_type__(self) -> None:
        """Get the data type of the dataset (classification, regression, etc.).

        This method analyzes the labels to determine if the dataset is for:
        - classification (integer or string labels)
        - regression (float labels)
        - multi-label (multiple labels per sample)
        - multi-regression (multiple float values per sample)
        """
        if isinstance(self.dataset, DatasetDict):
            keys = list(self.dataset.keys())
            if not keys:
                raise ValueError("DatasetDict is empty.")
            if "labels" in self.dataset[keys[0]].column_names:
                labels = self.dataset[keys[0]]["labels"]
            else:
                labels = None
        else:
            if "labels" in self.dataset.column_names:
                labels = self.dataset["labels"]
            else:
                labels = None
        
        # Handle empty datasets
        if labels is None:
            self.data_type = "unknown"
            return
        
        # Check if labels is empty
        try:
            if hasattr(labels, '__len__'):
                if len(labels) == 0:
                    self.data_type = "unknown"
                    return
            else:
                self.data_type = "unknown"
                return
        except (TypeError, AttributeError):
            self.data_type = "unknown"
            return
        
        # Check if labels is a list-like object
        if hasattr(labels, '__getitem__'):
            try:
                first_label = labels[0]
            except IndexError:
                self.data_type = "unknown"
                return
        else:
            self.data_type = "unknown"
            return
        
        if isinstance(first_label, str):
            if self.multi_label_sep and self.multi_label_sep in str(first_label):
                multi_labels = str(first_label).split(self.multi_label_sep)
                if '.' in multi_labels[0]:
                    self.data_type = "multi_regression"
                else:
                    self.data_type = "multi_label"
            else:
                if '.' in str(first_label):
                    self.data_type = "regression"
                else:
                    self.data_type = "classification"
        else:
            if isinstance(first_label, int):
                self.data_type = "classification"
            else:
                self.data_type = "regression"
    
    def statistics(self) -> dict:
        """Get statistics of the dataset.
        
        Includes number of samples, sequence length (min, max, average, median), 
        label distribution, GC content (by labels), nucleotide composition (by labels).
        
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

            if is_dataset:
                df = dataset.to_pandas()
            elif isinstance(dataset, pd.DataFrame):
                df = dataset.copy()
            else:
                raise ValueError('prepare_dataframe expects a datasets.Dataset or pandas.DataFrame')
            return df

        def compute_basic_stats(df: pd.DataFrame, seq_col: str = 'sequence') -> dict:
            """Compute number of samples and sequence length statistics."""
            seqs = df[seq_col].fillna('').astype(str)
            lens = seqs.str.len()
            return {
                'n_samples': int(len(lens)),
                'min_len': int(lens.min()) if len(lens) > 0 else 0,
                'max_len': int(lens.max()) if len(lens) > 0 else 0,
                'mean_len': float(lens.mean()) if len(lens) > 0 else float('nan'),
                'median_len': float(lens.median()) if len(lens) > 0 else float('nan'),
            }

        stats = {}
        seq_col = "sequence"
        label_col = "labels"
        if isinstance(self.dataset, DatasetDict):
            for split_name, split_ds in self.dataset.items():
                df = prepare_dataframe(split_ds)
                data_type = self.data_type
                basic = compute_basic_stats(df, seq_col)
                stats[split_name] = {'data_type': data_type, **basic}
        else:
            df = prepare_dataframe(self.dataset)
            data_type = self.data_type
            basic = compute_basic_stats(df, seq_col)
            stats['full'] = {'data_type': data_type, **basic}
        
        self.stats = stats  # Store stats in the instance for later use
        self.stats_for_plot = df

        return stats

    def plot_statistics(self, save_path: Optional[str] = None) -> None:
        """Plot statistics of the dataset.
        
        Includes sequence length distribution (histogram), 
        GC content distribution (box plot) for each sequence.
        If dataset is a DatasetDict, length plots and GC content plots from different datasets will be 
        concatenated into a single chart, respectively.
        Sequence length distribution is shown as a histogram, with min and max lengths for its' limit.

        Args:
            save_path: Path to save the plots. If None, plots will be shown interactively
            
        Raises:
            ValueError: If statistics have not been computed yet
        """

        import altair as alt
        from typing import Tuple
        alt.data_transformers.enable("vegafusion")

        def parse_multi_labels(series: pd.Series) -> pd.DataFrame:
            """Split semicolon-separated labels in a Series into a dataframe of columns.

            Example: '0;1;1' -> columns ['label_0','label_1','label_2']
            """
            rows = []
            maxlen = 0
            for v in series.fillna(''):
                if v == '':
                    parts = []
                else:
                    parts = [p.strip() for p in str(v).split(';')]
                rows.append(parts)
                if len(parts) > maxlen:
                    maxlen = len(parts)
            cols = [f'label_{i}' for i in range(maxlen)]
            parsed = [r + [''] * (maxlen - len(r)) for r in rows]
            df = pd.DataFrame(parsed, columns=cols)

            # try convert numeric types
            for c in df.columns:
                df[c] = pd.to_numeric(df[c].replace('', np.nan))
            return df

        def classification_plots(df: pd.DataFrame, label_col: str = 'labels', seq_col: str = 'sequence') -> alt.Chart:
            """Build histogram of seq lengths colorized by label and GC boxplot grouped by label.

            For multi-label (where label column contains semicolon), this function expects the
            caller to have split and called per-sublabel as necessary.
            """
            # ensure label is a categorical column
            df = df.copy()
            df['label_str'] = df[label_col].astype(str)
            df['seq_len'] = df[seq_col].fillna('').astype(str).str.len()
            df['gc'] = df[seq_col].fillna('').astype(str).map(calc_gc_content)

            # histogram: seq length, colored by label
            hist = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X('seq_len:Q', bin=alt.Bin(maxbins=60), title='Sequence length'),
                y=alt.Y('count():Q', title='Count'),
                color=alt.Color('label_str:N', title='Label')
            ).properties(width=300, height=240)

            # GC boxplot grouped by label
            box = alt.Chart(df).mark_boxplot(size=20).encode(
                x=alt.X('label_str:N', title='Label'),
                y=alt.Y('gc:Q', title='GC content'),
                color=alt.Color('label_str:N', legend=None)
            ).properties(width=300, height=240)

            return hist, box

        def regression_plots(df: pd.DataFrame, label_col: str = 'labels', seq_col: str = 'sequence') -> Tuple[alt.Chart, alt.Chart]:
            """Build histogram of seq lengths (ungrouped) and GC scatter (GC vs label value).

            For multi-regression, caller should split and call per target.
            """
            df = df.copy()
            df['seq_len'] = df[seq_col].fillna('').astype(str).str.len()
            df['gc'] = df[seq_col].fillna('').astype(str).map(calc_gc_content)

            hist = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X('seq_len:Q', bin=alt.Bin(maxbins=60), title='Sequence length'),
                y=alt.Y('count():Q', title='Count')
            ).properties(width=300, height=240)

            # ensure numeric label
            df['label_val'] = pd.to_numeric(df[label_col], errors='coerce')
            scatter = alt.Chart(df).mark_point().encode(
                x=alt.X('gc:Q', title='GC content'),
                y=alt.Y('label_val:Q', title='Label value'),
                tooltip=[alt.Tooltip('seq_len:Q'), alt.Tooltip('gc:Q'), alt.Tooltip('label_val:Q')]
            ).properties(width=300, height=240)

            return hist, scatter

        def per_split_charts(df: pd.DataFrame, data_type: str, seq_col: str, label_col: str) -> alt.Chart:
            """Return a combined Altair chart for a single split (DataFrame) based on data_type.

            Behavior aligned with user requirement:
            - For 'classification' or 'regression' (single-label): seq_len and GC plots are concatenated horizontally.
            - For 'multi-classification' and 'multi-regression': sublabels' results are concatenated horizontally
            and the pair (seq_len, GC) for each sublabel are concatenated vertically.
            """
            if data_type == 'classification':
                hist, box = classification_plots(df, label_col, seq_col)
                combined = alt.hconcat(hist, box)
                return combined.properties(title='Classification stats')

            if data_type == 'regression':
                hist, scatter = regression_plots(df, label_col, seq_col)
                combined = alt.hconcat(hist, scatter)
                return combined.properties(title='Regression stats')

            if data_type in ('multi-classification', 'multi-regression'):
                # split labels into subcolumns
                lbls_df = parse_multi_labels(df[label_col])
                per_subcharts = []
                for c in lbls_df.columns:
                    subdf = df.copy()
                    subdf[c] = lbls_df[c]
                    # drop nan labels (optional) but keep sequences
                    if data_type == 'multi-classification':
                        # treat each sublabel like single classification
                        subdf_for_plot = subdf.copy()
                        subdf_for_plot['labels_for_plot'] = subdf[c].astype('Int64').astype(str)
                        hist = alt.Chart(subdf_for_plot).mark_bar(opacity=0.7).encode(
                            x=alt.X('seq_len:Q', bin=alt.Bin(maxbins=50), title='Sequence length'),
                            y='count():Q',
                            color=alt.Color('labels_for_plot:N', title=f'{c}')
                        ).properties(width=260, height=200)

                        box = alt.Chart(subdf_for_plot).mark_boxplot(size=20).encode(
                            x=alt.X('labels_for_plot:N', title=f'{c}'),
                            y=alt.Y('gc:Q', title='GC content'),
                            color=alt.Color('labels_for_plot:N', legend=None)
                        ).properties(width=260, height=200)
                        pair = alt.vconcat(hist, box).properties(title=f'Sub-label {c}')
                        per_subcharts.append(pair)

                    else:  # multi-regression
                        subdf_for_plot = subdf.copy()
                        subdf_for_plot['label_val'] = pd.to_numeric(subdf[c], errors='coerce')
                        hist = alt.Chart(subdf_for_plot).mark_bar(opacity=0.7).encode(
                            x=alt.X('seq_len:Q', bin=alt.Bin(maxbins=50), title='Sequence length'),
                            y='count():Q'
                        ).properties(width=260, height=200)

                        scatter = alt.Chart(subdf_for_plot).mark_point().encode(
                            x=alt.X('gc:Q', title='GC content'),
                            y=alt.Y('label_val:Q', title='Label value'),
                            tooltip=[alt.Tooltip('seq_len:Q'), alt.Tooltip('gc:Q'), alt.Tooltip('label_val:Q')]
                        ).properties(width=260, height=200)
                        pair = alt.vconcat(hist, scatter).properties(title=f'Sub-target {c}')
                        per_subcharts.append(pair)

                # concat all subcharts horizontally
                combined = alt.hconcat(*per_subcharts)
                return combined.properties(title='Multi-target stats')

            raise ValueError(f'Unknown data_type: {data_type}')

        if self.stats is None or self.stats_for_plot is None:
            raise ValueError("Statistics have not been computed yet. Please call `statistics()` method first.")
        task_type = self.data_type
        df = self.stats_for_plot.copy()
        seq_col = "sequence"
        label_col = "labels"
        split_charts = []
        if isinstance(self.stats, dict):
            for split_name, split_stats in self.stats.items():
                chart = per_split_charts(df, task_type, seq_col, label_col).properties(title=split_name)
                split_charts.append(chart)
            # concatenate splits horizontally
            final = alt.hconcat(*split_charts).properties(title='Dataset splits')
        else:
            final = per_split_charts(df, task_type, seq_col, label_col).properties(title='Full dataset')

        if save_path:
            final.save(save_path)
        else:
            final.show()
            print("Successfully plotted dataset statistics.")


def show_preset_dataset() -> dict:
    """Show all preset datasets available in Hugging Face or ModelScope.

    Returns:
        A dictionary containing dataset names and their descriptions
    """
    from .dataset_auto import PRESET_DATASETS
    return PRESET_DATASETS


def load_preset_dataset(dataset_name: str, task: Optional[str] = None) -> 'DNADataset':
    """Load a preset dataset from Hugging Face or ModelScope.

    Args:
        dataset_name: Name of the dataset
        task: Task directory in a dataset

    Returns:
        An instance wrapping a datasets.Dataset
        
    Raises:
        ValueError: If dataset is not found in preset datasets
    """
    from modelscope import MsDataset
    from .dataset_auto import PRESET_DATASETS

    if dataset_name in PRESET_DATASETS:
        ds_info = PRESET_DATASETS[dataset_name]
        dataset_name = ds_info["name"]
        if task in ds_info["tasks"]:
            ds = MsDataset.load(dataset_name, data_dir=task)
        else:
            ds = MsDataset.load(dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not found in preset datasets.")

    seq_cols = ["s", "seq", "sequence", "sequences"]
    label_cols = ["l", "label", "labels", "target", "targets"]
    seq_col = "sequence"
    label_col = "labels"
    if isinstance(ds, DatasetDict):
        # Check if the dataset is a DatasetDict
        for dt in ds:
            # Rename columns if necessary
            for s in seq_cols:
                if s in ds[dt].column_names:
                    seq_col = s
                    break
            for l in label_cols:
                if l in ds[dt].column_names:
                    label_col = l
                    break
            if seq_col != "sequence":
                ds[dt] = ds[dt].rename_column(seq_col, "sequence")
            if label_col != "labels":
                ds[dt] = ds[dt].rename_column(label_col, "labels")
    else:
        # If it's a single dataset, rename columns directly
        if seq_col != "sequence":
            ds = ds.rename_column(seq_col, "sequence")
        if label_col != "labels":
            ds = ds.rename_column(label_col, "labels")
    
    dna_ds = DNADataset(ds, tokenizer=None, max_length=1024)
    dna_ds.sep = ds_info.get("separator", ",")
    dna_ds.multi_label_sep = ds_info.get("multi_separator", ";")

    return dna_ds


# Example usage:
"""
from dnallm import DNADataset
from dnallm.datahandling import show_preset_dataset, load_preset_dataset

# Show available preset datasets
show_preset_dataset()

# Load a specific dataset
ds = load_preset_dataset("nucleotide_transformer_downstream_tasks", task="enhancers_types")

# Get dataset statistics
ds.statistics()

# Plot dataset statistics
ds.plot_statistics()
"""


