"""DNA Dataset handling and processing utilities.

This module provides comprehensive tools for loading, processing, and managing DNA sequence datasets.
It supports various file formats, data augmentation techniques, and statistical analysis.
"""

import os
import random
from typing import Union, Optional
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
    
    def __init__(self, ds: Union[Dataset, DatasetDict], tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512):
        """Initialize the DNADataset.
        
        Args:
            ds: A Hugging Face Dataset containing at least 'sequence' and 'label' fields
            tokenizer: A Hugging Face tokenizer for encoding sequences
            max_length: Maximum length for tokenization
        """
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
    def load_local_data(cls, file_paths, seq_col: str = "sequence", label_col: str = "labels",
                        sep: str = None, fasta_sep: str = "|",
                        multi_label_sep: Union[str, None] = None,
                        tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512) -> 'DNADataset':
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
            DNADataset: An instance wrapping a Dataset or DatasetDict
            
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
    def _load_single_data(cls, file_path, seq_col: str = "sequence", label_col: str = "labels",
                          sep: str = None, fasta_sep: str = "|",
                          multi_label_sep: Union[str, None] = None) -> Dataset:
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
            Dataset: A Hugging Face Dataset with 'sequence' and 'labels' columns
            
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
                        lab = line[1:].strip().split(fasta_sep)[-1]  # Assume label is separated by `fasta_sep` in the header
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
        # Convert string labels to integer
        def format_labels(example):
            labels = example['labels']
            if isinstance(labels, str):
                if multi_label_sep:
                    example['labels'] = [float(x) for x in labels.split(multi_label_sep)]
                else:
                    example['labels'] = float(labels) if '.' in labels else int(labels)
            return example
        if 'labels' in ds.column_names:
            ds = ds.map(format_labels, desc="Format labels")
        # Return processed dataset
        return ds

    @classmethod
    def from_huggingface(cls, dataset_name: str,
                         seq_col: str = "sequence", label_col: str = "labels",
                         data_dir: Union[str, None]=None,
                         tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512) -> 'DNADataset':
        """Load a dataset from the Hugging Face Hub.

        Args:
            dataset_name: Name of the dataset
            seq_col: Column name for the DNA sequence
            label_col: Column name for the label
            data_dir: Data directory in a dataset
            tokenizer: Tokenizer for sequence encoding
            max_length: Max token length

        Returns:
            DNADataset: An instance wrapping a datasets.Dataset
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
                        data_dir: Union[str, None]=None,
                        tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512) -> 'DNADataset':
        """Load a dataset from the ModelScope.

        Args:
            dataset_name: Name of the dataset
            seq_col: Column name for the DNA sequence
            label_col: Column name for the label
            data_dir: Data directory in a dataset
            tokenizer: Tokenizer for sequence encoding
            max_length: Max token length

        Returns:
            DNADataset: An instance wrapping a datasets.Dataset
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
                         uppercase: bool=False, lowercase: bool=False,
                         task: Optional[str] = 'SequenceClassification'):
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
            raise ValueError("Tokenizer not provided.")
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

    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, seed: int = None):
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
    
    def shuffle(self, seed: int = None):
        """Shuffle the dataset.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.dataset.shuffle(seed=seed)

    def validate_sequences(self, minl: int = 20, maxl: int = 6000, gc: tuple = (0,1), valid_chars: str = "ACGTN"):
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
                              gc: tuple = (0,1), N_ratio: float = 0.0,
                              padding_size: int = 0, seed: int = None,
                              label_func = None, append: bool = False):
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

    def process_missing_data(self):
        """Filter out samples with missing or empty sequences or labels."""
        def non_missing(example):
            return example["sequence"] and example["labels"] is not None and example["sequence"].strip() != ""
        self.dataset = self.dataset.filter(non_missing)

    def raw_reverse_complement(self, ratio: float = 0.5, seed: int = None):
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

    def augment_reverse_complement(self, reverse=True, complement=True):
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

    def concat_reverse_complement(self, reverse=True, complement=True, sep: str = ""):
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
    
    def sampling(self, ratio: float=1.0, seed: int = None, overwrite: bool=False) -> Union[Dataset, DatasetDict]:
        """Randomly sample a fraction of the dataset.

        Args:
            ratio: Fraction of the dataset to sample. Default is 1.0 (no sampling)
            seed: Random seed for reproducibility
            overwrite: Whether to overwrite the original dataset with the sampled one

        Returns:
            A sampled dataset if overwrite=False, otherwise None
        """
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
        else:
            return dataset
    
    def head(self, head: int=10, show: bool=False) -> Union[dict, None]:
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
    
    def show(self, head: int=10):
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

    def __len__(self):
        """Return the length of the dataset.
        
        Returns:
            Length of the dataset or dict of lengths for DatasetDict
        """
        if isinstance(self.dataset, DatasetDict):
            return {dt: len(self.dataset[dt]) for dt in self.dataset}
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
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

    def __data_type__(self):
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
        if labels is not None:
            if isinstance(labels[0], str):
                if self.multi_label_sep and self.multi_label_sep in labels:
                    multi_labels = labels.split(self.multi_label_sep)
                    if '.' in multi_labels[0]:
                        self.data_type = "multi_regression"
                    else:
                        self.data_type = "multi_label"
                else:
                    if '.' in labels[0]:
                        self.data_type = "regression"
                    else:
                        self.data_type = "classification"
            else:
                if isinstance(labels[0], int):
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

        def get_stats(ds):
            """
            label_distribution: Distribution of labels in the dataset
            mean_gc_content: Mean GC content of sequences from each label in the dataset
            """
            if "sequence" not in ds.column_names or "labels" not in ds.column_names:
                raise ValueError("Dataset must contain 'sequence' and 'labels' columns.")
            df = ds.to_pandas()
            df['GC_content'] = df['sequence'].apply(calc_gc_content)
            if self.data_type == "multi_label" and self.multi_label_sep:
                df_split = df['labels'].str.split(self.multi_label_sep, expand=True).astype(int)
                df_split.columns = [f'label_{i+1}' for i in range(df_split.shape[1])]
                df = df.drop(columns=['labels']).join(df_split)
                label_distribution = df.filter(like='label_').apply(lambda col: col.value_counts())
            elif self.data_type == "multi_regression":
                df['labels'] = df['labels'].apply(lambda x: [float(i) for i in x.split(self.multi_label_sep)])
                label_distribution = None
            elif self.data_type == "regression":
                df['labels'] = df['labels'].astype(float)
                # Split labels into 10 groups based on the values
                bins = pd.cut(df['labels'], bins=10)
                label_distribution = bins.value_counts().to_dict()
                mean_gc_content = df.groupby(bins)['GC_content'].mean().to_dict()
                labels = df['labels']
            elif self.data_type == "classification":
                df['labels'] = df['labels'].astype(int)
                label_distribution = df['labels'].value_counts().to_dict()
                mean_gc_content = df.groupby('labels')['GC_content'].mean().to_dict()
                labels = df['labels']
            else:
                raise ValueError("Unsupported data type for statistics calculation.")
            # Calculate statistics
            stats = {
                "num_samples": len(ds),
                "sequence_length": {
                    "min": min(len(seq) for seq in ds["sequence"]),
                    "max": max(len(seq) for seq in ds["sequence"]),
                    "average": sum(len(seq) for seq in ds["sequence"]) / len(ds),
                    "median": sorted(len(seq) for seq in ds["sequence"])[len(ds) // 2],
                },
                "label_distribution": label_distribution,
                "mean_gc_content": mean_gc_content
            }
            stats_for_plot = {
                "sequence_length": {
                    "all": [len(seq) for seq in ds["sequence"]],
                    "min": stats["sequence_length"]["min"],
                    "max": stats["sequence_length"]["max"],
                },
                "gc_contents": df['GC_content'].tolist(),
                "labels": labels
            }
            return stats, stats_for_plot

        if self.stats is None:
            self.stats = {}
            self.stats_for_plot = {}
            if isinstance(self.dataset, DatasetDict):
                for dt in self.dataset:
                    ds = self.dataset[dt]
                    self.stats[dt], self.stats_for_plot[dt] = get_stats(ds)
            else:
                ds = self.dataset
                self.stats, self.stats_for_plot = get_stats(ds)
        else:
            if isinstance(self.stats, dict):
                return self.stats
            else:
                raise ValueError("Statistics have not been computed yet. Please call `statistics()` method first.")

        return self.stats

    def plot_statistics(self, save_path: str = None):
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
        alt.data_transformers.enable("vegafusion")

        if self.stats is None:
            raise ValueError("Statistics have not been computed yet. Please call `statistics()` method first.")
        if isinstance(self.dataset, DatasetDict):
            combined_charts = []
            for dt in self.stats_for_plot:
                if "sequence_length" in self.stats_for_plot[dt]:
                    seq_len_df = pd.DataFrame({
                        "length": self.stats_for_plot[dt]["sequence_length"]["all"],
                        "label": self.stats_for_plot[dt]["labels"]
                    })
                    min_length = self.stats_for_plot[dt]["sequence_length"]["min"]
                    max_length = self.stats_for_plot[dt]["sequence_length"]["max"]
                    if min_length == max_length:
                        # If min and max lengths are the same, use a single bin
                        seq_len_chart = alt.Chart(seq_len_df).mark_bar().encode(
                            alt.X("length:Q", bin=alt.Bin(step=1)),
                            alt.Y("count()"),
                            alt.Color("label:N")
                        ).properties(title=f"Sequence Length Distribution")
                    else:
                        # Use binning for sequence length distribution
                        seq_len_chart = alt.Chart(seq_len_df).mark_bar().encode(
                            alt.X("length:Q", bin=alt.Bin(maxbins=30,
                                                          extent=[min_length, max_length])),
                            alt.Y("count()"),
                            alt.Color("label:N")
                        ).properties(title=f"Sequence Length Distribution")
                if "gc_contents" in self.stats_for_plot[dt]:
                    gc_df = pd.DataFrame({
                        "GC content": self.stats_for_plot[dt]["gc_contents"],
                        "label": self.stats_for_plot[dt]["labels"]
                    })
                    gc_chart = alt.Chart(gc_df).mark_boxplot().encode(
                        alt.X("label:N"),
                        alt.Y("GC content:Q"),
                        alt.Color("label:N")
                    ).properties(title=f"GC Content Distribution")
                # concatenate charts
                if 'seq_len_chart' in locals() and 'gc_chart' in locals():
                    combined_chart = alt.hconcat(seq_len_chart, gc_chart
                    ).properties(title=f"Statistics for {dt}")
                    combined_charts.append(combined_chart)
            if combined_charts:
                combined_chart = alt.vconcat(*combined_charts)
                if save_path:
                    combined_chart.save(f"{save_path}_stats.pdf")
                else:
                    combined_chart.show()
        else:
            if "sequence_length" in self.stats_for_plot:
                seq_len_df = pd.DataFrame({
                    "length": self.stats_for_plot["sequence_length"]["all"],
                    "label": self.stats_for_plot[dt]["labels"]
                })
                min_length = self.stats_for_plot["sequence_length"]["min"]
                max_length = self.stats_for_plot["sequence_length"]["max"]
                if min_length == max_length:
                    # If min and max lengths are the same, use a single bin
                    seq_len_chart = alt.Chart(seq_len_df).mark_bar().encode(
                        alt.X("length:Q", bin=alt.Bin(step=1)),
                        alt.Y("count()"),
                        alt.Color("label:N")
                    ).properties(title=f"Sequence Length Distribution")
                else:
                    # Use binning for sequence length distribution
                    seq_len_chart = alt.Chart(seq_len_df).mark_bar().encode(
                        alt.X("length:Q", bin=alt.Bin(maxbins=30,
                                                      extent=[min_length, max_length])),
                        alt.Y("count()"),
                        alt.Color("label:N")
                    ).properties(title=f"Sequence Length Distribution")
            if "gc_contents" in self.stats_for_plot:
                gc_df = pd.DataFrame({
                    "GC content": self.stats_for_plot["gc_contents"],
                    "label": self.stats_for_plot[dt]["labels"]
                })
                gc_chart = alt.Chart(gc_df).mark_boxplot().encode(
                    alt.X("label:N"),
                    alt.Y("GC content:Q"),
                    alt.Color("label:N")
                ).properties(title="GC Content Distribution")
            if 'seq_len_chart' in locals() and 'gc_chart' in locals():
                combined_chart = alt.hconcat(seq_len_chart, gc_chart
                ).properties(title="Statistics for Dataset")
                if save_path:
                    combined_chart.save(f"{save_path}_stats.pdf")
                else:
                    combined_chart.show()

        if save_path:
            print(f"Statistics plots saved to {save_path}")
        else:
            print("Statistics plots generated successfully.")


def show_preset_dataset() -> dict:
    """Show all preset datasets available in Hugging Face or ModelScope.

    Returns:
        A dictionary containing dataset names and their descriptions
    """
    from .dataset_auto import PRESET_DATASETS
    return PRESET_DATASETS


def load_preset_dataset(dataset_name: str, task: str=None) -> 'DNADataset':
    """Load a preset dataset from Hugging Face or ModelScope.

    Args:
        dataset_name: Name of the dataset
        task: Task directory in a dataset

    Returns:
        DNADataset: An instance wrapping a datasets.Dataset
        
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
from dnallm.datasets import show_preset_dataset, load_preset_dataset

# Show available preset datasets
show_preset_dataset()

# Load a specific dataset
ds = load_preset_dataset("nucleotide_transformer_downstream_tasks", task="enhancers_types")

# Get dataset statistics
ds.statistics()

# Plot dataset statistics
ds.plot_statistics()
"""


