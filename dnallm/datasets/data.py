import os
import random
from typing import Union, Optional
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from ..utils.sequence import check_sequence, reverse_complement, random_generate_sequences


class DNADataset:
    def __init__(self, ds: Union[Dataset, DatasetDict], tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512):
        """
        Args:
            ds (datasets.Dataset or DatasetDict): A Hugging Face Dataset containing at least 'sequence' and 'label' fields.
            tokenizer (PreTrainedTokenizerBase, optional): A Hugging Face tokenizer for encoding sequences.
            max_length (int, optional): Maximum length for tokenization.
        """
        self.dataset = ds
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep = None
        self.multi_label_sep = None

    @classmethod
    def load_local_data(cls, file_paths, seq_col: str = "sequence", label_col: str = "labels",
                        sep: str = None, fasta_sep: str = "|",
                        multi_label_sep: Union[str, None] = None,
                        tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512):
        """
        Load DNA sequence datasets from one or multiple local files.
        
        Supports input formats: csv, tsv, json, parquet, arrow, dict, fasta, txt.
        
        Args:
            file_paths (str, list, or dict):  
                - Single dataset: Provide one file path (e.g., "data.csv").  
                - Pre-split datasets: Provide a dict like `{"train": "train.csv", "test": "test.csv"}`.
            seq_col (str): Column name for DNA sequences.
            label_col (str): Column name for labels.
            sep (str, optional): Delimiter for CSV, TSV, or TXT.
            fasta_sep (str, optional): Delimiter for FASTA files.
            multi_label_sep (str, optional): Delimiter for multi-label sequences.
            tokenizer (PreTrainedTokenizerBase, optional): A tokenizer.
            max_length (int, optional): Max token length.
        
        Returns:
            DNADataset: An instance wrapping a Dataset or DatasetDict.
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

        return cls(dataset, tokenizer=tokenizer, max_length=max_length)

    @classmethod
    def _load_single_data(cls, file_path, seq_col: str = "sequence", label_col: str = "labels",
                          sep: str = None, fasta_sep: str = "|",
                          multi_label_sep: Union[str, None] = None) -> Dataset:
        """
        Load DNA data (sequences and labels) from a local file.
        Supported file types: 
          - For structured formats (CSV, TSV, JSON, Parquet, Arrow, dict), uses load_dataset from datasets.
          - For FASTA and TXT, uses custom parsing.
        Args:
            file_path: For most file types, a path (or pattern) to the file(s). For 'dict', a dictionary.
            seq_col (str): Name of the column containing the DNA sequence.
            label_col (str): Name of the column containing the label.
            sep (str, optional): Delimiter for CSV, TSV, or TXT files.
            fasta_sep (str, optional): Delimiter for FASTA files.
            multi_label_sep (str, optional): Delimiter for multi-label sequences.
            tokenizer (PreTrainedTokenizerBase, optional): A tokenizer to pass along.
            max_length (int): Maximum length for tokenization.
        Returns:
            DNADataset: An instance wrapping a datasets.Dataset.
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
                    ds = ds.rename_column(seq_col, "sequence")
                if label_col in ds.column_names:
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
        ds = ds.map(format_labels, desc="Format labels")
        # Return processed dataset
        return ds

    @classmethod
    def from_huggingface(cls, dataset_name: str,
                         seq_col: str = "sequence", label_col: str = "labels",
                         data_dir: Union[str, None]=None,
                         tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512):
        """
        Load a dataset from the Hugging Face Hub.
        Args:
            dataset_name (str): Name of the dataset.
            seq_col (str): Column name for the DNA sequence.
            label_col (str): Column name for the label.
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
                        tokenizer: PreTrainedTokenizerBase = None, max_length: int = 512):
        """
        Load a dataset from the ModelScope.
        Args:
            dataset_name (str): Name of the dataset.
            seq_col (str): Column name for the DNA sequence.
            label_col (str): Column name for the label.
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
                         task: Optional[str] = 'SequenceClassification'):
        """
        Encode all sequences using the provided tokenizer.
        The dataset is mapped to include tokenized fields along with the label,
        making it directly usable with Hugging Face Trainer.
        
        Args:
            padding (str): Padding strategy for sequences. this can be 'max_length' or 'longest'.
                           Use 'longest' to pad to the length of the longest sequence in case of memory outage.
            return_tensors (str | TensorType): Returned tensor types, can be 'pt' or 'tf' or 'np'.
            remove_unused_columns: Whether to remove the original 'sequence' and 'label' columns
            task (str, optional): Task type for the tokenizer. If not provided, defaults to 'SequenceClassification'.
        """
        if self.tokenizer:
            sp_token_map = self.tokenizer.special_tokens_map
            pad_token = sp_token_map['pad_token'] if 'pad_token' in sp_token_map else None
            pad_id = self.tokenizer.encode(pad_token)[-1] if pad_token else None
            cls_token = sp_token_map['cls_token'] if 'cls_token' in sp_token_map else None
            sep_token = sp_token_map['sep_token'] if 'sep_token' in sp_token_map else None
            max_length = self.max_length
        else:
            raise ValueError("Tokenizer not provided.")
        def tokenize_for_sequence_classification(example):
            tokenized = self.tokenizer(
                example["sequence"],
                truncation=True,
                padding=padding,
                max_length=max_length
            )
            return tokenized
        def tokenize_for_token_classification(examples):
            tokenized_examples = {'sequence': [],
                                  'input_ids': [],
                                  'token_type_ids': [],
                                  'attention_mask': [],
                                  'labels': []}
            input_seqs = examples['sequence']
            if isinstance(input_seqs, str):
                input_seqs = input_seqs.split(self.multi_label_sep)
            for i, example_tokens in enumerate(input_seqs):
                all_ids = [x for x in self.tokenizer.encode(example_tokens, is_split_into_words=True) if x>=0]
                example_ner_tags = examples['labels'][i]
                pad_len = max_length - len(all_ids)
                if pad_len >= 0:
                    all_masks = [1] * len(all_ids) + [0] * pad_len
                    all_ids = all_ids + [pad_id] * pad_len
                    if cls_token:
                        if sep_token:
                            example_tokens = [cls_token] + example_tokens + [sep_token] + [pad_token] * pad_len
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
                tokenized_examples['token_type_ids'].append([0] * max_length)
                tokenized_examples['attention_mask'].append(all_masks)
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
            used_cols = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
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

    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
        """
        Split the dataset into train, test, and validation sets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.
            seed (int): Random seed for reproducibility.
        Returns:
            A tuple of DNADataset instances: (train, test, validation)
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
            self.dataset = DatasetDict({'train': train_ds, 'test': test_ds})
    
    def shuffle(self, seed: int = 42):
        """
        Shuffle the dataset
        
        Args:
            seed (int): Random seed for reproducibility.
        """
        self.dataset.shuffle(seed=seed)

    def validate_sequences(self, minl: int = 20, maxl: int = 6000, gc: tuple = (0,1), valid_chars: str = "ACGTN"):
        """
        Filter the dataset to keep sequences containing valid DNA bases or allowed length.
        
        Args:
            minl (int): Minimum length of the sequences.
            maxl (int): Maximum length of the sequences.
            gc (tuple): GC content range.
            valid_chars (str): Allowed characters in the sequences.
        """
        self.dataset = self.dataset.filter(
            lambda example: check_sequence(example["sequence"], minl, maxl, gc, valid_chars)
        )

    def random_generate(self, minl: int, maxl: int = 0, samples: int = 1,
                              gc: tuple = (0,1), N_ratio: float = 0.0,
                              padding_size: int = 0, seed: int = None,
                              label_func = None, append: bool = False):
        """
        Replace the current dataset with randomly generated DNA sequences.
        Args:
            minl: int, minimum length of the sequences
            maxl: int, maximum length of the sequences, default is the same as minl
            samples: int, number of sequences to generate, default 1
            with_N: bool, whether to include N in the base map, default False
            gc: tuple, GC content range, default (0,1)
            padding_size: int, padding size for sequence length, default 0
            seed: int, random seed, default None
            label_func (callable, optional): A function that generates a label from a sequence.
            append: bool, append the random generated data to the existed dataset or use the data as a dataset
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
        """
        Filter out samples with missing or empty sequences or labels.
        """
        def non_missing(example):
            return example["sequence"] and example["labels"] is not None and example["sequence"].strip() != ""
        self.dataset = self.dataset.filter(non_missing)

    def raw_reverse_complement(self, ratio: float = 0.5, seed: int = 42):
        """
        Do reverse complement of sequences in the dataset.
        
        Args:
            ratio (float): Ratio of sequences to reverse complement.
            seed (int): Random seed for reproducibility.
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
        """
        Augment the dataset by adding reverse complement sequences.
        This method doubles the dataset size.
        
        Args:
            reverse (bool): Whether to do reverse.
            complement (bool): Whether to do complement.
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
        """
        Augment each sample by concatenating the sequence with its reverse complement.
        
        Args:
            reverse (bool): Whether to do reverse.
            complement (bool): Whether to do complement.
            sep (str): Separator between the original and reverse complement sequences.
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

    def iter_batches(self, batch_size: int):
        """
        Generator that yields batches of examples from the dataset.
        
        Args:
            batch_size (int): Size of each batch.
        Yields:
            A batch of examples.
        """
        if isinstance(self.dataset, DatasetDict):
            raise ValueError("Dataset is a DatasetDict Object, please use `DNADataset.dataset[datatype].iter_batches(batch_size)` instead.")
        else:
            for i in range(0, len(self.dataset), batch_size):
                yield self.dataset[i: i + batch_size]

    def __len__(self):
        if isinstance(self.dataset, DatasetDict):
            return {dt: len(self.dataset[dt]) for dt in self.dataset}
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(self.dataset, DatasetDict):
            raise ValueError("Dataset is a DatasetDict Object, please use `DNADataset.dataset[datatype].__getitem__(idx)` instead.")
        else:
            return self.dataset[idx]



# Load data (fasta or csv format)
def load_fasta_data(data, labels=None, data_type='', sample=1e8, seed=7):
    # check if the data is existed
    if not data:
        return {}
    if not os.path.exists(data):
        return {}

    # Generate data dictionary
    dic = {'idx': [], 'sequence': [], 'label': []}
    idx = 0
    with open(data) as infile:
        for line in tqdm(infile, desc=data_type):
            if line.startswith(">"):
                name = line.strip()[1:].split("|")[0]
                name = "_".join(name.split("_")[:-1])
                label_info = line.strip()[1:].split("|")[1:]
                label = label_info[0]
            else:
                seq = line.strip()
                dic['idx'].append(name)
                dic['sequence'].append(seq)
                if labels:
                    dic['label'].append(int(label))
                else:
                    dic['label'].append(float(label))
                idx += 1
    # random sampling for fasta format data
    if sample < idx:
        random.seed(seed)
        print('Downsampling %s data to %s.' % (data_type, sample))
        indices = random.sample(range(idx), k=sample)
        sampled_dic = {'idx': [], 'sequence': [], 'label': []}
        # for label in labels:
        #     sampled_dic.update({label: []})
        for i in tqdm(indices):
            sampled_dic['idx'].append(dic['idx'][i])
            sampled_dic['sequence'].append(dic['sequence'][i])
            sampled_dic['label'].append(dic['label'][i])
        return sampled_dic
    else:
        return dic


def load_csv_data(train_data, eval_data=None, test_data=None, labels=None, split=0.1, shuffle=False, sample=1e8, seed=7):
    dataset ={}
    if train_data.endswith(".csv"):
        if eval_data and test_data:
            data_files = {'train': train_data, 'dev': eval_data, 'test': test_data}
            dataset = load_dataset('csv', data_files=data_files)
        elif eval_data:
            data_files = {'train': train_data, 'dev': eval_data}
            dataset = load_dataset('csv', data_files=data_files)
        elif test_data:
            data_files = {'train': train_data, 'test': test_data}
            dataset = load_dataset('csv', data_files=data_files)
        else:
            dataset = load_dataset('csv', data_files=train_data)
            dataset = dataset['train'].train_test_split(test_size=split)
        if shuffle:
            dataset["train"] = dataset["train"].shuffle(seed)
    return dataset
