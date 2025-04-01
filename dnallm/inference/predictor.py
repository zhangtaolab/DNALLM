import os
import warnings
import json
from typing import Optional, List, Dict, Union
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from ..datasets.data import DNADataset
from ..tasks.metrics import compute_metrics as Metrics

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

"""
DNA语言模型推理器模块

本模块实现了模型推理的核心功能，主要包括：

1. DNAPredictor类
   - 模型加载和初始化
   - 批量序列预测
   - 结果后处理
   - 设备管理
   - 半精度推理支持

2. 核心功能：
   - 模型状态管理
   - 批处理预测
   - 结果合并
   - 预测结果保存
   - 内存优化

3. 推理优化：
   - 批处理并行
   - GPU加速
   - 半精度计算
   - 内存效率优化

使用示例：
    predictor = DNAPredictor(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    results = predictor.predict(sequences)
"""

class DNAPredictor:
    """DNA sequence predictor using fine-tuned models"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: dict
    ):
        """
        Initialize predictor
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer for the model
            config: Configuration object containing task settings and inference parameters
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.task_config = config['task']
        self.pred_config = config['inference']
        self.device = self._get_device()
        if model:
            self.model.to(self.device)
        self.sequences = []
        self.labels = []

    def _get_device(self):
        # Get the device type
        device = self.pred_config.device.lower()
        if device == 'cpu':
            return torch.device('cpu')
        elif device == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('cuda')
        elif device == 'mps':
            if not torch.backends.mps.is_available():
                warnings.warn("MPS is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('mps')
        elif device == 'rocm':
            if not torch.backends.rocm.is_available():
                warnings.warn("ROCm is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('rocm')
        elif device == 'tpu':
            if not torch.backends.tpu.is_available():
                warnings.warn("TPU is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('tpu')
        elif device == 'xpu':
            if not torch.backends.xpu.is_available():
                warnings.warn("XPU is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('xpu')
        elif device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.backends.rocm.is_available():
                return torch.device('rocm')
            elif torch.backends.xpu.is_available():
                return torch.device('xpu')
            elif torch.backends.tpu.is_available():
                return torch.device('tpu')
            else:
                return torch.device('cpu')
        else:
            raise ValueError(f"Unsupported device type: {device}")

    def generate_dataset(self, seq_or_path: Union[str, List[str]], batch_size: int=1,
                         seq_col: str="sequence", label_col: str="labels",
                         keep_seqs: bool=True, do_encode: bool=True):
        """
        Generate dataset from sequences
        Args:
            seq_or_path: Single sequence or path to a file containing sequences
            batch_size: Batch size for DataLoader
            do_encode: Whether to encode sequences
        Returns:
            Dataset object
            DataLoader object
        """
        if isinstance(seq_or_path, str):
            suffix = seq_or_path.split(".")[-1]
            if suffix and os.path.isfile(seq_or_path):
                sequences = []
                dataset = DNADataset.load_local_data(seq_or_path, seq_col=seq_col, label_col=label_col,
                                                     tokenizer=self.tokenizer, max_length=self.pred_config.max_length)
            else:
                sequences = [seq_or_path]
        elif isinstance(seq_or_path, list):
            sequences = seq_or_path
        else:
            raise ValueError("Input should be a file path or a list of sequences.")
        if len(sequences) > 0:
            ds = Dataset.from_dict({"sequence": sequences})
            dataset = DNADataset(ds, self.tokenizer, max_length=self.pred_config.max_length)
        # If labels are provided, keep labels
        if keep_seqs:
            self.sequences = dataset.dataset["sequence"]
        # Encode sequences
        if do_encode:
            dataset.encode_sequences(remove_unused_columns=True)
        if "labels" in dataset.dataset.features:
            self.labels = dataset.dataset["labels"]
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.pred_config.num_workers
        )

        return dataset, dataloader

    def logits_to_preds(self, logits: list) -> tuple[torch.Tensor, list]:
        """
        Convert logits to predictions
        Args:
            logits: Model output logits
        Returns:
            preds: Model predictions
            labels: Human-readable labels
        """
        # Get task type and threshold from config
        task_type = self.task_config.task_type
        threshold = self.task_config.threshold
        label_names = self.task_config.label_names
        # Convert logits to predictions based on task type
        if task_type == "binary":
            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] > threshold).long()
            labels = [label_names[pred] for pred in preds]
        elif task_type == "multiclass":
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            labels = [label_names[pred] for pred in preds]
        elif task_type == "multilabel":
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            labels = []
            for pred in preds:
                label = [label_names[i] for i in range(len(pred)) if pred[i] == 1]
                labels.append(label)
        elif task_type == "regression":
            preds = logits.squeeze(-1)
            probs = preds
            labels = preds.tolist()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        return probs, labels

    def format_output(self, predictions: tuple[torch.Tensor, list]) -> dict:
        """
        Format output predictions
        Args:
            predictions: Tuple containing predictions
        Returns:
            Dictionaries containing formatted predictions
        """
        formatted_predictions = {}
        probs, labels = predictions
        probs = probs.numpy().tolist()
        keep_seqs = True if len(self.sequences) else False
        label_names = self.task_config.label_names
        for i, label in enumerate(labels):
            prob = probs[i]
            formatted_predictions[i] = {
                'sequence': self.sequences[i] if keep_seqs else '',
                'label': label,
                'scores': {label_names[j]: p for j, p in enumerate(prob)}
            }
        return formatted_predictions

    @torch.no_grad()
    def batch_predict(self, dataloader: DataLoader, do_pred=True) -> tuple[torch.Tensor, list]:
        """
        Predict for a batch of sequences
        Args:
            dataloader: DataLoader object containing sequences
        Returns:
            Dictionary containing predictions
        """
        # Set model to evaluation mode
        self.model.eval()
        all_logits = []
        # Iterate over batches
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            if self.pred_config.use_fp16:
                self.model = self.model.half()
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            logits = outputs.logits.cpu().detach()
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        # Get predictions
        predictions = None
        if do_pred:
            predictions = self.logits_to_preds(all_logits)
            predictions = self.format_output(predictions)

        return all_logits, predictions

    def predict_seqs(self, sequences: Union[str, List[str]], 
                     evaluate: bool = False,
                     save_to_file: bool = False) -> Union[tuple, dict]:
        """
        Predict for sequences
        
        Args:
            sequences: Single sequence or list of sequences
            save_to_file: Whether to save predictions to file
            
        Returns:
            Dictionary containing predictions
        """
        # Get dataset and dataloader from sequences
        _, dataloader = self.generate_dataset(sequences, batch_size=self.pred_config.batch_size)
        # Do batch prediction
        logits, predictions = self.batch_predict(dataloader)
        # Save predictions
        if save_to_file and self.config.output_dir:
            save_predictions(predictions, Path(self.config.output_dir))
        # Do evaluation
        if len(self.labels) == len(logits) and evaluate:
            metrics = self.calculate_metrics(logits, self.labels)
            if save_to_file and self.config.output_dir:
                save_metrics(metrics, Path(self.config.output_dir))
            return predictions, metrics

        return predictions


    def predict_file(self, file_path: str, evaluate: bool = False,
                     save_to_file: bool = False) -> Union[tuple, dict]:
        """
        Predict from a file containing sequences
        Args:
            file_path: Path to the file containing sequences
        Returns:
            List of dictionaries containing predictions
        """
        # Get dataset and dataloader from file
        _, dataloader = self.generate_dataset(file_path, batch_size=self.pred_config.batch_size)
        # Do batch prediction
        logits, predictions = self.batch_predict(dataloader)
        # Save predictions
        if save_to_file and self.config.output_dir:
            save_predictions(predictions, Path(self.config.output_dir))
        # Do evaluation
        if len(self.labels) == len(logits) and evaluate:
            metrics = self.calculate_metrics(logits, self.labels)
            if save_to_file and self.config.output_dir:
                save_metrics(metrics, Path(self.config.output_dir))
            return predictions, metrics

        return predictions

    def calculate_metrics(self, logits: Union[List, torch.Tensor],
                          labels: Union[List, torch.Tensor]) -> dict:
        """
        Calculate evaluation metrics
        Args:
            logits: Model predictions
            labels: True labels
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get task type and threshold from config
        task_type = self.task_config.task_type
        threshold = self.task_config.threshold
        label_names = self.task_config.label_names
        # Calculate metrics based on task type
        compute_metrics = Metrics(self.task_config)
        metrics = compute_metrics((logits, labels))
        
        return metrics


def save_predictions(predictions: Dict, output_dir: Path) -> None:
    """Save predictions to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)

def save_metrics(metrics: Dict, output_dir: Path) -> None:
    """Save metrics to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
