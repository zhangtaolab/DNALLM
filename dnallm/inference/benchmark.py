import os
import numpy as np
import json
from typing import Optional, List, Dict, Union
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from ..models import *
from ..datasets.data import DNADataset
from .predictor import DNAPredictor, save_predictions, save_metrics
from .plot import *

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class Benchmark:
    """
    Class for benchmarking DNA LLMs.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the Benchmark class.
        Args:
            config (dict): Configuration object containing task settings and inference parameters
        """
        self.config = config
        self.all_models = {'huggingface': set(np.concatenate([MODEL_INFO[m]['huggingface'] for m in MODEL_INFO]).tolist()),
                           'modelscope': set(np.concatenate([MODEL_INFO[m]['modelscope'] for m in MODEL_INFO]).tolist())}

    def get_predictor(self, model, tokenizer) -> DNAPredictor:
        """
        Create a predictor object for the model.
        Args:
            model: The model to be used for prediction.
            tokenizer: The tokenizer to be used for encoding sequences.
        Returns:
            DNAPredictor: The predictor object.
        """
        
        predictor = DNAPredictor(
            model=model,
            tokenizer=tokenizer,
            config=self.config
        )
        return predictor
    
    def get_dataset(self, seq_or_path: Union[str, List[str]],
                    seq_col: str="sequence", label_col: str="labels") -> DNADataset:
        """
        Load the dataset from the specified path or list of sequences.
        Args:
            seq_or_path (Union[str, List[str]]): Path to the sequence file or list of sequences.
        Returns:
            DNADataset: Dataset object containing the sequences.
        """
        predictor = DNAPredictor(
            model=None,
            tokenizer=None,
            config=self.config
        )
        ds, _ = predictor.generate_dataset(
            seq_or_path=seq_or_path,
            seq_col=seq_col,
            label_col=label_col,
            keep_seqs=False,
            do_encode=False
        )
        self.dataset = ds.dataset
        return ds

    def available_models(self, show_all: bool=True) -> List[str]:
        """
        List all available models.
        Args:
            show_all (bool): If True, show all models. If False, show only the models that are available.
        Returns:
            List[str]: List of model names.
        """
        # Load the model information
        if show_all:
            return MODEL_INFO
        else:
            models_list = {m: MODEL_INFO[m]["model_tags"] for m in MODEL_INFO}
            return models_list

    def run(self, model_names: Union[List[str], Dict] = None,
            source: str="local", use_mirror: bool=False,
            save_preds: bool=False, save_scores: bool=True) -> None:
        """
        Perform the benchmark.
        Args:
            model_names (Union[List[str], Dict]): List of model names or a dictionary mapping model names to paths.
            source (str): Source of the models ('local', 'huggingface', 'modelscope').
            use_mirror (bool): Whether to use a mirror for downloading models.
            save_preds (bool): Whether to save the predictions.
            save_scores (bool): Whether to save the metrics.
        Returns:
            None
        """
        task_config = self.config['task']
        pred_config = self.config['inference']
        all_results = {}
        metrics_save = {}
        labels = self.dataset['labels']
        for model_name in model_names:
            print(model_name)
            # Check if the model name is provided as a string or a dictionary
            if isinstance(model_names, dict):
                model_path = model_names[model_name]
            else:
                model_path = model_name
                model_name = os.path.basename(model_name)
            # Check if the model is local or remote
            if source == "local":
                model, tokenizer = load_model_and_tokenizer(model_path, task_config=task_config)
                predictor = self.get_predictor(model, tokenizer)
            else:
                # Check if the model is available in the model library
                # if model_path not in self.all_models[source]:
                #     print(f"Model \'{model_path}\' not found in our available models list.")
                #     continue
                try:
                    model, tokenizer = load_model_and_tokenizer(model_path, task_config=task_config,
                                                                source=source, use_mirror=use_mirror)
                except:
                    model, tokenizer = load_model_and_tokenizer(model_path, task_config=task_config)
                else:
                    raise NameError("Cannot find model in either the given source or local.")
                predictor = self.get_predictor(model, tokenizer)
            # Load the dataset
            dataset = DNADataset(self.dataset, tokenizer=tokenizer, max_length=pred_config.max_length)
            dataset.encode_sequences(remove_unused_columns=True)
            dataloader = DataLoader(
                dataset,
                batch_size=pred_config.batch_size,
                num_workers=pred_config.num_workers
            )
            # Perform the prediction
            logits, _, _ = predictor.batch_predict(dataloader, do_pred=False)
            if len(labels) == len(logits):
                metrics = predictor.calculate_metrics(logits, labels, plot=True)
                all_results[model_name] = metrics
                metrics2 = dict(metrics)
                if 'curve' in metrics2:
                    del metrics2['curve']
                if 'scatter' in metrics2:
                    del metrics2['scatter']
                metrics_save[model_name] = metrics2
        # Save the metrics
        if save_scores and pred_config.output_dir:
            save_metrics(metrics_save, Path(pred_config.output_dir))
        return all_results

    def plot(self, metrics: dict,
             show_score: bool = True,
             save_path: Optional[str] = None,
             separate: bool = False) -> None:
        """
        Plot the benchmark results.
        Args:
            metrics (dict): Dictionary containing model metrics.
            show_score (bool): Whether to show the score on the plot.
            save_path (Optional[str]): Path to save the plot.
            separate (bool): Whether to save the plots separately.
        Returns:
            None
        """
        task_config = self.config['task']
        task_type = task_config.task_type
        if task_type in ['binary', 'multiclass', 'multilabel', 'token']:
            # Prepare data for plotting
            bars_data, curves_data = prepare_data(metrics, task_type=task_type)
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    bar_chart = save_path.replace(suffix, "_metrics" + suffix)
                    line_chart = save_path.replace(suffix, "_roc" + suffix)
                else:
                    bar_chart = os.path.join(save_path, "metrics.pdf")
                    line_chart = os.path.join(save_path, "roc.pdf")
            else:
                bar_chart = None
                line_chart = None
            # Plot bar charts
            pbar = plot_bars(bars_data, show_score=show_score,
                             save_path=bar_chart, separate=separate)
            # Plot curve charts
            if task_type == 'token':
                pline = None
            else:
                pline = plot_curve(curves_data, show_score=show_score,
                                save_path=line_chart, separate=separate)
            return pbar, pline
        elif task_type == 'regression':
            # Prepare data for plotting
            bars_data, scatter_data = prepare_data(metrics, task_type=task_type)
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    bar_chart = save_path.replace(suffix, "_metrics" + suffix)
                    scatter_plot = save_path.replace(suffix, "_scatter" + suffix)
                else:
                    bar_chart = os.path.join(save_path, "metrics.pdf")
                    scatter_plot = os.path.join(save_path, "scatter.pdf")
            else:
                bar_chart = None
            # Plot bar charts
            pbar = plot_bars(bars_data, show_score=show_score,
                             save_path=bar_chart, separate=separate)
            # Plot scatter plots
            pdot = plot_scatter(scatter_data, show_score=show_score,
                                save_path=scatter_plot, separate=separate)
            return pbar, pdot
