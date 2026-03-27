"""DNA Language Model Benchmarking Module.

This module provides comprehensive benchmarking capabilities for DNA language
models,
including performance evaluation, metrics calculation, and
    result visualization.
"""

import os
import numpy as np
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ..models import MODEL_INFO, load_model_and_tokenizer
from ..datahandling.data import DNADataset
from ..configuration.configs import (
    TaskConfig,
    InferenceConfig,
    EvaluationConfig,
)
from .inference import DNAInference, save_metrics
from .plot import prepare_data, plot_bars, plot_curve, plot_scatter

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Benchmark:
    """Class for benchmarking DNA Language Models.

    This class provides methods to evaluate the performance of different DNA
        language
            models on various tasks, including classification, regression, and
            token classification.

        Attributes:
            config: Configuration dictionary containing task settings and
                inference parameters
            all_models: Dictionary mapping source names to sets of available
                model names
            dataset: The dataset used for benchmarking
    """

    def __init__(
        self,
        config: dict | None = None,
        models: list[dict] | None = None,
        datasets: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        batch_size: int = 1,
        device: str | None = None,
        **kwargs,
    ):
        """Initialize the Benchmark class.

        Args:
            config: Configuration object containing task settings and
                inference parameters
            models: List of dictionaries specifying models to benchmark, each
                with keys "name", "path", "source", and optionally "task_type"
            datasets: Dictionary mapping dataset names to pre-loaded dataset
                objects for config-less execution
            metrics: List of metric names to calculate and plot
            batch_size: Batch size for inference
                (used if config is not provided)
            device: Device to run inference on
                (used if config is not provided)
            **kwargs: Additional keyword arguments
        """
        self.all_models = {
            "huggingface": set(
                np.concatenate([
                    MODEL_INFO[m]["huggingface"] for m in MODEL_INFO
                ]).tolist()
            ),
            "modelscope": set(
                np.concatenate([
                    MODEL_INFO[m]["modelscope"] for m in MODEL_INFO
                ]).tolist()
            ),
        }
        self.datasets: list[str] = []
        # Store pre-loaded datasets for config-less execution
        self.datasets_dict = datasets or {}

        if config is not None:
            self.config = config
            if "benchmark" in config:
                self.prepared = self.__load_from_config()
            else:
                self.prepared = None
        else:
            # Code-based initialization (No YAML config)
            self.config = {}
            self.config["inference"] = InferenceConfig
            for k, v in dict(EvaluationConfig()).items():
                setattr(self.config["inference"], k, v)
            self.config["inference"].batch_size = batch_size
            self.config["inference"].device = (
                device
                if device
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            self.config["inference"].max_length = kwargs.get("max_length", 512)
            self.config["inference"].num_workers = kwargs.get("num_workers", 4)
            self.config["inference"].output_dir = kwargs.get(
                "output_dir", None
            )

            self.config["task"] = TaskConfig

            model_names = {}
            sources = []
            task_configs = []

            if models:
                for m in models:
                    model_names[m["name"]] = m["path"]
                    sources.append(m.get("source", "huggingface"))

            if self.datasets_dict:
                for _, dinfo in self.datasets_dict.items():
                    task_config = self.config["task"]
                    task_config.task_type = dinfo.get("task_type", "binary")
                    task_config.num_labels = dinfo.get("num_labels", 2)
                    task_config.label_names = dinfo.get(
                        "label_names",
                        [f"class_{i}" for i in range(task_config.num_labels)],
                    )
                    task_configs.append(task_config)

            self.prepared = {
                "models": model_names,
                "sources": sources,
                "tasks": task_configs,
                "dataset": [],
                "metrics": metrics or [],
                "plot_format": kwargs.get("plot_format", "pdf"),
            }

    def __load_from_config(self):
        """Load the benchmark-specific parameters from the configuration."""
        benchmark_config = self.config["benchmark"]
        config_path = os.path.dirname(benchmark_config.config_path)
        models = benchmark_config.models
        model_names = {m.name: m.path for m in models}
        sources = [m.source if hasattr(m, "source") else None for m in models]
        self.config["inference"] = InferenceConfig
        for k, v in dict(benchmark_config.evaluation).items():
            setattr(self.config["inference"], k, v)
        self.config["inference"].output_dir = benchmark_config.output.path
        if hasattr(benchmark_config, "datasets"):
            datasets = benchmark_config.datasets
            task_configs = []
            for d in datasets:
                self.config["task"] = TaskConfig
                self.config["task"].task_type = d.task
                self.config["task"].num_labels = d.num_labels
                self.config["task"].label_names = d.label_names
                self.config["task"].threshold = d.threshold
                data_path = (
                    d.path
                    if os.path.isfile(d.path)
                    else os.path.abspath(config_path + "/" + d.path)
                )
                self.get_dataset(data_path, d.text_column, d.label_column)
                task_configs.append(self.config["task"])
        else:
            datasets = []
            task_configs = [self.config["task"]]
        metrics = benchmark_config.metrics
        plot_format = benchmark_config.output.format
        return {
            "models": model_names,
            "sources": sources,
            "tasks": task_configs,
            "dataset": datasets,
            "metrics": metrics,
            "plot_format": plot_format,
        }

    def get_inference_engine(self, model: Any, tokenizer: Any) -> DNAInference:
        """Create an inference engine object for the model.

        Args:
            model: The model to be used for inference
            tokenizer: The tokenizer to be used for encoding sequences

        Returns:
            DNAInference: The inference engine object configured with the given
                model and tokenizer
        """

        inference_engine = DNAInference(
            model=model, tokenizer=tokenizer, config=self.config
        )
        return inference_engine

    def get_dataset(
        self,
        seq_or_path: str | list[str],
        seq_col: str = "sequence",
        label_col: str = "labels",
    ) -> DNADataset:
        """Load the dataset from the specified path or list of sequences.

        Args:
            seq_or_path: Path to the sequence file or list of sequences
            seq_col: Column name for DNA sequences, default "sequence"
            label_col: Column name for labels, default "labels"

        Returns:
            DNADataset: Dataset object containing the sequences and labels
        """
        inference_engine = DNAInference(
            model=None, tokenizer=None, config=self.config
        )
        ds, _ = inference_engine.generate_dataset(
            seq_or_path=seq_or_path,
            seq_col=seq_col,
            label_col=label_col,
            keep_seqs=False,
            do_encode=False,
        )
        self.datasets.append(ds.dataset)
        return ds

    def available_models(self, show_all: bool = True) -> dict[str, Any]:
        """List all available models.

        Args:
            show_all: If True, show all models. If False,
                show only the models that are available

        Returns:
            List of model names if show_all=True,
                otherwise dictionary mapping model names to tags
        """
        # Load the model information
        if show_all:
            return MODEL_INFO
        else:
            models_list = {m: MODEL_INFO[m]["model_tags"] for m in MODEL_INFO}
            return models_list

    def run(
        self,
        model_names: list[str] | dict | None = None,
        source: str = "huggingface",
        use_mirror: bool = False,
        save_preds: bool = False,
        save_scores: bool = True,
    ) -> dict[str, Any]:
        """Perform the benchmark evaluation on multiple models.

        This method loads each model, runs predictions on the dataset,
        calculates metrics, and optionally saves the results.

        Args:
            model_names: List of model names or
                a dictionary mapping model names to paths
            source: Source of the models ('local', 'huggingface', 'modelscope')
            use_mirror: Whether to use a mirror for downloading models
            save_preds: Whether to save the predictions
            save_scores: Whether to save the metrics

        Returns:
            dict[str, Any]: Dictionary containing benchmark results for each
                dataset and model

        Raises:
            NameError: If model cannot be found in either the given source or
                local storage
        """
        all_results: dict[str, Any] = {}
        selected_results: dict[str, Any] = {}
        metrics_save: dict[str, Any] = {}
        if self.prepared:
            task_configs = self.prepared["tasks"]
            pred_config = self.config["inference"]
            # Get datasets and model names from preset config
            dataset_names = [d.name for d in self.prepared["dataset"]]
            model_names = self.prepared["models"]
            sources = [s if s else source for s in self.prepared["sources"]]
            selected_metrics = self.prepared["metrics"]
        else:
            # Get configurations from the provided config
            task_configs = [self.config["task"]]
            pred_config = self.config["inference"]
            # Get dataset and model names manually
            dataset_names = ["custom"]
            sources = [source] * len(model_names)
            selected_metrics = []
            # Load the dataset
        for di, dname in enumerate(dataset_names):
            print("Dataset name:", dname)
            all_results[dname] = {}
            selected_results[dname] = {}
            metrics_save[dname] = {}
            labels = self.datasets[di]["labels"]
            task_config = (
                task_configs[di] if di < len(task_configs) else task_configs[0]
            )
            for mi, model_name in enumerate(model_names):
                print("Model name:", model_name)
                # Check if the model name is provided as a string or a
                # dictionary
                if isinstance(model_names, dict):
                    model_path = model_names[model_name]
                else:
                    model_path = model_name
                    model_name = os.path.basename(model_name)
                # Check if the model is local or remote
                source = sources[mi]
                # Load the model and tokenizer
                if source == "local":
                    model, tokenizer = load_model_and_tokenizer(
                        model_path, task_config=task_config
                    )
                else:
                    # Check if the model is available in the model library
                    # if model_path not in self.all_models[source]:
                    # print(f"Model \'{model_path}\' not found in our available
                    # models list.")
                    #     continue
                    # try:
                    model, tokenizer = load_model_and_tokenizer(
                        model_path,
                        task_config=task_config,
                        source=source,
                        use_mirror=use_mirror,
                    )
                    # except Exception:
                    #     if os.path.exists(model_path):
                    #         model, tokenizer = load_model_and_tokenizer(
                    #             model_path, task_config=task_config
                    #         )
                    #     else:
                    #         raise NameError(
                    # "Cannot find model in either the given source or local."
                    #         ) from None
                if hasattr(tokenizer, "model_max_length"):
                    max_length = min(
                        tokenizer.model_max_length, pred_config.max_length
                    )
                else:
                    max_length = pred_config.max_length
                dataset = DNADataset(
                    self.datasets[di],
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
                dataset.encode_sequences(remove_unused_columns=True)
                dataloader: DataLoader = DataLoader(
                    dataset,
                    batch_size=pred_config.batch_size,
                    num_workers=pred_config.num_workers,
                )
                inference_engine = self.get_inference_engine(model, tokenizer)
                # Perform the prediction
                logits, _, _ = inference_engine.batch_infer(
                    dataloader, do_pred=False
                )
                if len(labels) == len(logits):
                    metrics = inference_engine.calculate_metrics(
                        logits, labels, plot=True
                    )
                    all_results[dname][model_name] = metrics
                    # keep selected metrics
                    if selected_metrics:
                        selected_results[dname][model_name] = {}
                        if "all" in selected_metrics:
                            selected_results[dname][model_name] = metrics
                        else:
                            for metric in selected_metrics:
                                selected_results[dname][model_name][metric] = (
                                    all_results[dname][model_name][metric]
                                )
                    else:
                        selected_results[dname][model_name] = metrics
                    # keep all metrics if save_scores is True
                    if save_scores:
                        metrics2 = dict(metrics)
                        metrics2.pop("curve", None)
                        metrics2.pop("scatter", None)
                        metrics_save[dname][model_name] = metrics2
        # Save the metrics
        if save_scores and pred_config.output_dir:
            save_metrics(metrics_save, Path(pred_config.output_dir))
        if self.prepared:
            return selected_results
        else:
            return all_results

    def evaluate_single_model(
        self,
        model: Any,
        tokenizer: Any,
        val_data: Any,
        metrics_list: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate a single model on a given dataset/fold.

        Args:
            model: The model to be evaluated
            tokenizer: The tokenizer to be used for encoding sequences
            val_data: The validation dataset or subset to evaluate on
            metrics_list: Optional list of metric names to calculate.
                If None, all calculated metrics will be returned.
        Returns:
            Dictionary of calculated metrics for the model
                on the given dataset/fold
        """
        pred_config = self.config["inference"]

        if hasattr(tokenizer, "model_max_length"):
            max_length = min(
                tokenizer.model_max_length,
                getattr(pred_config, "max_length", 512),
            )
        else:
            max_length = getattr(pred_config, "max_length", 512)

        # Wrap with DNADataset to encode sequences for the specific tokenizer
        try:
            dataset = DNADataset(
                val_data,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            dataset.encode_sequences(remove_unused_columns=True)
        except Exception:
            # Fallback if val_data is already a fully processed tensor dataset
            dataset = val_data

        dataloader = DataLoader(
            dataset,
            batch_size=getattr(pred_config, "batch_size", 32),
            num_workers=getattr(pred_config, "num_workers", 4),
        )

        inference_engine = self.get_inference_engine(model, tokenizer)
        logits, _, _ = inference_engine.batch_infer(dataloader, do_pred=False)

        # Extract labels based on the dataset structure
        if hasattr(dataset, "labels"):
            labels = dataset.labels
        elif hasattr(val_data, "labels"):
            labels = val_data.labels
        elif isinstance(val_data, Subset) and hasattr(
            val_data.dataset, "labels"
        ):
            labels = [val_data.dataset.labels[i] for i in val_data.indices]
        else:
            labels = [
                item.get("labels", item.get("label")) for item in dataset
            ]

        if len(labels) == len(logits):
            calculated_metrics = inference_engine.calculate_metrics(
                logits, labels, plot=False
            )
            if metrics_list:
                return {
                    k: v
                    for k, v in calculated_metrics.items()
                    if k in metrics_list
                }
            return calculated_metrics
        else:
            raise ValueError(
                f"Label count ({len(labels)}) does not match "
                f"logit count ({len(logits)})."
            )

    def run_without_config(
        self, k_folds: int = 1, stratified: bool = False
    ) -> dict[str, Any]:
        """
        Run k-fold cross-validation benchmark.

        Args:
            k_folds: Number of folds for cross-validation
        Returns:
            Dictionary containing cross-validation results
                for each model and dataset

        """
        from sklearn.model_selection import KFold, StratifiedKFold

        if stratified:
            kfold = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=42
            )
        else:
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = {}

        if not self.prepared:
            raise ValueError(
                "Benchmark is not properly initialized "
                "with models and datasets."
            )

        model_names = self.prepared["models"]
        sources = self.prepared["sources"]
        tasks = self.prepared["tasks"]
        metrics_list = self.prepared.get("metrics", [])

        for mi, (model_name, model_path) in enumerate(model_names.items()):
            cv_results[model_name] = {}
            source = sources[mi]
            task_config = tasks[mi] if mi < len(tasks) else tasks[0]

            print(f"\nLoading model: {model_name}")
            model, tokenizer = load_model_and_tokenizer(
                model_path, task_config=task_config, source=source
            )

            for dataset_name, dataset_info in self.datasets_dict.items():
                dataset = dataset_info["dataset"]
                fold_scores = []
                indices = list(range(len(dataset)))

                if k_folds > 1:
                    kfold_split = kfold.split(indices)
                else:
                    kfold_split = [(indices, indices)]
                for fold, (_, val_idx) in enumerate(kfold_split):
                    print(
                        f"Running fold {fold + 1}/{k_folds} for "
                        f"{model_name} on {dataset_name}"
                    )

                    val_indices = val_idx.tolist()

                    # Support HuggingFace `.select()` or PyTorch `.Subset()`
                    if hasattr(dataset.dataset, "select"):
                        val_data = dataset.dataset.select(val_indices)
                    else:
                        val_data = Subset(dataset.dataset, val_indices)

                    fold_result = self.evaluate_single_model(
                        model=model,
                        tokenizer=tokenizer,
                        val_data=val_data,
                        metrics_list=metrics_list,
                    )
                    fold_scores.append(fold_result)

                # Aggregate metrics logic
                cv_results[model_name][dataset_name] = {
                    "fold_results": fold_scores
                }

                if fold_scores:
                    for k in fold_scores[0].keys():
                        try:
                            vals = [s[k] for s in fold_scores if k in s]
                            cv_results[model_name][dataset_name][
                                f"mean_{k}"
                            ] = float(np.mean(vals))
                            cv_results[model_name][dataset_name][
                                f"std_{k}"
                            ] = float(np.std(vals))
                        except (TypeError, ValueError):
                            continue

        return cv_results

    def plot(
        self,
        metrics: dict,
        show_score: bool = True,
        save_path: str | None = None,
        separate: bool = False,
        dataset: int | str = 0,
    ) -> None:
        """Plot the benchmark results.

        This method generates various types of plots based on the task type:
        - For classification tasks: bar charts for metrics and ROC curves
        - For regression tasks: bar charts for metrics and scatter plots
        - For token classification: bar charts for metrics only

        Args:
            metrics: Dictionary containing model metrics
            show_score: Whether to show the score on the plot
            save_path: Path to save the plot. If None,
                plots will be shown interactively
            separate: Whether to save the plots separately
            dataset: Index or name of the dataset to plot
                (if multiple datasets are provided)

        Returns:
            None
        """
        task_config = self.config["task"]
        task_type = task_config.task_type
        # Select dataset if multiple datasets are provided
        if isinstance(dataset, int):
            metrics = metrics[list(metrics.keys())[dataset]]
        elif isinstance(dataset, str):
            if dataset in metrics:
                metrics = metrics[dataset]
            else:
                raise ValueError(
                    f"Dataset name '{dataset}' not found in metrics."
                )
        else:
            metrics = metrics[next(iter(metrics.keys()))]
        if task_type in ["binary", "multiclass", "multilabel", "token"]:
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
            pbar = plot_bars(
                bars_data,
                show_score=show_score,
                save_path=bar_chart,
                separate=separate,
            )
            # Plot curve charts
            if task_type == "token":
                pline = None
            else:
                pline = plot_curve(
                    curves_data,
                    show_score=show_score,
                    save_path=line_chart,
                    separate=separate,
                )
            return pbar, pline
        elif task_type == "regression":
            # Prepare data for plotting
            bars_data, scatter_data = prepare_data(
                metrics, task_type=task_type
            )
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    bar_chart = save_path.replace(suffix, "_metrics" + suffix)
                    scatter_plot = save_path.replace(
                        suffix, "_scatter" + suffix
                    )
                else:
                    bar_chart = os.path.join(save_path, "metrics.pdf")
                    scatter_plot = os.path.join(save_path, "scatter.pdf")
            else:
                bar_chart = None
                scatter_plot = None
            # Plot bar charts
            pbar = plot_bars(
                bars_data,
                show_score=show_score,
                save_path=bar_chart,
                separate=separate,
            )
            # Plot scatter plots
            pdot = plot_scatter(
                scatter_data,
                show_score=show_score,
                save_path=scatter_plot,
                separate=separate,
            )
            return pbar, pdot
