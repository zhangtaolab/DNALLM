"""DNA Language Model Trainer Module.

This module implements the training process management for DNA language models,
    with the following main features:

1. DNATrainer Class
   - Unified management of model training, evaluation, and prediction processes
      - Support for multiple task types (
       classification,
       regression,
       masked language modeling)
   - Integration of task-specific prediction heads
   - Training parameter configuration
   - Training process monitoring and model saving

2. Core Features:
   - Model initialization and device management
   - Training parameter configuration
   - Training loop control
   - Evaluation metrics calculation
   - Model saving and loading
   - Prediction result generation

3. Supported Training Features:
   - Automatic evaluation and best model saving
   - Training log recording
   - Flexible batch size settings
   - Learning rate and weight decay configuration
   - Distributed training support
   - LoRA (Low-Rank Adaptation) for efficient fine-tuning

Usage Example:
    ```python
    trainer = DNATrainer(
        model=model,
        config=config,
        datasets=datasets
    )
    metrics = trainer.train()
    ```
"""

from pathlib import Path
from typing import Any
from collections.abc import Callable
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

from ..datahandling.data import DNADataset
from ..tasks.metrics import compute_metrics
from ..tasks.metrics import preprocess_logits_for_metrics as preprocess_logits


class DNATrainer:
    """DNA Language Model Trainer that supports multiple model types.

    This trainer class provides a unified interface for training, evaluating,
    and predicting with DNA language models. It supports various task types
    including classification, regression, and masked language modeling.
    Early stopping is supported via the callbacks configuration in TrainingConfig.
    QLoRA (4-bit quantized LoRA) is supported via use_qlora in TrainingConfig.

    Attributes:
        model: The DNA language model to be trained
        task_config: Configuration for the specific task
        train_config: Configuration for training parameters
        datasets: Dataset for training and evaluation
        extra_args: Additional training arguments
        trainer: HuggingFace Trainer instance
        training_args: Training arguments configuration
        data_split: Available dataset splits

    Examples:
        Standard LoRA training:
        ```python
        trainer = DNATrainer(
            model=model,
            config=config,
            datasets=datasets,
            use_lora=True,
        )
        metrics = trainer.train()
        ```

        QLoRA training (4-bit quantization):
        ```python
        # Model must be loaded with quantization_config before passing to trainer
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            task_config=task_config,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            },
        )
        trainer = DNATrainer(
            model=model,
            config=config,
            datasets=datasets,
            use_lora=True,
        )
        metrics = trainer.train()
        ```

    Plotting:
        After training, generate visualization plots:
        ```python
        trainer.plot_history(output_dir="./plots")
        ```
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        datasets: DNADataset | None = None,
        extra_args: dict | None = None,
        use_lora: bool = False,
    ):
        """Initialize the DNA trainer.

        Args:
            model: The DNA language model to be trained
                        config: Configuration dictionary containing task and
                training settings
            datasets: Dataset for training and evaluation
            extra_args: Additional training arguments to override defaults
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        self.model = model
        self.task_config = config["task"]
        self.train_config = config["finetune"]
        self.datasets = datasets
        self.extra_args = extra_args
        self.use_lora = use_lora

        # LoRA / QLoRA
        if use_lora:
            from ..models.model import peft_forward_compatiable

            print("[Info] Applying LoRA to the model...")

            # QLoRA: prepare 4-bit model for k-bit training
            if self.train_config.use_qlora:
                from peft import prepare_model_for_kbit_training

                print("[Info] Preparing model for 4-bit QLoRA training...")
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(**config["lora"].dict())
            model = peft_forward_compatiable(model)
            self.model = get_peft_model(model, lora_config)
            self.model.print_trainable_parameters()

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"[Info] Using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)

        self.set_up_trainer()

    def set_up_trainer(self):
        """Set up the HuggingFace Trainer with appropriate configurations.

        This method configures the training environment by:
        1. Setting up training arguments from configuration
        2. Configuring dataset splits (train/eval/test)
        3. Setting up task-specific metrics computation
        4. Configuring appropriate data collator for different task types
        5. Initializing the HuggingFace Trainer instance

        The method automatically handles:
        - Dataset split detection and validation
        - Task-specific data collator selection
        - Evaluation strategy configuration
        - Metrics computation setup
        """
        # Setup training arguments
        training_args = self.train_config.model_dump()
        if self.extra_args:
            training_args.update(self.extra_args)
        # Remove non-TrainingArguments fields
        training_args.pop("callbacks", None)
        training_args.pop("hyperparameter_search", None)
        training_args.pop("use_qlora", None)
        training_args.pop("quantization_config", None)
        self._save_safetensors = training_args.pop("save_safetensors", True)
        self.training_args = TrainingArguments(
            **training_args,
        )
        self.training_args.remove_unused_columns = (
            False
            if self.use_lora or "DNALLMforSequenceClassification" in self.model.__class__.__name__
            else self.training_args.remove_unused_columns
        )

        # Enable gradient checkpointing for QLoRA (required for memory efficiency)
        if self.train_config.use_qlora and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("[Info] Gradient checkpointing enabled for QLoRA.")
        if self.datasets is None:
            raise ValueError("Datasets are required for training")
        # Check if the dataset has been split
        if isinstance(self.datasets.dataset, DatasetDict):
            self.data_split = self.datasets.dataset.keys()
        else:
            self.data_split = []
        # Get datasets
        if "train" in self.data_split:
            train_dataset = self.datasets.dataset["train"]
        else:
            if len(self.data_split) == 0:
                train_dataset = self.datasets.dataset
            else:
                raise KeyError("Cannot find train data.")
        eval_key = [x for x in self.data_split if x not in ["train", "test"]]
        if eval_key:
            eval_dataset = self.datasets.dataset[eval_key[0]]
        elif "test" in self.data_split:
            eval_dataset = self.datasets.dataset["test"]
        else:
            eval_dataset = None
            self.training_args.eval_strategy = "no"

        # Set problem type specific settings
        if self.task_config.task_type == "regression":
            self.model.config.problem_type = "regression"
        # Get compute metrics
        if self.task_config.task_type in ["mask", "generation", "embedding"]:
            compute_metrics = None
        else:
            compute_metrics = self.compute_task_metrics()
        # Set data collator
        if self.task_config.task_type == "mask":
            from transformers import DataCollatorForLanguageModeling

            mlm_probability = self.task_config.mlm_probability
            mlm_probability = mlm_probability if mlm_probability else 0.15
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,  # type: ignore
                mlm=True,
                mlm_probability=mlm_probability,
            )
        elif self.task_config.task_type == "generation":
            from transformers import DataCollatorForLanguageModeling

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,  # type: ignore[arg-type]
                mlm=False,
            )
        else:
            data_collator = None

        # Assemble callbacks
        callbacks = []
        if (
            self.train_config.callbacks
            and self.train_config.callbacks.early_stopping
            and self.train_config.callbacks.early_stopping.patience is not None
        ):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_config.callbacks.early_stopping.patience,
                    early_stopping_threshold=self.train_config.callbacks.early_stopping.threshold,
                )
            )
            if not self.training_args.load_best_model_at_end:
                print(
                    "[Warning] Early stopping enabled but load_best_model_at_end=False. "
                    "Enabling load_best_model_at_end."
                )
                self.training_args.load_best_model_at_end = True

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits,
            callbacks=callbacks,
        )

    def customize_trainer(self, trainer_cls: Trainer):
        """Customize the HuggingFace Trainer instance.

        This method allows users to replace the default Trainer instance
        with a custom one, enabling advanced customization of the training
        process.

        Args:
            trainer_cls: A custom HuggingFace Trainer instance to replace the
                default one
        """
        # Use custom loss function if provided
        if isinstance(trainer_cls, type):
            # Directly replace the class of the existing trainer
            self.trainer.__class__ = trainer_cls
        else:
            # Replace the entire trainer instance
            self.trainer = trainer_cls

    def compute_task_metrics(self) -> Callable[..., dict[str, float]]:
        """Compute task-specific evaluation metrics.

        This method returns a callable function that computes appropriate
        metrics
                for the specific task type (classification, regression, etc.).

                Returns:
        Callable: A function that computes metrics for the specific task type
        """
        return compute_metrics(self.task_config)  # type: ignore[no-any-return]

    def _create_hp_space_fn(self) -> Callable:
        """Create the hp_space function for Optuna hyperparameter search.

        Returns a callable that takes an Optuna trial and returns
        a dict of hyperparameter values for TrainingArguments.
        """
        search_config = self.train_config.hyperparameter_search
        if not search_config or not search_config.search_space:
            return lambda trial: {}

        def hp_space(trial) -> dict[str, Any]:
            result = {}
            for param_name, distribution in search_config.search_space.items():
                if distribution.type == "float":
                    result[param_name] = trial.suggest_float(
                        param_name,
                        distribution.low,
                        distribution.high,
                        log=distribution.log,
                    )
                elif distribution.type == "int":
                    kwargs = {
                        "name": param_name,
                        "low": int(distribution.low),
                        "high": int(distribution.high),
                    }
                    if distribution.step is not None:
                        kwargs["step"] = distribution.step
                    result[param_name] = trial.suggest_int(**kwargs)
            return result

        return hp_space

    def train(self, save_tokenizer: bool = True) -> dict[str, float]:
        """Train the model and return training metrics.

        This method executes the training process using the configured
        HuggingFace Trainer, automatically saving the best model and optionally
        the tokenizer.

                Args:
            save_tokenizer: Whether to save the tokenizer along with the model,
                default True

                Returns:
            Dictionary containing training metrics including loss, learning
            rate, etc.
        """
        self.model.train()
        train_result = self.trainer.train()
        metrics: dict[str, float] = train_result.metrics
        # Save the model
        self.trainer.save_model()
        # check if have save_pretrained method
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(
                self.train_config.output_dir,
                safe_serialization=self._save_safetensors,
            )
        if save_tokenizer:
            self.datasets.tokenizer.save_pretrained(self.train_config.output_dir)  # type: ignore
        return metrics

    def search(self, save_tokenizer: bool = True) -> dict[str, Any]:
        """Run hyperparameter search using Optuna backend.

        This method runs multiple training trials with different
        hyperparameters and returns the best run configuration.

        Args:
            save_tokenizer: Whether to save the tokenizer with the best model.

        Returns:
            Dictionary containing:
            - "best_run": The best run object from hyperparameter_search
            - "best_hyperparameters": Dict of best hyperparameter values
            - "best_metric": The best objective metric value
        """
        if optuna is None:
            raise ImportError(
                "Optuna is required for hyperparameter search. Install it with: pip install optuna"
            )

        search_config = self.train_config.hyperparameter_search
        if not search_config or search_config.n_trials <= 0:
            raise ValueError(
                "Hyperparameter search is disabled. Set n_trials > 0 in "
                "hyperparameter_search configuration."
            )

        self.model.train()

        best_run = self.trainer.hyperparameter_search(
            direction=search_config.direction,
            backend="optuna",
            n_trials=search_config.n_trials,
            hp_space=self._create_hp_space_fn(),
            study_name=search_config.study_name,
        )

        result = {
            "best_run": best_run,
            "best_hyperparameters": best_run.hyperparameters,
            "best_metric": getattr(best_run, "objective", None),
        }

        # Save the best model
        self.trainer.save_model()
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(
                self.train_config.output_dir,
                safe_serialization=self._save_safetensors,
            )
        if save_tokenizer:
            self.datasets.tokenizer.save_pretrained(self.train_config.output_dir)  # type: ignore

        return result

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the evaluation dataset.

        This method runs evaluation on the configured evaluation dataset and
        returns task-specific metrics.

                Returns:
        Dictionary containing evaluation metrics for the current model state
        """
        self.model.eval()
        result: dict[str, float] = self.trainer.evaluate()
        return result

    def infer(self) -> dict[str, float]:
        """Generate inference results on the test dataset.

        This method generates inference results on the test dataset if
        available and returns both predictions and evaluation metrics.

        Returns:
                        Dictionary containing inference results and
                metrics if test dataset exists,
            otherwise empty dictionary
        """
        self.model.eval()
        result = {}
        if "test" in self.data_split:
            test_dataset = self.datasets.dataset["test"]  # type: ignore
            result = self.trainer.predict(test_dataset)
        return result

    def plot_history(
        self,
        output_dir: str | None = None,
        plot_loss: bool = True,
        plot_lr: bool = True,
    ) -> dict[str, Path]:
        """Generate training visualization plots from trainer state.

        This is a convenience method that delegates to the standalone
        plotting utilities in dnallm.utils.training_plots.

        Args:
            output_dir: Directory to save plots. Defaults to training output_dir.
            plot_loss: Whether to generate loss curve plot.
            plot_lr: Whether to generate learning rate schedule plot.

        Returns:
            Dictionary mapping plot names to saved file paths.
        """
        from pathlib import Path

        from dnallm.utils.training_plots import plot_loss_curve, plot_lr_schedule

        plot_dir = output_dir or self.train_config.output_dir or "."
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path] = {}
        log_history = self.trainer.state.log_history

        if plot_loss:
            loss_path = plot_path / "training_loss.png"
            plot_loss_curve(log_history, output_path=loss_path)
            results["loss_curve"] = loss_path

        if plot_lr:
            lr_path = plot_path / "lr_schedule.png"
            plot_lr_schedule(log_history, output_path=lr_path)
            results["lr_schedule"] = lr_path

        return results
