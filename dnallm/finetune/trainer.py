"""DNA Large Language Model Trainer Module.

This module implements the training process management for DNA large language models,
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
from collections.abc import Mapping
from typing import Any
from collections.abc import Callable
from datetime import datetime, timezone
import json
import math
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
import transformers
from packaging.version import Version
from peft import get_peft_model, LoraConfig

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

from ..datahandling.data import DNADataset
from ..tasks.metrics import compute_metrics
from ..tasks.metrics import preprocess_logits_for_metrics as preprocess_logits

transformers_version = Version(str(transformers.__version__))

# Pure-timing keys trainer.predict reports alongside metrics; evaluate(split=...)
# separates them into the JSON "runtime" block instead of the canonical
# metric-names dict (they are not metric-registry names).
PREDICT_RUNTIME_KEYS = ("runtime", "samples_per_second", "steps_per_second")


class DNATrainer:
    """DNA Large Language Model Trainer that supports multiple model types.

    This trainer class provides a unified interface for training, evaluating,
    and predicting with DNA large language models. It supports various task types
    including classification, regression, and masked language modeling.
    Early stopping is supported via the callbacks configuration in TrainingConfig.
    QLoRA (4-bit quantized LoRA) is supported via use_qlora in TrainingConfig.

    Evaluation semantics:
        The evaluation split is the first non-train/non-test split
        (dev/validation) present in the dataset. The test split is held out
        from evaluation unless ``finetune.allow_test_as_eval: true`` is set —
        when no dev split exists, per-step evaluation is disabled (with a
        loud warning) instead of silently evaluating on the test split, so
        held-out test metrics cannot leak into per-step evaluation or
        best-model selection. ``evaluate(split=...)`` evaluates the model
        you ended training with — the best checkpoint when
        load_best_model_at_end or early stopping fired, otherwise
        final-epoch weights.

    Attributes:
        model: The DNA large language model to be trained
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
        config: Mapping[str, Any],
        datasets: DNADataset | None = None,
        extra_args: dict | None = None,
        use_lora: bool = False,
    ):
        """Initialize the DNA trainer.

        Args:
            model: The DNA large language model to be trained
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
        training_args.pop("use_ia3", None)
        training_args.pop("allow_test_as_eval", None)
        training_args.pop("quantization_config", None)
        self._save_safetensors = training_args.pop("save_safetensors", True)
        # transformers v5 removed warmup_ratio from TrainingArguments;
        # convert it to warmup_steps once the train dataset size is known
        self._warmup_ratio = training_args.pop("warmup_ratio", None)
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
            # EVAL-01 guard: the test split is held out unless explicitly opted in.
            if self.train_config.allow_test_as_eval:
                eval_dataset = self.datasets.dataset["test"]
                print(
                    "[Warning] allow_test_as_eval=true: the test split will serve as "
                    "the evaluation set for per-step evaluation and best-model "
                    "selection. Metrics computed on it are leaked and must not be "
                    "reported as held-out performance."
                )
            else:
                eval_dataset = None
                self.training_args.eval_strategy = "no"
                print(
                    "[Warning] No dev split present: the test split is excluded from "
                    "evaluation and per-step evaluation is disabled. This differs "
                    "from previous dnallm versions, which silently used the test "
                    "split as the evaluation set. Set finetune.allow_test_as_eval="
                    "true to evaluate on the test split explicitly."
                )
        else:
            eval_dataset = None
            self.training_args.eval_strategy = "no"
        # EVAL-01: best-model selection needs an evaluation set no matter which
        # branch above excluded evaluation (held-out test split, train-only
        # split, or unsplit dataset) — fail loudly instead of silently never
        # selecting a best model (this mutation happens after
        # TrainingArguments.__post_init__, so transformers cannot catch it).
        if eval_dataset is None and self.training_args.load_best_model_at_end:
            raise ValueError(
                "load_best_model_at_end requires an evaluation split, but none is "
                "available (no dev split present, and the test split — if any — is "
                "excluded from evaluation when allow_test_as_eval is False). Provide "
                "a dev/validation split or set finetune.allow_test_as_eval=true."
            )

        # Convert warmup_ratio to warmup_steps (warmup_ratio was removed in transformers v5)
        if self._warmup_ratio and not self.training_args.warmup_steps:
            if self.training_args.max_steps and self.training_args.max_steps > 0:
                num_training_steps = self.training_args.max_steps
            else:
                steps_per_epoch = math.ceil(
                    len(train_dataset) / self.training_args.per_device_train_batch_size
                )
                num_training_steps = (
                    steps_per_epoch // self.training_args.gradient_accumulation_steps
                ) * self.training_args.num_train_epochs
            self.training_args.warmup_steps = int(self._warmup_ratio * num_training_steps)

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
            from transformers import DataCollatorForLanguageModeling  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live

            mlm_probability = self.task_config.mlm_probability
            mlm_probability = mlm_probability if mlm_probability else 0.15
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
            )
        elif self.task_config.task_type == "generation":
            from transformers import DataCollatorForLanguageModeling  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live

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
            # EVAL-01: best-model selection cannot be resurrected without an
            # eval set (PITFALLS #1 neighbor path).
            if eval_dataset is None:
                raise ValueError(
                    "Early stopping requires an evaluation split, but no eval dataset "
                    "is available (no dev split, and the test split is excluded from "
                    "evaluation unless allow_test_as_eval=true). Provide a "
                    "dev/validation split or set finetune.allow_test_as_eval=true."
                )
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
            # Transformers 5 enforces safetensors
            if transformers_version >= Version("5.0.0"):
                if self._save_safetensors:
                    self.model.save_pretrained(
                        self.train_config.output_dir,
                    )
                else:
                    torch.save(
                        self.model.state_dict(), f"{self.train_config.output_dir}/pytorch_model.bin"
                    )
            else:
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
        # check if have save_pretrained method
        if hasattr(self.model, "save_pretrained"):
            # Transformers 5 enforces safetensors
            if transformers_version >= Version("5.0.0"):
                if self._save_safetensors:
                    self.model.save_pretrained(
                        self.train_config.output_dir,
                    )
                else:
                    torch.save(
                        self.model.state_dict(), f"{self.train_config.output_dir}/pytorch_model.bin"
                    )
            else:
                self.model.save_pretrained(
                    self.train_config.output_dir,
                    safe_serialization=self._save_safetensors,
                )
        if save_tokenizer:
            self.datasets.tokenizer.save_pretrained(self.train_config.output_dir)  # type: ignore

        return result

    def evaluate(
        self,
        split: str | None = None,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model on a named split or the configured evaluation set.

        With ``split=None`` this keeps the Hugging Face ``Trainer.evaluate``
        calling convention for external HF-ecosystem callers: the legacy
        kwargs (``eval_dataset``, ``ignore_keys``, ``metric_key_prefix``)
        are forwarded to ``self.trainer.evaluate`` unchanged, and calling
        ``evaluate()`` with no arguments invokes ``self.trainer.evaluate()``
        with no kwargs.

        With ``split`` set to any split key present in the dataset dict, the
        held-out split is routed through ``self.trainer.predict`` — never
        ``trainer.evaluate`` — the ``test_`` predict prefix is stripped from
        the metric keys to produce canonical, unprefixed names, and a result
        JSON ``eval_{split}_result.json`` containing ``{"split", "timestamp",
        "metrics", "runtime"}`` is written under the finetune ``output_dir``.
        Pure-timing predict keys (``runtime``, ``samples_per_second``,
        ``steps_per_second``) are separated into the ``runtime`` block;
        ``loss`` stays among the metrics. This evaluates the weights the
        trainer currently holds; no checkpoint is reloaded and no checkpoint
        parameter exists.

        Args:
            split: Dataset split key to evaluate (e.g. "test", "val"). Any
                key present in the dataset dict is accepted.
            eval_dataset: Legacy kwarg forwarded to ``trainer.evaluate``
                (ignored when ``split`` is given).
            ignore_keys: Keys to ignore during evaluation/prediction.
            metric_key_prefix: Metric prefix for the legacy
                ``trainer.evaluate`` path.

        Returns:
            Dictionary of evaluation metrics (canonical, unprefixed metric
            names when ``split`` is given).

        Raises:
            ValueError: If ``split`` is not a key of the dataset dict.
        """
        if split is None:
            legacy_kwargs: dict[str, Any] = {}
            if eval_dataset is not None:
                legacy_kwargs["eval_dataset"] = eval_dataset
            if ignore_keys is not None:
                legacy_kwargs["ignore_keys"] = ignore_keys
            if metric_key_prefix != "eval":
                legacy_kwargs["metric_key_prefix"] = metric_key_prefix
            self.model.eval()
            result: dict[str, float] = self.trainer.evaluate(**legacy_kwargs)
            return result
        if split not in self.data_split:
            raise ValueError(
                f"Split '{split}' not found in dataset; available splits: {sorted(self.data_split)}"
            )
        self.model.eval()
        predict_result = self.trainer.predict(
            self.datasets.dataset[split],  # type: ignore
            ignore_keys=ignore_keys,
        )
        stripped: dict[str, float] = {
            key.removeprefix("test_"): value for key, value in predict_result.metrics.items()
        }
        runtime: dict[str, float] = {
            key: stripped.pop(key) for key in PREDICT_RUNTIME_KEYS if key in stripped
        }
        metrics: dict[str, float] = stripped
        result_path = Path(self.train_config.output_dir or ".") / f"eval_{split}_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics,
                    "runtime": runtime,
                },
                f,
                indent=2,
            )
        return metrics

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
