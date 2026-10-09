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
   - IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
     for parameter-efficient fine-tuning, symmetric to LoRA

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
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
import json
import math
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
import transformers
from packaging.version import Version
from peft import get_peft_model, LoraConfig, IA3Config

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

from ..configuration.configs import Ia3Config
from ..datahandling.data import DNADataset
from ..tasks.metrics import compute_metrics
from ..tasks.metrics import preprocess_logits_for_metrics as preprocess_logits

transformers_version = Version(str(transformers.__version__))

# Field names peft's IA3Config actually accepts (peft 0.14-0.21 span: exclude_modules
# landed mid-span) — dnallm's Ia3Config fields outside this set are dropped rather
# than crashing older peft with an unexpected-kwarg TypeError.
PEFT_IA3_FIELD_NAMES = frozenset(f.name for f in dataclass_fields(IA3Config))


def _load_peft_presets() -> dict:
    """Load the packaged per-family PEFT presets table (PEFT-02).

    Reads ``dnallm/configuration/presets/lora_targets.yaml`` via
    importlib.resources (wheel-safe, never CWD-relative) and validates the
    structure at load time so a corrupted table fails loudly here instead of
    silently freezing a backbone mid-training.

    Returns:
        The ``families`` mapping: family key -> preset row dict.

    Raises:
        ValueError: If the resource is missing or any row is malformed
            (empty target lists, feedforward modules outside the IA³ targets,
            or an inverted ratio band).
    """
    global _PEFT_PRESET_CACHE
    if _PEFT_PRESET_CACHE is not None:
        return _PEFT_PRESET_CACHE

    import yaml
    from importlib import resources

    try:
        resource = resources.files("dnallm.configuration").joinpath("presets/lora_targets.yaml")
        with resource.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise ValueError(f"Failed to load the packaged PEFT presets: {e}") from e

    families = data.get("families") if isinstance(data, dict) else None
    if not isinstance(families, dict) or not families:
        raise ValueError(
            "The packaged PEFT presets are malformed: the 'families' mapping is missing or empty."
        )
    for family, row in families.items():
        for key in ("lora_target_modules", "ia3_target_modules"):
            if not isinstance(row.get(key), list) or not row.get(key):
                raise ValueError(
                    f"The PEFT preset for family '{family}' is malformed: "
                    f"'{key}' must be a non-empty list."
                )
        ff = row.get("feedforward_modules") or []
        if not set(ff) <= set(row["ia3_target_modules"]):
            raise ValueError(
                f"The PEFT preset for family '{family}' is malformed: "
                f"feedforward_modules must be a subset of ia3_target_modules."
            )
        for key in ("ia3_ratio_band", "lora_ratio_band"):
            band = row.get(key)
            if not isinstance(band, list) or len(band) != 2 or not band[0] <= band[1]:
                raise ValueError(
                    f"The PEFT preset for family '{family}' is malformed: "
                    f"'{key}' must be a [lo, hi] pair with lo <= hi."
                )
    _PEFT_PRESET_CACHE = families
    return _PEFT_PRESET_CACHE


_PEFT_PRESET_CACHE: dict | None = None


def _resolve_peft_preset(model: Any) -> tuple[str, dict, str]:
    """Resolve the preset row for a live model (PEFT-02 auto-selection).

    Matching is two-tier: name markers from the table against the model's
    load path (longest marker wins), then the live ``config.model_type``.

    Args:
        model: The live backbone (its ``config`` carries ``_name_or_path``
            and ``model_type``).

    Returns:
        (family key, preset row, human-readable match description).

    Raises:
        ValueError: If no preset row matches — the user must set
            target_modules explicitly. Never a silent fallback.
    """
    families = _load_peft_presets()
    config_obj = getattr(model, "config", None)
    model_type = getattr(config_obj, "model_type", None)
    if not isinstance(model_type, str):
        model_type = None
    name_path = getattr(config_obj, "_name_or_path", None)
    name_blob = name_path.lower() if isinstance(name_path, str) else ""

    if name_blob:
        ranked = sorted(
            families.items(),
            key=lambda kv: -max(len(m) for m in kv[1].get("match_names") or []),
        )
        for family, row in ranked:
            for marker in row.get("match_names") or []:
                if marker in name_blob:
                    return family, row, f"name marker '{marker}'"
    if model_type:
        for family, row in families.items():
            if model_type in (row.get("model_types") or []):
                return family, row, f"config.model_type '{model_type}'"

    raise ValueError(
        f"No PEFT target-module preset found for this model (load path "
        f"'{name_blob or '<unknown>'}', model_type '{model_type}'). Set "
        f"target_modules explicitly in the lora:/ia3: config section, or run "
        f"with finetune.peft_dry_run=true to inspect the module names."
    )


def _peft_dry_run_report(model: Any, target_modules: list[str]) -> list[str]:
    """Match target_modules against the live model's modules (D-03).

    Mirrors peft's own matching rule (exact name or dotted-suffix match) and
    raises on zero matches — the silent-module-skip countermeasure.

    Args:
        model: The live backbone to inspect.
        target_modules: The final (preset- or user-resolved) target names.

    Returns:
        The matched module names.

    Raises:
        ValueError: If no module matches (wrong names for this backbone).
    """
    matched = []
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if any(name == target or name.endswith("." + target) for target in target_modules):
            matched.append(name)
    if not matched:
        raise ValueError(
            f"PEFT dry run: target_modules {target_modules} matched 0 of "
            f"{len(module_names)} modules — attaching the adapter would "
            f"silently freeze the whole model. Check the module names against "
            f"this backbone (finetune.peft_dry_run)."
        )
    shown = ", ".join(matched[:10])
    more = f" ... and {len(matched) - 10} more" if len(matched) > 10 else ""
    print(
        f"[Info] PEFT dry run: {len(matched)} modules matched target_modules "
        f"{target_modules}: {shown}{more}"
    )
    return matched


def _guard_trainable_ratio(
    model: Any, preset_family: str | None, preset_row: dict | None, adapter_kind: str
) -> tuple[int, int, float]:
    """Enforce the trainable-parameter ratio after adapter attach (D-04).

    Computed directly from requires_grad tensors (never parsed from
    print_trainable_parameters output). With a preset active the preset's
    ratio band is enforced; with user-supplied target_modules the guard
    enforces ratio > 0 (a fully frozen model is always wrong).

    Args:
        model: The adapter-wrapped model.
        preset_family: Family key when auto-selection fired, else None.
        preset_row: The preset row when auto-selection fired, else None.
        adapter_kind: "ia3" or "lora".

    Returns:
        (trainable count, total count, ratio).

    Raises:
        ValueError: Outside the preset band, or zero trainable parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / max(total, 1)
    if preset_row is not None:
        lo, hi = preset_row[f"{adapter_kind}_ratio_band"]
        if not (lo <= ratio <= hi):
            raise ValueError(
                f"PEFT preset '{preset_family}' attached {trainable}/{total} "
                f"trainable parameters (ratio {ratio:.2e}); expected band "
                f"[{lo:.2e}, {hi:.2e}]. A silent module-skip is the likely "
                f"cause — check target_modules against this backbone."
            )
    elif trainable == 0:
        raise ValueError(
            f"The PEFT adapter attached 0 trainable parameters out of {total}: "
            f"the model is fully frozen. target_modules matched no modules — "
            f"check the module names against this backbone."
        )
    return trainable, total, ratio


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

        # The ctor kwarg is invisible to Pydantic (use_lora is not a
        # TrainingConfig field), so the LoRA x IA³ combination is rejected
        # here, at trainer-init time, with a matchable message.
        if use_lora and self.train_config.use_ia3:
            raise ValueError(
                "use_lora=True cannot be combined with finetune.use_ia3=true: "
                "LoRA and IA³ are alternative adapter methods. Pass use_lora=False "
                "when finetune.use_ia3 is true."
            )

        # Shared PEFT target resolution: preset auto-selection (PEFT-02) and
        # the dry-run validator (D-03) run ahead of either adapter branch so
        # LoRA and IA³ share one resolution path.
        peft_kind = "lora" if use_lora else ("ia3" if self.train_config.use_ia3 else None)
        # WR-01: the dry-run flag is validate-and-exit by contract; without
        # an adapter method there is nothing to validate, and proceeding
        # would silently run the FULL fine-tune the user asked not to run.
        if self.train_config.peft_dry_run and peft_kind is None:
            raise ValueError(
                "finetune.peft_dry_run=true requires an adapter method: pass "
                "use_lora=True or set finetune.use_ia3=true. Refusing to start a "
                "full training run under a dry-run flag."
            )
        preset_family: str | None = None
        preset_row: dict | None = None
        if peft_kind is not None:
            section = config["lora"] if peft_kind == "lora" else config.get("ia3", Ia3Config())
            if peft_kind == "ia3" and "ia3" not in config:
                # Register the default section so the IA³ branch below (and
                # preset injection) mutate the same object the config carries.
                config["ia3"] = section
            targets = getattr(section, "target_modules", None)
            if targets is None:
                preset_family, preset_row, matched_by = _resolve_peft_preset(model)
                targets = list(
                    preset_row["ia3_target_modules"]
                    if peft_kind == "ia3"
                    else preset_row["lora_target_modules"]
                )
                if peft_kind == "ia3" and getattr(section, "feedforward_modules", None) is None:
                    section.feedforward_modules = list(preset_row.get("feedforward_modules") or [])
                section.target_modules = targets
                kind_label = "IA³" if peft_kind == "ia3" else "LoRA"
                print(
                    f"[Info] {kind_label} preset '{preset_family}' selected "
                    f"(matched by {matched_by}): target_modules={targets}"
                )
            if self.train_config.peft_dry_run:
                _peft_dry_run_report(model, section.target_modules)
                print("[Info] PEFT dry run complete — no training performed.")
                self._peft_dry_run = True
                return

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
            _guard_trainable_ratio(self.model, preset_family, preset_row, "lora")

        # IA³ (no k-bit prep: use_ia3 x use_qlora is rejected at config time)
        if self.train_config.use_ia3:
            from ..models.model import peft_forward_compatiable

            print("[Info] Applying IA³ to the model...")

            ia3_section = config.get("ia3", Ia3Config())
            peft_kwargs = {
                k: v for k, v in ia3_section.model_dump().items() if k in PEFT_IA3_FIELD_NAMES
            }
            ia3_config = IA3Config(**peft_kwargs)
            model = peft_forward_compatiable(model)
            self.model = get_peft_model(model, ia3_config)
            self.model.print_trainable_parameters()
            _guard_trainable_ratio(self.model, preset_family, preset_row, "ia3")

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
        training_args.pop("peft_dry_run", None)
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
            if self.use_lora
            or self.train_config.use_ia3
            or "DNALLMforSequenceClassification" in self.model.__class__.__name__
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
        if getattr(self, "_peft_dry_run", False):
            print("[Info] Skipping the training loop: finetune.peft_dry_run=true.")
            return {}
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
            ValueError: If ``split`` is not a key of the dataset dict, or if
                ``finetune.output_dir`` is not set (the result JSON would
                otherwise land in the current working directory).
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
        if not self.train_config.output_dir:
            raise ValueError(
                f"evaluate(split='{split}') writes eval_{split}_result.json under "
                "finetune.output_dir, but finetune.output_dir is not set. Set "
                "finetune.output_dir in the config so the result JSON has a "
                "deterministic home instead of the current working directory."
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
        result_path = Path(self.train_config.output_dir) / f"eval_{split}_result.json"
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
