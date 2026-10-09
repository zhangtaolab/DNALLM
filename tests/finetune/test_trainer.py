"""Fast unit tests for DNATrainer wiring.

The Hugging Face boundary (``Trainer`` / ``TrainingArguments``) is patched at
its import site in ``dnallm.finetune.trainer`` so every test exercises the
real task-config -> TrainingArguments mapping, metrics binding, LoRA/QLoRA
selection and early-stopping wiring without running any training. The slow
real-model file (``test_trainer_real_model.py``) owns the live-training path.

No test here performs a skip call, touches the network, or writes outside
pytest tmp_path.
"""

import json
import os
from unittest.mock import Mock, patch

import pytest
from conftest import SimpleDNATokenizer
from datasets import Dataset, DatasetDict
from packaging.version import Version

from dnallm.configuration.configs import (
    CallbackConfig,
    EarlyStoppingConfig,
    HyperparameterSearchConfig,
    LoraConfig,
    SearchSpaceDistribution,
    TaskConfig,
    load_config,
)
from dnallm.datahandling.data import DNADataset
from dnallm.finetune.trainer import DNATrainer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_finetune_config.yaml")


def make_datasets(splits, tmp_path=None):
    """Build a real DNADataset over tiny splits with a save-capable tokenizer.

    Named splits (even a single one) are wrapped in a DatasetDict; passing
    ``[None]`` yields an unsplit plain Dataset.
    """

    def one():
        return Dataset.from_dict({"sequence": ["ATCG"] * 4, "labels": [0, 1, 2, 1]})

    tokenizer = SimpleDNATokenizer()
    tokenizer.save_pretrained = Mock()
    if splits == [None]:
        return DNADataset(one(), tokenizer=tokenizer)
    return DNADataset(DatasetDict({name: one() for name in splits}), tokenizer=tokenizer)


@pytest.fixture
def trainer_config(tmp_path):
    """Load the tracked finetune fixture config with a tmp_path output dir."""
    config = load_config(CONFIG_PATH)
    config["finetune"].output_dir = str(tmp_path / "outputs")
    return config


@pytest.fixture
def mock_hf_boundary():
    """Patch Trainer and TrainingArguments at their import site."""
    with (
        patch("dnallm.finetune.trainer.Trainer") as trainer_cls,
        patch("dnallm.finetune.trainer.TrainingArguments") as args_cls,
    ):
        yield trainer_cls, args_cls


class TestTrainingArgumentsMapping:
    """Task-config -> TrainingArguments mapping at the mocked boundary."""

    def test_fixture_values_land_on_training_arguments(self, trainer_config, mock_hf_boundary):
        """The tracked config's epochs/bs/lr/report_to reach TrainingArguments."""
        _, args_cls = mock_hf_boundary

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"]))

        kwargs = args_cls.call_args.kwargs
        assert kwargs["num_train_epochs"] == 1
        assert kwargs["per_device_train_batch_size"] == 16
        assert kwargs["learning_rate"] == 2e-5
        assert kwargs["report_to"] == ["tensorboard"]

    def test_non_training_arguments_fields_are_popped(self, trainer_config, mock_hf_boundary):
        """Internal-only config fields never reach TrainingArguments."""
        _, args_cls = mock_hf_boundary

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"]))

        kwargs = args_cls.call_args.kwargs
        for popped in (
            "callbacks",
            "hyperparameter_search",
            "use_qlora",
            "use_ia3",
            "allow_test_as_eval",
            "quantization_config",
            "save_safetensors",
            "warmup_ratio",
        ):
            assert popped not in kwargs

    def test_extra_args_override_defaults(self, trainer_config, mock_hf_boundary):
        """extra_args override the config values handed to TrainingArguments."""
        _, args_cls = mock_hf_boundary

        DNATrainer(
            model=Mock(),
            config=trainer_config,
            datasets=make_datasets(["train", "val"]),
            extra_args={"learning_rate": 1e-3},
        )

        assert args_cls.call_args.kwargs["learning_rate"] == 1e-3

    def test_remove_unused_columns_disabled_for_classification_wrapper(
        self, trainer_config, mock_hf_boundary
    ):
        """The DNALLMforSequenceClassification wrapper keeps all columns."""

        class DNALLMforSequenceClassification(Mock):
            """Name-only stand-in for the classification wrapper."""

        DNATrainer(
            model=DNALLMforSequenceClassification(),
            config=trainer_config,
            datasets=make_datasets(["train", "val"]),
        )

        _, args_cls = mock_hf_boundary
        trainer = mock_hf_boundary[0].return_value
        assert trainer is not None
        # remove_unused_columns is forced False for the wrapper model
        assert args_cls.return_value.remove_unused_columns is False


class TestDatasetSplitWiring:
    """Dataset split detection and train/eval selection."""

    def test_datasets_required(self, trainer_config, mock_hf_boundary):
        """Construction without datasets raises."""
        with pytest.raises(ValueError, match="Datasets are required for training"):
            DNATrainer(model=Mock(), config=trainer_config, datasets=None)

    def test_unsplit_dataset_used_as_train_without_eval(self, trainer_config, mock_hf_boundary):
        """An unsplit dataset trains on itself with evaluation disabled."""
        trainer_cls, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False
        datasets = make_datasets([None])

        DNATrainer(model=Mock(), config=trainer_config, datasets=datasets)

        kwargs = trainer_cls.call_args.kwargs
        assert kwargs["train_dataset"] is datasets.dataset
        assert kwargs["eval_dataset"] is None
        assert args_cls.return_value.eval_strategy == "no"

    def test_eval_prefers_validation_split(self, trainer_config, mock_hf_boundary):
        """A validation split is preferred over test for evaluation."""
        trainer_cls, _ = mock_hf_boundary
        datasets = make_datasets(["train", "val", "test"])

        DNATrainer(model=Mock(), config=trainer_config, datasets=datasets)

        assert trainer_cls.call_args.kwargs["eval_dataset"] is datasets.dataset["val"]

    def test_eval_excludes_test_split_without_opt_in(self, trainer_config, mock_hf_boundary):
        """Without a dev split and allow_test_as_eval unset, the test split is
        excluded from evaluation (EVAL-01 guard)."""
        trainer_cls, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False
        datasets = make_datasets(["train", "test"])

        with patch("builtins.print"):
            DNATrainer(model=Mock(), config=trainer_config, datasets=datasets)

        assert trainer_cls.call_args.kwargs["eval_dataset"] is None
        assert args_cls.return_value.eval_strategy == "no"

    def test_train_only_split_disables_evaluation(self, trainer_config, mock_hf_boundary):
        """A lone train split disables the eval strategy."""
        trainer_cls, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train"]))

        assert trainer_cls.call_args.kwargs["eval_dataset"] is None
        assert args_cls.return_value.eval_strategy == "no"

    def test_missing_train_split_raises(self, trainer_config, mock_hf_boundary):
        """A split dict without train is rejected."""
        with pytest.raises(KeyError, match="Cannot find train data"):
            DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["val"]))


class TestEvalSemanticsGuard:
    """EVAL-01: the test split can never silently become the eval set."""

    def test_test_only_default_fires_flip_warn_exactly_once(self, trainer_config, mock_hf_boundary):
        """The guard WARN fires once at construction time, carrying all three facts."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False

        with patch("builtins.print") as mock_print:
            DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "test"])
            )

        flip_calls = [
            call
            for call in mock_print.call_args_list
            if all(
                fact in "".join(str(arg) for arg in call.args)
                for fact in ("[Warning]", "test split", "previous", "allow_test_as_eval")
            )
        ]
        assert len(flip_calls) == 1

    def test_opt_in_uses_test_as_eval_with_leak_warning(self, trainer_config, mock_hf_boundary):
        """allow_test_as_eval=True restores test-as-eval with a loud leak warning."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["finetune"].allow_test_as_eval = True
        datasets = make_datasets(["train", "test"])

        with patch("builtins.print") as mock_print:
            DNATrainer(model=Mock(), config=trainer_config, datasets=datasets)

        assert trainer_cls.call_args.kwargs["eval_dataset"] is datasets.dataset["test"]
        opt_in_calls = [
            call
            for call in mock_print.call_args_list
            if "allow_test_as_eval=true" in "".join(str(arg) for arg in call.args)
            and "leaked" in "".join(str(arg) for arg in call.args)
        ]
        assert len(opt_in_calls) == 1

    def test_train_only_emits_no_flip_warn(self, trainer_config, mock_hf_boundary):
        """A lone train split keeps today's behavior with no flip warning."""
        trainer_cls, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False

        with patch("builtins.print") as mock_print:
            DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train"]))

        assert trainer_cls.call_args.kwargs["eval_dataset"] is None
        assert args_cls.return_value.eval_strategy == "no"
        assert not [
            call
            for call in mock_print.call_args_list
            if "allow_test_as_eval" in "".join(str(arg) for arg in call.args)
        ]

    def test_unsplit_dataset_emits_no_flip_warn(self, trainer_config, mock_hf_boundary):
        """An unsplit dataset keeps today's behavior with no flip warning."""
        trainer_cls, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False

        with patch("builtins.print") as mock_print:
            DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets([None]))

        assert trainer_cls.call_args.kwargs["eval_dataset"] is None
        assert args_cls.return_value.eval_strategy == "no"
        assert not [
            call
            for call in mock_print.call_args_list
            if "allow_test_as_eval" in "".join(str(arg) for arg in call.args)
        ]


class TestEarlyStoppingCollision:
    """EVAL-01 collisions: best-model selection without an evaluation split."""

    def test_early_stopping_without_eval_split_raises(self, trainer_config, mock_hf_boundary):
        """Early stopping over a guarded test-only dataset raises with both remedies."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False
        trainer_config["finetune"].callbacks = CallbackConfig(
            early_stopping=EarlyStoppingConfig(patience=1)
        )

        with (
            patch("builtins.print"),
            pytest.raises(ValueError, match="Early stopping requires an evaluation split"),
        ):
            DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "test"])
            )

    def test_early_stopping_train_only_raises(self, trainer_config, mock_hf_boundary):
        """Early stopping over a train-only dataset raises the same ValueError."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False
        trainer_config["finetune"].callbacks = CallbackConfig(
            early_stopping=EarlyStoppingConfig(patience=2)
        )

        with (
            patch("builtins.print"),
            pytest.raises(ValueError, match="Early stopping requires an evaluation split"),
        ):
            DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train"]))

    def test_load_best_model_at_end_collision_raises(self, trainer_config, mock_hf_boundary):
        """A user-set load_best_model_at_end never survives the guard silently."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = True

        with (
            patch("builtins.print"),
            pytest.raises(ValueError, match="load_best_model_at_end requires an evaluation split"),
        ):
            DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "test"])
            )

    @pytest.mark.parametrize("splits", [["train"], [None]], ids=["train-only", "unsplit"])
    def test_load_best_model_at_end_collision_raises_without_test_split(
        self, trainer_config, mock_hf_boundary, splits
    ):
        """The collision guard also covers the train-only and unsplit paths."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = True

        with (
            patch("builtins.print"),
            pytest.raises(ValueError, match="load_best_model_at_end requires an evaluation split"),
        ):
            DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(splits))

    def test_opt_in_early_stopping_does_not_raise(self, trainer_config, mock_hf_boundary):
        """Opting into test-as-eval keeps early stopping working (opt-in edge)."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False
        trainer_config["finetune"].allow_test_as_eval = True
        trainer_config["finetune"].callbacks = CallbackConfig(
            early_stopping=EarlyStoppingConfig(patience=1)
        )

        with (
            patch("builtins.print"),
            patch("dnallm.finetune.trainer.EarlyStoppingCallback"),
        ):
            DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "test"])
            )

        # An eval set exists (test), so the force-enable path still works.
        assert args_cls.return_value.load_best_model_at_end is True


class TestEvaluateSplit:
    """evaluate(split=...) predict routing, canonical keys and result JSON."""

    def _guarded_trainer(self, trainer_config, mock_hf_boundary, splits=("train", "test")):
        """Build a DNATrainer over the given splits with the guard active."""
        _, args_cls = mock_hf_boundary
        args_cls.return_value.load_best_model_at_end = False
        datasets = make_datasets(list(splits))
        with patch("builtins.print"):
            return DNATrainer(model=Mock(), config=trainer_config, datasets=datasets), datasets

    def test_split_routes_through_predict_and_writes_result_json(
        self, trainer_config, mock_hf_boundary, tmp_path
    ):
        """split='test' predicts (never trainer.evaluate) and writes the JSON."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.predict.return_value.metrics = {
            "test_accuracy": 0.9,
            "test_AUROC": 0.8,
        }
        output_dir = tmp_path / "outputs"
        trainer_config["finetune"].output_dir = str(output_dir)

        trainer, datasets = self._guarded_trainer(trainer_config, mock_hf_boundary)
        result = trainer.evaluate(split="test")

        trainer_cls.return_value.predict.assert_called_once_with(
            datasets.dataset["test"], ignore_keys=None
        )
        trainer_cls.return_value.evaluate.assert_not_called()
        assert result == {"accuracy": 0.9, "AUROC": 0.8}

        result_path = output_dir / "eval_test_result.json"
        assert result_path.exists()
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["split"] == "test"
        assert "timestamp" in payload
        assert payload["metrics"] == {"accuracy": 0.9, "AUROC": 0.8}

    def test_unknown_split_raises_listing_available(self, trainer_config, mock_hf_boundary):
        """An absent split key raises a ValueError naming the available splits."""
        trainer, _ = self._guarded_trainer(trainer_config, mock_hf_boundary)

        with pytest.raises(ValueError, match="Split 'nonexistent' not found in dataset"):
            trainer.evaluate(split="nonexistent")

    def test_no_args_calls_trainer_evaluate_with_no_kwargs(self, trainer_config, mock_hf_boundary):
        """evaluate() with no arguments delegates with no kwargs (D-01)."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.evaluate.return_value = {"eval_loss": 0.3}
        trainer, _ = self._guarded_trainer(trainer_config, mock_hf_boundary, ("train", "val"))

        assert trainer.evaluate() == {"eval_loss": 0.3}
        trainer_cls.return_value.evaluate.assert_called_once_with()

    def test_legacy_kwargs_forward_unchanged(self, trainer_config, mock_hf_boundary):
        """Legacy HF kwargs pass through to trainer.evaluate (signature compat)."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.evaluate.return_value = {"eval_loss": 0.2}
        trainer, _ = self._guarded_trainer(trainer_config, mock_hf_boundary, ("train", "val"))
        eval_dataset = Mock()

        trainer.evaluate(
            eval_dataset=eval_dataset, ignore_keys=["logits"], metric_key_prefix="valid"
        )

        trainer_cls.return_value.evaluate.assert_called_once_with(
            eval_dataset=eval_dataset, ignore_keys=["logits"], metric_key_prefix="valid"
        )


class TestTaskTypeWiring:
    """Task-type specific settings, metrics binding and data collators."""

    def test_regression_sets_problem_type(self, trainer_config, mock_hf_boundary):
        """Regression tasks set problem_type on the model config."""
        trainer_config["task"] = TaskConfig(task_type="regression", num_labels=1)
        model = Mock()

        DNATrainer(model=model, config=trainer_config, datasets=make_datasets(["train", "val"]))

        assert model.config.problem_type == "regression"

    @pytest.mark.parametrize(
        ("task_type", "num_labels"),
        [
            ("binary", 2),
            ("multiclass", 3),
            ("multilabel", 3),
            ("regression", 1),
        ],
    )
    def test_compute_metrics_bound_for_prediction_tasks(
        self, trainer_config, mock_hf_boundary, task_type, num_labels
    ):
        """Prediction tasks bind a compute_metrics callable."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["task"] = TaskConfig(task_type=task_type, num_labels=num_labels)

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"]))

        assert callable(trainer_cls.call_args.kwargs["compute_metrics"])

    def test_mask_task_binds_mlm_collator_and_no_metrics(self, trainer_config, mock_hf_boundary):
        """Mask tasks use DataCollatorForLanguageModeling and no metrics."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["task"] = TaskConfig(task_type="mask", mlm_probability=0.2)

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"]))

        kwargs = trainer_cls.call_args.kwargs
        assert kwargs["compute_metrics"] is None
        collator = kwargs["data_collator"]
        assert collator.__class__.__name__ == "DataCollatorForLanguageModeling"
        assert collator.mlm_probability == 0.2

    def test_generation_task_binds_plain_mlm_collator(self, trainer_config, mock_hf_boundary):
        """Generation tasks use a whole-word collator without metrics."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["task"] = TaskConfig(task_type="generation")

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"]))

        kwargs = trainer_cls.call_args.kwargs
        assert kwargs["compute_metrics"] is None
        assert kwargs["data_collator"].mlm is False


class TestEarlyStopping:
    """Early stopping callback assembly."""

    def test_early_stopping_callback_wired(self, trainer_config, mock_hf_boundary):
        """Configured patience wires EarlyStoppingCallback with its args."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["finetune"].callbacks = CallbackConfig(
            early_stopping=EarlyStoppingConfig(patience=2, threshold=0.05)
        )
        mock_hf_boundary[1].return_value.load_best_model_at_end = False

        with patch("dnallm.finetune.trainer.EarlyStoppingCallback") as mock_esc:
            DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
            )

        mock_esc.assert_called_once_with(
            early_stopping_patience=2,
            early_stopping_threshold=0.05,
        )
        assert trainer_cls.call_args.kwargs["callbacks"] == [mock_esc.return_value]

    def test_load_best_model_forced_on_when_missing(self, trainer_config, mock_hf_boundary):
        """Early stopping with load_best_model_at_end off enables it loudly."""
        args_cls = mock_hf_boundary[1]
        trainer_config["finetune"].callbacks = CallbackConfig(
            early_stopping=EarlyStoppingConfig(patience=1, threshold=0.0)
        )
        args_cls.return_value.load_best_model_at_end = False

        with patch("dnallm.finetune.trainer.EarlyStoppingCallback"):
            DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
            )

        assert args_cls.return_value.load_best_model_at_end is True

    def test_no_early_stopping_without_patience(self, trainer_config, mock_hf_boundary):
        """A null patience yields an empty callbacks list."""
        trainer_cls, _ = mock_hf_boundary

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"]))

        assert trainer_cls.call_args.kwargs["callbacks"] == []


class TestLoraWiring:
    """LoRA / QLoRA model wrapping at the peft boundary."""

    def test_use_lora_wraps_model_via_peft(self, trainer_config, mock_hf_boundary):
        """use_lora applies LoraConfig + get_peft_model and trains the wrapper."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["lora"] = LoraConfig(r=4, lora_alpha=8)
        wrapped = Mock()

        with (
            patch("dnallm.finetune.trainer.get_peft_model", return_value=wrapped) as mock_gpm,
            patch("dnallm.finetune.trainer.LoraConfig") as mock_lora_config,
            patch("dnallm.models.model.peft_forward_compatiable", side_effect=lambda m: m),
        ):
            DNATrainer(
                model=Mock(),
                config=trainer_config,
                datasets=make_datasets(["train", "val"]),
                use_lora=True,
            )

        assert mock_lora_config.call_args.kwargs["r"] == 4
        assert mock_lora_config.call_args.kwargs["lora_alpha"] == 8
        assert mock_gpm.call_args.args[1] is mock_lora_config.return_value
        assert trainer_cls.call_args.kwargs["model"] is wrapped
        wrapped.print_trainable_parameters.assert_called_once()

    def test_use_qlora_prepares_kbit_model(self, trainer_config, mock_hf_boundary):
        """QLoRA prepares the 4-bit model and enables gradient checkpointing."""
        trainer_config["lora"] = LoraConfig()
        trainer_config["finetune"].use_qlora = True
        model = Mock()

        with (
            patch("dnallm.finetune.trainer.get_peft_model", return_value=Mock()),
            patch("dnallm.finetune.trainer.LoraConfig"),
            patch("dnallm.models.model.peft_forward_compatiable", side_effect=lambda m: m),
            patch("peft.prepare_model_for_kbit_training", side_effect=lambda m: m) as mock_prep,
        ):
            DNATrainer(
                model=model,
                config=trainer_config,
                datasets=make_datasets(["train", "val"]),
                use_lora=True,
            )

        mock_prep.assert_called_once_with(model)


class TestMultiGpu:
    """Multi-GPU DataParallel wrapping."""

    def test_multi_gpu_wraps_in_dataparallel(self, trainer_config, mock_hf_boundary):
        """More than one CUDA device wraps the model in DataParallel."""
        trainer_cls, _ = mock_hf_boundary
        wrapped = Mock()

        with (
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.nn.DataParallel", return_value=wrapped) as mock_dp,
        ):
            trainer = DNATrainer(
                model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
            )

        mock_dp.assert_called_once()
        assert trainer.model is wrapped
        assert trainer_cls.call_args.kwargs["model"] is wrapped

    def test_single_gpu_skips_dataparallel(self, trainer_config, mock_hf_boundary):
        """A single device leaves the model unwrapped."""
        model = Mock()
        with (
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.nn.DataParallel") as mock_dp,
        ):
            trainer = DNATrainer(
                model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
            )

        mock_dp.assert_not_called()
        assert trainer.model is model


class TestHyperparameterSpace:
    """Optuna hp_space construction."""

    def test_hp_space_empty_without_search_config(self, trainer_config, mock_hf_boundary):
        """No hyperparameter_search config yields an empty suggestion dict."""
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )

        assert trainer._create_hp_space_fn()(Mock()) == {}

    def test_hp_space_float_and_int_distributions(self, trainer_config, mock_hf_boundary):
        """Float and int distributions map onto trial suggest calls."""
        trainer_config["finetune"].hyperparameter_search = HyperparameterSearchConfig(
            search_space={
                "learning_rate": SearchSpaceDistribution(
                    type="float", low=1e-6, high=1e-3, log=True
                ),
                "per_device_train_batch_size": SearchSpaceDistribution(
                    type="int", low=4, high=32, step=4
                ),
            },
            n_trials=3,
        )
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        trial = Mock()

        result = trainer._create_hp_space_fn()(trial)

        trial.suggest_float.assert_called_once_with("learning_rate", 1e-6, 1e-3, log=True)
        trial.suggest_int.assert_called_once_with(
            name="per_device_train_batch_size", low=4, high=32, step=4
        )
        assert result == {
            "learning_rate": trial.suggest_float.return_value,
            "per_device_train_batch_size": trial.suggest_int.return_value,
        }

    def test_hp_space_int_without_step(self, trainer_config, mock_hf_boundary):
        """Int distributions without a step omit the step kwarg."""
        trainer_config["finetune"].hyperparameter_search = HyperparameterSearchConfig(
            search_space={"seed": SearchSpaceDistribution(type="int", low=1, high=42)},
        )
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        trial = Mock()

        trainer._create_hp_space_fn()(trial)

        assert trial.suggest_int.call_args.kwargs == {"name": "seed", "low": 1, "high": 42}


class TestSearch:
    """Optuna-backed hyperparameter search wiring."""

    def test_search_without_optuna_raises(self, trainer_config, mock_hf_boundary):
        """A missing optuna install raises an actionable ImportError."""
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )

        with patch("dnallm.finetune.trainer.optuna", None):
            with pytest.raises(ImportError, match="Optuna is required"):
                trainer.search()

    def test_search_disabled_raises(self, trainer_config, mock_hf_boundary):
        """A disabled search config raises instead of running."""
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )

        with pytest.raises(ValueError, match="Hyperparameter search is disabled"):
            trainer.search()

    def test_search_runs_optuna_backend_and_reports_best(self, trainer_config, mock_hf_boundary):
        """search() drives trainer.hyperparameter_search and returns the best run."""
        trainer_cls, _ = mock_hf_boundary
        search_config = HyperparameterSearchConfig(
            search_space={
                "learning_rate": SearchSpaceDistribution(type="float", low=1e-6, high=1e-3)
            },
            n_trials=3,
            direction="minimize",
            study_name="study-1",
        )
        trainer_config["finetune"].hyperparameter_search = search_config
        datasets = make_datasets(["train", "val"])
        best_run = Mock(hyperparameters={"learning_rate": 1e-4}, objective=0.12)
        trainer_cls.return_value.hyperparameter_search.return_value = best_run

        trainer = DNATrainer(model=Mock(), config=trainer_config, datasets=datasets)
        result = trainer.search(save_tokenizer=True)

        kwargs = trainer_cls.return_value.hyperparameter_search.call_args.kwargs
        assert kwargs["direction"] == "minimize"
        assert kwargs["backend"] == "optuna"
        assert kwargs["n_trials"] == 3
        assert kwargs["study_name"] == "study-1"
        assert callable(kwargs["hp_space"])
        assert result["best_run"] is best_run
        assert result["best_hyperparameters"] == {"learning_rate": 1e-4}
        assert result["best_metric"] == 0.12
        trainer_cls.return_value.save_model.assert_called_once()
        datasets.tokenizer.save_pretrained.assert_called_once_with(
            trainer_config["finetune"].output_dir
        )


class TestTrainEvaluate:
    """train/evaluate flows against the mocked Trainer boundary."""

    def test_train_returns_metrics_and_saves_model(self, trainer_config, mock_hf_boundary):
        """train() returns trainer metrics and saves model plus tokenizer."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.train.return_value.metrics = {"train_loss": 0.4}
        model = Mock()
        datasets = make_datasets(["train", "val"])

        trainer = DNATrainer(model=model, config=trainer_config, datasets=datasets)
        metrics = trainer.train()

        assert metrics == {"train_loss": 0.4}
        model.train.assert_called_once()
        trainer_cls.return_value.save_model.assert_called_once()
        model.save_pretrained.assert_called_once_with(trainer_config["finetune"].output_dir)
        datasets.tokenizer.save_pretrained.assert_called_once_with(
            trainer_config["finetune"].output_dir
        )

    def test_train_without_safetensors_uses_torch_save(self, trainer_config, mock_hf_boundary):
        """save_safetensors=False serializes the state dict via torch.save."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.train.return_value.metrics = {}
        trainer_config["finetune"].save_safetensors = False
        model = Mock()

        with patch("torch.save") as mock_save:
            trainer = DNATrainer(
                model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
            )
            trainer.train()

        mock_save.assert_called_once_with(
            model.state_dict(), f"{trainer_config['finetune'].output_dir}/pytorch_model.bin"
        )
        model.save_pretrained.assert_not_called()

    def test_train_skips_pretrained_save_without_method(self, trainer_config, mock_hf_boundary):
        """Models lacking save_pretrained skip direct serialization."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.train.return_value.metrics = {}
        model = Mock(spec=["train", "eval"])

        trainer = DNATrainer(
            model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        trainer.train(save_tokenizer=False)

        trainer_cls.return_value.save_model.assert_called_once()

    def test_evaluate_delegates_to_trainer(self, trainer_config, mock_hf_boundary):
        """evaluate() puts the model in eval mode and returns trainer metrics."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.evaluate.return_value = {"eval_loss": 0.3}
        model = Mock()

        trainer = DNATrainer(
            model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
        )

        assert trainer.evaluate() == {"eval_loss": 0.3}
        model.eval.assert_called_once()


class TestCustomizeAndPlots:
    """Trainer customization and history plotting."""

    def test_customize_trainer_with_class(self, trainer_config, mock_hf_boundary):
        """Passing a class swaps the trainer instance's class."""
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )

        class CustomTrainer:
            """Custom trainer class stand-in."""

        trainer.customize_trainer(CustomTrainer)

        assert trainer.trainer.__class__ is CustomTrainer

    def test_customize_trainer_with_instance(self, trainer_config, mock_hf_boundary):
        """Passing an instance replaces the trainer outright."""
        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        replacement = Mock()

        trainer.customize_trainer(replacement)

        assert trainer.trainer is replacement

    def test_plot_history_writes_both_plots(self, trainer_config, mock_hf_boundary, tmp_path):
        """plot_history saves loss and lr PNGs under the output dir."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.state.log_history = [
            {"step": 1, "loss": 1.5, "learning_rate": 0.01}
        ]

        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        results = trainer.plot_history(output_dir=str(tmp_path / "plots"))

        assert set(results) == {"loss_curve", "lr_schedule"}
        for path in results.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_plot_history_loss_only(self, trainer_config, mock_hf_boundary, tmp_path):
        """plot_history honors the per-plot flags."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.state.log_history = [{"step": 1, "loss": 1.5}]

        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        results = trainer.plot_history(output_dir=str(tmp_path / "plots"), plot_lr=False)

        assert list(results) == ["loss_curve"]


class TestWarmupConversion:
    """warmup_ratio to warmup_steps conversion for transformers v5."""

    def _configure_args(self, args_cls, **attrs):
        """Give the mocked TrainingArguments concrete numeric fields."""
        args_cls.return_value.warmup_steps = 0
        args_cls.return_value.per_device_train_batch_size = 1
        args_cls.return_value.gradient_accumulation_steps = 1
        args_cls.return_value.num_train_epochs = 1
        args_cls.return_value.max_steps = -1
        args_cls.return_value.load_best_model_at_end = False
        for key, value in attrs.items():
            setattr(args_cls.return_value, key, value)

    def test_warmup_steps_from_epoch_schedule(self, trainer_config, mock_hf_boundary):
        """warmup_ratio converts against steps derived from the train set."""
        _, args_cls = mock_hf_boundary
        trainer_config["finetune"].warmup_ratio = 0.5
        self._configure_args(args_cls)

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train"]))

        # ceil(4 rows / batch 1) = 4 steps per epoch, 1 epoch, ratio 0.5 -> 2
        assert args_cls.return_value.warmup_steps == 2

    def test_warmup_steps_prefers_max_steps(self, trainer_config, mock_hf_boundary):
        """A positive max_steps overrides the epoch-derived schedule."""
        _, args_cls = mock_hf_boundary
        trainer_config["finetune"].warmup_ratio = 0.5
        self._configure_args(args_cls, max_steps=10)

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train"]))

        assert args_cls.return_value.warmup_steps == 5

    def test_warmup_skipped_when_steps_already_set(self, trainer_config, mock_hf_boundary):
        """An explicit warmup_steps value suppresses the conversion."""
        _, args_cls = mock_hf_boundary
        trainer_config["finetune"].warmup_ratio = 0.5
        self._configure_args(args_cls, warmup_steps=7)

        DNATrainer(model=Mock(), config=trainer_config, datasets=make_datasets(["train"]))

        assert args_cls.return_value.warmup_steps == 7


class TestLegacyTransformersSave:
    """Pre-transformers-5 save branches."""

    def test_train_saves_with_safe_serialization_kwarg_pre_v5(
        self, trainer_config, mock_hf_boundary
    ):
        """transformers < 5 passes safe_serialization to save_pretrained."""
        trainer_cls, _ = mock_hf_boundary
        trainer_cls.return_value.train.return_value.metrics = {}
        model = Mock()

        with patch("dnallm.finetune.trainer.transformers_version", Version("4.55.0")):
            trainer = DNATrainer(
                model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
            )
            trainer.train(save_tokenizer=False)

        model.save_pretrained.assert_called_once_with(
            trainer_config["finetune"].output_dir, safe_serialization=True
        )

    def test_search_saves_with_safe_serialization_kwarg_pre_v5(
        self, trainer_config, mock_hf_boundary
    ):
        """The search save path uses the same pre-v5 kwarg contract."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["finetune"].hyperparameter_search = HyperparameterSearchConfig(
            search_space={
                "learning_rate": SearchSpaceDistribution(type="float", low=1e-6, high=1e-3)
            },
            n_trials=1,
        )
        best_run = Mock(hyperparameters={"learning_rate": 1e-4}, objective=0.1)
        trainer_cls.return_value.hyperparameter_search.return_value = best_run
        model = Mock()

        with patch("dnallm.finetune.trainer.transformers_version", Version("4.55.0")):
            trainer = DNATrainer(
                model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
            )
            trainer.search(save_tokenizer=False)

        model.save_pretrained.assert_called_once_with(
            trainer_config["finetune"].output_dir, safe_serialization=True
        )

    def test_search_without_safetensors_torch_saves_on_v5(self, trainer_config, mock_hf_boundary):
        """search() with save_safetensors=False serializes via torch.save on v5."""
        trainer_cls, _ = mock_hf_boundary
        trainer_config["finetune"].save_safetensors = False
        trainer_config["finetune"].hyperparameter_search = HyperparameterSearchConfig(
            search_space={
                "learning_rate": SearchSpaceDistribution(type="float", low=1e-6, high=1e-3)
            },
            n_trials=1,
        )
        best_run = Mock(hyperparameters={"learning_rate": 1e-4}, objective=0.1)
        trainer_cls.return_value.hyperparameter_search.return_value = best_run
        model = Mock()

        with patch("torch.save") as mock_save:
            trainer = DNATrainer(
                model=model, config=trainer_config, datasets=make_datasets(["train", "val"])
            )
            trainer.search(save_tokenizer=False)

        mock_save.assert_called_once_with(
            model.state_dict(), f"{trainer_config['finetune'].output_dir}/pytorch_model.bin"
        )
        model.save_pretrained.assert_not_called()


class TestInfer:
    """Test-set inference wiring."""

    def test_infer_predicts_on_test_split(self, trainer_config, mock_hf_boundary):
        """infer() predicts over the test split when present."""
        trainer_cls, args_cls = mock_hf_boundary
        # Default TrainingConfig leaves best-model selection off, so the
        # eval-semantics guard disables evaluation instead of colliding.
        args_cls.return_value.load_best_model_at_end = False
        trainer_cls.return_value.predict.return_value = {"metrics": {"test_loss": 0.2}}
        datasets = make_datasets(["train", "test"])

        with patch("builtins.print"):
            trainer = DNATrainer(model=Mock(), config=trainer_config, datasets=datasets)
        result = trainer.infer()

        trainer_cls.return_value.predict.assert_called_once_with(datasets.dataset["test"])
        assert result == {"metrics": {"test_loss": 0.2}}

    def test_infer_empty_without_test_split(self, trainer_config, mock_hf_boundary):
        """infer() returns an empty dict without a test split."""
        trainer_cls, _ = mock_hf_boundary

        trainer = DNATrainer(
            model=Mock(), config=trainer_config, datasets=make_datasets(["train", "val"])
        )
        result = trainer.infer()

        assert result == {}
        trainer_cls.return_value.predict.assert_not_called()
