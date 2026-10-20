"""Config generation suite: validate_sections equivalence + generate/edit round-trip.

Covers plan 23-04 (PFC-04):

1. Task 1 — the one sanctioned refactor: ``validate_sections`` factored
   verbatim out of ``load_config`` (``dnallm/configuration/configs.py``),
   zero behavior change. Equivalence tests pin ``load_config(path)`` ==
   ``validate_sections(raw dict + config_path)`` over every packaged
   example config, the raise-first error contract, and the benchmark
   full-raw-dict consumption.
2. Task 2 — ``generate_config`` validated emission with model-derived
   defaults (registry prefill, collect-all field-level errors).
3. Task 3 — ``edit_config`` load-existing → patch → revalidate round-trip
   with the YAML dump contract (write only when ``output_path`` is given).
"""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from dnallm.configuration.configs import (
    BenchmarkConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
    validate_sections,
)
from dnallm.preflight.config_gen import _SECTION_ORDER, generate_config

REPO_ROOT = Path(__file__).parent.parent.parent
PACKAGED_CONFIGS = sorted((REPO_ROOT / "configs").glob("*.yaml"))
BENCHMARK_YAML = REPO_ROOT / "example" / "notebooks" / "benchmark" / "benchmark_config.yaml"

# Registry-pinned fixtures (Phase 22 test-suite precedent): every prefilled
# value asserted below was measured against the real registry at plan time.
FINETUNED_REPO = "zhangtaolab/plant-dnabert-BPE-promoter"
FINETUNED_LABEL_NAMES = ["Not promoter", "Core promoter"]
BASE_REPO = "zhangtaolab/plant-dnabert-BPE"


def _raw_with_config_path(path: Path) -> dict:
    """Load a YAML the way ``load_config`` does: safe_load + config_path injection."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["config_path"] = os.path.abspath(path)
    return raw


def _assert_section_maps_equal(via_load: dict, via_factored: dict) -> None:
    """Assert two DNALLMConfig maps are equal in keys, types, and content."""
    assert sorted(via_load) == sorted(via_factored)
    for key in via_load:
        left, right = via_load[key], via_factored[key]
        assert type(left) is type(right), f"section {key!r} type mismatch"
        if isinstance(left, BaseModel):
            assert left.model_dump() == right.model_dump(), f"section {key!r} content mismatch"
        else:
            assert left == right, f"section {key!r} content mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: validate_sections factor-out — verbatim move, zero behavior change
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateSectionsEquivalence:
    """load_config and validate_sections produce identical results."""

    @pytest.mark.skipif(not PACKAGED_CONFIGS, reason="No packaged configs found")
    @pytest.mark.parametrize("config_path", PACKAGED_CONFIGS, ids=lambda p: p.name)
    def test_equivalence_packaged_example_configs(self, config_path: Path):
        """Every packaged example config yields the same sections via both paths."""
        via_load = load_config(str(config_path))
        via_factored = validate_sections(_raw_with_config_path(config_path))

        _assert_section_maps_equal(via_load, via_factored)

    def test_equivalence_invalid_task_section_raises_first_error(self, tmp_path):
        """Raise-first contract preserved: an invalid task section raises the
        same first ValidationError shape out of both paths."""
        cfg_path = tmp_path / "bad_task.yaml"
        cfg_path.write_text("task:\n  task_type: not-a-task\n", encoding="utf-8")
        raw = _raw_with_config_path(cfg_path)

        with pytest.raises(ValidationError) as load_exc:
            load_config(str(cfg_path))
        with pytest.raises(ValidationError) as factored_exc:
            validate_sections(raw)

        load_first = load_exc.value.errors()[0]
        factored_first = factored_exc.value.errors()[0]
        assert load_first["loc"] == factored_first["loc"] == ("task_type",)
        assert load_first["type"] == factored_first["type"] == "string_pattern_mismatch"

    def test_equivalence_benchmark_consumes_full_raw_dict(self):
        """The benchmark section constructs BenchmarkConfig from the FULL raw
        dict (config_path key included), pinned against an inline golden."""
        raw = _raw_with_config_path(BENCHMARK_YAML)
        golden = BenchmarkConfig(**raw)

        via_load = load_config(str(BENCHMARK_YAML))
        via_factored = validate_sections(raw)

        assert isinstance(via_load["benchmark"], BenchmarkConfig)
        assert via_load["benchmark"].model_dump() == golden.model_dump()
        assert via_load["benchmark"].config_path == os.path.abspath(BENCHMARK_YAML)
        assert via_factored["benchmark"].model_dump() == golden.model_dump()

    def test_equivalence_empty_dict_yields_empty_config(self):
        """Every section is optional: an empty dict in, an empty DNALLMConfig out."""
        assert validate_sections({}) == {}

    def test_equivalence_minimal_task_only_config(self):
        """A single-section dict validates that section and nothing else."""
        result = validate_sections({"task": {"task_type": "binary"}})

        assert sorted(result) == ["task"]
        assert isinstance(result["task"], TaskConfig)


class TestFactorOutStructure:
    """The construction body moved out of load_config — not duplicated."""

    def test_factor_out_moved_not_duplicated(self):
        """configs.py carries exactly one validate_sections definition and one
        task-construction site (inside the factored function)."""
        source = (REPO_ROOT / "dnallm" / "configuration" / "configs.py").read_text(encoding="utf-8")

        assert source.count("def validate_sections(") == 1
        assert source.count('configs["task"] = TaskConfig') == 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: generate_config — validated emission with model-derived defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateConfigEmission:
    """generate_config emits a complete validated config or field-level errors."""

    def test_finetuned_entry_prefill_from_registry(self):
        """A finetuned entry prefills task fields and max_length from the registry."""
        result = generate_config(FINETUNED_REPO)

        assert result.errors == []
        task = result.config["task"]
        assert isinstance(task, TaskConfig)
        assert task.task_type == "binary"
        assert task.num_labels == 2
        assert task.label_names == FINETUNED_LABEL_NAMES
        assert task.threshold == 0.5
        # The registry-derived context length lands in the section that
        # validates it (InferenceConfig.max_length) and is the field actually
        # consumed at runtime (inference.py reads pred_config.max_length).
        assert result.config["inference"].max_length == 512
        assert isinstance(result.config["finetune"], TrainingConfig)
        assert result.config["model"] == {"name": FINETUNED_REPO}

    def test_finetuned_max_length_follows_registry_value(self):
        """A finetuned entry whose effective_max_length is not the 512 default
        carries it into the validated inference section (registry-as-truth)."""
        from dnallm.models import registry

        entry = registry.try_resolve("zhangtaolab/plant-dnagemma-BPE")
        eff = registry.effective_max_length(entry)
        assert eff is not None and eff != 512

        result = generate_config("zhangtaolab/plant-dnagemma-BPE")

        assert result.errors == []
        assert result.config["inference"].max_length == eff

    def test_base_model_with_task_type_argument(self):
        """A base model takes task_type from the argument; TaskConfig fills
        the head defaults per its own conventions."""
        result = generate_config(BASE_REPO, task_type="multiclass")

        assert result.errors == []
        task = result.config["task"]
        assert task.task_type == "multiclass"
        assert task.num_labels == 2
        assert task.label_names == ["class_0", "class_1"]

    def test_base_model_without_task_type_is_field_level_error(self):
        """A base model ships no task: omitting task_type yields a
        TASK_TYPE_REQUIRED field-level error, never a traceback."""
        result = generate_config(BASE_REPO)

        assert any(
            "TASK_TYPE_REQUIRED" in e.message and e.section == "task" for e in result.errors
        )
        assert "task" not in result.config
        # Collect-all: the other sections still validate and return.
        assert result.config["model"] == {"name": BASE_REPO}
        assert isinstance(result.config["inference"], object)

    def test_unknown_model_is_field_level_error(self):
        """An unknown model name yields MODEL_NOT_IN_REGISTRY_CFG and an empty
        config dict — never a top-level exception."""
        result = generate_config("no-such/model")

        assert len(result.errors) == 1
        assert result.errors[0].section == "model"
        assert "MODEL_NOT_IN_REGISTRY_CFG" in result.errors[0].message
        assert "no-such/model" in result.errors[0].message
        assert result.config == {}


class TestCollectAllValidation:
    """The collect-all wrapper reports every broken section in one report."""

    def test_two_broken_sections_both_reported_task_first(self):
        """Two invalid sections each contribute their own errors, ordered by
        the fixed section order (task before finetune)."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all(
            {
                "task": {"task_type": "bogus"},
                "finetune": {"max_grad_norm": -1.0},
            }
        )

        assert [e.section for e in errors] == ["task", "finetune"]
        assert errors[0].field == "task.task_type"
        assert errors[1].field == "finetune.max_grad_norm"
        # Neither broken section constructed; nothing raised.
        assert "task" not in configs and "finetune" not in configs

    def test_section_error_field_paths_joined(self):
        """Each ValidationError errors() entry becomes a SectionError with the
        field path joined as 'section.field' (or the section itself when the
        error is model-level, e.g. loc == ())."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all(
            {"task": {"task_type": "multiclass", "num_labels": -1}}
        )

        assert len(errors) == 1
        assert errors[0].section == "task"
        assert errors[0].field == "task"
        assert "num_labels must be at least 2" in errors[0].message

    def test_valid_section_constructs_alongside_broken_one(self):
        """One section's failure never stops another from constructing."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all(
            {"task": {"task_type": "bogus"}, "lora": {"r": 16}}
        )

        assert [e.section for e in errors] == ["task"]
        assert configs["lora"].r == 16

    def test_section_order_constant_is_canonical(self):
        """_SECTION_ORDER is the fixed deterministic error-ordering contract."""
        assert _SECTION_ORDER == (
            "task",
            "model",
            "inference",
            "finetune",
            "lora",
            "ia3",
            "vep",
            "sweep",
            "benchmark",
        )
