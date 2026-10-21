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
from unittest.mock import patch

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
from dnallm.preflight.config_gen import (
    _SECTION_ORDER,
    edit_config,
    generate_config,
    render_yaml,
)

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

        entry = registry.try_resolve("zhangtaolab/plant-dnagpt-BPE-promoter")
        eff = registry.effective_max_length(entry)
        assert eff is not None
        assert eff != 512

        result = generate_config("zhangtaolab/plant-dnagpt-BPE-promoter")

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

        assert any("TASK_TYPE_REQUIRED" in e.message and e.section == "task" for e in result.errors)
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

    def test_missing_effective_length_omits_max_length_keys(self):
        """When the registry carries no effective length, the max_length keys
        are omitted and the Pydantic defaults apply (inference 512)."""
        with patch("dnallm.preflight.config_gen.registry.effective_max_length", return_value=None):
            result = generate_config(BASE_REPO, task_type="binary")

        assert result.errors == []
        assert result.config["inference"].max_length == 512


class TestCollectAllValidation:
    """The collect-all wrapper reports every broken section in one report."""

    def test_two_broken_sections_both_reported_task_first(self):
        """Two invalid sections each contribute their own errors, ordered by
        the fixed section order (task before finetune)."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all({
            "task": {"task_type": "bogus"},
            "finetune": {"max_grad_norm": -1.0},
        })

        assert [e.section for e in errors] == ["task", "finetune"]
        assert errors[0].field == "task.task_type"
        assert errors[1].field == "finetune.max_grad_norm"
        # Neither broken section constructed; nothing raised.
        assert "task" not in configs
        assert "finetune" not in configs

    def test_section_error_field_paths_joined(self):
        """Each ValidationError errors() entry becomes a SectionError with the
        field path joined as 'section.field' (or the section itself when the
        error is model-level, e.g. loc == ())."""
        from dnallm.preflight.config_gen import _validate_collect_all

        _configs, errors = _validate_collect_all({
            "task": {"task_type": "multiclass", "num_labels": -1}
        })

        assert len(errors) == 1
        assert errors[0].section == "task"
        assert errors[0].field == "task"
        assert "num_labels must be at least 2" in errors[0].message

    def test_valid_section_constructs_alongside_broken_one(self):
        """One section's failure never stops another from constructing."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all({"task": {"task_type": "bogus"}, "lora": {"r": 16}})

        assert [e.section for e in errors] == ["task"]
        assert configs["lora"].r == 16

    def test_model_section_passthrough_in_collect_all(self):
        """In the per-section path the model dict passes through unvalidated."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all({
            "task": {"task_type": "bogus"},
            "model": {"name": "x", "anything": 1},
        })

        assert [e.section for e in errors] == ["task"]
        assert configs["model"] == {"name": "x", "anything": 1}

    def test_broken_benchmark_section_reports_field_errors(self):
        """A malformed benchmark section contributes its own field-level
        errors (full-raw-dict construction attempt) without stopping others."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all({
            "task": {"task_type": "bogus"},
            "benchmark": {"models": []},
        })

        sections = [e.section for e in errors]
        assert sections[0] == "task"
        assert "benchmark" in sections
        assert "benchmark" not in configs

    def test_non_mapping_section_is_field_level_error(self):
        """A section holding a scalar yields a field-level error, not a
        traceback; sibling sections still construct."""
        from dnallm.preflight.config_gen import _validate_collect_all

        configs, errors = _validate_collect_all({"task": "binary", "finetune": {}})

        assert [e.section for e in errors] == ["task"]
        assert errors[0].field == "task"
        assert "must be a mapping" in errors[0].message
        assert isinstance(configs["finetune"], TrainingConfig)

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


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: edit_config round-trip (D-14) + YAML dump contract (D-15) + edges
# ─────────────────────────────────────────────────────────────────────────────

BASE_CONFIG_YAML = """\
task:
  task_type: binary
  num_labels: 2
  label_names: ["negative", "positive"]
finetune:
  learning_rate: 2.0e-05
  num_train_epochs: 3
"""


def _write_base_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "base_config.yaml"
    cfg_path.write_text(BASE_CONFIG_YAML, encoding="utf-8")
    return cfg_path


class TestEditConfigRoundTrip:
    """Load existing → patch → revalidate over path and dict inputs."""

    def test_round_trip_from_yaml_path(self, tmp_path):
        """A YAML path patched on one field revalidates; untouched fields
        survive verbatim (learning_rate bit-identical — precision edge)."""
        cfg_path = _write_base_config(tmp_path)
        original_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        result = edit_config(cfg_path, {"task": {"num_labels": 3}})

        assert result.errors == []
        assert result.config["task"].num_labels == 3
        assert result.config["task"].label_names == ["negative", "positive"]
        assert result.config["finetune"].learning_rate == original_raw["finetune"]["learning_rate"]

    def test_round_trip_from_dict_equals_path_variant(self, tmp_path):
        """A parsed dict input produces the same sections as the path variant."""
        cfg_path = _write_base_config(tmp_path)
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        patch = {"task": {"num_labels": 3}}

        from_path = edit_config(cfg_path, patch)
        from_dict = edit_config(raw, patch)

        _assert_section_maps_equal(from_path.config, from_dict.config)
        assert from_dict.errors == []

    def test_patch_overwrites_per_field_at_section_level(self, tmp_path):
        """Patch keys overwrite exactly the named fields; no recursive merge."""
        cfg_path = _write_base_config(tmp_path)

        result = edit_config(
            cfg_path,
            {"finetune": {"learning_rate": 1.0e-03, "num_train_epochs": 5}},
        )

        assert result.errors == []
        finetune = result.config["finetune"]
        assert finetune.learning_rate == 1.0e-03
        assert finetune.num_train_epochs == 5
        # Untouched fields keep their values (packaged value and default).
        assert finetune.seed == 42
        assert finetune.report_to == ["tensorboard"]

    def test_empty_patch_revalidates_unchanged(self, tmp_path):
        """An empty patch changes nothing: identical section values out."""
        cfg_path = _write_base_config(tmp_path)
        baseline = edit_config(cfg_path, {})

        assert baseline.errors == []
        again = edit_config(cfg_path, {})
        _assert_section_maps_equal(baseline.config, again.config)
        assert baseline.config["task"].num_labels == 2

    def test_patch_unknown_section_is_field_level_error(self, tmp_path):
        """A patch naming an unknown section yields UNKNOWN_SECTION; the
        bogus key never enters the config."""
        cfg_path = _write_base_config(tmp_path)

        result = edit_config(cfg_path, {"bogus": {"x": 1}})

        assert any("UNKNOWN_SECTION" in e.message for e in result.errors)
        assert "bogus" not in result.config
        # The valid sections still validate (collect-all).
        assert isinstance(result.config["task"], TaskConfig)

    def test_patch_invalid_value_is_field_level_error_not_traceback(self, tmp_path):
        """A patch to an invalid value yields the section's field-level
        errors — both the joined-path and model-level (empty loc) forms."""
        cfg_path = _write_base_config(tmp_path)

        result = edit_config(cfg_path, {"task": {"task_type": "multiclass", "num_labels": -1}})

        assert any(e.field == "task" for e in result.errors)
        assert "task" not in result.config
        assert isinstance(result.config["finetune"], TrainingConfig)

    def test_null_section_treated_as_absent_and_empty_section_defaults(self, tmp_path):
        """A null-valued section is absent; a {} section constructs defaults."""
        cfg_path = tmp_path / "edge.yaml"
        cfg_path.write_text("task:\nfinetune: {}\n", encoding="utf-8")

        result = edit_config(cfg_path, {})

        assert result.errors == []
        assert "task" not in result.config
        assert isinstance(result.config["finetune"], TrainingConfig)
        assert result.config["finetune"].num_train_epochs == 3

    def test_zero_recognized_sections_yields_no_sections_error(self, tmp_path):
        """A YAML with only an unrelated top key yields NO_SECTIONS."""
        cfg_path = tmp_path / "unrelated.yaml"
        cfg_path.write_text("notes: hello\n", encoding="utf-8")

        result = edit_config(cfg_path, {})

        assert any("NO_SECTIONS" in e.message for e in result.errors)
        assert result.config == {}

    def test_input_dict_not_mutated(self, tmp_path):
        """The caller's dict is patched on a copy, never mutated in place."""
        raw = {"task": {"task_type": "binary"}, "finetune": {"learning_rate": 2.0e-05}}
        snapshot = {"task": dict(raw["task"]), "finetune": dict(raw["finetune"])}

        result = edit_config(raw, {"task": {"num_labels": 4}})

        assert raw == snapshot
        assert result.config["task"].num_labels == 4

    def test_patch_replaces_model_section_wholesale(self, tmp_path):
        """The model section is the passthrough: any patch value replaces it."""
        cfg_path = _write_base_config(tmp_path)

        result = edit_config(cfg_path, {"model": {"name": "x", "path": "org/repo"}})

        assert result.errors == []
        assert result.config["model"] == {"name": "x", "path": "org/repo"}

    def test_patch_non_mapping_value_for_known_section_is_error(self, tmp_path):
        """A scalar patch value for a validated section is a field-level
        error; the section keeps its loaded value."""
        cfg_path = _write_base_config(tmp_path)

        result = edit_config(cfg_path, {"task": "binary"})

        assert any("must be a mapping" in e.message for e in result.errors)
        assert isinstance(result.config["task"], TaskConfig)

    def test_patch_creates_missing_section_dict(self, tmp_path):
        """Patching into an absent (null) section creates its dict fresh."""
        raw = {"task": None, "finetune": {"learning_rate": 2.0e-05}}

        result = edit_config(raw, {"task": {"task_type": "multiclass", "num_labels": 3}})

        assert result.errors == []
        assert result.config["task"].task_type == "multiclass"
        assert result.config["task"].label_names == ["class_0", "class_1", "class_2"]

    def test_nonexistent_existing_path_raises_matchable_value_error(self, tmp_path):
        """A nonexistent existing path raises ValueError, never a raw
        FileNotFoundError (WR-01) — the matchable contract validate_data
        already follows for the same user mistake."""
        missing = tmp_path / "no-such-config.yaml"

        with pytest.raises(ValueError, match=r"Existing config not found"):
            edit_config(missing, {})

    def test_unreadable_existing_path_raises_matchable_value_error(self, tmp_path):
        """A directory as the existing path raises ValueError too, not OSError."""
        with pytest.raises(ValueError, match=r"Existing config not readable"):
            edit_config(tmp_path, {})


class TestYamlDumpContract:
    """Write only when output_path is given AND the run is error-free (D-15, WR-02)."""

    def test_generate_config_with_errors_writes_no_file(self, tmp_path):
        """generate_config carrying errors never writes output_path (WR-02):
        a base model without task_type yields TASK_TYPE_REQUIRED and no file
        on disk — the caller cannot mistake file existence for validity."""
        out_path = tmp_path / "generated.yaml"

        result = generate_config(BASE_REPO, output_path=str(out_path))

        assert any("TASK_TYPE_REQUIRED" in e.message for e in result.errors)
        assert not out_path.exists()

    def test_edit_config_with_errors_writes_no_file(self, tmp_path):
        """edit_config carrying errors (patch or validation) never writes
        output_path (WR-02) — same rule on the round-trip path."""
        cfg_path = _write_base_config(tmp_path)
        out_path = tmp_path / "edited.yaml"

        result = edit_config(cfg_path, {"bogus": {"x": 1}}, output_path=str(out_path))

        assert any("UNKNOWN_SECTION" in e.message for e in result.errors)
        assert not out_path.exists()

    def test_output_written_when_output_path_given(self, tmp_path):
        """The dumped YAML re-loads to equal section values."""
        cfg_path = _write_base_config(tmp_path)
        out_path = tmp_path / "out" / "edited.yaml"
        out_path.parent.mkdir()
        out = str(out_path)

        result = edit_config(cfg_path, {"task": {"num_labels": 3}}, output_path=out)

        assert result.errors == []
        assert out_path.exists()
        reloaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert reloaded["task"]["num_labels"] == 3
        assert reloaded["finetune"]["learning_rate"] == 2.0e-05

    def test_no_file_written_when_output_path_omitted(self, tmp_path):
        """Without output_path nothing is created anywhere under tmp_path."""
        cfg_path = _write_base_config(tmp_path)
        before = sorted(os.listdir(tmp_path))

        edit_config(cfg_path, {"task": {"num_labels": 3}})

        assert sorted(os.listdir(tmp_path)) == before

    def test_input_file_never_modified(self, tmp_path):
        """Editing writes only to the named output; the input bytes stay put."""
        cfg_path = _write_base_config(tmp_path)
        original_bytes = cfg_path.read_bytes()
        out_path = tmp_path / "edited.yaml"

        edit_config(cfg_path, {"task": {"num_labels": 3}}, output_path=out_path)

        assert cfg_path.read_bytes() == original_bytes
        assert out_path.exists()

    def test_generate_config_dumps_complete_config(self, tmp_path):
        """generate_config(output_path=...) writes the complete config; the
        dumped YAML carries the finetune max_length key and reloads to the
        same section values."""
        out_path = tmp_path / "generated.yaml"

        result = generate_config(FINETUNED_REPO, output_path=out_path)

        assert result.errors == []
        assert out_path.exists()
        dumped = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert sorted(dumped) == ["finetune", "inference", "lora", "model", "task"]
        assert dumped["finetune"]["max_length"] == 512
        assert dumped["inference"]["max_length"] == 512
        assert dumped["model"] == {"name": FINETUNED_REPO}
        # The dump revalidates to the same sections the call returned.
        revalidated = edit_config(dumped, {})
        assert revalidated.errors == []
        _assert_section_maps_equal(revalidated.config, result.config)


class TestUnsafeYamlRefusal:
    """The round-trip loader refuses arbitrary constructors (T-23-08)."""

    def test_python_object_tag_refused(self, tmp_path):
        """A YAML carrying a python/object tag raises the safe-loader error —
        no arbitrary constructor ever executes."""
        cfg_path = tmp_path / "evil.yaml"
        cfg_path.write_text(
            'model: !!python/object/apply:os.system ["touch /tmp/pwned"]\n',
            encoding="utf-8",
        )

        with pytest.raises(yaml.YAMLError):
            edit_config(cfg_path, {})


# ─────────────────────────────────────────────────────────────────────────────
# 25-08 Task 1: GeneratedConfig.raw + render_yaml, the one serializer
# ─────────────────────────────────────────────────────────────────────────────


class TestRawMappingAndRenderYaml:
    """The raw mapping a shell edits is the mapping the writer emits."""

    def test_generated_raw_carries_the_five_core_sections(self, tmp_path):
        """raw names every emitted section, and the written file parses back
        to an equal mapping (the preview can never drift from the file)."""
        out_path = tmp_path / "generated.yaml"

        result = generate_config(FINETUNED_REPO, output_path=out_path)

        assert result.errors == []
        assert sorted(result.raw) == ["finetune", "inference", "lora", "model", "task"]
        assert yaml.safe_load(out_path.read_text(encoding="utf-8")) == result.raw
        assert result.raw["model"] == {"name": FINETUNED_REPO}
        assert result.raw["task"]["label_names"] == FINETUNED_LABEL_NAMES

    def test_edit_config_populates_raw_with_the_patched_mapping(self, tmp_path):
        """The round-trip raw carries the patch and every untouched field."""
        cfg_path = _write_base_config(tmp_path)

        result = edit_config(cfg_path, {"task": {"num_labels": 3}})

        assert result.errors == []
        assert result.raw["task"]["num_labels"] == 3
        assert result.raw["task"]["label_names"] == ["negative", "positive"]
        assert result.raw["finetune"]["learning_rate"] == 2.0e-05

    def test_render_yaml_writes_nothing(self, tmp_path, monkeypatch):
        """Rendering is a pure string transform: no file, no directory, no cwd
        side effect — tmp_path is exactly as empty as it started."""
        monkeypatch.chdir(tmp_path)
        mapping = {"task": {"task_type": "binary"}, "lora": {"r": 8}}

        text = render_yaml(mapping)

        assert yaml.safe_load(text) == mapping
        assert list(tmp_path.iterdir()) == []

    def test_the_written_file_is_byte_identical_to_render_yaml(self, tmp_path):
        """There is ONE serializer: the file is exactly render_yaml(raw)."""
        out_path = tmp_path / "written.yaml"

        result = edit_config(_write_base_config(tmp_path), {"task": {"num_labels": 3}})
        generated = edit_config(
            _write_base_config(tmp_path), {"task": {"num_labels": 3}}, output_path=out_path
        )

        assert generated.errors == []
        assert out_path.read_text(encoding="utf-8") == render_yaml(result.raw)

    def test_a_serialization_failure_leaves_the_existing_file_untouched(self, tmp_path):
        """WR-04: the destination is replaced atomically, never truncated first.

        The old writer opened the target with "w" (truncating it) before
        serializing, so an unserializable value destroyed the user's config.
        Now the document is rendered first, and the target keeps its bytes —
        with no scratch file left beside it.
        """
        out_path = tmp_path / "config.yaml"
        out_path.write_text("task:\n  task_type: binary\n", encoding="utf-8")
        before = out_path.read_bytes()

        with pytest.raises(yaml.representer.RepresenterError):
            edit_config(
                {"task": {"task_type": "binary"}, "model": {"path": Path("/tmp/x")}},
                {},
                output_path=out_path,
            )

        assert out_path.read_bytes() == before
        assert sorted(path.name for path in tmp_path.iterdir()) == ["config.yaml"]

    def test_render_yaml_keeps_the_mapping_key_order(self):
        """Keys stay where the mapping put them (sort_keys=False)."""
        text = render_yaml({"model": {"name": "x"}, "task": {"task_type": "binary"}})

        assert text.index("model:") < text.index("task:")

    def test_unknown_model_still_carries_an_empty_raw(self):
        """A registry miss has nothing to write, and raw says so."""
        result = generate_config("no-such/model")

        assert result.raw == {}
        assert result.errors


class TestEmptyPatchNoOp:
    """An empty patch revalidates unchanged and never drops a section (D-11)."""

    def test_revalidating_with_an_empty_patch_is_a_no_op(self, tmp_path):
        """patch={} returns zero errors and the same mapping it was handed,
        out-of-scope sections included — nothing is rewritten or dropped."""
        raw = {
            "task": {"task_type": "binary", "num_labels": 2},
            "finetune": {"learning_rate": 2.0e-05},
            "ia3": {"r": 16, "target_modules": ["query", "value"]},
        }

        result = edit_config(raw, {})

        assert result.errors == []
        assert result.raw == raw
        assert result.raw["ia3"] == {"r": 16, "target_modules": ["query", "value"]}
