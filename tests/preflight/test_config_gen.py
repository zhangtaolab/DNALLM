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
    load_config,
    validate_sections,
)

REPO_ROOT = Path(__file__).parent.parent.parent
PACKAGED_CONFIGS = sorted((REPO_ROOT / "configs").glob("*.yaml"))
BENCHMARK_YAML = REPO_ROOT / "example" / "notebooks" / "benchmark" / "benchmark_config.yaml"


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
