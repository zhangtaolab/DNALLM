#!/usr/bin/env python3
"""Tests for the multi-seed sweep protocol (dnallm.finetune.sweep).

Fast lane (``-m "not slow"``) is network-free: pure-function aggregation
tests on constructed arrays plus run_seeds orchestration tests with a
stubbed per-seed function. The real-model >= 3-seed acceptance trial
lives in the slow lane at the bottom of this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy import stats

from dnallm.configuration.configs import SweepConfig
from dnallm.finetune.sweep import aggregate_seeds, run_seeds, run_sweep_from_config

AGG_KWARGS = {"n_bootstrap": 2000, "bootstrap_seed": 42, "small_n_ci": "t-interval"}


def _stub_fn(seed: int, seed_dir: Path) -> dict[str, float]:
    """Deterministic per-seed run: metrics derived purely from the seed."""
    return {"auroc": 0.8 + 0.001 * seed, "loss": 0.5 - 0.0001 * seed}


class TestAggregateSeeds:
    """Pure-function statistics on constructed arrays (SEED-01, D-14)."""

    def test_aggregate_known_moments_exact(self):
        block = aggregate_seeds([1.0, 2.0, 3.0], **AGG_KWARGS)
        assert block["n_seeds"] == 3
        assert block["mean"] == 2.0
        assert block["sd"] == 1.0

    def test_aggregate_t_interval_matches_hand_computation(self):
        values = [1.0, 2.0, 3.0]
        block = aggregate_seeds(values, **AGG_KWARGS)
        assert block["method"] == "t"
        sem = 1.0 / math.sqrt(3)
        tcrit = stats.t.ppf(0.975, 2)
        assert block["ci95"] == pytest.approx([2.0 - tcrit * sem, 2.0 + tcrit * sem])

    def test_aggregate_n1_sd_and_ci_none(self):
        block = aggregate_seeds([5.0], **AGG_KWARGS)
        assert block == {
            "n_seeds": 1,
            "mean": 5.0,
            "sd": None,
            "ci95": None,
            "method": "none",
        }

    def test_aggregate_n2_guard_blocks_ci(self):
        block = aggregate_seeds([1.0, 2.0], **AGG_KWARGS)
        assert block["n_seeds"] == 2
        assert block["sd"] == pytest.approx(0.7071067811865476)
        assert block["ci95"] is None
        assert block["method"] == "none"

    def test_aggregate_n2_omit_is_still_none(self):
        # n < 3 precedes the small_n_ci policy: no interval in any branch.
        block = aggregate_seeds(
            [1.0, 2.0],
            small_n_ci="omit",
            **{k: v for k, v in AGG_KWARGS.items() if k != "small_n_ci"},
        )
        assert block["ci95"] is None
        assert block["method"] == "none"

    def test_aggregate_n3_boundary_uses_t(self):
        block = aggregate_seeds([0.5, 0.6, 0.7], **AGG_KWARGS)
        assert block["method"] == "t"
        assert block["ci95"] is not None

    def test_aggregate_n9_uses_t(self):
        block = aggregate_seeds([float(i) for i in range(9)], **AGG_KWARGS)
        assert block["method"] == "t"
        assert len(block["ci95"]) == 2

    def test_aggregate_n10_boundary_uses_bootstrap(self):
        block = aggregate_seeds([float(i) for i in range(10)], **AGG_KWARGS)
        assert block["method"] == "bootstrap-percentile"
        assert len(block["ci95"]) == 2

    def test_aggregate_n9_omit_policy_omits(self):
        kwargs = dict(AGG_KWARGS)
        kwargs["small_n_ci"] = "omit"
        block = aggregate_seeds([float(i) for i in range(9)], **kwargs)
        assert block["ci95"] is None
        assert block["method"] == "omitted"
        assert block["mean"] == 4.0

    def test_aggregate_n10_omit_still_bootstraps(self):
        # small_n_ci only governs the 3 <= n < 10 range (SweepConfig contract).
        kwargs = dict(AGG_KWARGS)
        kwargs["small_n_ci"] = "omit"
        block = aggregate_seeds([float(i) for i in range(10)], **kwargs)
        assert block["method"] == "bootstrap-percentile"

    def test_aggregate_bootstrap_is_seeded_and_reproducible(self):
        values = [float(i) for i in range(10)]
        first = aggregate_seeds(values, **AGG_KWARGS)
        second = aggregate_seeds(values, **AGG_KWARGS)
        assert first["ci95"] == second["ci95"]
        assert first == second

    def test_aggregate_bootstrap_seed_changes_interval(self):
        # Distinct, spread values give the resampled-mean percentiles enough
        # continuity that different RNG streams land on different bounds
        # (verified: seeds 42 and 7 differ on exactly this array).
        values = [0.13, 0.87, 0.41, 0.72, 0.05, 0.98, 0.29, 0.63, 0.51, 0.80]
        kwargs_a = dict(AGG_KWARGS)
        kwargs_b = dict(AGG_KWARGS)
        kwargs_b["bootstrap_seed"] = 7
        ci_a = aggregate_seeds(values, **kwargs_a)["ci95"]
        ci_b = aggregate_seeds(values, **kwargs_b)["ci95"]
        assert ci_a != ci_b

    def test_aggregate_invalid_small_n_ci_rejected(self):
        with pytest.raises(ValueError, match=r"small_n_ci must be one of"):
            aggregate_seeds([1.0, 2.0, 3.0], n_bootstrap=10, bootstrap_seed=42, small_n_ci="normal")

    def test_aggregate_empty_values_rejected(self):
        with pytest.raises(ValueError, match=r"at least one per-seed value"):
            aggregate_seeds([], **AGG_KWARGS)

    def test_aggregate_non_flat_values_rejected(self):
        with pytest.raises(ValueError, match=r"flat sequence"):
            aggregate_seeds([[1.0, 2.0], [3.0, 4.0]], **AGG_KWARGS)


class TestRunSeeds:
    """Orchestration: directory protocol, D-16 seed semantics, determinism."""

    def test_run_seeds_directory_protocol(self, tmp_path):
        payload = run_seeds(
            _stub_fn,
            [42, 43, 44],
            tmp_path,
            model_name="plant-dnabert-BPE",
            task_name="core_promoters",
        )
        task_root = tmp_path / "plant-dnabert-BPE" / "core_promoters"
        for seed in (42, 43, 44):
            seed_dir = task_root / f"seed_{seed}"
            assert seed_dir.is_dir()
            assert (seed_dir / "seed_result.json").is_file()
        assert (task_root / "statistics.json").is_file()
        assert payload["model_name"] == "plant-dnabert-BPE"
        assert payload["seeds"] == [42, 43, 44]

    def test_run_seeds_determinism_byte_identical(self, tmp_path):
        # Same seeds + deterministic fn -> identical statistics.json bytes
        # across two different out_roots (per-seed paths are stored relative
        # to the {model}/{task}/ dir precisely so this holds).
        run_seeds(_stub_fn, [42, 43, 44], tmp_path / "run_a", model_name="m", task_name="t")
        run_seeds(_stub_fn, [42, 43, 44], tmp_path / "run_b", model_name="m", task_name="t")
        bytes_a = (tmp_path / "run_a" / "m" / "t" / "statistics.json").read_bytes()
        bytes_b = (tmp_path / "run_b" / "m" / "t" / "statistics.json").read_bytes()
        assert bytes_a == bytes_b

    def test_run_seeds_fn_called_once_per_seed_with_seed_and_dir(self, tmp_path):
        # D-16 structural pin: run_seeds takes no dataset and cannot re-split;
        # each seed gets exactly one fn(seed, seed_dir) call.
        calls: list[tuple[int, str]] = []

        def recording_fn(seed: int, seed_dir: Path) -> dict[str, float]:
            calls.append((seed, Path(seed_dir).name))
            return {"m": float(seed)}

        run_seeds(recording_fn, [7, 8], tmp_path, model_name="m", task_name="t")
        assert calls == [(7, "seed_7"), (8, "seed_8")]

    def test_run_seeds_statistics_block_method_t_at_three_seeds(self, tmp_path):
        payload = run_seeds(_stub_fn, [42, 43, 44], tmp_path, model_name="m", task_name="t")
        block = payload["statistics"]["auroc"]
        assert block["n_seeds"] == 3
        assert block["method"] == "t"
        lo, hi = block["ci95"]
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo < block["mean"] < hi

    def test_run_seeds_bootstrap_at_ten_seeds(self, tmp_path):
        payload = run_seeds(
            lambda seed, seed_dir: {"m": 0.5 + 0.01 * seed},
            list(range(10)),
            tmp_path,
            model_name="m",
            task_name="t",
        )
        assert payload["statistics"]["m"]["method"] == "bootstrap-percentile"

    def test_run_seeds_per_seed_json_shape(self, tmp_path):
        run_seeds(_stub_fn, [42], tmp_path, model_name="m", task_name="t")
        data = json.loads((tmp_path / "m" / "t" / "seed_42" / "seed_result.json").read_text())
        assert data["seed"] == 42
        assert data["model_name"] == "m"
        assert data["task_name"] == "t"
        assert "timestamp" in data
        assert data["metrics"]["auroc"] == pytest.approx(0.842)

    def test_run_seeds_path_separator_in_model_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"path separators"):
            run_seeds(
                _stub_fn, [1], tmp_path, model_name="zhangtaolab/plant-dnabert-BPE", task_name="t"
            )

    def test_run_seeds_path_separator_in_task_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"path separators"):
            run_seeds(_stub_fn, [1], tmp_path, model_name="m", task_name="nested\\task")

    def test_run_seeds_dot_segment_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"special segment"):
            run_seeds(_stub_fn, [1], tmp_path, model_name="..", task_name="t")

    def test_run_seeds_empty_model_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"non-empty path segment"):
            run_seeds(_stub_fn, [1], tmp_path, model_name="  ", task_name="t")

    def test_run_seeds_empty_out_root_rejected(self):
        with pytest.raises(ValueError, match=r"out_root must be a non-empty path"):
            run_seeds(_stub_fn, [1], "", model_name="m", task_name="t")

    def test_run_seeds_none_out_root_rejected(self):
        with pytest.raises(ValueError, match=r"out_root must be a non-empty path"):
            run_seeds(_stub_fn, [1], None, model_name="m", task_name="t")  # type: ignore[arg-type]

    def test_run_seeds_empty_seeds_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"at least one seed"):
            run_seeds(_stub_fn, [], tmp_path, model_name="m", task_name="t")

    def test_run_seeds_metric_keys_missing_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"did not report"):
            run_seeds(
                _stub_fn,
                [1, 2],
                tmp_path,
                model_name="m",
                task_name="t",
                metric_keys=["nonexistent"],
            )

    def test_run_seeds_metric_keys_restricts_aggregation(self, tmp_path):
        payload = run_seeds(
            _stub_fn,
            [1, 2, 3],
            tmp_path,
            model_name="m",
            task_name="t",
            metric_keys=["loss"],
        )
        assert set(payload["statistics"]) == {"loss"}

    def test_run_seeds_non_numeric_metric_skipped(self, tmp_path):
        def mixed_fn(seed: int, seed_dir: Path) -> dict[str, object]:
            return {"m": float(seed), "name": f"run-{seed}"}

        payload = run_seeds(mixed_fn, [1, 2, 3], tmp_path, model_name="m", task_name="t")
        assert "m" in payload["statistics"]
        assert "name" not in payload["statistics"]

    def test_run_seeds_explicit_non_numeric_metric_rejected(self, tmp_path):
        # IN-09: a metric the caller explicitly requested via metric_keys
        # but that a seed reports non-numeric hard-fails instead of
        # silently vanishing from statistics (auto-discovery soft-skips).
        def mixed_fn(seed: int, seed_dir: Path) -> dict[str, object]:
            return {"m": float(seed), "name": f"run-{seed}"}

        with pytest.raises(ValueError, match=r"requested 'name'.*non-numeric"):
            run_seeds(
                mixed_fn,
                [1, 2],
                tmp_path,
                model_name="m",
                task_name="t",
                metric_keys=["name"],
            )

    def test_run_seeds_invalid_small_n_ci_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"small_n_ci must be one of"):
            run_seeds(
                _stub_fn, [1, 2], tmp_path, model_name="m", task_name="t", small_n_ci="normal"
            )


class TestRunSweepFromConfig:
    """SweepConfig adapter: fields consumed verbatim, no configs.py edits."""

    def test_adapter_threads_sweep_config_fields(self, tmp_path):
        config = SweepConfig(seeds=[1, 2, 3], out_root=str(tmp_path / "root"))
        payload = run_sweep_from_config(config, _stub_fn, model_name="m", task_name="t")
        assert (tmp_path / "root" / "m" / "t" / "statistics.json").is_file()
        assert payload["seeds"] == [1, 2, 3]
        assert payload["statistics"]["auroc"]["method"] == "t"

    def test_adapter_forwards_omit_policy(self, tmp_path):
        config = SweepConfig(
            seeds=[1, 2, 3],
            out_root=str(tmp_path / "root"),
            small_n_ci="omit",
        )
        payload = run_sweep_from_config(config, _stub_fn, model_name="m", task_name="t")
        assert payload["statistics"]["auroc"]["method"] == "omitted"

    def test_adapter_requires_out_root(self):
        with pytest.raises(ValueError, match=r"sweep.out_root is not set"):
            run_sweep_from_config(SweepConfig(), _stub_fn, model_name="m", task_name="t")

    def test_adapter_rejects_wrong_type(self, tmp_path):
        with pytest.raises(ValueError, match=r"expects a SweepConfig"):
            run_sweep_from_config(
                {"seeds": [1], "out_root": str(tmp_path)},
                _stub_fn,
                model_name="m",
                task_name="t",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Slow lane: real-model >= 3-seed end-to-end acceptance trial (D-15, SEED-01)
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_MODEL_REPO = "zhangtaolab/plant-dnabert-BPE"
SWEEP_MODEL_LABEL = "plant-dnabert-BPE"
SWEEP_DATASET = "zhangtaolab/plant-multi-species-core-promoters"
# D-16: fixed dataset-derived data-prep seed — deliberately NOT one of the
# sweep seeds, so sampling/splitting is identical across seeds by
# construction and seed-to-seed variance measures init/shuffle only.
SWEEP_DATA_SEED = 0


def _sweep_slow_lane_available() -> str | None:
    """Typed availability check for the slow-lane acceptance artifacts.

    Returns None when the pinned model and dataset are cached (offline
    runnable) or the modelscope route answers; otherwise a typed
    network-unavailable skip reason (allowlisted in expected_skips.yaml).
    """
    ms_cache = Path.home() / ".cache" / "modelscope" / "hub"
    model_cached = (ms_cache / "models" / "zhangtaolab" / "plant-dnabert-BPE").is_dir()
    dataset_glob = list(ms_cache.glob("datasets/zhangtaolab___plant-multi-species-core-promoters*"))
    if model_cached and dataset_glob:
        return None
    try:
        import socket

        with socket.create_connection(("modelscope.cn", 443), timeout=5):
            pass
    except OSError:
        return (
            "network-unavailable: models.lock-pinned plant-dnabert-BPE / "
            "plant-multi-species-core-promoters are not cached and the "
            "modelscope route is unreachable"
        )
    return None


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_sweep_three_seed_end_to_end(tmp_path):
    """D-15 acceptance: a >= 3-seed sweep of one small binary task, end-to-end.

    Pre-splits the dataset ONCE with a fixed data seed (D-16), then runs a
    tiny DNATrainer fine-tune of the pinned plant-dnabert-BPE per sweep seed
    via run_seeds; asserts the directory protocol, the per-seed result
    JSONs, and a t-interval statistics block at n=3.
    """
    reason = _sweep_slow_lane_available()
    if reason:
        pytest.skip(reason)
    import copy

    from datasets import DatasetDict

    from dnallm import DNADataset, DNATrainer, load_config, load_model_and_tokenizer

    from dnallm.finetune.sweep import run_seeds

    base_config = load_config(str(Path(__file__).parent / "test_finetune_config.yaml"))

    # D-16: data preparation happens ONCE, outside the per-seed fn — the
    # sweep seeds never influence sampling or the split.
    _, tokenizer = load_model_and_tokenizer(
        SWEEP_MODEL_REPO, task_config=base_config["task"], source="modelscope"
    )
    datasets = DNADataset.from_modelscope(
        SWEEP_DATASET,
        seq_col="sequence",
        label_col="label",
        tokenizer=tokenizer,
        max_length=128,
    )
    datasets.encode_sequences()
    sampled = datasets.sampling(0.02, seed=SWEEP_DATA_SEED, overwrite=True)
    if not isinstance(sampled.dataset, DatasetDict):
        sampled.split_data(test_size=0.2, val_size=0.1, seed=SWEEP_DATA_SEED)

    def train_one_seed(seed: int, seed_dir: Path) -> dict[str, float]:
        config = copy.deepcopy(base_config)
        finetune = config["finetune"]
        finetune.seed = seed  # D-16: threads init/shuffle only
        finetune.output_dir = str(seed_dir)
        finetune.num_train_epochs = 1
        finetune.max_steps = 5
        finetune.eval_strategy = "no"
        finetune.save_strategy = "no"
        finetune.load_best_model_at_end = False
        finetune.report_to = ["none"]
        finetune.save_total_limit = 1
        run_model, _ = load_model_and_tokenizer(
            SWEEP_MODEL_REPO, task_config=config["task"], source="modelscope"
        )
        trainer = DNATrainer(model=run_model, config=config, datasets=sampled)
        trainer.train(save_tokenizer=False)
        metrics = trainer.evaluate(split="test")
        del trainer, run_model
        return metrics

    run_seeds(
        train_one_seed,
        [42, 43, 44],
        tmp_path,
        model_name=SWEEP_MODEL_LABEL,
        task_name="core_promoters",
    )

    task_root = tmp_path / SWEEP_MODEL_LABEL / "core_promoters"
    for seed in (42, 43, 44):
        seed_dir = task_root / f"seed_{seed}"
        assert seed_dir.is_dir()
        assert (seed_dir / "seed_result.json").is_file()
        # The trainer's own per-seed result JSON lands in the same directory.
        assert (seed_dir / "eval_test_result.json").is_file()

    stats = json.loads((task_root / "statistics.json").read_text())
    assert stats["seeds"] == [42, 43, 44]
    assert stats["statistics"], "statistics block must aggregate at least one metric"
    auroc_block = next(
        (block for name, block in stats["statistics"].items() if name.lower() == "auroc"),
        None,
    )
    assert auroc_block is not None, (
        f"expected an auroc metric among aggregated keys, got {sorted(stats['statistics'])}"
    )
    assert auroc_block["n_seeds"] == 3
    assert auroc_block["method"] == "t"
    lower, upper = auroc_block["ci95"]
    assert math.isfinite(lower)
    assert math.isfinite(upper)
    assert lower < upper
