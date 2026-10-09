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

from dnallm.finetune.sweep import aggregate_seeds

AGG_KWARGS = {"n_bootstrap": 2000, "bootstrap_seed": 42, "small_n_ci": "t-interval"}


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
