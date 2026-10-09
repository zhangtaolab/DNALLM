"""Multi-seed sweep protocol with honest uncertainty aggregation.

This module answers the reviewer reproducibility challenge (R1-2a, REV-09):
multi-seed training runs reported with uncertainty aggregates that never fake
precision. The aggregation layer is a pure numpy/scipy function — no torch —
so the statistics contract is testable and reusable independently of any
model code, and the orchestration layer deliberately stays outside
``DNATrainer``: a sweep drives N trainer runs from the outside rather than
changing what one trainer run does.

Result-JSON ``statistics`` block spec (the SEED-01 deliverable):

Every ``statistics.json`` written under
``{out_root}/{model}/{task}/statistics.json`` maps each aggregated metric
name to a block with exactly these keys:

- ``n_seeds`` (int): number of seed runs aggregated.
- ``mean`` (float): arithmetic mean of the per-seed values.
- ``sd`` (float | null): sample standard deviation (ddof=1); ``None`` at
  n=1 where a deviation is undefined.
- ``ci95`` ([float, float] | null): 95% confidence interval on the mean as
  ``[lower, upper]``; ``None`` whenever the seed count cannot support an
  honest interval.
- ``method`` (str): how ``ci95`` was produced — one of ``"none"``
  (n < 3: no interval is reported at all), ``"t"`` (3 <= n < 10:
  Student-t interval), ``"omitted"`` (3 <= n < 10 with
  ``small_n_ci="omit"``), or ``"bootstrap-percentile"`` (n >= 10: seeded
  percentile bootstrap).

The n-guard is the point of the design (D-14): a bootstrap interval over
three seeds has only ten distinct resample multisets, so reporting one
would be vacuous precision. Below three seeds no interval is emitted in
any branch.

Features:

1. ``aggregate_seeds`` — pure, torch-free mean/sd/ci95 with the n-guard:
   no interval below 3 seeds, a Student-t interval for 3-9 seeds, and a
   seeded percentile bootstrap only from 10 seeds on.

Example:
    >>> from dnallm.finetune.sweep import aggregate_seeds
    >>> block = aggregate_seeds(
    ...     [0.71, 0.73, 0.75],
    ...     n_bootstrap=2000,
    ...     bootstrap_seed=42,
    ...     small_n_ci="t-interval",
    ... )
    >>> block["n_seeds"], block["method"]
    (3, 't')
    >>> block["ci95"] is not None
    True
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np
from scipy import stats

# Valid values for the small-n CI policy (mirrors SweepConfig.small_n_ci's
# pattern ^(t-interval|omit)$; validated here because aggregate_seeds is also
# callable directly, without going through the Pydantic boundary).
SMALL_N_CI_CHOICES = ("t-interval", "omit")

# Minimum seed count for ANY confidence interval (D-14): below this the
# statistics block reports method "none" and ci95 null.
CI_MIN_SEEDS = 3

# Minimum seed count for the percentile bootstrap: the resample space is too
# small to be meaningful below this (10 distinct resample multisets at n=3).
BOOTSTRAP_MIN_SEEDS = 10


def aggregate_seeds(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    small_n_ci: str,
) -> dict[str, Any]:
    """Aggregate per-seed metric values into the statistics block.

    Implements the n-guard (D-14): ``n < 3`` reports no interval
    (``method="none"``); ``3 <= n < 10`` reports a Student-t interval
    (``method="t"``) or omits it when ``small_n_ci="omit"``
    (``method="omitted"``); ``n >= 10`` reports a seeded percentile
    bootstrap (``method="bootstrap-percentile"``) regardless of
    ``small_n_ci`` (that policy only governs the small-n range).

    Args:
        values: Per-seed metric values (one per seed run).
        n_bootstrap: Number of bootstrap resamples (used only at
            ``n >= 10``).
        bootstrap_seed: Seed for the bootstrap RNG, so identical data and
            seed produce an identical interval (used only at ``n >= 10``).
        small_n_ci: Small-n interval policy, ``"t-interval"`` or
            ``"omit"`` (used only for ``3 <= n < 10``).

    Returns:
        The statistics block ``{"n_seeds", "mean", "sd", "ci95",
        "method"}`` per the module docstring spec.

    Raises:
        ValueError: If ``small_n_ci`` is not one of the valid policies, if
            ``values`` is empty, or if ``values`` is not a flat numeric
            sequence.
    """
    if small_n_ci not in SMALL_N_CI_CHOICES:
        raise ValueError(f"small_n_ci must be one of {SMALL_N_CI_CHOICES}, got {small_n_ci!r}.")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"aggregate_seeds expects a flat sequence of per-seed values, "
            f"got an array with shape {arr.shape}."
        )
    n = int(arr.size)
    if n < 1:
        raise ValueError("aggregate_seeds requires at least one per-seed value; got none.")
    out: dict[str, Any] = {
        "n_seeds": n,
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if n > 1 else None,
    }
    if n < CI_MIN_SEEDS:
        # Never vacuous: no interval is emitted below three seeds.
        out["ci95"] = None
        out["method"] = "none"
    elif n < BOOTSTRAP_MIN_SEEDS and small_n_ci == "t-interval":
        sem = arr.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        out["ci95"] = [
            float(arr.mean() - tcrit * sem),
            float(arr.mean() + tcrit * sem),
        ]
        out["method"] = "t"
    elif n >= BOOTSTRAP_MIN_SEEDS:
        rng = np.random.default_rng(bootstrap_seed)
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        means = arr[idx].mean(axis=1)
        out["ci95"] = [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ]
        out["method"] = "bootstrap-percentile"
    else:
        # 3 <= n < 10 with small_n_ci == "omit": point estimates only.
        out["ci95"] = None
        out["method"] = "omitted"
    return out
