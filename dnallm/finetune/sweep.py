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
2. ``run_seeds`` — the sweep orchestrator: one fully-seeded run per seed
   under ``{out_root}/{model}/{task}/seed_{s}/``, then the aggregate
   ``statistics.json`` under ``{out_root}/{model}/{task}/``.
3. ``run_sweep_from_config`` — adapter mapping a ``SweepConfig``
   (``dnallm.configuration.configs``) verbatim onto ``run_seeds``.

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

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


# Aggregate output filename under {out_root}/{model}/{task}/.
STATISTICS_FILENAME = "statistics.json"

# Per-seed result filename inside each seed_{s}/ directory; mirrors the
# Phase-10 trainer result-JSON shape {split -> seed, timestamp, metrics}.
SEED_RESULT_FILENAME = "seed_result.json"

# Characters a single directory segment must never contain (V12: model and
# task names come from config/user strings and flow into a recursive mkdir).
_FORBIDDEN_SEGMENT_CHARS = ("/", "\\", "\0")


def _sanitize_path_segment(name: str, field: str) -> str:
    """Validate one directory segment of the sweep output protocol.

    Args:
        name: The candidate segment (a model or task name).
        field: Which config field ``name`` came from, for the error message.

    Returns:
        The validated segment, unchanged.

    Raises:
        ValueError: If ``name`` is empty/blank, is ``.`` or ``..``, or
            contains a path separator — any of which would escape the
            per-run directory protocol under ``out_root``.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"sweep {field} must be a non-empty path segment, got {name!r}.")
    if name in (".", ".."):
        raise ValueError(
            f"sweep {field} must not be the special segment {name!r}: it would "
            f"not create a distinct per-{field} directory under out_root."
        )
    bad = sorted(ch for ch in _FORBIDDEN_SEGMENT_CHARS if ch in name)
    if bad:
        raise ValueError(
            f"sweep {field} {name!r} must not contain path separators "
            f"({', '.join(repr(c) for c in bad)}): it is used as a single "
            f"directory segment under the sweep out_root."
        )
    return name


def run_seeds(
    fn: Callable[[int, Path], Mapping[str, Any]],
    seeds: Iterable[int],
    out_root: str | Path,
    *,
    model_name: str,
    task_name: str,
    metric_keys: list[str] | None = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    small_n_ci: str = "t-interval",
) -> dict[str, Any]:
    """Run one fully-seeded training run per seed and aggregate statistics.

    Directory protocol: each seed ``s`` runs inside
    ``{out_root}/{model_name}/{task_name}/seed_{s}/`` (created with
    ``parents=True, exist_ok=True``); after all seeds,
    ``{out_root}/{model_name}/{task_name}/statistics.json`` receives the
    statistics block per the module docstring spec for every aggregated
    metric, plus the per-seed result paths.

    Seed semantics (D-16, same-split-across-seeds): ``run_seeds`` threads
    ONE seed into every stochastic stage of a run via ``fn`` —
    initialization and data-order/shuffle effects — while the dataset
    split stays fixed across seeds. The split must be performed ONCE by
    the caller (before the sweep, or with a dataset-derived fixed seed
    inside ``fn``'s closure), so seed-to-seed variance measures
    init/shuffle only, never split variance. ``run_seeds`` itself takes
    no dataset argument and cannot re-split per seed; ``fn`` receives the
    pre-split dataset via its closure and must not derive the split from
    the sweep seed. ``fn`` is called exactly once per seed as
    ``fn(seed, seed_dir)``.

    Determinism contract: identical seeds plus a deterministic ``fn``
    produce byte-identical ``statistics.json`` content on CPU. GPU kernel
    nondeterminism is explicitly out of contract — same-seed CUDA runs
    may differ in low-order bits; the canonical reproducibility guarantee
    is CPU-scoped.

    Args:
        fn: Callable running ONE fully-seeded training run; receives the
            sweep seed and the run's ``seed_{s}`` directory, and returns a
            mapping of metric name to numeric value (e.g. the dict from
            ``DNATrainer.evaluate(split=...)``).
        seeds: Sweep seeds to run (at least one).
        out_root: Root directory for the sweep output (never the current
            working directory — an empty path is rejected).
        model_name: Model label used as one directory segment (must not
            contain path separators).
        task_name: Task label used as one directory segment (must not
            contain path separators).
        metric_keys: Metric names to aggregate. When ``None``, every
            numeric key reported by ALL seeds is aggregated (sorted for
            deterministic output order).
        n_bootstrap: Bootstrap resample count, forwarded to
            :func:`aggregate_seeds`.
        bootstrap_seed: Bootstrap RNG seed, forwarded to
            :func:`aggregate_seeds`.
        small_n_ci: Small-n interval policy, forwarded to
            :func:`aggregate_seeds`.

    Returns:
        The statistics payload written to ``statistics.json``:
        ``{"model_name", "task_name", "seeds", "per_seed", "statistics"}``
        where ``per_seed`` entries carry ``{"seed", "path", "metrics"}``
        with paths relative to the ``{model}/{task}/`` directory (so the
        payload is byte-identical across equal runs regardless of where
        ``out_root`` lives) and ``statistics`` maps each metric to the
        :func:`aggregate_seeds` block.

    Raises:
        ValueError: If ``out_root`` is empty, ``seeds`` is empty,
            ``model_name``/``task_name`` are invalid path segments,
            ``small_n_ci`` is not a valid policy, or ``metric_keys``
            names a metric some seed did not report.
    """
    if out_root is None or not str(out_root).strip():
        raise ValueError(
            "sweep out_root must be a non-empty path; refusing to fall back "
            "to the current working directory."
        )
    if small_n_ci not in SMALL_N_CI_CHOICES:
        raise ValueError(f"small_n_ci must be one of {SMALL_N_CI_CHOICES}, got {small_n_ci!r}.")
    model_name = _sanitize_path_segment(model_name, "model_name")
    task_name = _sanitize_path_segment(task_name, "task_name")
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError(
            "run_seeds requires at least one seed so the protocol always reports a real run."
        )
    # Lazy import keeps the module's aggregation layer import-light (no
    # torch needed to import dnallm.finetune.sweep for statistics-only use).
    from ..utils import get_logger

    logger = get_logger("dnallm.finetune.sweep")

    task_root = Path(out_root) / model_name / task_name
    task_root.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []
    for seed in seed_list:
        seed_dir = task_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics = dict(fn(seed, seed_dir))
        result_path = seed_dir / SEED_RESULT_FILENAME
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "task_name": task_name,
                    "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        per_seed.append({
            "seed": seed,
            "path": f"seed_{seed}/{SEED_RESULT_FILENAME}",
            "metrics": metrics,
        })

    if metric_keys is None:
        common = set(per_seed[0]["metrics"])
        for entry in per_seed[1:]:
            common &= set(entry["metrics"])
        keys: list[str] = sorted(common)
    else:
        keys = list(metric_keys)
        for key in keys:
            missing = [entry["seed"] for entry in per_seed if key not in entry["metrics"]]
            if missing:
                raise ValueError(
                    f"metric_keys requested '{key}' but seed(s) {missing} did "
                    f"not report it; every aggregated metric must be present "
                    f"in every seed's result."
                )
    statistics: dict[str, dict[str, Any]] = {}
    for key in keys:
        values: list[float] = []
        numeric = True
        for entry in per_seed:
            value = entry["metrics"][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                logger.info(
                    f"[Info] Metric '{key}' is not numeric "
                    f"({type(value).__name__}); skipping its aggregation."
                )
                numeric = False
                break
            values.append(float(value))
        if not numeric:
            continue
        statistics[key] = aggregate_seeds(
            values,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            small_n_ci=small_n_ci,
        )

    payload: dict[str, Any] = {
        "model_name": model_name,
        "task_name": task_name,
        "seeds": seed_list,
        "per_seed": per_seed,
        "statistics": statistics,
    }
    stats_path = task_root / STATISTICS_FILENAME
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        f"[Success] Sweep over {len(seed_list)} seed(s) for {model_name}/{task_name} "
        f"written to {stats_path}"
    )
    return payload


def run_sweep_from_config(
    sweep_config: Any,
    fn: Callable[[int, Path], Mapping[str, Any]],
    *,
    model_name: str,
    task_name: str,
    metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Adapt a ``SweepConfig`` onto :func:`run_seeds` arguments verbatim.

    ``SweepConfig`` (``dnallm.configuration.configs``, the Phase-10
    scaffold) is consumed as-is: ``seeds`` and ``out_root`` positionally,
    ``n_bootstrap``/``bootstrap_seed``/``small_n_ci`` as the aggregation
    knobs. No configuration fields are added or interpreted beyond that.

    Args:
        sweep_config: The ``SweepConfig`` section of a loaded config.
        fn: Per-seed run callable, forwarded to :func:`run_seeds`.
        model_name: Model directory label, forwarded to :func:`run_seeds`.
        task_name: Task directory label, forwarded to :func:`run_seeds`.
        metric_keys: Optional metric allowlist, forwarded to
            :func:`run_seeds`.

    Returns:
        The statistics payload from :func:`run_seeds`.

    Raises:
        ValueError: If ``sweep_config`` is not a ``SweepConfig`` or its
            ``out_root`` is unset (plus everything :func:`run_seeds`
            raises).
    """
    from ..configuration.configs import SweepConfig

    if not isinstance(sweep_config, SweepConfig):
        raise ValueError(
            f"run_sweep_from_config expects a SweepConfig, got {type(sweep_config).__name__}."
        )
    if not sweep_config.out_root:
        raise ValueError(
            "sweep.out_root is not set: run_seeds needs an explicit output "
            "root and never falls back to the current working directory. "
            "Set sweep.out_root in the config."
        )
    return run_seeds(
        fn,
        sweep_config.seeds,
        sweep_config.out_root,
        model_name=model_name,
        task_name=task_name,
        metric_keys=metric_keys,
        n_bootstrap=sweep_config.n_bootstrap,
        bootstrap_seed=sweep_config.bootstrap_seed,
        small_n_ci=sweep_config.small_n_ci,
    )
