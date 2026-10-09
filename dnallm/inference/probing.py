"""Frozen-embedding probing for DNA large language models.

This module lets reviewers probe how much information frozen (non-fine-tuned)
DNA large language model embeddings carry about a binary classification
task, at any hidden-state layer and with a choice of pooling — the probing
baseline that makes fine-tuning gains interpretable (reviewer item R2-2,
REV-07). It is a pure consumer of the model-facing mechanics that already
exist in ``dnallm/inference/inference.py`` (the ``output_hidden_states``
forward path and the masked-mean pooling idiom) and of the Phase-10 metric
registry: probe metrics are emitted exclusively through
``dnallm.tasks.metric_registry``.

Features:

1. ``extract_embeddings`` — frozen-embedding extraction with a selectable
   hidden-state layer (``int``, negative indices allowed, default ``-1`` for
   the last layer) and pooling strategy (``"mean"`` masked mean over real
   tokens, or ``"cls"`` first token).
2. Embedding cache — when ``output_dir`` is supplied, embeddings are cached
   as ``float32`` npz files under ``{output_dir}/probe_cache/`` keyed by the
   ``(model, dataset, layer, pooling)`` tuple; filenames are sanitized hashes
   of the key (never raw model/dataset ids) and writes are temp-file plus
   ``os.replace`` so concurrent same-key writes leave one valid winner.
3. ``fit_probe`` — a fixed-hyperparameter sklearn probe (logistic or MLP)
   with a ``StandardScaler`` fit on the TRAIN split only, so held-out metrics
   never encode test-split statistics.
4. Registry metrics — probe outputs carry metric keys in the canonical
   registry spellings (``AUROC``, ``AUPRC``, ``accuracy``) resolved through
   ``metric_registry.resolve``.

Output schema (F4 contract for the dnallmmark probing-comparison lane):
every probe result row — ``ProbeResult.to_row()`` — carries exactly

    {"model": str | None,          # model_name given to extract_embeddings
     "dataset": str | None,        # dataset_name given to extract_embeddings
     "layer": int,                 # hidden-state layer the probe read
     "pooling": "mean" | "cls",    # pooling strategy used
     "kind": "logistic" | "mlp",   # probe estimator kind
     "metrics": {"AUROC": float, "AUPRC": float, "accuracy": float},
     "n_train": int,               # training rows the probe was fit on
     "n_test": int,                # held-out rows the metrics were computed on
     "cache_hit": bool | None}     # whether embeddings came from the cache

The fixed probe hyperparameters are module-level constants (no YAML config
surface): changing one is a visible, documented protocol decision, not a
per-run knob.

Example:
    >>> from dnallm.inference.probing import extract_embeddings, fit_probe
    >>> result = extract_embeddings(
    ...     model, tokenizer, sequences, labels,
    ...     layer=-1, pooling="mean",
    ...     model_name="plant-dnabert-BPE", dataset_name="core-promoters",
    ...     output_dir="results",  # enables the probe_cache
    ... )
    >>> train_X, train_y = result.embeddings[:40], result.labels[:40]
    >>> test_X, test_y = result.embeddings[40:], result.labels[40:]
    >>> probe = fit_probe(train_X, train_y, test_X, test_y, kind="logistic")
    >>> probe.metrics["AUROC"] > 0.5
    True
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..tasks.metric_registry import resolve, validate_emission
from ..utils import get_logger

logger = get_logger("dnallm.inference.probing")

__all__ = [
    "LOGISTIC_MAX_ITER",
    "LOGISTIC_SOLVER",
    "MLP_EARLY_STOP",
    "MLP_HIDDEN",
    "EmbeddingResult",
    "ProbeResult",
    "extract_embeddings",
    "fit_probe",
]

# D-13: fixed probe hyperparameters. SC3 fixes them this milestone; there is
# deliberately no ProbeConfig YAML surface — comparability across runs and
# across models requires every probe to use exactly these values.
LOGISTIC_MAX_ITER = 1000
"""int: Maximum lbfgs iterations for the logistic probe (``LogisticRegression``)."""

LOGISTIC_SOLVER = "lbfgs"
"""str: Solver for the logistic probe (``LogisticRegression``)."""

MLP_HIDDEN = (256,)
"""tuple[int, ...]: Hidden layer sizes for the MLP probe — one shallow layer, per the
probing literature convention that keeps probes from becoming fine-tuners."""

MLP_EARLY_STOP = True
"""bool: sklearn early stopping for the MLP probe (uses an internal train-split
fraction for validation, so the held-out test split is still untouched)."""

PROBE_KINDS = ("logistic", "mlp")
"""tuple[str, ...]: The probe estimator kinds ``fit_probe`` accepts."""

POOLING_STRATEGIES = ("mean", "cls")
"""tuple[str, ...]: The pooling strategies ``extract_embeddings`` accepts."""

_FORWARD_BATCH_SIZE = 32
"""int: Rows per model forward call during extraction (memory bound, not a protocol knob)."""


@dataclass
class EmbeddingResult:
    """Result of one frozen-embedding extraction.

    Attributes:
        embeddings: Pooled embeddings as a ``(n_rows, hidden_size)`` float32
            ndarray.
        labels: Labels aligned with ``embeddings`` rows (post-filtering).
        model_name: Model id recorded in the cache key; ``None`` when the
            caller did not supply one.
        dataset_name: Dataset id recorded in the cache key; ``None`` when the
            caller did not supply one.
        layer: The hidden-state layer that was read (as given; negative
            indices index from the last layer).
        pooling: The pooling strategy that produced each row.
        cache_hit: Whether the embeddings were loaded from the npz cache
            instead of recomputed.
        cache_path: Path of the cache file when ``output_dir`` was supplied,
            else ``None``.
        dtype: Storage dtype of the embeddings (always ``"float32"``).
    """

    embeddings: np.ndarray
    labels: np.ndarray
    model_name: str | None
    dataset_name: str | None
    layer: int
    pooling: str
    cache_hit: bool
    cache_path: str | None
    dtype: str = "float32"


@dataclass
class ProbeResult:
    """Result of fitting and evaluating one frozen-embedding probe.

    One row of the F4 output schema; see the module docstring for the exact
    contract the dnallmmark probing-comparison lane consumes.

    Attributes:
        kind: Probe estimator kind (``"logistic"`` or ``"mlp"``).
        layer: Hidden-state layer the embeddings were read from, when known
            (``None`` if the caller probed arrays of unknown provenance).
        pooling: Pooling strategy of the embeddings, when known.
        metrics: Held-out metrics with canonical registry keys (``AUROC``,
            ``AUPRC``, ``accuracy``) as floats.
        n_train: Number of training rows the probe was fit on.
        n_test: Number of held-out rows the metrics were computed on.
        model_name: Model id, when known.
        dataset_name: Dataset id, when known.
        cache_hit: Whether the embeddings came from the probe cache, when
            known.
        estimator: The fitted sklearn estimator.
        scaler: The fitted ``StandardScaler`` (fit on the train split only).
    """

    kind: str
    layer: int | None
    pooling: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    model_name: str | None = None
    dataset_name: str | None = None
    cache_hit: bool | None = None
    estimator: Any = None
    scaler: Any = None

    def to_row(self) -> dict[str, Any]:
        """Return this result as one F4-schema output row.

        Returns:
            Dict with keys ``model``, ``dataset``, ``layer``, ``pooling``,
            ``kind``, ``metrics``, ``n_train``, ``n_test``, ``cache_hit``.
        """
        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "layer": self.layer,
            "pooling": self.pooling,
            "kind": self.kind,
            "metrics": dict(self.metrics),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "cache_hit": self.cache_hit,
        }


def _cache_key(model_name: str | None, dataset_name: str | None, layer: int, pooling: str) -> list:
    """Build the canonical cache key list for the 4-tuple contract.

    Args:
        model_name: Model id (or ``None``).
        dataset_name: Dataset id (or ``None``).
        layer: Hidden-state layer index.
        pooling: Pooling strategy name.

    Returns:
        JSON-roundtrippable list form of ``(model, dataset, layer, pooling)``.
    """
    return [model_name, dataset_name, int(layer), pooling]


def _cache_filename(key: list) -> str:
    """Return the sanitized cache filename for a key.

    The filename is a stable sha256 of the key tuple's repr — hex-only, so
    model/dataset ids containing path separators can never traverse out of
    the cache directory (T-11-06).

    Args:
        key: Cache key list from ``_cache_key``.

    Returns:
        Filename of the form ``probe_<24-hex-chars>.npz``.
    """
    raw = "|".join(repr(part) for part in key)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"probe_{digest[:24]}.npz"


def _write_cache(cache_path: Path, embeddings: np.ndarray, labels: np.ndarray, meta: dict) -> None:
    """Atomically write one npz cache entry.

    The npz is written to a unique temp name in the same directory and then
    moved into place with ``os.replace`` (atomic within a filesystem), so two
    concurrent writers of the same key produce exactly one valid winner and
    readers never observe a half-written file.

    Args:
        cache_path: Final cache file path (already inside ``probe_cache/``).
        embeddings: float32 embedding matrix to store.
        labels: Label array aligned with the embeddings.
        meta: Metadata dict (key tuple + dtype) stored as a JSON string.
    """
    tmp_path = cache_path.parent / f"{cache_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "wb") as handle:
            np.savez(handle, embeddings=embeddings, labels=labels, meta=np.array(json.dumps(meta)))
        os.replace(tmp_path, cache_path)
    except OSError:
        # Never leave a temp file behind on a failed write.
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info(f"Wrote probe cache entry {cache_path.name} ({embeddings.shape[0]} rows)")


def _load_cache(
    cache_path: Path, model_name: str | None, dataset_name: str | None, layer: int, pooling: str
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Load a cache entry when one exists for exactly this key.

    Args:
        cache_path: Expected cache file path.
        model_name: Model id of the requested key.
        dataset_name: Dataset id of the requested key.
        layer: Layer index of the requested key.
        pooling: Pooling strategy of the requested key.

    Returns:
        Tuple ``(embeddings, labels, metadata)`` on a hit, ``None`` on a miss.
        Unreadable or key-mismatched files are treated as misses (with a
        warning), never as failures.
    """
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if _cache_key(
                meta.get("model"), meta.get("dataset"), meta.get("layer"), meta.get("pooling")
            ) != _cache_key(model_name, dataset_name, layer, pooling):
                return None
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            labels = np.asarray(data["labels"])
    except (OSError, ValueError, KeyError) as e:
        warnings.warn(
            f"Ignoring unreadable probe cache entry {cache_path.name}: {e}",
            stacklevel=2,
        )
        return None
    logger.info(f"Probe cache hit for {cache_path.name}")
    return embeddings, labels, meta


def _model_device(model: Any) -> torch.device:
    """Return the device the model's parameters live on.

    Args:
        model: Torch model (or model-like object).

    Returns:
        The device of the first parameter, or CPU when the object exposes no
        parameters.
    """
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _attention_mask_from_inputs(
    inputs: dict[str, Any], tokenizer: Any
) -> torch.Tensor | None:
    """Return the attention mask for a tokenized batch.

    Mirrors the ``DNAInference`` mask handling: prefer the tokenizer-provided
    mask, else derive it from the padding token id.

    Args:
        inputs: Tokenized batch (``input_ids`` and maybe ``attention_mask``).
        tokenizer: The tokenizer that produced the batch.

    Returns:
        LongTensor attention mask, or ``None`` when neither source is
        available.
    """
    if "attention_mask" in inputs:
        return inputs["attention_mask"].long()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None and "input_ids" in inputs:
        return (inputs["input_ids"] != pad_id).long()
    return None


def _hidden_states_from_outputs(outputs: Any) -> list[Any] | None:
    """Pull the per-layer hidden states out of a model output.

    Follows the same precedence as ``DNAInference._process_hidden_states``:
    ``outputs.hidden_states``, dict key, ``last_hidden_state``, then the first
    element of tuple/list outputs.

    Args:
        outputs: Whatever the model's forward returned.

    Returns:
        List of per-layer hidden-state tensors, or ``None`` when the output
        carries none.
    """
    hiddens = getattr(outputs, "hidden_states", None)
    if hiddens is None and isinstance(outputs, dict):
        hiddens = outputs.get("hidden_states")
    if hiddens is None:
        hiddens = getattr(outputs, "last_hidden_state", None)
    if hiddens is None and isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        hiddens = outputs[0]
    if hiddens is None:
        return None
    if isinstance(hiddens, (list, tuple)):
        return list(hiddens)
    return [hiddens]


def extract_embeddings(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
    labels: list[Any],
    *,
    layer: int = -1,
    pooling: str = "mean",
    model_name: str | None = None,
    dataset_name: str | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = _FORWARD_BATCH_SIZE,
) -> EmbeddingResult:
    """Extract pooled frozen embeddings from one hidden-state layer.

    The forward pass reuses the model's existing ``output_hidden_states``
    mechanics (forward-argument introspection plus the config flag, the same
    idiom ``DNAInference`` uses) — this function only adds layer selection,
    pooling, and the npz cache; it never re-implements model-facing behavior.

    Args:
        model: Model whose hidden states are read (any callable-forward
            model returning hidden states under ``output_hidden_states``).
        tokenizer: Hugging Face-style callable tokenizer.
        sequences: DNA sequences to embed.
        labels: Labels aligned with ``sequences`` (same length).
        layer: Hidden-state layer to read; negative indices count from the
            last layer (default ``-1``).
        pooling: ``"mean"`` (masked mean over real tokens) or ``"cls"``
            (first token).
        model_name: Model id recorded in the cache key and result.
        dataset_name: Dataset id recorded in the cache key and result.
        output_dir: When supplied, embeddings are cached under
            ``{output_dir}/probe_cache/`` keyed by
            ``(model_name, dataset_name, layer, pooling)``; a matching entry
            short-circuits the forward pass.
        batch_size: Rows per forward call (memory bound only).

    Returns:
        EmbeddingResult with float32 embeddings of shape
        ``(n_rows, hidden_size)``.

    Raises:
        ValueError: If ``pooling`` is unknown, ``layer`` is not an integer,
            ``sequences``/``labels`` lengths differ, no sequence remains
            after filtering out empty entries, or ``layer`` is out of range
            for the model's hidden-state stack.
    """
    if pooling not in POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy '{pooling}'. Valid strategies: {POOLING_STRATEGIES}"
        )
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise ValueError(f"layer must be an int, got {type(layer).__name__}.")
    layer = int(layer)

    sequences = [str(seq) for seq in sequences]
    labels = list(labels)
    if len(sequences) != len(labels):
        raise ValueError(
            f"sequences and labels lengths differ ({len(sequences)} vs {len(labels)})."
        )

    cache_path: Path | None = None
    if output_dir is not None:
        cache_path = Path(output_dir) / "probe_cache" / _cache_filename(
            _cache_key(model_name, dataset_name, layer, pooling)
        )
        cached = _load_cache(cache_path, model_name, dataset_name, layer, pooling)
        if cached is not None:
            embeddings, cached_labels, _meta = cached
            return EmbeddingResult(
                embeddings=embeddings,
                labels=cached_labels,
                model_name=model_name,
                dataset_name=dataset_name,
                layer=layer,
                pooling=pooling,
                cache_hit=True,
                cache_path=str(cache_path),
            )

    kept = [
        (seq, label) for seq, label in zip(sequences, labels) if seq.strip()
    ]
    if not kept:
        raise ValueError(
            f"No sequences remain after filtering ({len(sequences)} input sequences); "
            "extract_embeddings requires at least one non-empty sequence."
        )
    kept_sequences = [seq for seq, _ in kept]
    kept_labels = [label for _, label in kept]

    device = _model_device(model)
    if hasattr(model, "eval"):
        model.eval()

    # Reuse the output_hidden_states mechanics the same way DNAInference
    # does: pass the kwarg when the forward signature accepts it, and flip
    # the config flag for models that only read it from the config.
    params = inspect.signature(model.forward).parameters
    forward_kwargs: dict[str, Any] = {}
    if "output_hidden_states" in params:
        forward_kwargs["output_hidden_states"] = True
        try:
            model.config.output_hidden_states = True
        except (ValueError, AttributeError) as e:
            warnings.warn(f"Cannot enable output_hidden_states on config: {e}", stacklevel=2)

    pooled_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(kept_sequences), batch_size):
            batch = kept_sequences[start : start + batch_size]
            inputs = dict(tokenizer(batch, return_tensors="pt", padding=True, truncation=True))
            input_ids = inputs["input_ids"].to(device)
            attention_mask = _attention_mask_from_inputs(inputs, tokenizer)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask.to(device)
                if attention_mask is not None
                else None,
                **forward_kwargs,
            )
            states = _hidden_states_from_outputs(outputs)
            if states is None:
                raise ValueError(
                    "Model forward returned no hidden states; probing requires a model that "
                    "exposes them (output_hidden_states=True path)."
                )
            n_layers = len(states)
            if not -n_layers <= layer < n_layers:
                raise ValueError(
                    f"layer {layer} is out of range for a model exposing {n_layers} "
                    f"hidden-state tensors (valid range: -{n_layers}..{n_layers - 1})."
                )
            hidden = states[layer].detach().float().cpu()
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled_batches.append(pooled.numpy().astype(np.float32))

    embeddings = np.concatenate(pooled_batches, axis=0)
    labels_arr = np.asarray(kept_labels)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_cache(
            cache_path,
            embeddings,
            labels_arr,
            {
                "model": model_name,
                "dataset": dataset_name,
                "layer": layer,
                "pooling": pooling,
                "dtype": "float32",
            },
        )

    return EmbeddingResult(
        embeddings=embeddings,
        labels=labels_arr,
        model_name=model_name,
        dataset_name=dataset_name,
        layer=layer,
        pooling=pooling,
        cache_hit=False,
        cache_path=str(cache_path) if cache_path is not None else None,
    )


def fit_probe(
    train_embeddings: Any,
    train_labels: Any,
    test_embeddings: Any,
    test_labels: Any,
    *,
    kind: str = "logistic",
    layer: int | None = None,
    pooling: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    cache_hit: bool | None = None,
) -> ProbeResult:
    """Fit a fixed-hyperparameter sklearn probe on frozen embeddings.

    The ``StandardScaler`` is fit on the train split only and then applied to
    both splits, so held-out metrics never encode test-split statistics. All
    hyperparameters come from the module-level constants (D-13); metrics are
    computed exclusively through the metric registry with canonical keys.

    Args:
        train_embeddings: ``(n_train, hidden)`` training embeddings.
        train_labels: Training labels (binary).
        test_embeddings: ``(n_test, hidden)`` held-out embeddings.
        test_labels: Held-out labels (binary).
        kind: ``"logistic"`` or ``"mlp"``.
        layer: Layer the embeddings came from, recorded in the result.
        pooling: Pooling the embeddings used, recorded in the result.
        model_name: Model id, recorded in the result row.
        dataset_name: Dataset id, recorded in the result row.
        cache_hit: Whether the embeddings came from the cache, recorded in
            the result row.

    Returns:
        ProbeResult with registry-canonical held-out metrics.

    Raises:
        ValueError: If ``kind`` is not ``"logistic"`` or ``"mlp"``.
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"Unknown probe kind '{kind}'. Valid kinds: {PROBE_KINDS}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray(train_embeddings, dtype=np.float32)
    x_test = np.asarray(test_embeddings, dtype=np.float32)
    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    # Leakage discipline (Pitfall 8): the scaler sees ONLY the train split.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if kind == "logistic":
        estimator = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, solver=LOGISTIC_SOLVER)
    else:
        estimator = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            early_stopping=MLP_EARLY_STOP,
        )
    estimator.fit(x_train_scaled, y_train)

    proba = estimator.predict_proba(x_test_scaled)
    scores = proba[:, 1] if proba.shape[1] == 2 else proba
    preds = estimator.predict(x_test_scaled)

    metrics: dict[str, float] = {
        "AUROC": resolve("AUROC")(y_test, scores),
        "AUPRC": resolve("AUPRC")(y_test, scores),
        "accuracy": resolve("accuracy")(y_test, preds),
    }
    validate_emission(metrics.keys())

    return ProbeResult(
        kind=kind,
        layer=layer,
        pooling=pooling,
        metrics=metrics,
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        model_name=model_name,
        dataset_name=dataset_name,
        cache_hit=cache_hit,
        estimator=estimator,
        scaler=scaler,
    )
