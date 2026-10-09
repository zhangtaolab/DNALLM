"""DNA Large Language Model Metric Registry Module.

This module is the single metric name authority for dnallm and the companion
dnallmmark benchmark repository: every metric key emitted by any evaluation
path in ``dnallm/tasks/metrics.py`` resolves through this registry, and every
name the registry recognizes maps to exactly one canonical spelling.

1. Canonical names anchor the CURRENT emitted spellings (accuracy, precision,
   recall, f1 with ``_micro``/``_weighted``/``_samples`` variants, mcc,
   matthews_correlation, AUROC, AUPRC, AUROC_ovr, AUROC_ovo, TPR, TNR, FPR,
   FNR, mse, mae, r2, pearsonr, spearmanr).
2. Historical aliases (``eval_auroc``, ``eval_auprc``, ``eval_spearman_r``,
   ``eval_pearson_r``) and the ``eval_``-prefixed Trainer output spellings
   (``eval_accuracy``, ``eval_AUROC``, ...) are recognized for resolution but
   NEVER emitted - alias handling is strictly one-directional.
3. Matching is exact and case-sensitive: ``eval_auroc`` and ``eval_AUROC``
   are both registered aliases of ``AUROC``; any unregistered casing raises
   ``ValueError``.
4. The registry surface is frozen: it is built once at import through an
   invariant-checking builder, exposed as a read-only mapping, and offers no
   registration or mutation API.
5. The module is import-light: no torch or sklearn imports at module level -
   each metric callable lazily imports the library it wraps, so name
   resolution costs nothing until a metric is actually computed.

Example:
    from dnallm.tasks.metric_registry import canonical_name, resolve

    fn = resolve("eval_spearman_r")  # historical alias -> spearmanr callable
    name = canonical_name("eval_AUROC")  # -> 'AUROC'
    value = fn([1.0, 2.0, 3.0], [1.1, 2.2, 2.8])
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

__all__ = [
    "METRIC_REGISTRY",
    "canonical_name",
    "registered_names",
    "resolve",
    "validate_emission",
]


# ---------------------------------------------------------------------------
# Per-metric callables.
#
# Every callable takes (y_true, y_pred) and returns a float; the libraries it
# wraps are imported INSIDE the function body so importing this module (and
# resolving names) never pulls torch/sklearn/scipy.
#
# Input families, per canonical entry:
# - count metrics (accuracy, precision*, recall*, f1*, mcc,
#   matthews_correlation, TPR, TNR, FPR, FNR): hard predicted labels in
#   y_pred (1D binary/multiclass, or 2D multilabel indicator for *_samples);
# - score metrics (AUROC, AUPRC, AUROC_ovr, AUROC_ovo): probabilities in
#   y_pred (1D positive-class scores for binary, or a 2D class-probability /
#   label-indicator matrix);
# - continuous metrics (mse, mae, r2, pearsonr, spearmanr): floats.
# ---------------------------------------------------------------------------


def _accuracy(y_true: Any, y_pred: Any) -> float:
    """Accuracy over hard predictions."""
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_true, y_pred))


def _precision(y_true: Any, y_pred: Any) -> float:
    """Binary precision over hard predictions."""
    from sklearn.metrics import precision_score

    return float(precision_score(y_true, y_pred, zero_division=0))


def _recall(y_true: Any, y_pred: Any) -> float:
    """Binary recall over hard predictions."""
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, y_pred, zero_division=0))


def _f1(y_true: Any, y_pred: Any) -> float:
    """Binary F1 over hard predictions."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, zero_division=0))


def _f1_micro(y_true: Any, y_pred: Any) -> float:
    """Micro-averaged F1 over hard multiclass/multilabel predictions."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def _f1_weighted(y_true: Any, y_pred: Any) -> float:
    """Support-weighted F1 over hard multiclass/multilabel predictions."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _f1_samples(y_true: Any, y_pred: Any) -> float:
    """Per-sample averaged F1 over 2D multilabel hard predictions."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="samples", zero_division=0))


def _precision_micro(y_true: Any, y_pred: Any) -> float:
    """Micro-averaged precision over hard multiclass/multilabel predictions."""
    from sklearn.metrics import precision_score

    return float(precision_score(y_true, y_pred, average="micro", zero_division=0))


def _precision_weighted(y_true: Any, y_pred: Any) -> float:
    """Support-weighted precision over hard multiclass/multilabel predictions."""
    from sklearn.metrics import precision_score

    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


def _precision_samples(y_true: Any, y_pred: Any) -> float:
    """Per-sample averaged precision over 2D multilabel hard predictions."""
    from sklearn.metrics import precision_score

    return float(precision_score(y_true, y_pred, average="samples", zero_division=0))


def _recall_micro(y_true: Any, y_pred: Any) -> float:
    """Micro-averaged recall over hard multiclass/multilabel predictions."""
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, y_pred, average="micro", zero_division=0))


def _recall_weighted(y_true: Any, y_pred: Any) -> float:
    """Support-weighted recall over hard multiclass/multilabel predictions."""
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def _recall_samples(y_true: Any, y_pred: Any) -> float:
    """Per-sample averaged recall over 2D multilabel hard predictions."""
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, y_pred, average="samples", zero_division=0))


def _mcc(y_true: Any, y_pred: Any) -> float:
    """Matthews correlation coefficient over hard predictions."""
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred))


def _matthews_correlation(y_true: Any, y_pred: Any) -> float:
    """Matthews correlation coefficient (long spelling of ``mcc``)."""
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred))


def _tpr(y_true: Any, y_pred: Any) -> float:
    """True positive rate from the binary confusion matrix."""
    from sklearn.metrics import confusion_matrix

    _tn, _fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _tnr(y_true: Any, y_pred: Any) -> float:
    """True negative rate from the binary confusion matrix."""
    from sklearn.metrics import confusion_matrix

    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _fpr(y_true: Any, y_pred: Any) -> float:
    """False positive rate from the binary confusion matrix."""
    from sklearn.metrics import confusion_matrix

    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def _fnr(y_true: Any, y_pred: Any) -> float:
    """False negative rate from the binary confusion matrix."""
    from sklearn.metrics import confusion_matrix

    _tn, _fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0


def _auroc(y_true: Any, y_pred: Any) -> float:
    """Area under the ROC curve.

    Accepts 1D positive-class scores (binary), a 2D label-indicator/score
    pair (multilabel, macro-averaged), a 2D two-column probability matrix
    (binary, positive column used), or a 2D multi-class probability matrix
    (macro one-vs-rest).
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(y_true)
    scores = np.asarray(y_pred)
    if scores.ndim == 1 or labels.ndim == 2:
        return float(roc_auc_score(labels, scores))
    if scores.shape[1] == 2:
        return float(roc_auc_score(labels, scores[:, 1]))
    return float(roc_auc_score(labels, scores, average="macro", multi_class="ovr"))


def _auprc(y_true: Any, y_pred: Any) -> float:
    """Average precision (area under the precision-recall curve).

    Accepts the same score shapes as :func:`_auroc`.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    labels = np.asarray(y_true)
    scores = np.asarray(y_pred)
    if scores.ndim == 1:
        return float(average_precision_score(labels, scores))
    if scores.shape[1] == 2 and labels.ndim == 1:
        return float(average_precision_score(labels, scores[:, 1]))
    return float(average_precision_score(labels, scores, average="macro"))


def _auroc_ovr(y_true: Any, y_pred: Any) -> float:
    """Macro one-vs-rest ROC AUC over a 2D class-probability matrix."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(y_pred)
    if scores.ndim == 2 and scores.shape[1] > 2:
        return float(roc_auc_score(y_true, scores, average="macro", multi_class="ovr"))
    return float(roc_auc_score(y_true, scores))


def _auroc_ovo(y_true: Any, y_pred: Any) -> float:
    """Macro one-vs-one ROC AUC over a 2D class-probability matrix."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(y_pred)
    if scores.ndim == 2 and scores.shape[1] > 2:
        return float(roc_auc_score(y_true, scores, average="macro", multi_class="ovo"))
    return float(roc_auc_score(y_true, scores))


def _mse(y_true: Any, y_pred: Any) -> float:
    """Mean squared error over continuous predictions."""
    from sklearn.metrics import mean_squared_error

    return float(mean_squared_error(y_true, y_pred))


def _mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error over continuous predictions."""
    from sklearn.metrics import mean_absolute_error

    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination over continuous predictions."""
    from sklearn.metrics import r2_score

    return float(r2_score(y_true, y_pred))


def _pearsonr(y_true: Any, y_pred: Any) -> float:
    """Pearson correlation coefficient over continuous predictions."""
    from scipy.stats import pearsonr

    return float(pearsonr(y_true, y_pred).statistic)


def _spearmanr(y_true: Any, y_pred: Any) -> float:
    """Spearman rank correlation coefficient over continuous predictions."""
    from scipy.stats import spearmanr

    return float(spearmanr(y_true, y_pred).statistic)


def _build_registry(
    raw: dict[str, tuple[Callable[..., float], tuple[str, ...]]],
) -> dict[str, tuple[Callable[..., float], tuple[str, ...]]]:
    """Validate registry-construction invariants and return the table.

    Args:
        raw: Mapping of canonical metric name to ``(callable, aliases)``.

    Returns:
        A defensive copy of ``raw`` with every invariant verified.

    Raises:
        ValueError: If a canonical name is duplicated, if an alias equals the
            canonical name of a different entry, or if an alias is registered
            under two entries.
    """
    canonicals = list(raw)
    if len(set(canonicals)) != len(canonicals):
        raise ValueError("Canonical metric names must be unique.")
    alias_owners: dict[str, str] = {}
    for canonical, (_fn, aliases) in raw.items():
        for alias in aliases:
            if alias in raw and alias != canonical:
                raise ValueError(
                    f"Alias '{alias}' of '{canonical}' collides with the canonical "
                    f"name of another registry entry."
                )
            owner = alias_owners.setdefault(alias, canonical)
            if owner != canonical:
                raise ValueError(
                    f"Alias '{alias}' is registered under both '{owner}' and "
                    f"'{canonical}'; aliases must be unique across entries."
                )
    return dict(raw)


# Alias table: the eval_-prefixed form of every canonical name (the HF
# Trainer output spelling, e.g. eval_accuracy, eval_AUROC, eval_spearmanr)
# PLUS the historical spellings eval_auroc / eval_auprc / eval_spearman_r /
# eval_pearson_r. Recognition only - these are never emitted.
_RAW_REGISTRY: dict[str, tuple[Callable[..., float], tuple[str, ...]]] = {
    "accuracy": (_accuracy, ("eval_accuracy",)),
    "precision": (_precision, ("eval_precision",)),
    "recall": (_recall, ("eval_recall",)),
    "f1": (_f1, ("eval_f1",)),
    "f1_micro": (_f1_micro, ("eval_f1_micro",)),
    "f1_weighted": (_f1_weighted, ("eval_f1_weighted",)),
    "f1_samples": (_f1_samples, ("eval_f1_samples",)),
    "precision_micro": (_precision_micro, ("eval_precision_micro",)),
    "precision_weighted": (_precision_weighted, ("eval_precision_weighted",)),
    "precision_samples": (_precision_samples, ("eval_precision_samples",)),
    "recall_micro": (_recall_micro, ("eval_recall_micro",)),
    "recall_weighted": (_recall_weighted, ("eval_recall_weighted",)),
    "recall_samples": (_recall_samples, ("eval_recall_samples",)),
    "mcc": (_mcc, ("eval_mcc",)),
    "matthews_correlation": (_matthews_correlation, ("eval_matthews_correlation",)),
    "AUROC": (_auroc, ("eval_AUROC", "eval_auroc")),
    "AUPRC": (_auprc, ("eval_AUPRC", "eval_auprc")),
    "AUROC_ovr": (_auroc_ovr, ("eval_AUROC_ovr",)),
    "AUROC_ovo": (_auroc_ovo, ("eval_AUROC_ovo",)),
    "TPR": (_tpr, ("eval_TPR",)),
    "TNR": (_tnr, ("eval_TNR",)),
    "FPR": (_fpr, ("eval_FPR",)),
    "FNR": (_fnr, ("eval_FNR",)),
    "mse": (_mse, ("eval_mse",)),
    "mae": (_mae, ("eval_mae",)),
    "r2": (_r2, ("eval_r2",)),
    "pearsonr": (_pearsonr, ("eval_pearsonr", "eval_pearson_r")),
    "spearmanr": (_spearmanr, ("eval_spearmanr", "eval_spearman_r")),
}

METRIC_REGISTRY: MappingProxyType[str, tuple[Callable[..., float], tuple[str, ...]]] = (
    MappingProxyType(_build_registry(_RAW_REGISTRY))
)

_ALIAS_TO_CANONICAL: MappingProxyType[str, str] = MappingProxyType({
    alias: canonical for canonical, (_fn, aliases) in METRIC_REGISTRY.items() for alias in aliases
})

# Documented non-metric payload keys that metrics compute paths may attach to
# an emission alongside canonical metric names (plot data).
_PAYLOAD_KEYS: frozenset[str] = frozenset({"curve", "scatter"})


def resolve(name: str) -> Callable[..., float]:
    """Resolve a metric name (canonical or alias) to its metric callable.

    Args:
        name: Canonical metric name or registered alias (exact string,
            case-sensitive matching).

    Returns:
        The canonical metric callable with signature ``(y_true, y_pred) -> float``.

    Raises:
        ValueError: If the name is neither a canonical name nor a registered
            alias.
    """
    return METRIC_REGISTRY[canonical_name(name)][0]


def canonical_name(name: str) -> str:
    """Map a metric name or alias to its canonical spelling.

    Args:
        name: Canonical metric name or registered alias (exact string,
            case-sensitive matching).

    Returns:
        The canonical metric name.

    Raises:
        ValueError: If the name is neither a canonical name nor a registered
            alias.
    """
    if name in METRIC_REGISTRY:
        return name
    if name in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[name]
    raise ValueError(
        f"Unknown metric name: '{name}'. Valid canonical names: {sorted(METRIC_REGISTRY)}"
    )


def registered_names() -> tuple[str, ...]:
    """Return the sorted canonical metric names.

    Returns:
        Tuple of all canonical names in sorted order (the stable documented
        order); aliases are never included.
    """
    return tuple(sorted(METRIC_REGISTRY))


def validate_emission(keys: Any) -> None:
    """Validate that emitted metric keys are canonical registry names.

    Args:
        keys: Iterable of keys a metrics compute path is about to return.

    Raises:
        ValueError: If any key is neither a registered canonical name nor one
            of the documented non-metric payload keys (``curve``,
            ``scatter``). Aliases are NOT valid emission keys - emitting a
            historical alias is the exact drift this registry prevents.
    """
    valid = set(METRIC_REGISTRY) | set(_PAYLOAD_KEYS)
    offending = sorted(str(key) for key in keys if key not in valid)
    if offending:
        raise ValueError(
            f"Unregistered metric key(s) in emission: {offending}. "
            f"Emit canonical registry names only (aliases are recognized for "
            f"resolution but never emitted). Valid canonical names: {registered_names()}"
        )
