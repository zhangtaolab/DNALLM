"""Tests for the metric registry contract module (METR-01, REV-02).

Covers name resolution (canonical + alias), construction invariants,
immutability of the registry surface, the import-light contract (AST scan),
per-metric invocation of every registered callable, and the emission gate.
"""

import ast
import math
from itertools import chain
from pathlib import Path

import numpy as np
import pytest

import dnallm.tasks.metric_registry as metric_registry_module
from dnallm.tasks.metric_registry import (
    METRIC_REGISTRY,
    _build_registry,
    canonical_name,
    registered_names,
    resolve,
    validate_emission,
)

# The frozen canonical contract: exactly the keys emitted anywhere in
# dnallm/tasks/metrics.py today (spelling anchor, PITFALLS #2).
EXPECTED_CANONICAL = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_micro",
    "f1_weighted",
    "f1_samples",
    "precision_micro",
    "precision_weighted",
    "precision_samples",
    "recall_micro",
    "recall_weighted",
    "recall_samples",
    "mcc",
    "matthews_correlation",
    "AUROC",
    "AUPRC",
    "AUROC_ovr",
    "AUROC_ovo",
    "TPR",
    "TNR",
    "FPR",
    "FNR",
    "mse",
    "mae",
    "r2",
    "pearsonr",
    "spearmanr",
}

# Minimal valid (y_true, y_pred) pairs per metric input family.
_BINARY_TRUE = np.array([1, 0, 1, 0, 1])
_BINARY_PRED = np.array([1, 0, 1, 1, 0])
BINARY_HARD = (_BINARY_TRUE, _BINARY_PRED)

_MULTICLASS_TRUE = np.array([0, 1, 2, 1, 0])
_MULTICLASS_PRED = np.array([0, 1, 1, 1, 0])
MULTICLASS_HARD = (_MULTICLASS_TRUE, _MULTICLASS_PRED)

_MULTICLASS_PROBS = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.2, 0.7],
    [0.3, 0.5, 0.2],
    [0.6, 0.3, 0.1],
])
MULTICLASS_SCORED = (_MULTICLASS_TRUE, _MULTICLASS_PROBS)

_MULTILABEL_TRUE = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
_MULTILABEL_PRED = np.array([[1, 0], [0, 1], [1, 0], [0, 0], [1, 1]])
MULTILABEL_HARD = (_MULTILABEL_TRUE, _MULTILABEL_PRED)

_MULTILABEL_SCORES = np.array([[0.8, 0.3], [0.2, 0.7], [0.6, 0.9], [0.4, 0.2], [0.9, 0.1]])
MULTILABEL_SCORED = (_MULTILABEL_TRUE, _MULTILABEL_SCORES)

_BINARY_PROBS_1D = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
BINARY_SCORED_1D = (_BINARY_TRUE, _BINARY_PROBS_1D)

BINARY_PROBS_2D = (_BINARY_TRUE, np.column_stack([1 - _BINARY_PROBS_1D, _BINARY_PROBS_1D]))

_REG_TRUE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_REG_PRED = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
REGRESSION = (_REG_TRUE, _REG_PRED)

METRIC_INPUTS = {
    "accuracy": BINARY_HARD,
    "precision": BINARY_HARD,
    "recall": BINARY_HARD,
    "f1": BINARY_HARD,
    "f1_micro": MULTICLASS_HARD,
    "f1_weighted": MULTICLASS_HARD,
    "f1_samples": MULTILABEL_HARD,
    "precision_micro": MULTICLASS_HARD,
    "precision_weighted": MULTICLASS_HARD,
    "precision_samples": MULTILABEL_HARD,
    "recall_micro": MULTICLASS_HARD,
    "recall_weighted": MULTICLASS_HARD,
    "recall_samples": MULTILABEL_HARD,
    "mcc": BINARY_HARD,
    "matthews_correlation": MULTICLASS_HARD,
    "AUROC": BINARY_SCORED_1D,
    "AUPRC": BINARY_SCORED_1D,
    "AUROC_ovr": MULTICLASS_SCORED,
    "AUROC_ovo": MULTICLASS_SCORED,
    "TPR": BINARY_HARD,
    "TNR": BINARY_HARD,
    "FPR": BINARY_HARD,
    "FNR": BINARY_HARD,
    "mse": REGRESSION,
    "mae": REGRESSION,
    "r2": REGRESSION,
    "pearsonr": REGRESSION,
    "spearmanr": REGRESSION,
}

ALL_ALIASES = list(chain.from_iterable(aliases for _, aliases in METRIC_REGISTRY.values()))


def _stub_metric(y_true, y_pred):
    """Trivial callable for feeding deliberately invalid tables to the builder.

    The argument shapes are irrelevant to the builder's invariant checks.
    """
    return 0.0


class TestResolve:
    """Test resolve()."""

    def test_canonical_name_resolves_to_callable(self):
        assert callable(resolve("AUROC"))

    def test_alias_resolves_to_same_object_as_canonical(self):
        assert resolve("eval_auroc") is resolve("AUROC")
        assert resolve("eval_AUROC") is resolve("AUROC")

    def test_every_eval_prefixed_alias_resolves(self):
        for name in METRIC_REGISTRY:
            assert resolve("eval_" + name) is METRIC_REGISTRY[name][0]

    def test_historical_aliases_resolve_to_their_targets(self):
        assert resolve("eval_auprc") is resolve("AUPRC")
        assert resolve("eval_spearman_r") is resolve("spearmanr")
        assert resolve("eval_pearson_r") is resolve("pearsonr")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unknown metric name"):
            resolve("")

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown metric name: 'bogus'"):
            resolve("bogus")

    def test_unregistered_casing_raises(self):
        with pytest.raises(ValueError, match="Unknown metric name"):
            resolve("Eval_Auroc")

    def test_error_lists_valid_names(self):
        with pytest.raises(ValueError, match=r"accuracy.*spearmanr"):
            resolve("nope")


class TestCanonicalName:
    """Test canonical_name()."""

    def test_historical_alias_maps(self):
        assert canonical_name("eval_spearman_r") == "spearmanr"
        assert canonical_name("eval_pearson_r") == "pearsonr"
        assert canonical_name("eval_auprc") == "AUPRC"
        assert canonical_name("eval_auroc") == "AUROC"

    def test_eval_prefixed_alias_preserves_canonical_case(self):
        assert canonical_name("eval_AUROC") == "AUROC"
        assert canonical_name("eval_accuracy") == "accuracy"

    def test_canonical_name_is_identity(self):
        assert canonical_name("accuracy") == "accuracy"
        assert canonical_name("AUROC") == "AUROC"

    def test_unregistered_casing_raises(self):
        with pytest.raises(ValueError, match="Unknown metric name: 'Eval_Auroc'"):
            canonical_name("Eval_Auroc")


class TestInvariants:
    """Construction invariants of the registry builder."""

    def test_real_registry_satisfies_all_invariants(self):
        all_aliases = list(chain.from_iterable(a for _, a in METRIC_REGISTRY.values()))
        # (b) no alias duplicated across entries
        assert len(all_aliases) == len(set(all_aliases))
        # (a) no alias equals any canonical name
        assert not set(all_aliases) & set(METRIC_REGISTRY)
        # (c) canonical names are unique
        assert len(set(METRIC_REGISTRY)) == len(METRIC_REGISTRY)

    def test_builder_rejects_alias_colliding_with_other_canonical(self):
        colliding = {
            "a_metric": (_stub_metric, ("other_metric",)),
            "other_metric": (_stub_metric, ()),
        }
        with pytest.raises(ValueError, match="collides with the canonical"):
            _build_registry(colliding)

    def test_builder_rejects_alias_duplicated_across_entries(self):
        duplicated = {
            "a_metric": (_stub_metric, ("shared_alias",)),
            "b_metric": (_stub_metric, ("shared_alias",)),
        }
        with pytest.raises(ValueError, match="registered under both"):
            _build_registry(duplicated)

    def test_builder_accepts_a_valid_table(self):
        valid = {"a_metric": (_stub_metric, ("eval_a_metric",))}
        built = _build_registry(valid)
        assert built == valid


class TestImmutability:
    """The registry surface is frozen; resolution is side-effect-free."""

    def test_repeated_resolution_leaves_registry_unchanged(self):
        before = {name: (fn, aliases) for name, (fn, aliases) in METRIC_REGISTRY.items()}
        for name in chain(METRIC_REGISTRY, ALL_ALIASES):
            resolve(name)
            canonical_name(name)
        after = {name: (fn, aliases) for name, (fn, aliases) in METRIC_REGISTRY.items()}
        assert before == after

    def test_registry_mapping_rejects_item_assignment(self):
        with pytest.raises(TypeError):
            METRIC_REGISTRY["bogus_metric"] = (_stub_metric, ())  # type: ignore[index]

    def test_no_public_mutation_api_is_exported(self):
        assert set(metric_registry_module.__all__) == {
            "METRIC_REGISTRY",
            "canonical_name",
            "registered_names",
            "resolve",
            "validate_emission",
        }
        exported = [name for name in dir(metric_registry_module) if not name.startswith("_")]
        banned_exact = {"register", "append", "add", "update", "remove", "pop", "setdefault"}
        banned_prefixes = ("register_", "append_", "add_", "update_", "remove_", "unregister")
        offenders = [
            name for name in exported if name in banned_exact or name.startswith(banned_prefixes)
        ]
        assert offenders == [], f"mutation-like export {offenders} is forbidden"


class TestRegisteredNames:
    """registered_names() returns the sorted canonical set."""

    def test_result_is_sorted(self):
        names = registered_names()
        assert names == tuple(sorted(names))

    def test_equals_registry_key_set(self):
        assert set(registered_names()) == set(METRIC_REGISTRY)

    def test_equals_expected_contract_set(self):
        assert set(registered_names()) == EXPECTED_CANONICAL

    def test_aliases_are_not_registered_names(self):
        assert "eval_auroc" not in registered_names()
        assert "eval_spearman_r" not in registered_names()


class TestImportLight:
    """No module-level torch/sklearn imports (dnallmmark CI requirement)."""

    def test_no_top_level_torch_or_sklearn_imports(self):
        source = Path(metric_registry_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        banned = {"torch", "sklearn"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported = {(node.module or "").split(".")[0]}
            else:
                continue
            offenders = imported & banned
            assert offenders == set(), f"module-level import of {offenders} is forbidden"


class TestMetricCallables:
    """Every registered callable computes a finite float (coverage driver)."""

    def test_input_table_covers_every_registered_name(self):
        assert set(METRIC_INPUTS) == set(METRIC_REGISTRY)

    @pytest.mark.parametrize("name", sorted(METRIC_INPUTS))
    def test_every_registered_callable_computes_a_finite_float(self, name):
        y_true, y_pred = METRIC_INPUTS[name]
        value = resolve(name)(y_true, y_pred)
        assert isinstance(value, float)
        assert math.isfinite(value)

    def test_auroc_binary_two_column_scores_use_positive_column(self):
        from_column = resolve("AUROC")(*BINARY_PROBS_2D)
        from_1d = resolve("AUROC")(*BINARY_SCORED_1D)
        assert from_column == pytest.approx(from_1d)

    def test_auroc_multiclass_matrix_uses_macro_ovr(self):
        value = resolve("AUROC")(*MULTICLASS_SCORED)
        assert 0.0 <= value <= 1.0

    def test_auroc_multilabel_indicator_macro_averages(self):
        value = resolve("AUROC")(*MULTILABEL_SCORED)
        assert 0.0 <= value <= 1.0

    def test_auprc_binary_two_column_scores_use_positive_column(self):
        from_column = resolve("AUPRC")(*BINARY_PROBS_2D)
        from_1d = resolve("AUPRC")(*BINARY_SCORED_1D)
        assert from_column == pytest.approx(from_1d)

    def test_auprc_multiclass_matrix_macro_averages(self):
        value = resolve("AUPRC")(*MULTICLASS_SCORED)
        assert 0.0 <= value <= 1.0

    def test_auprc_multilabel_indicator_macro_averages(self):
        value = resolve("AUPRC")(*MULTILABEL_SCORED)
        assert 0.0 <= value <= 1.0

    def test_auroc_ovr_binary_scores_fall_back_to_plain_score(self):
        assert resolve("AUROC_ovr")(*BINARY_SCORED_1D) == pytest.approx(
            resolve("AUROC")(*BINARY_SCORED_1D)
        )

    def test_auroc_ovo_binary_scores_fall_back_to_plain_score(self):
        assert resolve("AUROC_ovo")(*BINARY_SCORED_1D) == pytest.approx(
            resolve("AUROC")(*BINARY_SCORED_1D)
        )

    def test_tpr_matches_hand_computed_confusion_rates(self):
        # labels [1,0,1,0,1] vs preds [1,0,1,1,0]: tp=2 fn=1 fp=1 tn=1
        assert resolve("TPR")(*BINARY_HARD) == pytest.approx(2 / 3)
        assert resolve("FNR")(*BINARY_HARD) == pytest.approx(1 / 3)
        assert resolve("FPR")(*BINARY_HARD) == pytest.approx(1 / 2)
        assert resolve("TNR")(*BINARY_HARD) == pytest.approx(1 / 2)


class TestValidateEmission:
    """validate_emission(): the emission gate."""

    def test_all_canonical_names_pass(self):
        validate_emission(METRIC_REGISTRY)

    def test_payload_keys_pass(self):
        validate_emission(["accuracy", "curve"])
        validate_emission(["mse", "scatter"])

    def test_unregistered_key_raises(self):
        with pytest.raises(ValueError, match="bogus_metric"):
            validate_emission(["accuracy", "bogus_metric"])

    def test_alias_is_never_a_valid_emission_key(self):
        with pytest.raises(ValueError, match="eval_auroc"):
            validate_emission(["eval_auroc"])

    def test_empty_emission_passes(self):
        validate_emission([])
