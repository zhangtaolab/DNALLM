"""PEFT target-module presets, dry-run validator, and ratio-guard tests (PEFT-02).

Network-free fast lane: the presets table is read from the packaged YAML via
importlib.resources, module trees are tiny in-memory torch modules, and the
trainable-ratio guard is exercised against fakes with controllable
requires_grad tensors. Real-backbone acceptance lives in the slow lane
(``tests/finetune/test_trainer_real_model.py``).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from dnallm.finetune import trainer as trainer_module
from dnallm.finetune.trainer import (
    _guard_trainable_ratio,
    _load_peft_presets,
    _peft_dry_run_report,
    _resolve_peft_preset,
)
from dnallm.models.modeling_auto import PRETRAIN_MODEL_MAPS


class FakeTree(torch.nn.Module):
    """Tiny module tree with controllable leaf names.

    ``leaves`` maps a dotted module path suffix to a child name, e.g.
    ``{"encoder.layer.0.attention.self": ["query", "key"]}`` builds
    encoder.layer.0.attention.self.query and .key Leaf modules.
    """

    def __init__(self, leaves):
        super().__init__()
        for path, names in leaves.items():
            node = self
            for part in path.split("."):
                if not hasattr(node, part):
                    setattr(node, part, torch.nn.Module())
                node = getattr(node, part)
            for name in names:
                setattr(node, name, torch.nn.Linear(4, 4))


def fake_config_model(name_path=None, model_type=None):
    """Minimal model stand-in carrying a config with the given identity."""
    model = FakeTree({"backbone.layer.0": ["query", "key", "value"]})
    model.config = SimpleNamespace(_name_or_path=name_path, model_type=model_type)
    return model


class TestPeftPresets:
    """Presets-table regression: the packaged YAML must stay complete and sane."""

    def test_presets_load_via_importlib_resources(self):
        """The table loads from the package (wheel-safe, never CWD-relative)."""
        families = _load_peft_presets()
        assert isinstance(families, dict)
        assert len(families) > 0

    def test_every_pretrain_model_maps_family_has_a_row(self):
        """The family-key set equals PRETRAIN_MODEL_MAPS' keys exactly."""
        families = _load_peft_presets()
        assert set(families) == set(PRETRAIN_MODEL_MAPS)

    def test_every_row_has_nonempty_target_lists(self):
        """No row ships an empty target_modules list (empty-scan edge)."""
        for family, row in _load_peft_presets().items():
            assert row["lora_target_modules"], family
            assert row["ia3_target_modules"], family

    def test_feedforward_modules_subset_of_ia3_targets(self):
        """peft requires feedforward_modules ⊆ target_modules for IA³."""
        for family, row in _load_peft_presets().items():
            ff = row.get("feedforward_modules") or []
            assert set(ff) <= set(row["ia3_target_modules"]), family

    def test_bands_present_with_lo_le_hi(self):
        """Ratio bands are [lo, hi] pairs with lo <= hi."""
        for family, row in _load_peft_presets().items():
            for key in ("ia3_ratio_band", "lora_ratio_band"):
                band = row[key]
                assert isinstance(band, list), (family, key)
                assert len(band) == 2, (family, key)
                assert band[0] <= band[1], (family, key)

    def test_lora_r_at_least_one_everywhere(self):
        """Recommended ranks are positive integers."""
        for family, row in _load_peft_presets().items():
            assert isinstance(row["lora_r"], int), family
            assert row["lora_r"] >= 1, family

    def test_benchmark_family_anchors_measured(self):
        """The two acceptance families carry the empirically-derived rows."""
        families = _load_peft_presets()
        assert families["Plant DNABERT"]["ia3_target_modules"] == [
            "key",
            "value",
            "intermediate.dense",
        ]
        assert families["Plant DNAMamba"]["ia3_target_modules"] == [
            "in_proj",
            "out_proj",
            "x_proj",
            "dt_proj",
        ]
        # out_proj excluded from LoRA on mamba (peft rejects it)
        assert "out_proj" not in families["Plant DNAMamba"]["lora_target_modules"]

    def test_missing_resource_fails_loud_at_load(self):
        """A missing packaged file raises a matchable error at load time."""
        with patch(
            "importlib.resources.files",
            side_effect=FileNotFoundError("missing presets file"),
        ):
            trainer_module._PEFT_PRESET_CACHE = None
            try:
                with pytest.raises(ValueError, match="Failed to load the packaged PEFT presets"):
                    _load_peft_presets()
            finally:
                trainer_module._PEFT_PRESET_CACHE = None

    def test_malformed_row_fails_loud_at_load(self):
        """Corrupted tables are rejected at load time with the exact defect."""

        class FakeResource:
            def __init__(self, content):
                self._content = content

            def open(self, *args, **kwargs):
                import io

                return io.StringIO(self._content)

            def joinpath(self, _path):
                return self

        good_tail = (
            "    lora_r: 8\n"
            "    ia3_ratio_band: [1.0e-4, 1.0e-2]\n"
            "    lora_ratio_band: [1.0e-4, 1.0e-2]\n"
        )
        cases = {
            "missing-families": "not_families: {}\n",
            "empty-targets": (
                "families:\n  BadFamily:\n    match_names: [bad]\n    model_types: []\n"
                "    lora_target_modules: []\n    ia3_target_modules: [key]\n"
                "    feedforward_modules: []\n" + good_tail
            ),
            "ff-not-subset": (
                "families:\n  BadFamily:\n    match_names: [bad]\n    model_types: []\n"
                "    lora_target_modules: [q]\n    ia3_target_modules: [key]\n"
                "    feedforward_modules: [not_a_target]\n" + good_tail
            ),
            "inverted-band": (
                "families:\n  BadFamily:\n    match_names: [bad]\n    model_types: []\n"
                "    lora_target_modules: [q]\n    ia3_target_modules: [key]\n"
                "    feedforward_modules: []\n"
                "    lora_r: 8\n"
                "    ia3_ratio_band: [1.0e-2, 1.0e-4]\n"
                "    lora_ratio_band: [1.0e-4, 1.0e-2]\n"
            ),
            "empty-match-names": (
                "families:\n  BadFamily:\n    match_names: []\n    model_types: []\n"
                "    lora_target_modules: [q]\n    ia3_target_modules: [key]\n"
                "    feedforward_modules: []\n" + good_tail
            ),
            "nonstring-match-names": (
                "families:\n  BadFamily:\n    match_names: [bad, 3]\n    model_types: []\n"
                "    lora_target_modules: [q]\n    ia3_target_modules: [key]\n"
                "    feedforward_modules: []\n" + good_tail
            ),
            "string-band": (
                "families:\n  BadFamily:\n    match_names: [bad]\n    model_types: []\n"
                "    lora_target_modules: [q]\n    ia3_target_modules: [key]\n"
                "    feedforward_modules: []\n"
                "    lora_r: 8\n"
                "    ia3_ratio_band: ['1.0e-4', '1.0e-2']\n"
                "    lora_ratio_band: [1.0e-4, 1.0e-2]\n"
            ),
        }
        expected = {
            "missing-families": "'families' mapping is missing or empty",
            "empty-targets": "'lora_target_modules' must be a non-empty list",
            "ff-not-subset": "feedforward_modules must be a subset of ia3_target_modules",
            "inverted-band": r"\[lo, hi\] pair with lo <= hi",
            "empty-match-names": "'match_names' must be a non-empty list of non-empty strings",
            "nonstring-match-names": "'match_names' must be a non-empty list of non-empty strings",
            "string-band": r"'ia3_ratio_band' must be a \[lo, hi\] pair of numbers",
        }
        for case, content in cases.items():
            with patch("importlib.resources.files", return_value=FakeResource(content)):
                trainer_module._PEFT_PRESET_CACHE = None
                try:
                    with pytest.raises(ValueError, match=expected[case]):
                        _load_peft_presets()
                finally:
                    trainer_module._PEFT_PRESET_CACHE = None


class TestPresetResolution:
    """Family resolution: name markers first, then config.model_type."""

    def test_name_marker_resolves_family(self):
        """A load path containing a family marker resolves to that family."""
        model = fake_config_model(
            name_path="/cache/hub/models/zhangtaolab/plant-dnamamba-BPE-open_chromatin",
            model_type="mamba",
        )
        family, _row, how = _resolve_peft_preset(model)
        assert family == "Plant DNAMamba"
        assert "name marker" in how

    def test_longest_marker_wins_over_shorter_family_markers(self):
        """dnabert-2 beats the shorter dnabert-style markers."""
        model = fake_config_model(name_path="zhihan1996/DNABERT-2-117M", model_type=None)
        assert _resolve_peft_preset(model)[0] == "DNABERT-2"

        model_bert = fake_config_model(name_path="zhihan1996/DNA_bert_6", model_type=None)
        assert _resolve_peft_preset(model_bert)[0] == "DNABERT"

    def test_model_type_fallback_resolves_architecture(self):
        """Without a name marker, config.model_type picks the architecture row."""
        model = fake_config_model(name_path=None, model_type="gemma")
        family, _row, how = _resolve_peft_preset(model)
        assert family == "Plant DNAGemma"
        assert "model_type" in how

    def test_unknown_backbone_raises_matchable_error(self):
        """No marker and no model_type row -> loud ValueError, no guessing."""
        model = fake_config_model(name_path="someone/new-dna-model", model_type="brandnew")
        with pytest.raises(ValueError, match="No PEFT target-module preset found"):
            _resolve_peft_preset(model)
        with pytest.raises(ValueError, match="target_modules explicitly"):
            _resolve_peft_preset(model)


class TestPeftDryRun:
    """Dry-run validator: match report, zero-match error (D-03)."""

    def test_matching_targets_reported(self, capsys):
        """Matched module names and count appear in the [Info] report."""
        tree = FakeTree({
            "encoder.layer.0.attention.self": ["query", "key", "value"],
            "encoder.layer.1.attention.self": ["query", "key", "value"],
        })
        matched = _peft_dry_run_report(tree, ["query"])
        assert len(matched) == 2
        out = capsys.readouterr().out
        assert "[Info] PEFT dry run" in out
        assert "2 modules matched" in out
        assert "encoder.layer.0.attention.self.query" in out

    def test_suffix_matching_mirrors_peft_rule(self):
        """Exact-name and dotted-suffix matches both count."""
        tree = FakeTree({"a.b": ["query"], "c": ["dense"]})
        matched = _peft_dry_run_report(tree, ["query", "c.dense", "c"])
        assert set(matched) == {"a.b.query", "c.dense", "c"}

    def test_zero_match_raises_matchable_error(self, capsys):
        """Wrong names for the backbone error instead of silently freezing."""
        tree = FakeTree({"encoder.layer.0.attention.self": ["query", "key", "value"]})
        with pytest.raises(ValueError, match="matched 0 of"):
            _peft_dry_run_report(tree, ["q_proj", "v_proj"])
        with pytest.raises(ValueError, match="silently freeze"):
            _peft_dry_run_report(tree, ["q_proj"])


class TestTrainableRatioGuard:
    """D-04 trainable-ratio guard computed from requires_grad tensors."""

    def _ratio_model(self, trainable, total):
        """Fake with an exact trainable/total split (numel as a method)."""

        class _Param:
            def __init__(self, count, trainable):
                self._count = count
                self.requires_grad = trainable

            def numel(self):
                return self._count

        class Fake:
            def parameters(self):
                return [
                    _Param(trainable, True),
                    _Param(total - trainable, False),
                ]

        return Fake()

    def test_in_band_passes(self):
        """A ratio inside the preset band passes and is returned."""
        model = self._ratio_model(trainable=10, total=10_000)  # 1e-3
        row = {"ia3_ratio_band": [1e-4, 1e-2], "lora_ratio_band": [1e-4, 1e-2]}
        trainable, total, ratio = _guard_trainable_ratio(model, "Fam", row, "ia3")
        assert (trainable, total) == (10, 10_000)
        assert ratio == pytest.approx(1e-3)

    def test_below_band_raises_d04_message(self):
        """Out-of-band raises naming the preset, band, and actual count."""
        model = self._ratio_model(trainable=1, total=10_000)  # 1e-4, below band
        row = {"ia3_ratio_band": [1e-3, 1e-2], "lora_ratio_band": [1e-3, 1e-2]}
        with pytest.raises(ValueError, match="PEFT preset 'Fam' attached 1/10000"):
            _guard_trainable_ratio(model, "Fam", row, "ia3")
        with pytest.raises(ValueError, match="silent module-skip"):
            _guard_trainable_ratio(model, "Fam", row, "ia3")
        with pytest.raises(ValueError, match=r"expected band \[1.00e-03, 1.00e-02\]"):
            _guard_trainable_ratio(model, "Fam", row, "ia3")

    def test_above_band_raises(self):
        """A suspiciously large ratio (nothing frozen) also fails."""
        model = self._ratio_model(trainable=9_000, total=10_000)
        row = {"ia3_ratio_band": [1e-4, 1e-2], "lora_ratio_band": [1e-4, 1e-2]}
        with pytest.raises(ValueError, match="PEFT preset 'Fam' attached 9000/10000"):
            _guard_trainable_ratio(model, "Fam", row, "lora")

    def test_user_targets_zero_trainable_raises(self):
        """Without a preset band, a fully frozen model still fails hard."""
        model = self._ratio_model(trainable=0, total=10_000)
        with pytest.raises(ValueError, match="0 trainable parameters"):
            _guard_trainable_ratio(model, None, None, "ia3")

    def test_user_targets_nonzero_trainable_passes(self):
        """User-supplied targets with a nonzero ratio pass without a band."""
        model = self._ratio_model(trainable=5, total=10_000)
        trainable, _, _ = _guard_trainable_ratio(model, None, None, "lora")
        assert trainable == 5
