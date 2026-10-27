"""Census-driven compat-rung tests (27-02): one ``Test*`` class per new rung.

Every rung here corresponds to a MEASURED failure class in the committed
census evidence (``tests/models/compat_census.json``, the 27-01 artifact):

- ``apply_chunking_to_forward`` gone from ``transformers.modeling_utils``
  under transformers 5.x (the remote ``modeling_bert.py`` import block of
  AIRI-Institute/gena-lm-bert-base and -bigbird-base-sparse, census
  ``import-fail`` records);
- the transformers 5 tied-weights bookkeeping (``all_tied_weights_keys``)
  missing on remote models whose ``__init__`` never calls ``post_init`` /
  ``init_weights`` (the bare ``Caduceus`` backbone of
  kuleshov-group/PlantCaduceus_l32, census load failure at
  ``_move_missing_keys_from_meta_to_device``).

The tests assert the rung contract shared with the eleven existing rungs:
absence-gated (never overwrite), idempotent (sentinel), probe-gated no-op on
transformers 4.x, and the closed-map discipline (no module-level
``__getattr__`` is ever installed on a patched transformers module).
"""

from __future__ import annotations

import types

import pytest
import torch
import transformers.cache_utils
import transformers.modeling_utils
import transformers.pytorch_utils
import transformers.utils.import_utils
from transformers.modeling_utils import PreTrainedModel

from dnallm.utils import transformers_compat
from dnallm.utils.transformers_compat import apply_patches


def _rung(name: str):
    """Return a new rung symbol, asserting it exists (the RED guard)."""
    rung = getattr(transformers_compat, name, None)
    assert rung is not None, f"rung symbol {name} must exist in transformers_compat"
    return rung


# ─────────────────────────────────────────────────────────────────────────────
# Rung 1: apply_chunking_to_forward re-attached on transformers.modeling_utils
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyChunkingToForward:
    """Contract: the 4.x ``modeling_utils`` chunking helper is restored on 5.x."""

    def test_name_resolves_on_live_modeling_utils(self):
        """After apply_patches, modeling_utils exposes a callable apply_chunking_to_forward."""
        assert callable(getattr(transformers.modeling_utils, "apply_chunking_to_forward", None))

    def test_restored_name_is_the_live_pytorch_utils_object(self):
        """The restore aliases the implementation transformers 5.x still ships.

        transformers 5.17 keeps ``apply_chunking_to_forward`` in
        ``pytorch_utils``; re-attaching THAT object (not a vendored copy)
        keeps semantics identical by construction.
        """
        restored = getattr(transformers.modeling_utils, "apply_chunking_to_forward", None)
        assert restored is not None
        assert restored is getattr(transformers.pytorch_utils, "apply_chunking_to_forward", None)

    def test_patch_is_idempotent(self):
        """A second rung call attaches nothing new and keeps the sentinel set."""
        patch = _rung("_patch_apply_chunking_to_forward")
        before = getattr(transformers.modeling_utils, "apply_chunking_to_forward", None)
        assert getattr(transformers.modeling_utils, "_dnallm_apply_chunking_patch", False) is True

        patch()

        assert (
            getattr(transformers.modeling_utils, "apply_chunking_to_forward", None) is before
        ), "repeat call must not rebind the attached name"

    def test_absence_gate_never_overwrites_an_existing_name(self):
        """A modeling_utils already exposing the name keeps its original object."""
        _rung("_patch_apply_chunking_to_forward")

        original = object()
        fake_modeling = types.SimpleNamespace(apply_chunking_to_forward=original)
        fake_pytorch = types.SimpleNamespace(apply_chunking_to_forward=object())

        attached = transformers_compat._attach_apply_chunking_to_forward(  # noqa: SLF001 - rung helper
            fake_modeling, fake_pytorch
        )
        assert attached is False
        assert fake_modeling.apply_chunking_to_forward is original

    def test_attach_prefers_the_live_pytorch_utils_object(self):
        """With the name absent on modeling_utils but present on pytorch_utils,
        the rung attaches the live pytorch_utils object (the alias shape)."""
        _rung("_patch_apply_chunking_to_forward")

        live = transformers.pytorch_utils.apply_chunking_to_forward
        fake_modeling = types.ModuleType("fake_modeling_utils")
        fake_pytorch = types.SimpleNamespace(apply_chunking_to_forward=live)

        transformers_compat._attach_apply_chunking_to_forward(  # noqa: SLF001 - rung helper
            fake_modeling, fake_pytorch
        )

        assert fake_modeling.apply_chunking_to_forward is live
        assert getattr(fake_modeling, "_dnallm_apply_chunking_patch", False) is True

    def test_attach_falls_back_to_the_vendored_v449_implementation(self):
        """When neither module exposes the name, the vendored v4.49.0
        implementation is attached so the rung survives a future removal
        from pytorch_utils as well."""
        _rung("_patch_apply_chunking_to_forward")

        fake_modeling = types.ModuleType("fake_modeling_utils")
        fake_pytorch = types.ModuleType("fake_pytorch_utils")

        transformers_compat._attach_apply_chunking_to_forward(  # noqa: SLF001 - rung helper
            fake_modeling, fake_pytorch
        )

        assert fake_modeling.apply_chunking_to_forward is transformers_compat._apply_chunking_to_forward  # noqa: SLF001 - rung helper

    def test_vendored_chunking_semantics_match_direct_application(self):
        """The vendored v4.49.0 chunker: chunked forward == direct forward,
        zero/negative chunk size passes through, arity mismatch raises."""
        _rung("_apply_chunking_to_forward")

        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        def double(x):
            return x * 2

        chunked = transformers_compat._apply_chunking_to_forward(  # noqa: SLF001 - rung helper
            double, 2, 0, tensor
        )
        assert torch.equal(chunked, double(tensor))

        passthrough = transformers_compat._apply_chunking_to_forward(  # noqa: SLF001 - rung helper
            double, 0, 0, tensor
        )
        assert torch.equal(passthrough, double(tensor))

        with pytest.raises(ValueError, match="arguments"):
            transformers_compat._apply_chunking_to_forward(  # noqa: SLF001 - rung helper
                double, 2, 0, tensor, tensor.clone()
            )

        with pytest.raises(ValueError, match="multiple of the chunk"):
            transformers_compat._apply_chunking_to_forward(  # noqa: SLF001 - rung helper
                double, 5, 0, tensor
            )

    def test_noop_when_neither_module_imports(self, monkeypatch):
        """The rung returns quietly when the target modules are absent
        (older transformers or a renamed module), touching nothing."""
        patch = _rung("_patch_apply_chunking_to_forward")

        import builtins

        real_import = builtins.__import__

        def broken_import(name, *args, **kwargs):
            if name.startswith("transformers.modeling_utils") or name.startswith(
                "transformers.pytorch_utils"
            ):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        patch()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Rung 2: tied-weights bookkeeping materialized at the from_pretrained boundary
# ─────────────────────────────────────────────────────────────────────────────


class _FakeBookkeepingModel:
    """Receiver shaped like a remote model that skipped ``post_init``.

    Carries the 5.x computation surface (``get_expanded_tied_weights_keys``,
    ``named_children``) but no ``all_tied_weights_keys`` attribute, exactly
    like the bare remote ``Caduceus`` backbone of PlantCaduceus_l32.
    """

    def __init__(self, own_mapping=None, child_mapping=None):
        self._own = own_mapping if own_mapping is not None else {"head.weight": "backbone.weight"}
        self._child = child_mapping
        self.calls = []

    def get_expanded_tied_weights_keys(self, all_submodels=False):
        self.calls.append(("expanded", all_submodels))
        return dict(self._own)

    def named_children(self):
        if self._child is None:
            return iter(())
        child = types.SimpleNamespace(all_tied_weights_keys=dict(self._child))
        return iter([("backbone", child)])


class TestTiedWeightsBookkeepingOnLoad:
    """Contract: remote __init__ paths that skip post_init still get the 5.x
    tied-weights bookkeeping before from_pretrained's loaders read it."""

    def test_rung_is_installed_on_the_live_class(self):
        """apply_patches wraps PreTrainedModel._finalize_model_loading with the
        materializing sentinel set."""
        assert getattr(PreTrainedModel, "_dnallm_tied_bookkeeping_patch", False) is True
        assert PreTrainedModel.__dict__.get("_dnallm_tied_bookkeeping_patch") is True

    def test_materialize_sets_the_post_init_equivalent_mapping(self):
        """A receiver missing the attribute gains exactly what post_init would
        compute: its own expanded map merged with each child's map."""
        _rung("_materialize_tied_weights_bookkeeping")

        model = _FakeBookkeepingModel(
            own_mapping={"head.weight": "backbone.weight"},
            child_mapping={"embed.weight": "tok.weight"},
        )

        transformers_compat._materialize_tied_weights_bookkeeping(model)  # noqa: SLF001 - rung helper

        assert model.all_tied_weights_keys == {
            "head.weight": "backbone.weight",
            "backbone.embed.weight": "backbone.tok.weight",
        }
        # post_init computes the own map with all_submodels=False:
        assert ("expanded", False) in model.calls

    def test_materialize_never_overwrites_an_existing_mapping(self):
        """A receiver that already carries the attribute keeps it untouched."""
        _rung("_materialize_tied_weights_bookkeeping")

        model = _FakeBookkeepingModel()
        original = {"already": "set"}
        model.all_tied_weights_keys = original

        transformers_compat._materialize_tied_weights_bookkeeping(model)  # noqa: SLF001 - rung helper

        assert model.all_tied_weights_keys is original

    def test_wrapped_finalize_materializes_then_delegates(self, monkeypatch):
        """The installed wrapper materializes the bookkeeping for a missing
        receiver, then delegates to the captured original with identical
        arguments."""
        _rung("_patch_tied_weights_bookkeeping_on_load")
        _rung("_materialize_tied_weights_bookkeeping")

        recorded = {}

        def fake_original(cls, model, load_config, loading_info):
            recorded["cls"] = cls
            recorded["model"] = model
            recorded["args"] = (load_config, loading_info)
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        model = _FakeBookkeepingModel()
        result = PreTrainedModel._finalize_model_loading(model, "load-config", "loading-info")

        assert result == "delegated"
        assert recorded["cls"] is PreTrainedModel
        assert recorded["model"] is model
        assert recorded["args"] == ("load-config", "loading-info")
        assert hasattr(model, "all_tied_weights_keys")

    def test_wrapped_finalize_leaves_compliant_models_untouched(self, monkeypatch):
        """A receiver already carrying the attribute is passed through without
        recomputation (the materializer's absence gate, via the wrapper)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")

        def fake_original(cls, model, load_config, loading_info):
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        model = _FakeBookkeepingModel()
        mapping = {"kept": "as-is"}
        model.all_tied_weights_keys = mapping

        PreTrainedModel._finalize_model_loading(model, None, None)

        assert model.all_tied_weights_keys is mapping
        assert model.calls == [], "no recomputation for a compliant receiver"

    def test_patch_is_idempotent(self):
        """A second rung call does not re-wrap the classmethod."""
        patch = _rung("_patch_tied_weights_bookkeeping_on_load")

        wrapper_before = PreTrainedModel.__dict__["_finalize_model_loading"]
        original_before = transformers_compat._tied_bookkeeping_original_finalize

        patch()

        assert PreTrainedModel.__dict__["_finalize_model_loading"] is wrapper_before
        assert transformers_compat._tied_bookkeeping_original_finalize is original_before

    def test_noop_when_post_init_lacks_the_bookkeeping(self, monkeypatch):
        """On transformers 4.x (probe False) the rung wraps nothing."""
        patch = _rung("_patch_tied_weights_bookkeeping_on_load")

        class _FakeModelClass:
            _finalize_model_loading = classmethod(lambda cls, *a: a)

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _FakeModelClass)
        monkeypatch.setattr(
            transformers_compat, "_post_init_computes_tied_weights_keys", lambda: False
        )

        before = _FakeModelClass.__dict__["_finalize_model_loading"]
        patch()

        assert _FakeModelClass.__dict__["_finalize_model_loading"] is before
        assert getattr(_FakeModelClass, "_dnallm_tied_bookkeeping_patch", False) is False

    def test_noop_when_finalize_model_loading_absent(self, monkeypatch):
        """A transformers without ``_finalize_model_loading`` is left untouched."""
        patch = _rung("_patch_tied_weights_bookkeeping_on_load")

        class _BareClass:
            pass

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _BareClass)
        monkeypatch.setattr(
            transformers_compat, "_post_init_computes_tied_weights_keys", lambda: True
        )

        patch()

        assert getattr(_BareClass, "_dnallm_tied_bookkeeping_patch", False) is False


# ─────────────────────────────────────────────────────────────────────────────
# Closed-map discipline (prohibition T-27-04)
# ─────────────────────────────────────────────────────────────────────────────


class TestClosedMapDiscipline:
    """No patched transformers module ever gains a module-level __getattr__."""

    def test_no_module_level_getattr_on_patched_modules(self):
        modules = [
            transformers.modeling_utils,
            transformers.pytorch_utils,
            transformers.cache_utils,
            transformers.utils.import_utils,
        ]
        for module in modules:
            assert "__getattr__" not in vars(module), (
                f"{module.__name__} must never carry a module-level __getattr__ "
                "(closed per-name map discipline, T-27-04)"
            )

    def test_apply_patches_stays_idempotent_with_new_rungs(self):
        """The extended ladder keeps apply_patches idempotent end to end."""
        finalize_before = PreTrainedModel.__dict__["_finalize_model_loading"]
        chunking_before = getattr(transformers.modeling_utils, "apply_chunking_to_forward", None)

        apply_patches()

        assert PreTrainedModel.__dict__["_finalize_model_loading"] is finalize_before
        assert (
            getattr(transformers.modeling_utils, "apply_chunking_to_forward", None)
            is chunking_before
        )
