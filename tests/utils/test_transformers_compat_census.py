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
import transformers.modeling_rope_utils
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

        assert getattr(transformers.modeling_utils, "apply_chunking_to_forward", None) is before, (
            "repeat call must not rebind the attached name"
        )

    def test_absence_gate_never_overwrites_an_existing_name(self):
        """A modeling_utils already exposing the name keeps its original object."""
        _rung("_patch_apply_chunking_to_forward")

        original = object()
        fake_modeling = types.SimpleNamespace(apply_chunking_to_forward=original)
        fake_pytorch = types.SimpleNamespace(apply_chunking_to_forward=object())

        attached = transformers_compat._attach_apply_chunking_to_forward(
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

        transformers_compat._attach_apply_chunking_to_forward(fake_modeling, fake_pytorch)

        assert fake_modeling.apply_chunking_to_forward is live
        assert getattr(fake_modeling, "_dnallm_apply_chunking_patch", False) is True

    def test_attach_falls_back_to_the_vendored_v449_implementation(self):
        """When neither module exposes the name, the vendored v4.49.0
        implementation is attached so the rung survives a future removal
        from pytorch_utils as well."""
        _rung("_patch_apply_chunking_to_forward")

        fake_modeling = types.ModuleType("fake_modeling_utils")
        fake_pytorch = types.ModuleType("fake_pytorch_utils")

        transformers_compat._attach_apply_chunking_to_forward(fake_modeling, fake_pytorch)

        assert (
            fake_modeling.apply_chunking_to_forward
            is transformers_compat._apply_chunking_to_forward
        )

    def test_vendored_chunking_semantics_match_direct_application(self):
        """The vendored v4.49.0 chunker: chunked forward == direct forward,
        zero/negative chunk size passes through, arity mismatch raises."""
        _rung("_apply_chunking_to_forward")

        tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        def double(x):
            return x * 2

        chunked = transformers_compat._apply_chunking_to_forward(double, 2, 0, tensor)
        assert torch.equal(chunked, double(tensor))

        passthrough = transformers_compat._apply_chunking_to_forward(double, 0, 0, tensor)
        assert torch.equal(passthrough, double(tensor))

        ragged = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        with pytest.raises(ValueError, match="arguments"):
            transformers_compat._apply_chunking_to_forward(double, 2, 0, tensor, tensor.clone())

        with pytest.raises(ValueError, match="multiple of the chunk"):
            transformers_compat._apply_chunking_to_forward(double, 2, 0, ragged)

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

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        """Native-shaped 5.x tie_weights (the bridge must leave it alone)."""


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

        transformers_compat._materialize_tied_weights_bookkeeping(model)

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

        transformers_compat._materialize_tied_weights_bookkeeping(model)

        assert model.all_tied_weights_keys is original

    def test_wrapped_finalize_materializes_then_delegates(self, monkeypatch):
        """The installed wrapper materializes the bookkeeping for a missing
        receiver, then delegates to the captured original with identical
        arguments (transformers 5.17 ships a static method: the model is the
        first argument)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")
        _rung("_materialize_tied_weights_bookkeeping")

        recorded = {}

        def fake_original(model, load_config, loading_info):
            recorded["model"] = model
            recorded["args"] = (load_config, loading_info)
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        model = _FakeBookkeepingModel()
        result = PreTrainedModel._finalize_model_loading(model, "load-config", "loading-info")

        assert result == "delegated"
        assert recorded["model"] is model
        assert recorded["args"] == ("load-config", "loading-info")
        assert hasattr(model, "all_tied_weights_keys")

    def test_wrapped_finalize_leaves_compliant_models_untouched(self, monkeypatch):
        """A receiver already carrying the attribute is passed through without
        recomputation (the materializer's absence gate, via the wrapper)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")

        def fake_original(model, load_config, loading_info):
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

    def test_wrapped_finalize_bridges_legacy_tie_weights_arity(self, monkeypatch):
        """A receiver with a zero-arg 4.x tie_weights override gains the
        per-instance arity bridge at the load boundary (the same skipped
        post_init entry leaves both 5.x protocol pieces missing -- measured
        live on the bare remote Caduceus backbone of PlantCaduceus_l32:
        all_tied_weights_keys materialization gets past meta-device move,
        then from_pretrained's model.tie_weights(missing_keys=...) call
        dies with TypeError on the un-bridged override)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")

        def fake_original(model, load_config, loading_info):
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        class _LegacyTieWeightsModel(_FakeBookkeepingModel):
            def tie_weights(self):
                """4.x zero-arg override (the Caduceus shape)."""

        model = _LegacyTieWeightsModel()
        PreTrainedModel._finalize_model_loading(model, None, None)

        # The 5.x post-load call contract now works through the bridge:
        model.tie_weights(missing_keys=set(), recompute_mapping=False)
        # And the native-shaped receiver from the fixture stays untouched
        # (the bridge self-guards on the signature -- no instance override):
        compliant = _FakeBookkeepingModel()
        PreTrainedModel._finalize_model_loading(compliant, None, None)
        assert compliant.__dict__.get("tie_weights") is None

    def test_kwargs_only_receiver_materializes(self, monkeypatch):
        """A kwargs-only call materializes the bookkeeping on the supplied model
        (IN-04: transformers may call the load-boundary hook with keywords)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")

        recorded = {}

        def fake_original(model, load_config, loading_info):
            recorded["model"] = model
            recorded["load_config"] = load_config
            recorded["loading_info"] = loading_info
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        model = _FakeBookkeepingModel()
        result = PreTrainedModel._finalize_model_loading(
            model=model, load_config="load-config", loading_info="loading-info"
        )

        assert result == "delegated"
        assert recorded["model"] is model
        assert recorded["load_config"] == "load-config"
        assert recorded["loading_info"] == "loading-info"
        assert hasattr(model, "all_tied_weights_keys"), (
            "a kwargs-only call must still materialize all_tied_weights_keys "
            "on the supplied model (IN-04)"
        )

    def test_kwargs_receiver_wins_over_class_shaped_positional(self, monkeypatch):
        """A kwargs ``model`` receiver beats any positional, so a class-shaped
        positional never receives model-instance bookkeeping (IN-04)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")

        class _ClassShapedReceiver:
            """Class carrying the computation surface, like a bound class receiver."""

            @classmethod
            def get_expanded_tied_weights_keys(cls, all_submodels=False):
                return {"head.weight": "backbone.weight"}

            @classmethod
            def named_children(cls):
                return iter(())

            @classmethod
            def tie_weights(cls, missing_keys=None, recompute_mapping=True):
                """Native-shaped tie_weights (the bridge must leave it alone)."""

        recorded = {}

        def fake_original(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        model = _FakeBookkeepingModel()
        result = PreTrainedModel._finalize_model_loading(
            _ClassShapedReceiver, model=model, load_config=None, loading_info=None
        )

        assert result == "delegated"
        assert recorded["kwargs"]["model"] is model, (
            "the delegated original must receive the caller's keyword arguments unchanged"
        )
        assert "all_tied_weights_keys" not in vars(_ClassShapedReceiver), (
            "a class-shaped positional must never gain model-instance bookkeeping"
        )
        assert hasattr(model, "all_tied_weights_keys")

    def test_missing_receiver_delegates(self, monkeypatch):
        """With no plausible receiver the call delegates unchanged and never
        reaches a receiver helper (IN-04)."""
        _rung("_patch_tied_weights_bookkeeping_on_load")

        def fail_helper(*_args, **_kwargs):
            raise AssertionError("no receiver helper may run without a receiver")

        monkeypatch.setattr(
            transformers_compat, "_materialize_tied_weights_bookkeeping", fail_helper
        )
        monkeypatch.setattr(transformers_compat, "_bridge_legacy_tie_weights", fail_helper)

        def fake_original(*args, **kwargs):
            return "delegated"

        monkeypatch.setattr(
            transformers_compat, "_tied_bookkeeping_original_finalize", fake_original
        )

        assert PreTrainedModel._finalize_model_loading() == "delegated"

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
# Rung 3 (27.2-01, D-04): the 'default' RoPE initializer restored for GENERanno
# ─────────────────────────────────────────────────────────────────────────────


class TestRopeDefaultInit:
    """Contract: the 4.x ``'default'`` ROPE_INIT_FUNCTIONS entry is restored on 5.x.

    Measured failure (27.2-01, lgq12697/GENERanno-eukaryote-0.5b-base, ms
    commit 0ae1f40f): the remote ``GenerannoRotaryEmbedding.__init__`` sets
    ``rope_type = "default"`` when ``config.rope_scaling`` is None and indexes
    the table, which transformers 5.17 no longer carries (its keys are the
    scaling variants only). The rung re-registers the vendored v4.49.0
    default initializer IN PLACE so every module holding the live dict object
    -- including the already-imported remote code -- sees the key (Pitfall 6:
    never reassign the dict, remote modules hold a bound reference).
    """

    def test_default_key_resolves_on_the_live_table(self):
        """After apply_patches, the live ROPE table answers 'default' with a callable."""
        table = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS
        assert "default" in table
        assert callable(table["default"])

    def test_repeat_install_preserves_the_installed_object(self):
        """A second rung call leaves the installed entry untouched (identity)."""
        patch = _rung("_patch_rope_default_init")
        table = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS
        installed = table["default"]

        patch()

        assert table["default"] is installed

    def test_absence_gate_never_overwrites_a_prepopulated_key(self):
        """A table already carrying 'default' (transformers 4.x, or already
        patched) keeps its existing object -- the rung only ever fills an
        ABSENT name."""
        patch = _rung("_patch_rope_default_init")
        table = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS
        original = table["default"]
        sentinel = object()
        table["default"] = sentinel
        try:
            patch()
            assert table["default"] is sentinel, "a present 'default' key must never be overwritten"
        finally:
            table["default"] = original

    def test_vendored_initializer_reproduces_the_v449_inv_freq_formula(self):
        """The vendored initializer computes the 4.49 default-RoPE frequencies:
        ``1 / rope_theta ** (arange(0, dim, 2) / dim)`` on the requested
        device, with attention scaling 1.0."""
        initializer = _rung("_compute_default_rope_parameters")

        config = types.SimpleNamespace(
            rope_theta=10000.0, head_dim=8, hidden_size=32, num_attention_heads=4
        )
        inv_freq, attention_factor = initializer(config, device="cpu")

        expected = 1.0 / (
            10000.0 ** (torch.arange(0, 8, 2, dtype=torch.int64).to(torch.float) / 8)
        )
        assert torch.equal(inv_freq, expected)
        assert inv_freq.device.type == "cpu"
        assert attention_factor == 1.0

    def test_initializer_derives_dim_from_hidden_size_when_head_dim_absent(self):
        """Without ``head_dim`` the frequency dimension is
        ``hidden_size // num_attention_heads`` (the 4.49 fallback)."""
        initializer = _rung("_compute_default_rope_parameters")

        config = types.SimpleNamespace(
            rope_theta=500.0, hidden_size=16, num_attention_heads=2
        )
        inv_freq, attention_factor = initializer(config, device=None)

        # head_dim = 16 // 2 = 8 -> arange(0, 8, 2) -> 4 frequency entries
        assert inv_freq.shape == (4,)
        expected = 1.0 / (500.0 ** (torch.arange(0, 8, 2, dtype=torch.int64).to(torch.float) / 8))
        assert torch.equal(inv_freq, expected)
        assert attention_factor == 1.0

    def test_initializer_honors_partial_rotary_factor(self):
        """``partial_rotary_factor`` scales the rotary dimension (4.49 reads
        it from the config when present, defaulting to 1.0)."""
        initializer = _rung("_compute_default_rope_parameters")

        config = types.SimpleNamespace(
            rope_theta=10000.0, head_dim=8, partial_rotary_factor=0.5
        )
        inv_freq, _ = initializer(config, device=None)

        # dim = int(8 * 0.5) = 4 -> arange(0, 4, 2) -> 2 frequency entries
        assert inv_freq.shape == (2,)
        expected = 1.0 / (
            10000.0 ** (torch.arange(0, 4, 2, dtype=torch.int64).to(torch.float) / 4)
        )
        assert torch.equal(inv_freq, expected)


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
        rope_before = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS.get("default")

        apply_patches()

        assert PreTrainedModel.__dict__["_finalize_model_loading"] is finalize_before
        assert (
            getattr(transformers.modeling_utils, "apply_chunking_to_forward", None)
            is chunking_before
        )
        assert transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS.get("default") is rope_before
