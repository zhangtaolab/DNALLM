"""Behavior-contract tests for the transformers compatibility shim (dnallm.utils.transformers_compat).

The patches under test were applied to the real ``transformers.PreTrainedModel``
at import time. These tests assert the CONTRACT of the patched live class —
idempotency, proxy semantics, candidate classification, and the swap/restore
flow — and never remove or undo the class patches (other tests in the suite
depend on the patched state).
"""

from __future__ import annotations

import copy
import sys
import types
from typing import ClassVar
from unittest.mock import Mock

import bitsandbytes.functional as bnb_functional
import pytest
import torch
import transformers.cache_utils
import transformers.configuration_utils
import transformers.modeling_utils
import transformers.models.deberta_v2.tokenization_deberta_v2
import transformers.pytorch_utils
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

from dnallm.utils import transformers_compat
from dnallm.utils.transformers_compat import (
    _QuantStatProxy,
    _attach_remote_code_pruning_helpers,
    _find_pruneable_heads_and_indices,
    _get_extended_attention_mask,
    _iter_uninitialized_quantized_weights,
    _patch_get_extended_attention_mask,
    _patch_get_parameter_or_buffer,
    _patch_initialize_weights_for_quantized_missing,
    _patch_remote_code_pruning_helpers,
    _prune_linear_layer,
    _restore_quantized,
    _swap_to_fp32,
    apply_patches,
)


class Params4bit:
    """Fake packed 4-bit parameter.

    The class NAME is load-bearing: the shim selects candidates by
    ``type(weight).__name__ == "Params4bit"``.
    """

    def __init__(self, data=None):
        self.data = data if data is not None else torch.zeros(4, dtype=torch.uint8)
        self.quant_state = {"absmax": 1.0}
        self.blocksize = 64
        self.quant_type = "fp4"
        self.compress_statistics = True
        self.quant_storage = torch.uint8

    def is_floating_point(self):
        """Packed uint8 storage is not floating point."""
        return False


class MarkedWeight:
    """Fake fp parameter already carrying the HF init mark (not a candidate)."""

    def __init__(self):
        self._is_hf_initialized = True
        self.quant_state = None


class FakeModule:
    """Fake nn.Module exposing weight both as an attr and in _parameters."""

    def __init__(self, weight):
        self.weight = weight
        self._parameters = {"weight": weight}


class WeightlessModule:
    """Fake module without any weight attribute."""


class FakeModel:
    """Fake model whose modules() yields a fixed list."""

    def __init__(self, modules):
        self._modules_list = list(modules)

    def modules(self):
        """Iterate the fixed module list."""
        return iter(self._modules_list)


class FakeInitSelf:
    """Fake receiver for the patched initialize_weights.

    The instance-attr ``smart_apply`` takes precedence over the class-level
    helper, so the REAL upstream ``initialize_weights`` body runs against a
    recording Mock instead of walking a real module tree.
    """

    def __init__(self, modules=()):
        self._initialize_weights = Mock(name="_initialize_weights")
        self.is_custom_code = Mock(return_value=False)
        self.smart_apply = Mock(name="smart_apply")
        self._modules_list = list(modules)

    def modules(self):
        """Iterate the fixed module list."""
        return iter(self._modules_list)


class TestApplyPatchesIdempotency:
    """Contract (a): apply_patches is idempotent on the live class."""

    def test_apply_patches_is_idempotent_on_live_class(self):
        """A second explicit apply_patches() must not rebind either accessor."""
        accessor_before = PreTrainedModel.get_parameter_or_buffer
        init_before = PreTrainedModel.initialize_weights

        apply_patches()

        assert accessor_before is PreTrainedModel.get_parameter_or_buffer
        assert init_before is PreTrainedModel.initialize_weights
        assert PreTrainedModel._dnallm_quant_key_patch is True
        assert PreTrainedModel._dnallm_quant_init_patch is True

    def test_patch_functions_return_early_when_already_flagged(self):
        """The class-patch guard flags short-circuit repeat patch calls."""
        accessor_before = PreTrainedModel.get_parameter_or_buffer
        init_before = PreTrainedModel.initialize_weights

        assert _patch_get_parameter_or_buffer() is None
        assert _patch_initialize_weights_for_quantized_missing() is None

        assert accessor_before is PreTrainedModel.get_parameter_or_buffer
        assert init_before is PreTrainedModel.initialize_weights

    def test_patch_skips_when_target_method_absent(self, monkeypatch):
        """Older transformers without the target method must be left untouched."""

        class _TargetlessModel:
            """Stand-in for a PreTrainedModel without the patchable methods."""

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _TargetlessModel)
        _patch_get_parameter_or_buffer()
        _patch_initialize_weights_for_quantized_missing()

        assert getattr(_TargetlessModel, "_dnallm_quant_key_patch", False) is False
        assert getattr(_TargetlessModel, "_dnallm_quant_init_patch", False) is False


class TestPatchedGetParameterOrBuffer:
    """Contract (b): the patched accessor proxies only what the original rejects."""

    PARENT_TENSOR = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class _FakeSelf:
        """Fake model receiver for direct accessor calls."""

        def __init__(self, parent_key="x.weight", resolve_via_parameter=True, tensor=None):
            self._parent_key = parent_key
            self._resolve_via_parameter = resolve_via_parameter
            self._tensor = tensor

        def get_parameter(self, target):
            """Resolve only the configured parent key, else AttributeError."""
            if self._resolve_via_parameter and target == self._parent_key:
                return self._tensor
            raise AttributeError(f"no parameter {target!r}")

        def get_buffer(self, target):
            """Resolve only the configured parent key, else AttributeError."""
            if not self._resolve_via_parameter and target == self._parent_key:
                return self._tensor
            raise AttributeError(f"no buffer {target!r}")

    def test_passthrough_when_original_succeeds(self):
        """Keys the original resolves must come back unchanged, not proxied."""
        fake = self._FakeSelf(tensor=self.PARENT_TENSOR)
        result = PreTrainedModel.get_parameter_or_buffer(fake, "x.weight")
        assert result is self.PARENT_TENSOR

    def test_returns_proxy_when_original_raises_and_parent_resolves(self):
        """A nested quant-stat key resolves to a proxy of the parent tensor."""
        fake = self._FakeSelf(parent_key="x.weight", tensor=self.PARENT_TENSOR)
        result = PreTrainedModel.get_parameter_or_buffer(fake, "x.weight.absmax")
        assert type(result).__name__ == "_QuantStatProxy"
        assert result.shape == self.PARENT_TENSOR.shape
        assert result.dtype == self.PARENT_TENSOR.dtype

    def test_proxy_via_get_buffer_fallback(self):
        """When get_parameter misses the parent, get_buffer must be tried."""
        fake = self._FakeSelf(
            parent_key="x.weight", resolve_via_parameter=False, tensor=self.PARENT_TENSOR
        )
        result = PreTrainedModel.get_parameter_or_buffer(fake, "x.weight.absmax")
        assert type(result).__name__ == "_QuantStatProxy"
        assert result.shape == self.PARENT_TENSOR.shape

    def test_reraises_when_no_parent_resolves(self):
        """A genuinely missing key must still raise the original AttributeError."""

        class _NoResolve:
            def get_parameter(self, target):
                raise AttributeError(f"no parameter {target!r}")

            def get_buffer(self, target):
                raise AttributeError(f"no buffer {target!r}")

        with pytest.raises(AttributeError):
            PreTrainedModel.get_parameter_or_buffer(_NoResolve(), "a.b.c")

    def test_non_string_target_reraises_original_error(self):
        """Non-str targets never enter the parent walk; the original error re-raises."""
        fake = self._FakeSelf(tensor=self.PARENT_TENSOR)
        with pytest.raises(AttributeError, match="neither a parameter"):
            PreTrainedModel.get_parameter_or_buffer(fake, ("x", "weight"))

    def test_non_attribute_errors_propagate_unpatched(self):
        """Only AttributeError activates the patch; other errors pass through."""
        fake = self._FakeSelf(tensor=self.PARENT_TENSOR)
        with pytest.raises(TypeError):
            # bytes targets crash inside the original with TypeError before
            # any AttributeError is raised — the shim must stay inert there.
            PreTrainedModel.get_parameter_or_buffer(fake, b"x.weight")


class TestQuantStatProxy:
    """Contract (c): attribute forwarding plus the init-flag drop."""

    def test_forwards_attribute_reads_to_parent(self):
        """Reads transparently resolve on the wrapped parent tensor."""
        tensor = torch.tensor([[1.0, 2.0]])
        proxy = _QuantStatProxy(tensor)
        assert proxy.shape == tensor.shape
        assert proxy.dtype == tensor.dtype
        assert proxy.is_floating_point() is True

    def test_forwards_regular_attribute_sets(self):
        """Ordinary setattr lands on the parent tensor."""
        tensor = torch.tensor([1.0])
        proxy = _QuantStatProxy(tensor)
        proxy.analysis_tag = "kept"
        assert tensor.analysis_tag == "kept"

    def test_drops_hf_initialized_flag_sets(self):
        """_is_hf_initialized sets must never reach the parent tensor."""
        tensor = torch.tensor([1.0])
        proxy = _QuantStatProxy(tensor)
        proxy._is_hf_initialized = True  # must be silently dropped, not raise

        assert not hasattr(tensor, "_is_hf_initialized")


class TestIterUninitializedQuantizedWeights:
    """Contract (d): candidate/marked classification over fake modules."""

    def test_classifies_candidates_and_marks(self):
        """Packed unmarked Params4bit weights are candidates; marks are reported."""
        candidate = Params4bit()
        candidate_module = FakeModule(candidate)
        model = FakeModel([
            WeightlessModule(),
            FakeModule(MarkedWeight()),
            candidate_module,
            torch.nn.Linear(4, 2),  # fp Parameter: neither candidate nor marked
        ])

        candidates, marked = _iter_uninitialized_quantized_weights(model)

        assert candidates == [(candidate_module, candidate)]
        assert marked is True

    def test_plain_model_yields_no_candidates_and_no_marks(self):
        """A model of ordinary fp modules is neither candidate-bearing nor marked."""
        model = FakeModel([WeightlessModule(), torch.nn.Linear(4, 2)])

        candidates, marked = _iter_uninitialized_quantized_weights(model)

        assert candidates == []
        assert marked is False


class TestSwapToFP32:
    """Unit contract for _swap_to_fp32 against a monkeypatched bnb.functional."""

    def test_replaces_packed_weight_with_dequantized_parameter(self, monkeypatch):
        """Each candidate is swapped to a no-grad Parameter of the dequantized value."""
        dequantize = Mock(return_value=torch.ones(4))
        monkeypatch.setattr(bnb_functional, "dequantize_4bit", dequantize)

        weight = Params4bit()
        module = FakeModule(weight)
        swapped = _swap_to_fp32([(module, weight)])

        assert swapped == [(module, weight)]
        # Compare call args field-by-field: Mock equality on tensor args is
        # ambiguous (element-wise bool).
        assert dequantize.call_count == 1
        call_args, _ = dequantize.call_args
        assert call_args[0] is weight.data
        assert call_args[1] is weight.quant_state
        replacement = module._parameters["weight"]
        assert replacement is not weight
        assert isinstance(replacement, torch.nn.Parameter)
        assert replacement.requires_grad is False
        assert torch.equal(replacement.data, torch.ones(4))


class TestRestoreQuantized:
    """Unit contract for _restore_quantized against a monkeypatched bnb.functional."""

    def test_packs_fp_value_and_restores_original_weight(self, monkeypatch):
        """The original Params4bit object returns with re-quantized data and the mark."""
        packed = torch.zeros(4, dtype=torch.uint8)
        quant_state = {"absmax": 2.0}
        quantize = Mock(return_value=(packed, quant_state))
        monkeypatch.setattr(bnb_functional, "quantize_4bit", quantize)

        weight = Params4bit()
        fp_value = torch.ones(4)
        module = FakeModule(weight)
        module._parameters["weight"] = torch.nn.Parameter(fp_value, requires_grad=False)

        _restore_quantized([(module, weight)])

        assert quantize.call_count == 1
        call_args, call_kwargs = quantize.call_args
        assert torch.equal(call_args[0], fp_value)
        assert call_kwargs == {
            "blocksize": weight.blocksize,
            "quant_type": weight.quant_type,
            "compress_statistics": weight.compress_statistics,
            "quant_storage": torch.uint8,
        }
        assert module._parameters["weight"] is weight
        assert weight.data is packed
        assert weight.quant_state is quant_state
        assert weight._is_hf_initialized is True


class TestInitializeWeightsWrapper:
    """Contract (e): the patched initialize_weights passthrough/swap/restore arms."""

    def _marked_candidate_model_modules(self):
        """A module list with one marked fp weight and one unmarked Params4bit."""
        candidate = Params4bit()
        module = FakeModule(candidate)
        return module, candidate, [module, FakeModule(MarkedWeight())]

    def test_passthrough_when_nothing_marked_and_no_candidates(self):
        """Ordinary models take the untouched original path exactly once."""
        fake = FakeInitSelf([torch.nn.Linear(4, 2)])

        PreTrainedModel.initialize_weights(fake)

        assert fake.smart_apply.call_count == 1
        assert fake.smart_apply.call_args.args[0] is fake._initialize_weights

    def test_passthrough_when_bitsandbytes_unavailable(self, monkeypatch):
        """Missing bitsandbytes must fall back to the original unswapped path."""
        module, weight, modules = self._marked_candidate_model_modules()
        fake = FakeInitSelf(modules)
        dequantize = Mock()
        monkeypatch.setattr(bnb_functional, "dequantize_4bit", dequantize)
        # None in sys.modules makes `import bitsandbytes` raise ImportError.
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)

        PreTrainedModel.initialize_weights(fake)

        assert fake.smart_apply.call_count == 1
        assert module._parameters["weight"] is weight  # never swapped
        assert dequantize.call_count == 0

    def test_swap_dequantize_original_restore_flow(self, monkeypatch):
        """Marked candidates are swapped, the original runs, then everything restores."""
        module, weight, modules = self._marked_candidate_model_modules()
        original_data = weight.data  # restore() reassigns both after the flow
        original_quant_state = weight.quant_state
        order = []
        packed = torch.zeros(4, dtype=torch.uint8)
        quant_state = {"absmax": 3.0}
        dequantize = Mock(side_effect=lambda data, qs: (order.append("deq"), torch.ones(4))[1])
        quantize = Mock(
            side_effect=lambda value, **kwargs: (order.append("quant"), (packed, quant_state))[1]
        )
        monkeypatch.setattr(bnb_functional, "dequantize_4bit", dequantize)
        monkeypatch.setattr(bnb_functional, "quantize_4bit", quantize)

        fake = FakeInitSelf(modules)
        fake.smart_apply = Mock(side_effect=lambda *args: order.append("orig"))

        PreTrainedModel.initialize_weights(fake)

        # Observable ordering: dequantize -> original -> re-quantize.
        assert order == ["deq", "orig", "quant"]
        assert dequantize.call_count == 1
        call_args, _ = dequantize.call_args
        assert call_args[0] is original_data
        assert call_args[1] is original_quant_state
        assert module._parameters["weight"] is weight
        assert weight.data is packed
        assert weight.quant_state is quant_state
        assert weight._is_hf_initialized is True

    def test_restore_runs_even_when_original_raises(self, monkeypatch):
        """The finally-clause restore must survive an exception in the original."""
        module, weight, modules = self._marked_candidate_model_modules()
        packed = torch.zeros(4, dtype=torch.uint8)
        monkeypatch.setattr(bnb_functional, "dequantize_4bit", Mock(return_value=torch.ones(4)))
        monkeypatch.setattr(
            bnb_functional, "quantize_4bit", Mock(return_value=(packed, {"absmax": 1.0}))
        )

        fake = FakeInitSelf(modules)
        fake.smart_apply = Mock(side_effect=RuntimeError("init exploded"))

        with pytest.raises(RuntimeError, match="init exploded"):
            PreTrainedModel.initialize_weights(fake)

        assert module._parameters["weight"] is weight
        assert weight._is_hf_initialized is True


class TestRemoteCodePruningHelpers:
    """Contract for the vendored v4.49.0 pruning helpers re-attached on transformers 5.x.

    Covers the helper arithmetic (head-pruning index math, linear-layer
    pruning shapes/values) and the live-module attachment contract on BOTH
    import sites 4.x remote code uses — ``transformers.modeling_utils`` and
    ``transformers.pytorch_utils`` — presence, idempotence, per-name absence
    gating (a native symbol is never overwritten), and version awareness:
    on transformers 5.x the exposed helpers ARE the vendored functions; on
    4.x they are upstream natives and the patch is an absence-gated no-op
    (the identity assertions invert rather than skip, so the file collects
    and passes cleanly across the whole CI matrix).
    """

    def test_find_pruneable_heads_and_indices_basic_head_removal(self):
        """Pruning head 1 of 4 heads (head_size 2) keeps 6 of 8 rows, removing 2 and 3."""
        heads, index = _find_pruneable_heads_and_indices([1], 4, 2, set())

        assert heads == {1}, f"expected heads {{1}}, got {heads}"
        assert isinstance(index, torch.Tensor)
        assert index.numel() == 6, f"expected 6 kept rows, got {index.numel()}"
        kept = set(index.tolist())
        assert kept == {0, 1, 4, 5, 6, 7}, f"positions 2 and 3 must be removed, got {sorted(kept)}"

    def test_find_pruneable_heads_and_indices_respects_already_pruned_heads(self):
        """An already-pruned head is not double-counted and shifts the index arithmetic.

        ``already_pruned_heads`` is the pruned-heads container the attention
        module passes — a set (the remote EsmAttention keeps
        ``self.pruned_heads = set()``), matching the v4.49.0 ``Set[int]``
        contract; the plan's ``{0: 0}`` dict literal would TypeError inside
        the mandated-verbatim ``set - already_pruned_heads`` subtraction.
        """
        heads, index = _find_pruneable_heads_and_indices([0, 1], 4, 2, {0})

        assert heads == {1}, f"already-pruned head 0 must not re-prune, got {heads}"
        kept = set(index.tolist())
        assert kept == {2, 3, 4, 5, 6, 7}, (
            f"head 1 shifts down by the prior pruning of head 0 (mask row 0 "
            f"cleared), expected rows 2-7 kept, got {sorted(kept)}"
        )

    def test_prune_linear_layer_dim1_prunes_input_features(self):
        """dim=1 keeps 2 of 8 input features: weight (4, 2) equal to index_select."""
        layer = torch.nn.Linear(8, 4)
        index = torch.tensor([3, 7])

        pruned = _prune_linear_layer(layer, index, dim=1)

        assert isinstance(pruned, torch.nn.Linear)
        assert tuple(pruned.weight.shape) == (4, 2), (
            f"expected weight (4, 2), got {tuple(pruned.weight.shape)}"
        )
        assert torch.equal(pruned.weight, layer.weight.index_select(1, index))
        assert torch.equal(pruned.bias, layer.bias), "dim=1 keeps all outputs, bias untouched"

    def test_prune_linear_layer_dim0_prunes_outputs_and_slices_bias(self):
        """dim=0 keeps 2 of 4 outputs: weight (2, 8) and the same bias rows sliced."""
        layer = torch.nn.Linear(8, 4)
        index = torch.tensor([1, 3])

        pruned = _prune_linear_layer(layer, index, dim=0)

        assert tuple(pruned.weight.shape) == (2, 8), (
            f"expected weight (2, 8), got {tuple(pruned.weight.shape)}"
        )
        assert torch.equal(pruned.weight, layer.weight.index_select(0, index))
        assert torch.equal(pruned.bias, layer.bias[index]), "dim=0 slices the bias the same way"

    def test_live_module_exposes_helpers_and_repeat_patch_is_idempotent(self):
        """Import-time apply_patches() exposes both helpers; repeat calls keep them callable."""
        modeling_utils = transformers.modeling_utils
        assert callable(modeling_utils.find_pruneable_heads_and_indices)
        assert callable(modeling_utils.prune_linear_layer)

        apply_patches()
        _patch_remote_code_pruning_helpers()  # already-flagged guard must return early

        heads, index = modeling_utils.find_pruneable_heads_and_indices([1], 4, 2, set())
        assert heads == {1}
        assert index.numel() == 6
        pruned = modeling_utils.prune_linear_layer(
            torch.nn.Linear(8, 4), torch.tensor([0, 1]), dim=1
        )
        assert tuple(pruned.weight.shape) == (4, 2)

    def test_attachment_identity_is_version_agnostic(self):
        """On transformers 5.x the exposed helpers ARE the vendored ones; on 4.x upstream's."""
        major = int(transformers.__version__.split(".")[0])
        modeling_utils = transformers.modeling_utils

        if major >= 5:
            assert (
                modeling_utils.find_pruneable_heads_and_indices is _find_pruneable_heads_and_indices
            ), "5.x attachment must be the vendored function"
            assert modeling_utils.prune_linear_layer is _prune_linear_layer, (
                "5.x attachment must be the vendored function"
            )
        else:
            assert (
                modeling_utils.find_pruneable_heads_and_indices
                is not _find_pruneable_heads_and_indices
            ), "4.x no-op gate must leave upstream's own helper in place"
            assert modeling_utils.prune_linear_layer is not _prune_linear_layer, (
                "4.x no-op gate must leave upstream's own helper in place"
            )

    def test_live_pytorch_utils_exposes_helpers_and_repeat_patch_is_idempotent(self):
        """Import-time apply_patches() also covers the pytorch_utils import site."""
        pytorch_utils = transformers.pytorch_utils
        assert callable(pytorch_utils.find_pruneable_heads_and_indices)
        assert callable(pytorch_utils.prune_linear_layer)

        apply_patches()
        _patch_remote_code_pruning_helpers()  # already-flagged guard must return early

        heads, index = pytorch_utils.find_pruneable_heads_and_indices([1], 4, 2, set())
        assert heads == {1}
        assert index.numel() == 6
        pruned = pytorch_utils.prune_linear_layer(
            torch.nn.Linear(8, 4), torch.tensor([0, 1]), dim=1
        )
        assert tuple(pruned.weight.shape) == (4, 2)

    def test_pytorch_utils_attachment_identity_is_version_agnostic(self):
        """5.x pytorch_utils receives the vendored find-helper; 4.x keeps upstream's own.

        ``prune_linear_layer`` gets no identity assertion on 5.x on purpose:
        5.17 pytorch_utils keeps its native implementation (which the patch
        must leave in place), while a future 5.x that drops it would receive
        the vendored one — both are the per-name absence-gated contract.
        """
        major = int(transformers.__version__.split(".")[0])
        pytorch_utils = transformers.pytorch_utils

        if major >= 5:
            assert (
                pytorch_utils.find_pruneable_heads_and_indices is _find_pruneable_heads_and_indices
            ), "5.x pytorch_utils attachment must be the vendored function"
        else:
            assert (
                pytorch_utils.find_pruneable_heads_and_indices
                is not _find_pruneable_heads_and_indices
            ), "4.x no-op gate must leave upstream's own helper in place"

    def test_attach_helper_attaches_only_missing_names_and_never_overwrites(self):
        """A module keeping a native prune_linear_layer (the 5.x pytorch_utils
        shape) receives only the missing find-helper; the native symbol survives."""
        fake = types.ModuleType("fake_pruning_module")
        native_prune = object()
        fake.prune_linear_layer = native_prune

        _attach_remote_code_pruning_helpers(fake)

        assert fake.find_pruneable_heads_and_indices is _find_pruneable_heads_and_indices
        assert fake.prune_linear_layer is native_prune
        assert fake._dnallm_remote_code_pruning_patch is True

    def test_attach_helper_noops_when_both_names_already_native(self):
        """A module exposing both names natively (the 4.x shape) is left
        untouched and does not even receive the sentinel."""
        fake = types.ModuleType("fake_pruning_module")
        native_find, native_prune = object(), object()
        fake.find_pruneable_heads_and_indices = native_find
        fake.prune_linear_layer = native_prune

        _attach_remote_code_pruning_helpers(fake)

        assert fake.find_pruneable_heads_and_indices is native_find
        assert fake.prune_linear_layer is native_prune
        assert not hasattr(fake, "_dnallm_remote_code_pruning_patch")

    def test_attach_helper_sentinel_short_circuits_repeat_calls(self):
        """A module already carrying the sentinel is never revisited."""
        fake = types.ModuleType("fake_pruning_module")
        fake._dnallm_remote_code_pruning_patch = True

        _attach_remote_code_pruning_helpers(fake)

        assert not hasattr(fake, "find_pruneable_heads_and_indices")
        assert not hasattr(fake, "prune_linear_layer")

    def test_patch_routes_synthetic_pytorch_utils_module(self, monkeypatch):
        """_patch_remote_code_pruning_helpers routes into pytorch_utils: a
        synthetic stand-in missing both names receives both vendored helpers
        (the live modules keep their already-patched state untouched).

        The shim's function-local ``import transformers.pytorch_utils`` binds
        whatever ``sys.modules["transformers"]`` holds at call time -- on
        transformers 5.x importing dnallm re-executes the lazy
        ``transformers/__init__`` and swaps that entry for a fresh
        ``_LazyModule``, so the module object this file's top-level
        ``import transformers`` bound can be a STALE parent the shim never
        sees.  Patch the CURRENT sys.modules parent (raising=False: the lazy
        parent exposes submodules through __getattr__, not instance attrs).
        """
        parent = sys.modules["transformers"]
        fake_pu = types.ModuleType("transformers.pytorch_utils")
        monkeypatch.setattr(parent, "pytorch_utils", fake_pu, raising=False)

        _patch_remote_code_pruning_helpers()

        assert fake_pu.find_pruneable_heads_and_indices is _find_pruneable_heads_and_indices
        assert fake_pu.prune_linear_layer is _prune_linear_layer
        assert fake_pu._dnallm_remote_code_pruning_patch is True


class TestGetExtendedAttentionMask:
    """Contract for the vendored v4.49.0 mask-expansion method re-attached on transformers 5.x.

    Covers the shape/value semantics of
    ``ModuleUtilsMixin.get_extended_attention_mask`` as 4.x-era remote code
    (``EsmModel.forward``) consumes it — 2D/3D broadcast expansion, the
    0.0 / dtype-min value mapping, dtype fallback and override — plus the
    live-class attachment contract: callable presence, idempotent
    absence-gated attachment, and version awareness: on transformers 5.x the
    exposed method IS the vendored function; on 4.x it is upstream's native
    one and the patch is an absence-gated no-op (the identity assertions
    invert rather than skip, so the file collects and passes cleanly across
    the whole CI matrix).
    """

    def test_2d_encoder_mask_expands_to_broadcastable_shape(self):
        """A 2D (2, 4) encoder mask expands to (2, 1, 1, 4) with 0.0 / dtype-min values."""
        fake = types.SimpleNamespace(dtype=torch.float32, config=types.SimpleNamespace())
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])

        extended = _get_extended_attention_mask(fake, mask, (2, 4))

        assert tuple(extended.shape) == (2, 1, 1, 4)
        assert float(extended[0, 0, 0, 0]) == 0.0, "attended position must become 0.0"
        assert float(extended[0, 0, 0, 2]) == torch.finfo(torch.float32).min, (
            "masked position must become the dtype minimum"
        )
        assert float(extended[1, 0, 0, 1]) == torch.finfo(torch.float32).min

    def test_3d_mask_expands_to_batched_pairwise_shape(self):
        """A 3D (2, 3, 4) mask expands to (2, 1, 3, 4) with the same value mapping."""
        fake = types.SimpleNamespace(dtype=torch.float32, config=types.SimpleNamespace())
        mask = torch.ones(2, 3, 4)
        mask[0, 1, 2] = 0.0

        extended = _get_extended_attention_mask(fake, mask, (2, 4))

        assert tuple(extended.shape) == (2, 1, 3, 4)
        assert float(extended[0, 0, 0, 0]) == 0.0, "attended position must become 0.0"
        assert float(extended[0, 0, 1, 2]) == torch.finfo(torch.float32).min

    def test_explicit_dtype_controls_output_dtype_and_fill_value(self):
        """dtype=torch.float16 yields float16 output masked with the float16 minimum."""
        fake = types.SimpleNamespace(dtype=torch.float32, config=types.SimpleNamespace())
        mask = torch.tensor([[1.0, 0.0]])

        extended = _get_extended_attention_mask(fake, mask, (1, 2), dtype=torch.float16)

        assert extended.dtype == torch.float16
        assert float(extended[0, 0, 0, 0]) == 0.0
        assert float(extended[0, 0, 0, 1]) == torch.finfo(torch.float16).min

    def test_dtype_none_falls_back_to_receiver_dtype(self):
        """With no dtype kwarg the output takes the receiver's own dtype."""
        fake = types.SimpleNamespace(dtype=torch.float64, config=types.SimpleNamespace())
        mask = torch.tensor([[1.0, 1.0, 1.0]])

        extended = _get_extended_attention_mask(fake, mask, (1, 3))

        assert extended.dtype == torch.float64
        assert float(extended[0, 0, 0, 2]) == 0.0

    def test_config_without_is_decoder_attribute_takes_encoder_branch(self):
        """A config lacking is_decoder entirely (the 05-04 D-07 next rung) still expands."""
        fake = types.SimpleNamespace(dtype=torch.float32, config=types.SimpleNamespace())
        assert not hasattr(fake.config, "is_decoder"), "fixture must exercise the missing case"

        extended = _get_extended_attention_mask(fake, torch.tensor([[1.0, 0.0]]), (1, 2))

        assert tuple(extended.shape) == (1, 1, 1, 2)
        assert float(extended[0, 0, 0, 1]) == torch.finfo(torch.float32).min

    def test_wrong_dimensionality_raises_value_error(self):
        """A 1D mask raises upstream's Wrong shape ValueError."""
        fake = types.SimpleNamespace(dtype=torch.float32, config=types.SimpleNamespace())

        with pytest.raises(ValueError, match="Wrong shape"):
            _get_extended_attention_mask(fake, torch.tensor([1.0, 0.0, 1.0]), (1, 3))

    def test_decoder_config_raises_not_implemented(self):
        """is_decoder=True fails loudly instead of returning a non-causal mask."""
        fake = types.SimpleNamespace(
            dtype=torch.float32, config=types.SimpleNamespace(is_decoder=True)
        )

        with pytest.raises(NotImplementedError, match="decoder"):
            _get_extended_attention_mask(fake, torch.tensor([[1.0, 1.0]]), (1, 2))

    def test_live_class_attachment_identity_is_version_agnostic(self):
        """On transformers 5.x the live method IS the vendored one; on 4.x it is NOT."""
        major = int(transformers.__version__.split(".")[0])

        assert callable(PreTrainedModel.get_extended_attention_mask)

        if major >= 5:
            assert PreTrainedModel.get_extended_attention_mask is _get_extended_attention_mask, (
                "5.x attachment must be the vendored function"
            )
        else:
            assert (
                PreTrainedModel.get_extended_attention_mask is not _get_extended_attention_mask
            ), "4.x no-op gate must leave upstream's native method in place"

    def test_apply_patches_rebind_is_idempotent_with_sentinel(self):
        """A second apply_patches() leaves the bound method identical and the sentinel stable."""
        major = int(transformers.__version__.split(".")[0])
        before = PreTrainedModel.get_extended_attention_mask

        apply_patches()

        assert PreTrainedModel.get_extended_attention_mask is before, "no rebind may happen"
        if major >= 5:
            assert PreTrainedModel._dnallm_extended_mask_patch is True
        else:
            assert not hasattr(PreTrainedModel, "_dnallm_extended_mask_patch"), (
                "4.x absence gate must not even set the sentinel"
            )

    def test_patch_attaches_only_where_class_lacks_the_method(self, monkeypatch):
        """A bare stand-in class receives the vendored method and sentinel; a class already
        exposing a native method is left untouched and gets no sentinel."""

        class _BareModel:
            """Stand-in for a PreTrainedModel without get_extended_attention_mask."""

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _BareModel)
        _patch_get_extended_attention_mask()

        assert _BareModel.get_extended_attention_mask is _get_extended_attention_mask
        assert _BareModel._dnallm_extended_mask_patch is True

        class _NativeModel:
            """Stand-in whose class already carries a native method (the 4.x shape)."""

            def get_extended_attention_mask(self, attention_mask, input_shape):
                """Native placeholder that must never be overwritten."""
                return attention_mask

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _NativeModel)
        _patch_get_extended_attention_mask()

        assert _NativeModel.get_extended_attention_mask is not _get_extended_attention_mask
        assert not hasattr(_NativeModel, "_dnallm_extended_mask_patch")


class TestPretrainedConfigLegacyDefaults:
    """Contract for the restored 4.x ``PretrainedConfig`` legacy defaults.

    transformers 5.x removed the 4.x instance defaults ``is_decoder=False``
    and ``add_cross_attention=False`` (4.x set both in ``__init__``), which
    4.x-era ``trust_remote_code`` checkpoints read at model build (the remote
    ``modeling_esm.py`` lines 335/584-585 of the NER/promoter mirrors).  The
    shim restores READ behavior through a ``PretrainedConfig.__getattr__``
    over a CLOSED default map -- explicitly set values, unknown attributes
    and deepcopy round-trips behave exactly as before.  This supersedes the
    05-04 D-07 rung termination per the owner instruction of 2026-10-02
    (fix all non-gated census failures now).
    """

    def test_bare_config_answers_legacy_defaults(self):
        """After apply_patches() a fresh config answers both legacy defaults with False."""
        config = PretrainedConfig()
        missing = [
            name for name in ("is_decoder", "add_cross_attention") if not hasattr(config, name)
        ]
        assert not missing, f"legacy config defaults missing after apply_patches(): {missing}"
        assert config.is_decoder is False
        assert config.add_cross_attention is False

    def test_explicitly_set_value_wins_over_default(self):
        """An explicitly set instance value shadows the restored default."""
        config = PretrainedConfig()
        config.is_decoder = True
        config.add_cross_attention = True

        apply_patches()  # a repeat application must not shadow explicit instance values

        assert config.is_decoder is True
        assert config.add_cross_attention is True

    def test_unknown_attribute_still_raises_attribute_error(self):
        """Attributes outside the closed map keep raising plain AttributeError."""
        config = PretrainedConfig()
        with pytest.raises(AttributeError, match="has no attribute"):
            config.definitely_not_a_real_attribute  # ruff: ignore[useless-expression]

    def test_deepcopy_round_trips_the_defaults(self):
        """A deepcopied config still answers both legacy defaults."""
        clone = copy.deepcopy(PretrainedConfig())
        assert clone.is_decoder is False
        assert clone.add_cross_attention is False

    def test_repeat_apply_is_idempotent(self):
        """A second apply_patches() never rebinds the class __getattr__."""
        major = int(transformers.__version__.split(".")[0])
        before = vars(PretrainedConfig).get("__getattr__")

        apply_patches()

        assert vars(PretrainedConfig).get("__getattr__") is before, "no rebind may happen"
        if major >= 5:
            assert getattr(PretrainedConfig, "_dnallm_config_legacy_defaults_patch", False) is True
        else:
            assert not hasattr(PretrainedConfig, "_dnallm_config_legacy_defaults_patch"), (
                "4.x native defaults must not even set the sentinel"
            )

    def test_patch_skips_when_defaults_resolve_natively(self, monkeypatch):
        """A config class resolving both names natively (the 4.x shape) is left untouched."""

        class _NativeConfig:
            is_decoder = False

            add_cross_attention = False

        monkeypatch.setattr(transformers.configuration_utils, "PretrainedConfig", _NativeConfig)
        transformers_compat._patch_pretrained_config_legacy_defaults()

        assert "__getattr__" not in vars(_NativeConfig)
        assert not hasattr(_NativeConfig, "_dnallm_config_legacy_defaults_patch")

    def test_patch_never_overwrites_existing_getattr(self, monkeypatch):
        """A config class already carrying its own __getattr__ keeps it verbatim."""

        def _existing_getattr(self, name):
            raise AttributeError(name)

        class _GuardedConfig:
            pass

        _GuardedConfig.__getattr__ = _existing_getattr
        monkeypatch.setattr(transformers.configuration_utils, "PretrainedConfig", _GuardedConfig)
        transformers_compat._patch_pretrained_config_legacy_defaults()

        assert vars(_GuardedConfig)["__getattr__"] is _existing_getattr
        assert not hasattr(_GuardedConfig, "_dnallm_config_legacy_defaults_patch")

    def test_patch_installs_on_config_missing_defaults(self, monkeypatch):
        """A bare config class missing both names receives the closed-map __getattr__."""

        class _BareConfig:
            pass

        monkeypatch.setattr(transformers.configuration_utils, "PretrainedConfig", _BareConfig)
        transformers_compat._patch_pretrained_config_legacy_defaults()

        probe = _BareConfig()
        assert probe.is_decoder is False
        assert probe.add_cross_attention is False
        with pytest.raises(AttributeError, match="has no attribute"):
            probe.something_else  # ruff: ignore[useless-expression]
        assert _BareConfig._dnallm_config_legacy_defaults_patch is True


class TestVendoredMambaCache:
    """Contract for the vendored v4.49.0 ``MambaCache`` re-attached on transformers 5.x.

    transformers 5.x removed ``MambaCache`` from ``cache_utils`` entirely, but
    4.x-era ``trust_remote_code`` checkpoints (the tRNADetector remote
    ``modeling_mamba.py`` line 27) import it from there and construct it with
    the 4.x signature ``MambaCache(config, batch_size, device=..., dtype=...)``.
    Covers construction shapes on a stub config, the update/reset semantics the
    remote forward calls, and the live-module attachment contract (absence
    gating per name, idempotent sentinel, version-aware identity).
    """

    @staticmethod
    def _stub_config() -> types.SimpleNamespace:
        """Two-layer mamba-shaped config with the attributes MambaCache reads."""
        return types.SimpleNamespace(
            num_hidden_layers=2, intermediate_size=8, conv_kernel=4, state_size=16
        )

    def test_cache_utils_exposes_mamba_cache_after_patches(self):
        """Import-time apply_patches() exposes MambaCache; on 5.x it IS the vendored class."""
        major = int(transformers.__version__.split(".")[0])
        assert hasattr(transformers.cache_utils, "MambaCache"), (
            "transformers.cache_utils must expose MambaCache after apply_patches()"
        )
        if major >= 5:
            assert transformers.cache_utils.MambaCache is transformers_compat._MambaCache, (
                "5.x attachment must be the vendored class"
            )

    def test_construction_shapes_dtype_and_device(self):
        """batch_size=3 on a 2-layer stub yields per-layer [3, 8, 4] / [3, 8, 16] states."""
        cache = transformers_compat._MambaCache(
            self._stub_config(), batch_size=3, device="cpu", dtype=torch.float32
        )
        assert cache.max_batch_size == 3
        assert cache.intermediate_size == 8
        assert cache.ssm_state_size == 16
        assert cache.conv_kernel_size == 4
        assert cache.device == torch.device("cpu")
        assert cache.dtype == torch.float32
        assert len(cache.conv_states) == 2
        assert len(cache.ssm_states) == 2
        for conv_state in cache.conv_states:
            assert tuple(conv_state.shape) == (3, 8, 4)
            assert conv_state.dtype == torch.float32
            assert conv_state.device.type == "cpu"
        for ssm_state in cache.ssm_states:
            assert tuple(ssm_state.shape) == (3, 8, 16)
            assert ssm_state.dtype == torch.float32
            assert ssm_state.device.type == "cpu"

    def test_update_conv_state_rolls_and_ssm_state_replaces(self):
        """update_conv_state writes the new column after rolling; ssm update replaces; reset zeros."""
        cache = transformers_compat._MambaCache(
            self._stub_config(), max_batch_size=2, device="cpu", dtype=torch.float32
        )
        out = cache.update_conv_state(0, torch.ones(2, 8, 1), torch.tensor([0]))
        assert torch.equal(out[:, :, 0], torch.ones(2, 8)), "new column must land at position 0"
        assert torch.equal(out[:, :, 1:], torch.zeros(2, 8, 3)), "rolled-away columns stay zero"

        ssm_state = torch.full((2, 8, 16), 2.0)
        assert cache.update_ssm_state(1, ssm_state) is cache.ssm_states[1]
        assert torch.equal(cache.ssm_states[1], ssm_state)

        cache.reset()
        assert torch.equal(cache.conv_states[0], torch.zeros(2, 8, 4))
        assert torch.equal(cache.ssm_states[1], torch.zeros(2, 8, 16))

    def test_deprecated_batch_size_argument_maps_to_max_batch_size(self):
        """The 4.x positional batch_size construction path keeps working (remote signature)."""
        cache = transformers_compat._MambaCache(self._stub_config(), batch_size=5, device="cpu")
        assert cache.max_batch_size == 5
        assert cache.batch_size == 5  # deprecated property mirrors max_batch_size

    def test_repeat_apply_is_idempotent(self):
        """A second apply_patches() leaves the attached class identity unchanged."""
        before = getattr(transformers.cache_utils, "MambaCache", None)

        apply_patches()

        assert transformers.cache_utils.MambaCache is before, "no re-attach may happen"

    def test_patch_leaves_native_mamba_cache_untouched(self, monkeypatch):
        """A cache_utils already exposing MambaCache natively (4.x shape) is left alone."""
        parent = sys.modules["transformers"]
        fake_cache_utils = types.ModuleType("transformers.cache_utils")
        native = object()
        fake_cache_utils.MambaCache = native
        monkeypatch.setattr(parent, "cache_utils", fake_cache_utils, raising=False)

        transformers_compat._patch_mamba_cache()

        assert fake_cache_utils.MambaCache is native
        assert not hasattr(fake_cache_utils, "_dnallm_mamba_cache_patch")

    def test_patch_attaches_vendored_cache_where_absent(self, monkeypatch):
        """A cache_utils lacking the name (5.x shape) receives the vendored class + sentinel."""
        parent = sys.modules["transformers"]
        fake_cache_utils = types.ModuleType("transformers.cache_utils")
        monkeypatch.setattr(parent, "cache_utils", fake_cache_utils, raising=False)

        transformers_compat._patch_mamba_cache()

        assert fake_cache_utils.MambaCache is transformers_compat._MambaCache
        assert fake_cache_utils._dnallm_mamba_cache_patch is True


class TestDebertaVocabDictNormalization:
    """Contract for the dict-vocab normalization on ``DebertaV2Tokenizer``.

    transformers 5.17 hands ``DebertaV2Tokenizer`` a ``vocab`` dict
    (``{token: score}``, insertion-ordered; proven live with the
    plant-dnabert-BPE checkpoint) through ``convert_to_native_format`` --
    the exact hook ``from_pretrained`` calls immediately before
    ``cls(*init_inputs, **init_kwargs)`` -- while the 5.x Unigram backend
    only accepts a sequence of ``(token, score)`` pairs.  The shim wraps
    that hook and normalizes dict vocabularies to ``list(vocab.items())``
    (insertion order = rank order; tuple equality keeps the
    ``vocab.index((str(unk_token), 0.0))`` lookup working with int scores).
    On transformers 4.x the hook does not exist and the patch no-ops.
    """

    VOCAB: ClassVar[dict[str, float]] = {"[PAD]": 0, "A": 1, "T": 2, "[UNK]": 0}

    def test_hook_normalizes_dict_vocab_to_ordered_pairs(self):
        """A dict vocab comes out of the hook as the insertion-ordered pair list."""
        converted = DebertaV2Tokenizer.convert_to_native_format(vocab=dict(self.VOCAB))
        assert converted["vocab"] == [("[PAD]", 0), ("A", 1), ("T", 2), ("[UNK]", 0)], (
            "dict vocab must be normalized to list(vocab.items()) in insertion order"
        )

    def test_tokenizer_builds_from_dict_vocab_and_tokenizes(self):
        """The normalized vocab builds a working tokenizer with rank order preserved."""
        converted = DebertaV2Tokenizer.convert_to_native_format(vocab=dict(self.VOCAB))
        tokenizer = DebertaV2Tokenizer(**converted)

        tokens = tokenizer.tokenize("AT")
        assert isinstance(tokens, list), "tokenization must work without TypeError"
        assert tokens, "tokenize('AT') must produce real pieces"
        ids = [tokenizer.convert_tokens_to_ids(token) for token in ["[PAD]", "A", "T", "[UNK]"]]
        assert ids == [0, 1, 2, 3], "dict insertion order must be preserved as rank order"

    def test_list_vocab_passes_through_unchanged(self):
        """A pair-list vocab already in native form is never reordered."""
        pairs = [("[PAD]", 0.0), ("A", 1.0), ("T", 2.0)]
        converted = DebertaV2Tokenizer.convert_to_native_format(vocab=list(pairs))
        assert converted["vocab"] == pairs

    def test_repeat_apply_does_not_wrap_twice(self):
        """A second apply_patches() keeps the same bound hook (sentinel idempotence)."""
        before = vars(DebertaV2Tokenizer).get("convert_to_native_format")

        apply_patches()

        assert vars(DebertaV2Tokenizer).get("convert_to_native_format") is before
        converted = DebertaV2Tokenizer.convert_to_native_format(vocab=dict(self.VOCAB))
        assert isinstance(converted["vocab"], list)

    def test_patch_noops_when_hook_absent(self, monkeypatch):
        """A DebertaV2Tokenizer without the hook (the 4.x shape) is left untouched."""

        class _HooklessTokenizer:
            """Stand-in for the 4.x sentencepiece-based DebertaV2Tokenizer."""

        monkeypatch.setattr(
            transformers.models.deberta_v2.tokenization_deberta_v2,
            "DebertaV2Tokenizer",
            _HooklessTokenizer,
        )
        transformers_compat._patch_deberta_vocab_dict()

        assert not hasattr(_HooklessTokenizer, "convert_to_native_format")
        assert not hasattr(_HooklessTokenizer, "_dnallm_deberta_vocab_patch")


class TestVendoredGetHeadMask:
    """Contract for the vendored v4.49.0 ``get_head_mask`` re-attached on transformers 5.x.

    transformers 5.x removed ``get_head_mask`` (and its private
    ``_convert_head_mask_to_5d`` helper) from ``ModuleUtilsMixin`` /
    ``PreTrainedModel``, but 4.x-era trust_remote_code checkpoints call it as
    a METHOD inside ``EsmModel.forward``
    (``self.get_head_mask(head_mask, self.config.num_hidden_layers)`` --
    five live call sites across the cached zhangtaolab/InstaDeepAI remote
    modeling files). Covers the None / 1D / 2D mask semantics and the
    live-class attachment contract (absence gating per name, idempotent
    sentinel, version-aware identity), mirroring
    :class:`TestGetExtendedAttentionMask`.
    """

    @staticmethod
    def _receiver() -> type:
        """A minimal receiver class exposing the two attributes the vendored method reads."""

        class _HeadMaskReceiver:
            dtype = torch.float32
            _convert_head_mask_to_5d = transformers_compat._convert_head_mask_to_5d

        return _HeadMaskReceiver

    def test_none_mask_returns_none_per_layer(self):
        """A None head mask expands to [None] * num_hidden_layers (the forward default)."""
        result = transformers_compat._get_head_mask(self._receiver()(), None, 3)
        assert result == [None, None, None]

    def test_1d_mask_expands_to_broadcastable_5d(self):
        """A [num_heads] mask expands to (layers, 1, heads, 1, 1) in the receiver dtype."""
        mask = torch.tensor([1.0, 0.0, 1.0])
        result = transformers_compat._get_head_mask(self._receiver()(), mask, 2)
        assert isinstance(result, torch.Tensor)
        assert tuple(result.shape) == (2, 1, 3, 1, 1)
        assert result.dtype == torch.float32
        assert float(result[0, 0, 1, 0, 0]) == 0.0, "masked head must carry the 0.0 entry"

    def test_2d_mask_expands_per_layer(self):
        """A [layers x heads] mask expands to (layers, 1, heads, 1, 1)."""
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = transformers_compat._get_head_mask(self._receiver()(), mask, 2)
        assert tuple(result.shape) == (2, 1, 2, 1, 1)
        assert float(result[0, 0, 1, 0, 0]) == 0.0
        assert float(result[1, 0, 0, 0, 0]) == 0.0

    def test_is_attention_chunked_adds_trailing_dim(self):
        """is_attention_chunked=True unsqueezes one more trailing dimension."""
        mask = torch.tensor([1.0, 1.0])
        result = transformers_compat._get_head_mask(
            self._receiver()(), mask, 2, is_attention_chunked=True
        )
        assert tuple(result.shape) == (2, 1, 2, 1, 1, 1)

    def test_wrong_dimensionality_raises_assertion_error(self):
        """A 3D mask hits upstream's head_mask.dim != 5 AssertionError."""
        with pytest.raises(AssertionError, match=r"head_mask\.dim != 5"):
            transformers_compat._get_head_mask(self._receiver()(), torch.zeros(2, 2, 2), 2)

    def test_live_class_attachment_identity_is_version_agnostic(self):
        """On transformers 5.x the live method IS the vendored one; on 4.x it is NOT."""
        major = int(transformers.__version__.split(".")[0])
        assert callable(PreTrainedModel.get_head_mask)
        if major >= 5:
            assert PreTrainedModel.get_head_mask is transformers_compat._get_head_mask, (
                "5.x attachment must be the vendored function"
            )
        else:
            assert PreTrainedModel.get_head_mask is not transformers_compat._get_head_mask, (
                "4.x no-op gate must leave upstream's native method in place"
            )

    def test_apply_patches_rebind_is_idempotent_with_sentinel(self):
        """A second apply_patches() leaves the bound method identical and the sentinel stable."""
        major = int(transformers.__version__.split(".")[0])
        before = PreTrainedModel.get_head_mask
        apply_patches()
        assert PreTrainedModel.get_head_mask is before, "no rebind may happen"
        if major >= 5:
            assert PreTrainedModel._dnallm_head_mask_patch is True
        else:
            assert not hasattr(PreTrainedModel, "_dnallm_head_mask_patch"), (
                "4.x absence gate must not even set the sentinel"
            )

    def test_patch_attaches_only_where_class_lacks_the_method(self, monkeypatch):
        """A bare stand-in class receives the vendored method and sentinel; a class already
        exposing a native method is left untouched and gets no sentinel."""

        class _BareModel:
            """Stand-in for a PreTrainedModel without get_head_mask."""

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _BareModel)
        transformers_compat._patch_get_head_mask()
        assert _BareModel.get_head_mask is transformers_compat._get_head_mask
        assert _BareModel._dnallm_head_mask_patch is True

        class _NativeModel:
            """Stand-in whose class already carries both native methods (the 4.x shape)."""

            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                """Native placeholder that must never be overwritten."""
                return head_mask

            def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
                """Native placeholder helper that must never be overwritten."""
                return head_mask

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _NativeModel)
        transformers_compat._patch_get_head_mask()
        assert _NativeModel.get_head_mask is not transformers_compat._get_head_mask
        assert _NativeModel._convert_head_mask_to_5d is not (
            transformers_compat._convert_head_mask_to_5d
        )
        assert not hasattr(_NativeModel, "_dnallm_head_mask_patch")


class TestLegacyInitWeightsBookkeeping:
    """Contract for the post_init bookkeeping restored behind bare ``init_weights()``.

    transformers 5.x moved the tied-weights/parallel-plan bookkeeping into
    ``PreTrainedModel.post_init`` (which ends by calling ``init_weights``),
    but 4.x-era trust_remote_code checkpoints end their ``__init__`` with the
    bare ``self.init_weights()`` entry (the remote ``EsmForMaskedLM`` /
    ``EsmForTokenClassification`` of the zhangtaolab NER and tRNAPointer
    mirrors -- 3 call sites each), skipping the bookkeeping entirely;
    ``from_pretrained`` then crashes at
    ``_move_missing_keys_from_meta_to_device`` reading
    ``self.all_tied_weights_keys``. The shim wraps ``init_weights`` so a
    receiver missing the bookkeeping first runs the real ``post_init``
    (which sets it, then calls this same wrapped ``init_weights`` again --
    bounded depth 2, never recursive). On transformers 4.x, where
    ``post_init`` computes no such attribute, the wrapper is never
    installed (probed once at patch time).
    """

    @staticmethod
    def _legacy_init_model(config=None):
        """Build a remote-shaped model whose __init__ ends with bare init_weights()."""

        class _LegacyInitModel(PreTrainedModel):
            """Remote-code shape: submodules built, then bare init_weights() (no post_init)."""

            def __init__(self, config):
                super().__init__(config)
                self.linear = torch.nn.Linear(4, 2)
                self.init_weights()

        return _LegacyInitModel(config or PretrainedConfig())

    def test_bare_init_weights_ends_with_post_init_bookkeeping(self):
        """A bare init_weights() receiver gains all_tied_weights_keys (a dict)."""
        model = self._legacy_init_model()
        assert isinstance(model.all_tied_weights_keys, dict), (
            "the post_init bookkeeping must run behind the bare init_weights() entry"
        )

    def test_bare_init_weights_still_initializes_weights(self):
        """The original init_weights body still runs exactly once (finite initialized weights)."""
        model = self._legacy_init_model()
        assert torch.isfinite(model.linear.weight).all(), "weights must be initialized"
        # a repeat explicit call takes the original path directly (attribute now present)
        model.init_weights()
        assert torch.isfinite(model.linear.weight).all()

    def test_native_post_init_flow_is_unchanged(self):
        """A model ending with post_init() (the 5.x-native shape) behaves identically."""

        class _NativeInitModel(PreTrainedModel):
            """Modern shape: post_init() at the end (bookkeeping + init_weights)."""

            def __init__(self, config):
                super().__init__(config)
                self.linear = torch.nn.Linear(4, 2)
                self.post_init()

        model = _NativeInitModel(PretrainedConfig())
        assert isinstance(model.all_tied_weights_keys, dict)
        assert torch.isfinite(model.linear.weight).all()

    def test_repeat_apply_is_idempotent(self):
        """A second apply_patches() never re-wraps init_weights."""
        before = vars(PreTrainedModel).get("init_weights")
        apply_patches()
        assert vars(PreTrainedModel).get("init_weights") is before

    def test_wrapper_delegates_via_post_init_exactly_once(self, monkeypatch):
        """The delegation path runs the REAL post_init exactly once per bare entry."""
        calls = []
        original_post_init = PreTrainedModel.post_init

        def counting_post_init(self):
            calls.append(type(self).__name__)
            return original_post_init(self)

        monkeypatch.setattr(PreTrainedModel, "post_init", counting_post_init)
        model = self._legacy_init_model()
        assert calls.count("_LegacyInitModel") == 1, (
            f"post_init must run exactly once for the bare entry, got {calls}"
        )
        assert isinstance(model.all_tied_weights_keys, dict)

    def test_patch_noops_when_post_init_computes_nothing(self, monkeypatch):
        """A transformers whose post_init computes no bookkeeping (the 4.x shape)
        never gets the wrapper installed (probe gate), so bare init_weights stays native."""

        class _FourXStyleModel:
            """Stand-in for a 4.x PreTrainedModel (no all_tied_weights_keys anywhere)."""

            def init_weights(self):
                """Native 4.x body that must stay bound verbatim."""
                self.marker = "native"

            def post_init(self):
                """4.x post_init computes no 5.x bookkeeping."""

        monkeypatch.setattr(transformers.modeling_utils, "PreTrainedModel", _FourXStyleModel)
        transformers_compat._patch_legacy_init_weights_bookkeeping()

        probe = _FourXStyleModel()
        probe.init_weights()
        assert probe.marker == "native", "4.x-shape classes must keep their native init_weights"
        assert not hasattr(_FourXStyleModel, "_dnallm_init_weights_patch")


def _collect_patch_installers():
    """Collect the sorted names of every ``_patch_*`` installer in the module.

    Collection from ``vars(transformers_compat)`` is dynamic on purpose: a
    future installer lands inside the absence-contract parametrization
    automatically, so a forgotten guard fails loudly in CI instead of
    surviving to the next code review (the exact IN-01 failure mode).

    Returns:
        A sorted list of the module's installer function names.
    """
    return sorted(
        name
        for name, value in vars(transformers_compat).items()
        if name.startswith("_patch_") and callable(value)
    )


EXPECTED_PATCH_INSTALLERS = frozenset({
    "_patch_device_type_query",
    "_patch_get_parameter_or_buffer",
    "_patch_initialize_weights_for_quantized_missing",
    "_patch_remote_code_pruning_helpers",
    "_patch_get_extended_attention_mask",
    "_patch_pretrained_config_legacy_defaults",
    "_patch_mamba_cache",
    "_patch_deberta_vocab_dict",
    "_patch_numpy_fromstring",
    "_patch_get_head_mask",
    "_patch_legacy_init_weights_bookkeeping",
})


class TestTransformersAbsenceContract:
    """Every installer degrades to a no-op when transformers is unimportable.

    The module contract at transformers_compat.py:7-11 promises that every
    patch no-ops when the relevant libraries are not installed, so importing
    DNALLM never breaks an otherwise working environment. ``apply_patches()``
    runs eagerly from the module body (reached at ``import dnallm`` through
    dnallm/utils/__init__.py), so one unguarded installer crashes the whole
    package import in a stripped environment or after a future transformers
    renames a submodule (IN-01, 05-REVIEW.md:234).

    The mechanism: a ``None`` entry for ``"transformers"`` in ``sys.modules``
    makes every import form the installers use raise ``ModuleNotFoundError``
    via parent-first resolution (live-probed at planning time) --
    ``import transformers.<submodule>`` fails with "'transformers' is not a
    package" and ``from transformers... import ...`` fails with "import of
    transformers halted; None in sys.modules". The guards' broad
    ``except Exception`` returning ``None`` is the behavior this contract
    requires. monkeypatch restores the sys.modules entry on teardown, and
    every installer is sentinel-gated, so the live patched classes on
    transformers 5.17 are never disturbed.
    """

    @pytest.mark.parametrize("installer_name", _collect_patch_installers())
    def test_installer_noops_when_transformers_unimportable(self, installer_name, monkeypatch):
        """Under a None transformers sys.modules entry each installer returns None.

        Each installer must honor the absence contract individually: the
        try-import guard turns the ModuleNotFoundError into a plain ``None``
        return, never a raise out of ``import dnallm``.
        """
        # None in sys.modules makes every transformers import form raise
        # ModuleNotFoundError (see class docstring).
        monkeypatch.setitem(sys.modules, "transformers", None)
        assert getattr(transformers_compat, installer_name)() is None

    def test_installer_roster_is_pinned(self):
        """The installer roster changes only through conscious extension here.

        Adding or renaming a ``_patch_*`` installer must fail this pin until
        the roster (and, through dynamic collection, the absence contract
        itself) is deliberately extended -- a silent roster drift is how an
        unguarded installer would sneak in.
        """
        assert set(_collect_patch_installers()) == EXPECTED_PATCH_INSTALLERS

    def test_apply_patches_survives_unimportable_transformers(self, monkeypatch):
        """apply_patches() returns None, never raises, when transformers is absent.

        ``apply_patches()`` executes eagerly at ``import dnallm`` time, so a
        single unguarded installer (on the current code the first unguarded
        one, ``_patch_pretrained_config_legacy_defaults``, is the fifth call
        in its body) crashes the whole package import instead of degrading to
        stock-transformers behavior.
        """
        monkeypatch.setitem(sys.modules, "transformers", None)
        assert apply_patches() is None
