"""Behavior-contract tests for the transformers compatibility shim (dnallm.utils.transformers_compat).

The patches under test were applied to the real ``transformers.PreTrainedModel``
at import time. These tests assert the CONTRACT of the patched live class —
idempotency, proxy semantics, candidate classification, and the swap/restore
flow — and never remove or undo the class patches (other tests in the suite
depend on the patched state).
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

import bitsandbytes.functional as bnb_functional
import pytest
import torch
import transformers.modeling_utils
from transformers.modeling_utils import PreTrainedModel

from dnallm.utils.transformers_compat import (
    _QuantStatProxy,
    _find_pruneable_heads_and_indices,
    _iter_uninitialized_quantized_weights,
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
    pruning shapes/values) and the live-module attachment contract on
    ``transformers.modeling_utils`` — presence, idempotence, and version
    awareness: on transformers 5.x the exposed helpers ARE the vendored
    functions; on 4.x they are upstream natives and the patch is an
    absence-gated no-op (the identity assertions invert rather than skip,
    so the file collects and passes cleanly across the whole CI matrix).
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
