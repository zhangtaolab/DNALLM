"""Compatibility shims for third-party library bugs.

This module centralizes small, defensive patches around bugs in
dependencies (e.g. transformers) that affect DNALLM's supported
workflows but are not fixed in the version range DNALLM supports.

All patches are gated so that they only change behavior on the exact
code paths that crash in unpatched transformers; every patch also
no-ops when the target method is absent (older transformers) or the
relevant libraries (bitsandbytes) are not installed, so importing
DNALLM never breaks an otherwise working environment.
"""

import torch


def _iter_uninitialized_quantized_weights(model):
    """Find packed (non-fp) bitsandbytes 4-bit params not yet HF-initialized.

    Returns ``(candidates, marked)`` where ``candidates`` is a list of
    ``(module, weight)`` pairs and ``marked`` indicates whether any
    parameter in the model carries an ``_is_hf_initialized`` flag (the
    signature of the missing-keys initialization flow).
    """
    candidates = []
    marked = False
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if getattr(weight, "_is_hf_initialized", None) is not None:
            marked = True
        if (
            type(weight).__name__ == "Params4bit"
            and not getattr(weight, "_is_hf_initialized", False)
            and getattr(weight, "quant_state", None) is not None
            and not weight.is_floating_point()
        ):
            candidates.append((module, weight))
    return candidates, marked


def _swap_to_fp32(candidates):
    """Temporarily replace packed 4-bit params with dequantized fp tensors."""
    import bitsandbytes as bnb

    swapped = []
    for module, weight in candidates:
        fp_value = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        module._parameters["weight"] = torch.nn.Parameter(fp_value, requires_grad=False)
        swapped.append((module, weight))
    return swapped


def _restore_quantized(swapped):
    """Re-quantize the fp tensors written by initialization and restore Params4bit."""
    import bitsandbytes as bnb

    for module, weight in swapped:
        fp_value = module._parameters.pop("weight").data
        packed, quant_state = bnb.functional.quantize_4bit(
            fp_value,
            blocksize=weight.blocksize,
            quant_type=weight.quant_type,
            compress_statistics=weight.compress_statistics,
            quant_storage=getattr(weight, "quant_storage", torch.uint8),
        )
        weight.data = packed
        weight.quant_state = quant_state
        module._parameters["weight"] = weight
        weight._is_hf_initialized = True


def _patch_get_parameter_or_buffer():
    """Make ``get_parameter_or_buffer`` tolerate nested quantizer keys.

    Returns a proxy of the parent tensor for nested quantization-statistic
    keys (e.g. ``...weight.absmax``, ``...weight.nested_quant_map``) so
    read-only consumers (e.g. ``caching_allocator_warmup``) keep working
    and ``_is_hf_initialized`` marks on them are dropped instead of
    crashing. Semantics for every other key are unchanged: the patched
    behavior only activates for keys that the original implementation
    already rejects with ``AttributeError``.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers not installed
        return

    # The method only exists in transformers >= 4.50; older supported
    # versions (dnallm allows >= 4.49) need no patch here.
    if not hasattr(PreTrainedModel, "get_parameter_or_buffer"):
        return

    if getattr(PreTrainedModel, "_dnallm_quant_key_patch", False):
        return

    original = PreTrainedModel.get_parameter_or_buffer

    def get_parameter_or_buffer(self, target):
        try:
            return original(self, target)
        except AttributeError:
            if isinstance(target, str):
                # Nested quantizer state-dict entries (bitsandbytes exposes
                # absmax/quant_map/etc. under "<param>.<stat>") are tensors
                # that live inside a parameter, not addressable parameters,
                # buffers or modules themselves.
                parts = target.split(".")
                for i in range(1, min(3, len(parts) - 1) + 1):
                    parent = ".".join(parts[:-i])
                    try:
                        tensor = self.get_parameter(parent)
                    except AttributeError:
                        try:
                            tensor = self.get_buffer(parent)
                        except AttributeError:
                            continue
                    # Proxy the parent: callers that only read get a real
                    # tensor to inspect, while `_is_hf_initialized` marks
                    # are dropped so that genuinely missing parent params
                    # still get initialized.
                    return _QuantStatProxy(tensor)
            raise

    PreTrainedModel.get_parameter_or_buffer = get_parameter_or_buffer  # type: ignore[method-assign]
    PreTrainedModel._dnallm_quant_key_patch = True  # type: ignore[attr-defined]


def _patch_initialize_weights_for_quantized_missing():
    """Let the missing-keys init flow handle bitsandbytes 4-bit params.

    In transformers 4.52-4.57, loading a quantized model whose checkpoint
    is missing some Linear weights (e.g. adding a classification head to a
    base model for QLoRA) ends with ``_initialize_missing_keys`` ->
    ``initialize_weights()``, where the model-specific ``_init_weights``
    calls ``normal_()`` on the packed uint8 weight of ``Linear4bit`` and
    crashes with ``NotImplementedError: normal_kernel ... not implemented
    for 'Byte'``. Fixed upstream in transformers v5.

    Workaround: during the missing-keys init flow only, temporarily
    dequantize the not-yet-initialized packed params to fp tensors so the
    regular init writes meaningful values, then re-quantize and write the
    values back into the original ``Params4bit`` objects.

    Versions <= 4.51 do not quantize missing weights at all (they stay
    fp32), so the crash cannot occur there; this patch is skipped when
    ``initialize_weights`` is absent. Note we deliberately do NOT hook
    ``_initialize_missing_keys`` on 4.50/4.51: in flows like
    ``ignore_mismatched_sizes=True`` that would re-initialize every module,
    swapping packed params would silently destroy loaded checkpoint values
    instead of crashing loudly like stock transformers.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers not installed
        return

    # `initialize_weights` only exists in transformers >= 4.52; on older
    # versions the crash this patch prevents cannot occur (missing weights
    # are not quantized there), so skip cleanly.
    if not hasattr(PreTrainedModel, "initialize_weights"):
        return

    if getattr(PreTrainedModel, "_dnallm_quant_init_patch", False):
        return

    original = PreTrainedModel.initialize_weights

    def initialize_weights(self):
        candidates, marked = _iter_uninitialized_quantized_weights(self)

        # Only intercept the missing-keys init flow (evidenced by the
        # `_is_hf_initialized` marks). In any other context (e.g. building
        # a fresh model with `from_config`) leave everything untouched.
        if not marked or not candidates:
            return original(self)

        try:
            import bitsandbytes  # noqa: F401
        except Exception:  # pragma: no cover - bitsandbytes not installed
            return original(self)

        swapped = _swap_to_fp32(candidates)
        try:
            original(self)
        finally:
            _restore_quantized(swapped)

    PreTrainedModel.initialize_weights = initialize_weights  # type: ignore[method-assign]
    PreTrainedModel._dnallm_quant_init_patch = True  # type: ignore[attr-defined]


class _QuantStatProxy:
    """Transparent proxy for nested quantizer state-dict entries.

    Forwards every attribute access to the underlying parent tensor (so
    read-only consumers keep working) while silently dropping
    ``_is_hf_initialized`` marks, which must not be set on the parent
    parameter through a nested quantizer-statistic key.
    """

    def __init__(self, tensor):
        object.__setattr__(self, "_tensor", tensor)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tensor"), name)

    def __setattr__(self, name, value):
        if name == "_is_hf_initialized":
            return
        setattr(object.__getattribute__(self, "_tensor"), name, value)


def apply_patches():
    """Apply all compatibility patches. Safe to call multiple times."""
    _patch_get_parameter_or_buffer()
    _patch_initialize_weights_for_quantized_missing()


# Apply patches on module import so they are active before any
# transformers model is loaded through DNALLM.
apply_patches()
