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


# The two pruning helpers below are vendored with semantics and docstrings kept
# verbatim from the upstream transformers reference implementation: tag v4.49.0,
# file src/transformers/pytorch_utils.py. transformers 5.x removed both helpers,
# but 4.x-era trust_remote_code checkpoints (e.g. the remote modeling_esm.py of
# zhangtaolab/nucleotide-transformer-v2-100m-promoter) still import them from
# transformers.modeling_utils; the patch below re-attaches these implementations
# under their upstream names. The only adaptation is import locality into this
# module (modern builtin-generic annotations; behavior is unchanged).


def _find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    # Convert to set and remove already pruned heads (upstream rebinding of the
    # `heads` parameter is renamed here only so the annotation flow stays sound;
    # the returned set and the mask arithmetic are unchanged).
    pruned_heads = set(heads) - already_pruned_heads
    for head in pruned_heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.Tensor = torch.arange(len(mask))[mask].long()
    return pruned_heads, index


def _prune_linear_layer(
    layer: torch.nn.Linear, index: torch.Tensor, dim: int = 0
) -> torch.nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    w = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(w.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


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
            import bitsandbytes  # ruff: ignore[unused-import]
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


def _patch_remote_code_pruning_helpers():
    """Re-attach the 4.x pruning helpers removed from transformers 5.x.

    transformers 5.x removed ``find_pruneable_heads_and_indices`` and
    ``prune_linear_layer`` from ``transformers.modeling_utils`` (and from
    ``pytorch_utils``), but 4.x-era ``trust_remote_code`` checkpoints (e.g.
    the remote ``modeling_esm.py`` of the nucleotide-transformer-v2 promoter
    mirror) still import both names from ``transformers.modeling_utils``.
    This patch attaches the vendored v4.49.0 implementations under their
    upstream names so remote code keeps resolving them.

    On transformers 4.x the names already exist and the patch no-ops
    (absence-gated per decision D-07); a module-level sentinel keeps repeat
    calls idempotent.
    """
    try:
        import transformers.modeling_utils
    except Exception:  # pragma: no cover - transformers not installed
        return

    # transformers 4.x still exposes the helpers natively - leave it untouched.
    if hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
        return

    if getattr(transformers.modeling_utils, "_dnallm_remote_code_pruning_patch", False):
        return

    transformers.modeling_utils.find_pruneable_heads_and_indices = (  # type: ignore[attr-defined]
        _find_pruneable_heads_and_indices
    )
    transformers.modeling_utils.prune_linear_layer = _prune_linear_layer  # type: ignore[attr-defined]
    transformers.modeling_utils._dnallm_remote_code_pruning_patch = True  # type: ignore[attr-defined]


def apply_patches():
    """Apply all compatibility patches. Safe to call multiple times."""
    _patch_get_parameter_or_buffer()
    _patch_initialize_weights_for_quantized_missing()
    _patch_remote_code_pruning_helpers()


# Apply patches on module import so they are active before any
# transformers model is loaded through DNALLM.
apply_patches()
