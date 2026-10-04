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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from transformers import PreTrainedConfig  # ty: ignore[unresolved-import]  # lazy export, resolves live
    from transformers import PreTrainedModel  # ty: ignore[unresolved-import]  # lazy export, resolves live


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
    import bitsandbytes.functional  # explicit submodule: makes bnb.functional below statically resolved

    swapped = []
    for module, weight in candidates:
        fp_value = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        module._parameters["weight"] = torch.nn.Parameter(fp_value, requires_grad=False)
        swapped.append((module, weight))
    return swapped


def _restore_quantized(swapped):
    """Re-quantize the fp tensors written by initialization and restore Params4bit."""
    import bitsandbytes as bnb
    import bitsandbytes.functional  # explicit submodule: makes bnb.functional below statically resolved

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
# file src/transformers/pytorch_utils.py. transformers 5.x removed both helpers
# from modeling_utils (and find_pruneable_heads_and_indices from pytorch_utils,
# which keeps its own prune_linear_layer), but 4.x-era trust_remote_code
# checkpoints (e.g. the remote modeling_esm.py of
# zhangtaolab/nucleotide-transformer-v2-100m-promoter) still import them from
# transformers.modeling_utils and transformers.pytorch_utils; the patch below
# re-attaches these implementations under their upstream names on both modules,
# per name only where the module does not already expose it. The only
# adaptation is import locality into this module (modern builtin-generic
# annotations; behavior is unchanged).


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


def _attach_remote_code_pruning_helpers(module: object) -> None:
    """Attach the vendored pruning helpers to *module* wherever missing.

    Absence-gated per name and per module: a name the module already exposes
    natively is never overwritten (transformers 4.x exposes both helpers;
    transformers 5.x ``pytorch_utils`` keeps its own ``prune_linear_layer``
    while lacking ``find_pruneable_heads_and_indices``). A module-level
    sentinel keeps repeat attachment idempotent.
    """
    if getattr(module, "_dnallm_remote_code_pruning_patch", False):
        return

    attached = False
    # setattr with a literal name is invisible to static attribute resolution,
    # keeping mypy AND ty/pyright clean without dialect-specific ignore
    # comments (ruff B010 is silenced because the dynamic form is deliberate).
    if not hasattr(module, "find_pruneable_heads_and_indices"):
        setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch (checker-agnostic)
            module,
            "find_pruneable_heads_and_indices",
            _find_pruneable_heads_and_indices,
        )
        attached = True
    if not hasattr(module, "prune_linear_layer"):
        setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch (checker-agnostic)
            module,
            "prune_linear_layer",
            _prune_linear_layer,
        )
        attached = True
    if attached:
        setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch (checker-agnostic)
            module, "_dnallm_remote_code_pruning_patch", True
        )


def _patch_remote_code_pruning_helpers():
    """Re-attach the 4.x pruning helpers removed from transformers 5.x.

    transformers 5.x removed ``find_pruneable_heads_and_indices`` and
    ``prune_linear_layer`` from ``transformers.modeling_utils``, and dropped
    ``find_pruneable_heads_and_indices`` from ``transformers.pytorch_utils``
    (which keeps its own ``prune_linear_layer``), but 4.x-era
    ``trust_remote_code`` checkpoints (e.g. the remote ``modeling_esm.py`` of
    the nucleotide-transformer-v2 promoter mirror) still import the names from
    ``transformers.modeling_utils`` and from ``transformers.pytorch_utils`` --
    the two canonical import sites of HF's own 4.x model files. This patch
    attaches the vendored v4.49.0 implementations under their upstream names
    to both modules, per name only where the module does not already expose
    it.

    On transformers 4.x both modules already expose the helpers natively and
    the patch no-ops (absence-gated per decision D-07); per-module sentinels
    keep repeat calls idempotent.
    """
    try:
        import transformers.modeling_utils
    except Exception:  # pragma: no cover - transformers not installed
        return
    _attach_remote_code_pruning_helpers(transformers.modeling_utils)

    # 4.x remote code canonically imports the helpers from pytorch_utils as
    # well; transformers 5.x still ships that module, but (as of 5.17) only
    # with its own prune_linear_layer -- only the missing name is attached.
    try:
        import transformers.pytorch_utils
    except Exception:  # pragma: no cover - module absent from this transformers
        return
    _attach_remote_code_pruning_helpers(transformers.pytorch_utils)


# The extended-attention-mask helper below is vendored with semantics and
# docstring kept verbatim from the upstream transformers reference
# implementation: tag v4.49.0, file src/transformers/modeling_utils.py
# (ModuleUtilsMixin.get_extended_attention_mask). transformers 5.x removed
# the method from modeling_utils and PreTrainedModel, but 4.x-era
# trust_remote_code checkpoints (the remote modeling_esm.py of
# zhangtaolab/nucleotide-transformer-v2-100m-promoter and its sibling NT/ESM
# caches) call it as a method inside EsmModel.forward
# (self.get_extended_attention_mask(attention_mask, input_shape)); the patch
# re-attaches it under its upstream name on the PreTrainedModel class only
# where the class does not already expose it. Three documented deviations
# from upstream: (1) is_decoder is read as
# getattr(self.config, "is_decoder", False) because transformers 5.x
# PretrainedConfig dropped the 4.x defaults and a remote 4.x config object
# may lack the attribute outright (the documented next rung of the 05-04
# D-07 ladder); (2) the decoder branch raises NotImplementedError instead of
# delegating to create_extended_attention_mask_for_decoder, which 5.x also
# removed, rather than returning a silently non-causal mask (every
# ESM-family remote checkpoint is an encoder); (3) upstream's cosmetic
# FutureWarning about the deprecated `device` argument is dropped, while the
# argument itself is kept in the signature for call compatibility.


def _get_extended_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_shape: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with the same dtype as `dtype`.
    """
    if dtype is None:
        dtype = self.dtype

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if getattr(self.config, "is_decoder", False):
            # Deviation (2) from upstream v4.49.0: the decoder branch there
            # delegated to ModuleUtilsMixin.create_extended_attention_mask_for_decoder,
            # which transformers 5.x also removed. Fail loudly instead of
            # returning a silently non-causal mask.
            raise NotImplementedError(
                "get_extended_attention_mask does not implement the decoder branch: upstream "
                "transformers 4.49 delegated it to create_extended_attention_mask_for_decoder, "
                "which transformers 5.x removed; dnallm's transformers-5 remote-code shim "
                "refuses to return a silently non-causal mask for a decoder config"
            )
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or "
            f"attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's minimum value for masked
    # positions. Since we are adding it to the raw scores before the softmax,
    # this is effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def _patch_get_extended_attention_mask():
    """Re-attach the 4.x ``get_extended_attention_mask`` removed from transformers 5.x.

    transformers 5.x removed ``get_extended_attention_mask`` from both
    ``transformers.modeling_utils`` and ``PreTrainedModel``, but 4.x-era
    ``trust_remote_code`` checkpoints (e.g. the remote ``modeling_esm.py`` of
    the nucleotide-transformer-v2 promoter mirror) call it as a METHOD
    (``self.get_extended_attention_mask(attention_mask, input_shape)``)
    inside ``EsmModel.forward``. The patch therefore attaches the vendored
    v4.49.0 implementation under its upstream name onto the
    ``PreTrainedModel`` class, which every remote ``EsmPreTrainedModel``
    subclass reaches through normal MRO.

    On transformers 4.x the class already exposes the native method and the
    patch no-ops (absence-gated per name, never overwrite); a class sentinel
    keeps repeat calls idempotent.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers not installed
        return

    if hasattr(PreTrainedModel, "get_extended_attention_mask"):
        return

    if getattr(PreTrainedModel, "_dnallm_extended_mask_patch", False):
        return

    PreTrainedModel.get_extended_attention_mask = (  # type: ignore[method-assign]
        _get_extended_attention_mask
    )
    PreTrainedModel._dnallm_extended_mask_patch = True  # type: ignore[attr-defined]


# The legacy config defaults below restore READ behavior that transformers 4.x
# provided by setting both attributes on every PretrainedConfig instance in
# ``__init__`` (upstream tag v4.49.0, src/transformers/configuration_utils.py:
# ``self.is_decoder = kwargs.pop("is_decoder", False)`` and the matching
# ``add_cross_attention`` line). transformers 5.x removed both defaults, and
# 4.x-era trust_remote_code checkpoints read them at model build (the remote
# modeling_esm.py lines 335/584-585 of the zhangtaolab NER/promoter mirrors:
# ``config.is_decoder`` and ``config.add_cross_attention``), which now raises
# ``AttributeError: 'EsmConfig' object has no attribute 'is_decoder'``. The
# patch installs a ``PretrainedConfig.__getattr__`` over a CLOSED default map
# instead of re-adding instance attributes, so serialization (``to_dict``
# iterates the instance dict) and every explicitly-set value stay exactly as
# transformers 5.x produces them. This rung SUPERSEDES the 05-04 D-07 rung
# termination (STATE.md had typed it "not vendored-pure-helper territory") per
# the owner instruction of 2026-10-02: fix all non-gated census failures now.
# If execution surfaces another removed 4.x config default that remote code
# reads, extend the closed map -- never a catch-all.

_LEGACY_PRETRAINED_CONFIG_DEFAULTS: dict[str, object] = {
    "is_decoder": False,
    "add_cross_attention": False,
}


def _legacy_config_defaults_missing(config_cls: type) -> bool:
    """Gate predicate: fresh instances of *config_cls* cannot resolve a legacy default.

    Args:
        config_cls: the config class to probe with a no-argument construction.

    Returns:
        True when at least one closed-map name fails to resolve on a freshly
        constructed instance (the transformers 5.x shape); False when every
        name resolves natively (the 4.x shape, or an already-patched class).
    """
    try:
        probe = config_cls()
    except Exception:
        # A config class that cannot be constructed bare cannot be probed
        # safely; leave it untouched rather than guessing.
        return False
    return any(not hasattr(probe, name) for name in _LEGACY_PRETRAINED_CONFIG_DEFAULTS)


def _pretrained_config_getattr(self: object, name: str) -> object:
    """Answer the closed map of removed 4.x config defaults; else AttributeError."""
    try:
        return _LEGACY_PRETRAINED_CONFIG_DEFAULTS[name]
    except KeyError:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None


def _patch_pretrained_config_legacy_defaults():
    """Restore the 4.x ``is_decoder``/``add_cross_attention`` config reads on 5.x.

    Installs ``PretrainedConfig.__getattr__`` answering the CLOSED default map
    above only when a freshly constructed ``PretrainedConfig()`` actually fails
    to resolve ``is_decoder`` (verified live: transformers 5.17 defines no
    ``__getattr__`` and no class-level defaults, while 4.x sets both names on
    every instance). An existing ``__getattr__`` is never overwritten; a class
    sentinel keeps repeat calls idempotent. Unknown attributes keep raising
    plain ``AttributeError``, and explicitly-set instance values shadow the
    defaults through normal attribute precedence.
    """
    try:
        import transformers.configuration_utils
    except Exception:  # pragma: no cover - transformers not installed / module renamed
        return

    config_cls = transformers.configuration_utils.PretrainedConfig

    if getattr(config_cls, "_dnallm_config_legacy_defaults_patch", False):
        return

    if "__getattr__" in vars(config_cls):
        return

    if not _legacy_config_defaults_missing(config_cls):
        return

    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic class patch
        config_cls,
        "__getattr__",
        _pretrained_config_getattr,
    )
    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic class patch
        config_cls,
        "_dnallm_config_legacy_defaults_patch",
        True,
    )


# The vendored MambaCache below is copied with semantics, docstring and
# deprecation warnings kept verbatim from the upstream transformers reference
# implementation: tag v4.49.0, file src/transformers/cache_utils.py. transformers
# 5.x removed MambaCache from cache_utils entirely (verified live on 5.17: the
# name exists in neither transformers.cache_utils nor
# transformers.models.mamba.modeling_mamba), but 4.x-era trust_remote_code
# checkpoints still import it from transformers.cache_utils and construct it
# with the 4.x signature (the tRNADetector remote modeling_mamba.py line 27
# ``from transformers.cache_utils import MambaCache``, constructed at its line
# 670 as ``MambaCache(config, batch_size, device=..., dtype=...)``); the patch
# below re-attaches the vendored class under its upstream name on the module,
# per name only where the module does not already expose it. The only
# adaptations are import locality into this module (modern builtin-generic
# annotations, typing-only PretrainedConfig import) and the upstream
# ``logger.warning_once`` deprecation notices re-emitted through a local
# warn-once helper over the ``warnings`` module; behavior is unchanged.

_MAMBA_CACHE_WARNED: set[str] = set()


def _warning_once(message: str) -> None:
    """Emit *message* as a UserWarning at most once per process.

    Stand-in for the upstream transformers ``logger.warning_once`` used by the
    vendored MambaCache deprecation paths (upstream logs each message once);
    the pytest suite ignores UserWarnings, so remote-code construction paths
    stay quiet under test.

    Args:
        message: the deprecation text to warn about exactly once.
    """
    if message in _MAMBA_CACHE_WARNED:
        return
    _MAMBA_CACHE_WARNED.add(message)
    warnings.warn(message, UserWarning, stacklevel=2)


class _MambaCache:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.
            The recommended way however is not not indicate any `device`, in that case cache will be initialized on `meta`
            device by default, and then moved to input device when updating.

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        device (`torch.device`):
            The default device on which the cache was initialized.
        intermediate_size: (`int`):
            Model's intermediate_size taken from config.
        ssm_state_size: (`int`):
            Model's state_size taken from config.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config
        conv_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, conv_kernel_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, ssm_state_size]` that holds ssm states

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    """

    is_compileable = True

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PreTrainedConfig,
        batch_size: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        max_batch_size: int | None = None,
    ):
        if batch_size is not None:
            _warning_once(
                f"The 'batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.49. Use the more precisely named 'max_batch_size' argument instead."
            )
        self.dtype = dtype
        self.max_batch_size = batch_size or max_batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.device = torch.device(device) if device is not None else torch.device("meta")

        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []
        for _ in range(config.num_hidden_layers):
            conv_state: torch.Tensor = torch.zeros(
                self.max_batch_size,
                self.intermediate_size,
                self.conv_kernel_size,
                device=self.device,
                dtype=dtype,
            )
            ssm_state: torch.Tensor = torch.zeros(
                self.max_batch_size,
                self.intermediate_size,
                self.ssm_state_size,
                device=self.device,
                dtype=dtype,
            )

            torch._dynamo.mark_static_address(conv_state)
            torch._dynamo.mark_static_address(ssm_state)
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        if self.conv_states[layer_idx].device.type == "meta":
            self.conv_states[layer_idx] = torch.zeros_like(
                self.conv_states[layer_idx],
                device=new_conv_state.device,
            )

        conv_state = self.conv_states[layer_idx]
        clamped_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, clamped_position] = new_conv_state.to(
            device=conv_state.device, dtype=conv_state.dtype
        )
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_states)):
            if self.conv_states[layer_idx].device.type != "meta":
                # In-place ops prevent breaking the static address
                self.conv_states[layer_idx].zero_()
                self.ssm_states[layer_idx].zero_()

    @property
    def batch_size(self):
        _warning_once(
            f"The 'batch_size' attribute of {self.__class__.__name__} is deprecated and will be removed in "
            "v4.49. Use the more precisely named 'self.max_batch_size' attribute instead."
        )
        return self.max_batch_size


def _patch_mamba_cache():
    """Re-attach the 4.x ``MambaCache`` removed from transformers 5.x cache_utils.

    transformers 5.x removed ``MambaCache`` from ``transformers.cache_utils``
    (verified live on 5.17: present in neither cache_utils nor
    transformers.models.mamba.modeling_mamba), but 4.x-era trust_remote_code
    checkpoints import it from there (the tRNADetector remote
    modeling_mamba.py line 27). The patch attaches the vendored v4.49.0
    implementation under its upstream name onto the module, only where the
    module does not already expose it; a module-level sentinel keeps repeat
    calls idempotent.
    """
    try:
        import transformers.cache_utils
    except Exception:  # pragma: no cover - transformers not installed / module renamed
        return

    module = transformers.cache_utils

    if hasattr(module, "MambaCache"):
        return

    if getattr(module, "_dnallm_mamba_cache_patch", False):
        return

    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch
        module,
        "MambaCache",
        _MambaCache,
    )
    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch
        module,
        "_dnallm_mamba_cache_patch",
        True,
    )


def _patch_deberta_vocab_dict():
    """Normalize a dict ``vocab`` for the 5.x Unigram ``DebertaV2Tokenizer``.

    transformers 5.17 hands ``DebertaV2Tokenizer`` a ``vocab`` dict
    (``{token: score}``, insertion-ordered -- proven live with the
    zhangtaolab/plant-dnabert-BPE checkpoint: 8000 entries with ``<unk>: 0``
    first) through ``convert_to_native_format``, the exact classmethod hook
    ``PreTrainedTokenizerBase.from_pretrained`` calls immediately before
    ``cls(*init_inputs, **init_kwargs)``, while the 5.x Unigram backend only
    accepts a sequence of ``(token, score)`` pairs and fails with
    ``TypeError: 'dict' object is not an instance of 'Sequence'``. 4.x loaded
    the same checkpoint through a sentencepiece ``spm.model`` file and never
    saw a dict vocab. The patch wraps the hook on ``DebertaV2Tokenizer`` only
    and normalizes a surviving dict vocab to ``list(vocab.items())`` --
    insertion order IS the rank order the backend expects, and tuple equality
    keeps the ``vocab.index((str(unk_token), 0.0))`` lookup working with int
    scores. A pair-list vocab passes through unchanged; a class sentinel keeps
    repeat calls idempotent, and the patch no-ops when the tokenizer class or
    the hook is absent (transformers 4.x has no ``convert_to_native_format``).
    """
    try:
        from transformers.models.deberta_v2.tokenization_deberta_v2 import (  # ty: ignore[unresolved-import]  # lazy export, resolves live
            DebertaV2Tokenizer,
        )
    except Exception:  # pragma: no cover - transformers not installed / module renamed
        return

    if getattr(DebertaV2Tokenizer, "convert_to_native_format", None) is None:
        return

    if getattr(DebertaV2Tokenizer, "_dnallm_deberta_vocab_patch", False):
        return

    # The bound classmethod resolves through the MRO (defined on the 5.x
    # TokenizersBackend base); capturing it here keeps subclass calls bound
    # to DebertaV2Tokenizer semantics.
    original = DebertaV2Tokenizer.convert_to_native_format

    def convert_to_native_format(cls, trust_remote_code=False, **kwargs):
        native = original(trust_remote_code=trust_remote_code, **kwargs)
        vocab = native.get("vocab")
        if isinstance(vocab, dict):
            native["vocab"] = list(vocab.items())
        return native

    DebertaV2Tokenizer.convert_to_native_format = classmethod(  # type: ignore[method-assign]
        convert_to_native_format
    )
    DebertaV2Tokenizer._dnallm_deberta_vocab_patch = True  # type: ignore[attr-defined]


# The head-mask helpers below are vendored with semantics and docstrings kept
# verbatim from the upstream transformers reference implementation: tag
# v4.49.0, file src/transformers/modeling_utils.py
# (ModuleUtilsMixin.get_head_mask and its private _convert_head_mask_to_5d
# helper). transformers 5.x removed both from modeling_utils and
# PreTrainedModel, but 4.x-era trust_remote_code checkpoints call
# get_head_mask as a METHOD inside EsmModel.forward
# (self.get_head_mask(head_mask, self.config.num_hidden_layers) -- five live
# call sites across the cached zhangtaolab/InstaDeepAI remote modeling
# files); the patch re-attaches the vendored v4.49.0 implementations under
# their upstream names on the PreTrainedModel class, per name only where the
# class does not already expose it. Three documented deviations from upstream:
# (1) modern builtin-generic annotations; (2) upstream's bare
# `assert head_mask.dim() == 5` is re-raised as an explicit
# `raise AssertionError` with the identical message because repo lint (S101)
# forbids bare asserts in dnallm/ -- the exception type and text are
# unchanged; (3) get_head_mask's tail is an early return instead of a
# reassigned `head_mask = [None] * num_hidden_layers` (the reassignment lies
# about the local's Tensor type; the early return is behavior-identical) and
# the `self` parameters are typed `PreTrainedModel` instead of upstream's
# implicit ModuleUtilsMixin receiver so attribute access checks honestly.


def _convert_head_mask_to_5d(
    self: PreTrainedModel, head_mask: torch.Tensor, num_hidden_layers: int
):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = (
            head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # We can specify head_mask for each layer
    if head_mask.dim() != 5:
        raise AssertionError(f"head_mask.dim != 5, instead {head_mask.dim()}")
    head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
    return head_mask


def _get_head_mask(
    self: PreTrainedModel,
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> torch.Tensor | list[None]:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 to discard).
        num_hidden_layers (`int`):
            The number of hidden layers in the model.
        is_attention_chunked (`bool`, *optional*, defaults to `False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
        `[None]` for each layer.
    """
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    return [None] * num_hidden_layers


def _patch_get_head_mask():
    """Re-attach the 4.x ``get_head_mask`` removed from transformers 5.x.

    transformers 5.x removed ``get_head_mask`` and ``_convert_head_mask_to_5d``
    from ``transformers.modeling_utils`` and ``PreTrainedModel``, but 4.x-era
    ``trust_remote_code`` checkpoints call the method inside
    ``EsmModel.forward``. The patch attaches the vendored v4.49.0
    implementations under their upstream names onto the ``PreTrainedModel``
    class (which every remote ``EsmPreTrainedModel`` subclass reaches through
    normal MRO), per name only where the class does not already expose it; a
    class sentinel keeps repeat calls idempotent.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers not installed
        return

    attached = False
    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = _get_head_mask  # type: ignore[method-assign]
        attached = True
    if not hasattr(PreTrainedModel, "_convert_head_mask_to_5d"):
        PreTrainedModel._convert_head_mask_to_5d = (  # type: ignore[method-assign]
            _convert_head_mask_to_5d
        )
        attached = True
    if attached:
        PreTrainedModel._dnallm_head_mask_patch = True  # type: ignore[attr-defined]


# The init_weights wrapper below restores the bookkeeping that transformers
# 5.x moved into PreTrainedModel.post_init (which ENDS by calling
# init_weights), while 4.x-era trust_remote_code checkpoints end their
# __init__ with the bare `self.init_weights()` entry (the remote
# EsmForMaskedLM / EsmForTokenClassification classes of the InstaDeepAI
# nucleotide-transformer-v2, zhangtaolab/plant-nucleotide-transformer-BPE and
# zhangtaolab/tRNAPointer mirrors -- 3 call sites each). With the bookkeeping
# skipped, from_pretrained crashes at _move_missing_keys_from_meta_to_device
# (modeling_utils) reading `self.all_tied_weights_keys` with
# AttributeError: 'EsmForMaskedLM' object has no attribute
# 'all_tied_weights_keys'. The wrapper routes a receiver MISSING the
# bookkeeping through the real self.post_init() first -- which sets the
# attribute and then calls this same wrapped init_weights again (bounded
# depth 2, never recursive) -- so weights are still initialized and tied
# exactly once, in the same order as a native 5.x model. On transformers 4.x,
# whose post_init computes no such attribute, a one-time probe keeps the
# wrapper from being installed at all (bare init_weights stays native).


def _post_init_computes_tied_weights_keys() -> bool:
    """Probe whether this transformers' ``post_init`` assigns the 5.x bookkeeping.

    Builds a bare ``PreTrainedModel`` and runs its own ``post_init``; True
    means the 5.x bookkeeping (``all_tied_weights_keys``) is produced there
    and the legacy-entry wrapper is meaningful. Any construction or probe
    failure safely maps to False (patch not installed).

    Returns:
        True when ``post_init`` assigns ``all_tied_weights_keys`` (5.x shape).
    """
    try:
        import transformers.modeling_utils

        from transformers import PreTrainedConfig  # ty: ignore[unresolved-import]  # lazy export, resolves live

        probe = transformers.modeling_utils.PreTrainedModel(PreTrainedConfig())
        probe.post_init()
        return hasattr(probe, "all_tied_weights_keys")
    except Exception:
        return False


def _patch_legacy_init_weights_bookkeeping():
    """Run the 5.x post_init bookkeeping behind the bare 4.x ``init_weights()`` entry.

    Wraps ``PreTrainedModel.init_weights`` so a receiver lacking
    ``all_tied_weights_keys`` (the signature of the bare 4.x-style entry that
    skipped ``post_init``) is routed through the real ``self.post_init()``
    before the original body runs. Install is gated on the
    :func:`_post_init_computes_tied_weights_keys` probe (no-op on
    transformers 4.x) and on a class sentinel for idempotency; an existing
    init_weights is only ever wrapped once, never replaced.
    """
    try:
        import transformers.modeling_utils
    except Exception:  # pragma: no cover - transformers not installed
        return

    model_cls = transformers.modeling_utils.PreTrainedModel

    if getattr(model_cls, "_dnallm_init_weights_patch", False):
        return

    original = vars(model_cls).get("init_weights")
    if original is None:
        return

    if not _post_init_computes_tied_weights_keys():
        return

    def init_weights(self):
        if not hasattr(self, "all_tied_weights_keys"):
            # 4.x remote entry: run the full post_init (bookkeeping, then this
            # same wrapped init_weights with the attribute now present).
            return self.post_init()
        return original(self)

    model_cls.init_weights = init_weights  # type: ignore[method-assign]
    model_cls._dnallm_init_weights_patch = True  # type: ignore[attr-defined]


# numpy 2.0 removed ``np.fromstring`` entirely (its binary mode had been
# deprecated since numpy 1.14 with "use frombuffer instead"; the text mode
# went with it). stripedhyena's ``CharLevelTokenizer`` -- the evo-1 family
# tokenizer loaded through dnallm/models/special/evo.py -- still calls
# ``np.fromstring`` on the utf-8 bytes of every sequence, which raises
# AttributeError on numpy 2.x. The vendored fallback below restores ONLY
# the historical binary-mode behavior: ``frombuffer`` semantics with the
# writable copy ``fromstring`` returned, and the ``count`` argument honored.
# The text mode (``sep != ''``) is refused with a ``loadtxt`` pointer --
# nothing in dnallm or stripedhyena uses it, and reproducing it would mean
# re-vendoring the removed C text parser. If another numpy API disappears
# that dnallm needs, add a NEW absence-gated rung -- never widen this one.


def _np_fromstring(string, dtype=float, count=-1, sep=""):
    """Vendored ``np.fromstring`` binary-mode fallback (see comment above)."""
    import numpy as np

    if sep:
        raise ValueError(
            "np.fromstring text mode (sep != '') is not provided by the dnallm "
            "compat shim; it was removed with numpy 2.0 and no dnallm/stripedhyena "
            "code path uses it. Use numpy.loadtxt instead."
        )
    # Historical fromstring returned a fresh writable array, while frombuffer
    # shares the read-only buffer -- copy so downstream in-place writes keep
    # working exactly as they did before numpy 2.0.
    return np.array(np.frombuffer(string, dtype=dtype, count=count))


def _numpy_fromstring_works(numpy) -> bool:
    """Probe whether ``numpy.fromstring`` is actually usable.

    numpy 2.x keeps the NAME ``fromstring`` as a stub that raises
    ``ValueError`` on every call ("The binary mode of fromstring is
    removed, use frombuffer instead"), so a mere ``hasattr`` gate would
    no-op on exactly the versions that need the shim. The probe performs
    one tiny binary-mode call: any raise counts as absent, a clean
    return (numpy 1.x native, or an already-installed shim) counts as
    present. The numpy 1.x call emits a DeprecationWarning, which this
    suite ignores globally.
    """
    fn = getattr(numpy, "fromstring", None)
    if fn is None:
        return False
    try:
        # getattr (not attribute access): the probe must fail on "fromstring
        # is broken", never on "this module clone lacks a uint8 alias".
        fn(b"AC", dtype=getattr(numpy, "uint8", None))
    except Exception:
        return False
    return True


def _patch_numpy_fromstring():
    """Restore ``np.fromstring`` (binary mode) on numpy 2.x for stripedhyena.

    Gated on the :func:`_numpy_fromstring_works` probe (numpy 1.x keeps its
    native working ``fromstring`` untouched; numpy 2.x ships a raising stub
    that must be replaced), idempotent via a module sentinel, and a plain
    no-op when numpy is not installed at all.
    """
    try:
        import numpy
    except Exception:  # pragma: no cover - numpy not installed
        return

    if _numpy_fromstring_works(numpy):
        return

    if getattr(numpy, "_dnallm_fromstring_patch", False):
        return

    numpy.fromstring = _np_fromstring
    numpy._dnallm_fromstring_patch = True  # type: ignore[attr-defined]


def apply_patches():
    """Apply all compatibility patches. Safe to call multiple times."""
    _patch_get_parameter_or_buffer()
    _patch_initialize_weights_for_quantized_missing()
    _patch_remote_code_pruning_helpers()
    _patch_get_extended_attention_mask()
    _patch_pretrained_config_legacy_defaults()
    _patch_mamba_cache()
    _patch_deberta_vocab_dict()
    _patch_get_head_mask()
    _patch_legacy_init_weights_bookkeeping()
    _patch_numpy_fromstring()


# Apply patches on module import so they are active before any
# transformers model is loaded through DNALLM.
apply_patches()
