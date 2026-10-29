import os
import sys
from importlib import import_module
from typing import Any

from ...utils import get_logger


logger = get_logger("dnallm.models.model")


def _triton_supports_trans_b() -> bool:
    """Return the triton capability result that controls flash neutralization.

    Semantics are inherited verbatim from the original inline probe: the
    return value answers "take the disable/neutralize path" -- ``True`` when
    ``tl.dot`` accepts ``trans_b`` (a NEW triton, whose kernels the repo's
    legacy ``flash_attn_triton.py`` cannot drive -- measured
    ``dot() got an unexpected keyword argument 'trans_b'``) or when the probe
    itself cannot run outside a kernel (``RuntimeError`` -> disable, the
    safe fallback on boxes without an active CUDA context); ``False`` only on
    the measured ``TypeError`` that says this triton genuinely lacks
    ``trans_b`` (the legacy path works).
    """

    try:
        # Typed Any: this is a runtime capability probe — `trans_b` is
        # deliberately passed to learn whether this triton build accepts
        # it (TypeError caught below), so the call must not be
        # signature-checked against whatever triton version is installed.
        tl: Any = import_module("triton.language")

        a = tl.zeros((1, 1), dtype=tl.float32)
        b = tl.zeros((1, 1), dtype=tl.float32)
        tl.dot(a, b, trans_b=True)  # capability probe: TypeError caught below
        return True
    except TypeError:
        return False
    except Exception:
        return True


def _wrap_rebuild_alibi_tensor(encoder_cls: type) -> bool:
    """Install the device-aligning wrapper on a remote ``BertEncoder`` class.

    Measured defect (zhihan1996/DNABERT-2-117M, bert_layers.py lines 394-399,
    snapshot 7bce263b): under transformers 5's meta-device construction
    context ``torch.arange(size, device=None)`` lands on meta while the
    legacy ``torch.Tensor(list)`` constructor ignores the default device and
    stays on cpu, so the slopes multiply crashes
    (``Tensor on device meta is not on the expected device cpu!``) inside
    ``__init__`` -- before any weight loads. The wrapper changes ONLY a
    missing device to ``"cpu"`` so both operands agree; the remote forward's
    own device catch-up (``self.alibi.to(hidden_states.device)``) then moves
    the tensor. An explicit device is forwarded unchanged. Meta-device
    fast/low-memory construction is preserved -- this is the owner-selected
    surgical alternative to forcing the whole model onto CPU construction
    (D-05).

    Args:
        encoder_cls: The remote ``BertEncoder`` class exposing
            ``rebuild_alibi_tensor``.

    Returns:
        True when the wrapper was installed, False when the class was
        already wrapped (sentinel) or carries no capturable method.
    """
    if getattr(encoder_cls, "_dnallm_alibi_device_patch", False):
        return False
    original = encoder_cls.__dict__.get("rebuild_alibi_tensor")
    if original is None:
        return False
    original_fn = original.__func__ if isinstance(original, (classmethod, staticmethod)) else original

    def rebuild_alibi_tensor(self, size, device=None):
        if device is None:
            # transformers 5 constructs remote models under a meta
            # default-device context; the legacy torch.Tensor(list)
            # constructor ignores it (cpu) while torch.arange honors it
            # (meta). Pin the one measured-broken operator to a real device
            # so both operands agree; forward's existing device catch-up
            # then moves the tensor.
            device = "cpu"
        return original_fn(self, size=size, device=device)

    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic class patch (checker-agnostic)
        encoder_cls,
        "rebuild_alibi_tensor",
        rebuild_alibi_tensor,
    )
    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic class patch (checker-agnostic)
        encoder_cls,
        "_dnallm_alibi_device_patch",
        True,
    )
    return True


def _neutralize_dnabert2_triton(module) -> bool:
    """Null an incompatible remote module's ``flash_attn_qkvpacked_func``.

    The repo's own module binds ``flash_attn_qkvpacked_func = None`` when its
    triton import fails, and its attention then routes to the pure-PyTorch
    path (``if ... flash_attn_qkvpacked_func is None``); setting the symbol
    to None on the module namespace reaches the same branch at call time
    (module-global lookup) without touching the snapshot files. Only armed
    when the capability probe selects the fallback, and only for modules the
    installer already narrowed to the DNABERT-2 shape.

    Args:
        module: The remote ``bert_layers`` module namespace.

    Returns:
        True when a live symbol was neutralized, False otherwise.
    """
    if not _triton_supports_trans_b():
        return False
    flash = getattr(module, "flash_attn_qkvpacked_func", None)
    if flash is None:
        return False
    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch (checker-agnostic)
        module,
        "flash_attn_qkvpacked_func",
        None,
    )
    return True


def _transformers_major_version(version: str) -> int:
    """Return the transformers major version, 0 when unparseable (never arm)."""
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _arm_dnabert2_dynamic_class_shims(dynamic_module_utils) -> bool:
    """Wrap ``get_class_in_module`` so resolved DNABERT-2 classes get the shims.

    The defect fires inside ``BertEncoder.__init__``, so a post-load patch is
    useless -- the wrapper must be installed on the remote class BEFORE any
    model constructor runs. ``transformers.dynamic_module_utils.get_class_in_module``
    is looked up as a module global at call time by the dynamic-class
    machinery (verified live on 5.17; the outer ``get_class_from_dynamic_module``
    is NOT patchable -- it is bound by from-import), which makes it the
    pre-construction seam. After delegation, a resolved class's module is
    inspected: one exposing ``BertEncoder.rebuild_alibi_tensor`` (the narrow
    DNABERT-2/MosaicBERT detection) receives the device-aligning wrapper and,
    when the capability probe selects the fallback, the triton neutralization.
    A module sentinel keeps repeat arming idempotent.

    Args:
        dynamic_module_utils: The ``transformers.dynamic_module_utils`` module.

    Returns:
        True when the resolver wrapper was installed, False when already
        armed or the target symbol is absent.
    """
    if getattr(dynamic_module_utils, "_dnallm_dnabert2_shim_patch", False):
        return False
    original = dynamic_module_utils.__dict__.get("get_class_in_module")
    if not callable(original):
        return False

    def get_class_in_module(class_name, module_path=None, **kwargs):
        target = original(class_name, module_path, **kwargs)
        module = sys.modules.get(getattr(target, "__module__", None))
        if module is not None:
            encoder_cls = getattr(module, "BertEncoder", None)
            if encoder_cls is not None and callable(
                getattr(encoder_cls, "rebuild_alibi_tensor", None)
            ):
                _wrap_rebuild_alibi_tensor(encoder_cls)
                _neutralize_dnabert2_triton(module)
        return target

    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch (checker-agnostic)
        dynamic_module_utils,
        "get_class_in_module",
        get_class_in_module,
    )
    setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic module patch (checker-agnostic)
        dynamic_module_utils,
        "_dnallm_dnabert2_shim_patch",
        True,
    )
    return True


def _install_dnabert2_tf5_shims() -> None:
    """Install the DNABERT-2 transformers-5 shims before any construction (D-05).

    Gated on transformers 5 (the meta-device construction path is a 5.x
    behavior; on 4.x models construct on cpu and the wrapper is unneeded --
    D-08 version gate for the construction-path class) and on
    ``transformers.dynamic_module_utils.get_class_in_module`` existing;
    idempotent through the module sentinel inside
    :func:`_arm_dnabert2_dynamic_class_shims`. No-op, never a raise, when
    transformers is absent or the seam moved.
    """
    try:
        import transformers
        import transformers.dynamic_module_utils
    except Exception:  # pragma: no cover - transformers not installed / module renamed
        return
    if _transformers_major_version(transformers.__version__) < 5:
        return
    _arm_dnabert2_dynamic_class_shims(transformers.dynamic_module_utils)


def _handle_dnabert2_models(model_path: str, load_args: list) -> tuple:
    """Handle special case for DNABERT-2 models.
    If the installed Triton version not supports trans_b in tl.dot,
    we disable the triton flash attention by renaming the file
    flash_attn_triton.py to flash_attn_triton.py.disabled.
    This is a workaround for the issue in triton 2.0.0+ that
    causes flash attention to produce incorrect results.
    Args:
        model_path: Path to the model directory
    """
    if (
        "dnabert-2" not in os.path.basename(model_path.lower())
        and "dnabert-s" not in os.path.basename(model_path).lower()
    ):
        return None, None

    triton_file_path = None
    original_content = None
    should_disable_triton = _triton_supports_trans_b()

    try:
        # D-05 (27.2-02): arm the pre-construction ALiBi device-alignment and
        # Triton shims BEFORE the delegated load -- transformers 5 constructs
        # the remote BertEncoder under a meta default-device context, so the
        # repair must be live on the remote class before __init__ runs.
        _install_dnabert2_tf5_shims()
        if should_disable_triton:
            triton_file_path = os.path.join(model_path, "flash_attn_triton.py")
            if os.path.exists(triton_file_path):
                # Step 1: Read and back up the original file content
                with open(triton_file_path, encoding="utf-8") as f:
                    original_content = f.read()
                # Step 2: Overwrite the file with code
                # that will raise an ImportError
                with open(triton_file_path, "w", encoding="utf-8") as f:
                    f.write(
                        'raise ImportError("Temporarily disabled '
                        'by script due to incompatible Triton version.")\n'
                    )
                logger.warning("Triton version does not support, flash_attn_triton disabled.")
            else:
                triton_file_path = None

        from ..model import _load_model_by_task_type

        model, tokenizer = _load_model_by_task_type(*load_args)
        return model, tokenizer
    finally:
        # This block will execute no matter what,
        # ensuring the original file is restored.
        if original_content is not None and triton_file_path is not None:
            with open(triton_file_path, "w", encoding="utf-8") as f:
                f.write(original_content)
