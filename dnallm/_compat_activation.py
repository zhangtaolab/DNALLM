"""Lazy activation of the transformers compatibility shims.

Contract: a plain ``import dnallm`` must activate dnallm's transformers
compat shims (``dnallm.utils.transformers_compat``, whose module tail runs
``apply_patches()``) before any 4.x-era ``trust_remote_code`` checkpoint
resolves its helper imports -- the exact failure of nightly run 40140203118,
where ``notebooks/embedding_attention.ipynb`` cell 1 imported only the root
package on transformers 5.x and the remote ``modeling_esm.py`` died on
``from transformers.pytorch_utils import find_pruneable_heads_and_indices``.

Activation is deliberately LAZY, not eager: importing
``dnallm.utils.transformers_compat`` at root-init time costs a measured
2.037 s (it pulls torch at module scope, then transformers.modeling_utils)
against the 0.177-0.191 s bare-``import dnallm`` baseline, and it would turn
the pinned Phase 25 purity contract
(``tests/test_package_facade.py::test_import_preflight_is_torch_free``, the
D-01 "wizard paints immediately" decision) red. Instead the root init ARMS
this module: a meta-path finder is inserted at ``sys.meta_path[0]`` that
watches exactly one module name and fires activation once, when the real
transformers package has finished executing.

Failure semantics are deliberate: activation errors PROPAGATE exactly as
they did in the pre-Phase-25 eager world. ``apply_patches`` already no-ops
per rung when its target is absent, so a merely old transformers never
raises; a genuinely broken environment fails loudly at the first
transformers import instead of poisoning every import silently. Errors are
never swallowed into the import machinery.

Thread safety: arming happens once at root-import time on a single thread;
the wrapped ``exec_module`` runs under the import system's own per-module
locking, like any import-time code.
"""

import sys
import types
from collections.abc import Sequence
from importlib.machinery import ModuleSpec

# One watch name covers every entry route: importing ANY transformers
# submodule first executes the root "transformers" package, so a finder
# watching the root name fires on `import transformers`, `import
# transformers.pytorch_utils`, `from transformers import AutoModel`, ... alike.
_WATCH_MODULE = "transformers"


class _CompatActivationFinder:
    """Meta-path finder that fires shim activation on the watched import.

    The finder sits at ``sys.meta_path[0]`` and is therefore consulted for
    every import in the process, which pins its tampering boundary
    (T-e4x-01): ``find_spec`` returns ``None`` for every fullname other than
    the single watch name -- a one-equality pass-through -- and for the watch
    name it never replaces module code. It delegates spec resolution to the
    remaining meta-path chain and wraps only the resolved loader's
    ``exec_module`` so the REAL transformers body always executes first;
    activation runs after it, before any submodule of the freshly imported
    package can execute. After one successful wrap the finder marks itself
    fired and becomes a permanent pass-through.
    """

    def __init__(self) -> None:
        self._fired = False

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Resolve *fullname*, wrapping only the watched module's execution.

        Args:
            fullname: the fully-qualified module name being imported.
            path: the parent package ``__path__`` for submodule imports;
                ``None`` for top-level modules such as the watch name.
            target: the existing module object for reloads, if any.

        Returns:
            The spec resolved by the remaining meta-path chain, with its
            loader's ``exec_module`` wrapped for the watch name; ``None``
            (pure pass-through) for every other name, after the finder has
            fired, and when delegation resolves nothing the normal
            ``ImportError`` path takes over.
        """
        if self._fired or fullname != _WATCH_MODULE:
            return None
        # Delegate to the remaining meta-path chain, skipping this instance.
        # Never assume PathFinder alone: editable installs register their own
        # finders; the first non-None spec wins, exactly like the interpreter.
        spec: ModuleSpec | None = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            # Genuinely absent module (let the normal ImportError happen) or a
            # namespace package (nothing to execute, nothing to wrap): return
            # the delegation result untouched.
            return spec
        loader = spec.loader
        original_exec_module = loader.exec_module

        def _wrapped_exec_module(module: types.ModuleType) -> None:
            # The real module body always executes FIRST; only then do the
            # compat shims run, against a fully initialized transformers.
            original_exec_module(module)
            _activate()

        setattr(  # ruff: ignore[set-attr-with-constant] - deliberate dynamic loader patch (checker-agnostic)
            loader,
            "exec_module",
            _wrapped_exec_module,
        )
        self._fired = True
        return spec


def _activate() -> None:
    """Import the compat shim module; its import tail applies the patches."""
    from .utils import transformers_compat  # ruff: ignore[unused-import] - the import IS the side effect


def arm_compat_activation() -> None:
    """Arm the lazy compat activation, or activate immediately if overdue.

    Idempotent: scans ``sys.meta_path`` for an existing
    :class:`_CompatActivationFinder` and inserts one at position 0 only when
    absent. When the watch module is already in ``sys.modules`` at arm time
    (the notebook cell order: transformers imported before dnallm), the
    finder would never fire -- no further import of the watched name will
    consult ``sys.meta_path`` -- so activation runs immediately instead and
    the finder is marked fired.
    """
    armed: _CompatActivationFinder | None = None
    for finder in sys.meta_path:
        if isinstance(finder, _CompatActivationFinder):
            armed = finder
            break
    if armed is None:
        armed = _CompatActivationFinder()
        sys.meta_path.insert(0, armed)
    if _WATCH_MODULE in sys.modules and not armed._fired:
        armed._fired = True
        _activate()
