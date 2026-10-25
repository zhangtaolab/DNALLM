"""Root-import activation contract: a bare ``import dnallm`` arms the shims.

The v1.4 nightly census failure (run 40140203118,
``test_notebook_execution[notebooks/embedding_attention.ipynb]``) was an
ImportError for ``transformers.pytorch_utils.find_pruneable_heads_and_indices``
raised by the 4.x-era ``trust_remote_code`` checkpoint of the
nucleotide-transformer-v2 promoter mirror. The notebook's first cell imports
ONLY the root package (``import dnallm  # activate dnallm's transformers 5.x
remote-code compat shims``), and the Phase 25 PEP 562 lazy facade stopped the
root init from ever importing ``dnallm.utils.transformers_compat`` -- so the
compat patches that re-attach the vendored v4.49 pruning helpers never ran in
that kernel.

These tests pin the restored contract in both import orders:

1. dnallm first, transformers second -- activation fires when the first
   transformers import completes (a lazy meta-path hook armed by the root
   init), leaving ``dnallm.utils.transformers_compat`` in ``sys.modules`` and
   the pruning helper resolvable on ``transformers.pytorch_utils``.
2. transformers first, dnallm second (the exact notebook cell order) -- the
   hook arms AND activates immediately because the watch module is already in
   ``sys.modules``.

Runtime purity and the arming mechanics are guarded too: a bare
``import dnallm`` must stay free of the training stack (the Phase 25 D-01
light-path contract) and must install exactly one activation finder in
``sys.meta_path``.

Every subprocess assertion runs in a fresh ``python -I`` interpreter because
the pytest process has already imported the training stack through
``tests/conftest.py`` -- an in-process ``sys.modules`` assertion would be
vacuous (the same rationale as ``tests/test_package_facade.py``). The
subprocess prints machine-readable ``SHIM=`` / ``PATCHED=`` / ``HEAVY=`` /
``FINDERS=`` marker lines, matched over markers because the dnallm logger
also writes to stdout.
"""

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import types
from pathlib import Path

import pytest

import dnallm

PACKAGE_ROOT = Path(dnallm.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent

# The training stack the armed root import must keep out of the light path.
HEAVY_MODULES = ("torch", "transformers", "datasets", "matplotlib")

# Version-independent activation marker: the shim module's presence in
# sys.modules. The PATCHED marker (the pruning helper itself) is only a
# meaningful assertion on transformers 5.x, where the helpers were removed;
# on 4.x they exist natively and the patch rung no-ops.
_SHIM = "dnallm.utils.transformers_compat"

# The activation-finder home module; a fresh import installs exactly one
# meta_path entry whose type is defined here.
_FINDER_MODULE = "dnallm._compat_activation"


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh isolated interpreter.

    Args:
        code: the Python source to execute with ``python -I -c``.

    Returns:
        The completed process, stdout/stderr captured as text.
    """
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv only
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
    )


def _marker(output: str, prefix: str) -> str:
    """Return the single output line starting with *prefix*.

    Args:
        output: captured subprocess stdout.
        prefix: the marker prefix written by the test's snippet.

    Returns:
        The full marker line.

    Raises:
        AssertionError: the marker line is absent.
    """
    for line in output.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"marker line {prefix!r} missing from subprocess output:\n{output}")


# The two-fact contract asserted by both import orders: the shim module ran
# (version-independent) and the vendored pruning helper it re-attaches is
# resolvable where the remote modeling_esm.py imports it from.
_CONTRACT_SNIPPET = (
    "import sys;"
    "print('SHIM=' + str('" + _SHIM + "' in sys.modules));"
    "print('PATCHED=' + str("
    "hasattr(transformers.pytorch_utils, 'find_pruneable_heads_and_indices')))"
)


def test_bare_import_activates_compat_shims() -> None:
    """``import dnallm`` then any transformers import ends with shims active.

    This is the restored notebook contract: the bare root import arms a lazy
    activation hook, the first transformers import (any submodule -- importing
    one always executes the root package first) fires it after the real module
    body executes, and the compat patches are in place before a
    trust_remote_code checkpoint could resolve its 4.x-era helper imports.
    """
    code = "import dnallm; import transformers.pytorch_utils; " + _CONTRACT_SNIPPET
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "SHIM=") == "SHIM=True", (
        "import dnallm did not activate dnallm.utils.transformers_compat on the "
        "first transformers import — the notebook contract (embedding_attention"
        ".ipynb cell 1, run 40140203118) is broken again"
    )
    assert _marker(result.stdout, "PATCHED=") == "PATCHED=True", (
        "the compat shims ran but find_pruneable_heads_and_indices is still "
        "absent from transformers.pytorch_utils"
    )


def test_notebook_order_import_activates_compat_shims() -> None:
    """transformers imported BEFORE dnallm also ends with the shims active.

    The exact nightly failure shape: a notebook whose earlier cells imported
    transformers directly, then runs ``import dnallm``. The arm step must
    notice the watch module already sitting in sys.modules and activate
    immediately instead of waiting for an import that already happened.
    """
    code = "import transformers.pytorch_utils; import dnallm; " + _CONTRACT_SNIPPET
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "SHIM=") == "SHIM=True", (
        "import dnallm did not activate the compat shims when transformers was "
        "imported first — the notebook cell order of run 40140203118 fails"
    )
    assert _marker(result.stdout, "PATCHED=") == "PATCHED=True", (
        "the compat shims ran but find_pruneable_heads_and_indices is still "
        "absent from transformers.pytorch_utils"
    )


def test_bare_import_stays_light() -> None:
    """A bare ``import dnallm`` pulls none of the training stack (D-01).

    The activation fix must not pay for the notebook contract with the Phase
    25 laziness win: arming the hook imports no torch/transformers/datasets/
    matplotlib, so the TUI first paint, ``dnallm --help`` and the MCP
    bootstrap keep their measured light path.
    """
    code = (
        "import sys;"
        "import dnallm;"
        "print('HEAVY=' + ','.join(sorted("
        "name for name in ('torch', 'transformers', 'datasets', 'matplotlib')"
        " if name in sys.modules)))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "HEAVY=") == "HEAVY=", (
        "import dnallm pulls the training stack in again — the activation hook "
        "went eager and regressed the D-01 light-path contract"
    )


def test_armed_finder_is_single_in_meta_path() -> None:
    """Importing dnallm twice (and re-arming) installs exactly one finder.

    The finder sits at ``sys.meta_path[0]`` and is consulted for every import
    in the process, so a second arming must be a no-op scan, never a second
    instance. Asserted with equality so it is also red before the fix exists
    (zero finders) and would catch a double-arm (two) equally.
    """
    code = (
        "import sys;"
        "import dnallm;"
        "import dnallm;"  # no-op re-import, and the explicit re-arm below
        "from dnallm import _compat_activation;"
        "_compat_activation.arm_compat_activation();"
        "print('FINDERS=' + str(sum("
        "1 for f in sys.meta_path if type(f).__module__ == '" + _FINDER_MODULE + "')))"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert _marker(result.stdout, "FINDERS=") == "FINDERS=1", (
        "the activation finder count in sys.meta_path is not exactly 1 — "
        "arming is not idempotent (0 = never armed, 2+ = double-arm)"
    )


def test_finder_passes_through_non_watch_names() -> None:
    """The finder returns None for every name except the watch module.

    Pass-through is the tampering boundary (T-e4x-01): a finder consulted for
    every import in the process must never intercept resolution of any module
    other than the single watched name. Skipped (not failed) until the hook
    module exists, so the RED proof on tip isolates the contract tests.
    """
    activation = pytest.importorskip("dnallm._compat_activation")
    finder = activation._CompatActivationFinder()
    assert finder.find_spec("dnallm.utils.sequence") is None
    # A submodule of the watch package is NOT the watch name: importing
    # transformers.pytorch_utils must resolve through the normal chain.
    assert finder.find_spec("transformers.pytorch_utils") is None
    assert finder.find_spec("torch") is None
    assert finder.find_spec("sys") is None


# ─────────────────────────────────────────────────────────────────────────────
# In-process unit variants for the branches the subprocess contract tests can
# never exercise under coverage (the subprocess runs untraced): delegation
# exhaustion, the idempotent re-arm scan, and the arm-time-already-imported
# activation. Each uses monkeypatch with restore -- no lingering global state.
# ─────────────────────────────────────────────────────────────────────────────


def test_find_spec_passes_through_when_watch_module_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watch name no finder resolves returns None unwrapped, unfired.

    Delegation exhaustion: with an empty meta-path chain the finder must hand
    the "found nothing" verdict straight back so the interpreter raises the
    normal ImportError -- activation never fabricates a spec, and the finder
    does not mark itself fired for an import that did not happen.
    """
    activation = pytest.importorskip("dnallm._compat_activation")
    monkeypatch.setattr(sys, "meta_path", [])
    finder = activation._CompatActivationFinder()
    assert finder.find_spec(activation._WATCH_MODULE) is None
    assert finder._fired is False


def test_find_spec_returns_namespace_spec_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loader-less (namespace) spec is returned untouched and never wrapped.

    A namespace package has no module body to execute, so there is nothing to
    wrap and no meaningful activation point; the finder stays unfired and the
    interpreter's own namespace semantics apply.
    """
    activation = pytest.importorskip("dnallm._compat_activation")

    class _NullLoaderFinder:
        """Meta-path stub that resolves everything to a loader-less spec."""

        def find_spec(self, fullname, path=None, target=None):
            from importlib.machinery import ModuleSpec

            return ModuleSpec(fullname, loader=None)

    monkeypatch.setattr(sys, "meta_path", [_NullLoaderFinder()])
    finder = activation._CompatActivationFinder()
    spec = finder.find_spec(activation._WATCH_MODULE)
    assert spec is not None
    assert spec.loader is None
    assert finder._fired is False


def test_rearm_leaves_meta_path_untouched() -> None:
    """A second arm call scans and exits without touching sys.meta_path.

    The pytest process armed the real finder at collection time (conftest
    imports dnallm), so this in-process re-arm exercises exactly the scan-hit
    branch the subprocess FINDERS=1 contract pins end to end: no second
    finder, no reorder.
    """
    activation = pytest.importorskip("dnallm._compat_activation")
    before = list(sys.meta_path)
    activation.arm_compat_activation()
    assert sys.meta_path == before
    assert (
        sum(isinstance(finder, activation._CompatActivationFinder) for finder in sys.meta_path) == 1
    )


def test_late_arm_activates_immediately_when_transformers_already_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm finds an unfired finder and an imported watch module -> activate.

    In-process variant of the notebook-order branch, fully hermetic: stub
    sys.modules entries stand in for the already-imported transformers and
    the already-run shim module (the activation import becomes a no-op
    lookup), so a stand-in unfired finder at meta_path[0] makes arm() take
    the "the finder would never fire" branch -- activation runs immediately
    and the finder is marked fired, with no heavy import pulled in.
    """
    activation = pytest.importorskip("dnallm._compat_activation")
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(
        sys.modules,
        "dnallm.utils.transformers_compat",
        types.ModuleType("dnallm.utils.transformers_compat"),
    )
    fresh = activation._CompatActivationFinder()
    monkeypatch.setattr(sys, "meta_path", [fresh, *sys.meta_path])
    activation.arm_compat_activation()
    assert fresh._fired is True
