"""Private notebook-execution harness for the example tests.

This module mirrors the ``dnallm/mcp/tests/_network_skip.py`` seam: a
``_``-prefixed helper module that lives beside its consumer tests, is
never imported by any root conftest or ``dnallm/`` module, and never
ships in the wheel.  It carries the whole execution machinery for the
example-notebook tests:

* :data:`NOTEBOOK_EXEC_SPECS` -- per-notebook execution budgets
  (per-cell timeout, per-test timeout, extra sandbox inputs);
* :func:`seed_sandbox` -- copy an example directory into a pytest
  ``tmp_path`` sandbox so the kernel cwd never touches the repo tree;
* :func:`run_notebook` -- execute a notebook via nbclient with per-cell
  timeout, immediate kernel shutdown and partial-failure artifacts;
* :func:`assert_tree_clean` -- scoped ``git status`` tripwire proving an
  execution never dirtied ``example/`` or ``docs/example/``.

nbclient 0.11.0 semantics (live-probed, 05-RESEARCH.md Pattern 1):
``NotebookClient`` is NOT a context manager; a plain ``execute()`` call
wraps the cells in a start/finally-cleanup chain, so kernel shutdown is
guaranteed on success, on cell error and on cell timeout.  Cell errors
and cell timeouts are always re-raised here -- never converted to skips
or soft-passes.  The only sanctioned skip paths for execution tests are
narrow typed-skip helpers whose stable prefixes are registered in
``tests/expected_skips.yaml``.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from nbformat import NotebookNode

# Repo and example anchors -- identical derivation to
# tests/examples/test_examples.py (its EXAMPLE_DIR line).
REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLE_DIR = REPO_ROOT / "example"

# Per-notebook execution budgets.  Keys are str() of the absolute
# notebook paths so parametrized lookups stay exact; values carry the
# per-cell timeout, the per-test timeout mark the test layer must
# apply, and any out-of-dir sandbox inputs.  Phase 8 expands this dict
# as more notebooks join the execution rollout.
NOTEBOOK_EXEC_SPECS: dict[str, dict] = {
    str(EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb"): {
        "cell_timeout": 600,
        "test_timeout": 1800,
        "extra_inputs": [],
    },
}

# Deterministic headless-execution overrides.  nbclient 0.11.0 has no
# ``env`` trait and kernels inherit the parent process environment at
# spawn time, so these are applied as a save/restore sandwich around
# execute() rather than constructor arguments.
_ENV_OVERRIDES: dict[str, str] = {
    "MPLBACKEND": "Agg",
    "TOKENIZERS_PARALLELISM": "true",
    "WANDB_MODE": "disabled",
}


def seed_sandbox(src_dir: Path, tmp_path: Path, extra_inputs: list[Path] | None = None) -> Path:
    """Copy an example directory into a pytest tmp sandbox for execution.

    Whole-dir copy, not just the ``.ipynb``: the example notebooks read
    sibling inputs (``./inference_config.yaml``, ``./test.csv``) with
    paths relative to their own directory, and those resolve inside the
    sandbox once the kernel cwd points there (Pitfall 8).

    Args:
        src_dir: example directory holding the notebook and its siblings.
        tmp_path: pytest function-scoped tmp dir; the sandbox is created
            under it as ``tmp_path / src_dir.name``.
        extra_inputs: optional out-of-dir inputs copied into the sandbox
            (none exist for the pilot; kept for Phase 8 generality).

    Returns:
        The sandbox path the kernel should use as its cwd.
    """
    sandbox = tmp_path / src_dir.name
    shutil.copytree(
        src_dir,
        sandbox,
        ignore=shutil.ignore_patterns(
            ".ipynb_checkpoints",
            "__pycache__",
            "outputs*",
            "results*",
            "*.gz",
        ),
    )
    for extra in extra_inputs or []:
        dest = sandbox / extra.name
        if not dest.exists():
            shutil.copy2(extra, dest)
    return sandbox


def run_notebook(
    nb_path: Path,
    sandbox: Path,
    cell_timeout: int = 600,
    artifact_dir: Path | None = None,
) -> NotebookNode:
    """Execute a notebook inside *sandbox* and return the executed node.

    The kernel cwd is the sandbox (via ``resources.metadata.path``, the
    nbclient kernel-cwd mechanism), each cell gets ``cell_timeout``
    seconds, and the kernel is SIGKILLed immediately on shutdown.
    ``NotebookClient`` is not a context manager in nbclient 0.11.0:
    plain ``execute()`` self-cleans through its internal finally (on
    success, cell error and cell timeout alike), which is why the
    client is never wrapped in a with-statement here.

    Args:
        nb_path: notebook to execute (read from the repo tree; never modified).
        sandbox: kernel working directory (a seeded tmp copy).
        cell_timeout: per-cell timeout in seconds; must stay strictly
            below the per-test pytest timeout mark.
        artifact_dir: directory for partial-failure artifacts.  When
            given, a cell error or timeout writes the executed notebook
            node plus the exception text here before re-raising.

    Returns:
        The executed notebook node with every cell's outputs collected.

    Raises:
        CellExecutionError: a cell raised; re-raised after artifact capture.
        CellTimeoutError: a cell exceeded ``cell_timeout``; re-raised
            after artifact capture (the timed-out cell carries no error
            output in the node, hence the separate ``.error.txt``).
    """
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=cell_timeout,
        allow_errors=False,  # fail at first error -- the repair signal this milestone exists for
        kernel_name="python3",
        startup_timeout=120,
        shutdown_kernel="immediate",
        resources={"metadata": {"path": str(sandbox)}},
    )
    saved_env = {key: os.environ.get(key) for key in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    try:
        client.execute()  # internally: setup_kernel -> cells -> finally _cleanup_kernel()
        return nb
    except (CellExecutionError, CellTimeoutError) as exc:
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, artifact_dir / f"{nb_path.stem}.executed.ipynb")
            (artifact_dir / f"{nb_path.stem}.error.txt").write_text(str(exc), encoding="utf-8")
        raise
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_tree_clean(paths: tuple[str, ...] = ("example", "docs/example")) -> None:
    """Fail if an execution wrote anything into the guarded repo paths.

    Runs a scoped ``git status --porcelain -- <paths>`` from the repo
    root; any output line means an execution dirtied the tree, which is
    exactly the false green this harness exists to prevent.  Scoped on
    purpose: untracked ``.planning/`` / ``.gsd/`` working dirs are none
    of the harness's business.

    Args:
        paths: repo-relative pathspecs to watch.
    """
    # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert not result.stdout.strip(), (
        f"notebook execution dirtied the repo tree under {paths}:\n{result.stdout}"
    )
