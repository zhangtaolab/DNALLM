"""Private notebook-execution harness for the example tests.

This module mirrors the ``dnallm/mcp/tests/_network_skip.py`` seam: a
``_``-prefixed helper module that lives beside its consumer tests, is
never imported by any root conftest or ``dnallm/`` module, and never
ships in the wheel.  It carries the whole execution machinery for the
example-notebook tests:

* :data:`NOTEBOOK_EXEC_SPECS` -- per-notebook execution budgets
  (per-cell timeout, extra sandbox inputs);
* :data:`MARIMO_EXEC_SPECS` -- per-marimo-app execution specs (wall
  timeout for the export subprocess);
* :func:`seed_sandbox` -- copy an example directory into a pytest
  ``tmp_path`` sandbox so the kernel cwd never touches the repo tree;
* :func:`run_notebook` -- execute a notebook via nbclient with per-cell
  timeout, immediate kernel shutdown and partial-failure artifacts;
* :func:`run_marimo_app` -- execute a marimo app headlessly through the
  venv ``marimo`` CLI (export-html flavor) inside a sandbox cwd;
* :func:`run_example_script` -- run an example helper script with the
  venv interpreter inside a sandbox cwd, always leaving a run log;
* :func:`assert_tree_clean` -- scoped ``git status`` tripwire proving an
  execution never dirtied ``example/`` or ``docs/example/``;
* :func:`environment_unavailable_skip` / :func:`optional_dep_skip` /
  :func:`network_unavailable_skip` -- the only sanctioned skip paths,
  emitting stable junit-greppable prefixes registered in
  ``tests/expected_skips.yaml``.

nbclient 0.11.0 semantics (live-probed, 05-RESEARCH.md Pattern 1):
``NotebookClient`` is NOT a context manager; a plain ``execute()`` call
wraps the cells in a start/finally-cleanup chain, so kernel shutdown is
guaranteed on success, on cell error and on cell timeout.  Cell errors
and cell timeouts are always re-raised here -- never converted to skips
or soft-passes.  The only sanctioned skip paths for execution tests are
narrow typed-skip helpers whose stable prefixes are registered in
``tests/expected_skips.yaml``.

marimo execution flavor: ``export html`` is the 05-FEASIBILITY.md spike
decision -- it yields a deterministic exit code AND a concrete artifact
(the rendered HTML carrying the executed outputs) per app.  Script-mode
(``python app.py``, which the same spike proved terminates cleanly and
binds no port) remains the sanctioned fallback flavor.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from nbformat import NotebookNode

# Repo and example anchors -- identical derivation to
# tests/examples/test_examples.py (its EXAMPLE_DIR line).
REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLE_DIR = REPO_ROOT / "example"

# Per-notebook execution budgets.  Keys are str() of the absolute
# notebook paths so parametrized lookups stay exact; values carry the
# per-cell timeout and any out-of-dir sandbox inputs.  The per-test
# pytest-timeout marks are enforced in the TEST modules (class-ladder
# marks plus explicit per-entry overrides) and are deliberately NOT
# duplicated here: a spec-side copy of the outer budget was never
# consulted by the test layer and went stale enough to hide a
# strictly-below violation once (05 review WR-02/WR-03), so the specs
# carry only budgets this harness itself enforces.  All 21 example
# notebooks carry starter budgets (05-05, D-08 census); the execution
# test's ACTIVE_NOTEBOOKS list gates which ones actually run -- 05-06
# grows it with census-green notebooks.  Budgets follow the per-class ladder
# of the existing real-model precedents (tests/models/test_model.py
# 900s downloads; tests/finetune/test_trainer_real_model.py 3600-7200s
# real training): 600-900 inference, 1800 evo/generation-LoRA-inference,
# 1800-3600 data prep, 3600 finetune.  Starter budgets; 05-06 records
# actuals and may tune.
NOTEBOOK_EXEC_SPECS: dict[str, dict] = {
    str(EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb"): {
        "cell_timeout": 600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "inference_for_tRNA" / "inference.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "generation" / "inference.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "generation_evo_models" / "inference.ipynb"): {
        "cell_timeout": 1800,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "generation_megaDNA" / "inference.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "in_silico_mutagenesis" / "in_silico_mutagenesis.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "interpretation" / "interpretation.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "embedding_attention.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "data_prepare" / "finetune" / "finetune_data.ipynb"): {
        "cell_timeout": 1800,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "data_prepare" / "predict" / "predict_data.ipynb"): {
        "cell_timeout": 900,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "finetune_binary" / "finetune_binary.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "finetune_custom_head" / "finetune.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "finetune_generation" / "finetune_generation.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "finetune_multi_labels" / "finetune_multi_labels.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "finetune_NER_task" / "data_generation_and_inference.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "finetune_NER_task" / "finetune_NER_task.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "lora_finetune_inference" / "lora_finetune.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "lora_finetune_inference" / "lora_inference.ipynb"): {
        "cell_timeout": 1800,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "notebooks" / "benchmark" / "benchmark.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_langchain_agents.ipynb"): {
        "cell_timeout": 600,
        "extra_inputs": [],
    },
    str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_pydantic_ai.ipynb"): {
        "cell_timeout": 600,
        "extra_inputs": [],
    },
}

# Per-marimo-app execution specs.  Keys are str() of the absolute app
# paths in the same literal form as the notebook keys above; values
# carry the wall timeout for the export subprocess, which must stay
# strictly below the per-test pytest-timeout mark the test module
# applies (its 7200s class mark).  The export flavor is NOT spec data:
# run_marimo_app hardcodes ``marimo export html`` -- the 05-FEASIBILITY.md
# decision (spike export ~40s warm; 1200s wall leaves headroom for a cold
# model fetch, the MARIMO_FLAVOR_TIMEOUT_S=1200 precedent) -- until a
# second flavor actually exists.
MARIMO_EXEC_SPECS: dict[str, dict] = {
    str(EXAMPLE_DIR / "marimo" / "inference" / "inference_demo.py"): {
        "timeout_s": 1200,
    },
    # 05-06 census-green apps (both export in <10s: config/dataset load runs
    # at module level; the model actions are button-gated by app design).
    # 3600s wall leaves headroom for a cold model fetch behind a future
    # button-triggered flavor; the 7200 test mark matches the class ladder.
    str(EXAMPLE_DIR / "marimo" / "benchmark" / "benchmark_demo.py"): {
        "timeout_s": 3600,
    },
    str(EXAMPLE_DIR / "marimo" / "finetune" / "finetune_demo.py"): {
        "timeout_s": 3600,
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


def run_marimo_app(
    app_path: Path,
    sandbox: Path,
    timeout: int = 1200,
    artifact_dir: Path | None = None,
) -> Path:
    """Execute a marimo app headlessly via ``marimo export html``.

    Export-html is the 05-FEASIBILITY.md flavor decision: a deterministic
    exit code plus a concrete HTML artifact per app.  The subprocess runs
    with ``cwd=sandbox`` -- a seeded tmp copy of the app's directory (the
    caller seeds it; this helper seeds nothing) -- so sibling inputs
    (configs, ``.xlsx`` model lists) resolve inside the sandbox and the
    repo tree is never touched.

    Args:
        app_path: marimo app to export (read from the repo tree; never
            modified) -- its basename is the CLI argument, so the app must
            also exist inside the sandbox copy.
        sandbox: working directory for the export subprocess.
        timeout: wall budget in seconds for the export; must stay
            strictly below the per-test pytest timeout mark.
        artifact_dir: directory receiving the HTML artifact and, on
            failure, an error artifact (returncode + stdout + stderr).
            When ``None`` the HTML lands in the sandbox itself.

    Returns:
        Path of the exported HTML artifact (always larger than 1000
        bytes on success).

    Raises:
        ValueError: no ``marimo`` console script resolvable next to the
            running interpreter nor on ``PATH``.
        subprocess.TimeoutExpired: the export exceeded ``timeout``; the
            child is killed first (``subprocess.run`` semantics: the
            timeout expiry kills and waits for the child, then the
            exception is re-raised).
        AssertionError: non-zero exit, or the HTML artifact is missing or
            undersized (< 1000 bytes); the error artifact carries the
            captured stdout/stderr for the census evidence.
    """
    marimo_bin = Path(sys.executable).with_name("marimo")
    if not marimo_bin.is_file():
        which = shutil.which("marimo")
        if which is None:
            raise ValueError(
                "marimo CLI not found: expected a console script next to "
                f"{sys.executable} or a marimo on PATH -- the notebook extra "
                "must be installed in the active venv"
            )
        marimo_bin = Path(which)
    out_dir = artifact_dir if artifact_dir is not None else sandbox
    out_dir.mkdir(parents=True, exist_ok=True)
    html_out = out_dir / "_census_export.html"
    if html_out.exists():
        html_out.unlink()  # a stale file from an earlier run would fake a fresh export
    cmd = [str(marimo_bin), "export", "html", app_path.name, "-o", str(html_out)]
    saved_env = {key: os.environ.get(key) for key in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    try:
        # ruff: ignore[subprocess-without-shell-equals-true]
        proc = subprocess.run(
            cmd,
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    artifact_bytes = html_out.stat().st_size if html_out.is_file() else 0
    if proc.returncode != 0 or artifact_bytes < 1000:
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / f"{app_path.stem}.export.error.txt").write_text(
                f"returncode={proc.returncode}\n\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
                encoding="utf-8",
            )
        stderr_tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise AssertionError(
            f"marimo export failed for {app_path.name}: returncode={proc.returncode}, "
            f"artifact {html_out} is {artifact_bytes} bytes (must exceed 1000); "
            f"last stderr lines:\n{stderr_tail}"
        )
    return html_out


def run_example_script(
    script_path: Path,
    sandbox: Path,
    timeout: int = 3000,
    artifact_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an example helper script with the venv interpreter in a sandbox.

    The script runs as ``[sys.executable, script_path.name]`` with
    ``cwd=sandbox`` -- a seeded tmp copy of its directory -- so sibling
    inputs (configs, downloaded data) resolve inside the sandbox and the
    repo tree is never touched.

    Args:
        script_path: script to run (read from the repo tree; never
            modified) -- its basename is the argv, so the script must
            also exist inside the sandbox copy.
        sandbox: working directory for the subprocess.
        timeout: wall budget in seconds; must stay strictly below the
            per-test pytest timeout mark.
        artifact_dir: when given, the combined stdout/stderr run log is
            ALWAYS written here (success evidence for the census, not
            only failure artifacts).

    Returns:
        The completed process (returncode is 0 -- non-zero exits raise).

    Raises:
        subprocess.TimeoutExpired: the script exceeded ``timeout``; the
            child is killed first (``subprocess.run`` semantics).
        AssertionError: non-zero exit; the message carries the script
            name, the returncode and the tail of stderr.
    """
    cmd = [sys.executable, script_path.name]
    saved_env = {key: os.environ.get(key) for key in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    try:
        # ruff: ignore[subprocess-without-shell-equals-true]
        proc = subprocess.run(
            cmd,
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{script_path.stem}.run.log").write_text(
            f"returncode={proc.returncode}\n\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
            encoding="utf-8",
        )
    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise AssertionError(
            f"example script failed: {script_path.name} returncode={proc.returncode}; "
            f"last stderr lines:\n{stderr_tail}"
        )
    return proc


def _scoped_dirty_lines(paths: tuple[str, ...]) -> frozenset[str]:
    """Return the scoped ``git status --porcelain`` lines for *paths*.

    Fails when the guard itself could not run (not a git repo, missing
    git binary, contended index.lock): an empty stdout from a failed
    call would otherwise read as "clean" and silently disable the
    tripwire.
    """
    # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"git status failed (rc={result.returncode}): {result.stderr.strip()}"
    )
    return frozenset(line for line in result.stdout.splitlines() if line.strip())


# Pre-existing dirt under the guarded paths, captured at import time
# (collection, before any test executes).  The tripwire's contract is to
# catch dirt written by THIS harness's executions, not to police the
# developer's working tree: the owner's IDE session keeps execution-output
# churn in three tracked notebooks (documented owner-sanctioned baseline,
# 05-05 dispatch note), which is none of the harness's business.  Same
# delta-zero principle as _kernel_count in test_notebook_execution.py:
# compare against a captured pre-session baseline, never against absolute
# zero; on a clean checkout (CI) the baseline is empty and the assert is
# exactly the original absolute check.  Narrow blind spot, accepted:
# further churn on a file ALREADY dirty at import time is indistinguish-
# able from its baseline line -- every execution runs in a tmp sandbox,
# so an actual write into the repo tree is prevented by construction and
# this guard stays belt-and-braces.
_PREEXISTING_DIRTY: frozenset[str] = _scoped_dirty_lines(("example", "docs/example"))


def assert_tree_clean(paths: tuple[str, ...] = ("example", "docs/example")) -> None:
    """Fail if an execution wrote anything new into the guarded repo paths.

    Runs a scoped ``git status --porcelain -- <paths>`` from the repo
    root and compares the result against :data:`_PREEXISTING_DIRTY`
    (captured at import time): any line NOT present at session start
    means an execution of this session dirtied the tree, which is
    exactly the false green this harness exists to prevent.  Pre-existing
    working-tree modifications that predate the session are tolerated
    (delta-zero).  Scoped on purpose: untracked ``.planning/`` /
    ``.gsd/`` working dirs are none of the harness's business.

    Args:
        paths: repo-relative pathspecs to watch.
    """
    current = _scoped_dirty_lines(paths)
    baseline = _PREEXISTING_DIRTY if paths == ("example", "docs/example") else frozenset()
    new_dirt = sorted(current - baseline)
    assert not new_dirt, f"notebook execution dirtied the repo tree under {paths}:\n{new_dirt}"


def environment_unavailable_skip(action: str, evidence: str) -> None:
    """Skip with the stable ``environment-unavailable:`` prefix.

    The narrow typed-skip path for execution tests whose environment
    cannot provide what the notebook needs (per D-06: only after the
    notebook variant AND the smallest viable variant both failed, with
    the failure recorded).  Nothing calls this yet in Phase 5; the
    prefix is registered now so Phase 8 skip decisions already carry an
    allowlist contract.

    Args:
        action: stable label naming what the test was attempting.
        evidence: recorded infeasibility evidence (e.g. the exact error
            text), so the junit skip message carries the decision basis.
    """
    pytest.skip(f"environment-unavailable: {action} ({evidence})")


def optional_dep_skip(action: str, evidence: str) -> None:
    """Skip with the stable ``optional-dep:`` prefix.

    The narrow typed-skip path for execution tests blocked on an
    optional dependency that is not installed on the executing box.
    Nothing calls this yet in Phase 5; the prefix is registered now so
    Phase 8 skip decisions already carry an allowlist contract.

    Args:
        action: stable label naming what the test was attempting.
        evidence: the missing-dependency evidence (e.g. ImportError text).
    """
    pytest.skip(f"optional-dep: {action} ({evidence})")


def network_unavailable_skip(action: str, evidence: str) -> None:
    """skip with the stable ``network-unavailable:`` prefix.

    The narrow typed-skip path for execution tests blocked on a local
    network service the notebook needs (the MCP-01 convention from
    ``dnallm/mcp/tests/_network_skip.py``; the prefix has been registered
    in ``tests/expected_skips.yaml`` since 05-01).  Used by the 05-06
    gated layer for the ollama/mcp-server example notebooks: the skip
    fires only when a required endpoint is genuinely unreachable, with
    the live probe results recorded in the message.

    Args:
        action: stable label naming what the test was attempting.
        evidence: the live probe evidence (endpoint URLs + results), so
            the junit skip message carries the decision basis.
    """
    pytest.skip(f"network-unavailable: {action} ({evidence})")
