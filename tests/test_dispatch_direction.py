"""Dispatch-direction contracts for the platform seams (WPAR-06).

Phase 16 (16-01) structure + runtime tests for the FIRST dispatch seam --
the CRE notebook's jaccard agreement cell (plant_helixseek_cre.ipynb cell
11). The file proves, by execution and by AST inspection:

* the seam dispatches on the platform equality predicate with a fail-loud
  direction: a win32 platform runs the numpy evaluator in-process with
  ZERO subprocess invocation (contract A), while a POSIX platform with an
  unusable CLI binary RAISES instead of quietly taking the numpy path
  (contract B -- the WPAR-06 "cannot silently execute on Linux" clause);
* no exception-based import-fallback ladder wraps the dispatch (an
  ImportError on Linux must fail, never fall back);
* the showcase gate's PATH probe stays fail-loud on POSIX and is preceded
  by the platform guard that routes win32 past it (no skip machinery).

Posture (copied from tests/test_genomic_intervals.py): platform-neutral,
no bedtools binary, no subprocess of our own, no skip machinery and no
custom markers anywhere in this file -- it auto-runs on test-windows and
every Linux leg. Root placement keeps the examples census triple
(216/217) untouched.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import pytest

from dnallm.utils import jaccard

REPO_ROOT = Path(__file__).resolve().parents[1]
CRE_NB = REPO_ROOT / "example" / "notebooks" / "plant_helixseek_cre" / "plant_helixseek_cre.ipynb"
SHOWCASE_TEST = REPO_ROOT / "tests" / "examples" / "test_plant_helixseek_showcase.py"
CELL_11 = 11
MD_CELLS = (0, 10, 22)


def _load_nb() -> dict:
    """Read the committed CRE notebook as plain JSON (kernel-free reader)."""
    return json.loads(CRE_NB.read_text(encoding="utf-8"))


def _cell_text(cell: dict) -> str:
    """Join a notebook cell's source list into one string."""
    return "".join(cell["source"])


def _cell11_source() -> str:
    nb = _load_nb()
    cell = nb["cells"][CELL_11]
    assert cell["cell_type"] == "code", "CRE cell 11 must stay a code cell"
    return _cell_text(cell)


def _cell11_tree() -> ast.Module:
    return ast.parse(_cell11_source())


def _dispatch_if(tree: ast.Module) -> ast.If:
    """Return the ``if sys.platform == "win32":`` node of cell 11."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left, (right,) = test.left, test.comparators
        platform_attr = (
            isinstance(left, ast.Attribute)
            and left.attr == "platform"
            and isinstance(left.value, ast.Name)
            and left.value.id == "sys"
        )
        win32_literal = isinstance(right, ast.Constant) and right.value == "win32"
        if platform_attr and win32_literal:
            return node
    pytest.fail(
        'cell 11 has no `if sys.platform == "win32":` dispatch node -- the seam '
        "must branch on the platform equality predicate (WPAR-06)"
    )


# ---------------------------------------------------------------------------
# Structure tests (CRE seam)
# ---------------------------------------------------------------------------


def test_cell11_dispatches_on_platform_equality() -> None:
    """Cell 11 branches on the win32 platform equality predicate (AS-1)."""
    src = _cell11_source()
    assert "sys.platform" in src, "platform attribute missing from cell 11"
    assert 'sys.platform == "win32"' in src, (
        'cell 11 must dispatch on the equality predicate `sys.platform == "win32"` '
        "-- never an exception-based fallback (WPAR-06)"
    )


def test_cell11_posix_branch_keeps_cli_call() -> None:
    """The POSIX (else) branch still calls the CLI tool with check semantics."""
    node = _dispatch_if(_cell11_tree())
    cli_calls = [
        call
        for stmt in node.orelse
        for call in ast.walk(stmt)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
    ]
    assert cli_calls, "POSIX branch no longer calls subprocess.run (the CLI truth path)"
    argv_consts = {
        element.value
        for call in cli_calls
        for element in ast.walk(call)
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    assert "bedtools" in argv_consts, "POSIX branch argv no longer names the bedtools CLI"
    assert "jaccard" in argv_consts, "POSIX branch argv no longer names the jaccard subcommand"
    check_true = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run"
        and any(
            kw.arg == "check" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in call.keywords
        )
        for stmt in node.orelse
        for call in ast.walk(stmt)
        if isinstance(call, ast.Call)
    )
    assert check_true, "POSIX branch subprocess.run lost check=True (fail-loud CLI call)"


def test_cell11_no_try_wrapped_import_ladder() -> None:
    """No try-statement anywhere in cell 11 encloses an import (WPAR-06).

    The exception-based import-fallback ladder
    (``try: from pybedtools ... except ImportError: <numpy>``) is the
    prohibited dispatch shape: it would let the numpy path silently
    execute on a Linux box whose CLI wrapper broke, corrupting the
    bedtools truth chain. Platform equality is the only sanctioned
    predicate, so a try-wrapped import must be structurally impossible.
    """
    offenders: list[int] = []
    for node in ast.walk(_cell11_tree()):
        if isinstance(node, (ast.Try, ast.TryStar)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    offenders.append(sub.lineno)
    assert not offenders, (
        f"cell 11 contains a try-wrapped import at line(s) {sorted(set(offenders))} -- "
        "the exception-based import-fallback ladder is the WPAR-06 target and must "
        "not exist (dispatch on sys.platform instead)"
    )


def test_cell11_shared_print_keeps_the_stream_format() -> None:
    """The branch-shared trailing print keeps the exact 4-decimal format.

    The nightly stream regex ``^jaccard=([0-9.eE+-]+)`` and CRE cell 21's
    comparison parse this one printed line -- it is the normalization
    point between the bedtools 6-dp stdout and the numpy full-precision
    float (Pitfall 8).
    """
    src = _cell11_source()
    assert 'print(f"jaccard={jaccard_value:.4f}")' in src, (
        "cell 11 must keep the exact branch-shared print "
        '`print(f"jaccard={jaccard_value:.4f}")` (Pitfall 8)'
    )


def test_md_cells_stay_bedtools_worded() -> None:
    """md cells 0/10/22 keep their frozen bedtools-worded conventions."""
    nb = _load_nb()
    for index in MD_CELLS:
        cell = nb["cells"][index]
        assert cell["cell_type"] == "markdown", f"CRE cell {index} must stay markdown"
        assert "bedtools jaccard" in _cell_text(cell), (
            f"CRE md cell {index} lost its bedtools-worded line (owner directive: "
            "frozen conventions; the dispatch seam is cell 11 only)"
        )


# ---------------------------------------------------------------------------
# Runtime contract A -- win32 dispatch executes the numpy evaluator, zero subprocess
# ---------------------------------------------------------------------------

_BED_A = "Chr1\t100\t200\tp1\t0\t+\nChr1\t1000\t1100\tp2\t0\t+\n"
# One overlapping B row plus one NON-overlapping B row: the union math is
# non-trivial (a union-only contribution no branch could fake).
_BED_B = "Chr1\t150\t260\tt1\t0\t+\nChr1\t5000\t5100\tt2\t0\t+\n"


def test_win32_contract_cell11_runs_evaluator_zero_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a monkeypatched win32 platform cell 11 computes in-process."""
    bed_a = tmp_path / "a.bed"
    bed_b = tmp_path / "b.bed"
    bed_a.write_text(_BED_A, encoding="utf-8")
    bed_b.write_text(_BED_B, encoding="utf-8")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run invoked on the win32 dispatch path")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(sys, "platform", "win32")

    buf = io.StringIO()
    namespace = {"predicted_bed_path": bed_a, "truth_bed_path": bed_b}
    with contextlib.redirect_stdout(buf):
        exec(compile(_cell11_source(), "cre-cell11-win32-contract", "exec"), namespace)  # ruff: ignore[exec-builtin]

    expected = f"jaccard={jaccard(bed_a, bed_b)['jaccard']:.4f}"
    assert buf.getvalue().strip() == expected, (
        f"win32 cell 11 printed {buf.getvalue().strip()!r}; the numpy evaluator "
        f"formatted with the cell's 4-decimal format prints {expected!r}"
    )


# ---------------------------------------------------------------------------
# Runtime contract B -- POSIX fail-loud (cannot silently execute on Linux)
# ---------------------------------------------------------------------------


def test_posix_contract_cell11_fails_loud_when_cli_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A POSIX platform with an unusable CLI binary fails loudly (WPAR-06).

    The process runner is poisoned with a distinctive sentinel error
    (standing in for a missing/unusable bedtools binary); exec'ing the
    cell on a POSIX platform MUST propagate that sentinel. If the cell
    silently substituted the numpy evaluator, nothing would raise and
    this test fails -- that is the exact WPAR-06 violation.
    """
    bed_a = tmp_path / "a.bed"
    bed_b = tmp_path / "b.bed"
    bed_a.write_text(_BED_A, encoding="utf-8")
    bed_b.write_text(_BED_B, encoding="utf-8")

    def _sentinel(*args: object, **kwargs: object) -> None:
        raise RuntimeError("SENTINEL-CLI-UNUSABLE: bedtools subprocess unavailable")

    monkeypatch.setattr(subprocess, "run", _sentinel)
    monkeypatch.setattr(sys, "platform", "linux")

    namespace = {"predicted_bed_path": bed_a, "truth_bed_path": bed_b}
    with pytest.raises(RuntimeError, match="SENTINEL-CLI-UNUSABLE"):
        exec(compile(_cell11_source(), "cre-cell11-posix-contract", "exec"), namespace)  # ruff: ignore[exec-builtin]


# ---------------------------------------------------------------------------
# Gate structure assertion (showcase gate stays fail-loud, guard precedes probe)
# ---------------------------------------------------------------------------


def test_showcase_gate_guard_precedes_path_probe() -> None:
    """The showcase gate: platform guard precedes the PATH probe, probe kept."""
    text = SHOWCASE_TEST.read_text(encoding="utf-8")
    probe = 'assert shutil.which("bedtools") is not None'
    guard = 'if sys.platform != "win32":'

    lines = text.splitlines()
    probe_hits = [i for i, line in enumerate(lines) if probe in line]
    assert probe_hits, "the showcase gate's hard PATH-probe assert is gone (POSIX fail-loud lost)"
    probe_line = probe_hits[0]

    fn_start = max(i for i in range(probe_line + 1) if lines[i].startswith("def "))
    fn_end = next(
        (
            probe_line + offset
            for offset in range(1, len(lines) - probe_line)
            if lines[probe_line + offset].startswith(("def ", "class "))
        ),
        len(lines),
    )
    region = lines[fn_start:fn_end]

    guard_hits = [i for i, line in enumerate(region) if line.strip() == guard]
    assert guard_hits, (
        "the inverse platform guard is missing from the gate function -- Windows "
        "cannot route past the PATH probe (WPAR-04)"
    )
    region_probe = [i for i, line in enumerate(region) if probe in line]
    assert region_probe, "PATH probe must stay inside the gate function"
    assert guard_hits[0] < region_probe[0], (
        "the platform guard must textually precede the PATH probe inside the gate "
        "function -- a future edit cannot make the gate ever-green on either side"
    )

    gate_text = "\n".join(region)
    for banned in ("pytest.skip", "optional_dep_skip", "environment_unavailable_skip"):
        assert banned not in gate_text, (
            f"skip machinery ({banned}) inside the showcase gate function -- the "
            "Windows route must proceed into numpy execution, never skip (WPAR-04)"
        )
