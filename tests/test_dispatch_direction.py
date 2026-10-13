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
import importlib.util
import io
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import pytest

from dnallm.utils import jaccard, loj_intersect

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


# ---------------------------------------------------------------------------
# NER script runtime contracts (16-02) -- generate_bpe_dataset.py seam
# ---------------------------------------------------------------------------

NER_SCRIPT = REPO_ROOT / "example" / "notebooks" / "finetune_NER_task" / "generate_bpe_dataset.py"


def _is_win32_platform_test(test: ast.expr) -> bool:
    """True for the ``sys.platform == "win32"`` equality predicate node."""
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
    ):
        return False
    left, (right,) = test.left, test.comparators
    platform_attr = (
        isinstance(left, ast.Attribute)
        and left.attr == "platform"
        and isinstance(left.value, ast.Name)
        and left.value.id == "sys"
    )
    return platform_attr and isinstance(right, ast.Constant) and right.value == "win32"


def _script_module_defs(names: tuple[str, ...]) -> dict[str, str]:
    """AST-extract named top-level def/class source from the NER script.

    The script is a linear example (module-level execution), so the runtime
    contracts exec the extracted segments instead of importing the module
    (16-02 AS-2: the module-level helper is the extraction target).
    """
    src = NER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = {
        node.name: ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    }
    missing = sorted(set(names) - set(found))
    assert not missing, (
        f"generate_bpe_dataset.py must define {sorted(names)} at module level "
        "above the consumer -- the inline adapter + dispatch helper are the "
        f"contract-test extraction targets (16-02 AS-1/AS-2); missing: {missing}"
    )
    return found


def _script_import_seam_segment() -> str:
    """Return the source segment of the script's platform import seam."""
    src = NER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If) and _is_win32_platform_test(node.test):
            return ast.get_source_segment(src, node) or ""
    pytest.fail(
        'generate_bpe_dataset.py has no top-level `if sys.platform == "win32":` '
        "import seam -- the bare pybedtools import must be the "
        "platform-conditional pair (WPAR-03, fail-loud per WPAR-06)"
    )


def _exec_win32_loj_rows(bed_a: Path, bed_b: Path, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Run the script's extracted adapter + dispatch helper under win32."""
    defs = _script_module_defs(("LojRow", "_loj_rows"))
    monkeypatch.setattr(sys, "platform", "win32")
    namespace: dict[str, object] = {"sys": sys, "loj_intersect": loj_intersect}
    exec(  # ruff: ignore[exec-builtin]
        compile(
            "\n\n".join(defs[name] for name in ("LojRow", "_loj_rows")),
            "ner-script-win32-contract",
            "exec",
        ),
        namespace,
    )
    return namespace["_loj_rows"](bed_a, bed_b)


# A: 5 BED6 token rows across three genes. Rows 2-3 are CONSECUTIVE no-hit
# rows inside geneA (Pitfall 1's guaranteed-crash class: a re-encoded null
# score slot KeyErrors the label maps on the second consecutive intergenic
# token); row 4 is the multi-hit row (two overlapping B rows -> join
# multiplicity must survive).
_NER_A_BED = (
    "Chr1\t100\t200\tATCG\tgeneA\t+\n"
    "Chr1\t300\t380\tGG\tgeneA\t+\n"
    "Chr1\t390\t440\tCC\tgeneA\t+\n"
    "Chr1\t505\t550\tTTTT\tgeneB\t-\n"
    "Chr1\t900\t950\tAA\tgeneC\t+\n"
)
_NER_B_BED = (
    "Chr1\t150\t260\tgeneX\texon\t+\n"
    "Chr1\t500\t560\tgeneY\texon\t-\n"
    "Chr1\t520\t555\tgeneZ\tintron\t+\n"
)

# bedtools v2.31.1 `intersect -loj` truth over the pair above, measured via
# pybedtools 0.12.1 on the parity-record box (2026-10-13; same provenance
# discipline as tests/fixtures/genomic_intervals/PROVENANCE.md). A-file
# order, one line per A/B hit pair in B-file order, and the count-6 null
# sextet ('.', '-1', '-1', '.', '-1', '.') VERBATIM on no-hit rows.
_NER_EXPECTED_FIELDS = [
    ["Chr1", "100", "200", "ATCG", "geneA", "+", "Chr1", "150", "260", "geneX", "exon", "+"],
    ["Chr1", "300", "380", "GG", "geneA", "+", ".", "-1", "-1", ".", "-1", "."],
    ["Chr1", "390", "440", "CC", "geneA", "+", ".", "-1", "-1", ".", "-1", "."],
    ["Chr1", "505", "550", "TTTT", "geneB", "-", "Chr1", "500", "560", "geneY", "exon", "-"],
    ["Chr1", "505", "550", "TTTT", "geneB", "-", "Chr1", "520", "555", "geneZ", "intron", "+"],
    ["Chr1", "900", "950", "AA", "geneC", "+", ".", "-1", "-1", ".", "-1", "."],
]


def test_ner_script_win32_rows_match_frozen_loj_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Win32 dispatch rows reproduce the pinned 12-column loj truth exactly.

    Same length, same order, element-wise equal fields -- including the two
    consecutive no-hit null sextets and the multi-hit A row emitted twice
    (the consumer's gene flush and label maps depend on all three).
    """
    bed_a = tmp_path / "a.bed"
    bed_b = tmp_path / "b.bed"
    bed_a.write_text(_NER_A_BED, encoding="utf-8")
    bed_b.write_text(_NER_B_BED, encoding="utf-8")

    rows = _exec_win32_loj_rows(bed_a, bed_b, monkeypatch)

    assert [list(row.fields) for row in rows] == _NER_EXPECTED_FIELDS
    for row in rows:
        # Type contract (Pitfall 2): start/end are ints, every field a str.
        assert isinstance(row.start, int)
        assert isinstance(row.end, int)
        assert isinstance(row.chrom, str)
        assert isinstance(row.name, str)
        assert all(isinstance(column, str) for column in row.fields)
    # The count-6 null score slot rides VERBATIM (Pitfall 1): the consumer's
    # intergenic check is `name == "-1"` on fields[10] -- never a re-encoded
    # "." there.
    assert rows[1].fields[10] == "-1"
    assert rows[2].fields[10] == "-1"


def test_ner_script_win32_rows_match_live_pybedtools_loj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-row equality against a LIVE pybedtools loj (the Linux truth side).

    Runs only where the full truth chain is installed (pybedtools importable
    AND the bedtools binary on PATH: the parity-record dev box with the
    nightly prefix exported, coverage-nightly, example-nightly). Elsewhere
    the frozen-truth test above carries the identical contract; this is a
    conditional contract-strengthening, not skip machinery (no pytest.skip,
    no audit surface -- the file posture stands).
    """
    if importlib.util.find_spec("pybedtools") is None or shutil.which("bedtools") is None:
        return

    import pybedtools

    bed_a = tmp_path / "a.bed"
    bed_b = tmp_path / "b.bed"
    bed_a.write_text(_NER_A_BED, encoding="utf-8")
    bed_b.write_text(_NER_B_BED, encoding="utf-8")

    rows = _exec_win32_loj_rows(bed_a, bed_b, monkeypatch)
    live = [
        list(iv.fields) for iv in pybedtools.BedTool(str(bed_a)).intersect(str(bed_b), loj=True)
    ]

    assert len(rows) == len(live), (
        f"win32 dispatch returned {len(rows)} rows, live loj returned {len(live)}"
    )
    for index, (row, truth) in enumerate(zip(rows, live, strict=True)):
        assert list(row.fields) == truth, (
            f"row {index} diverges from live pybedtools loj: {list(row.fields)} != {truth}"
        )


def test_ner_script_posix_seam_import_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """A POSIX platform with a broken pybedtools propagates the ImportError.

    The WPAR-06 "cannot silently execute on Linux" clause for the NER seam:
    with the platform attribute monkeypatched to a POSIX value and the
    pybedtools entry in sys.modules poisoned (None), exec'ing the seam's
    conditional import MUST raise. An exception-based fallback to the numpy
    evaluator would swallow this and corrupt the bedtools truth chain --
    that ladder shape is the exact prohibition.
    """
    seam = _script_import_seam_segment()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "pybedtools", None)
    namespace: dict[str, object] = {"sys": sys}
    with pytest.raises(ImportError):
        exec(  # ruff: ignore[exec-builtin]
            compile(seam, "ner-script-seam-posix-contract", "exec"), namespace
        )
