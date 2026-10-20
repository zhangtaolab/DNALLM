"""Dispatch-direction contracts for the platform seams (WPAR-06).

Phase 16 (16-01) structure + runtime tests for the FIRST dispatch seam --
the CRE notebook's jaccard agreement cell (plant_helixseek_cre.ipynb cell
11) -- extended by 16-02 with the NER seams (generate_bpe_dataset.py
import/acquisition + data_generation_and_inference.ipynb cells 0/1/2/8)
and both NER test gates. The file proves, by execution and by AST
inspection:

* the seam dispatches on the platform equality predicate with a fail-loud
  direction: a win32 platform runs the numpy evaluator in-process with
  ZERO subprocess invocation (contract A), while a POSIX platform with an
  unusable CLI binary RAISES instead of quietly taking the numpy path
  (contract B -- the WPAR-06 "cannot silently execute on Linux" clause);
* the NER win32 dispatch reproduces the pinned bedtools loj rows exactly
  (12-column fields element-wise, null sextet verbatim, A-file order,
  join multiplicity, pybedtools row types) and equals a LIVE pybedtools
  loj wherever the truth chain is installed;
* no exception-based import-fallback ladder wraps the dispatch (an
  ImportError on Linux must fail, never fall back);
* the duplicated inline NER adapter (script + cell 8) stays textually
  identical (the AS-1 drift guard);
* the three gates' probes stay fail-loud on POSIX and are preceded by the
  platform guard that routes win32 past them (no skip machinery on any
  Windows route).

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


# ---------------------------------------------------------------------------
# NER structure contracts (16-02) -- seam shape, notebook cells, both gates
# ---------------------------------------------------------------------------

NER_NB = (
    REPO_ROOT
    / "example"
    / "notebooks"
    / "finetune_NER_task"
    / "data_generation_and_inference.ipynb"
)
NER_NB_ID = "notebooks/finetune_NER_task/data_generation_and_inference.ipynb"
SCRIPT_GATE_TEST = REPO_ROOT / "tests" / "examples" / "test_script_execution.py"
NOTEBOOK_GATE_TEST = REPO_ROOT / "tests" / "examples" / "test_notebook_execution.py"


def _ner_cell(index: int) -> str:
    """Return one NER notebook code cell's source (kernel-free JSON reader)."""
    nb = json.loads(NER_NB.read_text(encoding="utf-8"))
    cell = nb["cells"][index]
    assert cell["cell_type"] == "code", f"NER cell {index} must stay a code cell"
    return _cell_text(cell)


def _imports_in(stmts: list[ast.stmt]) -> list[tuple[str | None, tuple[str, ...]]]:
    """Collect (module, names) for every import inside the given statements."""
    found: list[tuple[str | None, tuple[str, ...]]] = []
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.ImportFrom):
                found.append((node.module or "", tuple(alias.name for alias in node.names)))
            elif isinstance(node, ast.Import):
                found.append((None, tuple(alias.name for alias in node.names)))
    return found


def _top_level_win32_if(tree: ast.Module) -> ast.If:
    """Return the first top-level ``if sys.platform == "win32":`` node."""
    for node in tree.body:
        if isinstance(node, ast.If) and _is_win32_platform_test(node.test):
            return node
    pytest.fail('no top-level `if sys.platform == "win32":` node in the parsed source')


def _normalize_ws(source: str) -> str:
    """Whitespace-normalize source for the script-vs-cell drift comparison."""
    return "\n".join(line.strip() for line in source.splitlines())


def _function_region(text: str, signature: str) -> list[str]:
    """Lines of one top-level function, from its def to the next top-level line."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(signature)]
    assert len(starts) == 1, f"expected exactly one {signature!r} definition"
    start = starts[0]
    end = next(
        (
            start + offset
            for offset in range(1, len(lines) - start)
            if lines[start + offset] and not lines[start + offset][0].isspace()
        ),
        len(lines),
    )
    return lines[start:end]


def test_ner_script_import_seam_is_the_conditional_pair() -> None:
    """Script seam: win32 imports the evaluator, POSIX imports BedTool verbatim."""
    src = NER_SCRIPT.read_text(encoding="utf-8")
    assert 'sys.platform == "win32"' in src, "script lost the win32 predicate"
    seam = _top_level_win32_if(ast.parse(src))
    assert _imports_in(seam.body) == [("dnallm.utils", ("loj_intersect",))], (
        "win32 branch must import the loj evaluator from the dnallm.utils package surface"
    )
    assert _imports_in(seam.orelse) == [("pybedtools", ("BedTool",))], (
        "POSIX branch must keep the BedTool import verbatim (the bedtools truth chain)"
    )


def test_ner_nb_cell2_import_seam_matches_script_shape() -> None:
    """Notebook cell 2 carries the same conditional import pair as the script."""
    cell2 = _ner_cell(2)
    assert 'sys.platform == "win32"' in cell2, "cell 2 lost the win32 predicate"
    seam = _top_level_win32_if(ast.parse(cell2))
    assert _imports_in(seam.body) == [("dnallm.utils", ("loj_intersect",))]
    assert _imports_in(seam.orelse) == [("pybedtools", ("BedTool",))]


def test_ner_seams_have_no_try_wrapped_import_ladder() -> None:
    """No try-statement encloses an import in the script or cell 2 (WPAR-06).

    The exception-based import-fallback ladder is the prohibited dispatch
    shape -- it would let the numpy path silently execute on a Linux box
    whose pybedtools broke. (The script's two pre-existing try blocks wrap
    FASTA slicing, never imports.)
    """
    offenders: list[str] = []
    for label, source in (
        ("script", NER_SCRIPT.read_text(encoding="utf-8")),
        ("cell 2", _ner_cell(2)),
    ):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Try, ast.TryStar)):
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        offenders.append(f"{label}:{sub.lineno}")
    assert not offenders, (
        f"try-wrapped import at {offenders} -- the exception-based fallback ladder "
        "is the WPAR-06 target and must not exist (dispatch on sys.platform instead)"
    )


def test_ner_adapter_textually_identical_script_and_cell8() -> None:
    """AS-1 drift guard: the duplicated inline adapter must not diverge.

    The adapter + helper are deliberately INLINE in both the script and
    notebook cell 8 (the locked dnallm.utils surface stays untouched), so
    this textual-identity assertion is the drift pin between the copies.
    """
    script_defs = _script_module_defs(("LojRow", "_loj_rows"))
    cell8 = _ner_cell(8)
    cell_defs = {
        node.name: ast.get_source_segment(cell8, node)
        for node in ast.parse(cell8).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("LojRow", "_loj_rows")
    }
    assert set(cell_defs) == {"LojRow", "_loj_rows"}, "cell 8 lost the adapter/helper definitions"
    for name in ("LojRow", "_loj_rows"):
        assert _normalize_ws(script_defs[name]) == _normalize_ws(cell_defs[name]), (
            f"{name} drifted between generate_bpe_dataset.py and notebook cell 8 -- "
            "the two inline copies must stay textually identical (AS-1)"
        )


def test_ner_cell8_acquisition_dispatched_and_null_check_intact() -> None:
    """Cell 8 acquires through the helper and keeps the null-score check."""
    cell8 = _ner_cell(8)
    assert "_loj_rows(tokens_bed, annotation_bed)" in cell8, (
        "cell 8's acquisition statement is not dispatched through _loj_rows"
    )
    assert "BedTool(tokens_bed)" not in cell8, (
        "cell 8 still acquires through pybedtools directly (seam incomplete)"
    )
    script = NER_SCRIPT.read_text(encoding="utf-8")
    for label, source in (("script", script), ("cell 8", cell8)):
        assert 'name == "-1"' in source, (
            f"{label} lost the intergenic null-score check -- fields[10] accessor is "
            "load-bearing (Pitfall 1)"
        )


def test_ner_install_and_verify_cells_are_posix_guarded() -> None:
    """Cells 0/1: POSIX install lines verbatim; Windows touches no POSIX tooling.

    T-16-06: the POSIX branch must keep both install lines byte-verbatim
    and the Windows branch must attempt no pybedtools/bedtools tooling;
    cell 1's which-magic lives inside the POSIX guard (Pitfall 5).
    """
    cell0 = _ner_cell(0)
    cell1 = _ner_cell(1)
    guard = 'if sys.platform != "win32":'
    assert guard in cell0, "cell 0 lost the POSIX guard"
    assert guard in cell1, "cell 1 lost the POSIX guard (Pitfall 5)"
    assert "!brew install bedtools" in cell0, "cell 0 lost the brew bedtools install line"
    assert "!uv pip install pyfastx pybedtools" in cell0, (
        "cell 0 lost the uv pip line naming both packages verbatim"
    )
    windows_branch = cell0.split("else:", 1)[1]
    assert "!uv pip install pyfastx" in windows_branch, "Windows branch must still install pyfastx"
    assert "pybedtools" not in windows_branch, (
        "Windows branch attempts pybedtools tooling (WPAR-03 install clause)"
    )
    assert "bedtools" not in windows_branch, (
        "Windows branch attempts bedtools tooling (WPAR-03 install clause)"
    )
    for label, source in (("cell 0", cell0), ("cell 1", cell1)):
        lines = [line for line in source.splitlines() if line.strip()]
        guard_index = next((i for i, line in enumerate(lines) if line.strip() == guard), None)
        magic_index = next(i for i, line in enumerate(lines) if line.strip().startswith("!"))
        assert guard_index is not None, f"{label}: no POSIX guard line found"
        assert magic_index > guard_index, (
            f"{label}: a shell magic sits outside the POSIX guard body"
        )
        assert lines[magic_index].startswith("    "), (
            f"{label}: the magic must be indented INSIDE the guard (Pitfall 5 for cell 1)"
        )
    assert "!which intersectBed" in cell1, "cell 1 lost the intersectBed verify magic"


# ---------------------------------------------------------------------------
# NER gate structure assertions (16-02) -- guard precedes probe, no win32 skip
# ---------------------------------------------------------------------------


def test_script_gate_guard_precedes_probe() -> None:
    """Script gate: win32 guard first statement, probe + typed skip retained."""
    region = _function_region(
        SCRIPT_GATE_TEST.read_text(encoding="utf-8"), "def _gate_script_pybedtools"
    )
    guard_hits = [
        i for i, line in enumerate(region) if line.strip() == 'if sys.platform == "win32":'
    ]
    probe_hits = [i for i, line in enumerate(region) if 'find_spec("pybedtools")' in line]
    assert guard_hits, "script gate lost the win32 early-return (WPAR-04)"
    assert probe_hits, "script gate lost the find_spec probe (POSIX fail-loud)"
    assert guard_hits[0] < probe_hits[0], "the win32 guard must precede the probe"
    assert "optional_dep_skip(" in "\n".join(region), "typed skip call not retained"

    # No skip call reachable on the win32 path: between guard and probe
    # there is only the plain return.
    between = [line for line in region[guard_hits[0] + 1 : probe_hits[0]] if line.strip()]
    assert between, "the win32 guard body is empty -- the early return is gone"
    assert all("return" in line for line in between), (
        "the win32 guard body must be the plain early return only"
    )
    for banned in ("pytest.skip", "optional_dep_skip", "environment_unavailable_skip"):
        assert banned not in "\n".join(between), (
            f"skip machinery ({banned}) reachable on the win32 path (WPAR-04 no-ever-green-skip)"
        )


def test_notebook_gate_guard_precedes_probe_and_registration_untouched() -> None:
    """Notebook gate: win32 guard precedes the probe call; map entry kept."""
    text = NOTEBOOK_GATE_TEST.read_text(encoding="utf-8")
    region = _function_region(text, "def _gate_pybedtools")
    guard_hits = [
        i for i, line in enumerate(region) if line.strip() == 'if sys.platform == "win32":'
    ]
    probe_hits = [i for i, line in enumerate(region) if "_gate_optional_deps(" in line]
    assert guard_hits, "notebook gate lost the win32 early-return (WPAR-04)"
    assert probe_hits, "notebook gate lost the optional-deps probe call (POSIX fail-loud)"
    assert guard_hits[0] < probe_hits[0], "the win32 guard must precede the probe call"

    between = [line for line in region[guard_hits[0] + 1 : probe_hits[0]] if line.strip()]
    assert between, "the win32 guard body is empty -- the early return is gone"
    assert all("return" in line for line in between), (
        "the win32 guard body must be the plain early return only"
    )
    for banned in ("pytest.skip", "optional_dep_skip", "environment_unavailable_skip"):
        assert banned not in "\n".join(between), (
            f"skip machinery ({banned}) reachable on the win32 path (WPAR-04 no-ever-green-skip)"
        )

    assert f'"{NER_NB_ID}": _gate_pybedtools' in text, (
        "the _ACTIVE_NOTEBOOK_GATES registration entry for the NER notebook changed"
    )
