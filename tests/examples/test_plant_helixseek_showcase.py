"""PlantHelixSeek showcase notebooks: structure tests + nightly execution lane.

Phase 7 (07-01) landing module for the CRE showcase notebook and, from 07-02,
the Anno sibling.  Two layers per D-13/D-05:

* **Fast, kernel-free structure tests** (:class:`TestPlantHelixSeekShowcaseStructure`)
  pin the committed notebook's contract surface: the provenance cell, the D-16
  fla guard shape (and the absence of any ``fla`` import statement), the
  illustrative-loci captions, the embedded vega outputs, the 2 MB size budget
  (D-12), the SHOW-07 no-genome-wide-claim rule, and a parse guard proving
  :func:`_parse_floors`/:func:`_parse_bands` can read the frozen
  ``selection.md`` contract (failing red on any missing key or band row, D-06).

* **Slow nightly execution tests** re-execute the notebooks end to end through
  the Phase-5 harness (``seed_sandbox`` with per-file shared-data extras,
  ``run_notebook``, ``assert_tree_clean``) and assert the observed metrics
  against the bands parsed from ``selection.md`` at test startup -- never
  against copied literals (D-07), with D-08 named-cause messages on failure.

The showcase notebooks NEVER join ``ACTIVE_NOTEBOOKS``/``GATED_NOTEBOOKS`` in
``test_notebook_execution.py``: their census semantics would double-execute
each ~10-20 min notebook per nightly run (research Open Question 1).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from tests.examples._execution import (
    EXAMPLE_DIR,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    run_notebook,
    seed_sandbox,
)

# Committed showcase surface (Phase-6 data + Phase-7 notebooks).
SHARED_DATA = EXAMPLE_DIR / "notebooks" / "plant_helixseek_shared" / "data"
SELECTION_MD = SHARED_DATA / "selection.md"
CRE_NB = EXAMPLE_DIR / "notebooks" / "plant_helixseek_cre" / "plant_helixseek_cre.ipynb"

# The pinned disclaimer sentence (D-03/SHOW-07) every metric figure or
# conclusion cell carries; structure tests pin it, selection.md explains it.
ILLUSTRATIVE_DISCLAIMER = "illustrative locus, not genome-wide accuracy"

# Slow-lane budgets: the per-test pytest-timeout marks.  The NOTEBOOK_EXEC_SPECS
# cell_timeout for each notebook must stay strictly BELOW its mark (Pitfall 6:
# an outer kill at the cell budget would preempt nbclient's clean
# CellTimeoutError handling and the partial-failure artifact capture).
CRE_TEST_TIMEOUT_S = 2400


def _cell_text(cell: dict) -> str:
    """Return a cell's source as one string (source may be a list or a str)."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _load_nb(path: Path) -> dict:
    """Read a committed notebook as plain JSON (kernel-free)."""
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_floors() -> dict[str, float | int]:
    """Parse the frozen observed-value keys from selection.md (line-start anchors).

    The four frozen keys are matched with ``^key=`` MULTILINE anchors; a
    missing key is a parse-guard failure (D-06) -- the frozen contract is the
    single source of truth and silently proceeding without it would assert
    against nothing.

    Returns:
        ``jaccard``/``neg_cre_fraction``/``neg_anno_fraction`` as floats and
        ``genes_above_floor`` as int.
    """
    text = SELECTION_MD.read_text(encoding="utf-8")
    floors: dict[str, float | int] = {}
    for key in ("jaccard", "genes_above_floor", "neg_cre_fraction", "neg_anno_fraction"):
        match = re.search(rf"^{key}=([0-9.]+)", text, re.MULTILINE)
        if match is None:
            pytest.fail(
                f"selection.md is missing frozen key '{key}=' at a line start "
                "(parse guard, D-06 -- the Phase-6 contract is the assertion source)"
            )
        floors[key] = int(match.group(1)) if key == "genes_above_floor" else float(match.group(1))
    return floors


def _parse_bands() -> dict[str, tuple[float, float] | int]:
    """Parse ALL FOUR tolerance-band rows from the selection.md band table.

    Row lookup is by metric-column substring; bracket intervals ``[a, b]``
    feed the three fraction/jaccard rows and the ``>= N genes`` lower bound
    feeds the Anno gene-models row (parsed here in 07-01, consumed by 07-02's
    Anno test in this same module).  No bound is ever copied as a literal
    (D-07); an unparseable or missing row fails red (D-06).

    Returns:
        ``cre_jaccard``/``neg_cre_fraction``/``neg_anno_fraction`` as
        ``(low, high)`` tuples and ``genes_above_floor`` as the int lower bound.
    """
    lines = SELECTION_MD.read_text(encoding="utf-8").splitlines()
    try:
        header_idx = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("## Thresholds and tolerance bands")
        )
    except StopIteration:
        pytest.fail(
            "selection.md is missing the '## Thresholds and tolerance bands' section "
            "(parse guard, D-06)"
        )
    table_lines = [line for line in lines[header_idx:] if line.startswith("|")]

    def _row(label: str) -> str:
        for line in table_lines:
            columns = [column.strip() for column in line.split("|")]
            if len(columns) > 2 and label in columns[1]:
                return line
        pytest.fail(f"selection.md band table is missing the '{label}' row (parse guard, D-06)")
        return ""  # pragma: no cover -- pytest.fail raises

    bands: dict[str, tuple[float, float] | int] = {}
    for key, label in (
        ("cre_jaccard", "CRE jaccard"),
        ("neg_cre_fraction", "Negative CRE peak-base fraction (flanking)"),
        ("neg_anno_fraction", "Negative Anno genic-base fraction (intergenic)"),
    ):
        interval = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", _row(label))
        if interval is None:
            pytest.fail(
                f"selection.md band row '{label}' carries no '[a, b]' tolerance interval "
                "(parse guard, D-06)"
            )
        bands[key] = (float(interval.group(1)), float(interval.group(2)))

    gene_bound = re.search(r">=\s*([0-9]+)\s*genes", _row("Anno gene models with exon-F1"))
    if gene_bound is None:
        pytest.fail(
            "selection.md gene-models band row carries no '>= N genes' lower bound "
            "(parse guard, D-06)"
        )
    bands["genes_above_floor"] = int(gene_bound.group(1))
    return bands


def _stream_text(nb) -> str:
    """Join every stream output across all cells (text may be a list or a str)."""
    parts: list[str] = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []) or []:
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


class TestPlantHelixSeekShowcaseStructure:
    """Kernel-free structure tests on the COMMITTED showcase notebook (D-13)."""

    def test_selection_contract_parse_guard(self):
        """_parse_floors/_parse_bands read every frozen key and all four band rows."""
        floors = _parse_floors()
        assert set(floors) == {
            "jaccard",
            "genes_above_floor",
            "neg_cre_fraction",
            "neg_anno_fraction",
        }
        assert isinstance(floors["genes_above_floor"], int)
        for key in ("jaccard", "neg_cre_fraction", "neg_anno_fraction"):
            assert isinstance(floors[key], float)

        bands = _parse_bands()
        assert set(bands) == {
            "cre_jaccard",
            "neg_cre_fraction",
            "neg_anno_fraction",
            "genes_above_floor",
        }
        for key in ("cre_jaccard", "neg_cre_fraction", "neg_anno_fraction"):
            low, high = bands[key]
            assert isinstance(low, float)
            assert isinstance(high, float)
            assert low <= high, f"band {key} is inverted: [{low}, {high}]"
        assert isinstance(bands["genes_above_floor"], int)

    def test_provenance_markdown_cell(self):
        """Cell 0 is the consolidated provenance markdown (D-01/D-03)."""
        nb = _load_nb(CRE_NB)
        first = nb["cells"][0]
        assert first["cell_type"] == "markdown", "first cell must be the provenance markdown"
        text = _cell_text(first)
        assert "plant_helixseek_shared/data/selection.md" in text, "selection.md link missing"
        assert "cre_locus=Chr1:5100001-5300000" in text, "locus coordinate line missing"
        assert "illustrative locus" in text.lower(), "illustrative-loci disclaimer missing"

    def test_first_code_cell_is_the_fla_guard(self):
        """The first code cell carries the D-16 find_spec guard + version keys."""
        nb = _load_nb(CRE_NB)
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        assert code_cells, "notebook has no code cells"
        src = _cell_text(code_cells[0])
        assert 'importlib.util.find_spec("fla")' in src, "find_spec guard missing"
        assert "RuntimeError" in src, "guard must raise RuntimeError on missing fla"
        for key in ("transformers_version=", "torch_version=", "fla_version="):
            assert key in src, f"version print {key!r} missing from the guard cell"

    def test_no_import_statement_names_fla(self):
        """AST-level: no Import/ImportFrom node names the fla module (D-16).

        tests/examples/test_examples.py::test_notebook_imports execs every
        import statement node on legs without the fla extra, so the notebook
        must read the fla version through importlib.metadata calls only.
        """
        nb = _load_nb(CRE_NB)
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = _cell_text(cell)
            if not src.strip():
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    is_fla_module = name == "fla" or name.startswith("fla.")
                    assert not is_fla_module, (
                        f"cell {index} contains a fla import statement ({name!r}) -- "
                        "the fast-lane import-exec test would fail on fla-less legs; "
                        "read the version via importlib.metadata instead (D-16)"
                    )

    def test_illustrative_caption_follows_the_metric_figure(self):
        """The markdown after the altair figure carries the pinned disclaimer (D-03)."""
        nb = _load_nb(CRE_NB)
        figure_idx = next(
            (
                i
                for i, cell in enumerate(nb["cells"])
                if cell["cell_type"] == "code" and "alt.Chart(" in _cell_text(cell)
            ),
            None,
        )
        assert figure_idx is not None, "CRE notebook has no altair figure cell"
        following_markdown = [
            _cell_text(cell)
            for cell in nb["cells"][figure_idx + 1 :]
            if cell["cell_type"] == "markdown"
        ]
        assert following_markdown, "no markdown cell follows the metric figure"
        assert ILLUSTRATIVE_DISCLAIMER in following_markdown[0].lower(), (
            f"the caption following the metric figure must carry the pinned disclaimer "
            f"{ILLUSTRATIVE_DISCLAIMER!r}"
        )

    def test_committed_notebook_has_executed_outputs(self):
        """The committed blob carries stream outputs, a vega mime OUTPUT, and stays <= 2 MB (D-12)."""
        nb = _load_nb(CRE_NB)
        output_mimes = [
            mime
            for cell in nb["cells"]
            for output in cell.get("outputs", []) or []
            if output.get("output_type") in ("display_data", "execute_result")
            for mime in (output.get("data", {}) or {})
        ]
        vega_mimes = [mime for mime in output_mimes if mime.startswith("application/vnd.vega")]
        assert vega_mimes, (
            "committed notebook carries no vega mime output -- nbstripout activated, or the "
            "figure cell never executed (its source also names the mime strings, so this "
            "checks outputs, not raw text)"
        )
        assert CRE_NB.stat().st_size <= 2097152, (
            f"committed notebook is {CRE_NB.stat().st_size} bytes (budget 2097152, D-12)"
        )
        assert _stream_text(nb).strip(), "committed notebook carries no stream outputs"

    def test_no_genome_wide_claim_phrasing(self):
        """SHOW-07: 'genome-wide' may appear only inside the negated disclaimer."""
        nb = _load_nb(CRE_NB)
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "markdown":
                continue
            for line in _cell_text(cell).splitlines():
                if "genome-wide" in line:
                    assert "not genome-wide" in line, (
                        f"markdown cell {index} uses genome-wide phrasing outside the "
                        f"illustrative-loci disclaimer: {line!r} (SHOW-07)"
                    )


@pytest.mark.slow
@pytest.mark.timeout(CRE_TEST_TIMEOUT_S)
def test_cre_notebook_executes_within_selection_bands(tmp_path: Path) -> None:
    """Re-execute the committed CRE notebook in-sandbox; assert parsed bands (D-13/D-14)."""
    spec = NOTEBOOK_EXEC_SPECS[str(CRE_NB)]
    # Pitfall 6: the harness contract requires cell_timeout < the outer mark,
    # else pytest-timeout preempts nbclient's clean CellTimeoutError handling.
    assert spec["cell_timeout"] < CRE_TEST_TIMEOUT_S, (
        f"NOTEBOOK_EXEC_SPECS cell_timeout {spec['cell_timeout']} must stay strictly "
        f"below this test's {CRE_TEST_TIMEOUT_S}s pytest-timeout mark (Pitfall 6)"
    )

    # Pattern 6: shared data rides along as per-FILE tuple extras seeding the
    # sibling ../plant_helixseek_shared/data/ directory inside the sandbox.
    extras = [
        (SELECTION_MD, "../plant_helixseek_shared/data/selection.md"),
        (
            SHARED_DATA / "chr1_5351001_5371000.fas",
            "../plant_helixseek_shared/data/chr1_5351001_5371000.fas",
        ),
        (
            SHARED_DATA / "TAIR10_DHSs_chr1_5351001_5371000.gff",
            "../plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff",
        ),
    ]
    sandbox = seed_sandbox(CRE_NB.parent, tmp_path, extra_inputs=extras)
    artifacts = tmp_path / "artifacts"
    nb = run_notebook(CRE_NB, sandbox, cell_timeout=spec["cell_timeout"], artifact_dir=artifacts)

    errored = [
        (cell_index, output)
        for cell_index, cell in enumerate(nb["cells"])
        for output in cell.get("outputs", []) or []
        if output.get("output_type") == "error"
    ]
    assert not errored, f"CRE notebook executed with error outputs: {errored}"

    assert_tree_clean()

    stream = _stream_text(nb)
    floors = _parse_floors()
    bands = _parse_bands()

    observed: dict[str, float] = {}
    for key in ("jaccard", "neg_cre_fraction"):
        match = re.search(rf"^{key}=([0-9.eE+-]+)", stream, re.MULTILINE)
        assert match is not None, f"CRE notebook stream is missing the '{key}=' key=value line"
        observed[key] = float(match.group(1))

    low, high = bands["cre_jaccard"]
    assert low <= observed["jaccard"] <= high, (
        f"CRE jaccard={observed['jaccard']} outside the selection.md band [{low}, {high}] "
        f"(selection.md observed {floors['jaccard']}; the headroom above the floor absorbs "
        "transformers 4.49-5.x drift). Re-run selection (Phase-6 methodology) if environment "
        "drift is suspected."
    )

    low, high = bands["neg_cre_fraction"]
    assert low <= observed["neg_cre_fraction"] <= high, (
        f"negative-CRE peak-base fraction={observed['neg_cre_fraction']} outside the "
        f"selection.md band [{low}, {high}] (selection.md observed "
        f"{floors['neg_cre_fraction']}). Re-run selection (Phase-6 methodology) if "
        "environment drift is suspected."
    )
