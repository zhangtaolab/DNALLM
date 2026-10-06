"""Unit tests for the docs mirror gate (scripts/check_docs_sync.py).

The gate is a CI hard gate (docs-validation.yml runs the script): any drift
between example/ and docs/example/ fails the job. These tests pin the .pdf
exemption's path shape -- exempt exactly at example/notebooks/*/ depth (the
.gitignore pattern ``example/notebooks/*/*.pdf`` that justifies the
exemption), drift everywhere else -- so the exemption cannot silently widen
back to any-depth and mask real mirror drift.
"""

import filecmp
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_docs_sync.py"
_SPEC = importlib.util.spec_from_file_location("check_docs_sync", _SCRIPT_PATH)
check_docs_sync = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_docs_sync)


def _stray_pdf_errors(tmp_path, pdf_relpath: str, side: str) -> list[str]:
    """Run check_sync over a minimal mirror tree carrying one stray PDF.

    Both halves get the same directory skeleton (notebooks/demo/data and
    marimo/), so dircmp recurses into every level and the stray PDF itself
    surfaces in left_only/right_only instead of its parent directory.

    Args:
        tmp_path: pytest fixture directory for the test.
        pdf_relpath: path of the stray PDF relative to that mirror half's root.
        side: "example" or "docs" -- which half of the mirror carries the PDF.

    Returns:
        The check_sync error list for the tree.
    """
    left = tmp_path / "example"
    right = tmp_path / "docs" / "example"
    for base in (left, right):
        for sub in ("notebooks/demo/data", "marimo"):
            (base / sub).mkdir(parents=True)
    stray_base = left if side == "example" else right
    (stray_base / pdf_relpath).write_bytes(b"%PDF-stray")
    dcmp = filecmp.dircmp(str(left), str(right))
    return check_docs_sync.check_sync(dcmp)


class TestPdfExemptionShape:
    """The .pdf exemption applies only at example/notebooks/*/ depth."""

    def test_pdf_at_justified_depth_is_exempt_on_the_example_side(self):
        """A notebooks/<dir>/x.pdf stray on the example side is ignored."""
        assert check_docs_sync._should_ignore("plot.pdf", "notebooks/demo")

    def test_pdf_at_justified_depth_is_exempt_on_the_docs_side(self):
        """A notebooks/<dir>/x.pdf stray on the docs side is ignored too."""
        # Both mirror halves share one _should_ignore, so the suffix
        # exemptions' both-sides behavior holds for the .pdf exemption.
        assert check_docs_sync._should_ignore("plot.pdf", "notebooks/benchmark")

    def test_pdf_deeper_than_the_gitignore_pattern_is_drift(self):
        """A notebooks/<dir>/<deeper>/x.pdf stray is NOT exempt."""
        assert not check_docs_sync._should_ignore("plot.pdf", "notebooks/demo/data")

    def test_pdf_outside_notebooks_is_drift(self):
        """A x.pdf stray in a non-notebooks directory is NOT exempt."""
        assert not check_docs_sync._should_ignore("plot.pdf", "marimo")

    def test_top_level_pdf_is_drift(self):
        """A x.pdf stray directly under the mirror root is NOT exempt."""
        assert not check_docs_sync._should_ignore("plot.pdf", "")

    def test_suffix_exemptions_stay_depth_free(self):
        """.gz/.log runtime artifacts remain ignored at every depth."""
        assert check_docs_sync._should_ignore("model.bin.gz", "notebooks/demo")
        assert check_docs_sync._should_ignore("run.log", "")
        assert check_docs_sync._should_ignore("run.log", "marimo")

    def test_ignore_names_stay_ignored(self):
        """IGNORE directory names remain ignored at every depth."""
        assert check_docs_sync._should_ignore("logs", "notebooks/demo")
        assert check_docs_sync._should_ignore("outputs", "")


class TestStrayPdfDriftDetection:
    """check_sync flags a stray PDF outside the exempt depth, end to end."""

    def test_justified_depth_stray_on_example_side_passes(self, tmp_path):
        """example/notebooks/demo/plot.pdf only on the example side: no error."""
        errors = _stray_pdf_errors(tmp_path, "notebooks/demo/plot.pdf", "example")
        assert errors == []

    def test_justified_depth_stray_on_docs_side_passes(self, tmp_path):
        """docs/example/notebooks/demo/plot.pdf only on the docs side: no error."""
        errors = _stray_pdf_errors(tmp_path, "notebooks/demo/plot.pdf", "docs")
        assert errors == []

    def test_too_deep_stray_is_reported(self, tmp_path):
        """notebooks/demo/data/plot.pdf only on the example side: drift."""
        errors = _stray_pdf_errors(tmp_path, "notebooks/demo/data/plot.pdf", "example")
        assert errors == ["ONLY in example/: notebooks/demo/data/plot.pdf"]

    def test_stray_outside_notebooks_is_reported(self, tmp_path):
        """marimo/plot.pdf only on the example side: drift."""
        errors = _stray_pdf_errors(tmp_path, "marimo/plot.pdf", "example")
        assert errors == ["ONLY in example/: marimo/plot.pdf"]
