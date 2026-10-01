"""Real end-to-end execution tests for example notebooks.

Pilot layer of the v1.1 execution rollout: the parametrized test runs
whole notebooks through the private nbclient harness inside a tmp
sandbox, and every test here is slow-marked so hosted fast legs never
spawn kernels.  Phase 8 expands :data:`NOTEBOOK_EXEC_SPECS` and this
parametrization to the full notebook census.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.examples._execution import (
    EXAMPLE_DIR,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    run_notebook,
)

# Pilot notebooks -- Phase 8 replaces this list with the expanded
# NOTEBOOK_EXEC_SPECS-driven rollout.
PILOTS = [EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb"]


@pytest.mark.slow
@pytest.mark.timeout(1800)
class TestNotebookExecution:
    """Real end-to-end notebook execution through the private harness."""

    @pytest.mark.parametrize(
        "nb_path",
        PILOTS,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_notebook_executes_end_to_end(
        self,
        nb_path: Path,
        tmp_path: Path,
        notebook_sandbox: Path,
    ) -> None:
        """Execute the whole notebook in a tmp sandbox and prove it stays error-free."""
        spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
        sandbox = notebook_sandbox
        artifacts = tmp_path / "artifacts"
        nb = run_notebook(
            nb_path,
            sandbox,
            cell_timeout=spec["cell_timeout"],
            artifact_dir=artifacts,
        )

        # Structure-only invariants: every non-empty code cell ran to
        # completion with no error output (values are checked by the
        # notebook's own assertions, not duplicated here).
        code_cells = [
            cell
            for cell in nb.cells
            if cell.cell_type == "code"
            and (
                "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            ).strip()
        ]
        errored = [
            (index, output)
            for index, cell in enumerate(code_cells)
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errored, f"{nb_path.name} executed with error outputs: {errored}"
        assert_tree_clean()
