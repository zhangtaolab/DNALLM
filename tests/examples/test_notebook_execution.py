"""Real end-to-end execution tests for example notebooks.

Pilot layer of the v1.1 execution rollout: the parametrized test runs
whole notebooks through the private nbclient harness inside a tmp
sandbox, and the kill test proves a hung kernel is cleaned up (EXEC-06).
Every test here is slow-marked so hosted fast legs never spawn kernels.
Phase 8 expands :data:`NOTEBOOK_EXEC_SPECS` and this parametrization to
the full notebook census.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from pathlib import Path

import nbformat.v4 as nbf
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError

from tests.examples._execution import (
    EXAMPLE_DIR,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    run_notebook,
)

# Pilot notebooks -- Phase 8 replaces this list with the expanded
# NOTEBOOK_EXEC_SPECS-driven rollout.
PILOTS = [EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb"]


def _kernel_count() -> int:
    """Count live ipykernel_launcher processes.

    In-pytest use is self-match-safe: the pytest argv never contains the
    pattern, unlike the enclosing ``bash -c`` of a CI step, which would
    self-match (05-RESEARCH.md Pitfall 1).  Callers therefore compare
    against a captured pre-test baseline (delta-zero), never against
    absolute zero -- unrelated jupyter servers on the box are none of
    this test's business.
    """
    # ruff: ignore[start-process-with-partial-path]
    result = subprocess.run(
        ["pgrep", "-f", "ipykernel_launcher"],
        capture_output=True,
        text=True,
        check=False,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


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


class TestKernelLifecycle:
    """Deliberate-hang proof that the harness kills a stuck kernel (EXEC-06)."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)  # outer backstop: kernel start + 3s cell + kill + poll
    def test_hung_kernel_is_killed_and_cleaned_up(self, tmp_path: Path) -> None:
        """Prove a cell sleeping past the per-cell timeout gets killed with delta-zero kernels."""
        nb = nbf.new_notebook(
            cells=[
                nbf.new_code_cell('print("ok")'),
                nbf.new_code_cell("import time; time.sleep(300)"),
            ]
        )
        before = _kernel_count()
        client = NotebookClient(
            nb,
            timeout=3,
            kernel_name="python3",
            shutdown_kernel="immediate",
            resources={"metadata": {"path": str(tmp_path)}},
        )
        with pytest.raises(CellTimeoutError):
            client.execute()
        # Delta-zero, polled: the kill is async-ish and CI boxes differ
        # from the GB10 probe (which showed the kernel gone immediately),
        # so poll up to 15s in 0.5s steps before asserting.
        deadline = time.time() + 15
        while time.time() < deadline and _kernel_count() > before:
            time.sleep(0.5)
        assert _kernel_count() == before, "hung kernel survived the harness"
