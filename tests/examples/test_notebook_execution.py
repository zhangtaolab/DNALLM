"""Real end-to-end execution tests for example notebooks.

Census-driven rollout layer of the v1.1 example execution (D-08): the
parametrized test runs whole notebooks through the private nbclient
harness inside a tmp sandbox, and the kill test proves a hung kernel is
cleaned up (EXEC-06).  Every test here is slow-marked so hosted fast
legs never spawn kernels.  :data:`NOTEBOOK_EXEC_SPECS` carries budgets
for all 21 notebooks; the :data:`ACTIVE_NOTEBOOKS` list below gates
which ones actually execute -- 05-06 grows it with census-green
notebooks (and gated entries), never by silently widening.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from collections.abc import Iterator
from pathlib import Path

import nbformat
import nbformat.v4 as nbf
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

from tests.examples._execution import (
    EXAMPLE_DIR,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    run_notebook,
    seed_sandbox,
)

# Census rollout list: every notebook here executes for real on each run.
# Initially exactly the pilot; 05-06 adds census-green notebooks one by
# one (entries must exist in NOTEBOOK_EXEC_SPECS, which already carries
# budgets for all 21).
ACTIVE_NOTEBOOKS = [EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb"]


# The sandbox fixture lives in this module (not a tests/examples/conftest.py):
# tests/ is not a package, so a second conftest.py would win the bare
# ``conftest`` module-name race and break the three existing test files that
# do ``from conftest import ...`` (test_trainer/test_benchmark/test_dna_dataset).
@pytest.fixture
def notebook_sandbox(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """Yield a seeded tmp sandbox of the executed notebook's directory.

    Seeds the parametrized notebook's parent directory over ``tmp_path``
    and asserts the repo tree stayed clean on teardown -- belt-and-braces
    beyond the in-test guard.  Generalized over the expanded spec dict
    (05-05): the seeded directory follows ``nb_path`` from the test's
    parametrization, so every census notebook gets a faithful sandbox.
    """
    nb_path = Path(request.node.callspec.params["nb_path"])
    yield seed_sandbox(nb_path.parent, tmp_path)
    assert_tree_clean()


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
@pytest.mark.timeout(7200)
class TestNotebookExecution:
    """Real end-to-end notebook execution through the private harness."""

    @pytest.mark.parametrize(
        "nb_path",
        ACTIVE_NOTEBOOKS,
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


class TestPartialFailureArtifacts:
    """A failing notebook leaves its partial-execution artifacts behind (EXEC-01)."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_cell_error_captures_artifacts_and_reraises(self, tmp_path: Path) -> None:
        """A raising cell re-raises AND writes the executed node + error text to artifact_dir."""
        nb_path = tmp_path / "failing.ipynb"
        nbformat.write(
            nbf.new_notebook(
                cells=[
                    nbf.new_code_cell('marker = "cell-1-ran"'),
                    nbf.new_code_cell("raise RuntimeError('deliberate failure')"),
                    nbf.new_code_cell('print("never reached")'),
                ]
            ),
            nb_path,
        )
        artifacts = tmp_path / "artifacts"
        with pytest.raises(CellExecutionError):
            run_notebook(nb_path, tmp_path, cell_timeout=120, artifact_dir=artifacts)

        executed = artifacts / "failing.executed.ipynb"
        error_txt = artifacts / "failing.error.txt"
        assert executed.is_file(), "partial-execution notebook artifact missing"
        assert error_txt.is_file(), "error text artifact missing"
        partial = nbformat.read(executed, as_version=4)
        sources = [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in partial.cells
            if c.cell_type == "code"
        ]
        assert 'marker = "cell-1-ran"' in sources, "pre-error cell missing from partial artifact"
        assert "deliberate failure" in error_txt.read_text(encoding="utf-8")
        assert_tree_clean()
