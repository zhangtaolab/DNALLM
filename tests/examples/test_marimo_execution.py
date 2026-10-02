"""Real headless-execution tests for the marimo example apps.

Census lane of the v1.1 example rollout (D-08): every app listed in
:data:`MARIMO_EXEC_SPECS` runs through the private harness via the
``marimo export html`` flavor -- the 05-FEASIBILITY.md spike decision
(deterministic exit code plus a concrete HTML artifact per app) --
inside a tmp sandbox, proving a real export without touching the repo
tree.  Every test here is slow-marked so hosted fast legs never spawn
an app runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.examples._execution import (
    EXAMPLE_DIR,
    MARIMO_EXEC_SPECS,
    assert_tree_clean,
    run_marimo_app,
    seed_sandbox,
)


# The sandbox fixture lives in this module (not a tests/examples/conftest.py):
# tests/ is not a package, so a second conftest.py would win the bare
# ``conftest`` module-name race and break the three existing test files that
# do ``from conftest import ...`` (test_trainer/test_benchmark/test_dna_dataset).
@pytest.fixture
def marimo_sandbox(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """Yield a seeded tmp sandbox of the executed app's directory.

    Seeds the parametrized app's parent directory over ``tmp_path`` and
    asserts the repo tree stayed clean on teardown -- belt-and-braces
    beyond the in-test guard.
    """
    app_path = Path(request.node.callspec.params["app_path"])
    yield seed_sandbox(app_path.parent, tmp_path)
    assert_tree_clean()


@pytest.mark.slow
@pytest.mark.timeout(1500)
class TestMarimoAppExecution:
    """Real headless marimo-app execution through the private harness."""

    @pytest.mark.parametrize(
        "app_path",
        [Path(key) for key in MARIMO_EXEC_SPECS],
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_inference_demo_exports_html(
        self,
        app_path: Path,
        tmp_path: Path,
        marimo_sandbox: Path,
    ) -> None:
        """Export the app to HTML headlessly and prove a real artifact results."""
        spec = MARIMO_EXEC_SPECS[str(app_path)]
        html_out = run_marimo_app(
            app_path,
            marimo_sandbox,
            timeout=spec["timeout_s"],
            artifact_dir=tmp_path / "artifacts",
        )
        assert html_out.is_file(), f"export artifact missing: {html_out}"
        assert html_out.stat().st_size > 1000, (
            f"export artifact undersized: {html_out.stat().st_size} bytes"
        )
        assert_tree_clean()
