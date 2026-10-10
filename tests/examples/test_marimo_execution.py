"""Real headless-execution tests for the marimo example apps.

Census lane of the v1.1 example rollout (D-08): every app listed in
:data:`MARIMO_EXEC_SPECS` runs through the private harness via the
``marimo export html`` flavor -- the 05-FEASIBILITY.md spike decision
(deterministic exit code plus a concrete HTML artifact per app) --
inside a tmp sandbox, proving a real export without touching the repo
tree.  Every test here is slow-marked so hosted fast legs never spawn
an app runtime.

D-18 deepening (08-01): a shallow export-plus-size check can pass on an
empty shell, so each app is asserted on the full quadruple -- (1) a
successful headless run, (2) the exit-code contract, (3) the app's UI
default-value literals present in the exported HTML, and (4) a
key-content marker.  Artifacts stay in the tmp sandbox (``assert_tree_clean``
enforces nothing is committed).
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.examples._execution import (
    EXAMPLE_DIR,
    MARIMO_EXEC_SPECS,
    _resolve_marimo_cli,
    assert_tree_clean,
    run_marimo_app,
    seed_sandbox,
)

# D-18 (3): per-app UI default-value literals, calibrated from each app's
# ``mo.ui`` constructors (``value=`` arguments in example/marimo/**) by real
# export runs on the dev box. The exported HTML embeds both the app code and
# the rendered initial UI state, so these literals prove the export carries
# the app's real default UI values -- not an empty shell. Keys are POSIX
# paths relative to EXAMPLE_DIR. A new app added to MARIMO_EXEC_SPECS
# without a defaults entry fails loudly here (KeyError) -- that is the
# inheritance mechanism: calibrate the literals when the app joins the lane.
MARIMO_EXPORT_DEFAULTS: dict[str, tuple[str, ...]] = {
    "marimo/inference/inference_demo.py": ("open chromatin", "Plant DNABERT", "BPE"),
    "marimo/benchmark/benchmark_demo.py": ("config.yaml", "test.csv", "modelscope"),
    "marimo/finetune/finetune_demo.py": (
        "finetune_config.yaml",
        "zhangtaolab/plant-dnagpt-BPE",
        "512",
    ),
}

# D-18 (4): key-content marker present in every real marimo export --
# calibrated against a real benchmark_demo export (marimo >=0.16 emits the
# embedded <marimo-code> element; this version has no marimo-root element).
# An empty-shell export (or a truncated one) fails this check.
MARIMO_EXPORT_MARKER = "marimo-code"


class TestMarimoCliResolution:
    """Cross-platform CLI-discovery contracts (Windows first exposure, 2026-10-08).

    POSIX layouts put console scripts in ``bin/`` beside the interpreter;
    Windows conda/venv layouts put them in ``Scripts/`` -- the resolver
    must find the installed CLI under both, and fail loudly (never
    silently) when nothing resolves.
    """

    def test_resolves_installed_cli_on_this_interpreter(self) -> None:
        """marimo is installed in the test env (notebook extra) -- it must resolve."""
        cli = _resolve_marimo_cli()
        assert cli.is_file(), f"resolved a non-existent CLI path: {cli}"
        assert "marimo" in cli.name

    def test_missing_cli_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All candidates missing + PATH empty -> the descriptive ValueError."""
        fake_exe = tmp_path / "python.exe"
        fake_exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "executable", str(fake_exe))
        monkeypatch.setattr(sysconfig, "get_path", lambda *_args, **_kwargs: str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
        with pytest.raises(ValueError, match="marimo CLI not found"):
            _resolve_marimo_cli()

    def test_windows_scripts_layout_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Scripts\\marimo.exe beside a root python.exe resolves (conda/venv layout)."""
        (tmp_path / "Scripts").mkdir()
        cli_exe = tmp_path / "Scripts" / "marimo.exe"
        cli_exe.write_text("", encoding="utf-8")
        fake_exe = tmp_path / "python.exe"
        fake_exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "executable", str(fake_exe))
        monkeypatch.setattr(
            sysconfig, "get_path", lambda *_args, **_kwargs: str(tmp_path / "Scripts")
        )
        monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
        assert _resolve_marimo_cli() == cli_exe


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
@pytest.mark.timeout(7200)
class TestMarimoAppExecution:
    """Real headless marimo-app execution through the private harness."""

    @pytest.mark.parametrize(
        "app_path",
        [Path(key) for key in MARIMO_EXEC_SPECS],
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_app_exports_html_with_default_ui_and_content(
        self,
        app_path: Path,
        tmp_path: Path,
        marimo_sandbox: Path,
    ) -> None:
        """Export the app headlessly and assert the D-18 quadruple."""
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
        # D-18 (2) exit-code contract: run_marimo_app never returns a
        # process handle -- it raises AssertionError on any non-zero
        # ``marimo export`` return code, and it writes
        # ``<stem>.export.error.txt`` ONLY on that failure path. Reaching
        # this assertion therefore proves the subprocess exited 0; the
        # artifact's absence is the recorded exit-0 evidence.
        error_artifact = tmp_path / "artifacts" / f"{app_path.stem}.export.error.txt"
        assert not error_artifact.exists(), (
            f"marimo export recorded a failure artifact (non-zero exit "
            f"returncode): {error_artifact}"
        )
        # D-18 (3) + (4): default-value literals and key-content marker in
        # the exported HTML -- plain ``in`` checks, no DOM scraping.
        rel = app_path.relative_to(EXAMPLE_DIR).as_posix()
        html_text = html_out.read_text(encoding="utf-8")
        for literal in MARIMO_EXPORT_DEFAULTS[rel]:
            assert literal in html_text, (
                f"{rel}: UI default literal {literal!r} absent from the export"
            )
        assert MARIMO_EXPORT_MARKER in html_text, (
            f"{rel}: key-content marker {MARIMO_EXPORT_MARKER!r} absent "
            f"from the export (empty-shell export?)"
        )
        assert_tree_clean()
