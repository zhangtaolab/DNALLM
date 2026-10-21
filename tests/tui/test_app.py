"""Pilot tests for the pre-flight app shell and the environment panel (25-05).

The app is driven headless: every seam is injected (a fake collector here), so
nothing touches the network, the filesystem, or torch. The panel's fast/slow
split is asserted directly — version rows are on screen before the worker
finishes, and the device/disk placeholders are replaced afterwards.
"""

from __future__ import annotations

import asyncio
import threading
import time

from textual.widgets import Static

from dnallm.tui import copy, run_preflight
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.screens.env import EnvScreen
from dnallm.tui.workers import PreflightWorkers
from dnallm.utils.env_info import EnvRow


class FakeCollector:
    """A collector with known rows that records which methods ran.

    Args:
        device_rows: Rows ``collect_device`` returns (a CPU-only row by default).
        disk_rows: Rows ``collect_disk`` returns (one free-space row by default).
        gate: Optional event ``collect_device`` waits on, so a test can hold the
            worker open and assert the placeholder state.
        failure: Optional exception ``collect_device`` raises.
    """

    def __init__(self, device_rows=None, disk_rows=None, gate=None, failure=None) -> None:
        self.device_rows = (
            device_rows
            if device_rows is not None
            else [EnvRow("torch_cuda", "GPU acceleration (CUDA)", "CPU only — no GPU detected")]
        )
        self.disk_rows = (
            disk_rows
            if disk_rows is not None
            else [EnvRow("disk:/tmp", "Free disk space for model downloads (/tmp)", "100 GB free")]
        )
        self.gate = gate
        self.failure = failure
        self.calls: list[str] = []
        self.device_started = threading.Event()

    def collect_versions(self) -> list[EnvRow]:
        """Return the instant version rows, as the real collector does."""
        self.calls.append("versions")
        return [
            EnvRow("dnallm_version", "DNALLM version", "0.5.2"),
            EnvRow("python_version", "Python version", "3.13.15"),
        ]

    def collect_device(self) -> list[EnvRow]:
        """Return the device rows, optionally waiting on the test's gate."""
        self.calls.append("device")
        self.device_started.set()
        if self.gate is not None:
            self.gate.wait(timeout=30)
        if self.failure is not None:
            raise self.failure
        return list(self.device_rows)

    def collect_disk(self) -> list[EnvRow]:
        """Return the free-disk rows."""
        self.calls.append("disk")
        return list(self.disk_rows)


def _rendered_text(screen) -> str:
    """Join the literal content of every Static on *screen*, in layout order."""
    return "\n".join(str(widget.content) for widget in screen.query(Static))


def test_app_is_a_preflight_workers_composition_root() -> None:
    """The worker mixin is in the MRO, so seam methods resolve on the app."""
    assert issubclass(DNALLMPreflightApp, PreflightWorkers)
    app = DNALLMPreflightApp(collector=FakeCollector())
    for name in ("validate_data_worker", "download_worker"):
        assert callable(getattr(app, name))


def test_run_preflight_starts_the_app_with_the_config_path(monkeypatch) -> None:
    """run_preflight builds the app with --config and hands control to it."""
    recorded = {}

    def fake_run(self) -> None:
        recorded["config_path"] = self.state.config_path
        recorded["collector"] = self.collector

    monkeypatch.setattr(DNALLMPreflightApp, "run", fake_run)
    run_preflight(config_path="example.yaml")

    assert recorded["config_path"] == "example.yaml"
    assert recorded["collector"].__name__ == "dnallm.utils.env_info"


async def test_env_screen_renders_version_rows_before_any_heavy_work() -> None:
    """The env panel shows the version rows and checking placeholders on mount.

    The collector is held open so the assertion sees the exact state a real
    startup shows: version rows painted, the slow rows still checking.
    """
    gate = threading.Event()
    collector = FakeCollector(gate=gate)
    app = DNALLMPreflightApp(collector=collector)

    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, EnvScreen)
            text = _rendered_text(app.screen)
    finally:
        gate.set()

    assert "DNALLM version: 0.5.2" in text
    assert "Python version: 3.13.15" in text
    assert f"{copy.ENV_DEVICE_SECTION}: {copy.CHECKING_PLACEHOLDER}" in text
    assert f"{copy.ENV_DISK_SECTION}: {copy.CHECKING_PLACEHOLDER}" in text


async def test_worker_fills_the_placeholders_and_keeps_row_order() -> None:
    """The placeholder holds until the worker reports, then the real rows land."""
    gate = threading.Event()
    collector = FakeCollector(
        device_rows=[EnvRow("torch_vram", "GPU memory (VRAM)", "24 GB total, 21 GB free")],
        gate=gate,
    )
    app = DNALLMPreflightApp(collector=collector)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        before = _rendered_text(app.screen)
        assert copy.CHECKING_PLACEHOLDER in before
        assert "24 GB total" not in before

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        after = _rendered_text(app.screen)

    assert copy.CHECKING_PLACEHOLDER not in after
    assert "GPU memory (VRAM): 24 GB total, 21 GB free" in after
    assert "Free disk space for model downloads (/tmp): 100 GB free" in after
    # versions -> device -> disk, the order fixed by the collector
    assert after.index("DNALLM version") < after.index("GPU memory (VRAM)")
    assert after.index("GPU memory (VRAM)") < after.index("Free disk space")


async def test_failing_collector_renders_one_plain_language_row() -> None:
    """A raising collector becomes one readable row; the app keeps running."""
    app = DNALLMPreflightApp(collector=FakeCollector(failure=RuntimeError("boom")))

    async with app.run_test(size=(100, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _rendered_text(app.screen)
        assert copy.ENV_CHECK_FAILED in text
        assert copy.CHECKING_PLACEHOLDER not in text
        assert "RuntimeError" not in text
        assert app.last_failure == copy.ENV_CHECK_FAILED
        # The quit binding still works after the failure.
        await pilot.press("q")


async def test_quitting_with_a_worker_in_flight_is_prompt() -> None:
    """Quit does not wait for a blocked worker thread."""
    release = threading.Event()
    collector = FakeCollector(gate=release)
    app = DNALLMPreflightApp(collector=collector)
    started = time.monotonic()

    try:
        async with app.run_test(size=(100, 40)) as pilot:
            for _ in range(500):
                if collector.device_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert collector.device_started.is_set(), "the env worker never started"
            await pilot.press("q")
    finally:
        release.set()

    # Far below the 300 s pytest timeout: quitting never blocks on the thread.
    assert time.monotonic() - started < 20


def test_render_text_shows_brackets_literally() -> None:
    """render_text disables markup, so a "[" cannot be parsed (T-25-08)."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    widget = app.render_text("[bold]not-a-tag[/bold]")

    # visual.plain is what actually reaches the terminal; markup parsing would
    # strip the bracketed words and leave "not-a-tag".
    assert widget.visual.plain == "[bold]not-a-tag[/bold]"


async def test_bracketed_collector_value_renders_verbatim() -> None:
    """A bracketed value read from the machine cannot alter the layout."""
    bracketed = "[bold]not-a-tag[/bold] 24 GB"
    collector = FakeCollector(device_rows=[EnvRow("torch_vram", "GPU memory (VRAM)", bracketed)])
    app = DNALLMPreflightApp(collector=collector)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        row = next(
            widget
            for widget in app.screen.query(Static)
            if "GPU memory (VRAM)" in str(widget.content)
        )

    assert row.visual.plain == f"GPU memory (VRAM): {bracketed}"


async def test_config_path_is_stored_and_never_loaded(tmp_path) -> None:
    """--config lands on WizardState; the app never opens the file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("task:\n  task_type: binary\n")
    app = DNALLMPreflightApp(collector=FakeCollector(), config_path=str(config_file))

    # Deleting the file proves the running app does not read it: a load here
    # would fail loudly instead of the wizard starting normally.
    config_file.unlink()

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert app.state.config_path == str(config_file)
        assert app.config_path == str(config_file)
