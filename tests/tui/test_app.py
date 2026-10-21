"""Pilot tests for the pre-flight app shell and the environment panel (25-05).

The app is driven headless: every seam is injected (a fake collector here), so
nothing touches the network, the filesystem, or torch. The panel's fast/slow
split is asserted directly — version rows are on screen before the worker
finishes, and the device/disk placeholders are replaced afterwards.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest
import yaml
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Static

from dnallm.preflight import (
    AvailabilityReport,
    AvailabilityState,
    CompatibilityReport,
    GeneratedConfig,
    Hub,
    edit_config,
    validate_data,
)
from dnallm.tui import copy, run_preflight
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.messages import (
    AvailabilityUpdated,
    CompatibilityReady,
    ConfigReady,
    DataReportReady,
    DownloadFinished,
    DownloadProgress,
    EnvReportReady,
    WorkerFailed,
)
from dnallm.tui.screens.config import ConfigScreen
from dnallm.tui.screens.data import DataScreen
from dnallm.tui.screens.download import DownloadScreen
from dnallm.tui.screens.entry import EntryScreen
from dnallm.tui.screens.env import EnvScreen
from dnallm.tui.screens.suitability import SuitabilityScreen
from dnallm.tui.screens.summary import SummaryScreen
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


# ─────────────────────────────────────────────────────────────────────────────
# 25-07: the app routes the suitability worker payloads to the live panel
# ─────────────────────────────────────────────────────────────────────────────


class _AvailabilityRecorder(Screen):
    """A screen that records the availability updates the app hands it."""

    def __init__(self) -> None:
        super().__init__()
        self.updates: list[tuple[str, AvailabilityReport]] = []

    async def apply_availability_update(self, model: str, availability: AvailabilityReport) -> None:
        """Record one update, exactly as the app's routing hook delivers it."""
        self.updates.append((model, availability))


async def test_the_app_routes_an_availability_update_to_the_active_screen() -> None:
    """An on-demand probe result reaches only the panel that is showing."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    recorder = _AvailabilityRecorder()
    report = AvailabilityReport(state=AvailabilityState.UNKNOWN, reason="RATE_LIMITED")

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(recorder)
        await pilot.pause()
        app.post_message(AvailabilityUpdated("org/model", report))
        await pilot.pause()
        updates = list(recorder.updates)

    assert updates == [("org/model", report)]


async def test_an_availability_update_with_no_panel_is_harmless() -> None:
    """A late probe result on a panel without the hook never reaches the user."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, EnvScreen)  # no apply_availability_update hook
        app.post_message(
            AvailabilityUpdated("org/model", AvailabilityReport(state=AvailabilityState.UNKNOWN))
        )
        await pilot.pause()
        assert app.last_failure is None
        await pilot.press("q")


# ─────────────────────────────────────────────────────────────────────────────
# 25-08: the app routes the config worker payloads to the live panel
# ─────────────────────────────────────────────────────────────────────────────


class _ConfigRecorder(Screen):
    """A screen that records the config results the app hands it."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[GeneratedConfig, str | None]] = []

    async def apply_config_ready(
        self, generated: GeneratedConfig, output_path: str | None = None
    ) -> None:
        """Record one result, exactly as the app's routing hook delivers it."""
        self.results.append((generated, output_path))


async def test_the_app_routes_a_config_ready_to_the_active_screen() -> None:
    """A validated config (and a save path) reaches only the showing panel."""
    generated = GeneratedConfig(config={}, errors=[], raw={"task": {"task_type": "binary"}})
    app = DNALLMPreflightApp(collector=FakeCollector())
    recorder = _ConfigRecorder()

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(recorder)
        await pilot.pause()
        app.post_message(ConfigReady(generated, "out.yaml"))
        await pilot.pause()
        results = list(recorder.results)

    assert results == [(generated, "out.yaml")]


async def test_a_config_ready_with_no_config_panel_is_harmless() -> None:
    """A late config result on a panel without the hook never reaches the user."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, EnvScreen)  # no apply_config_ready hook
        app.post_message(ConfigReady(GeneratedConfig(config={}, errors=[], raw={})))
        await pilot.pause()
        assert app.last_failure is None
        await pilot.press("q")


# ─────────────────────────────────────────────────────────────────────────────
# 25-08 Task 3: --config jumps straight to the config panel (D-07)
# ─────────────────────────────────────────────────────────────────────────────


class FakeEditConfig:
    """An ``edit_config`` stand-in that records every argument."""

    def __init__(self, generated: GeneratedConfig) -> None:
        self.generated = generated
        self.calls: list[dict] = []

    def __call__(self, existing, patch=None, *, output_path=None) -> GeneratedConfig:
        """Record the request and return the canned result."""
        self.calls.append({"existing": existing, "patch": patch, "output_path": output_path})
        return self.generated


async def test_the_config_flag_opens_the_config_panel_instead_of_the_entry_choice(
    monkeypatch, tmp_path
) -> None:
    """--config skips the choice and loads the file through edit_config."""
    config_file = tmp_path / "existing.yaml"
    config_file.write_text("task:\n  task_type: binary\n", encoding="utf-8")
    expected = edit_config(config_file, {})
    fake = FakeEditConfig(expected)
    monkeypatch.setattr("dnallm.preflight.edit_config", fake)
    app = DNALLMPreflightApp(collector=FakeCollector(), config_path=str(config_file))

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, EnvScreen)
        await pilot.click("#env-continue")
        await pilot.pause()
        first_non_env = app.screen
        await app.workers.wait_for_complete()
        await pilot.pause()
        preview = str(app.screen.query_one("#config-preview", Static).content)

    assert isinstance(first_non_env, ConfigScreen)
    assert fake.calls[0]["existing"] == str(config_file)
    assert fake.calls[0]["patch"] == {}
    assert fake.calls[0]["output_path"] is None
    assert "task_type: binary" in preview


# ─────────────────────────────────────────────────────────────────────────────
# 25-10 Task 1: one headless walk of the whole wizard (TUI-08 / SC3)
#
# The walk crosses every seam the wizard has — collector, prober, downloader —
# and reads one small CSV from tmp_path. Nothing here touches the network, a
# GPU, or a model checkpoint; the core work (data validation, suitability,
# config generation, the config write) is the real one.
# ─────────────────────────────────────────────────────────────────────────────

# A tiny labelled CSV: three sequences, a 0/1 label column. Long enough for the
# core's length check, short enough to read at a glance.
_WIZARD_CSV = (
    "sequence,label\n"
    "ACGTACGTACGTACGTACGTAC,0\n"
    "TGCATGCATGCATGCATGCATG,1\n"
    "AAAACCCCGGGGTTTTAAAACCC,0\n"
)

# The path the fake downloader "stores the files in" — shown verbatim on the summary.
_WIZARD_DOWNLOAD_PATH = "/home/user/.cache/modelscope/agront-1b-H3K27ac"

# A task type, so the ready-made-model listing contains binary models and the
# generated config needs no further user input.
_WIZARD_TASK_TYPE = "binary"


class _FixedProber:
    """An availability prober that answers without dialling anything.

    Args:
        report: The report every call returns (a ModelScope confirmation by
            default).
    """

    def __init__(self, report: AvailabilityReport | None = None) -> None:
        self.report = (
            report
            if report is not None
            else AvailabilityReport(
                state=AvailabilityState.REMOTE_CONFIRMED,
                size_bytes=1_500_000_000,
                revision="rev-1",
                source_hub=Hub.MODELSCOPE,
            )
        )
        self.calls: list[str] = []

    def __call__(self, model: str) -> AvailabilityReport:
        """Record the probed model id and return the fixed report."""
        self.calls.append(model)
        return self.report


class _FixedDownloader:
    """A hub-downloader stand-in that returns one fixed path.

    Args:
        path: The path the fake download reports it stored the files in.
    """

    def __init__(self, path: str = _WIZARD_DOWNLOAD_PATH) -> None:
        self.path = path
        self.calls: list[str] = []
        self.kwargs: list[dict] = []

    def __call__(self, model_name: str, **kwargs: object) -> str:
        """Record the attempt and return the fixed path."""
        self.calls.append(model_name)
        self.kwargs.append(dict(kwargs))
        return self.path


async def _settle_wizard(pilot, app: DNALLMPreflightApp) -> None:
    """Let every queued worker wave finish (report first, local batch second).

    The download runs on its own daemon thread rather than as a Textual worker
    (WR-02), so its operation is joined here explicitly.
    """
    await app.workers.wait_for_complete()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()
    operation = app._download_operation
    if operation is not None:
        for _ in range(2000):
            if not operation.is_alive:
                break
            await asyncio.sleep(0.005)
        await pilot.pause()
        await pilot.pause()


async def test_the_whole_wizard_walks_to_a_downloaded_model_summary(tmp_path) -> None:
    """One Pilot walk: env -> entry -> data -> suitability -> config -> download -> summary.

    Every seam is injected (a fake collector, a fake prober, a fake downloader)
    and the only file read is a small CSV written here, so the walk is
    network-free and GPU-free by construction. It proves the flow SC3
    describes: the user leaves with a validated config file and, optionally,
    the model files — listed in that order on the closing page.
    """
    data_file = tmp_path / "promoters.csv"
    data_file.write_text(_WIZARD_CSV, encoding="utf-8")
    out_path = tmp_path / "wizard_config.yaml"
    prober = _FixedProber()
    downloader = _FixedDownloader()
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=prober, downloader=downloader)
    started = time.monotonic()

    async with app.run_test(size=(100, 40)) as pilot:
        # 1. Environment facts: the instant version rows are painted, and the
        #    worker replaces the two checking placeholders.
        await _settle_wizard(pilot, app)
        assert isinstance(app.screen, EnvScreen)
        env_text = _rendered_text(app.screen)
        assert copy.CHECKING_PLACEHOLDER not in env_text
        await pilot.click("#env-continue")
        await pilot.pause()

        # 2. Entry choice (D-07): a new run starts from the user's own data.
        assert isinstance(app.screen, EntryScreen)
        await pilot.click("#entry-from-data")
        await pilot.pause()

        # 3. Data: the pasted path is read by the core, not by the screen.
        assert isinstance(app.screen, DataScreen)
        app.screen.query_one("#data-path", Input).value = str(data_file)
        await pilot.click("#data-check")
        await _settle_wizard(pilot, app)
        data_text = _rendered_text(app.screen)
        await pilot.click("#data-continue")
        await pilot.pause()

        # 4. Suitability: ready-made models for a binary task, first row chosen.
        await pilot.pause()
        assert isinstance(app.screen, SuitabilityScreen)
        app.screen.query_one("#suitability-task-type", Input).value = _WIZARD_TASK_TYPE
        await pilot.click("#suitability-intent-predict")
        await _settle_wizard(pilot, app)
        app.screen.query_one("#suitability-listing", DataTable).move_cursor(row=0)
        await pilot.pause()
        chosen = app.state.selected_entry
        assert chosen is not None
        use_enabled = not app.screen.query_one("#suitability-use", Button).disabled
        await pilot.click("#suitability-use")
        await _settle_wizard(pilot, app)

        # 5. Config: built by the core from the chosen model, with no errors.
        assert isinstance(app.screen, ConfigScreen)
        preview = str(app.screen.query_one("#config-preview", Static).content)
        errors_pane = str(app.screen.query_one("#config-errors", Static).content)
        save_enabled = not app.screen.query_one("#config-save", Button).disabled
        app.screen.query_one("#config-output-path", Input).value = str(out_path)
        await pilot.click("#config-save")
        await _settle_wizard(pilot, app)
        continue_enabled = not app.screen.query_one("#config-continue", Button).disabled

        # 6. Download: the explicit "Continue to the download" action (25-09).
        await pilot.click("#config-continue")
        await pilot.pause()
        assert isinstance(app.screen, DownloadScreen)
        await pilot.click("#download-start")
        await _settle_wizard(pilot, app)
        summary = app.screen
        assert isinstance(summary, SummaryScreen)
        summary_text = _rendered_text(summary)

    elapsed = time.monotonic() - started

    # The environment page showed the instant rows before the walk moved on.
    assert "DNALLM version: 0.5.2" in env_text
    # The data page rendered the core's own split line for the pasted file.
    assert 'Split "data": 3 sequences' in data_text
    # The suitability page's listing came from the core, probed only through
    # the injected prober (never the network).
    assert use_enabled
    assert chosen.model in prober.calls
    # The config the core built for a ready-made model needed no fixes.
    assert errors_pane == ""
    assert save_enabled
    assert continue_enabled
    # The download went through the injected downloader, for the chosen model only.
    assert downloader.calls == [chosen.model]
    # The saved file exists and is exactly the previewed document.
    written = out_path.read_text(encoding="utf-8")
    assert written == preview
    assert yaml.safe_load(written) == app.state.generated_config.raw
    # The summary lists the model files first, then the config path.
    assert _WIZARD_DOWNLOAD_PATH in summary_text
    assert str(out_path) in summary_text
    assert summary_text.index(_WIZARD_DOWNLOAD_PATH) < summary_text.index(str(out_path))
    # Well inside the plan's 60 s per-test budget (pytest-timeout is the 300 s ceiling).
    assert elapsed < 60


# ─────────────────────────────────────────────────────────────────────────────
# 25-10 Task 3: the composition root's degraded paths
#
# The app routes every typed payload to the active screen's optional hook, and
# imports each later panel lazily so an older build still runs. Both are safety
# valves; these tests hold them open.
# ─────────────────────────────────────────────────────────────────────────────


def test_active_screen_is_none_before_the_first_panel() -> None:
    """Before anything is pushed there is no active screen, and that is not an error."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    assert app._active_screen() is None


class _BareScreen(Screen):
    """A screen with none of the wizard's ``apply_*`` hooks."""

    def compose(self):
        """Draw nothing: this screen only exists to receive (and drop) messages."""
        return iter(())


async def test_routing_hooks_ignore_a_screen_without_the_hook() -> None:
    """Every payload is harmless on a panel that does not implement its hook."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    data_report = validate_data(["ACGT" * 10, "TGCA" * 10])

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(_BareScreen())
        await pilot.pause()
        app.post_message(EnvReportReady({"device": [], "disk": []}))
        app.post_message(DataReportReady(data_report))
        app.post_message(CompatibilityReady(CompatibilityReport(verdicts=[], listing=[]), {}))
        app.post_message(DownloadProgress(done=0, total=0, label="nothing yet"))
        app.post_message(DownloadFinished(path=None, cancelled=True))
        app.post_message(WorkerFailed("Nobody on this page can show this."))
        await pilot.pause()
        failure = app.last_failure
        await pilot.press("q")

    # The failure is still recorded — only its on-screen rendering is skipped.
    assert failure == "Nobody on this page can show this."


_MISSING_PANEL_CASES = [
    ("dnallm.tui.screens.download", "show_download", {}),
    ("dnallm.tui.screens.entry", "show_entry", {}),
    ("dnallm.tui.screens.config", "show_config", {"model": "zhangtaolab/FungiHelixSeek"}),
    ("dnallm.tui.screens.suitability", "show_suitability", {}),
]


@pytest.mark.parametrize(("module_name", "action", "kwargs"), _MISSING_PANEL_CASES)
async def test_a_missing_panel_module_logs_instead_of_pushing(
    monkeypatch, module_name: str, action: str, kwargs: dict
) -> None:
    """An older build without one panel logs a line; the wizard keeps its page."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        before = app.screen
        # ``None`` in sys.modules makes ``from <module> import X`` raise ImportError,
        # which is exactly the older-build case the guarded import exists for.
        monkeypatch.setitem(sys.modules, module_name, None)
        getattr(app, action)(**kwargs)
        await pilot.pause()
        still = app.screen
        await pilot.press("q")

    assert still is before
