"""Pilot tests for the data and suitability panels (TUI-04/TUI-05).

The panels are presentation only, so the tests are table tests against the
core: a synthetic file is validated twice — once directly through
``validate_data`` and once through the running panel — and the rendered rows
must equal the report the core returned. The suitability panel is driven the
same way: a fake core returns a frozen report whose row order is its own, and
the panel must show exactly that order and pass the core's intent vocabulary
through unchanged. Every seam is injected, so nothing touches the network or
torch; only a small CSV in ``tmp_path`` is read.
"""

from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, Input, ProgressBar, Static

import dnallm
import dnallm.preflight as preflight_facade
import dnallm.preflight.compatibility as compatibility_core
from dnallm.preflight import (
    AvailabilityReport,
    AvailabilityState,
    CompatibilityReport,
    DimensionVerdict,
    GeneratedConfig,
    Hub,
    ListingEntry,
    ModelVerdict,
    VerdictStatus,
    edit_config,
    render_yaml,
    validate_data,
)
from dnallm.tui import copy
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.messages import CompatibilityReady
from dnallm.tui.screens.config import ConfigScreen
from dnallm.tui.screens.data import DataScreen
from dnallm.tui.screens.download import DownloadScreen
from dnallm.tui.screens.entry import EntryScreen
from dnallm.tui.screens.env import EnvScreen
from dnallm.tui.screens.suitability import INTENT_FINETUNE, INTENT_PREDICT, SuitabilityScreen
from dnallm.tui.screens.summary import SummaryScreen
from dnallm.tui.workers import DOWNLOAD_MAX_TRY

# Two short sequences (below the default 20-character minimum) and one long
# one; the duplicate pair and the missing-too-short rows give the core a
# report-level finding and two split-level findings to render.
_FINDINGS_CSV = "sequence,label\nACGT,0\nACGT,0\nACGTACGTACGTACGTACGTAC,1\n"

# Clean data: single sequence column, distinct rows well past the minimum,
# only A/C/G/T — the core has nothing to report.
_CLEAN_CSV = "sequence\nACGTACGTACGTACGTACGTAC\nTGCATGCATGCATGCATGCATG\n"


class FakeCollector:
    """The minimum collector the app needs so the env step mounts instantly."""

    def collect_versions(self) -> list:
        """Return no version rows; this file tests the data panel."""
        return []

    def collect_device(self) -> list:
        """Return no device rows."""
        return []

    def collect_disk(self) -> list:
        """Return no disk rows."""
        return []


def _rendered_text(screen) -> str:
    """Join the literal content of every Static on *screen*, in layout order."""
    return "\n".join(str(widget.content) for widget in screen.query(Static))


def _rows_the_text_needs(widget: Static) -> int:
    """How many rows Textual's own layout needs for *widget*'s text at its width.

    Uses the same ``Content.get_height`` call Textual uses to size a ``Static``,
    so the number moves with the renderer instead of a re-implemented wrapper.
    """
    return widget.visual.get_height(widget.styles, widget.content_region.width)


def _table_rows(table: DataTable) -> list[list[str]]:
    """The findings table's cells as plain strings, in row order."""
    return [[str(cell) for cell in table.get_row_at(index)] for index in range(table.row_count)]


async def _open_data_screen(app: DNALLMPreflightApp, pilot) -> DataScreen:
    """Push the data panel and wait for it to be the visible screen."""
    app.push_screen(DataScreen())
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, DataScreen)
    return screen


async def test_env_continue_opens_the_entry_choice() -> None:
    """The env panel's Continue action offers the two ways in (D-07)."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, EnvScreen)
        await pilot.click("#env-continue")
        await pilot.pause()
        assert isinstance(app.screen, EntryScreen)


async def test_data_panel_renders_the_core_report_verbatim(tmp_path) -> None:
    """Every row on the page equals a field of the report the core returned."""
    data_file = tmp_path / "found.csv"
    data_file.write_text(_FINDINGS_CSV)
    expected = validate_data(str(data_file))

    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        screen.query_one("#data-path", Input).value = str(data_file)
        await pilot.click("#data-check")
        await app.workers.wait_for_complete()
        await pilot.pause()

        text = _rendered_text(screen)
        rows = _table_rows(screen.query_one("#data-findings", DataTable))
        stored = app.state.data_report

    assert stored is not None
    assert stored == expected
    split = expected.splits[0]
    assert stored.splits[0].row_count == split.row_count
    # The rendered split line carries the core's own row count and lengths.
    assert f'"{split.name}": {split.row_count} sequences' in text
    assert f"shortest {split.length_stats.min}" in text
    assert f"longest {split.length_stats.max}" in text
    for entry in split.charset_census:
        assert (
            copy.DATA_CHARSET_ROW.format(split=split.name, char=entry.char, count=entry.count)
            in text
        )

    # Every finding the core returned is one row, in the core's order, with its
    # severity, reason code, and count (verbatim) — nothing filtered or sorted.
    findings = list(expected.findings)
    findings.extend(split.findings)
    assert [row[1] for row in rows] == [finding.reason_code for finding in findings]
    assert [row[0] for row in rows] == [finding.severity.value for finding in findings]
    assert [row[2] for row in rows] == [finding.message for finding in findings]
    assert [row[3] for row in rows] == [
        copy.DATA_COUNT_NOT_APPLICABLE if finding.count is None else str(finding.count)
        for finding in findings
    ]


async def test_skip_keeps_no_report_and_leaves_the_step(monkeypatch) -> None:
    """Skipping makes no core call, leaves data_report None, and moves on."""
    calls: list[object] = []
    monkeypatch.setattr("dnallm.preflight.validate_data", lambda source: calls.append(source))
    advanced: list[str] = []
    app = DNALLMPreflightApp(collector=FakeCollector())
    monkeypatch.setattr(app, "show_suitability", lambda: advanced.append("suitability"))

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        assert copy.NO_DATA_PROVIDED in _rendered_text(screen)
        await pilot.click("#data-skip")
        await pilot.pause()
        text = _rendered_text(screen)
        assert app.state.data_report is None

    assert advanced == ["suitability"]
    assert calls == []
    # The consequence of skipping is stated in the panel, in plain words (D-08).
    assert copy.DATA_SKIP_NOTE in text


async def test_empty_path_prompts_instead_of_calling_the_core(monkeypatch) -> None:
    """An empty path is a prompt, not a read: the worker never starts."""
    calls: list[object] = []
    monkeypatch.setattr("dnallm.preflight.validate_data", lambda source: calls.append(source))
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        screen.query_one("#data-path", Input).value = "   "
        await pilot.click("#data-check")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _rendered_text(screen)

    assert copy.EMPTY_PATH_PROMPT in text
    assert calls == []
    assert app.state.data_report is None


async def test_reentering_after_a_skip_shows_the_no_data_state(monkeypatch) -> None:
    """Coming back to the panel after a skip shows the explained empty state."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    monkeypatch.setattr(app, "show_suitability", lambda: None)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        await pilot.click("#data-skip")
        await pilot.pause()
        app.pop_screen()
        await pilot.pause()
        screen = await _open_data_screen(app, pilot)
        text = _rendered_text(screen)
        table = screen.query_one("#data-findings", DataTable)
        table_hidden = not table.display

    assert copy.NO_DATA_PROVIDED in text
    assert copy.DATA_SKIP_NOTE in text
    assert table_hidden


async def test_a_clean_report_shows_an_explicit_all_clear_row(tmp_path) -> None:
    """Zero findings is a stated result, never an empty table to interpret."""
    data_file = tmp_path / "clean.csv"
    data_file.write_text(_CLEAN_CSV)
    expected = validate_data(str(data_file))
    assert expected.findings == []
    assert expected.splits[0].findings == []

    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        screen.query_one("#data-path", Input).value = str(data_file)
        await pilot.click("#data-check")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _rendered_text(screen)
        no_findings = screen.query_one("#data-no-findings", Static)
        table = screen.query_one("#data-findings", DataTable)
        cleared_row = copy.NO_FINDINGS in text and no_findings.display
        row_count = table.row_count

    assert cleared_row
    assert row_count == 0


async def test_a_pickle_path_renders_the_refusal_row_with_its_code(tmp_path) -> None:
    """A .pkl path becomes a plain row naming UNSUPPORTED_FORMAT_PKL; no raise."""
    pickle_file = tmp_path / "labels.pkl"
    pickle_file.write_bytes(b"not really a pickle")

    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        # Handed straight to the worker: this proves the CORE's refusal row is
        # what renders, not a screen-side file-type check.
        app.validate_data_worker(str(pickle_file))
        await app.workers.wait_for_complete()
        await pilot.pause()
        rows = _table_rows(screen.query_one("#data-findings", DataTable))
        stored = app.state.data_report
        failure = app.last_failure

    assert failure is None
    assert stored is not None
    assert stored.findings[0].reason_code == "UNSUPPORTED_FORMAT_PKL"
    assert rows[0][1] == "UNSUPPORTED_FORMAT_PKL"
    assert rows[0][0] == "error"
    assert rows[0][2] == stored.findings[0].message


async def test_a_missing_path_becomes_a_plain_language_row(tmp_path) -> None:
    """A path that does not exist never produces a traceback on screen."""
    missing = tmp_path / "nowhere.csv"

    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_data_screen(app, pilot)
        app.validate_data_worker(str(missing))
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _rendered_text(screen)
        failure = app.last_failure

    assert failure is not None
    assert failure in text
    assert "could not read this data" in text
    assert "ValueError" not in text
    assert "Traceback" not in text


async def test_the_panel_is_usable_at_80x24(tmp_path) -> None:
    """The action and the first findings row are on screen at the smallest size."""
    data_file = tmp_path / "found.csv"
    data_file.write_text(_FINDINGS_CSV)
    expected = validate_data(str(data_file))
    expected_codes = [finding.reason_code for finding in expected.findings]
    expected_codes += [finding.reason_code for finding in expected.splits[0].findings]

    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_data_screen(app, pilot)
        screen.query_one("#data-path", Input).value = str(data_file)
        await pilot.click("#data-check")
        await app.workers.wait_for_complete()
        await pilot.pause()

        body = screen.query_one("#data-body", VerticalScroll).region
        check_button = screen.query_one("#data-check", Button).region
        table = screen.query_one("#data-findings", DataTable)
        # A slow runner can still be mid-layout here (the Windows CI leg saw a
        # one-row table region): wait, bounded, until the header plus the
        # first findings row are really in the geometry before snapshotting.
        layout_settled = await _wait_until(lambda: table.region.height >= table.header_height + 1)
        assert layout_settled, "the findings table never reached header + one row at 80x24"
        table_region = table.region
        header_height = table.header_height
        first_row = _table_rows(table)[0]
        height = app.size.height

    # The action is inside the visible page.
    assert check_button.bottom <= body.bottom <= height
    # The table is inside the page, tall enough for header + one finding, and
    # that first finding row sits above the bottom of the page.
    assert table_region.bottom <= body.bottom
    assert table_region.height >= header_height + 1
    assert table_region.y + header_height + 1 <= body.bottom
    # And that first row really is the core's first finding.
    assert first_row[1] == expected_codes[0]


# ─────────────────────────────────────────────────────────────────────────────
# Suitability panel (TUI-05, 25-07): the core's order and vocabulary
# ─────────────────────────────────────────────────────────────────────────────


class FakeProber:
    """An availability prober that records what it was asked and dials nothing.

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
                size_bytes=1,
                revision="rev-1",
                source_hub=Hub.MODELSCOPE,
            )
        )
        self.calls: list[str] = []

    def __call__(self, model: str) -> AvailabilityReport:
        """Record the probed model id and return the canned report."""
        self.calls.append(model)
        return self.report


class FakeCompatibilityCore:
    """A ``check_compatibility`` stand-in that records every argument.

    Args:
        report: The frozen report every call returns.
        gate: Optional event the call waits on, so a test can hold the worker
            open and assert the probing state the user actually sees.
    """

    def __init__(self, report: CompatibilityReport, gate: threading.Event | None = None) -> None:
        self.report = report
        self.gate = gate
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        data: Any = None,
        *,
        task_type: str | None = None,
        models: list[str] | None = None,
        intent: str | None = None,
        prober: Any = None,
        probe_shortlist: int | None = None,
    ) -> CompatibilityReport:
        """Record the call, wait on the gate when set, and return the report."""
        if self.gate is not None:
            self.gate.wait(timeout=30)
        self.calls.append({
            "data": data,
            "task_type": task_type,
            "models": models,
            "intent": intent,
            "prober": prober,
            "probe_shortlist": probe_shortlist,
        })
        return self.report


def _synthetic_report() -> CompatibilityReport:
    """A report whose row order only the screen could break.

    The second row is the lab model, so a screen-side lab-first sort would
    flip the two — exactly the reordering the panel must never do.
    """
    return CompatibilityReport(
        verdicts=[
            ModelVerdict(
                model="zhangtaolab/lab-b",
                entry_name="Lab B",
                is_lab=True,
                dimensions=[
                    DimensionVerdict(dimension="task", status=VerdictStatus.OK),
                    DimensionVerdict(
                        dimension="length",
                        status=VerdictStatus.UNKNOWN,
                        reason_code="NO_DATA",
                    ),
                ],
                overall=VerdictStatus.UNKNOWN,
            )
        ],
        listing=[
            ListingEntry(
                model="third-party/file-a",
                entry_name="Third A",
                is_lab=False,
                task_type="mask",
                availability=AvailabilityReport(
                    state=AvailabilityState.REMOTE_CONFIRMED, source_hub=Hub.HUGGINGFACE
                ),
                support="works",
            ),
            ListingEntry(
                model="zhangtaolab/lab-b",
                entry_name="Lab B",
                is_lab=True,
                task_type="binary",
                num_labels=2,
                label_names=["0", "1"],
                availability=AvailabilityReport(
                    state=AvailabilityState.UNKNOWN, reason="RATE_LIMITED"
                ),
                support="works with built-in fixes",
            ),
            # A third-party family the shortlist never reached: no report at
            # all, which is the D-05 case — listed, unverified, choosable.
            ListingEntry(
                model="third-party/file-c",
                entry_name="Third C",
                is_lab=False,
                task_type="mask",
                support="needs pip install gpn",
            ),
        ],
    )


async def _open_suitability_screen(app: DNALLMPreflightApp, pilot) -> SuitabilityScreen:
    """Push the suitability panel and wait for it to be the visible screen."""
    app.push_screen(SuitabilityScreen())
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, SuitabilityScreen)
    return screen


async def test_suitability_renders_the_core_listing_in_core_order(monkeypatch) -> None:
    """Rows appear in ``report.listing`` order, with the core's own facts."""
    report = _synthetic_report()
    gate = threading.Event()
    fake_core = FakeCompatibilityCore(report, gate=gate)
    monkeypatch.setattr(preflight_facade, "check_compatibility", fake_core)
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    prober = FakeProber()
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=prober)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        # Before the report lands the page says plainly that it is probing.
        probing = copy.PROBING_PLACEHOLDER in _rendered_text(screen)
        gate.set()
        await _settle(pilot, app)
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))
        verdict_rows = _table_rows(screen.query_one("#suitability-verdicts", DataTable))
        status = _rendered_text(screen)

    assert probing
    # The lab row stays second: the screen never re-sorts the core's listing.
    assert [row[0] for row in rows] == [entry.entry_name for entry in report.listing]
    assert [row[1] for row in rows] == [
        copy.THIRD_PARTY_ROW_FLAG,
        copy.LAB_ROW_FLAG,
        copy.THIRD_PARTY_ROW_FLAG,
    ]
    assert rows[1][2] == "binary"
    assert rows[1][3] == "2"
    assert rows[2][2] == "mask"
    # Each availability cell names the hub the core reported — never another.
    assert copy.AVAILABILITY_REMOTE.format(hub="Hugging Face") == rows[0][4]
    assert rows[1][4] == copy.HOSTING_NOT_VERIFIED
    # A third-party row nobody probed is listed like any other and says so.
    assert rows[2][4] == copy.HOSTING_NOT_VERIFIED
    # One verdict row per dimension, carrying the core's status and reason.
    assert [row[1] for row in verdict_rows] == ["task", "length"]
    assert [row[2] for row in verdict_rows] == ["ok", copy.NO_DATA_YET]
    assert verdict_rows[1][3] == "NO_DATA"
    assert copy.SUITABILITY_READY in status
    # The core got the pinned vocabulary, the app's prober, the 20-row cap,
    # and the wizard's stored data report (None: the data step was skipped).
    assert fake_core.calls[0]["intent"] == "finetune"
    assert fake_core.calls[0]["prober"] is prober
    assert fake_core.calls[0]["probe_shortlist"] == 20
    assert fake_core.calls[0]["data"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Suitability listing, support column (D-16, 27.2-05): one trailing text cell
# derived by the core, with the availability column still at index 4.
# ─────────────────────────────────────────────────────────────────────────────


async def test_the_listing_table_declares_support_after_availability(monkeypatch) -> None:
    """The sixth column is Support, immediately after Availability (D-16)."""
    from dnallm.tui.screens.suitability import _AVAILABILITY_COLUMN

    monkeypatch.setattr(
        preflight_facade, "check_compatibility", FakeCompatibilityCore(_synthetic_report())
    )
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        table = screen.query_one("#suitability-listing", DataTable)
        headers = [str(column.label) for column in table.columns.values()]

    assert headers == list(copy.SUITABILITY_LISTING_COLUMNS)
    assert headers[-2:] == ["Availability", "Support"]
    assert len(headers) == 6
    # The refresh code keeps writing availability at index 4 (never the new cell).
    assert _AVAILABILITY_COLUMN == 4


async def test_each_listing_row_shows_the_support_word_verbatim(monkeypatch) -> None:
    """Every row's last cell is exactly its entry's support word."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))

    assert [row[5] for row in rows] == [entry.support for entry in report.listing]
    assert [row[5] for row in rows] == [
        "works",
        "works with built-in fixes",
        "needs pip install gpn",
    ]
    # Availability still occupies its own index-4 cell.
    assert copy.AVAILABILITY_REMOTE.format(hub="Hugging Face") == rows[0][4]


async def test_availability_updates_leave_the_support_cell_untouched(monkeypatch) -> None:
    """A refreshed or on-demand availability write changes only column 4."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())
    late = AvailabilityReport(state=AvailabilityState.CACHED)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        listing = screen.query_one("#suitability-listing", DataTable)
        await screen.apply_availability_update(report.listing[1].model, late)
        await pilot.pause()
        after_on_demand = [str(cell) for cell in listing.get_row_at(1)]
        screen._refresh_availability_cells()
        after_refresh = [str(cell) for cell in listing.get_row_at(1)]

    assert after_on_demand[4] == copy.AVAILABILITY_CACHED_LOCAL
    assert after_refresh[4] == copy.AVAILABILITY_CACHED_LOCAL
    assert after_on_demand[5] == "works with built-in fixes"
    assert after_refresh[5] == "works with built-in fixes"


async def test_a_finetuned_row_shows_its_inherited_base_support(monkeypatch) -> None:
    """A finetuned entry's inherited base word reaches the rendered cell verbatim."""
    from dnallm.models.registry import try_resolve
    from dnallm.preflight.compatibility import _build_listing

    rows_entries = _build_listing([try_resolve("zhangtaolab/dnabert2-promoter")])
    report = CompatibilityReport(verdicts=[], listing=rows_entries)
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))

    assert rows_entries[0].support == "works with built-in fixes"
    assert rows[0][5] == "works with built-in fixes"


def test_screen_intent_constants_match_the_core_vocabulary() -> None:
    """The screen pins the core's exact values, so copy can never drift in."""
    assert INTENT_FINETUNE == compatibility_core.INTENT_FINETUNE
    assert INTENT_PREDICT == compatibility_core.INTENT_PREDICT
    # The UI's words are copy, and they are not the core's vocabulary.
    assert copy.SUITABILITY_INTENT_FINETUNE not in (INTENT_FINETUNE, INTENT_PREDICT)
    assert copy.SUITABILITY_INTENT_PREDICT not in (INTENT_FINETUNE, INTENT_PREDICT)


async def test_choosing_predict_now_asks_the_core_with_predict(monkeypatch) -> None:
    """The second route re-runs the core with ``intent="predict"``."""
    fake_core = FakeCompatibilityCore(_synthetic_report())
    monkeypatch.setattr(preflight_facade, "check_compatibility", fake_core)
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        await _open_suitability_screen(app, pilot)
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#suitability-intent-predict")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert [call["intent"] for call in fake_core.calls] == ["finetune", "predict"]


async def test_selecting_a_listing_row_records_it_in_wizard_state(monkeypatch) -> None:
    """Focusing a row stores that exact frozen entry and enables the action."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = screen.query_one("#suitability-listing", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()
        # The ten-row legend puts the use action below the body's bottom edge at
        # this size, so scroll it in before clicking (never shrink the legend).
        screen.scroll_to_widget(screen.query_one("#suitability-use", Button), animate=False)
        await pilot.pause()
        selected = app.state.selected_entry
        use_enabled = not screen.query_one("#suitability-use", Button).disabled
        await pilot.click("#suitability-use")
        await pilot.pause()
        status = _rendered_text(screen)

    assert selected is report.listing[1]
    assert use_enabled
    assert copy.USE_MODEL_CONFIRMED.format(model="Lab B") in status


async def test_data_continue_opens_the_suitability_panel(monkeypatch) -> None:
    """The data step's Continue action routes on to the suitability panel."""
    monkeypatch.setattr(
        preflight_facade, "check_compatibility", FakeCompatibilityCore(_synthetic_report())
    )
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        await _open_data_screen(app, pilot)
        await pilot.click("#data-continue")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, SuitabilityScreen)


# ─────────────────────────────────────────────────────────────────────────────
# Suitability panel, two-phase availability (D-04): every row local, the
# shortlist remote. Nothing here touches a hub or the real disk caches.
# ─────────────────────────────────────────────────────────────────────────────


async def _settle(pilot, app: DNALLMPreflightApp) -> None:
    """Let both worker waves finish (report first, local batch second)."""
    await app.workers.wait_for_complete()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def _fake_local(cached: dict[str, AvailabilityReport] | None = None):
    """A local_availability stand-in: canned hits, UNKNOWN for everything else."""

    def local_availability(model: str) -> AvailabilityReport:
        canned = (cached or {}).get(model)
        if canned is not None:
            return canned
        return AvailabilityReport(state=AvailabilityState.UNKNOWN)

    return local_availability


def test_the_probe_shortlist_cap_is_twenty() -> None:
    """D-04/D-10: the automatic probe window is 20 rows, never all entries."""
    from dnallm.tui.workers import PROBE_SHORTLIST

    assert PROBE_SHORTLIST == 20


async def test_only_the_lab_first_shortlist_is_remote_probed(monkeypatch) -> None:
    """A long listing probes exactly 20 rows; the rest stay honestly unknown."""
    prober = FakeProber()
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=prober)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))
        report = app.state.compatibility_report

    assert report is not None
    assert len(rows) > 20
    assert len(prober.calls) == 20
    # The probed rows are the first 20 in the core's lab-first order.
    assert prober.calls == [entry.model for entry in report.listing[:20]]
    assert rows[0][4] == copy.AVAILABILITY_REMOTE.format(hub="ModelScope")
    assert rows[20][4] == copy.HOSTING_NOT_VERIFIED


async def test_a_locally_cached_row_outside_the_shortlist_shows_cached(monkeypatch) -> None:
    """The local pass classifies rows the remote shortlist never reaches."""
    listing = compatibility_core.check_compatibility(intent="finetune").listing
    assert len(listing) > 26
    beyond = listing[25]
    later = listing[26]
    cached_report = AvailabilityReport(
        state=AvailabilityState.CACHED, size_bytes=5, source_hub=Hub.MODELSCOPE
    )
    monkeypatch.setattr(
        preflight_facade, "local_availability", _fake_local({beyond.model: cached_report})
    )
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))
        stored = app.state.availability_by_entry

    assert rows[25][4] == copy.AVAILABILITY_CACHED_HUB.format(hub="ModelScope")
    assert rows[26][4] == copy.HOSTING_NOT_VERIFIED
    assert stored[beyond.model] == cached_report
    assert stored[later.model].state == AvailabilityState.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Suitability panel, honesty rules (D-05/D-06/D-08): UNKNOWN is not
# unavailable, NO_DATA is not a failure, and an on-demand probe touches one row.
# ─────────────────────────────────────────────────────────────────────────────


async def test_an_unknown_row_is_honest_and_still_choosable(monkeypatch) -> None:
    """The unverified row says so, shows the reason, and stays selectable."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        table = screen.query_one("#suitability-listing", DataTable)
        rows = _table_rows(table)
        table.move_cursor(row=1)
        await pilot.pause()
        text = _rendered_text(screen)
        use_enabled = not screen.query_one("#suitability-use", Button).disabled
        selected = app.state.selected_entry

    assert rows[1][4] == copy.HOSTING_NOT_VERIFIED
    # The probe's own reason is shown verbatim, never translated or dropped.
    assert f"{copy.AVAILABILITY_DETAIL_LABEL} RATE_LIMITED" in text
    # D-06: an unverified row is choosable, and the page says why that is safe.
    assert copy.UNKNOWN_STILL_DOWNLOADABLE in text
    assert use_enabled
    assert selected is report.listing[1]


async def test_a_no_data_verdict_reads_as_no_data_yet(monkeypatch) -> None:
    """A dimension that needs data says so; it never reads as a failure."""
    monkeypatch.setattr(
        preflight_facade, "check_compatibility", FakeCompatibilityCore(_synthetic_report())
    )
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        rows = _table_rows(screen.query_one("#suitability-verdicts", DataTable))

    no_data_rows = [row for row in rows if row[3] == "NO_DATA"]
    assert no_data_rows, "the synthetic report must carry a NO_DATA dimension"
    assert all(row[2] == copy.NO_DATA_YET for row in no_data_rows)
    assert all("fail" not in row[2].lower() for row in no_data_rows)


async def test_the_on_demand_probe_replaces_exactly_one_row(monkeypatch) -> None:
    """Pressing p probes the focused row only, and rewrites only its cell."""
    on_demand = AvailabilityReport(
        state=AvailabilityState.REMOTE_CONFIRMED,
        size_bytes=9,
        revision="rev-2",
        source_hub=Hub.HUGGINGFACE,
    )
    prober = FakeProber(report=on_demand)
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=prober)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        table = screen.query_one("#suitability-listing", DataTable)
        before = _table_rows(table)
        table.move_cursor(row=25)
        await pilot.pause()
        await pilot.press("p")
        await _settle(pilot, app)
        after = _table_rows(table)
        report = app.state.compatibility_report
        stored = app.state.availability_by_entry

    assert report is not None
    # Twenty automatic shortlist probes, then exactly one on-demand probe.
    assert len(prober.calls) == 21
    assert prober.calls[-1] == report.listing[25].model
    changed = [
        index for index, (old, new) in enumerate(zip(before, after, strict=True)) if old != new
    ]
    assert changed == [25]
    assert after[25][4] == copy.AVAILABILITY_REMOTE.format(hub="Hugging Face")
    assert stored[report.listing[25].model] is on_demand


# ─────────────────────────────────────────────────────────────────────────────
# Config panel (TUI-06, 25-08): the form, the ordered error pane, the preview
# ─────────────────────────────────────────────────────────────────────────────

# Registry-pinned fixtures, measured against the real registry at plan time
# (same entries the Phase-23 core suite pins).
CONFIG_FINETUNED_REPO = "zhangtaolab/plant-dnabert-BPE-promoter"
CONFIG_BASE_REPO = "zhangtaolab/plant-dnabert-BPE"


async def _open_config_screen(app: DNALLMPreflightApp, pilot, **kwargs) -> ConfigScreen:
    """Push the config panel and wait for it to be the visible screen."""
    app.push_screen(ConfigScreen(**kwargs))
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, ConfigScreen)
    return screen


async def _settle_config(pilot, app: DNALLMPreflightApp) -> None:
    """Let queued input changes start their worker, then let it finish."""
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def test_render_yaml_is_a_facade_member() -> None:
    """The one serializer is public: shells preview exactly what would be written."""
    from dnallm.preflight import render_yaml as facade_render_yaml

    assert facade_render_yaml({"task": {"task_type": "binary"}}) == ("task:\n  task_type: binary\n")


async def test_the_config_screen_renders_the_five_core_sections() -> None:
    """Every core section and field is on the page once the core reports."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        text = _rendered_text(screen)
        headings = [
            str(screen.query_one(f"#config-heading-{section}", Static).content)
            for section in ("task", "model", "finetune", "inference", "lora")
        ]
        fields = [
            screen.query_one(f"#config-field-{section}-{name}", Input).value
            for section, name in (
                ("task", "task_type"),
                ("model", "name"),
                ("finetune", "num_train_epochs"),
                ("inference", "max_length"),
                ("lora", "r"),
            )
        ]

    for title in (
        copy.CONFIG_SECTION_TASK,
        copy.CONFIG_SECTION_MODEL,
        copy.CONFIG_SECTION_FINETUNE,
        copy.CONFIG_SECTION_INFERENCE,
        copy.CONFIG_SECTION_LORA,
    ):
        assert title in headings
    assert copy.CONFIG_PANEL_TITLE in text
    assert fields[0] == "binary"
    assert fields[1] == CONFIG_FINETUNED_REPO
    # Absent from the generated mapping: the field is simply empty, not invented.
    assert fields[2] == ""
    assert fields[3] == "512"


async def test_the_preview_equals_the_serialized_working_mapping() -> None:
    """The preview text is exactly render_yaml(raw) — the bytes a save writes."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        preview = str(screen.query_one("#config-preview", Static).content)
        stored = app.state.generated_config

    assert stored is not None
    assert preview == render_yaml(stored.raw)
    assert "task_type: binary" in preview


async def test_a_base_model_without_a_task_type_shows_the_error() -> None:
    """The core's TASK_TYPE_REQUIRED reaches the pane as one plain sentence.

    GAP A: the pane says what to choose (A2) and never the reason code; the
    core's exact message is one press away behind the details control.
    """
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_BASE_REPO)
        await _settle_config(pilot, app)
        plain_errors = str(screen.query_one("#config-errors", Static).content)
        status = _rendered_text(screen)
        stored = app.state.generated_config
        await pilot.click("#config-error-details")
        await pilot.pause()
        detail_errors = str(screen.query_one("#config-errors", Static).content)
        details_label = screen.query_one("#config-error-details", Button).label

    assert stored is not None
    expected = copy.CONFIG_ERROR_MESSAGES["task", "task_type", "task_type_required"]
    # The error line carries the core's field path and the plain sentence.
    assert copy.CONFIG_ERROR_LINE.format(field=stored.errors[0].field, message=expected) in (
        plain_errors
    )
    assert "TASK_TYPE_REQUIRED" not in plain_errors
    assert "detail from the check:" not in plain_errors
    assert copy.CONFIG_HAS_ERRORS in status
    # The core's exact message is never lost — it is one press away.
    assert "detail from the check:" in detail_errors
    assert "TASK_TYPE_REQUIRED" in detail_errors
    assert str(details_label) == copy.CONFIG_ERROR_DETAILS_HIDE


async def test_an_unknown_model_shows_the_plain_model_sentence() -> None:
    """GAP A, the hand-built model error: ("model", "model") reads as A4.

    This is the real path for a hand-built ``SectionError`` that is not the
    base-model case — the one the earlier suite had no live assertion for.
    """
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model="no/such-model-xyz")
        await _settle_config(pilot, app)
        pane = str(screen.query_one("#config-errors", Static).content)
        rendered = _rendered_text(screen)
        save_disabled = screen.query_one("#config-save", Button).disabled

    expected = copy.CONFIG_ERROR_MESSAGES["model", "model", "model_not_in_registry"]
    assert expected in pane
    assert "MODEL_NOT_IN_REGISTRY_CFG" not in pane
    assert save_disabled
    assert copy.CONFIG_SAVE_DISABLED_REASON in rendered


async def test_at_80x24_the_plain_error_fits_and_the_raw_one_is_a_press_away(tmp_path) -> None:
    """GAP A on the smallest screen: A1 fits, the save reason stays visible.

    The details control is composed above the error pane, so the pane's growth
    cannot push it off-screen; it is pressed directly rather than clicked
    because this test is about the pane, not pointer targeting. A second press
    removes the detail lines again — nothing is truncated away or duplicated.
    """
    config_file = tmp_path / "bad_task.yaml"
    config_file.write_text("task:\n  task_type: not-a-task\n", encoding="utf-8")
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_config_screen(app, pilot, config_path=str(config_file))
        await _settle_config(pilot, app)
        pane = str(screen.query_one("#config-errors", Static).content)
        errors_widget = screen.query_one("#config-errors", Static)
        side = screen.query_one("#config-side", VerticalScroll)
        save_disabled = screen.query_one("#config-save", Button).disabled
        reason = str(screen.query_one("#config-save-reason", Static).content)
        reason_bottom = screen.query_one("#config-save-reason", Static).region.bottom
        height = app.size.height
        details = screen.query_one("#config-error-details", Button)
        enabled = not details.disabled
        label_before = str(details.label)
        assert errors_widget.region.height >= _rows_the_text_needs(errors_widget)
        tail_line = errors_widget.render_line(errors_widget.region.height - 1).text
        assert "binary_classification" in tail_line
        # Textual's scroll_y setter clamps into [0, max_scroll_y] itself.
        side.scroll_y += errors_widget.region.bottom - side.region.bottom
        await pilot.pause()
        assert side.region.contains_point((
            errors_widget.region.x + 1,
            errors_widget.region.bottom - 1,
        ))
        details.press()
        await pilot.pause()
        pane_with_details = str(screen.query_one("#config-errors", Static).content)
        assert errors_widget.region.height >= _rows_the_text_needs(errors_widget)
        side.scroll_y += errors_widget.region.bottom - side.region.bottom
        await pilot.pause()
        assert side.region.contains_point((
            errors_widget.region.x + 1,
            errors_widget.region.bottom - 1,
        ))
        label_after = str(details.label)
        details.press()
        await pilot.pause()
        pane_again = str(screen.query_one("#config-errors", Static).content)
        side.scroll_y = side.max_scroll_y
        await pilot.pause()
        screen.query_one("#config-field-task-task_type", Input).value = "binary"
        await _settle_config(pilot, app)
        pane_after_repair = str(errors_widget.content)
        details_disabled_after_repair = details.disabled
        save_enabled_after_repair = not screen.query_one("#config-save", Button).disabled
        side.scroll_home(animate=False)
        await pilot.pause()
        pane_top_visible_after_repair = side.region.contains_region(errors_widget.region)

        wide_config_file = tmp_path / "several_bad.yaml"
        wide_config_file.write_text(
            "task:\n  task_type: not-a-task\n  num_labels: 1\n"
            "finetune:\n  num_train_epochs: abc\n"
            "inference:\n  batch_size: xyz\n"
            "lora:\n  r: nope\n",
            encoding="utf-8",
        )
        await pilot.resize_terminal(100, 40)
        wide_screen = await _open_config_screen(app, pilot, config_path=str(wide_config_file))
        await _settle_config(pilot, app)
        wide_pane = wide_screen.query_one("#config-errors", Static)
        wide_side = wide_screen.query_one("#config-side", VerticalScroll)
        wide_details = wide_screen.query_one("#config-error-details", Button)
        wide_details.press()
        await pilot.pause()
        wide_laid_out = wide_pane.region.height
        wide_needed = _rows_the_text_needs(wide_pane)
        wide_side.scroll_y += wide_pane.region.bottom - wide_side.region.bottom
        await pilot.pause()
        wide_tail_visible = wide_side.region.contains_point((
            wide_pane.region.x + 1,
            wide_pane.region.bottom - 1,
        ))

    expected = copy.CONFIG_ERROR_MESSAGES["task", "task_type", "not_an_allowed_value"]
    assert expected in pane
    assert "String should match pattern" not in pane
    assert save_disabled
    assert reason == copy.CONFIG_SAVE_DISABLED_REASON
    assert reason_bottom <= height
    assert enabled
    assert label_before == copy.CONFIG_ERROR_DETAILS_SHOW
    assert "detail from the check:" in pane_with_details
    assert "String should match pattern" in pane_with_details
    assert label_after == copy.CONFIG_ERROR_DETAILS_HIDE
    assert pane_again == pane
    # Appended by 25-13 (all pre-existing assertions above stay byte-identical):
    # repairing the last error repaints an empty pane and, once the side is
    # scrolled home, the errors block's top edge is laid out inside its viewport.
    # WR-10 removed the old `scroll_y <= max_scroll_y` check here: Textual's
    # validate_scroll_y clamps into [0, max_scroll_y], so it could never fail.
    assert pane_after_repair == ""
    assert details_disabled_after_repair
    assert save_enabled_after_repair
    assert pane_top_visible_after_repair
    # The several-error case rides this same app run (one resize, one push).
    assert wide_laid_out >= wide_needed
    assert wide_tail_visible


async def test_the_details_control_is_off_while_the_config_is_clean() -> None:
    """With nothing to explain, the detail control is disabled and says Show."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        details = screen.query_one("#config-error-details", Button)
        disabled = details.disabled
        label = str(details.label)
        pane = str(screen.query_one("#config-errors", Static).content)

    assert disabled
    assert label == copy.CONFIG_ERROR_DETAILS_SHOW
    assert pane == ""


# ─────────────────────────────────────────────────────────────────────────────
# Config panel glosses (GAP B, 25-11 Task 2): one plain hint per named term,
# with the config keys and values still verbatim.
# ─────────────────────────────────────────────────────────────────────────────


async def test_every_glossed_field_renders_its_hint_and_keys_stay_verbatim() -> None:
    """Each named term carries its sentence; typing a value keeps the raw key."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        rendered = {
            key: str(screen.query_one(f"#config-hint-{key[0]}-{key[1]}", Static).content)
            for key in copy.CONFIG_FIELD_HINTS
        }
        headings = {
            section: str(screen.query_one(f"#config-heading-{section}", Static).content)
            for section in ("task", "model", "finetune", "inference", "lora")
        }
        unglossed = list(screen.query("#config-hint-model-name"))
        epochs_label = str(
            screen.query_one("#config-label-finetune-num_train_epochs", Static).content
        )
        screen.query_one("#config-field-finetune-num_train_epochs", Input).value = "5"
        await _settle_config(pilot, app)
        preview = str(screen.query_one("#config-preview", Static).content)

    assert rendered == dict(copy.CONFIG_FIELD_HINTS)
    assert headings["inference"] == copy.CONFIG_SECTION_INFERENCE
    assert headings["lora"] == copy.CONFIG_SECTION_LORA
    assert headings["inference"].endswith("(making predictions)")
    assert headings["lora"].endswith("(lightweight fine-tuning)")
    assert unglossed == []
    assert epochs_label == "Epochs"
    assert "num_train_epochs: 5" in preview


async def test_the_suitability_task_legend_names_all_eight_and_survives_typing() -> None:
    """The eight task types are a static legend, not a placeholder that vanishes."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        legend = screen.query_one("#suitability-task-legend", Static)
        placeholder = screen.query_one("#suitability-task-type", Input).placeholder
        before = str(legend.content)
        screen.query_one("#suitability-task-type", Input).value = "binary"
        await pilot.pause()
        after = str(legend.content)

    assert before == copy.SUITABILITY_TASK_LEGEND
    assert placeholder == copy.SUITABILITY_TASK_PROMPT
    assert after == copy.SUITABILITY_TASK_LEGEND
    for value, gloss in copy.TASK_TYPE_GLOSSES.items():
        assert value in before
        assert gloss in before


async def test_the_task_type_legend_is_reachable_at_80x24(monkeypatch) -> None:
    """The ten-row legend must not put the two actions out of reach on 80x24.

    Measured with the legend mounted: the use action starts below the body's
    bottom edge, so it is reachable only by scrolling. Every one of the three
    widgets is proven contained in ``#suitability-body`` after
    ``scroll_to_widget``.
    """
    monkeypatch.setattr(
        preflight_facade, "check_compatibility", FakeCompatibilityCore(_synthetic_report())
    )
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        body = screen.query_one("#suitability-body", VerticalScroll)
        legend = screen.query_one("#suitability-task-legend", Static)
        assert str(legend.content) == copy.SUITABILITY_TASK_LEGEND
        contained = {}
        for selector in ("#suitability-task-legend", "#suitability-check", "#suitability-use"):
            widget = screen.query_one(selector)
            screen.scroll_to_widget(widget, animate=False)
            await pilot.pause()
            contained[selector] = body.region.contains_region(widget.region)

    assert contained == {
        "#suitability-task-legend": True,
        "#suitability-check": True,
        "#suitability-use": True,
    }


async def test_the_hinted_form_stays_legible_at_80x24() -> None:
    """The UAT's Test 2 condition, re-proven with twelve hint lines present.

    Clean leg: the five headings, the read-only facts note, the preview and an
    enabled save are all there, the form scrolls, and the last LoRA hint is
    reachable by scrolling rather than clipped. Broken leg: the save-disabled
    reason stays on screen at 24 rows while the task-type hint still renders.
    """
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        headings = [
            str(screen.query_one(f"#config-heading-{section}", Static).content)
            for section in ("task", "model", "finetune", "inference", "lora")
        ]
        facts = str(screen.query_one("#config-facts-note", Static).content)
        preview = str(screen.query_one("#config-preview", Static).content)
        save_enabled = not screen.query_one("#config-save", Button).disabled
        form = screen.query_one("#config-form", VerticalScroll)
        task_type_input = screen.query_one("#config-field-task-task_type", Input)
        task_type_hint = screen.query_one("#config-hint-task-task_type", Static)
        form_at_top = form.scroll_y == 0
        first_field_contained = form.region.contains_region(task_type_input.region)
        hint_below_input = task_type_hint.region.y == task_type_input.region.bottom
        expected_task_hint = copy.CONFIG_FIELD_HINTS["task", "task_type"]
        hint_text_exact = str(task_type_hint.content) == expected_task_hint
        last_hint = "config-hint-lora-target_modules"
        screen.scroll_to_widget(screen.query_one(f"#{last_hint}", Static), animate=False)
        await pilot.pause()
        hint = screen.query_one(f"#{last_hint}", Static)
        hint_text = str(hint.content)
        hint_reachable = form.region.contains_region(hint.region)
        scrollable = form.max_scroll_y > 0
        form.scroll_to_widget(task_type_hint, animate=False)
        await pilot.pause()
        glosses_top_visible = form.region.contains_point((
            task_type_hint.region.x + 1,
            task_type_hint.region.y,
        ))
        form.scroll_y = min(
            form.max_scroll_y,
            form.scroll_y + (task_type_hint.region.bottom - form.region.bottom),
        )
        await pilot.pause()
        glosses_bottom_visible = form.region.contains_point((
            task_type_hint.region.x + 1,
            task_type_hint.region.bottom - 1,
        ))

        screen.query_one("#config-field-task-task_type", Input).value = "not-a-task"
        await _settle_config(pilot, app)
        save_disabled = screen.query_one("#config-save", Button).disabled
        reason = str(screen.query_one("#config-save-reason", Static).content)
        reason_bottom = screen.query_one("#config-save-reason", Static).region.bottom
        height = app.size.height
        task_hint = str(screen.query_one("#config-hint-task-task_type", Static).content)
        form.scroll_home(animate=False)
        await pilot.pause()
        error_state_contained = form.region.contains_region(task_type_input.region)

    assert headings == [
        copy.CONFIG_SECTION_TASK,
        copy.CONFIG_SECTION_MODEL,
        copy.CONFIG_SECTION_FINETUNE,
        copy.CONFIG_SECTION_INFERENCE,
        copy.CONFIG_SECTION_LORA,
    ]
    assert facts == copy.CONFIG_READONLY_FACTS_NOTE
    assert "task_type: binary" in preview
    assert save_enabled
    assert scrollable
    assert hint_text == copy.CONFIG_FIELD_HINTS["lora", "target_modules"]
    assert hint_reachable
    assert save_disabled
    assert reason == copy.CONFIG_SAVE_DISABLED_REASON
    assert reason_bottom <= height
    assert task_hint == copy.CONFIG_FIELD_HINTS["task", "task_type"]
    # Appended by 25-13 (every pre-existing assertion above stays byte-identical):
    # the panel's first control sits inside the 80x24 form before any scrolling,
    # in the clean state and again while a validation error is showing, and its
    # hint renders directly beneath it with all eight glosses reachable.
    assert form_at_top
    assert first_field_contained
    assert error_state_contained
    assert hint_below_input
    assert hint_text_exact
    assert glosses_top_visible
    assert glosses_bottom_visible


async def test_the_preview_widget_is_read_only_text() -> None:
    """No raw-YAML editor: the preview is rendered text, never an input."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        preview = screen.query_one("#config-preview")

    assert isinstance(preview, Static)
    assert not isinstance(preview, Input)


# ─────────────────────────────────────────────────────────────────────────────
# Config panel (TUI-06, 25-08 Task 2): edit round-trip, read-only facts,
# live revalidation, and the save gate (D-11 / D-12 / D-13)
# ─────────────────────────────────────────────────────────────────────────────

# A loaded config that carries a section the form does not cover. ia3 is
# outside the five core sections, so it must ride through an edit + save
# byte-identically (D-11, 23-04 zero-data-loss round-trip). The fixture is
# written in the shared serializer's own style, so the byte comparison below
# tests data survival rather than a difference in quoting style.
CONFIG_WITH_IA3 = """\
task:
  task_type: binary
  num_labels: 2
  label_names:
  - negative
  - positive
finetune:
  learning_rate: 2.0e-05
  num_train_epochs: 3
ia3:
  target_modules:
  - query
  - value
  feedforward_modules:
  - dense
  r: 16
"""

# Two broken sections in two different sections: the core always reports task
# first, so a screen-side sort (alphabetical, or by field name) would show it.
CONFIG_TWO_PROBLEMS = """\
finetune:
  max_grad_norm: -1.0
task:
  task_type: not-a-task
"""


def _section_block(text: str, section: str) -> str:
    """The exact text of one top-level YAML section, header line included."""
    lines = text.splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.startswith(f"{section}:"))
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip()):
        end += 1
    return "".join(lines[start:end])


async def test_an_out_of_scope_section_survives_an_edit_and_save_verbatim(tmp_path) -> None:
    """Editing a core field never drops ia3: it is kept byte-identically."""
    config_file = tmp_path / "with_ia3.yaml"
    config_file.write_text(CONFIG_WITH_IA3, encoding="utf-8")
    out_path = tmp_path / "saved.yaml"
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, config_path=str(config_file))
        await _settle_config(pilot, app)
        screen.query_one("#config-field-finetune-num_train_epochs", Input).value = "7"
        await _settle_config(pilot, app)
        screen.query_one("#config-output-path", Input).value = str(out_path)
        await pilot.click("#config-save")
        await _settle_config(pilot, app)
        saved_text = out_path.read_text(encoding="utf-8")

    assert _section_block(saved_text, "ia3") == _section_block(CONFIG_WITH_IA3, "ia3")
    saved = yaml.safe_load(saved_text)
    assert saved["ia3"] == {
        "target_modules": ["query", "value"],
        "feedforward_modules": ["dense"],
        "r": 16,
    }
    # The edit really happened, so the survival is not a no-op save.
    assert saved["finetune"]["num_train_epochs"] == 7


async def test_the_kept_as_is_block_names_an_out_of_scope_section(tmp_path) -> None:
    """A preserved section is shown, read-only, so nothing is lost unseen."""
    config_file = tmp_path / "with_ia3.yaml"
    config_file.write_text(CONFIG_WITH_IA3, encoding="utf-8")
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, config_path=str(config_file))
        await _settle_config(pilot, app)
        kept_text = _rendered_text(screen)
        heading_visible = screen.query_one("#config-kept-as-is-heading", Static).display
        input_ids = [widget.id for widget in screen.query(Input)]

    assert copy.CONFIG_KEPT_AS_IS_HEADING in kept_text
    assert heading_visible
    assert copy.CONFIG_KEPT_AS_IS_LINE.format(section="ia3") in kept_text
    assert all(not (widget_id or "").startswith("config-field-ia3") for widget_id in input_ids)


async def test_the_error_pane_lists_errors_in_the_core_section_order(tmp_path) -> None:
    """The pane is the core's list, in the core's order — never re-sorted."""
    config_file = tmp_path / "broken.yaml"
    config_file.write_text(CONFIG_TWO_PROBLEMS, encoding="utf-8")
    expected = edit_config(config_file, {})
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, config_path=str(config_file))
        await _settle_config(pilot, app)
        lines = str(screen.query_one("#config-errors", Static).content).splitlines()

    assert len(expected.errors) >= 2
    assert lines == [
        copy.CONFIG_ERROR_LINE.format(
            field=error.field,
            message=copy.config_error_message(error.section, error.field, error.message),
        )
        for error in expected.errors
    ]
    # The plain sentences replaced the raw messages; no jargon leaked through.
    assert "String should match pattern" not in "\n".join(lines)
    # task is reported before finetune, exactly as the core ordered them.
    assert expected.errors[0].section == "task"
    assert expected.errors[-1].section == "finetune"


async def test_architectural_facts_are_read_only_and_policy_fields_editable() -> None:
    """D-12: checkpoint facts are disabled; policy fields stay open."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        facts = {
            name: screen.query_one(f"#config-field-task-{name}", Input).disabled
            for name in ("num_labels", "label_names", "threshold")
        }
        editable = {
            "max_length": screen.query_one("#config-field-inference-max_length", Input).disabled,
            "epochs": screen.query_one("#config-field-finetune-num_train_epochs", Input).disabled,
            "learning_rate": screen.query_one(
                "#config-field-finetune-learning_rate", Input
            ).disabled,
            "batch_size": screen.query_one(
                "#config-field-finetune-per_device_train_batch_size", Input
            ).disabled,
            "output_dir": screen.query_one("#config-field-finetune-output_dir", Input).disabled,
        }
        note = _rendered_text(screen)

    assert facts == {"num_labels": True, "label_names": True, "threshold": True}
    assert editable == {
        "max_length": False,
        "epochs": False,
        "learning_rate": False,
        "batch_size": False,
        "output_dir": False,
    }
    assert copy.CONFIG_READONLY_FACTS_NOTE in note


async def test_editing_a_field_revalidates_live_and_gates_the_save(tmp_path) -> None:
    """A bad value disables saving; fixing it enables it again (D-13)."""
    out_path = tmp_path / "never.yaml"
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        clean_button = screen.query_one("#config-save", Button).disabled

        screen.query_one("#config-field-finetune-learning_rate", Input).value = "not-a-number"
        await _settle_config(pilot, app)
        broken_button = screen.query_one("#config-save", Button).disabled
        broken_errors = str(screen.query_one("#config-errors", Static).content)

        screen.query_one("#config-field-finetune-learning_rate", Input).value = "0.001"
        await _settle_config(pilot, app)
        fixed_button = screen.query_one("#config-save", Button).disabled
        fixed_errors = str(screen.query_one("#config-errors", Static).content)
        stored = app.state.generated_config

    assert clean_button is False
    assert broken_button is True
    assert "finetune.learning_rate" in broken_errors
    assert fixed_button is False
    assert fixed_errors == ""
    assert stored is not None
    assert stored.raw["finetune"]["learning_rate"] == 0.001
    assert not out_path.exists()


async def test_save_with_an_error_writes_nothing(tmp_path) -> None:
    """SC3: with any problem on the list, no file is ever written."""
    out_path = tmp_path / "invalid.yaml"
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_BASE_REPO)
        await _settle_config(pilot, app)
        screen.query_one("#config-output-path", Input).value = str(out_path)
        disabled = screen.query_one("#config-save", Button).disabled
        reason = _rendered_text(screen)
        # Even a forced call is a no-op while the list is non-empty.
        screen._save()
        await _settle_config(pilot, app)
        exists = out_path.exists()

    assert disabled
    assert copy.CONFIG_SAVE_DISABLED_REASON in reason
    assert exists is False


async def test_save_writes_the_previewed_document_when_clean(tmp_path) -> None:
    """A clean save writes exactly render_yaml(working mapping)."""
    out_path = tmp_path / "clean.yaml"
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        preview = str(screen.query_one("#config-preview", Static).content)
        screen.query_one("#config-output-path", Input).value = str(out_path)
        await pilot.click("#config-save")
        await _settle_config(pilot, app)
        status = _rendered_text(screen)
        stored = app.state.generated_config
        export_path = app.state.export_path
        written = out_path.read_text(encoding="utf-8")

    assert stored is not None
    assert written == render_yaml(stored.raw)
    assert written == preview
    assert export_path == str(out_path)
    assert copy.CONFIG_SAVED.format(path=out_path) in status


async def test_back_navigation_keeps_the_working_mapping(tmp_path) -> None:
    """Escape leaves the panel; the mapping stays on the wizard state."""
    config_file = tmp_path / "with_ia3.yaml"
    config_file.write_text(CONFIG_WITH_IA3, encoding="utf-8")
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, config_path=str(config_file))
        await _settle_config(pilot, app)
        screen.query_one("#config-field-finetune-num_train_epochs", Input).value = "9"
        await _settle_config(pilot, app)
        stored_before = app.state.generated_config
        await pilot.press("escape")
        await pilot.pause()
        kept = app.state.generated_config
        returned = isinstance(app.screen, EnvScreen)

    assert returned
    assert kept is stored_before
    assert kept is not None
    assert kept.raw["ia3"] == {
        "target_modules": ["query", "value"],
        "feedforward_modules": ["dense"],
        "r": 16,
    }
    assert kept.raw["finetune"]["num_train_epochs"] == 9


# ─────────────────────────────────────────────────────────────────────────────
# Dual entry (D-07) and the suitability → config edge (D-09), 25-08 Task 3
# ─────────────────────────────────────────────────────────────────────────────


class FakeConfigCore:
    """``generate_config`` / ``edit_config`` stand-ins that record every call.

    Args:
        generated: The frozen result both stand-ins return.
    """

    def __init__(self, generated: GeneratedConfig) -> None:
        self.generated = generated
        self.generate_calls: list[dict[str, Any]] = []
        self.edit_calls: list[dict[str, Any]] = []

    def generate(self, model: str, *, task_type: str | None = None) -> GeneratedConfig:
        """Record the generation request and return the canned result."""
        self.generate_calls.append({"model": model, "task_type": task_type})
        return self.generated

    def edit(
        self,
        existing: Any,
        patch: Any = None,
        *,
        output_path: Any = None,
    ) -> GeneratedConfig:
        """Record the edit request and return the canned result."""
        self.edit_calls.append({"existing": existing, "patch": patch, "output_path": output_path})
        return self.generated


def _install_config_core(monkeypatch, fake: FakeConfigCore) -> None:
    """Point the preflight facade's config entry points at *fake*."""
    monkeypatch.setattr(preflight_facade, "generate_config", fake.generate)
    monkeypatch.setattr(preflight_facade, "edit_config", fake.edit)


async def _open_entry_screen(app: DNALLMPreflightApp, pilot) -> EntryScreen:
    """Reach the entry panel the way a user does: through the env panel."""
    await pilot.pause()
    await pilot.click("#env-continue")
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, EntryScreen)
    return screen


async def test_start_from_my_data_opens_the_data_panel() -> None:
    """The first door keeps the D-09 order: data comes next."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        await _open_entry_screen(app, pilot)
        await pilot.click("#entry-from-data")
        await pilot.pause()
        assert isinstance(app.screen, DataScreen)


async def test_edit_an_existing_config_opens_it_through_the_core(tmp_path, monkeypatch) -> None:
    """The second door loads the pasted path through edit_config, never raw YAML."""
    config_file = tmp_path / "existing.yaml"
    config_file.write_text("task:\n  task_type: binary\n", encoding="utf-8")
    expected = edit_config(config_file, {})
    fake = FakeConfigCore(expected)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_entry_screen(app, pilot)
        screen.query_one("#entry-config-path", Input).value = str(config_file)
        await pilot.click("#entry-from-config")
        await pilot.pause()
        active = await _wait_until(lambda: isinstance(app.screen, ConfigScreen))
        await _settle_config(pilot, app)

    assert active
    assert fake.edit_calls[0]["existing"] == str(config_file)
    assert fake.edit_calls[0]["patch"] == {}
    assert fake.edit_calls[0]["output_path"] is None


async def test_edit_an_existing_config_with_an_empty_path_prompts(monkeypatch) -> None:
    """An empty path is a prompt, not a read: no core call is made."""
    fake = FakeConfigCore(GeneratedConfig(config={}, errors=[], raw={}))
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_entry_screen(app, pilot)
        await pilot.click("#entry-from-config")
        await pilot.pause()
        text = _rendered_text(screen)
        still_entry = isinstance(app.screen, EntryScreen)

    assert still_entry
    assert copy.ENTRY_CONFIG_EMPTY in text
    assert fake.edit_calls == []


async def test_use_this_model_opens_the_config_panel_with_that_model(monkeypatch) -> None:
    """The suitability → config edge (D-09) hands the chosen model to the core."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    generated = edit_config({"task": {"task_type": "binary"}}, {})
    fake = FakeConfigCore(generated)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        table = screen.query_one("#suitability-listing", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()
        await pilot.click("#suitability-use")
        await pilot.pause()
        active = isinstance(app.screen, ConfigScreen)
        await _settle_config(pilot, app)

    assert active
    assert fake.generate_calls[0]["model"] == report.listing[1].model
    assert app.state.selected_entry is report.listing[1]


async def test_use_this_model_stays_disabled_without_a_selection(monkeypatch) -> None:
    """With no listing row to choose, the action is disabled and says why."""
    empty = CompatibilityReport(verdicts=[], listing=[])
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(empty))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    fake = FakeConfigCore(GeneratedConfig(config={}, errors=[], raw={}))
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        disabled = screen.query_one("#suitability-use", Button).disabled
        await pilot.click("#suitability-use")
        await pilot.pause()
        still_suitability = isinstance(app.screen, SuitabilityScreen)

    assert disabled
    assert still_suitability
    assert fake.generate_calls == []


async def test_back_from_the_config_panel_returns_to_the_entry_screen(
    tmp_path, monkeypatch
) -> None:
    """Escape goes back one step and the loaded mapping survives the trip."""
    config_file = tmp_path / "existing.yaml"
    config_file.write_text(CONFIG_WITH_IA3, encoding="utf-8")
    expected = edit_config(config_file, {})
    fake = FakeConfigCore(expected)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_entry_screen(app, pilot)
        screen.query_one("#entry-config-path", Input).value = str(config_file)
        await pilot.click("#entry-from-config")
        await pilot.pause()
        await _settle_config(pilot, app)
        loaded = app.state.generated_config
        await pilot.press("escape")
        await pilot.pause()
        returned = isinstance(app.screen, EntryScreen)
        kept = app.state.generated_config

    assert returned
    assert loaded is not None
    assert kept is loaded
    assert kept.raw["ia3"]["r"] == 16


# ─────────────────────────────────────────────────────────────────────────────
# Download and summary panels (TUI-07), 25-09 Task 1
# ─────────────────────────────────────────────────────────────────────────────

# A real registry entry (lab model), so the hub default the worker picks is the
# same one production would pick for it.
DOWNLOAD_MODEL_REPO = "zhangtaolab/plant-dnabert-BPE-promoter"
DOWNLOAD_RETURNED_PATH = "/home/user/.cache/modelscope/plant-dnabert-BPE-promoter"
CONFIG_EXPORT_PATH = "/home/user/work/dnallm_config.yaml"


class FakeDownloader:
    """A hub-downloader stand-in: counts calls, can block, can fail.

    Args:
        path: The string the fake "stores the files in".
        gate: When set, the first call blocks until this event is set.
        failure: When set, every call raises it instead of returning a path.
    """

    def __init__(self, path: str = DOWNLOAD_RETURNED_PATH, gate=None, failure=None) -> None:
        self.path = path
        self.gate = gate
        self.failure = failure
        self.calls: list[str] = []

    def __call__(self, model_name: str, **kwargs: object) -> str:
        """Record the attempt, then block, fail, or return the stored path."""
        self.calls.append(model_name)
        if self.gate is not None:
            self.gate.wait(timeout=20)
        if self.failure is not None:
            raise self.failure
        return self.path


def _download_entry(availability: AvailabilityReport | None = None) -> ListingEntry:
    """The frozen listing row the suitability panel would have recorded."""
    return ListingEntry(
        model=DOWNLOAD_MODEL_REPO,
        entry_name="plant-dnabert-BPE-promoter",
        is_lab=True,
        task_type="binary_classification",
        availability=availability
        or AvailabilityReport(
            state=AvailabilityState.REMOTE_CONFIRMED,
            size_bytes=1_500_000_000,
            source_hub=Hub.HUGGINGFACE,
        ),
    )


async def _wait_until(condition, tries: int = 500) -> bool:
    """Poll *condition* while yielding to the event loop, for up to ~5 seconds."""
    for _ in range(tries):
        if condition():
            return True
        await asyncio.sleep(0.01)
    return False


async def _open_download_screen(app: DNALLMPreflightApp, pilot) -> DownloadScreen:
    """Push the download panel and wait for it to settle as the visible screen."""
    app.push_screen(DownloadScreen())
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, DownloadScreen)
    return screen


async def _wait_for_the_download(pilot, app: DNALLMPreflightApp) -> None:
    """Wait for the transfer's own daemon thread, then let its messages land.

    The download is deliberately not a Textual worker (WR-02), so
    ``app.workers.wait_for_complete()`` cannot see it.
    """
    operation = app._download_operation
    if operation is not None:
        for _ in range(2000):
            if not operation.is_alive:
                break
            await asyncio.sleep(0.005)
    await pilot.pause()
    await pilot.pause()


async def test_the_download_panel_shows_progress_then_the_summary() -> None:
    """Progress is visible while the download runs, then both paths are listed."""
    gate = threading.Event()
    downloader = FakeDownloader(gate=gate)
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()
    app.state.export_path = CONFIG_EXPORT_PATH

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        assert downloader.calls == []
        await pilot.click("#download-start")
        started = await _wait_until(lambda: bool(downloader.calls))
        await pilot.pause()
        in_progress = str(screen.query_one("#download-progress", Static).content)
        still_downloading = app.screen is screen
        gate.set()
        await _wait_for_the_download(pilot, app)
        summary = app.screen
        assert isinstance(summary, SummaryScreen)
        text = _rendered_text(summary)

    assert started, "the download worker never reached the downloader"
    assert still_downloading
    assert in_progress == copy.DOWNLOAD_ATTEMPT.format(attempt=1, max_try=DOWNLOAD_MAX_TRY)
    assert downloader.calls == [DOWNLOAD_MODEL_REPO]
    # The path is shown exactly as the downloader returned it, and the config
    # path follows it (the fixed order of D-09).
    assert DOWNLOAD_RETURNED_PATH in text
    assert CONFIG_EXPORT_PATH in text
    assert text.index(DOWNLOAD_RETURNED_PATH) < text.index(CONFIG_EXPORT_PATH)


async def test_the_summary_lists_the_model_path_before_the_config_path() -> None:
    """The two rows keep their fixed order, with both headings visible."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(
            SummaryScreen(model_path=DOWNLOAD_RETURNED_PATH, config_path=CONFIG_EXPORT_PATH)
        )
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SummaryScreen)
        text = _rendered_text(screen)
        model_heading = str(screen.query_one("#summary-model-heading", Static).content)
        config_heading = str(screen.query_one("#summary-config-heading", Static).content)

    assert model_heading == copy.SUMMARY_HEADINGS["model"]
    assert config_heading == copy.SUMMARY_HEADINGS["config"]
    assert text.index(model_heading) < text.index(DOWNLOAD_RETURNED_PATH)
    assert text.index(DOWNLOAD_RETURNED_PATH) < text.index(config_heading)
    assert text.index(config_heading) < text.index(CONFIG_EXPORT_PATH)


async def test_a_saved_config_opens_the_download_step() -> None:
    """The config panel's way on appears only after a save, and reaches download."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    app.state.selected_entry = _download_entry()

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        disabled_before = screen.query_one("#config-continue", Button).disabled
        app.state.export_path = CONFIG_EXPORT_PATH
        screen._update_continue_state()  # the same refresh a save performs
        await pilot.pause()
        disabled_after = screen.query_one("#config-continue", Button).disabled
        await pilot.click("#config-continue")
        await pilot.pause()
        opened_download = isinstance(app.screen, DownloadScreen)

    assert disabled_before
    assert not disabled_after
    assert opened_download


# ─────────────────────────────────────────────────────────────────────────────
# Hub choice, byte progress, and the plain-language failure state (25-09 Task 2)
# ─────────────────────────────────────────────────────────────────────────────


def test_format_size_is_honest_about_an_unknown_size() -> None:
    """A missing or zero size renders a phrase, never a division by zero."""
    from dnallm.tui.screens.download import _format_size

    assert _format_size(None) == copy.DOWNLOAD_SIZE_UNKNOWN
    assert _format_size(0) == copy.DOWNLOAD_SIZE_UNKNOWN
    assert _format_size(1_500_000_000) == "1.5 GB"


async def test_the_download_panel_shows_the_reported_size_or_that_it_is_unknown() -> None:
    """The size line comes from the availability report, and says so when absent."""
    known_entry = _download_entry(
        AvailabilityReport(
            state=AvailabilityState.REMOTE_CONFIRMED,
            size_bytes=2_400_000_000,
            source_hub=Hub.MODELSCOPE,
        )
    )
    unknown_entry = _download_entry(
        AvailabilityReport(state=AvailabilityState.UNKNOWN, size_bytes=None)
    )

    texts: list[str] = []
    for entry in (known_entry, unknown_entry):
        app = DNALLMPreflightApp(collector=FakeCollector())
        app.state.selected_entry = entry
        async with app.run_test(size=(100, 40)) as pilot:
            screen = await _open_download_screen(app, pilot)
            texts.append(str(screen.query_one("#download-availability", Static).content))

    assert texts[0] == copy.DOWNLOAD_SIZE_LABEL.format(size="2.4 GB")
    assert texts[1] == copy.DOWNLOAD_SIZE_LABEL.format(size=copy.DOWNLOAD_SIZE_UNKNOWN)


async def test_a_failed_download_reads_plainly_and_retry_tries_again() -> None:
    """download_model's ValueError becomes a plain sentence, and retry re-runs it."""
    # "not found" makes download_model stop after one attempt (no retry sleep).
    downloader = FakeDownloader(failure=RuntimeError("repo is not found"))
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        await pilot.click("#download-start")
        started = await _wait_until(lambda: bool(downloader.calls))
        await _wait_for_the_download(pilot, app)
        text = _rendered_text(screen)
        retry_visible = screen.query_one("#download-retry", Button).display
        still_on_download = app.screen is screen
        calls_after_failure = list(downloader.calls)
        await pilot.click("#download-retry")
        await _wait_for_the_download(pilot, app)
        calls_after_retry = list(downloader.calls)

    assert started
    assert still_on_download, "a failed download must not jump to the summary"
    assert retry_visible
    assert copy.DOWNLOAD_FAILED in text
    assert "RuntimeError" not in text
    assert "ValueError" not in text
    assert "Traceback" not in text
    assert calls_after_failure == [DOWNLOAD_MODEL_REPO]
    assert calls_after_retry == [DOWNLOAD_MODEL_REPO, DOWNLOAD_MODEL_REPO]


# ─────────────────────────────────────────────────────────────────────────────
# Cooperative cancel, quit safety, and the skip path (25-09 Task 3)
# ─────────────────────────────────────────────────────────────────────────────


async def test_cancel_between_attempts_stops_after_exactly_one_call(monkeypatch) -> None:
    """A stop mid-attempt lets that attempt finish and blocks the next one."""
    from types import SimpleNamespace

    from dnallm.models import model as model_module

    # download_model sleeps one second before each retry; keep it out of wall time.
    monkeypatch.setattr(model_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    gate = threading.Event()
    downloader = FakeDownloader(gate=gate, failure=ConnectionError("connection reset"))
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        await pilot.click("#download-start")
        started = await _wait_until(lambda: bool(downloader.calls))
        await pilot.click("#download-cancel")
        await pilot.pause()
        pending_note = str(screen.query_one("#download-status", Static).content)
        gate.set()
        await _wait_for_the_download(pilot, app)
        text = _rendered_text(screen)
        retry_visible = screen.query_one("#download-retry", Button).display
        calls = list(downloader.calls)

    assert started
    # The stop is honest while the attempt is still running...
    assert pending_note == copy.CANCEL_PENDING_NOTE
    # ...and the sentinel escapes the retry loop: no second attempt, ever.
    assert calls == [DOWNLOAD_MODEL_REPO]
    assert copy.DOWNLOAD_CANCELLED in text
    assert retry_visible
    assert "Cancelled" not in text.replace(copy.DOWNLOAD_CANCELLED, "")


async def test_quitting_during_a_download_is_prompt() -> None:
    """Quit never waits for a blocked transfer — not even at process exit.

    The previous version released its gate in ``finally`` and measured only
    the app's teardown, so it passed by construction. The property WR-02 is
    about is structural: the transfer must run on a daemon thread outside
    Textual's executor, because ``asyncio.run`` and the interpreter both join
    that executor at exit — which would freeze a real process for as long as
    the hub call takes. Here the transfer is still blocked on its gate after
    the app has gone, and its thread is a daemon, so nothing can join it.
    """
    gate = threading.Event()
    downloader = FakeDownloader(gate=gate)
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()
    started_at = time.monotonic()
    operation = None
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await _open_download_screen(app, pilot)
            await pilot.click("#download-start")
            assert await _wait_until(lambda: bool(downloader.calls))
            operation = app._download_operation
            assert operation is not None
            await pilot.press("q")
            # Quitting asked it to stop, but the in-flight attempt is
            # uninterruptible and the app did not wait for it.
            still_blocked = operation.is_alive
        elapsed = time.monotonic() - started_at

        assert still_blocked, "quit waited for the transfer"
        assert operation.is_daemon, "a non-daemon thread would freeze process exit"
        # Only an app that did not join the thread can get here while it is
        # still blocked: a Textual worker would have kept teardown waiting.
        assert elapsed < 20
    finally:
        gate.set()
        if operation is not None:
            operation.join(10)


async def test_skipping_the_download_reaches_the_summary_without_downloading() -> None:
    """Skip is a first-class path: no downloader call, both rows still in order."""
    downloader = FakeDownloader()
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()
    app.state.export_path = CONFIG_EXPORT_PATH

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        await pilot.click("#download-skip")
        await pilot.pause()
        summary = app.screen
        assert isinstance(summary, SummaryScreen)
        # Header mounts HeaderTitle asynchronously; poll for it before rendering.
        header_mounted = await _wait_until(lambda: len(summary.query("HeaderTitle")) == 1)
        assert header_mounted, "summary HeaderTitle did not mount within the poll window"
        text = _rendered_text(summary)

    assert downloader.calls == []
    assert copy.SUMMARY_MODEL_SKIPPED in text
    assert CONFIG_EXPORT_PATH in text
    # The fixed order holds even when nothing was downloaded.
    assert text.index(copy.SUMMARY_HEADINGS["model"]) < text.index(copy.SUMMARY_MODEL_SKIPPED)
    assert text.index(copy.SUMMARY_MODEL_SKIPPED) < text.index(copy.SUMMARY_HEADINGS["config"])
    assert text.index(copy.SUMMARY_HEADINGS["config"]) < text.index(CONFIG_EXPORT_PATH)


async def test_the_closing_step_without_a_model_goes_straight_to_the_summary() -> None:
    """With no model chosen the step offers nothing to download (TUI-07/empty)."""
    downloader = FakeDownloader()
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = None

    async with app.run_test(size=(100, 40)) as pilot:
        app.show_download()
        await pilot.pause()
        summary = app.screen
        assert isinstance(summary, SummaryScreen)
        text = _rendered_text(summary)

    assert downloader.calls == []
    assert copy.SUMMARY_MODEL_SKIPPED in text
    assert copy.SUMMARY_CONFIG_MISSING in text


def test_the_download_panel_never_imports_a_hub_client() -> None:
    """The screen is presentation only: hub clients stay behind the seam.

    This is the plan's "no network in tests" guard on the screen itself: a hub
    import here would mean the panel could download without the app's worker,
    its cancel flag, and its plain-language error mapping.
    """
    import ast
    from pathlib import Path

    import dnallm.tui.screens.download as download_module

    source = Path(download_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported & {"huggingface_hub", "modelscope"} == set()
    assert imported & {"torch", "transformers", "datasets"} == set()
    # The only way to a download is the app's worker, never download_model.
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    referenced |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "download_model" not in referenced


# ─────────────────────────────────────────────────────────────────────────────
# Panel→core table (TUI-08), 25-10 Task 2
#
# Presentation-only is a claim, so it is a table: every screen module names the
# exact set of core functions it may reach, and the test proves the table in
# both directions — nothing missing, nothing extra. Screens reach the core two
# ways: directly (an import or a call) or through the 25-02 thread-worker seam
# (`self.app.<worker>` / `self._preflight_app.<worker>`), which is why the seam
# map is declared next to the allowlist and expanded through it.
# ─────────────────────────────────────────────────────────────────────────────

PACKAGE_ROOT = Path(dnallm.__file__).parent
SCREENS_DIR = PACKAGE_ROOT / "tui" / "screens"
WORKERS_MODULE = PACKAGE_ROOT / "tui" / "workers.py"
_SCREEN_PACKAGE = "dnallm.tui.screens"
_CORE_MODULE_PREFIXES = ("dnallm.preflight", "dnallm.models", "dnallm.utils.env_info")

#: The presentation-only contract. One entry per module under dnallm/tui/screens/.
#: ``__init__.py`` is listed with an empty set so "every module present" is
#: literal: a new screen module cannot appear without an allowlist decision.
SCREEN_CORE_ALLOWLIST: dict[str, frozenset[str]] = {
    "__init__.py": frozenset(),
    "config.py": frozenset({"generate_config", "edit_config", "render_yaml"}),
    "data.py": frozenset({"validate_data"}),
    "download.py": frozenset({"download_model"}),
    "entry.py": frozenset(),
    "env.py": frozenset({"collect_versions", "collect_device", "collect_disk"}),
    "suitability.py": frozenset({
        "check_compatibility",
        "local_availability",
        "probe_availability",
    }),
    "summary.py": frozenset(),
}

#: What each worker seam method reaches in the core. A screen that calls a
#: worker method inherits that method's core surface; a worker method without an
#: entry here fails the test, and so does an entry naming a method that
#: ``PreflightWorkers`` does not define.
SEAM_CORE_MAP: dict[str, frozenset[str]] = {
    "collect_env_worker": frozenset({"collect_device", "collect_disk"}),
    "validate_data_worker": frozenset({"validate_data"}),
    "check_compatibility_worker": frozenset({"check_compatibility"}),
    "local_availability_batch_worker": frozenset({"local_availability"}),
    "probe_availability_worker": frozenset({"probe_availability"}),
    "generate_config_worker": frozenset({"generate_config"}),
    "edit_config_worker": frozenset({"edit_config"}),
    "download_worker": frozenset({"download_model"}),
}


def _screen_module_paths() -> dict[str, Path]:
    """Every .py file under dnallm/tui/screens/, keyed by file name."""
    return {path.name: path for path in sorted(SCREENS_DIR.glob("*.py"))}


def _top_level_function_names(path: Path) -> set[str]:
    """Names of the functions defined at module scope in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _core_function_names() -> set[str]:
    """Every function the three core surfaces define.

    Types are deliberately excluded: a screen may import a frozen report type
    for typing, and that is not a reach into core behaviour. Only callables
    count, which is what "presentation-only" is about.
    """
    paths = [
        *(PACKAGE_ROOT / "preflight").glob("*.py"),
        PACKAGE_ROOT / "utils" / "env_info.py",
        PACKAGE_ROOT / "models" / "model.py",
    ]
    names: set[str] = set()
    for path in paths:
        names |= _top_level_function_names(path)
    return names


def _resolve_relative_module(module: str, level: int) -> str:
    """Resolve a screen module's relative import to its absolute dotted target."""
    if level == 0:
        return module
    parts = _SCREEN_PACKAGE.split(".")
    parts = parts[: len(parts) - (level - 1)]
    base = ".".join(parts)
    return f"{base}.{module}" if module else base


def _is_core_module(target: str) -> bool:
    """True when *target* is one of the three core surfaces a screen may reach."""
    return any(
        target == prefix or target.startswith(f"{prefix}.") for prefix in _CORE_MODULE_PREFIXES
    )


def _called_name(func: ast.expr) -> str | None:
    """The name a call expression invokes (``f`` for ``f(...)``, ``g`` for ``x.g(...)``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _direct_core_names(tree: ast.AST, core_names: set[str]) -> set[str]:
    """Core functions *tree* imports from, or calls, without going through a worker."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _is_core_module(_resolve_relative_module(node.module or "", node.level)):
                found |= {alias.name for alias in node.names if alias.name in core_names}
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_core_module(alias.name):
                    found.add(alias.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in core_names:
                found.add(name)
    return found


def _is_self_app_chain(node: ast.expr) -> bool:
    """True for the expression ``self.app`` or ``self._preflight_app``."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in {"app", "_preflight_app"}
    )


def _app_alias_names(tree: ast.AST) -> set[str]:
    """Local names a screen binds to its own app (``app = self._preflight_app``).

    An alias is still the app: without this, a one-line local assignment would
    hide a seam call from the table.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not _is_self_app_chain(value):
            continue
        aliases |= {target.id for target in targets if isinstance(target, ast.Name)}
    return aliases


def _is_app_bound_attribute(node: ast.expr, aliases: set[str]) -> bool:
    """True for ``<app>.<name>``, where ``<app>`` is ``self.app``, the typed
    app property, or a local alias of either."""
    if not isinstance(node, ast.Attribute):
        return False
    if _is_self_app_chain(node.value):
        return True
    return isinstance(node.value, ast.Name) and node.value.id in aliases


def _app_bound_call_names(tree: ast.AST) -> set[str]:
    """Every method name a screen calls on its app, aliases included."""
    aliases = _app_alias_names(tree)
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_app_bound_attribute(node.func, aliases)
    }


def _preflight_worker_methods() -> set[str]:
    """Every method ``PreflightWorkers`` defines (AST scan, no import of textual)."""
    tree = ast.parse(WORKERS_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PreflightWorkers":
            return {
                member.name
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError("dnallm/tui/workers.py must define PreflightWorkers")


def _screen_expanded_core_names(module_name: str) -> tuple[set[str], set[str]]:
    """The core surface one screen reaches, plus the seam methods it used.

    Args:
        module_name: The screen module's file name.

    Returns:
        ``(core_names, seam_methods)`` — direct imports/calls unioned with every
        seam method expanded through :data:`SEAM_CORE_MAP`.
    """
    tree = ast.parse(_screen_module_paths()[module_name].read_text(encoding="utf-8"))
    core_names = _core_function_names()
    direct = _direct_core_names(tree, core_names)
    seam_methods = _app_bound_call_names(tree) & _preflight_worker_methods()
    expanded = set(direct)
    for seam in seam_methods:
        expanded |= SEAM_CORE_MAP.get(seam, frozenset())
    return expanded, seam_methods


def test_the_allowlist_names_every_screen_module() -> None:
    """A new screen module cannot appear without an allowlist decision."""
    present = set(_screen_module_paths())
    assert set(SCREEN_CORE_ALLOWLIST) == present, (
        f"SCREEN_CORE_ALLOWLIST and dnallm/tui/screens/ disagree: "
        f"unlisted={sorted(present - set(SCREEN_CORE_ALLOWLIST))}, "
        f"missing={sorted(set(SCREEN_CORE_ALLOWLIST) - present)} (TUI-08)"
    )


@pytest.mark.parametrize("module_name", sorted(SCREEN_CORE_ALLOWLIST))
def test_each_screen_reaches_exactly_its_allowlisted_core_functions(module_name: str) -> None:
    """The table holds in both directions for one screen module.

    Extra names mean the screen started doing domain work; missing names mean
    the allowlist is stale. Either one fails.
    """
    expanded, seam_methods = _screen_expanded_core_names(module_name)
    allowed = set(SCREEN_CORE_ALLOWLIST[module_name])

    unmapped = sorted(seam_methods - set(SEAM_CORE_MAP))
    assert unmapped == [], (
        f"{module_name} calls seam methods with no SEAM_CORE_MAP entry: {unmapped} — "
        f"declare what each worker reaches in the core (TUI-08)"
    )

    invented = sorted(expanded - _core_function_names())
    assert invented == [], (
        f"{module_name}'s expanded surface names non-core callables: {invented} — "
        f"SEAM_CORE_MAP may only name core functions (TUI-08)"
    )

    extra = sorted(expanded - allowed)
    missing = sorted(allowed - expanded)
    assert extra == [], (
        f"{module_name} reaches core functions its allowlist does not name: {extra} "
        f"(allowed: {sorted(allowed)}) — screens are presentation only (TUI-08)"
    )
    assert missing == [], (
        f"{module_name} no longer reaches allowlisted core functions: {missing} — "
        f"the allowlist is the contract, so it must match (TUI-08)"
    )


def test_every_seam_method_a_screen_calls_has_a_mapping() -> None:
    """Contract direction 1: no screen may reach an unmapped worker seam."""
    called: set[str] = set()
    for path in _screen_module_paths().values():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called |= _app_bound_call_names(tree) & _preflight_worker_methods()
    assert called - set(SEAM_CORE_MAP) == set(), (
        f"seam methods called by screens with no SEAM_CORE_MAP entry: "
        f"{sorted(called - set(SEAM_CORE_MAP))} (TUI-08)"
    )


def test_every_mapped_seam_name_exists_on_preflight_workers() -> None:
    """Contract direction 2: the map cannot name a method the seam does not define."""
    methods = _preflight_worker_methods()
    unknown = sorted(set(SEAM_CORE_MAP) - methods)
    assert unknown == [], (
        f"SEAM_CORE_MAP names methods PreflightWorkers does not define: {unknown} "
        f"(known: {sorted(methods)}) (TUI-08)"
    )


def test_no_screen_imports_yaml_pandas_or_a_hub_client() -> None:
    """Screens render reports; parsing YAML or talking to a hub is core work."""
    banned = {"yaml", "pandas", "huggingface_hub", "modelscope"}
    offenders: dict[str, list[str]] = {}
    for module_name, path in _screen_module_paths().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        found = sorted(imported & banned)
        if found:
            offenders[module_name] = found
    assert offenders == {}, (
        f"screen modules import a library that belongs to the core: {offenders} — "
        f"the panel renders reports and never parses a config or dials a hub (TUI-08)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 25-10 Task 3: the panel branches the happy-path tests never reach
#
# Every test below exists because a line of a screen could only be reached by a
# specific user action or a specific degraded state: switching a route back,
# pressing Enter instead of clicking, a stale batch arriving late, a section
# that is not a mapping, a stop pressed before anything started. They keep the
# per-module coverage floor honest instead of leaving the branches unasserted.
# ─────────────────────────────────────────────────────────────────────────────


async def test_the_route_controls_switch_both_ways_and_recheck(monkeypatch) -> None:
    """Predict → finetune → check re-runs the core, always with its own vocabulary."""
    fake_core = FakeCompatibilityCore(_synthetic_report())
    monkeypatch.setattr(preflight_facade, "check_compatibility", fake_core)
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        await pilot.click("#suitability-intent-predict")
        await _settle(pilot, app)
        await pilot.click("#suitability-intent-finetune")
        await _settle(pilot, app)
        finetune_variant = screen.query_one("#suitability-intent-finetune", Button).variant
        await pilot.click("#suitability-check")
        await _settle(pilot, app)

    assert [call["intent"] for call in fake_core.calls] == [
        "finetune",
        "predict",
        "finetune",
        "finetune",
    ]
    assert finetune_variant == "primary"


async def test_enter_on_a_listing_row_selects_it_and_verdict_rows_are_ignored(monkeypatch) -> None:
    """Enter is the keyboard equivalent of a click; the verdict table selects nothing."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        # A highlight on the verdicts table must not become a model choice: the
        # listing's own first-row highlight is what the state already holds.
        selection_before = app.state.selected_entry
        screen.query_one("#suitability-verdicts", DataTable).move_cursor(row=0)
        await pilot.pause()
        assert app.state.selected_entry is selection_before
        listing = screen.query_one("#suitability-listing", DataTable)
        listing.focus()
        listing.move_cursor(row=1)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        selected = app.state.selected_entry

    assert selected is report.listing[1]


async def test_a_batch_for_a_listing_the_user_already_replaced_is_ignored(monkeypatch) -> None:
    """A late local batch never overwrites a report the panel has moved past."""
    monkeypatch.setattr(
        preflight_facade, "check_compatibility", FakeCompatibilityCore(_synthetic_report())
    )
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        current = app.state.compatibility_report
        stale = CompatibilityReport(verdicts=[], listing=[])
        await screen.apply_compatibility_ready(stale, {})
        await pilot.pause()
        still = app.state.compatibility_report

    assert still is current
    assert stale is not current


class _GatedSecondCheck:
    """A ``check_compatibility`` stand-in whose second call waits for the test.

    The first call answers at once, so the panel renders a listing and starts
    its offline local pass. The second call blocks, which opens the window the
    CR-01 order needs: the previous listing's local batch can land *before*
    the new report exists.

    Args:
        first: The report the first call returns.
        second: The report the second call returns once released.
    """

    def __init__(self, first: CompatibilityReport, second: CompatibilityReport) -> None:
        self.first = first
        self.second = second
        self.calls = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def __call__(
        self,
        data: Any = None,
        *,
        task_type: str | None = None,
        models: list[str] | None = None,
        intent: str | None = None,
        prober: Any = None,
        probe_shortlist: int | None = None,
    ) -> CompatibilityReport:
        """Answer the first call, and hold the second until released."""
        self.calls += 1
        if self.calls == 1:
            return self.first
        self.second_started.set()
        self.release_second.wait(timeout=30)
        return self.second


async def test_a_stale_local_batch_never_becomes_the_current_report(monkeypatch) -> None:
    """CR-01: a local batch that lands during a re-check is not "the report".

    The failing order is the normal one on a cold cache: the offline local pass
    for the previous listing finishes while the new check is still probing, so
    the batch arrives first. Adoption would set the stale report as current and
    the real report would then be dropped for good — the panel would keep the
    old listing and stay on "probing…" until Check was pressed again.
    """
    first = _synthetic_report()
    second = CompatibilityReport(
        verdicts=[],
        listing=[
            ListingEntry(
                model="zhangtaolab/b",
                entry_name="Lab B2",
                is_lab=True,
                task_type="binary",
            )
        ],
    )
    core = _GatedSecondCheck(first, second)
    monkeypatch.setattr(preflight_facade, "check_compatibility", core)
    batch_gate = threading.Event()
    local_calls: list[str] = []

    def gated_local(model: str) -> AvailabilityReport:
        batch_gate.wait(timeout=30)
        local_calls.append(model)
        return AvailabilityReport(state=AvailabilityState.UNKNOWN)

    monkeypatch.setattr(preflight_facade, "local_availability", gated_local)
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        assert await _wait_until(lambda: app.state.compatibility_report is first)
        # The user asks for a new check while the previous listing's local pass
        # is still running.
        await pilot.click("#suitability-intent-predict")
        assert await _wait_until(core.second_started.is_set)
        batch_gate.set()
        assert await _wait_until(lambda: len(local_calls) == len(first.listing))
        # Deliver the stale batch the way the worker's message would arrive:
        # through the app, before the new report exists.
        app.post_message(
            CompatibilityReady(
                first,
                {"third-party/file-a": AvailabilityReport(state=AvailabilityState.UNKNOWN)},
            )
        )
        await pilot.pause()
        await pilot.pause()
        core.release_second.set()
        assert await _wait_until(lambda: app.state.compatibility_report is second)
        await pilot.pause()
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))
        status = str(screen.query_one("#suitability-status", Static).content)

    assert app.state.compatibility_report is second
    assert [row[0] for row in rows] == ["Lab B2"]
    assert copy.PROBING_PLACEHOLDER not in status


async def test_a_local_batch_is_only_valid_for_the_report_on_screen(monkeypatch) -> None:
    """A batch may repaint the current report; it may never become the report."""
    current = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(current))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        stale = CompatibilityReport(verdicts=[], listing=[])
        await screen.apply_compatibility_ready(
            stale,
            {"some/model": AvailabilityReport(state=AvailabilityState.UNKNOWN)},
            screen._generation,
        )
        await pilot.pause()
        still = app.state.compatibility_report

    assert still is current


async def test_a_local_batch_before_the_listing_mounts_stores_state_without_raising(
    monkeypatch,
) -> None:
    """A local batch landing before the listing table mounts must not raise.

    Windows CI run 41081401307 (PR #69): the current report's local batch
    reached the screen before ``#suitability-listing`` existed, so the
    refresh crashed on the listing query after both state writes had already
    happened. A bare screen that is never run reproduces that moment
    deterministically — no Pilot, no worker, no message ordering.
    """
    report = _synthetic_report()
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())
    screen = SuitabilityScreen()
    screen._report = report
    screen._listing = list(report.listing)
    # A bare screen has no app of its own; hand it the real (unrunning) app so
    # the hook's state writes land in a real WizardState.
    monkeypatch.setattr(SuitabilityScreen, "_preflight_app", property(lambda self: app))

    assert not screen.query("#suitability-listing")
    cached = AvailabilityReport(state=AvailabilityState.CACHED)
    await screen.apply_compatibility_ready(
        report, {report.listing[0].model: cached}, screen._generation
    )

    # The batch was stored in both places — the exact object, not a copy.
    assert app.state.availability_by_entry[report.listing[0].model] is cached
    assert screen._local_by_model[report.listing[0].model] is cached


async def test_a_check_with_no_listing_clears_the_stale_selection(monkeypatch) -> None:
    """WR-07: "Use this model" can never point at a row that left the screen."""
    first = _synthetic_report()
    empty = CompatibilityReport(verdicts=[], listing=[])
    reports = [first, empty]

    class _SequenceCheck:
        calls = 0

        def __call__(self, *_args: object, **_kwargs: object) -> CompatibilityReport:
            report = reports[min(self.calls, len(reports) - 1)]
            self.calls += 1
            return report

    monkeypatch.setattr(preflight_facade, "check_compatibility", _SequenceCheck())
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        screen.query_one("#suitability-listing", DataTable).move_cursor(row=1)
        await pilot.pause()
        selected_before = app.state.selected_entry
        enabled_before = not screen.query_one("#suitability-use", Button).disabled
        # A check that returns no listing (explicit-model mode) replaces the
        # rows, so the previous choice must not survive.
        await pilot.click("#suitability-check")
        await _settle(pilot, app)
        selected_after = app.state.selected_entry
        disabled_after = screen.query_one("#suitability-use", Button).disabled
        detail = str(screen.query_one("#suitability-detail", Static).content)

    assert selected_before is first.listing[1]
    assert enabled_before
    assert selected_after is None
    assert disabled_after
    assert detail == ""


async def test_a_failed_suitability_check_reads_plainly(monkeypatch) -> None:
    """A raising compatibility call leaves a readable row, never a traceback."""

    def boom(*_args: object, **_kwargs: object) -> CompatibilityReport:
        raise RuntimeError("boom")

    monkeypatch.setattr(preflight_facade, "check_compatibility", boom)
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        status = str(screen.query_one("#suitability-status", Static).content)

    assert status == copy.SUITABILITY_FAILED


async def test_the_row_being_probed_says_so_while_it_is_in_flight(monkeypatch) -> None:
    """A row marked probing shows the probing sentence even on a refresh."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        listing = screen.query_one("#suitability-listing", DataTable)
        listing.cursor_type = "row"
        listing.move_cursor(row=2)
        await pilot.pause()
        screen.action_probe_row()
        # Refresh while the probe is still in flight: the row must stay honest.
        screen._refresh_availability_cells()
        cell = str(listing.get_row_at(2)[4])
        await _settle(pilot, app)

    assert cell == copy.PROBING_PLACEHOLDER


async def test_an_update_for_an_unlisted_model_is_still_recorded(monkeypatch) -> None:
    """A probe result arriving after the listing changed is kept, not applied."""
    monkeypatch.setattr(
        preflight_facade, "check_compatibility", FakeCompatibilityCore(_synthetic_report())
    )
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())
    late = AvailabilityReport(state=AvailabilityState.REMOTE_CONFIRMED, source_hub=Hub.HUGGINGFACE)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        await screen.apply_availability_update("not/listed-any-more", late)
        await pilot.pause()
        recorded = app.state.availability_by_entry["not/listed-any-more"]
        status = str(screen.query_one("#suitability-status", Static).content)

    assert recorded is late
    assert status == copy.SUITABILITY_READY


async def test_probing_or_confirming_with_no_row_explains_what_to_do(monkeypatch) -> None:
    """With an empty listing both actions say how to pick a model."""
    empty = CompatibilityReport(verdicts=[], listing=[])
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(empty))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        screen.action_probe_row()
        probe_status = str(screen.query_one("#suitability-status", Static).content)
        assert screen._entry_at(5) is None
        screen._confirm_selection()
        confirm_status = str(screen.query_one("#suitability-status", Static).content)

    assert probe_status == copy.SELECT_MODEL_HINT
    assert confirm_status == copy.SELECT_MODEL_HINT


async def test_the_availability_refresh_stops_at_the_table_edge(monkeypatch) -> None:
    """A listing longer than the rendered table refreshes only the rows it has."""
    report = _synthetic_report()
    monkeypatch.setattr(preflight_facade, "check_compatibility", FakeCompatibilityCore(report))
    monkeypatch.setattr(preflight_facade, "local_availability", _fake_local())
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        await _settle(pilot, app)
        listing = screen.query_one("#suitability-listing", DataTable)
        screen._listing.append(report.listing[0])  # a row the table never received
        screen._refresh_availability_cells()
        rows = listing.row_count

    assert rows == len(report.listing)


def test_a_cached_report_without_a_hub_says_it_is_on_this_computer() -> None:
    """The local state is named plainly whether the hub is absent or LOCAL."""
    from dnallm.tui.screens.suitability import _availability_sentence

    assert (
        _availability_sentence(AvailabilityReport(state=AvailabilityState.CACHED))
        == copy.AVAILABILITY_CACHED_LOCAL
    )
    assert (
        _availability_sentence(
            AvailabilityReport(state=AvailabilityState.CACHED, source_hub=Hub.LOCAL)
        )
        == copy.AVAILABILITY_CACHED_LOCAL
    )


# ── Config panel branches ───────────────────────────────────────────────────


def test_the_field_parsers_are_total() -> None:
    """An unparseable number stays text so the core can report it in its own words."""
    from dnallm.tui.screens.config import _parse, _unchanged

    assert _parse("int", "7") == 7
    assert _parse("int", "seven") == "seven"
    assert _parse("float", "0.5") == 0.5
    assert _parse("float", "half") == "half"
    assert _parse("labels", "query, value") == ["query", "value"]
    assert _parse("text", " x ") == " x "
    # An absent value and an empty field are the same thing: no patch at all.
    assert _unchanged("text", "", None) is True
    assert _unchanged("int", 3, 3) is True
    assert _unchanged("float", 0.5, 0.5) is True
    assert _unchanged("float", 1, 1.0) is True
    assert _unchanged("int", 4, 3) is False


async def test_the_panel_resumes_the_wizards_working_mapping(monkeypatch) -> None:
    """Re-entering the panel revalidates the mapping already on the wizard state."""
    generated = edit_config({"task": {"task_type": "binary"}}, {})
    fake = FakeConfigCore(generated)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())
    app.state.generated_config = generated

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot)
        await _settle_config(pilot, app)
        status = str(screen.query_one("#config-status", Static).content)

    assert fake.edit_calls[0]["existing"] is generated.raw
    assert status == copy.CONFIG_READY


async def test_the_panel_falls_back_to_the_config_flag_file(monkeypatch, tmp_path) -> None:
    """With nothing in the wizard, the --config path is what the core is handed."""
    config_file = tmp_path / "existing.yaml"
    config_file.write_text("task:\n  task_type: binary\n", encoding="utf-8")
    fake = FakeConfigCore(edit_config(config_file, {}))
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector(), config_path=str(config_file))

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot)
        await _settle_config(pilot, app)
        status = str(screen.query_one("#config-status", Static).content)

    assert fake.edit_calls[0]["existing"] == str(config_file)
    assert status == copy.CONFIG_READY


async def test_the_panel_with_nothing_to_configure_says_so(monkeypatch) -> None:
    """No model, no file, no mapping: the panel explains the way back."""
    fake = FakeConfigCore(GeneratedConfig(config={}, errors=[], raw={}))
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot)
        await pilot.pause()
        status = str(screen.query_one("#config-status", Static).content)

    assert status == copy.CONFIG_NO_SOURCE
    assert fake.generate_calls == []
    assert fake.edit_calls == []


async def test_saving_with_a_blank_name_prompts_instead_of_writing(monkeypatch) -> None:
    """A blank save name is a prompt, never a write to an empty path."""
    generated = edit_config({"task": {"task_type": "binary"}}, {})
    fake = FakeConfigCore(generated)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        writes_before = [call for call in fake.edit_calls if call["output_path"] is not None]
        screen.query_one("#config-output-path", Input).value = "   "
        await pilot.click("#config-save")
        await _settle_config(pilot, app)
        status = str(screen.query_one("#config-status", Static).content)
        export_path = app.state.export_path

    assert writes_before == []
    assert status == copy.CONFIG_OUTPUT_PATH_PROMPT
    assert export_path is None


async def test_editing_a_section_that_is_not_a_mapping_still_records(monkeypatch) -> None:
    """A malformed section is replaced by a mapping rather than crashing the form."""
    generated = edit_config({"task": {"task_type": "binary"}}, {})
    fake = FakeConfigCore(generated)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        screen._raw["lora"] = "not-a-mapping"
        screen._record("lora", "r", 8)
        recorded = screen._raw["lora"]

    assert recorded == {"r": 8}


async def test_a_field_edit_hands_the_worker_a_snapshot_of_the_mapping(monkeypatch) -> None:
    """WR-01: the live working mapping is copied before it leaves this thread.

    The worker deep-copies its input on its own thread, while the UI thread
    keeps writing into the same nested dicts on the next keystroke. Adding a
    key mid-copy raises "dictionary changed size during iteration", so the
    snapshot has to be taken here, where the mapping is stable for the call.
    """
    generated = edit_config(
        {"task": {"task_type": "binary"}, "finetune": {"learning_rate": 1e-05}}, {}
    )
    fake = FakeConfigCore(generated)
    _install_config_core(monkeypatch, fake)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        live = screen._raw
        dispatched: list[tuple[Any, Any, dict]] = []
        monkeypatch.setattr(
            app,
            "edit_config_worker",
            lambda existing, patch=None, **kwargs: dispatched.append((existing, patch, kwargs)),
        )

        screen.query_one("#config-field-finetune-num_train_epochs", Input).value = "7"
        await pilot.pause()
        edit_existing, edit_patch, _ = dispatched[0]
        # The value at dispatch time is in the copy, and the copy is not live.
        assert edit_patch == {"finetune": {"num_train_epochs": 7}}
        snapshot_at_dispatch = edit_existing == live
        live["finetune"]["added_while_the_worker_runs"] = True
        copy_saw_the_late_key = "added_while_the_worker_runs" in edit_existing["finetune"]

        dispatched.clear()
        screen.query_one("#config-output-path", Input).value = "out.yaml"
        await pilot.pause()
        dispatched.clear()
        screen._save()
        await pilot.pause()
        save_existing = dispatched[0][0]

    assert snapshot_at_dispatch
    assert edit_existing is not live
    assert save_existing is not live
    assert copy_saw_the_late_key is False


async def test_a_failed_config_build_reads_plainly(monkeypatch) -> None:
    """A raising generate_config leaves one readable row on the panel."""

    def boom(*_args: object, **_kwargs: object) -> GeneratedConfig:
        raise RuntimeError("boom")

    monkeypatch.setattr(preflight_facade, "generate_config", boom)
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_FINETUNED_REPO)
        await _settle_config(pilot, app)
        status = str(screen.query_one("#config-status", Static).content)

    assert "could not build this config" in status
    assert "RuntimeError" not in status


# ── Download panel branches ─────────────────────────────────────────────────


def test_format_size_uses_bytes_below_a_kilobyte() -> None:
    """Small repos are quoted in whole bytes, larger ones in decimal units."""
    from dnallm.tui.screens.download import _format_size

    assert _format_size(999) == "999 bytes"
    assert _format_size(1000) == "1.0 KB"


def test_format_size_falls_back_when_the_unit_table_is_empty(monkeypatch) -> None:
    """The defensive fallback says the size is unknown rather than raising."""
    import dnallm.tui.screens.download as download_module

    monkeypatch.setattr(download_module, "_SIZE_UNITS", ())

    assert download_module._format_size(10**12) == copy.DOWNLOAD_SIZE_UNKNOWN


async def test_progress_is_clamped_and_late_messages_are_ignored() -> None:
    """Inside one measurement the bar never runs backwards; after the end nothing moves."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    app.state.selected_entry = _download_entry()
    app.state.export_path = CONFIG_EXPORT_PATH

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        await screen.apply_download_progress(done=10, total=100, label="first")
        bar = screen.query_one("#download-bar", ProgressBar)
        shown_after_first = bar.progress
        await screen.apply_download_progress(done=4, total=100, label="still going")
        shown_after_second = bar.progress
        await screen.apply_download_progress(done=1, total=0, label="unknown total")
        await screen.apply_download_finished(DOWNLOAD_RETURNED_PATH, False, None)
        await pilot.pause()
        summary_once = isinstance(app.screen, SummaryScreen)
        # Both late arrivals must be no-ops rather than a second summary.
        await screen.apply_download_progress(done=1, total=100, label="late")
        await screen.apply_download_finished("/models/again", False, None)
        await pilot.pause()
        still_summary = isinstance(app.screen, SummaryScreen)

    assert shown_after_first == 10
    assert shown_after_second == 10  # clamped, never 4
    assert summary_once
    assert still_summary


async def test_a_failed_seam_call_on_the_download_page_offers_a_retry() -> None:
    """A raising worker leaves the retry action on screen and says what happened."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    app.state.selected_entry = _download_entry()
    app.state.export_path = CONFIG_EXPORT_PATH

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        await screen.apply_worker_failure("Something went wrong.")
        retry_visible = screen.query_one("#download-retry", Button).display
        status = str(screen.query_one("#download-status", Static).content)

    assert retry_visible is True
    assert status == "Something went wrong."


async def test_leaving_the_download_page_goes_back_but_never_mid_download() -> None:
    """Escape returns to the previous step while idle, and is inert while running."""
    gate = threading.Event()
    downloader = FakeDownloader(gate=gate)
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()
    app.state.export_path = CONFIG_EXPORT_PATH

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        screen.action_go_back()
        await pilot.pause()
        left_idle_page = not isinstance(app.screen, DownloadScreen)

        running = await _open_download_screen(app, pilot)
        await pilot.click("#download-start")
        await _wait_until(lambda: bool(downloader.calls))
        running.action_go_back()
        await pilot.pause()
        stayed = app.screen is running
        gate.set()
        await _wait_for_the_download(pilot, app)

    assert left_idle_page
    assert stayed


async def test_stopping_or_ticking_before_a_download_starts_is_a_no_op() -> None:
    """The idle page ignores stop and the elapsed tick, and keeps saying so."""
    app = DNALLMPreflightApp(collector=FakeCollector())
    app.state.selected_entry = _download_entry()

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_download_screen(app, pilot)
        screen._cancel()
        screen._tick()
        screen._start_download()
        screen._start_download()  # a second start while the first is running
        status = str(screen.query_one("#download-status", Static).content)
        await _wait_for_the_download(pilot, app)

    assert status == copy.DOWNLOAD_PROGRESS.format(model=DOWNLOAD_MODEL_REPO)


async def test_finishing_without_the_files_mid_download_stops_it_first(monkeypatch) -> None:
    """Skip while running asks for a stop, then describes what really happened.

    WR-03: the old code jumped straight to the summary, which then claimed no
    model files were downloaded while the attempt was still writing into the
    hub cache. Now the summary waits for the worker's outcome and uses the
    "you stopped it, files may be incomplete" sentence.
    """
    from types import SimpleNamespace

    from dnallm.models import model as model_module

    # Keep the retry loop's one-second sleep out of the test's wall time.
    monkeypatch.setattr(model_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    gate = threading.Event()
    downloader = FakeDownloader(gate=gate, failure=ConnectionError("connection reset"))
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()
    app.state.export_path = CONFIG_EXPORT_PATH

    try:
        async with app.run_test(size=(100, 40)) as pilot:
            screen = await _open_download_screen(app, pilot)
            await pilot.click("#download-start")
            await _wait_until(lambda: bool(downloader.calls))
            await pilot.click("#download-skip")
            await pilot.pause()
            # Still on the download panel: the stop is pending, not finished.
            stayed = app.screen is screen
            pending = str(screen.query_one("#download-status", Static).content)
            gate.set()
            await _wait_for_the_download(pilot, app)
            summary = app.screen
            text = _rendered_text(summary)
    finally:
        gate.set()

    assert stayed, "skip mid-download must not reach the summary before the stop"
    assert pending == copy.CANCEL_PENDING_NOTE
    assert isinstance(summary, SummaryScreen)
    assert copy.SUMMARY_MODEL_STOPPED in text
    assert copy.SUMMARY_MODEL_SKIPPED not in text
    assert downloader.calls == [DOWNLOAD_MODEL_REPO]


async def test_the_quit_action_leaves_the_wizard() -> None:
    """The closing page's quit button exits the app."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(SummaryScreen(model_path="/models/x", config_path="out.yaml"))
        await pilot.pause()
        await pilot.click("#summary-quit")
        await pilot.pause()
        exited = bool(app._exit)

    assert exited


# ─────────────────────────────────────────────────────────────────────────────
# Copy inventory (D-10): every sentence a screen shows comes from copy.py
# ─────────────────────────────────────────────────────────────────────────────


_WIDGET_TEXT_CLASSES = frozenset({"Static", "Button", "Label"})
_WIDGET_TEXT_KEYWORDS = frozenset({"placeholder", "label"})


def _literal_widget_text(tree: ast.AST) -> list[str]:
    """Non-empty literal strings handed straight to a text-showing widget."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node.func) not in _WIDGET_TEXT_CLASSES:
            continue
        candidates = list(node.args[:1]) + [
            keyword.value for keyword in node.keywords if keyword.arg in _WIDGET_TEXT_KEYWORDS
        ]
        for candidate in candidates:
            if (
                isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
                and candidate.value.strip()
            ):
                found.append(candidate.value)
    return found


def test_no_screen_builds_a_widget_from_a_literal_sentence() -> None:
    """D-10: user-facing words live in dnallm/tui/copy.py, not beside the layout.

    Empty strings are allowed — a status line has to start somewhere — and so
    are the values the core produced (a model name, a path, a report message):
    those are not sentences this project writes.
    """
    offenders: dict[str, list[str]] = {}
    for module_name, path in _screen_module_paths().items():
        literals = _literal_widget_text(ast.parse(path.read_text(encoding="utf-8")))
        if literals:
            offenders[module_name] = literals
    assert offenders == {}, (
        f"screen widgets are built from literal sentences: {offenders} — move "
        f"them into dnallm/tui/copy.py so the phase-end plain-language review "
        f"is a single-file read (D-10)"
    )


def test_every_copy_constant_the_tui_uses_is_declared_in_copy_all() -> None:
    """The copy inventory is complete: a used sentence is always an exported one."""
    from dnallm.tui import copy as copy_module

    exported = set(copy_module.__all__)
    used: dict[str, set[str]] = {}
    for path in sorted((PACKAGE_ROOT / "tui").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "copy"
        }
        if names:
            used[path.name] = names
    referenced = set().union(*used.values())
    undeclared = sorted(referenced - exported)
    assert undeclared == [], (
        f"copy constants used but not declared in dnallm/tui/copy.py __all__: "
        f"{undeclared} (used by {used}) — the inventory is what the human copy "
        f"review reads (D-10)"
    )


def test_the_unverified_hosting_sentence_promises_only_one_hub() -> None:
    """D-10 honesty: the download picks one hub and retries it, never a second.

    The earlier wording said "ModelScope first, then Hugging Face", which the
    seam does not do (``workers._choose_hub`` picks one hub; a retry stays on
    it). This pins the corrected sentence so the claim cannot drift back.
    """
    assert "then Hugging Face" not in copy.UNKNOWN_STILL_DOWNLOADABLE
    assert "try again if that does not work" in copy.UNKNOWN_STILL_DOWNLOADABLE


#: Engine words that must never reach a user-facing sentence (D-10).
_BANNED_COPY_TOKENS = (
    "Exception",
    "Traceback",
    "ValueError",
    "RuntimeError",
    "KeyError",
    "kwargs",
    "stderr",
)


def _copy_strings(value: object) -> list[str]:
    """Every string inside one copy constant (a str, a tuple, or a dict's values)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _copy_strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _copy_strings(item)]
    return []


def test_copy_never_leaks_engine_words() -> None:
    """No exception class name, no traceback word, no keyword name reaches a user."""
    from dnallm.tui import copy as copy_module

    offenders: dict[str, list[str]] = {}
    for name in copy_module.__all__:
        for text in _copy_strings(getattr(copy_module, name)):
            hits = [token for token in _BANNED_COPY_TOKENS if token in text]
            if hits:
                offenders[name] = hits
    assert offenders == {}, (
        f"copy constants carry engine words: {offenders} — every sentence is "
        f"written for a biologist and exception details stay in the log (D-10)"
    )
