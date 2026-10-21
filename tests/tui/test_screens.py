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

import threading
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, Input, Static

import dnallm.preflight as preflight_facade
import dnallm.preflight.compatibility as compatibility_core
from dnallm.preflight import (
    AvailabilityReport,
    AvailabilityState,
    CompatibilityReport,
    DimensionVerdict,
    Hub,
    ListingEntry,
    ModelVerdict,
    VerdictStatus,
    validate_data,
)
from dnallm.tui import copy
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.screens.data import DataScreen
from dnallm.tui.screens.env import EnvScreen
from dnallm.tui.screens.suitability import INTENT_FINETUNE, INTENT_PREDICT, SuitabilityScreen

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


async def test_env_continue_opens_the_data_panel() -> None:
    """The env panel's Continue action pushes the data panel."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, EnvScreen)
        await pilot.click("#env-continue")
        await pilot.pause()
        assert isinstance(app.screen, DataScreen)


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
    prober = FakeProber()
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=prober)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_suitability_screen(app, pilot)
        # Before the report lands the page says plainly that it is probing.
        probing = copy.PROBING_PLACEHOLDER in _rendered_text(screen)
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rows = _table_rows(screen.query_one("#suitability-listing", DataTable))
        verdict_rows = _table_rows(screen.query_one("#suitability-verdicts", DataTable))
        status = _rendered_text(screen)

    assert probing
    # The lab row stays second: the screen never re-sorts the core's listing.
    assert [row[0] for row in rows] == [entry.entry_name for entry in report.listing]
    assert [row[1] for row in rows] == [copy.THIRD_PARTY_ROW_FLAG, copy.LAB_ROW_FLAG]
    assert rows[1][2] == "binary"
    assert rows[1][3] == "2"
    # Each availability cell names the hub the core reported — never another.
    assert copy.AVAILABILITY_REMOTE.format(hub="Hugging Face") == rows[0][4]
    assert rows[1][4] == copy.HOSTING_NOT_VERIFIED
    # One verdict row per dimension, carrying the core's status and reason.
    assert [row[1] for row in verdict_rows] == ["task", "length"]
    assert [row[2] for row in verdict_rows] == ["ok", "unknown"]
    assert verdict_rows[1][3] == "NO_DATA"
    assert copy.SUITABILITY_READY in status
    # The core got the pinned vocabulary, the app's prober, the 20-row cap,
    # and the wizard's stored data report (None: the data step was skipped).
    assert fake_core.calls[0]["intent"] == "finetune"
    assert fake_core.calls[0]["prober"] is prober
    assert fake_core.calls[0]["probe_shortlist"] == 20
    assert fake_core.calls[0]["data"] is None


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
