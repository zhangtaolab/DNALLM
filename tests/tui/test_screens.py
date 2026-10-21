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

import asyncio
import threading
import time
from typing import Any

import yaml
from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, Input, ProgressBar, Static

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
            # A third-party family the shortlist never reached: no report at
            # all, which is the D-05 case — listed, unverified, choosable.
            ListingEntry(
                model="third-party/file-c",
                entry_name="Third C",
                is_lab=False,
                task_type="mask",
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
    """The core's TASK_TYPE_REQUIRED reaches the pane as one readable line."""
    app = DNALLMPreflightApp(collector=FakeCollector())

    async with app.run_test(size=(140, 50)) as pilot:
        screen = await _open_config_screen(app, pilot, model=CONFIG_BASE_REPO)
        await _settle_config(pilot, app)
        errors = str(screen.query_one("#config-errors", Static).content)
        status = _rendered_text(screen)
        stored = app.state.generated_config

    assert stored is not None
    assert "TASK_TYPE_REQUIRED" in errors
    assert "task_type" in errors
    # The error line carries the core's field path and message, core order intact.
    assert (
        copy.CONFIG_ERROR_LINE.format(
            field=stored.errors[0].field, message=stored.errors[0].message
        )
        in errors
    )
    assert copy.CONFIG_HAS_ERRORS in status


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
        copy.CONFIG_ERROR_LINE.format(field=error.field, message=error.message)
        for error in expected.errors
    ]
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
        active = isinstance(app.screen, ConfigScreen)
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
        await app.workers.wait_for_complete()
        await pilot.pause()
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
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _rendered_text(screen)
        retry_visible = screen.query_one("#download-retry", Button).display
        still_on_download = app.screen is screen
        calls_after_failure = list(downloader.calls)
        await pilot.click("#download-retry")
        await app.workers.wait_for_complete()
        await pilot.pause()
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
        await app.workers.wait_for_complete()
        await pilot.pause()
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
    """Quit never waits for a blocked download thread (D-09)."""
    gate = threading.Event()
    downloader = FakeDownloader(gate=gate)
    app = DNALLMPreflightApp(collector=FakeCollector(), downloader=downloader)
    app.state.selected_entry = _download_entry()
    started_at = time.monotonic()

    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await _open_download_screen(app, pilot)
            await pilot.click("#download-start")
            assert await _wait_until(lambda: bool(downloader.calls))
            await pilot.press("q")
    finally:
        gate.set()

    assert time.monotonic() - started_at < 20


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
