"""Pilot tests for the data panel (TUI-04, 25-06).

The panel is presentation only, so the tests are table tests against the core:
a synthetic file is validated twice — once directly through ``validate_data``
and once through the running panel — and the rendered rows must equal the
report the core returned. Every seam is injected, so nothing touches the
network or torch; only a small CSV in ``tmp_path`` is read.
"""

from __future__ import annotations

from textual.widgets import DataTable, Input, Static

from dnallm.preflight import validate_data
from dnallm.tui import copy
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.screens.data import DataScreen
from dnallm.tui.screens.env import EnvScreen

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
