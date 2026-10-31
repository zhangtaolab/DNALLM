"""The data panel (TUI-04): what is in the user's data file.

Presentation only. The panel hands the path the user pasted to
``validate_data`` through the app's thread-worker seam and renders the frozen
:class:`~dnallm.preflight.report.DataReport` that comes back — split row
counts, length statistics, the character census, and every finding with its
severity, reason code, and row count. It never opens, parses, or counts the
file itself, and it never sorts, filters, or groups a finding: the core's order
is the order on screen.

Reason codes and counts are shown verbatim (TUI-04) beside a plain-language
explanation; every other sentence comes from :mod:`dnallm.tui.copy` and is
written for a biologist (D-10). File paths and report text are rendered
markup-free, so a ``[`` in a path cannot corrupt the page (T-25-10).

Example:
    app.push_screen(DataScreen())  # or the env panel's Continue action
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from .. import copy

if TYPE_CHECKING:
    from ...preflight import DataReport, Finding, SplitReport
    from ..app import DNALLMPreflightApp

__all__ = ["DataScreen"]

_PATH_INPUT_ID = "data-path"
_CHECK_BUTTON_ID = "data-check"
_SKIP_BUTTON_ID = "data-skip"
_CONTINUE_BUTTON_ID = "data-continue"
_FINDINGS_TABLE_ID = "data-findings"
_SPLITS_CONTAINER_ID = "data-splits"
_CHARSET_CONTAINER_ID = "data-charset"

# The areas that only make sense once a report exists (D-08: with no data the
# page shows the unchanged no-data state instead of empty tables).
_REPORT_AREA_SELECTORS = (
    "#data-summary-heading",
    f"#{_SPLITS_CONTAINER_ID}",
    "#data-charset-heading",
    f"#{_CHARSET_CONTAINER_ID}",
    "#data-findings-heading",
    f"#{_FINDINGS_TABLE_ID}",
    "#data-no-findings",
)


class DataScreen(Screen):
    """Render ``validate_data``'s report; never derive one."""

    @property
    def _preflight_app(self) -> DNALLMPreflightApp:
        """The typed app, for the worker seam and the markup-safe renderer."""
        return cast("DNALLMPreflightApp", self.app)

    def compose(self) -> ComposeResult:
        """Lay out the path input, the actions, and the report areas."""
        yield Header()
        with VerticalScroll(id="data-body"):
            yield Static(copy.DATA_PANEL_TITLE, id="data-title")
            yield Input(placeholder=copy.DATA_PATH_PROMPT, id=_PATH_INPUT_ID)
            with Horizontal(id="data-actions"):
                yield Button(copy.DATA_CHECK_ACTION, id=_CHECK_BUTTON_ID, variant="primary")
                yield Button(copy.DATA_SKIP_ACTION, id=_SKIP_BUTTON_ID)
                yield Button(copy.DATA_CONTINUE_ACTION, id=_CONTINUE_BUTTON_ID)
            yield Static("", markup=False, id="data-status")
            yield Static(copy.DATA_SUMMARY_HEADING, id="data-summary-heading")
            yield Container(id=_SPLITS_CONTAINER_ID)
            yield Static(copy.DATA_CHARSET_HEADING, id="data-charset-heading")
            yield Container(id=_CHARSET_CONTAINER_ID)
            yield Static(copy.DATA_FINDINGS_HEADING, id="data-findings-heading")
            yield DataTable(id=_FINDINGS_TABLE_ID)
            yield Static("", markup=False, id="data-no-findings")
        yield Static(copy.QUIT_HINT, id="data-quit-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Declare the findings columns and show the state already reached.

        Re-entering the panel after a skip shows the same plain no-data state
        rather than an empty table or an error; re-entering after a check
        re-renders the stored report without re-reading the file.
        """
        table = self.query_one(f"#{_FINDINGS_TABLE_ID}", DataTable)
        for label in copy.DATA_FINDINGS_COLUMNS:
            table.add_column(label)
        data_report = self._preflight_app.state.data_report
        if data_report is None:
            self._show_no_data_state()
            return
        self._set_report_areas_visible(True)
        self._set_status(copy.DATA_READY)
        self._render_report(data_report)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Start reading the pasted path, or skip the step entirely.

        Args:
            event: The button press; the check, skip and continue actions are
                ours.
        """
        if event.button.id == _CHECK_BUTTON_ID:
            self._check_data()
        elif event.button.id == _SKIP_BUTTON_ID:
            self._skip_data()
        elif event.button.id == _CONTINUE_BUTTON_ID:
            self._preflight_app.show_suitability()

    def _check_data(self) -> None:
        """Hand the pasted path to the core through the app's thread worker.

        An empty path never reaches the core: it is a prompt, not a read.
        """
        path = self.query_one(f"#{_PATH_INPUT_ID}", Input).value.strip()
        if not path:
            self._set_status(copy.EMPTY_PATH_PROMPT)
            return
        self._set_status(copy.CHECKING_PLACEHOLDER)
        self._preflight_app.validate_data_worker(path)

    def _skip_data(self) -> None:
        """Leave the step without data and move on (D-08).

        ``WizardState.data_report`` stays None, so the suitability step judges
        models from the model and the task alone; no core call is made.
        """
        self._preflight_app.state.data_report = None
        self.query_one(f"#{_PATH_INPUT_ID}", Input).value = ""
        self._show_no_data_state()
        self._preflight_app.show_suitability()

    def _show_no_data_state(self) -> None:
        """State plainly that there is no data and what that means downstream."""
        self._set_status(f"{copy.NO_DATA_PROVIDED}\n{copy.DATA_SKIP_NOTE}")
        self._set_report_areas_visible(False)

    def _set_report_areas_visible(self, visible: bool) -> None:
        """Show or hide the areas that need a report to make sense."""
        for selector in _REPORT_AREA_SELECTORS:
            self.query_one(selector).display = visible

    async def apply_data_report(self, data_report: DataReport) -> None:
        """Store the frozen report and render it exactly as the core returned it.

        Args:
            data_report: The report ``validate_data`` produced inside the
                worker. Every number shown on this page comes from it.
        """
        self._preflight_app.state.data_report = data_report
        await self._clear_report()
        self._set_report_areas_visible(True)
        self._set_status(copy.DATA_READY)
        self._render_report(data_report)

    async def apply_worker_failure(self, message: str) -> None:
        """Show a failed read as one plain-language row.

        Args:
            message: The already-plain-language sentence the seam produced —
                never an exception class name.
        """
        self._set_status(message)

    async def _clear_report(self) -> None:
        """Empty the split, character, and findings areas before re-rendering."""
        for container_id in (_SPLITS_CONTAINER_ID, _CHARSET_CONTAINER_ID):
            await self.query_one(f"#{container_id}", Container).remove_children()
        self.query_one(f"#{_FINDINGS_TABLE_ID}", DataTable).clear()

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one("#data-status", Static).update(text)

    def _render_report(self, data_report: DataReport) -> None:
        """Paint the splits, the character census, and every finding in order.

        Args:
            data_report: The frozen report to render. Nothing here computes,
                reorders, or filters — each row is one report field.
        """
        splits = self.query_one(f"#{_SPLITS_CONTAINER_ID}", Container)
        charset = self.query_one(f"#{_CHARSET_CONTAINER_ID}", Container)
        table = self.query_one(f"#{_FINDINGS_TABLE_ID}", DataTable)

        for split in data_report.splits:
            splits.mount(self._preflight_app.render_text(_split_line(split)))
            for entry in split.charset_census:
                charset.mount(
                    self._preflight_app.render_text(
                        copy.DATA_CHARSET_ROW.format(
                            split=split.name, char=entry.char, count=entry.count
                        )
                    )
                )

        findings = _findings_in_order(data_report)
        for finding in findings:
            table.add_row(
                _cell(finding.severity.value),
                _cell(finding.reason_code),
                _cell(finding.message),
                _cell(
                    copy.DATA_COUNT_NOT_APPLICABLE if finding.count is None else str(finding.count)
                ),
            )
        # A report with nothing to say gets an explicit all-clear row instead of
        # an empty table the user would have to interpret.
        no_findings = self.query_one("#data-no-findings", Static)
        no_findings.update("" if findings else copy.NO_FINDINGS)
        no_findings.display = not findings


def _findings_in_order(data_report: DataReport) -> list[Finding]:
    """Every finding the core returned, in the order it returned them.

    The read-level findings come first (they describe the whole file), then
    each split's findings in split order. This is a flattening of two ordered
    lists — never a sort, a filter, or a re-grouping.

    Args:
        data_report: The frozen report whose findings are being listed.

    Returns:
        The findings in core order.
    """
    findings: list[Finding] = list(data_report.findings)
    for split in data_report.splits:
        findings.extend(split.findings)
    return findings


def _split_line(split: SplitReport) -> str:
    """Describe one split's row count and length range from the report alone.

    Args:
        split: The split census to describe.

    Returns:
        One plain-language line; an empty split says so instead of showing
        zeros the core never measured.
    """
    stats = split.length_stats
    if stats is None:
        return copy.DATA_SPLIT_ROW_EMPTY.format(name=split.name)
    return copy.DATA_SPLIT_ROW.format(
        name=split.name,
        row_count=split.row_count,
        shortest=stats.min,
        typical=stats.p50,
        longest=stats.max,
    )


def _cell(value: str) -> Text:
    """Wrap a table cell so its text is displayed literally.

    ``DataTable`` parses a plain ``str`` cell as Rich markup; a report message
    or a path containing ``[`` would then either vanish or raise. A
    :class:`~rich.text.Text` cell is never markup-parsed (T-25-10).

    Args:
        value: The exact text to display.

    Returns:
        The text as a markup-free Rich cell.
    """
    return Text(str(value), end="")
