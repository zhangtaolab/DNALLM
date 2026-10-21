"""The suitability panel (TUI-05): which model can do this job.

Presentation only. The panel hands the app's stored data report, the chosen
route and the optional task type to ``check_compatibility`` through the app's
thread-worker seam and renders the frozen
:class:`~dnallm.preflight.report.CompatibilityReport` that comes back: the
per-model verdicts for models the user asked about, and the intent-routed
listing.

Three rules the renderer exists to keep:

1. **The core's order is the order on screen.** The listing is rendered in
   exactly the sequence ``check_compatibility`` returned — lab-first ordering
   belongs to the core (PFC-07), so the screen never sorts or filters a row.
   Task and intent selection are arguments to the core, never a list
   comprehension here.
2. **Every fact is shown honestly (D-05/D-06).** An availability state the
   core could not verify reads "hosting not verified"; the hub is named only
   when the core reported one; no row is hidden and no row is relabelled. An
   unknown state never disables the row: UNKNOWN is not unavailable.
3. **The intent vocabulary is the core's.** :data:`INTENT_FINETUNE` and
   :data:`INTENT_PREDICT` mirror ``dnallm.preflight.compatibility`` exactly and
   are what the worker receives; the sentences a user reads ("Fine-tune a base
   model on my data") come from :mod:`dnallm.tui.copy` and are never sent to
   the core.

Choosing a row stores the frozen :class:`~dnallm.preflight.report.ListingEntry`
in :attr:`dnallm.tui.state.WizardState.selected_entry`. The "Use this model"
control is built here; its destination (the config panel) is wired by 25-08,
which owns that transition.

Example:
    app.push_screen(SuitabilityScreen())  # or the data panel's Continue action
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from .. import copy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...preflight import AvailabilityReport, CompatibilityReport, ListingEntry
    from ..app import DNALLMPreflightApp

__all__ = ["INTENT_FINETUNE", "INTENT_PREDICT", "SuitabilityScreen"]

# The core's ONLY accepted intent vocabulary (dnallm/preflight/compatibility.py).
# Pinned here as constants so the UI's words can never drift into the core call.
INTENT_FINETUNE = "finetune"
INTENT_PREDICT = "predict"

_TITLE_ID = "suitability-title"
_FINETUNE_BUTTON_ID = "suitability-intent-finetune"
_PREDICT_BUTTON_ID = "suitability-intent-predict"
_TASK_INPUT_ID = "suitability-task-type"
_MODELS_INPUT_ID = "suitability-models"
_CHECK_BUTTON_ID = "suitability-check"
_STATUS_ID = "suitability-status"
_VERDICTS_TABLE_ID = "suitability-verdicts"
_LISTING_TABLE_ID = "suitability-listing"
_USE_BUTTON_ID = "suitability-use"

# Core values the renderer branches on. ``AvailabilityState`` and ``Hub`` are
# string enums, so comparing against the value string needs no runtime import
# of the core — the screen stays presentation-only, and the type imports above
# are for typing only.
_STATE_CACHED = "cached"
_STATE_REMOTE_CONFIRMED = "remote_confirmed"
_HUB_LOCAL = "local"


class SuitabilityScreen(Screen):
    """Render ``check_compatibility``'s report; never compute a verdict."""

    def __init__(self) -> None:
        """Start on the fine-tune route (base models) with nothing selected."""
        super().__init__()
        self._intent = INTENT_FINETUNE
        self._report: CompatibilityReport | None = None
        self._listing: list[ListingEntry] = []

    @property
    def _preflight_app(self) -> DNALLMPreflightApp:
        """The typed app, for the worker seam and the markup-safe renderer."""
        return cast("DNALLMPreflightApp", self.app)

    def compose(self) -> ComposeResult:
        """Lay out the route controls, the verdict table, and the listing."""
        yield Header()
        with VerticalScroll(id="suitability-body"):
            yield Static(copy.SUITABILITY_PANEL_TITLE, id=_TITLE_ID)
            with Horizontal(id="suitability-intent"):
                yield Button(
                    copy.SUITABILITY_INTENT_FINETUNE,
                    id=_FINETUNE_BUTTON_ID,
                    variant="primary",
                )
                yield Button(copy.SUITABILITY_INTENT_PREDICT, id=_PREDICT_BUTTON_ID)
            yield Input(placeholder=copy.SUITABILITY_TASK_PROMPT, id=_TASK_INPUT_ID)
            yield Input(placeholder=copy.SUITABILITY_MODELS_PROMPT, id=_MODELS_INPUT_ID)
            yield Button(copy.SUITABILITY_CHECK_ACTION, id=_CHECK_BUTTON_ID)
            yield Static("", markup=False, id=_STATUS_ID)
            yield Static(copy.SUITABILITY_VERDICTS_HEADING, id="suitability-verdicts-heading")
            yield DataTable(id=_VERDICTS_TABLE_ID)
            yield Static(copy.SUITABILITY_LISTING_HEADING, id="suitability-listing-heading")
            yield DataTable(id=_LISTING_TABLE_ID)
            yield Button(copy.USE_MODEL_ACTION, id=_USE_BUTTON_ID, disabled=True)
        yield Static(copy.QUIT_HINT, id="suitability-quit-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Declare the columns, show the probing state, and start the worker."""
        verdicts = self.query_one(f"#{_VERDICTS_TABLE_ID}", DataTable)
        for label in copy.SUITABILITY_VERDICT_COLUMNS:
            verdicts.add_column(label)
        listing = self.query_one(f"#{_LISTING_TABLE_ID}", DataTable)
        for label in copy.SUITABILITY_LISTING_COLUMNS:
            listing.add_column(label)
        # Rows are chosen whole, not cell by cell (the selection is a model).
        listing.cursor_type = "row"
        self._start_compatibility()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the intent controls, the check action, and the selection action.

        Args:
            event: The button press; only this panel's buttons are handled.
        """
        if event.button.id == _FINETUNE_BUTTON_ID:
            self._choose_intent(INTENT_FINETUNE, _FINETUNE_BUTTON_ID, _PREDICT_BUTTON_ID)
        elif event.button.id == _PREDICT_BUTTON_ID:
            self._choose_intent(INTENT_PREDICT, _PREDICT_BUTTON_ID, _FINETUNE_BUTTON_ID)
        elif event.button.id == _CHECK_BUTTON_ID:
            self._start_compatibility()
        elif event.button.id == _USE_BUTTON_ID:
            self._confirm_selection()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Record the focused listing row as the user's current choice.

        Args:
            event: The highlight event; only the listing table selects models.
        """
        if event.data_table.id != _LISTING_TABLE_ID:
            return
        entry = self._entry_at(event.cursor_row)
        if entry is not None:
            self._select(entry)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Treat Enter on a listing row as choosing that model.

        Args:
            event: The row-selected event; only the listing table selects models.
        """
        if event.data_table.id != _LISTING_TABLE_ID:
            return
        entry = self._entry_at(event.cursor_row)
        if entry is not None:
            self._select(entry)

    def _choose_intent(self, intent: str, active_id: str, other_id: str) -> None:
        """Switch the route and ask the core again with the new intent.

        Args:
            intent: The core's vocabulary value for the chosen route.
            active_id: The button that becomes primary (the chosen route).
            other_id: The button that returns to the default style.
        """
        self._intent = intent
        self.query_one(f"#{active_id}", Button).variant = "primary"
        self.query_one(f"#{other_id}", Button).variant = "default"
        self._start_compatibility()

    def _start_compatibility(self) -> None:
        """Hand the current route, task type and model names to the core."""
        self._set_status(copy.PROBING_PLACEHOLDER)
        self._preflight_app.check_compatibility_worker(
            self._preflight_app.state.data_report,
            task_type=self._task_type(),
            models=self._requested_models(),
            intent=self._intent,
            prober=self._preflight_app.prober,
        )

    def _task_type(self) -> str | None:
        """The task type the user typed, or None when the field is blank."""
        value = self.query_one(f"#{_TASK_INPUT_ID}", Input).value.strip()
        return value or None

    def _requested_models(self) -> list[str] | None:
        """The model names the user listed, or None to ask for a listing."""
        raw = self.query_one(f"#{_MODELS_INPUT_ID}", Input).value
        names = [name.strip() for name in raw.split(",") if name.strip()]
        return names or None

    async def apply_compatibility_ready(
        self, report: CompatibilityReport, local_by_name: Mapping[str, AvailabilityReport]
    ) -> None:
        """Store the frozen report and render it in the core's own order.

        Args:
            report: The report ``check_compatibility`` produced in the worker.
            local_by_name: Availability already known from the local cache,
                keyed by model id (rendering only; never re-probed).
        """
        self._report = report
        self._preflight_app.state.compatibility_report = report
        self._preflight_app.state.availability_by_entry.update(local_by_name)
        self._render_report(report)
        if report.listing or report.verdicts:
            self._set_status(copy.SUITABILITY_READY)

    async def apply_worker_failure(self, message: str) -> None:
        """Show a failed check as one plain-language row.

        Args:
            message: The already-plain-language sentence the seam produced.
        """
        self._set_status(message)

    def _render_report(self, report: CompatibilityReport) -> None:
        """Paint the verdicts and every listing row in the core's order.

        Args:
            report: The frozen report to render. Nothing here reorders,
                filters, or recomputes a row.
        """
        self._listing = list(report.listing)
        verdicts = self.query_one(f"#{_VERDICTS_TABLE_ID}", DataTable)
        verdicts.clear()
        for verdict in report.verdicts:
            for dimension in verdict.dimensions:
                verdicts.add_row(
                    _cell(verdict.entry_name or verdict.model),
                    _cell(dimension.dimension),
                    _cell(dimension.status.value),
                    _cell(dimension.reason_code or ""),
                )

        listing = self.query_one(f"#{_LISTING_TABLE_ID}", DataTable)
        listing.clear()
        for entry in self._listing:
            listing.add_row(*self._listing_cells(entry))

    def _listing_cells(self, entry: ListingEntry) -> list[Text]:
        """One listing row's cells, straight from the entry's fields.

        Args:
            entry: The frozen listing row.

        Returns:
            The cells, markup-free.
        """
        return [
            _cell(entry.entry_name),
            _cell(copy.LAB_ROW_FLAG if entry.is_lab else copy.THIRD_PARTY_ROW_FLAG),
            _cell(entry.task_type or ""),
            _cell(copy.NO_HEAD_LABELS if entry.num_labels is None else str(entry.num_labels)),
            _cell(_availability_text(entry.availability)),
        ]

    def _entry_at(self, position: int) -> ListingEntry | None:
        """The listing entry at *position*, or None when the row is gone.

        Args:
            position: The table's cursor row index.
        """
        if 0 <= position < len(self._listing):
            return self._listing[position]
        return None

    def _select(self, entry: ListingEntry) -> None:
        """Record the chosen row as the wizard's model.

        Args:
            entry: The frozen listing row the user focused.
        """
        self._preflight_app.state.selected_entry = entry
        self.query_one(f"#{_USE_BUTTON_ID}", Button).disabled = False

    def _confirm_selection(self) -> None:
        """Acknowledge the recorded choice; the config step owns the transition."""
        entry = self._preflight_app.state.selected_entry
        if entry is None:
            self._set_status(copy.SELECT_MODEL_HINT)
            return
        self._set_status(copy.USE_MODEL_CONFIRMED.format(model=entry.entry_name))

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one(f"#{_STATUS_ID}", Static).update(text)


def _availability_text(availability: AvailabilityReport | None) -> str:
    """Turn one availability report into the sentence a reader sees.

    Args:
        availability: The core's report for one model, or None when nothing is
            known about that row yet.

    Returns:
        A plain-language sentence. The hub name is shown only when the core
        reported one; everything else reads "hosting not verified" (D-05).
    """
    if availability is None:
        return copy.HOSTING_NOT_VERIFIED
    if availability.state == _STATE_CACHED:
        if availability.source_hub is None or str(availability.source_hub) == _HUB_LOCAL:
            return copy.AVAILABILITY_CACHED_LOCAL
        return copy.AVAILABILITY_CACHED_HUB.format(hub=_hub_name(availability.source_hub))
    if availability.state == _STATE_REMOTE_CONFIRMED and availability.source_hub is not None:
        return copy.AVAILABILITY_REMOTE.format(hub=_hub_name(availability.source_hub))
    return copy.HOSTING_NOT_VERIFIED


def _hub_name(hub: object) -> str:
    """The reader-facing name of one hub value (the value itself when unknown).

    Args:
        hub: A ``Hub`` member (a string enum) or its raw value.
    """
    return str(copy.HUB_NAMES.get(str(hub), str(hub)))


def _cell(value: str) -> Text:
    """Wrap a table cell so its text is displayed literally.

    ``DataTable`` parses a plain ``str`` cell as Rich markup; a model name or a
    probe reason containing ``[`` would then either vanish or raise.

    Args:
        value: The exact text to display.

    Returns:
        The text as a markup-free Rich cell.
    """
    return Text(str(value), end="")
