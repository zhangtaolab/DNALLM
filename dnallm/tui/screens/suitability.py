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
control is built here and its destination is the config panel: activating it
pushes :class:`~dnallm.tui.screens.config.ConfigScreen` seeded with the chosen
model (the suitability → config edge of D-09's flow order).

Example:
    app.push_screen(SuitabilityScreen())  # or the data panel's Continue action
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from .. import copy
from ..workers import PROBE_SHORTLIST

if TYPE_CHECKING:
    from collections.abc import Mapping

    from textual.worker import Worker

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
_DETAIL_ID = "suitability-detail"
_USE_BUTTON_ID = "suitability-use"

# Column index of the availability cell in the listing table (one constant, so
# an in-place refresh cannot drift from the column it labels).
_AVAILABILITY_COLUMN = 4

# Core values the renderer branches on. ``AvailabilityState`` and ``Hub`` are
# string enums, so comparing against the value string needs no runtime import
# of the core — the screen stays presentation-only, and the type imports above
# are for typing only.
_STATE_CACHED = "cached"
_STATE_REMOTE_CONFIRMED = "remote_confirmed"
_STATE_UNKNOWN = "unknown"
_HUB_LOCAL = "local"
_REASON_NO_DATA = "NO_DATA"


class SuitabilityScreen(Screen):
    """Render ``check_compatibility``'s report; never compute a verdict."""

    #: "p" re-checks the focused row's hosting — one row, on the user's request.
    BINDINGS: ClassVar[list[Binding]] = [Binding("p", "probe_row", "Check hosting")]

    def __init__(self) -> None:
        """Start on the fine-tune route (base models) with nothing selected."""
        super().__init__()
        self._intent = INTENT_FINETUNE
        self._report: CompatibilityReport | None = None
        self._listing: list[ListingEntry] = []
        self._local_by_model: dict[str, AvailabilityReport] = {}
        self._on_demand: dict[str, AvailabilityReport] = {}
        self._probing: set[str] = set()
        self._local_requested = False
        #: Bumped by every check request. Reports and batches carry the value
        #: they were asked for, so a slower arrival from a listing the user has
        #: already left is ignored instead of overwriting the current one.
        self._generation = 0
        #: The local batch running for the current listing, cancelled when a
        #: newer check replaces that listing (CR-01).
        self._batch_worker: Worker[None] | None = None

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
            yield Static("", markup=False, id=_DETAIL_ID)
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
        """Hand the current route, task type and model names to the core.

        A fresh run starts a fresh local pass: the listing may be entirely
        different, so yesterday's local results can never be shown against it.
        The request gets a new generation, and the previous listing's local
        pass is cancelled — a slow worker from a listing the user has left can
        neither repaint nor become the current report (CR-01).
        """
        self._set_status(copy.PROBING_PLACEHOLDER)
        self._generation += 1
        if self._batch_worker is not None:
            self._batch_worker.cancel()
            self._batch_worker = None
        self._report = None
        self._local_by_model = {}
        self._on_demand = {}
        self._probing = set()
        self._local_requested = False
        self._preflight_app.check_compatibility_worker(
            self._preflight_app.state.data_report,
            task_type=self._task_type(),
            models=self._requested_models(),
            intent=self._intent,
            prober=self._preflight_app.prober,
            probe_shortlist=PROBE_SHORTLIST,
            generation=self._generation,
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
        self,
        report: CompatibilityReport,
        local_by_name: Mapping[str, AvailabilityReport],
        generation: int = 0,
    ) -> None:
        """Render the core's report, then fill in every row's local state.

        Two arrivals share this hook: the compatibility worker posts the report
        (shortlist probes already attached), and the local batch worker posts
        the same report with the offline local states. The second arrival only
        rewrites the availability column, so the user's selection survives.

        Two rules keep a slower arrival from undoing a newer request (CR-01):

        1. An arrival from an older check generation is dropped outright.
        2. A local batch is valid ONLY for the report already on screen. It
           repaints that report; it never *becomes* the current report. Without
           this, a batch for the previous listing landing before the new report
           would be adopted, and the report the user asked for would then be
           dropped for good — leaving the panel on "probing…" forever.

        Args:
            report: The report ``check_compatibility`` produced in the worker.
            local_by_name: Availability already known from the local cache,
                keyed by model id (rendering only; never re-probed).
            generation: The check request this arrival answers; anything older
                than :attr:`_generation` is stale.
        """
        if generation != self._generation:
            # A result for a listing the user has already replaced: ignore it.
            return
        if local_by_name:
            if report is not self._report:
                # A batch can only repaint the report on screen, never replace it.
                return
            self._preflight_app.state.availability_by_entry.update(local_by_name)
            self._local_by_model.update(local_by_name)
            self._refresh_availability_cells()
            return
        self._report = report
        self._preflight_app.state.compatibility_report = report
        self._render_report(report)
        if report.listing or report.verdicts:
            self._set_status(copy.SUITABILITY_READY)
        if report.listing and not self._local_requested:
            self._local_requested = True
            self._batch_worker = self._preflight_app.local_availability_batch_worker(
                report, self._generation
            )

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
                    _cell(_verdict_result_text(dimension.status.value, dimension.reason_code)),
                    _cell(dimension.reason_code or ""),
                )

        listing = self.query_one(f"#{_LISTING_TABLE_ID}", DataTable)
        listing.clear()
        for position, entry in enumerate(self._listing):
            listing.add_row(*self._listing_cells(entry, position))

    def _listing_cells(self, entry: ListingEntry, position: int) -> list[Text]:
        """One listing row's cells, straight from the entry's fields.

        Args:
            entry: The frozen listing row.
            position: Its index in the core's order.

        Returns:
            The cells, markup-free.
        """
        return [
            _cell(entry.entry_name),
            _cell(copy.LAB_ROW_FLAG if entry.is_lab else copy.THIRD_PARTY_ROW_FLAG),
            _cell(entry.task_type or ""),
            _cell(copy.NO_HEAD_LABELS if entry.num_labels is None else str(entry.num_labels)),
            _cell(self._availability_text(entry, position)),
        ]

    def _availability_text(self, entry: ListingEntry, position: int) -> str:
        """One row's availability, in the documented fallback order.

        An on-demand result (the user asked about this row) wins, then the
        shortlist's remote result, then whatever the local pass found, then the
        honest default: a row the core has not verified says "hosting not
        verified" (D-05), and a row still inside the probing window says
        "probing…".

        Args:
            entry: The frozen listing row.
            position: Its index in the core's order.

        Returns:
            The availability sentence for the row.
        """
        if entry.model in self._probing:
            return copy.PROBING_PLACEHOLDER
        availability = self._availability_for(entry)
        is_pending_shortlist_row = availability is None and position < PROBE_SHORTLIST
        if is_pending_shortlist_row:
            return copy.PROBING_PLACEHOLDER
        return _availability_sentence(availability)

    def _availability_for(self, entry: ListingEntry) -> AvailabilityReport | None:
        """Everything known about one row's availability, most recent first.

        Args:
            entry: The frozen listing row.

        Returns:
            The newest report for this model, or None when nothing is known.
        """
        return (
            self._on_demand.get(entry.model)
            or entry.availability
            or self._local_by_model.get(entry.model)
        )

    def _refresh_availability_cells(self) -> None:
        """Rewrite only the availability column, keeping every other cell."""
        listing = self.query_one(f"#{_LISTING_TABLE_ID}", DataTable)
        for position, entry in enumerate(self._listing):
            if position >= listing.row_count:
                return
            listing.update_cell_at(
                Coordinate(row=position, column=_AVAILABILITY_COLUMN),
                _cell(self._availability_text(entry, position)),
            )

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

        An unknown availability never blocks the choice (D-06): the row is
        chosen exactly like a confirmed one, and the detail line says why.

        Args:
            entry: The frozen listing row the user focused.
        """
        self._preflight_app.state.selected_entry = entry
        self.query_one(f"#{_USE_BUTTON_ID}", Button).disabled = False
        self._show_detail(entry)

    def _show_detail(self, entry: ListingEntry) -> None:
        """Show what is known about the chosen row, in the core's own words.

        Args:
            entry: The frozen listing row the user focused.
        """
        availability = self._availability_for(entry)
        lines: list[str] = []
        if availability is not None and availability.reason:
            # The probe's reason is a technical identifier: shown verbatim.
            lines.append(f"{copy.AVAILABILITY_DETAIL_LABEL} {availability.reason}")
        if availability is None or availability.state == _STATE_UNKNOWN:
            # D-06: nothing verified is not "unavailable". Say the user may
            # still choose this model, and why that is safe.
            lines.append(copy.UNKNOWN_STILL_DOWNLOADABLE)
        self.query_one(f"#{_DETAIL_ID}", Static).update("\n".join(lines))

    def _position_of(self, model: str) -> int | None:
        """The listing row index for *model*, or None when it is not listed.

        Args:
            model: The repo id to locate.
        """
        for position, entry in enumerate(self._listing):
            if entry.model == model:
                return position
        return None

    async def apply_availability_update(self, model: str, availability: AvailabilityReport) -> None:
        """Replace exactly one row's availability with a re-checked report.

        Args:
            model: The repo id the on-demand probe was asked about.
            availability: The report the core returned (an unknown state is a
                valid answer and is shown as such, PFC-06).
        """
        self._probing.discard(model)
        self._on_demand[model] = availability
        self._preflight_app.state.availability_by_entry[model] = availability
        position = self._position_of(model)
        if position is not None:
            listing = self.query_one(f"#{_LISTING_TABLE_ID}", DataTable)
            listing.update_cell_at(
                Coordinate(row=position, column=_AVAILABILITY_COLUMN),
                _cell(self._availability_text(self._listing[position], position)),
            )
            selected = self._preflight_app.state.selected_entry
            if selected is not None and selected.model == model:
                self._show_detail(selected)
        self._set_status(copy.SUITABILITY_READY)

    def action_probe_row(self) -> None:
        """Re-check the focused row's hosting — that row only (D-04).

        The automatic pass is capped; this is the user's explicit request for
        one row, so it probes one row and replaces one row.
        """
        listing = self.query_one(f"#{_LISTING_TABLE_ID}", DataTable)
        position = listing.cursor_row
        entry = self._entry_at(position)
        if entry is None:
            self._set_status(copy.SELECT_MODEL_HINT)
            return
        self._probing.add(entry.model)
        listing.update_cell_at(
            Coordinate(row=position, column=_AVAILABILITY_COLUMN),
            _cell(copy.PROBING_PLACEHOLDER),
        )
        self._set_status(copy.PROBING_PLACEHOLDER)
        self._preflight_app.probe_availability_worker(entry.model, self._preflight_app.prober)

    def _confirm_selection(self) -> None:
        """Open the config panel seeded with the chosen model (D-09).

        This is the suitability → config edge, and the only place it is
        implemented: ``25-07`` built the control, this plan gives it its
        destination. The model and the task type the user typed are handed to
        the panel, which builds the config through the core's
        ``generate_config``.
        """
        entry = self._preflight_app.state.selected_entry
        if entry is None:
            self._set_status(copy.SELECT_MODEL_HINT)
            return
        self._set_status(copy.USE_MODEL_CONFIRMED.format(model=entry.entry_name))
        self._preflight_app.show_config(model=entry.model, task_type=self._task_type())

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one(f"#{_STATUS_ID}", Static).update(text)


def _availability_sentence(availability: AvailabilityReport | None) -> str:
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


def _verdict_result_text(status: str, reason_code: str | None) -> str:
    """The result cell for one dimension verdict.

    ``NO_DATA`` is the core's way of saying this check needs data (D-08). It is
    stated in words instead of being shown as a status that could read as a
    failure: the dimension was not measured, so nothing was rejected.

    Args:
        status: The verdict status value (``ok``/``warn``/``fail``/``unknown``).
        reason_code: The dimension's reason code, when it has one.

    Returns:
        The cell text for the result column.
    """
    if reason_code == _REASON_NO_DATA:
        return copy.NO_DATA_YET
    return status


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
