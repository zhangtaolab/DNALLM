"""The download panel (TUI-07): the optional last step, with an honest stop.

Presentation only. Pressing "Download this model" calls the app's
``download_worker`` seam — the one place ``download_model`` is reached, wrapped
with the cooperative pre-attempt cancellation check built in 25-02 — and renders
the typed :class:`~dnallm.tui.messages.DownloadProgress` /
:class:`~dnallm.tui.messages.DownloadFinished` payloads that come back. The
screen never imports ``download_model``, never constructs a path, and never
runs a retry loop of its own.

Four rules this panel exists to keep:

1. **Cancel is cooperative, and the copy says so.** Textual thread workers
   cannot be interrupted, so a stop takes effect before the *next* attempt; an
   attempt already talking to the hub finishes. The status line is
   ``CANCEL_PENDING_NOTE`` ("Stopping — this download attempt will finish
   first"), never "cancelled" while an attempt is still running (RESEARCH
   Flag 2, Pitfall 1).
2. **Progress never goes backwards within one measurement.** Updates arrive
   from the downloader's own callbacks. A new attempt (or a new file bar, which
   reports its own total) starts a new measurement; inside one measurement the
   shown value is clamped to stay non-decreasing.
3. **Downloading is optional (D-09).** Skip from an idle panel reaches the
   summary with no downloader call. Skip while a transfer is running first asks
   it to stop and waits for the worker's outcome, so the closing page never
   claims nothing was written while an attempt is still writing (WR-03).
   Quitting is never blocked.
4. **One model, sequential (D-09).** The screen holds exactly the single model
   the suitability panel recorded — no queue, no persistence.

Example:
    app.push_screen(DownloadScreen())  # after the config panel's save
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ProgressBar, Static

from .. import copy
from .summary import SummaryScreen

if TYPE_CHECKING:
    from textual.timer import Timer
    from textual.worker import Worker

    from ...preflight import AvailabilityReport, ListingEntry
    from ..app import DNALLMPreflightApp

__all__ = ["DownloadScreen"]

_TITLE_ID = "download-title"
_MODEL_ID = "download-model"
_AVAILABILITY_ID = "download-availability"
_PROGRESS_ID = "download-progress"
_BAR_ID = "download-bar"
_ELAPSED_ID = "download-elapsed"
_STATUS_ID = "download-status"
_START_ID = "download-start"
_CANCEL_ID = "download-cancel"
_RETRY_ID = "download-retry"
_SKIP_ID = "download-skip"

# Decimal size units, matching how hub pages quote a model's size.
_SIZE_UNITS = ("bytes", "KB", "MB", "GB", "TB")


def _format_size(size_bytes: int | None) -> str:
    """Render a repo size for a reader, or the plain unknown-size phrase.

    Args:
        size_bytes: The repo size the availability report carried, or None.

    Returns:
        A human-readable size, or :data:`dnallm.tui.copy.DOWNLOAD_SIZE_UNKNOWN`
        when the size is missing or zero. No arithmetic is attempted on a
        missing value (PFC-06), so an unknown size can never divide by zero.
    """
    if not size_bytes:
        return copy.DOWNLOAD_SIZE_UNKNOWN
    size = float(size_bytes)
    for unit in _SIZE_UNITS:
        if size < 1000 or unit == _SIZE_UNITS[-1]:
            if unit == "bytes":
                return f"{int(size)} bytes"
            return f"{size:.1f} {unit}"
        size /= 1000
    return copy.DOWNLOAD_SIZE_UNKNOWN


def _format_elapsed(seconds: float) -> str:
    """Render elapsed time as ``minutes:seconds``.

    Args:
        seconds: Whole seconds spent on the download so far.

    Returns:
        A ``m:ss`` string.
    """
    whole = max(int(seconds), 0)
    return f"{whole // 60}:{whole % 60:02d}"


class DownloadScreen(Screen):
    """Offer one model's download, with visible progress and a cooperative stop."""

    #: Escape returns to the config panel, unless a download is still running.
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "go_back", "Back", show=False)]

    def __init__(self) -> None:
        """Build the panel; the model comes from the wizard state on mount."""
        super().__init__()
        self._model: str | None = None
        self._entry: ListingEntry | None = None
        self._availability: AvailabilityReport | None = None
        self._worker: Worker[Any] | None = None
        self._timer: Timer | None = None
        self._started: float | None = None
        self._done = 0
        self._total = 0
        self._downloading = False
        self._finished = False
        #: Set when Skip is pressed mid-transfer: the summary waits for the
        #: worker's own outcome, so it can describe what really happened.
        self._skip_pending = False

    @property
    def _preflight_app(self) -> DNALLMPreflightApp:
        """The typed app, for the worker seam and the markup-safe renderer."""
        return cast("DNALLMPreflightApp", self.app)

    def compose(self) -> ComposeResult:
        """Lay out the model line, the progress widget, and the four actions."""
        yield Header()
        with VerticalScroll(id="download-body"):
            yield Static(copy.DOWNLOAD_PANEL_TITLE, id=_TITLE_ID)
            yield Static(copy.DOWNLOAD_BLURB, markup=False, id="download-blurb")
            yield Static(copy.DOWNLOAD_MODEL_LABEL, id="download-model-label")
            yield Static("", markup=False, id=_MODEL_ID)
            yield Static("", markup=False, id=_AVAILABILITY_ID)
            yield Static("", markup=False, id=_PROGRESS_ID)
            yield ProgressBar(total=None, show_eta=False, id=_BAR_ID)
            yield Static("", markup=False, id=_ELAPSED_ID)
            yield Static("", markup=False, id=_STATUS_ID)
            with Horizontal(id="download-actions"):
                yield Button(copy.DOWNLOAD_START_ACTION, id=_START_ID, variant="primary")
                yield Button(copy.DOWNLOAD_CANCEL_ACTION, id=_CANCEL_ID, disabled=True)
                yield Button(copy.DOWNLOAD_RETRY_ACTION, id=_RETRY_ID)
                yield Button(copy.DOWNLOAD_SKIP_ACTION, id=_SKIP_ID)
        yield Static(copy.QUIT_HINT, id="download-quit-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Read the chosen model from the wizard state and set the first state.

        With no model selected the step has nothing to offer (D-09's empty
        case): the download action stays switched off and the wizard goes
        straight to the summary instead of dead-ending on an empty panel.
        """
        self.query_one(f"#{_RETRY_ID}", Button).display = False
        state = self._preflight_app.state
        entry = state.selected_entry
        self._entry = entry
        if entry is None:
            self._set_status(copy.DOWNLOAD_NO_MODEL)
            self._show_summary(None)
            return
        self._model = entry.model
        self._availability = state.availability_by_entry.get(entry.model) or entry.availability
        self.query_one(f"#{_MODEL_ID}", Static).update(entry.model)
        self.query_one(f"#{_AVAILABILITY_ID}", Static).update(self._availability_line())
        self._set_status(copy.DOWNLOAD_NOT_STARTED)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the start, cancel, retry and skip actions.

        Args:
            event: The button press; only this panel's buttons are handled.
        """
        if event.button.id == _START_ID:
            self._start_download()
        elif event.button.id == _CANCEL_ID:
            self._cancel()
        elif event.button.id == _RETRY_ID:
            self._start_download()
        elif event.button.id == _SKIP_ID:
            self._skip()

    async def apply_download_progress(self, done: int, total: int, label: str) -> None:
        """Repaint the progress line and bar from one downloader callback.

        The bar is honest about the scale it is showing: a new ``total`` is a
        new measurement (a fresh attempt, or one file's own byte count) and
        restarts the bar, while updates inside one measurement are clamped so
        the bar never moves backwards (T-25-17).

        Args:
            done: Units finished in the current measurement.
            total: Units the current measurement covers; 0 when unknown.
            label: Plain-language description of the current stage.
        """
        if self._finished:
            return
        self.query_one(f"#{_PROGRESS_ID}", Static).update(label)
        bar = self.query_one(f"#{_BAR_ID}", ProgressBar)
        if total <= 0:
            bar.update(total=None, progress=0)
            return
        if total != self._total:
            self._total = total
            self._done = max(done, 0)
        else:
            self._done = max(done, self._done)
        bar.update(total=total, progress=min(self._done, total))

    async def apply_download_finished(
        self, path: str | None, cancelled: bool, error: str | None
    ) -> None:
        """Show the outcome: the summary on success, a retryable state otherwise.

        Args:
            path: The path ``download_model`` returned, verbatim, or None.
            cancelled: True when the cooperative stop took effect.
            error: Plain-language failure sentence, or None.
        """
        if self._finished:
            return
        self._stop_running()
        if self._skip_pending:
            # Skip was pressed while the transfer was running. Only now can the
            # closing page tell the truth: the files are in place, or the stop
            # left them incomplete — it must never claim nothing was written
            # while an attempt was (WR-03).
            self._skip_pending = False
            self._show_summary(
                path,
                model_note=None if path is not None else copy.SUMMARY_MODEL_STOPPED,
            )
            return
        if path is not None and not cancelled and error is None:
            self.query_one(f"#{_STATUS_ID}", Static).update(copy.DOWNLOAD_DONE)
            self._show_summary(path)
            return
        self.query_one(f"#{_CANCEL_ID}", Button).disabled = True
        self.query_one(f"#{_RETRY_ID}", Button).display = True
        self.query_one(f"#{_SKIP_ID}", Button).disabled = False
        self._set_status(copy.DOWNLOAD_CANCELLED if cancelled else (error or copy.DOWNLOAD_FAILED))

    async def apply_worker_failure(self, message: str) -> None:
        """Show a failed seam call as one plain-language row.

        Args:
            message: The already-plain-language sentence the seam produced.
        """
        self._stop_running()
        self.query_one(f"#{_RETRY_ID}", Button).display = True
        self._set_status(message)

    def action_go_back(self) -> None:
        """Return to the config panel, but never mid-download."""
        if self._downloading:
            return
        self._preflight_app.pop_screen()

    def _start_download(self) -> None:
        """Ask the seam to download this one model, and start the timer."""
        if self._model is None or self._downloading:
            return
        self._finished = False
        self._downloading = True
        self._done = 0
        self._total = 0
        self._started = time.monotonic()
        self.query_one(f"#{_START_ID}", Button).disabled = True
        self.query_one(f"#{_RETRY_ID}", Button).display = False
        self.query_one(f"#{_SKIP_ID}", Button).disabled = False
        self.query_one(f"#{_CANCEL_ID}", Button).disabled = False
        self.query_one(f"#{_BAR_ID}", ProgressBar).update(total=None, progress=0)
        self.query_one(f"#{_PROGRESS_ID}", Static).update(
            copy.DOWNLOAD_PROGRESS.format(model=self._model)
        )
        self._set_status(copy.DOWNLOAD_PROGRESS.format(model=self._model))
        self._tick()
        self._timer = self.set_interval(1.0, self._tick)
        self._worker = self._preflight_app.download_worker(
            self._model,
            self._preflight_app.downloader,
            entry=self._entry,
            availability=self._availability,
        )

    def _cancel(self) -> None:
        """Ask the worker to stop before its next attempt, and say so honestly."""
        if self._worker is None or not self._downloading:
            return
        self._worker.cancel()
        self.query_one(f"#{_CANCEL_ID}", Button).disabled = True
        self._set_status(copy.CANCEL_PENDING_NOTE)

    def _skip(self) -> None:
        """Leave for the summary without downloading (D-09's first-class skip).

        While a transfer is running this only *asks* it to stop: cancelling is
        cooperative, so the attempt in flight keeps writing until it returns.
        Jumping to the summary now would state that nothing was downloaded while
        bytes are still landing, so the summary waits for the worker's outcome
        (WR-03).
        """
        if self._downloading:
            self._skip_pending = True
            self._cancel()
            return
        self._stop_running()
        self._show_summary(None)

    def _show_summary(self, model_path: str | None, *, model_note: str | None = None) -> None:
        """Push the closing summary with the path exactly as it came back.

        Args:
            model_path: The downloader's return value, or None when skipped.
            model_note: What to say instead of the plain "skipped" sentence
                when there is no path (a stopped transfer may have written
                part of the files).
        """
        self._finished = True
        self._stop_running()
        self._preflight_app.push_screen(
            SummaryScreen(
                model_path=model_path,
                config_path=self._preflight_app.state.export_path,
                model_note=model_note,
            )
        )

    def _stop_running(self) -> None:
        """Mark the download as no longer running and stop the elapsed timer."""
        self._downloading = False
        self._worker = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        """Repaint the elapsed-time line once a second while a download runs."""
        if self._started is None:
            return
        elapsed = time.monotonic() - self._started
        self.query_one(f"#{_ELAPSED_ID}", Static).update(
            copy.DOWNLOAD_ELAPSED.format(elapsed=_format_elapsed(elapsed))
        )

    def _availability_line(self) -> str:
        """The size line, from the report the core produced (never a guess)."""
        size = getattr(self._availability, "size_bytes", None)
        return copy.DOWNLOAD_SIZE_LABEL.format(size=_format_size(size))

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one(f"#{_STATUS_ID}", Static).update(text)
