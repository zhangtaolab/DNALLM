"""The single thread-worker seam between the wizard's event loop and the core.

Every blocking core call (data validation, compatibility probes, config
generation, downloads) runs in a Textual thread worker: the event loop keeps
painting, and the only thing that crosses the thread boundary back to the UI is
a typed ``post_message`` payload from :mod:`dnallm.tui.messages`. Widgets are
never touched from a worker thread — ``tests/tui/test_structure.py`` enforces
that by parsing this module and asserting ``post_message`` is the only ``self``
call inside every ``@work`` body.

Heavy core modules are imported inside the worker bodies, never at module
scope: importing this module must not pull in torch or transformers.
"""

from __future__ import annotations

from typing import Any

from textual import work
from textual.worker import get_current_worker

from ..utils import get_logger
from .messages import DataReportReady, WorkerFailed

logger = get_logger("dnallm.tui.workers")

__all__ = ["PreflightWorkers"]

# UI copy: written for a biologist, never an exception class name (D-10).
_DATA_FAILED_COPY = (
    "We could not read this data. Check that the path is right and that the "
    "file format is one of CSV, TSV, JSON, FASTA or TXT, then try again."
)


class PreflightWorkers:
    """Thread workers for the pre-flight wizard, mixed into the Textual app.

    Each worker owns one blocking core call, reports progress/results with
    typed messages, and never touches a widget. Later plans extend this mixin
    with the remaining core calls.
    """

    @work(thread=True, exit_on_error=False)
    def validate_data_worker(self, source: Any) -> None:
        """Validate *source* off the event loop and post the resulting report.

        Args:
            source: Anything ``dnallm.preflight.validate_data`` accepts — a
                path, a split mapping, or a list of raw sequences.
        """
        worker = get_current_worker()
        try:
            from ..preflight import validate_data

            data_report = validate_data(source)
        except Exception:
            # Class name and traceback go to the log; the user reads plain words.
            logger.exception("validate_data failed inside the TUI worker")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(_DATA_FAILED_COPY))
            return
        if worker.is_cancelled:
            return
        self.post_message(DataReportReady(data_report))
