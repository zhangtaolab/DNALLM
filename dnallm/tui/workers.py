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

from typing import TYPE_CHECKING, Any

from textual import work
from textual.worker import get_current_worker

from ..utils import get_logger
from .copy import ENV_CHECK_FAILED
from .messages import (
    DataReportReady,
    DownloadFinished,
    DownloadProgress,
    EnvReportReady,
    WorkerFailed,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.worker import Worker

logger = get_logger("dnallm.tui.workers")

__all__ = ["DownloadCancelled", "PreflightWorkers", "make_cancellable_downloader"]

# UI copy: written for a biologist, never an exception class name (D-10).
_DATA_FAILED_COPY = (
    "We could not read this data. Check that the path is right and that the "
    "file format is one of CSV, TSV, JSON, FASTA or TXT, then try again."
)
_DOWNLOAD_FAILED_COPY = "The download did not finish. Check your internet connection and try again."


class DownloadCancelled(BaseException):
    """Raised by a cancellable downloader once the user has asked to stop.

    This derives from :class:`BaseException` on purpose. ``download_model``
    wraps every attempt in ``except Exception`` and retries up to ``max_try``
    times with a one-second sleep, so an ``Exception``-derived sentinel would
    be swallowed as if it were a network hiccup and retried instead of
    stopping the download. ``tests/tui/test_workers.py`` proves the escape
    through the real ``download_model``.
    """


def make_cancellable_downloader(
    worker: Worker[Any], downloader: Callable[..., str]
) -> Callable[..., str]:
    """Wrap *downloader* with the cooperative pre-attempt cancellation check.

    Textual thread workers cannot be interrupted, so cancelling is
    cooperative: an attempt already inside the hub client finishes, and the
    next attempt never starts.

    Args:
        worker: The thread worker running the download; its ``cancelled_event``
            is set when the user asks to stop.
        downloader: The real download callable (for example
            ``snapshot_download``), called with the same keyword arguments
            ``download_model`` forwards.

    Returns:
        A callable that raises :class:`DownloadCancelled` instead of starting
        an attempt the user has already cancelled, and otherwise delegates to
        *downloader* unchanged.
    """

    def cancellable_downloader(model_name: str, **kwargs: Any) -> str:
        if worker.cancelled_event.is_set():
            raise DownloadCancelled()
        return downloader(model_name, **kwargs)

    return cancellable_downloader


class PreflightWorkers:
    """Thread workers for the pre-flight wizard, mixed into the Textual app.

    Each worker owns one blocking core call, reports progress/results with
    typed messages, and never touches a widget. Later plans extend this mixin
    with the remaining core calls.
    """

    @work(thread=True, exit_on_error=False)
    def collect_env_worker(self, collector: Any) -> None:
        """Collect the slower environment facts off the event loop.

        The version rows are already on screen when this runs; only the
        graphics-card and free-disk facts need the worker (D-01).

        Args:
            collector: The app's environment collector. It must expose
                ``collect_device`` and ``collect_disk``; the default is
                :mod:`dnallm.utils.env_info`, and tests inject a fake so no row
                touches torch or the filesystem.
        """
        worker = get_current_worker()
        try:
            device_rows = collector.collect_device()
            disk_rows = collector.collect_disk()
        except Exception as exc:
            # Details go to the log; the user reads plain words, never a class name.
            logger.error(f"Environment collection failed inside the TUI worker: {exc!r}")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(ENV_CHECK_FAILED))
            return
        if worker.is_cancelled:
            return
        self.post_message(EnvReportReady({"device": device_rows, "disk": disk_rows}))

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
        except Exception as exc:
            # Details go to the log; the user reads plain words, never a class name.
            logger.error(f"validate_data failed inside the TUI worker: {exc!r}")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(_DATA_FAILED_COPY))
            return
        if worker.is_cancelled:
            return
        self.post_message(DataReportReady(data_report))

    @work(thread=True, exit_on_error=False)
    def download_worker(self, model_name: str, downloader: Callable[..., str]) -> None:
        """Download *model_name* off the event loop, with a working cancel.

        Args:
            model_name: Registry/repo id passed through to ``download_model``.
            downloader: The real hub download callable. The worker wraps it so
                a cancel stops the download before its next attempt.
        """
        worker = get_current_worker()
        self.post_message(DownloadProgress(done=0, total=1, label=f"Downloading {model_name}"))
        try:
            # Function-local: importing dnallm.models pulls torch/transformers in.
            from ..models.model import download_model

            path = download_model(model_name, make_cancellable_downloader(worker, downloader))
        except DownloadCancelled:
            logger.info(f"Download of {model_name} cancelled by the user")
            self.post_message(DownloadFinished(path=None, cancelled=True))
            return
        except ValueError as exc:
            # download_model's own failure signal: every attempt was retried.
            logger.warning(f"Download of {model_name} failed after every retry: {exc}")
            self.post_message(
                DownloadFinished(path=None, cancelled=False, error=_DOWNLOAD_FAILED_COPY)
            )
            return
        except Exception as exc:
            logger.error(f"Unexpected failure downloading {model_name}: {exc!r}")
            self.post_message(
                DownloadFinished(path=None, cancelled=False, error=_DOWNLOAD_FAILED_COPY)
            )
            return
        self.post_message(DownloadFinished(path=path, cancelled=False))
