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
from .copy import DOWNLOAD_PROGRESS, ENV_CHECK_FAILED, SUITABILITY_FAILED
from .messages import (
    AvailabilityUpdated,
    CompatibilityReady,
    ConfigReady,
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

__all__ = [
    "PROBE_SHORTLIST",
    "DownloadCancelled",
    "PreflightWorkers",
    "make_cancellable_downloader",
]

# UI copy: written for a biologist, never an exception class name (D-10).
_DATA_FAILED_COPY = (
    "We could not read this data. Check that the path is right and that the "
    "file format is one of CSV, TSV, JSON, FASTA or TXT, then try again."
)
_DOWNLOAD_FAILED_COPY = "The download did not finish. Check your internet connection and try again."
_CONFIG_FAILED_COPY = (
    "We could not build this config. Check that the model name is right and "
    "that the config file exists, then try again."
)

#: Worker group for the config panel. ``exclusive`` makes a newer validation
#: run replace an older one, so a fast typist's last keystroke is the result
#: that renders (results can never arrive out of order and overwrite it).
CONFIG_WORKER_GROUP = "config"

#: How many listing rows the compatibility worker asks the core to probe
#: (D-04/D-10: the lab-first shortlist only — never every entry).
PROBE_SHORTLIST = 20


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

    @work(thread=True, exit_on_error=False, group=CONFIG_WORKER_GROUP, exclusive=True)
    def generate_config_worker(self, model: str, *, task_type: str | None = None) -> None:
        """Build a validated config for *model* off the event loop.

        Args:
            model: The model name the suitability panel recorded.
            task_type: The task type to use for a base model; the core ignores
                it for a finetuned registry entry, whose own task wins.
        """
        worker = get_current_worker()
        try:
            from ..preflight import generate_config

            generated = generate_config(model, task_type=task_type)
        except Exception as exc:
            # Details go to the log; the user reads plain words, never a class name.
            logger.error(f"generate_config failed inside the TUI worker: {exc!r}")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(_CONFIG_FAILED_COPY))
            return
        if worker.is_cancelled:
            return
        self.post_message(ConfigReady(generated))

    @work(thread=True, exit_on_error=False, group=CONFIG_WORKER_GROUP, exclusive=True)
    def edit_config_worker(
        self,
        existing: Any,
        patch: Any = None,
        *,
        output_path: str | None = None,
    ) -> None:
        """Revalidate an existing config (or raw mapping) off the event loop.

        Args:
            existing: A YAML path or the raw mapping the config panel is
                editing. Both are the core's documented round-trip inputs.
            patch: Field overrides per section; ``{}`` / None revalidates the
                input unchanged (the panel's empty-edit no-op).
            output_path: When given, the core writes exactly there — and only
                when the run carries no error records (WR-02), so the file the
                user sees can never be invalid.
        """
        worker = get_current_worker()
        try:
            from ..preflight import edit_config

            generated = edit_config(existing, patch, output_path=output_path)
        except Exception as exc:
            # Details go to the log; the user reads plain words, never a class name.
            logger.error(f"edit_config failed inside the TUI worker: {exc!r}")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(_CONFIG_FAILED_COPY))
            return
        if worker.is_cancelled:
            return
        self.post_message(ConfigReady(generated, output_path))

    @work(thread=True, exit_on_error=False)
    def check_compatibility_worker(
        self,
        data: Any,
        *,
        task_type: str | None = None,
        models: list[str] | None = None,
        intent: str | None = None,
        prober: Callable[[str], Any] | None = None,
        probe_shortlist: int = PROBE_SHORTLIST,
    ) -> None:
        """Work out model suitability and the intent listing off the event loop.

        The core probes only the first ``probe_shortlist`` listing rows (labs
        first); every visible row gets its offline local state from a second,
        separate worker so the automatic probes stay bounded (D-04).

        Args:
            data: The frozen ``DataReport`` already validated, or ``None`` when
                the user skipped the data step (D-08).
            task_type: Optional task type filter, passed through verbatim.
            models: Optional explicit model names; when given the core returns
                verdicts instead of a listing.
            intent: The core's intent vocabulary (``"finetune"`` /
                ``"predict"``). Screens pass their pinned constants, never a
                UI label.
            prober: The app's injected availability prober. ``None`` means
                production: the core's own ``probe_availability`` is used.
                Tests inject a fake so nothing touches the network.
            probe_shortlist: How many lab-first listing rows the core may
                probe (D-04/D-10: never all entries).
        """
        worker = get_current_worker()
        try:
            from ..preflight import check_compatibility
            from ..preflight.availability import probe_availability

            effective_prober = prober if prober is not None else probe_availability
            report = check_compatibility(
                data,
                task_type=task_type,
                models=models,
                intent=intent,
                prober=effective_prober,
                probe_shortlist=probe_shortlist,
            )
        except Exception as exc:
            # Details go to the log; the user reads plain words, never a class name.
            logger.error(f"check_compatibility failed inside the TUI worker: {exc!r}")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(SUITABILITY_FAILED))
            return
        if worker.is_cancelled:
            return
        self.post_message(CompatibilityReady(report, {}))

    @work(thread=True, exit_on_error=False)
    def local_availability_batch_worker(self, report: Any) -> None:
        """Classify every listing row's local cache state off the event loop.

        This is the free, offline half of D-04: pure path and cache-directory
        checks for the whole listing, so a row outside the probed shortlist
        still shows whether the files are already on this computer. No hub is
        contacted here.

        Args:
            report: The frozen ``CompatibilityReport`` whose ``listing`` rows
                are classified. Its rows are read, never changed.
        """
        worker = get_current_worker()
        try:
            from ..preflight import local_availability

            by_model = {entry.model: local_availability(entry.model) for entry in report.listing}
        except Exception as exc:
            # Details go to the log; the user reads plain words, never a class name.
            logger.error(f"local_availability failed inside the TUI worker: {exc!r}")
            if not worker.is_cancelled:
                self.post_message(WorkerFailed(SUITABILITY_FAILED))
            return
        if worker.is_cancelled:
            return
        self.post_message(CompatibilityReady(report, by_model))

    @work(thread=True, exit_on_error=False)
    def probe_availability_worker(
        self, model: str, prober: Callable[[str], Any] | None = None
    ) -> None:
        """Re-check one model's hosting off the event loop (the on-demand probe).

        Only the row the user asked about is probed: the shortlist cap is a
        policy for the automatic pass, and an explicit request is the user's
        own single probe (D-04), never a sweep of the rest of the listing.

        Args:
            model: The repo id of the focused listing row.
            prober: The app's injected availability prober. ``None`` means
                production: the core's own ``probe_availability`` is used.
        """
        worker = get_current_worker()
        from ..preflight.availability import probe_availability

        effective_prober = prober if prober is not None else probe_availability
        availability = probe_availability(model, prober=effective_prober)
        if worker.is_cancelled:
            return
        self.post_message(AvailabilityUpdated(model, availability))

    @work(thread=True, exit_on_error=False)
    def download_worker(
        self,
        model_name: str,
        downloader: Callable[..., str] | None = None,
        *,
        entry: Any = None,
        availability: Any = None,
    ) -> None:
        """Download *model_name* off the event loop, with a working cancel.

        Args:
            model_name: Registry/repo id passed through to ``download_model``.
            downloader: The real hub download callable. An explicit callable is
                the test hook; ``None`` falls back to the app's injected
                ``downloader`` seam, then to the hub the availability report
                names (25-09).
            entry: The frozen listing row the user chose, used only to pick a
                hub when the report names none.
            availability: The frozen availability report for that row.
        """
        worker = get_current_worker()
        self.post_message(
            DownloadProgress(done=0, total=0, label=DOWNLOAD_PROGRESS.format(model=model_name))
        )
        resolved = downloader if callable(downloader) else getattr(self, "downloader", None)
        try:
            # Function-local: importing dnallm.models pulls torch/transformers in.
            from ..models.model import download_model

            if not callable(resolved):
                from huggingface_hub import snapshot_download

                resolved = snapshot_download
            path = download_model(model_name, make_cancellable_downloader(worker, resolved))
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
