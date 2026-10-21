"""The single thread-worker seam between the wizard's event loop and the core.

Every blocking core call (data validation, compatibility probes, config
generation) runs in a Textual thread worker: the event loop keeps painting, and
the only thing that crosses the thread boundary back to the UI is a typed
``post_message`` payload from :mod:`dnallm.tui.messages`. Widgets are never
touched from a worker thread — ``tests/tui/test_structure.py`` enforces that by
parsing this module and asserting ``post_message`` is the only ``self`` call
inside every ``@work`` body.

The download is the exception: it runs on its own daemon thread
(:class:`DownloadOperation`). Textual thread workers use the loop's default
executor, which ``asyncio.run`` and the interpreter both join at exit — an
uninterruptible transfer would then keep the process alive after the user
quits. The daemon thread keeps the same message-only boundary and the same
cooperative ``cancelled_event`` contract, but nothing ever joins it.

Heavy core modules are imported inside the worker bodies, never at module
scope: importing this module must not pull in torch or transformers.
"""

from __future__ import annotations

import functools
import inspect
import threading
from typing import TYPE_CHECKING, Any

from textual import work
from textual.worker import get_current_worker

from ..utils import get_logger
from .copy import (
    DOWNLOAD_ATTEMPT,
    DOWNLOAD_FAILED,
    DOWNLOAD_PROGRESS,
    ENV_CHECK_FAILED,
    SUITABILITY_FAILED,
)
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

logger = get_logger("dnallm.tui.workers")

__all__ = [
    "COMPATIBILITY_WORKER_GROUP",
    "PROBE_SHORTLIST",
    "DownloadCancelled",
    "DownloadOperation",
    "PreflightWorkers",
    "make_cancellable_downloader",
]

# UI copy: written for a biologist, never an exception class name (D-10).
_DATA_FAILED_COPY = (
    "We could not read this data. Check that the path is right and that the "
    "file format is one of CSV, TSV, JSON, FASTA or TXT, then try again."
)
_DOWNLOAD_FAILED_COPY = DOWNLOAD_FAILED
_CONFIG_FAILED_COPY = (
    "We could not build this config. Check that the model name is right and "
    "that the config file exists, then try again."
)

#: ``download_model``'s attempt budget (T-25-17: core-owned, never a UI knob).
#: Mirrored here only so the attempt line can say "attempt N of M".
DOWNLOAD_MAX_TRY = 10

#: Worker group for the config panel. ``exclusive`` makes a newer validation
#: run replace an older one, so a fast typist's last keystroke is the result
#: that renders (results can never arrive out of order and overwrite it).
CONFIG_WORKER_GROUP = "config"

#: Worker group for the suitability panel's compatibility check. Exclusive, so
#: a new check supersedes an older one: a slow run for the previous listing is
#: cancelled rather than racing the run the user just asked for (CR-01).
COMPATIBILITY_WORKER_GROUP = "compatibility"

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


class DownloadOperation:
    """One model transfer running on its own daemon thread (WR-02).

    Textual thread workers run on ``loop.run_in_executor(None, ...)``. Both
    ``asyncio.run`` and the interpreter itself join that default executor at
    exit, and an attempt already inside a hub client cannot be interrupted, so
    quitting mid-download would leave the process blocked until the transfer
    finished. This operation owns a plain daemon thread instead: the app asks
    it to stop cooperatively and never waits for it, and interpreter exit does
    not wait either.

    It exposes the same ``cancelled_event`` contract as a Textual worker, so
    :func:`make_cancellable_downloader` works unchanged.

    Args:
        target: The thread body (the blocking download call).
    """

    def __init__(self, target: Callable[[], None]) -> None:
        self.cancelled_event = threading.Event()
        self._thread = threading.Thread(target=target, name="dnallm-download", daemon=True)

    def start(self) -> DownloadOperation:
        """Start the transfer thread and return this handle.

        Returns:
            This operation, for ``operation = ...start()`` at the call site.
        """
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Ask the transfer to stop before its next attempt (cooperative)."""
        self.cancelled_event.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the transfer thread — for tests, never for the UI path.

        Args:
            timeout: Seconds to wait at most, or None to wait forever.
        """
        self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        """True while the transfer thread is still running."""
        return self._thread.is_alive()

    @property
    def is_daemon(self) -> bool:
        """True when interpreter exit will not wait for this thread."""
        return self._thread.daemon


def make_cancellable_downloader(worker: Any, downloader: Callable[..., str]) -> Callable[..., str]:
    """Wrap *downloader* with the cooperative pre-attempt cancellation check.

    Textual thread workers (and the download's own daemon thread) cannot be
    interrupted, so cancelling is cooperative: an attempt already inside the
    hub client finishes, and the next attempt never starts.

    Args:
        worker: The running operation; its ``cancelled_event`` is set when the
            user asks to stop.
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


def _choose_hub(entry: Any, availability: Any) -> Any:
    """Pick the hub to download a model from (D-11 owner directive).

    The rules, in order: a hub the availability report actually confirmed;
    otherwise ModelScope for a lab entry (ModelScope first, lab first);
    otherwise Hugging Face. This is plumbing, not domain logic — the core's own
    retry policy inside ``download_model`` still applies, and the screen offers
    a retry rather than trying another hub behind the user's back.

    Args:
        entry: The frozen listing row the user chose, or None.
        availability: That row's frozen availability report, or None.

    Returns:
        A :class:`dnallm.preflight.report.Hub` member.
    """
    from ..preflight.report import Hub

    confirmed = getattr(availability, "source_hub", None)
    if confirmed in (Hub.MODELSCOPE, Hub.HUGGINGFACE):
        return confirmed
    if entry is not None and getattr(entry, "is_lab", False):
        return Hub.MODELSCOPE
    return Hub.HUGGINGFACE


def _resolve_hub_downloader(hub: Any) -> Callable[..., str]:
    """Return the hub client's ``snapshot_download`` for *hub*.

    The imports stay function-local, like every other heavy import in this
    package, so importing the TUI never pulls a hub client in.

    Args:
        hub: A :class:`dnallm.preflight.report.Hub` member.

    Returns:
        The download callable ``download_model`` expects.
    """
    from ..preflight.report import Hub

    if hub == Hub.MODELSCOPE:
        from modelscope.hub.snapshot_download import snapshot_download
    else:
        from huggingface_hub import snapshot_download
    return snapshot_download


def _make_progress_reporter(emit: Callable[[int, int, str], None]) -> Any | None:
    """Build a ``tqdm``-compatible class that forwards byte counts to *emit*.

    Args:
        emit: Callback receiving ``(done, total, label)`` for one update.

    Returns:
        A subclass of ``tqdm`` usable as ``tqdm_class``, or None when tqdm is
        not importable (attempt-level progress is the honest fallback).
    """
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is a declared dependency
        logger.debug("tqdm is unavailable; byte progress stays attempt-level")
        return None

    class _ProgressReporter(tqdm):
        """A tqdm bar that reports its own numbers to the UI thread."""

        def update(self, n: int = 1) -> Any:
            """Advance the bar and post the new position.

            Args:
                n: Units to advance by (bytes for a file bar).

            Returns:
                Whatever the parent bar returned.
            """
            shown = super().update(n)
            total = int(self.total) if self.total else 0
            if total:
                emit(int(self.n), total, str(self.desc or ""))
            return shown

    return _ProgressReporter


def _bind_progress(
    downloader: Callable[..., str], emit: Callable[[int, int, str], None]
) -> Callable[..., str]:
    """Bind a byte-progress reporter when *downloader* accepts one.

    Only an explicitly declared ``tqdm_class`` parameter counts: passing the
    keyword to a client that merely swallows ``**kwargs`` could forward it
    somewhere that rejects it, so a downloader without the parameter is left
    exactly as it was (attempt-level progress only).

    Args:
        downloader: The resolved hub download callable.
        emit: Callback receiving ``(done, total, label)``.

    Returns:
        The downloader unchanged, or a partial with ``tqdm_class`` bound.
    """
    try:
        parameters = inspect.signature(downloader).parameters
    except (TypeError, ValueError):
        return downloader
    if "tqdm_class" not in parameters:
        return downloader
    reporter = _make_progress_reporter(emit)
    if reporter is None:
        return downloader
    return functools.partial(downloader, tqdm_class=reporter)


class PreflightWorkers:
    """Thread workers for the pre-flight wizard, mixed into the Textual app.

    Each worker owns one blocking core call, reports progress/results with
    typed messages, and never touches a widget. Later plans extend this mixin
    with the remaining core calls.

    ``download_worker`` is the one method that is not a Textual ``@work``
    worker: it returns a :class:`DownloadOperation` on a daemon thread so a
    quit never waits for a transfer (WR-02).
    """

    #: The transfer started by :meth:`download_worker`, if any. Quitting asks
    #: it to stop; nobody joins it.
    _download_operation: DownloadOperation | None = None

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

    @work(
        thread=True,
        exit_on_error=False,
        group=COMPATIBILITY_WORKER_GROUP,
        exclusive=True,
    )
    def check_compatibility_worker(
        self,
        data: Any,
        *,
        task_type: str | None = None,
        models: list[str] | None = None,
        intent: str | None = None,
        prober: Callable[[str], Any] | None = None,
        probe_shortlist: int = PROBE_SHORTLIST,
        generation: int = 0,
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
            generation: The panel's check generation, echoed back on the
                result so a newer check can supersede a slower older one
                (CR-01). The worker is also exclusive, so Textual cancels an
                older run in the same group before starting this one.
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
        self.post_message(CompatibilityReady(report, {}, generation))

    @work(thread=True, exit_on_error=False)
    def local_availability_batch_worker(self, report: Any, generation: int = 0) -> None:
        """Classify every listing row's local cache state off the event loop.

        This is the free, offline half of D-04: pure path and cache-directory
        checks for the whole listing, so a row outside the probed shortlist
        still shows whether the files are already on this computer. No hub is
        contacted here.

        Args:
            report: The frozen ``CompatibilityReport`` whose ``listing`` rows
                are classified. Its rows are read, never changed.
            generation: The check generation the report belongs to, echoed so
                the panel can tell a batch for the current listing from one
                for a listing it has already replaced (CR-01).
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
        self.post_message(CompatibilityReady(report, by_model, generation))

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

    def download_worker(
        self,
        model_name: str,
        downloader: Callable[..., str] | None = None,
        *,
        entry: Any = None,
        availability: Any = None,
    ) -> DownloadOperation:
        """Download *model_name* on its own daemon thread, with a working cancel.

        This is deliberately NOT a Textual thread worker (WR-02): workers run
        on the loop's default executor, which is joined at interpreter exit, so
        quitting mid-download would block the process until the transfer
        finished. The returned operation is cancelled cooperatively and is
        never joined by the UI.

        Args:
            model_name: Registry/repo id passed through to ``download_model``.
            downloader: The real hub download callable. An explicit callable is
                the test hook; ``None`` falls back to the app's injected
                ``downloader`` seam, then to the hub the availability report
                names (25-09).
            entry: The frozen listing row the user chose, used only to pick a
                hub when the report names none.
            availability: The frozen availability report for that row.

        Returns:
            The running :class:`DownloadOperation`; ``cancel()`` stops it before
            the next attempt.
        """
        progress_label = DOWNLOAD_PROGRESS.format(model=model_name)
        # Posted from the UI thread, before the transfer starts: the opener is
        # on screen before any byte is requested.
        self.post_message(DownloadProgress(done=0, total=0, label=progress_label))
        resolved = downloader if callable(downloader) else getattr(self, "downloader", None)

        def run() -> None:
            attempt = {"count": 0}

            def post(message: Any) -> None:
                # Thread-safe (Textual routes a cross-thread post through the
                # event loop). A closed loop means the user has already quit,
                # and a transfer that outlives the UI just stops reporting.
                try:
                    self.post_message(message)
                except RuntimeError:
                    logger.debug("The download outlived the message loop; dropping an update")

            def emit_progress(done: int, total: int, label: str) -> None:
                # Called from the hub client's own callbacks, still inside the
                # transfer thread: the typed message is the only thing crossing.
                post(DownloadProgress(done=done, total=total, label=label or progress_label))

            try:
                # Function-local: importing dnallm.models pulls torch/transformers in.
                from ..models.model import download_model

                target = resolved
                if not callable(target):
                    hub = _choose_hub(entry, availability)
                    logger.info(f"Downloading {model_name} from {getattr(hub, 'value', hub)}")
                    target = _resolve_hub_downloader(hub)
                reporting = _bind_progress(target, emit_progress)

                def attempt_downloader(name: str, **kwargs: Any) -> str:
                    # One line per real attempt, posted before the hub is asked.
                    attempt["count"] += 1
                    post(
                        DownloadProgress(
                            done=attempt["count"],
                            total=DOWNLOAD_MAX_TRY,
                            label=DOWNLOAD_ATTEMPT.format(
                                attempt=attempt["count"], max_try=DOWNLOAD_MAX_TRY
                            ),
                        )
                    )
                    return reporting(name, **kwargs)

                path = download_model(
                    model_name,
                    make_cancellable_downloader(operation, attempt_downloader),
                    max_try=DOWNLOAD_MAX_TRY,
                )
            except DownloadCancelled:
                logger.info(f"Download of {model_name} cancelled by the user")
                post(DownloadFinished(path=None, cancelled=True))
                return
            except ValueError as exc:
                # download_model's own failure signal: every attempt was retried.
                logger.warning(f"Download of {model_name} failed after every retry: {exc}")
                post(DownloadFinished(path=None, cancelled=False, error=_DOWNLOAD_FAILED_COPY))
                return
            except Exception as exc:
                logger.error(f"Unexpected failure downloading {model_name}: {exc!r}")
                post(DownloadFinished(path=None, cancelled=False, error=_DOWNLOAD_FAILED_COPY))
                return
            post(DownloadFinished(path=path, cancelled=False))

        operation = DownloadOperation(run)
        #: Kept on the app so quitting can ask it to stop, and so tests can
        #: wait for it without going through the worker manager (it has no entry
        #: there — it is not a Textual worker).
        self._download_operation = operation
        return operation.start()
