"""Seam tests for ``dnallm.tui.workers`` (25-02).

The seam is the only bridge between the Textual event loop and every blocking
core call, so it is tested end-to-end: a headless Pilot run drives a real
``validate_data`` call inside a thread worker and asserts the typed payload
reaches the host app.

The download tests carry the Wave-0 probe (RESEARCH assumption A3): the cancel
sentinel is pushed through the REAL ``download_model`` so the design decision —
``DownloadCancelled`` derives from ``BaseException`` and therefore escapes the
``except Exception`` retry loop — is proven before any screen exists. The fake
downloader counts calls; nothing here asserts that a thread stopped.
"""

import threading
import time
from types import SimpleNamespace

import pytest
from textual.app import App

import dnallm.tui.workers as workers_module
from dnallm.tui.messages import DataReportReady, DownloadFinished, WorkerFailed
from dnallm.tui.workers import (
    DownloadCancelled,
    PreflightWorkers,
    make_cancellable_downloader,
)


class _Harness(PreflightWorkers, App[None]):
    """Minimal host app: records every typed payload the seam delivers."""

    def __init__(self) -> None:
        super().__init__()
        self.data_reports: list = []
        self.worker_failures: list = []
        self.download_events: list = []

    def on_data_report_ready(self, message: DataReportReady) -> None:
        self.data_reports.append(message.data_report)

    def on_worker_failed(self, message: WorkerFailed) -> None:
        self.worker_failures.append(message.message)

    def on_download_finished(self, message: DownloadFinished) -> None:
        self.download_events.append(message)


async def test_seam_round_trip_delivers_a_real_data_report() -> None:
    """A real validate_data call runs in a worker and its report is delivered."""
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        app.validate_data_worker(["ACGT" * 10, "ACGTN" * 8])
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert len(app.data_reports) == 1
    report = app.data_reports[0]
    assert len(report.splits) == 1
    assert report.splits[0].row_count == 2
    assert app.worker_failures == []


async def test_worker_failure_is_reported_in_plain_language() -> None:
    """A core call that raises becomes a plain-language WorkerFailed payload."""
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        app.validate_data_worker(123)  # an int is not a valid validate_data source
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert app.data_reports == []
    assert len(app.worker_failures) == 1
    message = app.worker_failures[0]
    assert "ValueError" not in message  # class names stay in the log (D-10)
    assert "could not read this data" in message


def test_download_cancelled_escapes_download_models_retry_loop() -> None:
    """Wave-0 probe: the BaseException sentinel leaves download_model in one attempt."""
    from dnallm.models.model import download_model

    calls: list[tuple[str, dict]] = []

    def fake_downloader(model_name: str, **kwargs: object) -> str:
        calls.append((model_name, kwargs))
        raise DownloadCancelled()

    started = time.monotonic()
    with pytest.raises(DownloadCancelled):
        download_model("probe/model", fake_downloader, max_try=10)
    elapsed = time.monotonic() - started

    # One call, no retry: an Exception-derived sentinel would be swallowed by
    # the `except Exception` loop and retried ten times, one second apart.
    assert len(calls) == 1
    assert elapsed < 2.0


def test_cancellable_downloader_blocks_the_retry_attempt(monkeypatch) -> None:
    """A cancel landing mid-attempt stops the download before the retry."""
    from dnallm.models import model as model_module

    # Keep the (single) retry-loop sleep out of the test's wall time.
    monkeypatch.setattr(model_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    cancel_state = SimpleNamespace(cancelled_event=threading.Event())
    calls: list[str] = []

    def flaky_downloader(model_name: str, **kwargs: object) -> str:
        calls.append(model_name)
        cancel_state.cancelled_event.set()  # the user pressed stop during this attempt
        raise ConnectionError("connection reset")

    cancellable = make_cancellable_downloader(cancel_state, flaky_downloader)
    with pytest.raises(DownloadCancelled):
        model_module.download_model("probe/model", cancellable, max_try=10)

    assert calls == ["probe/model"]


async def test_worker_cancel_before_first_attempt_delivers_cancelled(monkeypatch) -> None:
    """A worker whose cancel flag is already set never calls the downloader."""
    # A pre-set cancel flag stands in for a cancel that lands before the thread
    # body starts (threads cannot be interrupted, so only the flag is checkable).
    cancelled_worker = SimpleNamespace(cancelled_event=threading.Event(), is_cancelled=True)
    cancelled_worker.cancelled_event.set()
    monkeypatch.setattr(workers_module, "get_current_worker", lambda: cancelled_worker)

    calls: list[str] = []

    def fake_downloader(model_name: str, **kwargs: object) -> str:
        calls.append(model_name)
        return "/tmp/never-used"

    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        app.download_worker("probe/model", fake_downloader)
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == []
    assert len(app.download_events) == 1
    finished = app.download_events[0]
    assert finished.cancelled is True
    assert finished.path is None
    assert finished.error is None
