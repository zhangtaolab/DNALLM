"""Seam tests for ``dnallm.tui.workers`` (25-02).

The seam is the only bridge between the Textual event loop and every blocking
core call, so it is tested end-to-end: a headless Pilot run drives a real
``validate_data`` call inside a thread worker and asserts the typed payload
reaches the host app.
"""

from textual.app import App

from dnallm.tui.messages import DataReportReady, DownloadFinished, WorkerFailed
from dnallm.tui.workers import PreflightWorkers


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
