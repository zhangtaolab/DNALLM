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

import asyncio
import functools
import threading
import time
from types import SimpleNamespace

import pytest
from textual.app import App

import dnallm.tui.workers as workers_module
from dnallm.preflight import AvailabilityReport, AvailabilityState, Hub, ListingEntry
from dnallm.tui.messages import DataReportReady, DownloadFinished, DownloadProgress, WorkerFailed
from dnallm.tui.workers import (
    DOWNLOAD_MAX_TRY,
    DownloadCancelled,
    PreflightWorkers,
    _bind_progress,
    _choose_hub,
    make_cancellable_downloader,
)


class _Harness(PreflightWorkers, App[None]):
    """Minimal host app: records every typed payload the seam delivers."""

    def __init__(self) -> None:
        super().__init__()
        self.data_reports: list = []
        self.config_events: list = []
        self.compatibility_events: list = []
        self.availability_events: list = []
        self.worker_failures: list = []
        self.download_events: list = []
        self.progress_events: list[DownloadProgress] = []

    def on_data_report_ready(self, message: DataReportReady) -> None:
        self.data_reports.append(message.data_report)

    def on_config_ready(self, message) -> None:
        self.config_events.append(message)

    def on_compatibility_ready(self, message) -> None:
        self.compatibility_events.append(message)

    def on_availability_updated(self, message) -> None:
        self.availability_events.append(message)

    def on_worker_failed(self, message: WorkerFailed) -> None:
        self.worker_failures.append(message.message)

    def on_download_finished(self, message: DownloadFinished) -> None:
        self.download_events.append(message)

    def on_download_progress(self, message: DownloadProgress) -> None:
        self.progress_events.append(message)


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


async def test_validate_worker_drops_the_report_after_a_cancel(monkeypatch) -> None:
    """A cancel that lands during validation suppresses the UI effect."""
    cancelled_worker = SimpleNamespace(cancelled_event=threading.Event(), is_cancelled=True)
    monkeypatch.setattr(workers_module, "get_current_worker", lambda: cancelled_worker)

    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        app.validate_data_worker(["ACGT" * 10])
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert app.data_reports == []
    assert app.worker_failures == []


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


def _cancellable_state() -> SimpleNamespace:
    """An object with the ``cancelled_event`` contract a cancel check needs."""
    return SimpleNamespace(cancelled_event=threading.Event())


def test_cancellable_downloader_blocks_an_attempt_after_a_pre_set_cancel() -> None:
    """A cancel that landed before the attempt started blocks it entirely."""
    state = _cancellable_state()
    state.cancelled_event.set()
    calls: list[str] = []

    cancellable = make_cancellable_downloader(
        state, lambda model_name, **kwargs: calls.append(model_name)
    )

    with pytest.raises(DownloadCancelled):
        cancellable("probe/model")

    assert calls == []


async def _wait_for_download(operation, pilot) -> None:
    """Wait for the transfer's daemon thread, then let its messages land."""
    for _ in range(2000):
        if not operation.is_alive:
            break
        await asyncio.sleep(0.005)
    await pilot.pause()
    await pilot.pause()


async def test_a_cancelled_download_delivers_the_cancelled_outcome(monkeypatch) -> None:
    """A stop during an attempt finishes it and delivers ``cancelled=True``.

    The attempt fails (so the retry loop wants another one); the cooperative
    check blocks that next attempt, which is exactly what a user's stop means.
    """
    from dnallm.models import model as model_module

    # Keep the retry loop's one-second sleep out of the test's wall time.
    monkeypatch.setattr(model_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    started = threading.Event()
    gate = threading.Event()

    def fake_downloader(model_name: str, **kwargs: object) -> str:
        started.set()
        gate.wait(timeout=10)
        raise ConnectionError("connection reset")

    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        operation = app.download_worker("probe/model", fake_downloader)
        for _ in range(500):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        operation.cancel()
        gate.set()
        await _wait_for_download(operation, pilot)

    assert len(app.download_events) == 1
    finished = app.download_events[0]
    assert finished.cancelled is True
    assert finished.path is None
    assert finished.error is None


async def test_download_worker_delivers_the_stored_path() -> None:
    """A download that succeeds delivers the path, not cancelled, no error."""
    app = _Harness()

    def fake_downloader(model_name: str, **kwargs: object) -> str:
        return f"/models/{model_name}"

    async with app.run_test(size=(100, 40)) as pilot:
        operation = app.download_worker("probe/model", fake_downloader)
        await _wait_for_download(operation, pilot)

    assert len(app.download_events) == 1
    finished = app.download_events[0]
    assert finished.path == "/models/probe/model"
    assert finished.cancelled is False
    assert finished.error is None


async def test_download_worker_reports_exhausted_retries_in_plain_language(monkeypatch) -> None:
    """download_model's ValueError becomes plain-language copy, never a class name."""
    from dnallm.models import model as model_module

    # The real retry loop sleeps one second per attempt; keep that out of wall time.
    monkeypatch.setattr(model_module, "time", SimpleNamespace(sleep=lambda _seconds: None))
    attempts: list[str] = []

    def failing_downloader(model_name: str, **kwargs: object) -> str:
        attempts.append(model_name)
        raise ConnectionError("connection reset")

    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        operation = app.download_worker("probe/model", failing_downloader)
        await _wait_for_download(operation, pilot)

    assert len(attempts) == 10  # download_model's default max_try
    assert len(app.download_events) == 1
    finished = app.download_events[0]
    assert finished.cancelled is False
    assert finished.path is None
    assert "ValueError" not in finished.error
    assert "did not finish" in finished.error


# ─────────────────────────────────────────────────────────────────────────────
# Hub choice, progress binding, and plain-language failure copy (25-09 Task 2)
# ─────────────────────────────────────────────────────────────────────────────


def _entry(is_lab: bool) -> ListingEntry:
    """A frozen listing row, lab or third-party."""
    return ListingEntry(model="owner/model", entry_name="model", is_lab=is_lab)


def _availability(state: AvailabilityState, hub: Hub | None) -> AvailabilityReport:
    """A frozen availability report with (or without) a confirmed hub."""
    return AvailabilityReport(state=state, size_bytes=None, source_hub=hub)


def test_choose_hub_uses_a_confirmed_hub_even_for_a_lab_entry() -> None:
    """A hub the probe actually confirmed wins over the lab default."""
    availability = _availability(AvailabilityState.REMOTE_CONFIRMED, Hub.HUGGINGFACE)

    assert _choose_hub(_entry(is_lab=True), availability) is Hub.HUGGINGFACE


def test_choose_hub_defaults_to_modelscope_for_a_lab_entry() -> None:
    """With no confirmed hub, a lab model keeps the ModelScope-first rule."""
    availability = _availability(AvailabilityState.UNKNOWN, None)

    assert _choose_hub(_entry(is_lab=True), availability) is Hub.MODELSCOPE
    assert _choose_hub(_entry(is_lab=True), None) is Hub.MODELSCOPE


def test_choose_hub_defaults_to_hugging_face_for_a_third_party_entry() -> None:
    """A third-party model with no confirmed hub falls back to Hugging Face."""
    availability = _availability(AvailabilityState.UNKNOWN, None)

    assert _choose_hub(_entry(is_lab=False), availability) is Hub.HUGGINGFACE
    assert _choose_hub(None, None) is Hub.HUGGINGFACE


def test_bind_progress_binds_tqdm_class_only_when_declared() -> None:
    """Byte progress is bound for a client that declares tqdm_class."""
    seen: list[tuple[int, int, str]] = []

    def with_bytes(model_name: str, *, tqdm_class=None, **kwargs: object) -> str:
        return model_name

    def attempt_only(model_name: str, **kwargs: object) -> str:
        return model_name

    bound = _bind_progress(with_bytes, lambda done, total, label: seen.append((done, total, label)))

    assert isinstance(bound, functools.partial)
    assert bound.keywords["tqdm_class"] is not None
    # A downloader that only swallows **kwargs is left exactly as it was.
    assert _bind_progress(attempt_only, lambda done, total, label: None) is attempt_only


async def test_a_byte_reporting_downloader_posts_increasing_progress() -> None:
    """A tqdm-compatible fake reports its byte positions, and they only grow."""
    app = _Harness()

    def byte_downloader(model_name: str, *, tqdm_class=None, **kwargs: object) -> str:
        bar = tqdm_class(total=4, desc="bytes")
        bar.update(1)
        bar.update(1)
        bar.close()
        return "/models/reported"

    async with app.run_test(size=(100, 40)) as pilot:
        operation = app.download_worker("probe/model", byte_downloader)
        await _wait_for_download(operation, pilot)

    byte_events = [message for message in app.progress_events if message.total == 4]
    assert [message.done for message in byte_events] == [1, 2]
    assert all(message.label == "bytes" for message in byte_events)
    attempt_events = [
        message for message in app.progress_events if message.total == DOWNLOAD_MAX_TRY
    ]
    assert [message.done for message in attempt_events] == [1]
    assert app.download_events[0].path == "/models/reported"


async def test_an_attempt_only_downloader_still_reports_each_attempt() -> None:
    """Without tqdm_class the seam reports attempt-level progress, honestly."""
    app = _Harness()
    attempts: list[str] = []

    def attempt_only(model_name: str, **kwargs: object) -> str:
        attempts.append(model_name)
        return "/models/attempt"

    async with app.run_test(size=(100, 40)) as pilot:
        operation = app.download_worker("probe/model", attempt_only)
        await _wait_for_download(operation, pilot)

    assert attempts == ["probe/model"]
    attempts_seen = [m.done for m in app.progress_events if m.total == DOWNLOAD_MAX_TRY]
    assert attempts_seen == [1]
    assert app.progress_events[0].done == 0  # the "downloading…" opener


def test_resolve_hub_downloader_returns_that_hub_client() -> None:
    """Each hub resolves to its own snapshot_download, imported on demand."""
    hugging_face = workers_module._resolve_hub_downloader(Hub.HUGGINGFACE)

    assert hugging_face.__name__ == "snapshot_download"
    assert "huggingface_hub" in hugging_face.__module__

    model_scope = workers_module._resolve_hub_downloader(Hub.MODELSCOPE)

    assert model_scope.__name__ == "snapshot_download"
    assert "modelscope" in model_scope.__module__


def test_bind_progress_leaves_a_downloader_it_cannot_inspect(monkeypatch) -> None:
    """No readable signature, or no tqdm, means no binding: progress degrades."""
    unreadable = object()

    assert _bind_progress(unreadable, lambda done, total, label: None) is unreadable

    monkeypatch.setattr(workers_module, "_make_progress_reporter", lambda emit: None)

    def with_bytes(model_name: str, *, tqdm_class=None, **kwargs: object) -> str:
        return model_name

    assert _bind_progress(with_bytes, lambda done, total, label: None) is with_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 25-10 Task 3: the seam's failure and cancel paths, one test per worker
#
# Every worker has the same shape — core call, plain-language failure, cancel
# check — so each one gets the same pair of tests: a raising core must surface
# readable copy (never a class name), and a cancel that lands during the call
# must deliver nothing at all.
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY_MODEL = "zhangtaolab/plant-dnabert-BPE"


def _safe_prober(_model: str) -> AvailabilityReport:
    """A prober that answers UNKNOWN without touching a hub."""
    return AvailabilityReport(state=AvailabilityState.UNKNOWN)


def _cancelled_worker(monkeypatch) -> None:
    """Make every worker body start already cancelled (threads cannot be interrupted)."""
    cancelled = SimpleNamespace(cancelled_event=threading.Event(), is_cancelled=True)
    monkeypatch.setattr(workers_module, "get_current_worker", lambda: cancelled)


def _raising(monkeypatch, target, module, name: str) -> None:
    """Replace ``module.name`` with a callable that raises RuntimeError."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(target, name, boom)


async def _run(pilot, app: _Harness, start) -> None:
    """Start one worker, then let the seam settle.

    The download is not a Textual worker (WR-02), so its returned operation is
    waited for explicitly; every other worker is managed by Textual.
    """
    operation = start()
    await app.workers.wait_for_complete()
    await pilot.pause()
    if hasattr(operation, "is_alive"):
        await _wait_for_download(operation, pilot)


async def test_generate_config_failure_is_plain_language(monkeypatch) -> None:
    """A raising generate_config becomes read copy, never 'RuntimeError'."""
    import dnallm.preflight as facade

    _raising(monkeypatch, facade, "generate_config", "generate_config")
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.generate_config_worker(_REGISTRY_MODEL))

    assert app.config_events == []
    assert len(app.worker_failures) == 1
    assert "could not build this config" in app.worker_failures[0]
    assert "RuntimeError" not in app.worker_failures[0]


async def test_a_cancelled_generate_worker_delivers_nothing(monkeypatch) -> None:
    """A cancel during generation suppresses the result and the failure alike."""
    _cancelled_worker(monkeypatch)
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.generate_config_worker(_REGISTRY_MODEL))

    assert app.config_events == []
    assert app.worker_failures == []


async def test_edit_config_failure_is_plain_language(monkeypatch) -> None:
    """A raising edit_config becomes the same read sentence as generate."""
    import dnallm.preflight as facade

    _raising(monkeypatch, facade, "edit_config", "edit_config")
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.edit_config_worker({"task": {"task_type": "binary"}}))

    assert app.config_events == []
    assert len(app.worker_failures) == 1
    assert "could not build this config" in app.worker_failures[0]


async def test_a_cancelled_edit_worker_delivers_nothing(monkeypatch) -> None:
    """A cancel during a revalidation suppresses every UI effect."""
    _cancelled_worker(monkeypatch)
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.edit_config_worker({"task": {"task_type": "binary"}}))

    assert app.config_events == []
    assert app.worker_failures == []


async def test_a_save_failure_names_the_path_not_the_model(tmp_path) -> None:
    """WR-06: a write that fails on the output path gets save-specific copy.

    The generic sentence tells the user to check the model name and that the
    config file exists; neither has anything to do with an unwritable save
    location, and SC3 depends on the user knowing the file was not written.
    """
    app = _Harness()
    out_path = tmp_path / "no-such-folder" / "config.yaml"

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(
            pilot,
            app,
            lambda: app.edit_config_worker(
                {"task": {"task_type": "binary"}}, {}, output_path=str(out_path)
            ),
        )

    assert app.config_events == []
    assert len(app.worker_failures) == 1
    message = app.worker_failures[0]
    assert str(out_path) in message
    assert "folder" in message
    assert "model name" not in message
    assert "OSError" not in message
    assert not out_path.exists()


async def test_check_compatibility_failure_is_plain_language(monkeypatch) -> None:
    """A raising suitability check says the data is safe and retry is open."""
    import dnallm.preflight as facade

    _raising(monkeypatch, facade, "check_compatibility", "check_compatibility")
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(
            pilot,
            app,
            lambda: app.check_compatibility_worker(None, intent="finetune", prober=_safe_prober),
        )

    assert app.compatibility_events == []
    assert len(app.worker_failures) == 1
    assert "could not work out which models fit" in app.worker_failures[0]
    assert "RuntimeError" not in app.worker_failures[0]


async def test_a_cancelled_compatibility_worker_delivers_nothing(monkeypatch) -> None:
    """A cancel during the suitability check suppresses the report."""
    _cancelled_worker(monkeypatch)
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(
            pilot,
            app,
            lambda: app.check_compatibility_worker(
                None, intent="finetune", prober=_safe_prober, probe_shortlist=1
            ),
        )

    assert app.compatibility_events == []
    assert app.worker_failures == []


async def test_local_availability_failure_is_plain_language(monkeypatch) -> None:
    """A raising local pass reports the same suitability sentence."""
    import dnallm.preflight as facade

    _raising(monkeypatch, facade, "local_availability", "local_availability")
    report = facade.check_compatibility(
        None, intent="finetune", prober=_safe_prober, probe_shortlist=1
    )
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.local_availability_batch_worker(report))

    assert app.compatibility_events == []
    assert len(app.worker_failures) == 1
    assert "could not work out which models fit" in app.worker_failures[0]


async def test_a_cancelled_local_batch_delivers_nothing(monkeypatch) -> None:
    """A cancel during the offline local pass suppresses the batch."""
    _cancelled_worker(monkeypatch)
    import dnallm.preflight as facade

    report = facade.check_compatibility(
        None, intent="finetune", prober=_safe_prober, probe_shortlist=1
    )
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.local_availability_batch_worker(report))

    assert app.compatibility_events == []
    assert app.worker_failures == []


async def test_a_cancelled_on_demand_probe_delivers_nothing(monkeypatch) -> None:
    """A cancel during the single-row probe suppresses its update."""
    _cancelled_worker(monkeypatch)
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.probe_availability_worker("owner/model", _safe_prober))

    assert app.availability_events == []


async def test_a_failed_on_demand_probe_reads_plainly(monkeypatch) -> None:
    """WR-05: a raising probe posts readable copy instead of staying silent.

    Every other worker has this guard; without it the row the user pressed
    ``p`` on would stay on "probing…" forever with nothing else on screen.
    """
    import dnallm.preflight.availability as availability_module

    _raising(monkeypatch, availability_module, "probe_availability", "probe_availability")
    app = _Harness()

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.probe_availability_worker("owner/model"))

    assert app.availability_events == []
    assert len(app.worker_failures) == 1
    assert "could not work out which models fit" in app.worker_failures[0]
    assert "RuntimeError" not in app.worker_failures[0]


async def test_a_download_with_no_downloader_resolves_the_chosen_hub(monkeypatch) -> None:
    """With no explicit downloader the seam picks a hub and resolves its client."""
    resolved: list[Hub] = []

    def fake_resolver(hub: Hub):
        resolved.append(hub)

        def fake_downloader(model_name: str, **_kwargs: object) -> str:
            return f"/models/{model_name}"

        return fake_downloader

    monkeypatch.setattr(workers_module, "_resolve_hub_downloader", fake_resolver)
    app = _Harness()  # no ``downloader`` seam, exactly like a production app run

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(
            pilot,
            app,
            lambda: app.download_worker("owner/lab-model", None, entry=_entry(is_lab=True)),
        )

    # A lab entry with no confirmed hub: ModelScope first (D-11 / PFC-07).
    assert resolved == [Hub.MODELSCOPE]
    assert app.download_events[0].path == "/models/owner/lab-model"
    assert app.download_events[0].error is None


async def test_an_unexpected_download_failure_is_plain_language(monkeypatch) -> None:
    """A non-ValueError failure is still one readable sentence, with no class name."""
    import dnallm.models.model as model_module

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(model_module, "download_model", boom)
    app = _Harness()

    def fake_downloader(model_name: str, **_kwargs: object) -> str:
        return "/models/never-used"

    async with app.run_test(size=(100, 40)) as pilot:
        await _run(pilot, app, lambda: app.download_worker("probe/model", fake_downloader))

    assert len(app.download_events) == 1
    finished = app.download_events[0]
    assert finished.path is None
    assert finished.cancelled is False
    assert "did not finish" in finished.error
    assert "RuntimeError" not in finished.error
