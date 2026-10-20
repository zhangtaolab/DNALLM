"""Behavior suite for probe_availability: availability probing (23-03).

Task 1 covers the offline half (PFC-03/D-12): the three-way local cache
detection (local path isdir, ModelScope dual-layout isdir, HF cache scan),
the in-process memo, and the effective-endpoint reporting. Every test runs
network-free: remote probing is isolated behind the ``_probe_remote`` module
seam or injected fakes, and hub-library functions are patched at the module
attribute level (the adapters import them lazily at call time).

Registry pins are live (Phase 22 data): entry name ``Plant DNABERT``
resolves to repo id ``zhangtaolab/plant-dnabert-BPE``.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

import dnallm.preflight.availability as availability
from dnallm.preflight.availability import (
    clear_probe_memo,
    effective_endpoints,
    modelscope_cache_candidates,
    probe_availability,
)
from dnallm.preflight.report import AvailabilityReport, AvailabilityState, Hub

# Live-registry fixtures (Phase 22 lane): the pretrained BPE base entry and
# its human-readable registry name resolve to the same repo id.
BPE_BASE_REPO = "zhangtaolab/plant-dnabert-BPE"
BPE_BASE_NAME = "Plant DNABERT"

# Repo ids guaranteed outside the packaged registry and off disk.
UNREGISTERED_HF_REPO = "preflight-hf-test/definitely-not-registered"
UNREGISTERED_MS_REPO = "org/definitely-not-registered"

CACHED = AvailabilityState.CACHED
UNKNOWN = AvailabilityState.UNKNOWN


@pytest.fixture(autouse=True)
def _isolated_probe_memo():
    """Empty the in-process probe memo around every test (D-12 test hook)."""
    clear_probe_memo()
    yield
    clear_probe_memo()


def _sentinel_report(reason: str = "sentinel") -> AvailabilityReport:
    """A report whose only job is identity: proves which path produced it."""
    return AvailabilityReport(state=UNKNOWN, reason=reason)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1a: local path detection (D-12 branch 1)
# ─────────────────────────────────────────────────────────────────────────────


def test_local_directory_input_returns_cached_local(monkeypatch, tmp_path) -> None:
    """An existing directory path probes as cached/local with zero remote work."""
    model_dir = tmp_path / "my-local-model"
    model_dir.mkdir()
    remote = Mock(return_value=_sentinel_report("remote"))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    report = probe_availability(str(model_dir))

    assert report.state == CACHED
    assert report.source_hub == Hub.LOCAL
    assert report.size_bytes is None
    assert report.reason is None
    remote.assert_not_called()


def test_local_nonexistent_path_falls_through_to_remote(monkeypatch, tmp_path) -> None:
    """A nonexistent local path never raises: it degrades to the remote probe."""
    missing = tmp_path / "definitely-not-on-disk"
    remote = Mock(return_value=_sentinel_report("remote-ran"))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    report = probe_availability(str(missing))

    assert report is remote.return_value
    remote.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1b: HF cache detection (D-12 branch 2, library-honoring)
# ─────────────────────────────────────────────────────────────────────────────


def test_local_hf_cache_snapshot_yields_cached_huggingface(monkeypatch) -> None:
    """A cached HF snapshot is detected via the hub's own cache helpers."""
    snapshot_path = "/hf-root/hub/models--org--name/snapshots/deadbeef/config.json"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=snapshot_path))
    scan = Mock(
        return_value=SimpleNamespace(
            repos=[SimpleNamespace(repo_id=UNREGISTERED_HF_REPO, size_on_disk=1234)]
        )
    )
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", scan)
    remote = Mock(return_value=_sentinel_report("remote"))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    report = probe_availability(UNREGISTERED_HF_REPO)

    assert report.state == CACHED
    assert report.source_hub == Hub.HUGGINGFACE
    assert report.size_bytes == 1234
    assert report.revision == "deadbeef"
    remote.assert_not_called()


def test_local_hf_cache_miss_and_sentinel_fall_through(monkeypatch) -> None:
    """Cache miss (None) and the negative-cache sentinel are both misses."""
    remote = Mock(return_value=_sentinel_report("remote-ran"))
    monkeypatch.setattr(availability, "_probe_remote", remote)
    for miss_value in (None, huggingface_hub._CACHED_NO_EXIST):
        clear_probe_memo()
        cache_lookup = Mock(return_value=miss_value)
        scan = Mock()
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cache_lookup)
        monkeypatch.setattr(huggingface_hub, "scan_cache_dir", scan)

        report = probe_availability(UNREGISTERED_HF_REPO)

        assert report is remote.return_value, f"miss value {miss_value!r} must fall through"
        scan.assert_not_called()
    remote.assert_called()


def test_local_hf_scan_failure_degrades_to_size_none(monkeypatch) -> None:
    """A raising scan_cache_dir only costs the size field, never the verdict."""
    snapshot_path = "/hf-root/hub/models--org--name/snapshots/feedface/config.json"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=snapshot_path))
    monkeypatch.setattr(
        huggingface_hub, "scan_cache_dir", Mock(side_effect=RuntimeError("scan broke"))
    )
    monkeypatch.setattr(
        availability, "_probe_remote", Mock(return_value=_sentinel_report("remote"))
    )

    report = probe_availability(UNREGISTERED_HF_REPO)

    assert report.state == CACHED
    assert report.source_hub == Hub.HUGGINGFACE
    assert report.size_bytes is None
    assert report.revision == "feedface"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1c: ModelScope dual-layout cache candidates (D-12 branch 3)
# ─────────────────────────────────────────────────────────────────────────────


def test_modelscope_cache_candidates_dual_layout_dots(monkeypatch, tmp_path) -> None:
    """Both MS cache layouts are produced, with dots mapped to triple underscores."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))

    candidates = modelscope_cache_candidates("org/name.with.dots")

    assert candidates == [
        f"{tmp_path}/models/org/name___with___dots",
        f"{tmp_path}/org/name___with___dots",
    ]


def test_local_modelscope_cache_dir_yields_cached_modelscope(monkeypatch, tmp_path) -> None:
    """An isdir MS candidate (current layout) probes cached with an exact du size."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    cached_dir = tmp_path / "models" / "org" / "name___with___dots"
    cached_dir.mkdir(parents=True)
    (cached_dir / "weights.bin").write_bytes(b"abc")  # 3 bytes
    (cached_dir / "config.json").write_bytes(b"abcd")  # 4 bytes
    remote = Mock(return_value=_sentinel_report("remote"))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    report = probe_availability("org/name.with.dots")

    assert report.state == CACHED
    assert report.source_hub == Hub.MODELSCOPE
    assert report.size_bytes == 7  # exact integer byte sum, no rounding
    assert report.revision is None
    remote.assert_not_called()


def test_local_modelscope_cache_legacy_layout_also_hits(monkeypatch, tmp_path) -> None:
    """The legacy pre-restructure layout is honored as the second candidate."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    legacy_dir = tmp_path / "org" / "name"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_bytes(b"abcdef")  # 6 bytes
    monkeypatch.setattr(
        availability, "_probe_remote", Mock(return_value=_sentinel_report("remote"))
    )

    report = probe_availability("org/name")

    assert report.state == CACHED
    assert report.source_hub == Hub.MODELSCOPE
    assert report.size_bytes == 6


# ─────────────────────────────────────────────────────────────────────────────
# Task 1d: in-process memo (D-12: same model never re-probed per process)
# ─────────────────────────────────────────────────────────────────────────────


def test_memo_second_probe_skips_resolver_and_remote(monkeypatch) -> None:
    """A second probe of the same id resolves and probes at most once."""
    sentinel = _sentinel_report("memoized")
    resolver = Mock(return_value=("memo/org", None))
    remote = Mock(return_value=sentinel)
    monkeypatch.setattr(availability, "_resolve_model_ref", resolver)
    monkeypatch.setattr(availability, "_probe_remote", remote)

    first = probe_availability("memo/model")
    second = probe_availability("memo/model")

    assert first is sentinel
    assert second is sentinel
    assert resolver.call_count == 1
    assert remote.call_count == 1


def test_memo_aliases_registry_name_and_repo_id_share_one_probe(monkeypatch) -> None:
    """A registry name and its repo id dedupe onto one remote probe."""
    # Local detection is isolated to keep the test environment-independent
    # (this box carries real hub caches; CI does not).
    monkeypatch.setattr(availability, "_probe_cached_local", Mock(return_value=None))
    sentinel = _sentinel_report("memoized")
    remote = Mock(return_value=sentinel)
    monkeypatch.setattr(availability, "_probe_remote", remote)

    by_repo = probe_availability(BPE_BASE_REPO)
    by_name = probe_availability(BPE_BASE_NAME)

    assert by_repo is sentinel
    assert by_name is sentinel
    assert remote.call_count == 1


def test_memo_clear_forces_reprobe(monkeypatch) -> None:
    """clear_probe_memo empties the memo so the next probe runs again."""
    remote = Mock(return_value=_sentinel_report("reprobe"))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    probe_availability("memo/model")
    clear_probe_memo()
    probe_availability("memo/model")

    assert remote.call_count == 2


def test_memo_concurrent_probes_resolve_once(monkeypatch) -> None:
    """8 concurrent probes of one uncached id resolve and probe at most once."""
    sentinel = _sentinel_report("concurrent")
    resolver = Mock(return_value=("concurrent/org", None))
    monkeypatch.setattr(availability, "_resolve_model_ref", resolver)
    prober = Mock(return_value=sentinel)

    barrier = threading.Barrier(8)
    results: list[AvailabilityReport] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        report = probe_availability("concurrent/model", prober=prober)
        with results_lock:
            results.append(report)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(report is sentinel for report in results)
    assert resolver.call_count == 1
    assert prober.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 1e: effective endpoint reporting (Spike 1: constants, never os.environ)
# ─────────────────────────────────────────────────────────────────────────────


def test_effective_endpoints_reports_library_constants(monkeypatch) -> None:
    """Endpoints come from library constants, never from a lying env var."""
    import huggingface_hub.constants as hf_constants
    from modelscope.utils.file_utils import get_modelscope_cache_dir

    monkeypatch.setenv("HF_ENDPOINT", "https://env-vars-lie.example.com")

    endpoints = effective_endpoints()

    assert endpoints["huggingface"] == hf_constants.ENDPOINT
    assert "modelscope_cache" in endpoints
    assert endpoints["modelscope_cache"] == get_modelscope_cache_dir()
