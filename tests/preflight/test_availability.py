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

import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

import dnallm.preflight.availability as availability
from dnallm.preflight.availability import (
    DEFAULT_PROBE_TIMEOUT,
    clear_probe_memo,
    effective_endpoints,
    modelscope_cache_candidates,
    probe_availability,
)
from dnallm.preflight.report import (
    AvailabilityReasonCode,
    AvailabilityReport,
    AvailabilityState,
    Hub,
)

# The modelscope package is a lazy module whose attribute access can raise
# ("modelscope.hub" resolves through its lazy loader); bind the hub-api
# submodule through importlib so patching targets the module object itself.
ms_hub_api = importlib.import_module("modelscope.hub.api")

# Live-registry fixtures (Phase 22 lane): the pretrained BPE base entry and
# its human-readable registry name resolve to the same repo id.
BPE_BASE_REPO = "zhangtaolab/plant-dnabert-BPE"
BPE_BASE_NAME = "Plant DNABERT"

# Repo ids guaranteed outside the packaged registry and off disk.
UNREGISTERED_HF_REPO = "preflight-hf-test/definitely-not-registered"
UNREGISTERED_MS_REPO = "org/definitely-not-registered"

CACHED = AvailabilityState.CACHED
REMOTE_CONFIRMED = AvailabilityState.REMOTE_CONFIRMED
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


def _confirmed_report(hub: Hub, size: int = 100, revision: str = "rev-1") -> AvailabilityReport:
    """A canned remote-confirmation report for one hub."""
    return AvailabilityReport(
        state=REMOTE_CONFIRMED, size_bytes=size, revision=revision, source_hub=hub
    )


def _hf_error(exc_cls, status_code: int | None, message: str):
    """Construct a real huggingface_hub error carrying a fake response."""
    response = (
        None
        if status_code is None
        else SimpleNamespace(status_code=status_code, headers={}, request=None)
    )
    return exc_cls(message, response=response)


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

    # Path-joined, not f-string-with-"/": the implementation builds candidates
    # with os.path.join (platform-native separators for the isdir probes), and
    # a forward-slash expectation fails on Windows where the joined paths use
    # backslashes.
    assert candidates == [
        str(tmp_path / "models" / "org" / "name___with___dots"),
        str(tmp_path / "org" / "name___with___dots"),
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

    by_name = probe_availability(BPE_BASE_NAME)
    by_repo = probe_availability(BPE_BASE_REPO)
    # Both memo key directions: the name probe stores the alias entry, the
    # repo probe hits the raw fast path; a fresh repo-first probe then lets
    # the name probe hit the resolved-repo-id memo entry.
    clear_probe_memo()
    by_repo_first = probe_availability(BPE_BASE_REPO)
    by_name_again = probe_availability(BPE_BASE_NAME)

    assert by_name is sentinel
    assert by_repo is sentinel
    assert by_repo_first is sentinel
    assert by_name_again is sentinel
    assert remote.call_count == 2


def test_memo_clear_forces_reprobe(monkeypatch) -> None:
    """clear_probe_memo empties the memo so the next probe runs again."""
    remote = Mock(return_value=_sentinel_report("reprobe"))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    probe_availability("memo/model")
    clear_probe_memo()
    probe_availability("memo/model")

    assert remote.call_count == 2


def test_memo_concurrent_probes_resolve_once(monkeypatch) -> None:
    """8 concurrent DEFAULT probes of one uncached id resolve and probe at most once.

    Uses the default seam (patched _probe_remote), not the ``prober``
    kwarg: injected-prober results never enter the memo (WR-03), so the
    memo-based dedup contract is pinned on the default flow it serves.
    """
    sentinel = _sentinel_report("concurrent")
    resolver = Mock(return_value=("concurrent/org", None))
    remote = Mock(return_value=sentinel)
    monkeypatch.setattr(availability, "_resolve_model_ref", resolver)
    monkeypatch.setattr(availability, "_probe_cached_local", Mock(return_value=None))
    monkeypatch.setattr(availability, "_probe_remote", remote)

    barrier = threading.Barrier(8)
    results: list[AvailabilityReport] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        report = probe_availability("concurrent/model")
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
    assert remote.call_count == 1


def test_injected_prober_result_never_poisons_default_probe(monkeypatch) -> None:
    """A fake prober's report never enters the memo (WR-03).

    A subsequent DEFAULT probe of the same model must run the default flow
    and return its own report — not the injected fake.
    """
    monkeypatch.setattr(availability, "_probe_cached_local", Mock(return_value=None))
    fake_report = _confirmed_report(Hub.MODELSCOPE, size=999, revision="fake")
    default_report = _sentinel_report("default-ran")
    remote = Mock(return_value=default_report)
    monkeypatch.setattr(availability, "_probe_remote", remote)

    injected = probe_availability("memo-poison/org", prober=Mock(return_value=fake_report))
    default = probe_availability("memo-poison/org")

    assert injected is fake_report
    assert default is default_report  # not the memoized fake report
    assert remote.call_count == 1


def test_two_injected_probers_each_get_their_own_report(monkeypatch) -> None:
    """Two different injected probers across calls each get their own report
    back — the first call's result is not served to the second (WR-03)."""
    monkeypatch.setattr(availability, "_probe_cached_local", Mock(return_value=None))
    monkeypatch.setattr(
        availability, "_probe_remote", Mock(return_value=_sentinel_report("default"))
    )
    first = _confirmed_report(Hub.MODELSCOPE, size=1)
    second = _confirmed_report(Hub.HUGGINGFACE, size=2)

    assert probe_availability("seam-a/org", prober=Mock(return_value=first)) is first
    assert probe_availability("seam-a/org", prober=Mock(return_value=second)) is second


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


# ─────────────────────────────────────────────────────────────────────────────
# Task 2a: remote fallback chain — ModelScope first, HF fallback (D-11)
# ─────────────────────────────────────────────────────────────────────────────


def test_remote_ms_confirms_and_hf_is_never_called(monkeypatch) -> None:
    """MS confirms existence; the HF adapter is never invoked (D-11 order)."""
    order: list[str] = []
    ms = Mock(side_effect=lambda rid: (order.append("ms"), _confirmed_report(Hub.MODELSCOPE))[1])
    hf = Mock(side_effect=lambda rid: (order.append("hf"), None)[1])
    monkeypatch.setattr(availability, "_ms_adapter", ms)
    monkeypatch.setattr(availability, "_hf_adapter", hf)

    report = probe_availability("order-ms-first/org")

    assert order == ["ms"]
    assert hf.call_count == 0
    assert report.state == REMOTE_CONFIRMED
    assert report.source_hub == Hub.MODELSCOPE


def test_remote_ms_clean_miss_falls_back_to_hf(monkeypatch) -> None:
    """A clean MS miss (falsy repo_exists) probes HF next and confirms there."""
    order: list[str] = []
    ms = Mock(side_effect=lambda rid: (order.append("ms"), None)[1])
    hf = Mock(side_effect=lambda rid: (order.append("hf"), _confirmed_report(Hub.HUGGINGFACE))[1])
    monkeypatch.setattr(availability, "_ms_adapter", ms)
    monkeypatch.setattr(availability, "_hf_adapter", hf)

    report = probe_availability("order-fallback/org")

    assert order == ["ms", "hf"]
    assert report.state == REMOTE_CONFIRMED
    assert report.source_hub == Hub.HUGGINGFACE


def test_remote_ms_adapter_body_exact_size_and_revision(monkeypatch) -> None:
    """The MS adapter sums the listing's Size fields exactly and reads Revision."""

    class FakeHubApi:
        def repo_exists(self, repo_id, **kwargs):
            return True

        def get_model_files(self, repo_id, **kwargs):
            return [
                {"Name": "weights.bin", "Size": 100, "Revision": "v1.2.0"},
                {"Name": "config.json", "Size": 23, "Revision": "v1.2.0"},
            ]

    monkeypatch.setattr(ms_hub_api, "HubApi", FakeHubApi)

    report = probe_availability("ms-adapter-body/org")

    assert report.state == REMOTE_CONFIRMED
    assert report.source_hub == Hub.MODELSCOPE
    assert report.size_bytes == 123  # exact integer sum, no rounding
    assert report.revision == "v1.2.0"
    assert report.reason is None


def test_remote_hf_adapter_body_exact_size_and_revision(monkeypatch) -> None:
    """The HF adapter sums sibling sizes exactly and reports info.sha."""
    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))

    class FakeHfApi:
        def __init__(self, endpoint=None):
            self.endpoint = endpoint

        def model_info(self, repo_id, files_metadata=True):
            return SimpleNamespace(
                sha="rev-abc",
                siblings=[
                    SimpleNamespace(rfilename="weights.bin", size=100),
                    SimpleNamespace(rfilename="config.json", size=23),
                    SimpleNamespace(rfilename="unknown-size.bin", size=None),
                ],
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    report = probe_availability("hf-adapter-body/org")

    assert report.state == REMOTE_CONFIRMED
    assert report.source_hub == Hub.HUGGINGFACE
    assert report.size_bytes == 123  # exact integer sum; None sizes skipped
    assert report.revision == "rev-abc"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2b: failure taxonomy — every failure degrades to unknown + reason
# (PFC-03/PFC-06: a transient failure can never masquerade as unavailable)
# ─────────────────────────────────────────────────────────────────────────────


def test_remote_both_hubs_clean_miss_is_not_found_on_either(monkeypatch) -> None:
    """Clean misses on both hubs conclude NOT_FOUND_ON_EITHER_HUB (unknown)."""
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))
    monkeypatch.setattr(
        availability, "_hf_adapter", Mock(side_effect=_hf_error(RepositoryNotFoundError, 404, "no"))
    )

    report = probe_availability("taxonomy/both-miss")

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.NOT_FOUND_ON_EITHER_HUB.value


def test_remote_gated_repo_is_a_clean_miss_not_unknown(monkeypatch) -> None:
    """GatedRepoError on one hub means 'try the other hub', never an error."""
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))
    monkeypatch.setattr(
        availability, "_hf_adapter", Mock(side_effect=_hf_error(GatedRepoError, 403, "gated"))
    )

    report = probe_availability("taxonomy/gated")

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.NOT_FOUND_ON_EITHER_HUB.value


def test_remote_hf_rate_limited_maps_to_unknown_rate_limited(monkeypatch) -> None:
    """A 429 HfHubHTTPError degrades to unknown/RATE_LIMITED — never unavailable."""
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))
    monkeypatch.setattr(
        availability,
        "_hf_adapter",
        Mock(side_effect=_hf_error(HfHubHTTPError, 429, "too many requests")),
    )

    report = probe_availability("taxonomy/rate-limited")

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.RATE_LIMITED.value


def test_remote_hf_http_error_other_status_is_probe_error(monkeypatch) -> None:
    """Non-429 HTTP errors carry the exception class name under PROBE_ERROR."""
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))
    monkeypatch.setattr(
        availability,
        "_hf_adapter",
        Mock(side_effect=_hf_error(HfHubHTTPError, 503, "server error")),
    )

    report = probe_availability("taxonomy/http-503")

    assert report.state == UNKNOWN
    assert report.reason == f"{AvailabilityReasonCode.PROBE_ERROR.value}: HfHubHTTPError"


def test_remote_offline_mode_maps_to_unknown_offline(monkeypatch) -> None:
    """OfflineModeIsEnabled degrades to unknown/OFFLINE."""
    from huggingface_hub.errors import OfflineModeIsEnabled

    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))
    monkeypatch.setattr(
        availability, "_hf_adapter", Mock(side_effect=OfflineModeIsEnabled("offline mode"))
    )

    report = probe_availability("taxonomy/offline")

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.OFFLINE.value


def test_remote_connection_error_carries_class_name(monkeypatch) -> None:
    """A plain ConnectionError degrades to unknown with the class name in reason."""
    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))
    monkeypatch.setattr(availability, "_hf_adapter", Mock(side_effect=ConnectionError("dns broke")))

    report = probe_availability("taxonomy/connection")

    assert report.state == UNKNOWN
    assert report.reason == (f"{AvailabilityReasonCode.PROBE_ERROR.value}: ConnectionError")


def test_remote_ms_error_falls_through_to_hf(monkeypatch) -> None:
    """A raising MS hub never short-circuits the probe: HF fallback runs."""
    order: list[str] = []

    def raising_ms(rid):
        order.append("ms")
        raise RuntimeError("modelscope exploded")

    def confirming_hf(rid):
        order.append("hf")
        return _confirmed_report(Hub.HUGGINGFACE)

    monkeypatch.setattr(availability, "_ms_adapter", raising_ms)
    monkeypatch.setattr(availability, "_hf_adapter", confirming_hf)

    report = probe_availability("taxonomy/ms-error")

    assert order == ["ms", "hf"]
    assert report.state == REMOTE_CONFIRMED
    assert report.source_hub == Hub.HUGGINGFACE


def test_remote_ms_error_and_hf_miss_reports_ms_error_class(monkeypatch) -> None:
    """When HF also misses cleanly, the remembered MS error names the reason."""
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.setattr(
        availability, "_ms_adapter", Mock(side_effect=RuntimeError("modelscope exploded"))
    )
    monkeypatch.setattr(
        availability,
        "_hf_adapter",
        Mock(side_effect=_hf_error(RepositoryNotFoundError, 404, "no")),
    )

    report = probe_availability("taxonomy/ms-error-hf-miss")

    assert report.state == UNKNOWN
    assert report.reason == f"{AvailabilityReasonCode.PROBE_ERROR.value}: RuntimeError"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2c: bounded wait + the injectable prober seam (PFC-03)
# ─────────────────────────────────────────────────────────────────────────────


def test_remote_timeout_returns_probe_timeout_within_bound(monkeypatch) -> None:
    """A call at or beyond the bounded wait degrades to unknown/PROBE_TIMEOUT."""
    monkeypatch.setattr(
        availability, "_hf_adapter", Mock(return_value=_confirmed_report(Hub.HUGGINGFACE))
    )

    def slow_ms(rid):
        time.sleep(0.4)
        return None

    monkeypatch.setattr(availability, "_ms_adapter", slow_ms)

    started = time.monotonic()
    report = probe_availability("timeout/org", timeout=0.1)
    elapsed = time.monotonic() - started

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.PROBE_TIMEOUT.value
    assert elapsed < 0.35  # the guard fired at the bound, not after the sleep


def test_remote_timeout_hf_branch_returns_probe_timeout(monkeypatch) -> None:
    """The HF-side bounded wait fires the same PROBE_TIMEOUT degradation."""
    monkeypatch.setattr(availability, "_ms_adapter", Mock(return_value=None))

    def slow_hf(rid):
        time.sleep(0.4)
        raise ConnectionError("never reached in time")

    monkeypatch.setattr(availability, "_hf_adapter", slow_hf)

    report = probe_availability("timeout-hf/org", timeout=0.1)

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.PROBE_TIMEOUT.value


def test_remote_injected_prober_bypasses_adapters(monkeypatch) -> None:
    """The injected prober's report is returned verbatim; adapters never run."""
    verbatim = _confirmed_report(Hub.MODELSCOPE, size=10, revision="x")
    ms = Mock(return_value=_confirmed_report(Hub.MODELSCOPE))
    hf = Mock(return_value=_confirmed_report(Hub.HUGGINGFACE))
    monkeypatch.setattr(availability, "_ms_adapter", ms)
    monkeypatch.setattr(availability, "_hf_adapter", hf)
    seen: list[str] = []

    def prober(repo_id):
        seen.append(repo_id)
        return verbatim

    report = probe_availability("seam/org", prober=prober, timeout=1.0)

    assert report is verbatim
    assert ms.call_count == 0
    assert hf.call_count == 0
    assert seen == ["seam/org"]


def test_remote_injected_prober_raising_degrades_to_unknown(monkeypatch) -> None:
    """A raising injected prober still never escapes (PFC-06 never-raise)."""

    def bad_prober(repo_id):
        raise ValueError("bad fake")

    report = probe_availability("seam-raising/org", prober=bad_prober, timeout=1.0)

    assert report.state == UNKNOWN
    assert report.reason == f"{AvailabilityReasonCode.PROBE_ERROR.value}: ValueError"


def test_default_probe_timeout_is_five_seconds() -> None:
    """The configurable default bound is 5.0 s (D-12)."""
    assert DEFAULT_PROBE_TIMEOUT == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Task 3a: degraded-mode proofs (PFC-06 / ROADMAP SC-5) — network severed
# ─────────────────────────────────────────────────────────────────────────────


def _neutralize_hub_caches(monkeypatch, tmp_path) -> None:
    """Point both hub cache detectors at an empty world (hermetic probes)."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=None))


def _raising_adapters(monkeypatch) -> None:
    """Patch both remote adapters to raise — the network is severed."""
    monkeypatch.setattr(availability, "_ms_adapter", Mock(side_effect=RuntimeError("network down")))
    monkeypatch.setattr(availability, "_hf_adapter", Mock(side_effect=RuntimeError("network down")))


def test_degraded_cached_local_id_with_raising_adapters(monkeypatch, tmp_path) -> None:
    """With the network severed, a locally cached id still probes cached."""
    _neutralize_hub_caches(monkeypatch, tmp_path)
    _raising_adapters(monkeypatch)
    model_dir = tmp_path / "offline-model"
    model_dir.mkdir()

    report = probe_availability(str(model_dir))

    assert report.state == CACHED
    assert report.source_hub == Hub.LOCAL


def test_degraded_uncached_id_returns_unknown_with_reason(monkeypatch, tmp_path) -> None:
    """With the network severed, an uncached id degrades to unknown, no raise."""
    _neutralize_hub_caches(monkeypatch, tmp_path)
    _raising_adapters(monkeypatch)

    report = probe_availability("degraded/never-cached")

    assert report.state == UNKNOWN
    assert report.reason == f"{AvailabilityReasonCode.PROBE_ERROR.value}: RuntimeError"


def test_degraded_registry_entry_name_resolves_and_degrades(monkeypatch, tmp_path) -> None:
    """A registry entry name resolves through try_resolve, then degrades."""
    _neutralize_hub_caches(monkeypatch, tmp_path)
    _raising_adapters(monkeypatch)
    real_try_resolve = availability.try_resolve
    resolve_spy = Mock(side_effect=real_try_resolve)
    monkeypatch.setattr(availability, "try_resolve", resolve_spy)

    report = probe_availability(BPE_BASE_NAME)

    resolve_spy.assert_called_once_with(BPE_BASE_NAME)
    assert report.state == UNKNOWN
    assert report.reason is not None


# ─────────────────────────────────────────────────────────────────────────────
# Task 3b: coverage completion — adapter-miss body, revision fallback,
# injected-prober timeout
# ─────────────────────────────────────────────────────────────────────────────


def test_remote_ms_adapter_body_clean_miss_returns_none(monkeypatch) -> None:
    """A falsy repo_exists is a clean miss: the chain proceeds to HF."""

    class FakeHubApi:
        def repo_exists(self, repo_id, **kwargs):
            return False

    monkeypatch.setattr(ms_hub_api, "HubApi", FakeHubApi)
    monkeypatch.setattr(
        availability, "_hf_adapter", Mock(return_value=_confirmed_report(Hub.HUGGINGFACE))
    )

    report = probe_availability("ms-clean-miss/org")

    assert report.state == REMOTE_CONFIRMED
    assert report.source_hub == Hub.HUGGINGFACE


def test_local_hf_cache_non_snapshot_path_revision_none(monkeypatch) -> None:
    """A cached path without a snapshots segment yields revision None."""
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        Mock(return_value="/hf-root/hub/models--org--name/config.json"),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "scan_cache_dir",
        Mock(side_effect=RuntimeError("scan unavailable")),
    )

    report = probe_availability(UNREGISTERED_HF_REPO)

    assert report.state == CACHED
    assert report.source_hub == Hub.HUGGINGFACE
    assert report.revision is None


def test_timeout_injected_prober_returns_probe_timeout(monkeypatch) -> None:
    """An over-slow injected prober hits the same bound (the seam contract)."""

    def slow_prober(repo_id):
        time.sleep(0.4)
        return _confirmed_report(Hub.MODELSCOPE)

    started = time.monotonic()
    report = probe_availability("seam-timeout/org", prober=slow_prober, timeout=0.1)
    elapsed = time.monotonic() - started

    assert report.state == UNKNOWN
    assert report.reason == AvailabilityReasonCode.PROBE_TIMEOUT.value
    assert elapsed < 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Task 3c: real-network acceptance (slow lane, exactly one network test).
# Deselected locally via -m "not slow"; exercised by CI's slow leg. The typed
# skip prefix matches tests/expected_skips.yaml's existing network entry.
# WR-04: the reachability check runs INSIDE the test body (lazy skip), never
# at collection time, and uses only the stdlib — no undeclared httpx import.
# ─────────────────────────────────────────────────────────────────────────────

_SLOW_LANE_MODEL = "zhangtaolab/plant-dnabert-BPE"  # dual-homed lab model


def _any_model_hub_reachable(timeout: float = 5.0) -> bool:
    """True when at least one of the two probed hubs answers at all.

    stdlib ``urllib.request`` only (WR-04: httpx is not a declared test
    dependency); any HTTP response at all — including a 4xx/5xx — proves
    the host is reachable.
    """
    import urllib.error
    import urllib.request

    for url in ("https://www.modelscope.cn", "https://huggingface.co"):
        try:
            urllib.request.urlopen(url, timeout=timeout)  # ruff: ignore[suspicious-url-open-usage]  # fixed https constants only
            return True
        except urllib.error.HTTPError:
            return True  # an HTTP error response still proves reachability
        except OSError:
            continue
    return False


@pytest.mark.slow
def test_slow_real_network_dual_homed_lab_model() -> None:
    """Probe a real dual-homed lab model end-to-end; the report is well-formed.

    Local caches short-circuit when present (this box caches the lab
    repos); otherwise the real adapters run under a generous 30 s bound.
    The only hard assertions: no exception, a three-value state, and a
    positive exact integer size whenever the probe confirms remotely.
    """
    if not _any_model_hub_reachable():
        pytest.skip("network-unavailable: no model hub reachable (modelscope.cn / huggingface.co)")
    report = probe_availability(_SLOW_LANE_MODEL, timeout=30.0)

    assert report.state in (CACHED, REMOTE_CONFIRMED, UNKNOWN)
    if report.state == REMOTE_CONFIRMED:
        assert isinstance(report.size_bytes, int)
        assert report.size_bytes > 0


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: local_availability — the offline bulk path for every listing row
# (D-04, 25-07). Local cache detection only: no hub client, no memo, no raise.
# ─────────────────────────────────────────────────────────────────────────────


def test_local_availability_existing_directory_is_cached_local(monkeypatch, tmp_path) -> None:
    """An existing directory reads cached/local; no hub client is touched."""
    model_dir = tmp_path / "bulk-local-model"
    model_dir.mkdir()
    remote = Mock(side_effect=AssertionError("local_availability must never probe"))
    hub_scan = Mock(side_effect=AssertionError("local_availability must not scan hubs"))
    monkeypatch.setattr(availability, "_probe_remote", remote)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", hub_scan)

    report = availability.local_availability(str(model_dir))

    assert report.state == CACHED
    assert report.source_hub == Hub.LOCAL
    assert report.reason is None
    remote.assert_not_called()
    hub_scan.assert_not_called()


def test_local_availability_modelscope_layout_reports_exact_size(monkeypatch, tmp_path) -> None:
    """A repo in the ModelScope dual layout reads cached with its on-disk size."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    cached_dir = tmp_path / "models" / "org" / "name___with___dots"
    cached_dir.mkdir(parents=True)
    (cached_dir / "weights.bin").write_bytes(b"abc")  # 3 bytes

    report = availability.local_availability("org/name.with.dots")

    assert report.state == CACHED
    assert report.source_hub == Hub.MODELSCOPE
    assert report.size_bytes == 3


def test_local_availability_uncached_path_like_id_is_unknown_with_no_reason(
    monkeypatch, tmp_path
) -> None:
    """Nothing cached is UNKNOWN with no reason — never a raise, never a verdict."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=None))

    report = availability.local_availability(str(tmp_path / "not-on-disk"))

    assert report.state == UNKNOWN
    assert report.reason is None
    assert report.source_hub is None


def test_local_availability_degrades_instead_of_raising(monkeypatch) -> None:
    """A failing local detector is UNKNOWN with no reason, never an exception."""
    monkeypatch.setattr(
        availability, "_probe_cached_local", Mock(side_effect=OSError("unreadable path"))
    )

    report = availability.local_availability("broken/org")

    assert report.state == UNKNOWN
    assert report.reason is None


def test_local_availability_never_memoizes_and_never_probes(monkeypatch, tmp_path) -> None:
    """Every call re-detects, and the remote probe memo stays untouched."""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=None))
    detector = Mock(return_value=None)
    monkeypatch.setattr(availability, "_probe_cached_local", detector)
    monkeypatch.setattr(
        availability,
        "_probe_remote",
        Mock(side_effect=AssertionError("local_availability must never probe")),
    )

    first = availability.local_availability("bulk/org")
    second = availability.local_availability("bulk/org")

    assert first.state == UNKNOWN
    assert second.state == UNKNOWN
    assert detector.call_count == 2  # not memoized: every row is classified fresh
    assert availability._MEMO == {}  # the remote-probe memo is never written here


def test_local_availability_is_exported_from_the_facade() -> None:
    """The offline bulk path is part of the documented preflight API."""
    import dnallm.preflight as preflight

    assert "local_availability" in preflight.__all__
    assert preflight.local_availability is availability.local_availability
