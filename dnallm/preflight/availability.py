"""probe_availability: side-effect-free model availability probing (PFC-03).

Design invariants:

1. **No weights, ever.** Probing never downloads: local cache detection is
   pure path/scan checks (directory ``isdir``, the HF cache scan helpers,
   ModelScope dual-layout candidates) and remote detection asks each hub
   only for existence plus file listings. The ModelScope snapshot fetcher
   is deliberately NOT used for detection — on a cache miss it writes a
   marker file into the cache (a side effect a read-only probe must never
   have); actual downloads belong to the later select-to-download flow.
2. **ModelScope-first (D-11), via the lab-model proxy.** Registry entries
   carry no ``source`` field (the schema is frozen), so the documented
   stand-in is :func:`dnallm.models.registry.is_lab_model`: every
   ``zhangtaolab/*`` repo is dual-homed and ModelScope is the
   owner-directed primary source. The remote fallback chain therefore
   probes ModelScope first for EVERY entry and falls back to Hugging Face;
   when both hubs carry a repo, ``source_hub`` reports modelscope.
3. **Failure is UNKNOWN (PFC-06).** Rate limits, offline mode, timeouts,
   and every other probe failure map to ``state=unknown`` with an
   attached reason — never to an "unavailable" verdict and never to an
   exception: :func:`probe_availability` cannot raise.
4. **The core owns the time bound.** The hub libraries expose no
   per-call timeouts, so every remote call runs under a bounded wait on a
   shared executor; an abandoned call finishes naturally in the
   background. The default bound is :data:`DEFAULT_PROBE_TIMEOUT` seconds.
5. **Hub clients live only here, lazily.** This is the one module under
   ``dnallm/preflight/`` permitted to import ``huggingface_hub`` /
   ``modelscope`` (enforced by ``tests/preflight/test_structure.py``);
   imports sit inside the adapter functions so this module's own source
   pulls no hub client at import time, and the unit suite stays
   network-free via the injectable prober seam (PFC-03). Runtime
   ``sys.modules`` purity (the parent package inits' transitive imports)
   is a known open constraint deferred to Phase 25.
6. **Endpoints are reported, not assumed.** ``effective_endpoints``
   reads library constants (``huggingface_hub.constants.ENDPOINT`` and
   the resolved ModelScope cache root) — never ``os.environ``, which can
   lie relative to what the libraries actually resolved at import.

Example:
    from dnallm.preflight.availability import probe_availability

    report = probe_availability("zhangtaolab/plant-dnabert-BPE")
    # AvailabilityReport(state='cached' | 'remote_confirmed' | 'unknown', ...)
"""

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from ..models.registry import RegistryEntry, is_lab_model, try_resolve
from ..utils import get_logger
from .report import (
    AvailabilityReasonCode,
    AvailabilityReport,
    AvailabilityState,
    Hub,
)

logger = get_logger("dnallm.preflight.availability")

__all__ = [
    "DEFAULT_PROBE_TIMEOUT",
    "clear_probe_memo",
    "effective_endpoints",
    "modelscope_cache_candidates",
    "probe_availability",
]

#: Default bounded wait for one remote hub call, in seconds (D-12).
DEFAULT_PROBE_TIMEOUT = 5.0

#: The file whose presence in the HF snapshot proves the repo is cached.
_HF_CACHE_PROBE_FILENAME = "config.json"

# In-process memo (D-12): the same model is never re-probed within a
# process, by DEFAULT probes only (WR-03: injected-prober results never
# enter the memo — see probe_availability's prober arg). Keyed by resolved
# repo id AND by every raw input alias that resolved to it; registry data
# is immutable, so no cross-process persistence is needed. Guarded by
# _MEMO_LOCK; the probe itself runs under the lock so concurrent default
# probes of one id resolve and probe at most once (lock-guarded
# double-check).
_MEMO: dict[str, AvailabilityReport] = {}
_MEMO_LOCK = threading.Lock()

# Shared bounded-wait executor: submitting is cheap, worker threads spawn
# on demand, and timed-out calls are simply abandoned here (they finish
# naturally; no per-call thread spawn, no cancellation games).
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dnallm-preflight-probe")

T = TypeVar("T")


def probe_availability(
    model: str,
    *,
    prober: Callable[[str], AvailabilityReport] | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> AvailabilityReport:
    """Probe one model's availability without downloading anything (PFC-03).

    Resolution order (D-12 three-way): an existing directory on disk, the
    local hub caches, then the remote fallback chain (ModelScope first,
    Hugging Face fallback — see module invariant 2 for the is_lab_model
    proxy standing in for the absent registry ``source`` field).

    Args:
        model: Registry entry name, raw repo id, or local filesystem path.
        prober: Optional replacement probe callable receiving the resolved
            repo id and returning an :class:`AvailabilityReport` — the
            injectable seam the compatibility layer (23-02) wires in; the
            unit suite injects fakes so it never touches the network. A
            raising or over-slow injected prober degrades to unknown with
            a reason, exactly like the default adapters (PFC-06). Caching
            (WR-03): injected-prober results are never memoized, and an
            injected prober never reads the memo — the memo caches
            default-probe results only, so one injected call cannot poison
            later probes.
        timeout: Bounded wait in seconds for any one remote call
            (default :data:`DEFAULT_PROBE_TIMEOUT`).

    Returns:
        The availability report: ``cached`` (local path or a hub cache
        hit, with size/revision when known), ``remote_confirmed`` (the
        repo exists on a hub, with exact size and revision), or
        ``unknown`` with a reason — never an exception, never an
        "unavailable" verdict.
    """
    with _MEMO_LOCK:
        # WR-03: the memo caches DEFAULT-probe results only. An injected
        # prober never reads from and never writes to the memo — otherwise
        # one fake report would be served to every later probe of the
        # model (including default probes) for the rest of the process.
        if prober is None:
            memoized = _MEMO.get(model)
            if memoized is not None:
                return memoized
        repo_id, entry = _resolve_model_ref(model)
        if prober is None:
            memoized = _MEMO.get(repo_id)
            if memoized is not None:
                _MEMO[model] = memoized
                return memoized
        try:
            if prober is not None:
                report = _bounded_wait(lambda: prober(repo_id), timeout)
            else:
                report = _default_prober(model, repo_id, entry, timeout=timeout)
        except TimeoutError:
            logger.info(f"probe_availability: {repo_id!r} exceeded {timeout}s bound")
            report = _unknown(AvailabilityReasonCode.PROBE_TIMEOUT.value)
        except Exception as exc:  # PFC-06: the probe never raises
            logger.info(f"probe_availability: {repo_id!r} degraded to unknown: {exc!r}")
            report = _unknown(
                f"{AvailabilityReasonCode.PROBE_ERROR.value}: {exc.__class__.__name__}"
            )
        if prober is None:
            _MEMO[repo_id] = report
            if model != repo_id:
                _MEMO[model] = report
        return report


def clear_probe_memo() -> None:
    """Empty the in-process probe memo (test/invalidation hook, D-12)."""
    with _MEMO_LOCK:
        _MEMO.clear()


def effective_endpoints() -> dict[str, str]:
    """Report the endpoints/roots the libraries actually resolved.

    Reads library constants — the Hugging Face endpoint from
    ``huggingface_hub.constants.ENDPOINT`` (frozen at that library's
    import; an ``HF_ENDPOINT`` env var set afterwards is inert) and the
    ModelScope cache root via its own resolver (honors ``MODELSCOPE_CACHE``).
    Pure constants read, zero network; feeds the environment panel of the
    later TUI shell.

    Returns:
        Mapping with ``"huggingface"`` (the effective HF endpoint URL) and
        ``"modelscope_cache"`` (the resolved ModelScope cache root path).
    """
    import huggingface_hub.constants as hf_constants
    from modelscope.utils.file_utils import get_modelscope_cache_dir

    return {
        "huggingface": hf_constants.ENDPOINT,
        "modelscope_cache": get_modelscope_cache_dir(),
    }


def modelscope_cache_candidates(repo_id: str) -> list[str]:
    """Return the two ModelScope cache-layout candidate paths for a repo id.

    ModelScope resolves its cache root through its own file utility
    (honoring ``MODELSCOPE_CACHE``) and stores repos under
    ``{root}/models/{org}/{name}`` with every dot in the name replaced by
    ``___`` (the library's own convention); the legacy pre-restructure
    layout ``{root}/{org}/{name}`` persists until a download migrates it.
    Detection is pure ``os.path.isdir`` over these candidates — the
    snapshot fetcher is never used (it writes a marker file on a miss).

    Args:
        repo_id: The ModelScope repo id (``org/name``).

    Returns:
        The current-layout candidate first, then the legacy-layout one.
    """
    from modelscope.utils.file_utils import get_modelscope_cache_dir

    org, _, name = repo_id.partition("/")
    flat_name = name.replace(".", "___")
    root = get_modelscope_cache_dir()
    return [
        os.path.join(root, "models", org, flat_name),
        os.path.join(root, org, flat_name),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Resolution + the default probe flow
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_model_ref(model: str) -> tuple[str, RegistryEntry | None]:
    """Resolve a model reference to ``(repo_id, registry_entry_or_None)``.

    A registry hit (entry name or repo id) yields the entry's canonical
    repo id; anything else is treated as a raw repo id (or local path)
    verbatim.
    """
    entry = try_resolve(model)
    if entry is not None:
        return entry.model, entry
    return model, None


def _default_prober(
    model: str,
    repo_id: str,
    entry: RegistryEntry | None,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> AvailabilityReport:
    """The default probe flow for a resolved model reference.

    Local three-way cache detection first (offline-safe, zero side
    effects); the remote fallback chain only when nothing is cached.
    """
    lab = is_lab_model(entry) if entry is not None else False
    logger.debug(f"probe_availability: {repo_id!r} (lab={lab}) default probe starting")
    local_report = _probe_cached_local(model, repo_id)
    if local_report is not None:
        return local_report
    return _probe_remote(repo_id, timeout=timeout)


def _probe_cached_local(model: str, repo_id: str) -> AvailabilityReport | None:
    """Three-way local cache detection (D-12); None when nothing is cached.

    Order: an existing directory at the (expanded) input path, the
    ModelScope cache (dual layout, pure ``isdir``), then the HF cache
    (the hub's own scan helpers). Pure reads — nothing is ever written.
    """
    if os.path.isdir(os.path.expanduser(model)):
        return AvailabilityReport(state=AvailabilityState.CACHED, source_hub=Hub.LOCAL)
    ms_report = _ms_cache_report(repo_id)
    if ms_report is not None:
        return ms_report
    return _hf_cache_report(repo_id)


def _ms_cache_report(repo_id: str) -> AvailabilityReport | None:
    """Detect a ModelScope-cached repo via the dual-layout candidates."""
    for candidate in modelscope_cache_candidates(repo_id):
        if os.path.isdir(candidate):
            return AvailabilityReport(
                state=AvailabilityState.CACHED,
                size_bytes=_dir_size_bytes(candidate),
                revision=None,
                source_hub=Hub.MODELSCOPE,
            )
    return None


def _hf_cache_report(repo_id: str) -> AvailabilityReport | None:
    """Detect an HF-cached repo via the hub's own cache helpers.

    A cached ``config.json`` snapshot path proves the repo is cached; the
    scan view supplies the exact on-disk size when available and the
    revision comes from the snapshot path. Both helpers are best-effort:
    a failure simply means "not detected here".
    """
    from huggingface_hub import _CACHED_NO_EXIST, scan_cache_dir, try_to_load_from_cache

    try:
        snapshot = try_to_load_from_cache(repo_id, _HF_CACHE_PROBE_FILENAME)
    except Exception as exc:
        # Path-like and otherwise unparseable ids raise on the way in —
        # a cache lookup failure is a miss, never a probe failure.
        logger.debug(f"probe_availability: HF cache lookup unavailable for {repo_id!r}: {exc!r}")
        return None
    if snapshot is _CACHED_NO_EXIST or not isinstance(snapshot, str) or not snapshot:
        return None
    size: int | None = None
    try:
        for scanned in scan_cache_dir().repos:
            if scanned.repo_id == repo_id:
                size = int(scanned.size_on_disk)
                break
    except Exception as exc:
        logger.debug(f"probe_availability: HF cache scan unavailable for {repo_id!r}: {exc!r}")
    return AvailabilityReport(
        state=AvailabilityState.CACHED,
        size_bytes=size,
        revision=_revision_from_snapshot_path(snapshot),
        source_hub=Hub.HUGGINGFACE,
    )


def _dir_size_bytes(path: str) -> int:
    """Recursive disk-usage byte sum over a directory tree."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _revision_from_snapshot_path(snapshot_path: str) -> str | None:
    """Extract the revision from an HF snapshot file path.

    Snapshot paths look like ``.../models--org--name/snapshots/<rev>/<file>``.
    """
    parts = snapshot_path.replace("\\", "/").split("/")
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Remote half: bounded wait + the MS-first fallback chain
# ─────────────────────────────────────────────────────────────────────────────


def _bounded_wait(fn: Callable[[], T], timeout: float) -> T:
    """Run ``fn`` under a bounded wait on the shared executor.

    Raises TimeoutError when the bound is exceeded — the abandoned call
    finishes naturally in its worker thread; it is never cancelled.
    """
    future = _EXECUTOR.submit(fn)
    return future.result(timeout=timeout)


def _probe_remote(repo_id: str, *, timeout: float = DEFAULT_PROBE_TIMEOUT) -> AvailabilityReport:
    """Remote fallback chain for an uncached repo id: ModelScope first (D-11).

    A confirmed ModelScope repo is returned without ever calling the HF
    adapter; a clean miss (falsy existence) falls through to Hugging
    Face. Every failure degrades to unknown with a reason (PFC-06): a
    timeout ends the probe immediately (the core owns the time bound),
    a raising hub falls through to the other one, and clean misses on
    both hubs conclude ``NOT_FOUND_ON_EITHER_HUB``.
    """
    ms_error: Exception | None = None
    try:
        confirmed = _bounded_wait(lambda: _ms_adapter(repo_id), timeout)
    except TimeoutError:
        return _unknown(AvailabilityReasonCode.PROBE_TIMEOUT.value)
    except Exception as exc:
        ms_error = exc
        logger.info(f"probe_availability: modelscope probe failed for {repo_id!r}: {exc!r}")
    else:
        if confirmed is not None:
            return confirmed

    try:
        confirmed = _bounded_wait(lambda: _hf_adapter(repo_id), timeout)
    except TimeoutError:
        return _unknown(AvailabilityReasonCode.PROBE_TIMEOUT.value)
    except Exception as exc:
        hf_verdict = _classify_hf_error(exc)
        if hf_verdict is not None:
            return hf_verdict
    else:
        return confirmed

    if ms_error is not None:
        # MS errored and HF only confirmed the absence: name the MS failure.
        return _unknown(
            f"{AvailabilityReasonCode.PROBE_ERROR.value}: {ms_error.__class__.__name__}"
        )
    return _unknown(AvailabilityReasonCode.NOT_FOUND_ON_EITHER_HUB.value)


def _ms_adapter(repo_id: str) -> AvailabilityReport | None:
    """ModelScope remote probe: a confirmed report, or None on a clean miss.

    Existence via the pinned library's HubApi (no hand-rolled REST); on a
    hit, ``size_bytes`` is the exact integer sum of the file listing's
    ``Size`` fields and the revision comes from the listing's
    ``Revision`` field (``"master"`` default). Transport/auth failures
    raise for the chain to classify — never a not-found verdict.
    """
    from modelscope.hub.api import HubApi

    api = HubApi()
    if not api.repo_exists(repo_id):
        return None
    files = api.get_model_files(repo_id)
    size_bytes = sum(
        int(entry["Size"])
        for entry in files
        if isinstance(entry, dict) and entry.get("Size") is not None
    )
    revision = "master"
    for entry in files:
        if isinstance(entry, dict) and entry.get("Revision"):
            revision = str(entry["Revision"])
            break
    return AvailabilityReport(
        state=AvailabilityState.REMOTE_CONFIRMED,
        size_bytes=size_bytes,
        revision=revision,
        source_hub=Hub.MODELSCOPE,
    )


def _hf_adapter(repo_id: str) -> AvailabilityReport:
    """Hugging Face remote probe: a confirmed report, or raises for classification.

    The endpoint is pinned to the library constant (no user-controlled
    URL construction). A missing repo raises the library's not-found /
    gated errors, which :func:`_classify_hf_error` treat as a clean miss
    on this hub rather than a probe failure.
    """
    import huggingface_hub.constants as hf_constants
    from huggingface_hub import HfApi

    api = HfApi(endpoint=hf_constants.ENDPOINT)
    info = api.model_info(repo_id, files_metadata=True)
    size_bytes = sum(
        int(sibling.size)
        for sibling in (info.siblings or [])
        if getattr(sibling, "size", None) is not None
    )
    return AvailabilityReport(
        state=AvailabilityState.REMOTE_CONFIRMED,
        size_bytes=size_bytes,
        revision=info.sha,
        source_hub=Hub.HUGGINGFACE,
    )


def _classify_hf_error(exc: Exception) -> AvailabilityReport | None:
    """Map an HF-side exception to an unknown report; None means a clean miss.

    ``RepositoryNotFoundError`` / ``GatedRepoError`` mean the repo really
    is not usable from this hub — the chain continues to its both-hubs
    conclusion instead of degrading to unknown. A 429
    ``HfHubHTTPError`` is ``RATE_LIMITED``; ``OfflineModeIsEnabled`` is
    ``OFFLINE``; every other failure is ``PROBE_ERROR`` carrying the
    exception class name (PFC-03/PFC-06 taxonomy).
    """
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )

    if isinstance(exc, (RepositoryNotFoundError, GatedRepoError)):
        return None
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            return _unknown(AvailabilityReasonCode.RATE_LIMITED.value)
    if isinstance(exc, OfflineModeIsEnabled):
        return _unknown(AvailabilityReasonCode.OFFLINE.value)
    return _unknown(f"{AvailabilityReasonCode.PROBE_ERROR.value}: {exc.__class__.__name__}")


def _unknown(reason: str) -> AvailabilityReport:
    """Build the unknown-state degradation report (PFC-06)."""
    return AvailabilityReport(state=AvailabilityState.UNKNOWN, reason=reason)
