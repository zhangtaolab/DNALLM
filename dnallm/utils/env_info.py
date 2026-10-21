"""Core-owned environment fact collector for the pre-flight wizard (D-02, D-03).

The wizard's first panel shows what this machine can actually do before the
user commits to a model. Every fact it renders comes from here, because the
TUI is presentation-only (TUI-03) and a screen must never have to catch an
exception:

1. Instant half of the report. :func:`collect_versions` and
   :func:`collect_disk` are stdlib-only: they read package metadata and query
   the model-cache filesystems without importing torch, so version and disk
   rows paint while the terminal is still starting. :func:`collect` composes
   the row groups this module owns in a stable order.
2. Degrade, never raise. A missing package or an unreadable cache folder each
   produce a normal row whose value says in plain words why the check could
   not complete.
3. Rows are data, not paragraphs. :class:`EnvRow` carries ``key``, ``label``,
   and ``value`` so the screen decides how to lay them out; the plain-language
   phrasing of ``label`` and ``value`` is produced here, once.
4. Cache roots are resolved with the standard library alone -- never through
   ``platformdirs`` or a hub client. This module is core (``dnallm/utils/``)
   and must not depend on the optional ``tui`` extra at runtime.

Example:
    from dnallm.utils.env_info import collect

    for row in collect().rows:
        print(f"{row.label}: {row.value}")
"""

import os
import platform
import shutil
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from .logger import get_logger

__all__ = [
    "EnvReport",
    "EnvRow",
    "collect",
    "collect_disk",
    "collect_versions",
]

logger = get_logger("dnallm.utils.env_info")

# One gibibyte, used so a 24 GiB card reads back as "24 GB" rather than 25.8.
_GIGABYTE = 1024**3

# Libraries whose version the panel reports, in display order (D-03). The
# first element is the row key, the second the name shown to the user, and the
# third the distribution name importlib.metadata looks up.
_VERSIONED_LIBRARIES: tuple[tuple[str, str, str], ...] = (
    ("transformers_version", "Transformers version", "transformers"),
    ("numpy_version", "NumPy version", "numpy"),
    ("datasets_version", "Datasets version", "datasets"),
)

# Environment variables a user may set to move a model cache elsewhere; the
# resolution order below mirrors each library's own precedence.
_HF_HUB_CACHE_ENV = "HF_HUB_CACHE"
_HF_HOME_ENV = "HF_HOME"
_MODELSCOPE_CACHE_ENV = "MODELSCOPE_CACHE"


@dataclass(frozen=True)
class EnvRow:
    """One labelled environment fact.

    Attributes:
        key: Stable machine-readable identifier (for example
            ``"python_version"``) so a screen can style or route a row without
            matching on its display text.
        label: Plain-language name of the fact, shown to the user.
        value: Plain-language value; on a degraded check this says why the
            value is unknown instead of being empty.
    """

    key: str
    label: str
    value: str


@dataclass(frozen=True)
class EnvReport:
    """The complete environment picture the wizard's env panel renders.

    Produced by :func:`collect` (and by the part-collectors) and consumed by
    the environment screen in the TUI (25-05); nothing else is expected to
    read it.

    Attributes:
        rows: The environment rows, in a fixed order.
    """

    rows: list[EnvRow] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Versions (stdlib only, instant)
# ─────────────────────────────────────────────────────────────────────────────


def _distribution_version(distribution: str) -> str | None:
    """Read an installed distribution's version without importing it.

    Args:
        distribution: The distribution name on the Python path.

    Returns:
        The version string, or ``None`` when the metadata cannot be read
        (not installed, or a broken install). Never raises.
    """
    try:
        return metadata.version(distribution)
    except Exception as exc:  # PackageNotFoundError, plus any broken-metadata error
        logger.debug(f"Could not read the {distribution} version: {exc}")
        return None


def _version_or_unknown(distribution: str, display_name: str) -> str:
    """Return the version string, or a plain-language note that it is unknown."""
    version = _distribution_version(distribution)
    if version is None:
        return f"unknown — the installed {display_name} version could not be read"
    return version


def collect_versions() -> list[EnvRow]:
    """Describe which software versions this machine is running.

    This is the instant half of the report: it reads package metadata through
    ``importlib.metadata`` and the interpreter's own version, so no library is
    imported and no graphics card is touched.

    Returns:
        The dnallm and Python rows followed by one row per library in
        :data:`_VERSIONED_LIBRARIES`. A distribution whose metadata cannot be
        read still gets a row, with a plain-language unknown value.
    """
    rows = [
        EnvRow(
            key="dnallm_version",
            label="DNALLM version",
            value=_version_or_unknown("dnallm", "DNALLM"),
        ),
        EnvRow(
            key="python_version",
            label="Python version",
            value=platform.python_version(),
        ),
    ]
    rows.extend(
        EnvRow(
            key=key,
            label=label,
            value=_version_or_unknown(distribution, label.replace(" version", "")),
        )
        for key, label, distribution in _VERSIONED_LIBRARIES
    )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Disk space for the model caches (stdlib only, instant)
# ─────────────────────────────────────────────────────────────────────────────


def _cache_roots() -> list[Path]:
    """Resolve the model-cache roots whose free space the panel reports.

    Resolution is deliberately huggingface_hub's own precedence, implemented
    with nothing but ``os`` and ``pathlib``: ``HF_HUB_CACHE`` first, then
    ``HF_HOME`` (whose downloads live in the ``hub`` subfolder), then the
    ``~/.cache/huggingface/hub`` default. ModelScope honours
    ``MODELSCOPE_CACHE`` and otherwise uses ``~/.cache/modelscope``.

    Returns:
        One Hugging Face root followed by one ModelScope root, in that order.
    """
    roots: list[Path] = []

    hub_cache = os.environ.get(_HF_HUB_CACHE_ENV)
    if hub_cache:
        roots.append(Path(hub_cache).expanduser())
    else:
        hf_home = os.environ.get(_HF_HOME_ENV)
        base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
        roots.append(base / "hub")

    modelscope_cache = os.environ.get(_MODELSCOPE_CACHE_ENV)
    if modelscope_cache:
        roots.append(Path(modelscope_cache).expanduser())
    else:
        roots.append(Path.home() / ".cache" / "modelscope")

    return roots


def _dir_free_bytes(path: Path | str) -> int | None:
    """Return the free bytes on the filesystem holding *path*, or ``None``.

    A cache root may sit on a network mount that is currently unreachable, so
    the disk query is wrapped: a failure is reported to the log and becomes an
    unknown row, never an exception in the wizard's face (T-25-07).

    Args:
        path: The cache root to query.

    Returns:
        Free bytes, or ``None`` when the filesystem could not be queried.
    """
    try:
        return shutil.disk_usage(str(path)).free
    except OSError as exc:
        logger.debug(f"Could not check the free disk space of {path}: {exc}")
        return None


def collect_disk() -> list[EnvRow]:
    """Describe how much room the model caches have left.

    Only the cache roots from :func:`_cache_roots` are queried -- never an
    arbitrary user path -- and each query degrades to a plain-language unknown
    value rather than raising.

    Returns:
        One row per resolved cache root, in :func:`_cache_roots` order.
    """
    rows: list[EnvRow] = []
    for root in _cache_roots():
        free_bytes = _dir_free_bytes(root)
        if free_bytes is None:
            value = "unknown — this folder could not be checked"
        else:
            value = f"{free_bytes / _GIGABYTE:.0f} GB free"
        rows.append(
            EnvRow(
                key=f"disk:{root}",
                label=f"Free disk space for model downloads ({root})",
                value=value,
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# The composed report
# ─────────────────────────────────────────────────────────────────────────────


def collect() -> EnvReport:
    """Collect the environment rows this module can produce without torch.

    Versions first (instant), then the cache-root disk rows, so two runs on
    the same machine render identically.

    Returns:
        The report the environment panel renders.
    """
    return EnvReport(rows=[*collect_versions(), *collect_disk()])
