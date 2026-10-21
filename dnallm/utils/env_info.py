"""Core-owned environment fact collector for the pre-flight wizard (D-02, D-03).

The wizard's first panel shows what this machine can actually do before the
user commits to a model. Every fact it renders comes from here, because the
TUI is presentation-only (TUI-03) and a screen must never have to catch an
exception:

1. Two speeds, one report. :func:`collect_versions` and :func:`collect_disk`
   are stdlib-only and return instantly, so version and disk rows paint while
   the terminal is still starting. :func:`collect_device` needs torch, which
   it imports inside the function body only -- importing this module never
   pulls torch, transformers, or datasets into ``sys.modules``. :func:`collect`
   composes all three in a stable order (versions, device, disk).
2. Degrade, never raise. A missing package, an absent GPU, an unreadable cache
   folder, or a torch that is not installed each produce a normal row whose
   value says in plain words why the check could not complete.
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
    "collect_device",
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
# Device and GPU memory (torch, imported inside the function body -- D-02)
# ─────────────────────────────────────────────────────────────────────────────


def _torch():
    """Return the torch module, importing it only when called (D-02 seam).

    The import lives inside the body so that importing this module stays
    torch-free; tests replace this seam to exercise the device branches on
    machines without a GPU.

    Returns:
        The torch module.

    Raises:
        ImportError: torch is not installed in this environment.
    """
    import torch

    return torch


def _torch_missing_row() -> EnvRow:
    """The single row shown when PyTorch itself is unavailable."""
    return EnvRow(
        key="torch_cuda",
        label="GPU acceleration (CUDA)",
        value="PyTorch is not installed — GPU details cannot be checked",
    )


def collect_device() -> list[EnvRow]:
    """Describe the graphics card torch can see, its name, and its memory.

    torch is imported here, inside the body, so the version and disk rows can
    paint before the heavy import finishes. Every branch degrades to a row:
    torch absent, CUDA absent, and a driver that answers nothing all produce
    plain-language values instead of exceptions.

    Returns:
        For a usable GPU: the CUDA-availability, device-name, and GPU-memory
        rows. For a CPU-only machine: a single "CPU only" row. For a missing
        or unusable torch: a single row saying why the check could not run.
    """
    try:
        torch = _torch()
    except ImportError as exc:
        logger.debug(f"torch is not importable: {exc}")
        return [_torch_missing_row()]

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        logger.debug(f"Could not query CUDA availability: {exc}")
        return [
            EnvRow(
                key="torch_cuda",
                label="GPU acceleration (CUDA)",
                value="unknown — the graphics card could not be queried on this machine",
            )
        ]

    if not cuda_available:
        return [
            EnvRow(
                key="torch_cuda",
                label="GPU acceleration (CUDA)",
                value="CPU only — no GPU detected",
            )
        ]

    rows = [
        EnvRow(
            key="torch_cuda",
            label="GPU acceleration (CUDA)",
            value="available — PyTorch can use the graphics card",
        )
    ]

    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        logger.debug(f"Could not read the GPU name: {exc}")
        device_name = None
    rows.append(
        EnvRow(
            key="torch_device",
            label="GPU model",
            value=device_name or "unknown — the graphics card name could not be read",
        )
    )

    total_bytes: int | None = None
    free_bytes: int | None = None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    except Exception as exc:
        logger.debug(f"Could not read GPU memory: {exc}")
    if total_bytes is None or free_bytes is None:
        vram_value = "unknown — the graphics card memory could not be read"
    else:
        vram_value = f"{total_bytes / _GIGABYTE:.0f} GB total, {free_bytes / _GIGABYTE:.0f} GB free"
    rows.append(EnvRow(key="torch_vram", label="GPU memory (VRAM)", value=vram_value))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# The composed report
# ─────────────────────────────────────────────────────────────────────────────


def collect() -> EnvReport:
    """Collect the full environment report in its stable order.

    Versions first (instant), then the device (which imports torch), then the
    cache-root disk rows -- so two runs on the same machine render identically.

    Returns:
        The report the environment panel renders.
    """
    return EnvReport(rows=[*collect_versions(), *collect_device(), *collect_disk()])
