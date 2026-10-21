"""Typed payloads crossing the worker-thread → UI boundary.

Every blocking core call in the wizard runs in a Textual thread worker and
reports back by posting one of these messages; nothing else crosses the thread
boundary (``tests/tui/test_structure.py`` parses the worker bodies and asserts
that ``post_message`` is the only ``self`` call inside them).

The payloads carry the frozen report objects from ``dnallm.preflight``
verbatim — a screen renders what the core computed and never re-derives a
verdict from a reshaped tuple. ``DownloadProgress`` / ``DownloadFinished``
carry primitives because a download has no core report object of its own.

Plain-language rule (D-10): every string a user reads is written for a
biologist — no exception class names, no reason-code spelling in UI copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.message import Message

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..preflight import (
        AvailabilityReport,
        CompatibilityReport,
        DataReport,
        GeneratedConfig,
    )

__all__ = [
    "CompatibilityReady",
    "ConfigReady",
    "DataReportReady",
    "DownloadFinished",
    "DownloadProgress",
    "EnvReportReady",
    "WorkerFailed",
]


class EnvReportReady(Message):
    """The environment collector finished; ``env_report`` holds its rows.

    Attributes:
        env_report: The collector's row mapping, rendered verbatim (TUI-03).
    """

    def __init__(self, env_report: Mapping[str, object]) -> None:
        super().__init__()
        self.env_report = env_report


class DataReportReady(Message):
    """Data validation finished.

    Attributes:
        data_report: The frozen ``DataReport`` returned by ``validate_data``.
    """

    def __init__(self, data_report: DataReport) -> None:
        super().__init__()
        self.data_report = data_report


class CompatibilityReady(Message):
    """Suitability checking finished.

    Attributes:
        report: The frozen ``CompatibilityReport`` from ``check_compatibility``.
        local_by_name: Availability already known from the local cache, keyed
            by model id; rendering only, never re-probed.
    """

    def __init__(
        self, report: CompatibilityReport, local_by_name: Mapping[str, AvailabilityReport]
    ) -> None:
        super().__init__()
        self.report = report
        self.local_by_name = local_by_name


class ConfigReady(Message):
    """Config generation/editing finished and validated clean.

    Attributes:
        generated: The frozen ``GeneratedConfig`` (validated YAML plus the
            section map the panels render).
    """

    def __init__(self, generated: GeneratedConfig) -> None:
        super().__init__()
        self.generated = generated


class DownloadProgress(Message):
    """One download stage started.

    Attributes:
        done: Stages finished so far.
        total: Stages this download is expected to run.
        label: Plain-language description of the stage.
    """

    def __init__(self, done: int, total: int, label: str) -> None:
        super().__init__()
        self.done = done
        self.total = total
        self.label = label


class DownloadFinished(Message):
    """The download worker stopped — finished, cancelled, or failed.

    Attributes:
        path: Where the model files were stored, or None when nothing was
            downloaded.
        cancelled: True when the user stopped the download before it finished.
        error: Plain-language failure description, or None on success/cancel.
    """

    def __init__(self, path: str | None, cancelled: bool, error: str | None = None) -> None:
        super().__init__()
        self.path = path
        self.cancelled = cancelled
        self.error = error


class WorkerFailed(Message):
    """A core call raised; ``message`` is the plain-language explanation.

    Attributes:
        message: User-facing sentence — never an exception class name.
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message
