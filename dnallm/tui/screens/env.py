"""The environment panel (TUI-03): what this computer can do.

The panel is presentation only. It renders the rows
:mod:`dnallm.utils.env_info` returns — no sorting, no verdicts, no filesystem
scanning — and it never imports a heavy core module at screen level.

Two speeds, one page (D-01): the version rows are read from package metadata
and paint immediately, while the graphics-card and free-disk rows show a plain
"checking…" placeholder until the app's thread worker reports. The user sees a
useful page before torch is ever imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from .. import copy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...utils.env_info import EnvRow
    from ..app import DNALLMPreflightApp

__all__ = ["EnvScreen"]

# Section keys shared with ``PreflightWorkers.collect_env_worker``.
_DEVICE_SECTION = "device"
_DISK_SECTION = "disk"


class EnvScreen(Screen):
    """Show the machine's environment facts; never derive them."""

    @property
    def _preflight_app(self) -> DNALLMPreflightApp:
        """The typed app, for the collector seam and the markup-safe renderer."""
        return cast("DNALLMPreflightApp", self.app)

    def compose(self) -> ComposeResult:
        """Lay out the title, the version rows, and the two worker-filled areas."""
        yield Header()
        with VerticalScroll(id="env-body"):
            yield Static(copy.ENV_PANEL_TITLE, id="env-title")
            yield Static(copy.ENV_VERSIONS_SECTION, id="env-versions-heading")
            yield Container(id="env-versions")
            yield Container(id="env-device")
            yield Container(id="env-disk")
        yield Static(copy.QUIT_HINT, id="env-quit-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Paint the instant rows, the checking placeholders, then start the worker."""
        app = self._preflight_app
        for row in app.collector.collect_versions():
            self._mount_row("env-versions", row)
        self._mount_placeholder("env-device", copy.ENV_ROW_LABELS[_DEVICE_SECTION])
        self._mount_placeholder("env-disk", copy.ENV_ROW_LABELS[_DISK_SECTION])
        app.collect_env_worker(app.collector)

    async def apply_env_report(self, env_report: Mapping[str, object]) -> None:
        """Replace the checking placeholders with the worker's real rows.

        Only the device and disk areas are rewritten, so the version rows
        painted on mount stay exactly where they are (versions → device →
        disk).

        Args:
            env_report: The collector's rows keyed by section — ``device`` and
                ``disk``, each a list of ``EnvRow``.
        """
        for section in (_DEVICE_SECTION, _DISK_SECTION):
            await self._replace_rows(
                f"env-{section}", cast("list[EnvRow]", env_report.get(section) or [])
            )

    async def apply_worker_failure(self, message: str) -> None:
        """Collapse both placeholders into one plain-language failure row."""
        for section in (_DEVICE_SECTION, _DISK_SECTION):
            await self.query_one(f"#env-{section}", Container).remove_children()
        self.query_one("#env-device", Container).mount(self._preflight_app.render_text(message))

    async def _replace_rows(self, container_id: str, rows: list[EnvRow]) -> None:
        """Empty one area and mount *rows* into it, in the order the core gave."""
        container = self.query_one(f"#{container_id}", Container)
        await container.remove_children()
        for row in rows:
            self._mount_row(container_id, row)

    def _mount_row(self, container_id: str, row: EnvRow) -> None:
        """Mount one labelled fact as a literal (markup-free) row."""
        self.query_one(f"#{container_id}", Container).mount(
            self._preflight_app.render_text(f"{row.label}: {row.value}")
        )

    def _mount_placeholder(self, container_id: str, label: str) -> None:
        """Mount one "label: checking…" row that the worker will replace."""
        self.query_one(f"#{container_id}", Container).mount(
            self._preflight_app.render_text(f"{label}: {copy.CHECKING_PLACEHOLDER}")
        )
