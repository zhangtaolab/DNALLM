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
    from ...utils.env_info import EnvRow
    from ..app import DNALLMPreflightApp

__all__ = ["EnvScreen"]


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
        """Paint the instant rows and the two checking placeholders."""
        app = self._preflight_app
        for row in app.collector.collect_versions():
            self._mount_row("env-versions", row)
        self._mount_placeholder("env-device", copy.ENV_ROW_LABELS["device"])
        self._mount_placeholder("env-disk", copy.ENV_ROW_LABELS["disk"])

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
