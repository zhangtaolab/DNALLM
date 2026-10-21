"""The closing summary (TUI-07): where the model files and the config went.

Presentation only, and the last screen of the wizard. It shows exactly two
facts, in a fixed order (D-09):

1. the model-files path — the string ``download_model`` returned, verbatim. The
   screen never builds, joins or normalises a path (T-25-16); a skipped
   download keeps the row and says so, so the order never depends on whether
   the user downloaded.
2. the config path — the path the config panel wrote.

Quitting from here is always allowed: downloading is optional (D-09).

Example:
    app.push_screen(SummaryScreen(model_path="/models/demo", config_path="out.yaml"))
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from .. import copy

__all__ = ["SummaryScreen"]

_TITLE_ID = "summary-title"
_MODEL_HEADING_ID = "summary-model-heading"
_MODEL_PATH_ID = "summary-model-path"
_CONFIG_HEADING_ID = "summary-config-heading"
_CONFIG_PATH_ID = "summary-config-path"
_QUIT_ID = "summary-quit"


class SummaryScreen(Screen):
    """List the model-files path, then the config path, and let the user leave."""

    #: ``q`` leaves the wizard from here; there is no step after the summary.
    BINDINGS: ClassVar[list[Binding]] = [Binding("q", "quit_all", "Quit", show=False)]

    def __init__(
        self,
        *,
        model_path: str | None = None,
        config_path: str | None = None,
    ) -> None:
        """Store the two paths exactly as the core returned them.

        Args:
            model_path: The path ``download_model`` returned, or None when the
                download was skipped or stopped.
            config_path: The path the config panel wrote, or None when no
                config was saved in this run.
        """
        super().__init__()
        self._model_path = model_path
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        """Lay out the two paths in the fixed order, both rendered literally."""
        yield Header()
        with VerticalScroll(id="summary-body"):
            yield Static(copy.SUMMARY_PANEL_TITLE, id=_TITLE_ID, markup=False)
            yield Static(copy.SUMMARY_HEADINGS["model"], id=_MODEL_HEADING_ID)
            yield Static(self._shown_model_path(), markup=False, id=_MODEL_PATH_ID)
            yield Static(copy.SUMMARY_HEADINGS["config"], id=_CONFIG_HEADING_ID)
            yield Static(self._shown_config_path(), markup=False, id=_CONFIG_PATH_ID)
            yield Button(copy.SUMMARY_QUIT_ACTION, id=_QUIT_ID, variant="primary")
        yield Static(copy.QUIT_HINT, id="summary-quit-hint")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Leave the wizard.

        Args:
            event: The button press; only this panel's quit action is handled.
        """
        if event.button.id == _QUIT_ID:
            self.action_quit_all()

    def action_quit_all(self) -> None:
        """Quit the wizard; an in-flight worker is cancelled on unmount."""
        self.app.exit()

    def _shown_model_path(self) -> str:
        """The model path verbatim, or the plain "skipped" sentence."""
        if self._model_path:
            return self._model_path
        return copy.SUMMARY_MODEL_SKIPPED

    def _shown_config_path(self) -> str:
        """The config path verbatim, or the plain "not saved" sentence."""
        if self._config_path:
            return self._config_path
        return copy.SUMMARY_CONFIG_MISSING
