"""The entry choice (TUI-06, D-07): start from data, or edit a config file.

The wizard has two doors, and this panel is where the user picks one:

1. **Start from my data** — the usual path: read the data, judge which models
   fit, then build the config from the chosen model (env → data → suitability
   → config, D-09's order).
2. **Edit an existing config file** — straight to the config panel with that
   file loaded. The file is handed to the core's ``edit_config``; this panel
   never opens, parses, or validates YAML itself.

``dnallm preflight --config my.yaml`` skips this choice entirely and lands on
the config panel (D-07); the app owns that decision, so this screen is only
ever pushed when there is no ``--config``.

Example:
    app.show_entry()  # the env panel's Continue action
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static

from .. import copy

if TYPE_CHECKING:
    from ..app import DNALLMPreflightApp

__all__ = ["EntryScreen"]

_FROM_DATA_ID = "entry-from-data"
_CONFIG_PATH_ID = "entry-config-path"
_FROM_CONFIG_ID = "entry-from-config"
_STATUS_ID = "entry-status"


class EntryScreen(Screen):
    """Offer the two ways in; never read a file or build a config itself."""

    @property
    def _preflight_app(self) -> DNALLMPreflightApp:
        """The typed app, for the navigation seam."""
        return cast("DNALLMPreflightApp", self.app)

    def compose(self) -> ComposeResult:
        """Lay out the two routes and the path input the second one needs."""
        yield Header()
        with VerticalScroll(id="entry-body"):
            yield Static(copy.ENTRY_PANEL_TITLE, id="entry-title")
            yield Static(copy.ENTRY_BLURB, id="entry-blurb")
            yield Button(copy.ENTRY_START_FROM_DATA, id=_FROM_DATA_ID, variant="primary")
            yield Input(placeholder=copy.ENTRY_CONFIG_PROMPT, id=_CONFIG_PATH_ID)
            yield Button(copy.ENTRY_EDIT_CONFIG, id=_FROM_CONFIG_ID)
            yield Static("", markup=False, id=_STATUS_ID)
        yield Static(copy.QUIT_HINT, id="entry-quit-hint")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route each door to its panel.

        Args:
            event: The button press; only this panel's two routes are handled.
        """
        if event.button.id == _FROM_DATA_ID:
            self._start_from_data()
        elif event.button.id == _FROM_CONFIG_ID:
            self._edit_existing_config()

    def _start_from_data(self) -> None:
        """Open the data panel (function-local import: screens stay independent)."""
        from .data import DataScreen

        self.app.push_screen(DataScreen())

    def _edit_existing_config(self) -> None:
        """Open the config panel on the pasted path — an empty path prompts."""
        path = self.query_one(f"#{_CONFIG_PATH_ID}", Input).value.strip()
        if not path:
            self._set_status(copy.ENTRY_CONFIG_EMPTY)
            return
        self._preflight_app.show_config(config_path=path)

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one(f"#{_STATUS_ID}", Static).update(text)
