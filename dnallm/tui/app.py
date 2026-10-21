"""The composition root of the pre-flight wizard (25-05).

:class:`DNALLMPreflightApp` is the one place the wizard is wired together:

1. It mixes in the thread-worker seam from :mod:`dnallm.tui.workers`, so every
   blocking core call a screen makes resolves through ``self.app.<worker>``
   and runs off the event loop (``validate_data_worker`` and
   ``download_worker`` today; later plans extend the same mixin).
2. It owns the single :class:`~dnallm.tui.state.WizardState` instance, the only
   shared mutable state in the wizard.
3. It holds the injectable seams — ``collector``, ``prober`` and
   ``downloader`` — so Pilot tests run without a graphics card, a network, or
   the training stack.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import App
from textual.binding import Binding
from textual.widgets import Static

from ..utils import env_info, get_logger
from . import copy
from .screens.env import EnvScreen
from .state import WizardState
from .workers import PreflightWorkers

logger = get_logger("dnallm.tui.app")

__all__ = ["DNALLMPreflightApp"]


class DNALLMPreflightApp(PreflightWorkers, App[None]):
    """The pre-flight wizard's application object.

    The worker mixin comes first in the bases so every seam method it defines
    resolves on this instance — that is the contract later screens rely on
    when they call ``self.app.validate_data_worker(...)``.

    Attributes:
        collector: Object exposing ``collect_versions``, ``collect_device`` and
            ``collect_disk``. Defaults to :mod:`dnallm.utils.env_info`; tests
            inject a fake so no row touches torch or the filesystem.
        prober: Availability prober handed to the suitability panel (25-07).
            ``None`` means "let the core decide".
        downloader: Hub downloader handed to the download panel (25-09).
            ``None`` means "let the core decide".
        state: The single shared :class:`~dnallm.tui.state.WizardState`.
    """

    TITLE = copy.APP_TITLE

    BINDINGS: ClassVar[list[Binding]] = [Binding("q", "quit", "Quit", show=True)]

    CSS = """
    #env-body { height: 1fr; padding: 1 2; }
    #env-title { text-style: bold; }
    #env-versions-heading { margin-top: 1; }
    #env-quit-hint { padding: 0 2; }
    """

    def __init__(
        self,
        collector: Any | None = None,
        prober: Any | None = None,
        downloader: Any | None = None,
        config_path: str | None = None,
    ) -> None:
        """Build the app and store its injected seams.

        Args:
            collector: Environment collector; defaults to the
                :mod:`dnallm.utils.env_info` module.
            prober: Availability prober for the suitability panel, or ``None``.
            downloader: Download callable for the download panel, or ``None``.
            config_path: The ``--config`` path (D-07). It is stored on
                :attr:`state` and never opened here.
        """
        super().__init__()
        self.collector = env_info if collector is None else collector
        self.prober = prober
        self.downloader = downloader
        self.state = WizardState(config_path=config_path)
        self.last_failure: str | None = None

    @property
    def config_path(self) -> str | None:
        """The config file from ``--config``; read-only and never loaded here."""
        return self.state.config_path

    def render_text(self, text: str, *, id: str | None = None) -> Static:
        """Return a :class:`~textual.widgets.Static` that shows *text* literally.

        ``markup=False`` is load-bearing (Pitfall 5, T-25-08). Every non-literal
        string a screen shows — a folder path, a model name, a value read from
        the machine — can contain ``[``. Textual would otherwise parse it as
        markup, which either corrupts the layout or raises mid-render.

        Args:
            text: The exact text to display.
            id: Optional widget id.

        Returns:
            A Static widget with markup parsing disabled.
        """
        return Static(text, markup=False, id=id)

    def on_mount(self) -> None:
        """Open on the environment panel — the first step of the wizard."""
        self.push_screen(EnvScreen())

    def on_unmount(self) -> None:
        """Stop in-flight workers without waiting for a thread to finish.

        Textual thread workers cannot be interrupted; ``cancel_all`` sets each
        worker's cancelled flag (checked before it posts anything) so quitting
        never blocks on a slow core call.
        """
        self.workers.cancel_all()
