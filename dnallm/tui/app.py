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

Worker-failure contract: every worker is declared with
``@work(thread=True, exit_on_error=False)``, so a core call that raises never
tears the app down. The worker posts a :class:`~dnallm.tui.messages.WorkerFailed`
whose ``message`` is already plain language; this app records it on
``last_failure`` and hands it to the active screen's async
``apply_worker_failure(message)`` hook, which renders it as a status row. A
screen without that hook leaves the failure in the log rather than crashing.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import App, ScreenStackError
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

from ..utils import env_info, get_logger
from . import copy
from .messages import DataReportReady, EnvReportReady, WorkerFailed
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
    #env-continue { margin: 1 2; width: auto; }
    #env-quit-hint { padding: 0 2; }
    #data-body { height: 1fr; padding: 0 2; }
    #data-title { text-style: bold; }
    #data-actions { height: auto; }
    #data-check { width: auto; }
    #data-skip { width: auto; margin-left: 2; }
    #data-status { text-style: italic; }
    #data-splits { height: auto; }
    #data-charset { height: auto; }
    #data-summary-heading { text-style: bold; }
    #data-charset-heading { text-style: bold; }
    #data-findings-heading { text-style: bold; }
    #data-findings { max-height: 14; }
    #data-no-findings { text-style: italic; }
    #data-quit-hint { padding: 0 2; }
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

    def _active_screen(self) -> Screen | None:
        """The visible screen, or ``None`` before the first one is pushed."""
        try:
            return self.screen
        except ScreenStackError:
            return None

    async def on_env_report_ready(self, message: EnvReportReady) -> None:
        """Hand the collector's rows to the panel that asked for them.

        The worker method lives on this app (it is mixed in), so its message
        arrives here rather than at the screen. Screens opt in by defining an
        async ``apply_env_report(env_report)`` hook.

        Args:
            message: The typed payload carrying the collector's rows.
        """
        screen = self._active_screen()
        apply_env_report = getattr(screen, "apply_env_report", None)
        if apply_env_report is None:
            logger.debug("EnvReportReady arrived with no environment panel showing")
            return
        await apply_env_report(message.env_report)

    async def on_data_report_ready(self, message: DataReportReady) -> None:
        """Hand the data report to the panel that asked for it.

        ``validate_data_worker`` is mixed into this app, so its message arrives
        here rather than at the screen. The data panel opts in by defining an
        async ``apply_data_report(data_report)`` hook.

        Args:
            message: The typed payload carrying the frozen ``DataReport``.
        """
        screen = self._active_screen()
        apply_data_report = getattr(screen, "apply_data_report", None)
        if apply_data_report is None:
            logger.debug("DataReportReady arrived with no data panel showing")
            return
        await apply_data_report(message.data_report)

    def show_suitability(self) -> None:
        """Move on to the suitability step after the data step ends.

        The import is function-local for two reasons: this module must import
        without the later panels, and the suitability panel (25-07) is the one
        place that decides what it needs. A build without that panel logs a
        line instead of raising into the UI thread.
        """
        try:
            from .screens.suitability import SuitabilityScreen
        except ImportError:
            logger.warning("The suitability panel is not part of this build yet")
            return
        self.push_screen(SuitabilityScreen())

    async def on_worker_failed(self, message: WorkerFailed) -> None:
        """Show any worker error as a plain-language row, never an app exit.

        ``exit_on_error=False`` on the seam keeps Textual from tearing the app
        down when a core call raises; this handler is what the user sees
        instead. Screens opt in by defining an async
        ``apply_worker_failure(message)`` hook.

        Args:
            message: The typed payload whose ``message`` is already plain
                language (exception details stay in the log).
        """
        self.last_failure = message.message
        logger.warning(f"A wizard worker failed: {message.message}")
        screen = self._active_screen()
        apply_failure = getattr(screen, "apply_worker_failure", None)
        if apply_failure is None:
            logger.debug("WorkerFailed arrived with no screen able to show it")
            return
        await apply_failure(message.message)
