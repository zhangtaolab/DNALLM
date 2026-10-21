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
from .messages import (
    AvailabilityUpdated,
    CompatibilityReady,
    ConfigReady,
    DataReportReady,
    DownloadFinished,
    DownloadProgress,
    EnvReportReady,
    WorkerFailed,
)
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
    #data-continue { width: auto; margin-left: 2; }
    #suitability-body { height: 1fr; padding: 0 2; }
    #suitability-title { text-style: bold; }
    #suitability-intent { height: auto; }
    #suitability-intent-finetune { width: auto; }
    #suitability-intent-predict { width: auto; margin-left: 2; }
    #suitability-task-legend { color: $text-muted; }
    #suitability-check { width: auto; margin-top: 1; }
    #suitability-status { text-style: italic; }
    #suitability-verdicts-heading { text-style: bold; }
    #suitability-listing-heading { text-style: bold; }
    #suitability-verdicts { max-height: 12; }
    #suitability-listing { max-height: 14; }
    #suitability-detail { text-style: italic; }
    #suitability-use { width: auto; margin: 1 0; }
    #suitability-quit-hint { padding: 0 2; }
    #config-body { height: 1fr; }
    #config-form { width: 1fr; padding: 0 2; }
    #config-side { width: 1fr; padding: 0 2; }
    #config-title { text-style: bold; }
    #config-status { text-style: italic; }
    #config-section-task, #config-section-model, #config-section-finetune,
    #config-section-inference, #config-section-lora { height: auto; margin-top: 1; }
    #config-heading-task, #config-heading-model, #config-heading-finetune,
    #config-heading-inference, #config-heading-lora { text-style: bold; }
    #config-facts-note { text-style: italic; }
    #config-errors-heading { text-style: bold; }
    #config-errors { max-height: 14; overflow-y: auto; }
    #config-error-details { width: auto; margin: 1 0; }
    .config-hint { color: $text-muted; }
    #config-preview-heading { text-style: bold; margin-top: 1; }
    #config-preview { max-height: 24; }
    #config-kept-as-is-heading { text-style: bold; margin-top: 1; }
    #config-kept-as-is { height: auto; }
    #config-output-row { height: auto; margin: 1 2; }
    #config-output-label { width: auto; padding: 0 1 0 0; }
    #config-output-path { width: 1fr; }
    #config-save { width: auto; margin-left: 2; }
    #config-continue { width: auto; }
    #config-save-reason { padding: 0 2; }
    #config-quit-hint { padding: 0 2; }
    #download-body { height: 1fr; padding: 0 2; }
    #download-title { text-style: bold; }
    #download-blurb { margin-bottom: 1; }
    #download-model-label { text-style: bold; margin-top: 1; }
    #download-model { padding: 0 2; }
    #download-availability { padding: 0 2; text-style: italic; }
    #download-progress { margin-top: 1; }
    #download-bar { margin: 0 2; }
    #download-elapsed { padding: 0 2; text-style: italic; }
    #download-status { margin-top: 1; }
    #download-actions { height: auto; margin-top: 1; }
    #download-start { width: auto; }
    #download-cancel { width: auto; margin-left: 2; }
    #download-retry { width: auto; margin-left: 2; }
    #download-skip { width: auto; margin-left: 2; }
    #download-quit-hint { padding: 0 2; }
    #summary-body { height: 1fr; padding: 0 2; }
    #summary-title { text-style: bold; }
    #summary-model-heading { text-style: bold; margin-top: 1; }
    #summary-model-path { padding: 0 2; }
    #summary-config-heading { text-style: bold; margin-top: 1; }
    #summary-config-path { padding: 0 2; }
    #summary-quit { width: auto; margin-top: 1; }
    #summary-quit-hint { padding: 0 2; }
    #entry-body { height: 1fr; padding: 0 2; }
    #entry-title { text-style: bold; }
    #entry-blurb { margin-bottom: 1; }
    #entry-from-data { width: auto; }
    #entry-config-path { margin-top: 1; }
    #entry-from-config { width: auto; margin-top: 1; }
    #entry-status { text-style: italic; }
    #entry-quit-hint { padding: 0 2; }
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
        """Stop in-flight work without waiting for a thread to finish.

        Textual thread workers cannot be interrupted; ``cancel_all`` sets each
        worker's cancelled flag (checked before it posts anything) so quitting
        never blocks on a slow core call. The download runs on its own daemon
        thread (WR-02), outside the worker manager, so it is asked to stop the
        same cooperative way and is never joined: the process can exit while a
        hub call is still in flight.
        """
        self.workers.cancel_all()
        if self._download_operation is not None:
            self._download_operation.cancel()

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

    async def on_compatibility_ready(self, message: CompatibilityReady) -> None:
        """Hand the suitability report to the panel that asked for it.

        ``check_compatibility_worker`` and ``local_availability_batch_worker``
        are mixed into this app, so their messages arrive here rather than at
        the screen. The suitability panel opts in by defining an async
        ``apply_compatibility_ready(report, local_by_name, generation)`` hook.

        Args:
            message: The typed payload carrying the frozen report, whatever
                local availability is already known, and the check generation
                it belongs to.
        """
        screen = self._active_screen()
        apply_report = getattr(screen, "apply_compatibility_ready", None)
        if apply_report is None:
            logger.debug("CompatibilityReady arrived with no suitability panel showing")
            return
        await apply_report(message.report, message.local_by_name, message.generation)

    async def on_config_ready(self, message: ConfigReady) -> None:
        """Hand a validated config result to the config panel.

        ``generate_config_worker`` and ``edit_config_worker`` are mixed into
        this app, so their message arrives here rather than at the screen. The
        config panel opts in by defining an async
        ``apply_config_ready(generated, output_path)`` hook.

        Args:
            message: The typed payload carrying the frozen result and, for a
                save, the path the core wrote.
        """
        screen = self._active_screen()
        apply_config_ready = getattr(screen, "apply_config_ready", None)
        if apply_config_ready is None:
            logger.debug("ConfigReady arrived with no config panel showing")
            return
        await apply_config_ready(message.generated, message.output_path)

    async def on_availability_updated(self, message: AvailabilityUpdated) -> None:
        """Hand a re-checked row's availability back to the panel that asked.

        Args:
            message: The typed payload naming the model and its new report.
        """
        screen = self._active_screen()
        apply_update = getattr(screen, "apply_availability_update", None)
        if apply_update is None:
            logger.debug("AvailabilityUpdated arrived with no suitability panel showing")
            return
        await apply_update(message.model, message.availability)

    async def on_download_progress(self, message: DownloadProgress) -> None:
        """Hand one downloader callback to the panel that asked for the download.

        ``download_worker`` is mixed into this app, so its messages arrive here
        rather than at the screen. The download panel opts in by defining an
        async ``apply_download_progress(done, total, label)`` hook.

        Args:
            message: The typed payload carrying the stage's units and label.
        """
        screen = self._active_screen()
        apply_progress = getattr(screen, "apply_download_progress", None)
        if apply_progress is None:
            logger.debug("DownloadProgress arrived with no download panel showing")
            return
        await apply_progress(message.done, message.total, message.label)

    async def on_download_finished(self, message: DownloadFinished) -> None:
        """Hand the download outcome to the panel that started it.

        Args:
            message: The typed payload carrying the returned path, the cancel
                flag, and any plain-language error.
        """
        screen = self._active_screen()
        apply_finished = getattr(screen, "apply_download_finished", None)
        if apply_finished is None:
            logger.debug("DownloadFinished arrived with no download panel showing")
            return
        await apply_finished(message.path, message.cancelled, message.error)

    def show_download(self) -> None:
        """Open the optional closing step: download the chosen model, or skip.

        This is the config → download edge of D-09's flow order, and the only
        place it is implemented. The import stays function-local for the same
        reason :meth:`show_suitability`'s does: this module must import without
        the later panels.
        """
        try:
            from .screens.download import DownloadScreen
        except ImportError:
            logger.warning("The download panel is not part of this build yet")
            return
        self.push_screen(DownloadScreen())

    def show_entry(self) -> None:
        """Move past the environment panel: ask how to start, or jump to config.

        D-07's dual entry. ``--config`` means the user already told us which
        file to open, so the choice is skipped and the config panel opens on
        that path; otherwise the entry panel offers "start from your data"
        versus "edit an existing config file".

        The import is function-local so this module imports without the later
        panels, exactly as :meth:`show_suitability` does.
        """
        if self.state.config_path:
            self.show_config(config_path=self.state.config_path)
            return
        try:
            from .screens.entry import EntryScreen
        except ImportError:
            logger.warning("The entry panel is not part of this build yet")
            return
        self.push_screen(EntryScreen())

    def show_config(
        self,
        *,
        model: str | None = None,
        task_type: str | None = None,
        config_path: str | None = None,
    ) -> None:
        """Open the config panel, seeded with a model, a file, or nothing.

        Args:
            model: The model the suitability panel handed over (D-09's
                suitability → config edge lands here, and only here).
            task_type: The task type for a base model, passed through verbatim.
            config_path: An existing YAML to edit; the panel loads it through
                the core's ``edit_config``, never a screen-side read.
        """
        try:
            from .screens.config import ConfigScreen
        except ImportError:
            logger.warning("The config panel is not part of this build yet")
            return
        self.push_screen(ConfigScreen(model=model, task_type=task_type, config_path=config_path))

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
