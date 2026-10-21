"""The config panel (TUI-06): a validated config, and never an invalid file.

Presentation only. The panel asks the core for a config through the app's
thread-worker seam — ``generate_config`` when a model was chosen, ``edit_config``
when an existing file (or a mapping already in the wizard) is being edited — and
renders the frozen :class:`~dnallm.preflight.report.GeneratedConfig` that comes
back.

Three rules the panel exists to keep:

1. **The form edits the raw mapping, never a parsed copy of it.** The working
   mapping is exactly ``GeneratedConfig.raw`` — the mapping the writer emits —
   so a section the form does not understand (``ia3``, ``vep``, ``sweep``) rides
   along untouched and is shown read-only instead of being dropped (D-11).
2. **Every problem is listed, in the core's order.** The side pane prints one
   ``section.field → message`` line per :class:`~dnallm.preflight.report.SectionError`,
   in the order the core returned them (its fixed ``_SECTION_ORDER`` then field
   order), and the save action stays switched off until the list is empty (D-13).
3. **The preview is the file.** It is produced by the core's own
   ``render_yaml`` — the one serializer ``_dump_yaml`` writes — so what the user
   reads is byte-for-byte what a save would put on disk. The panel never
   imports yaml and never writes a file itself.

Example:
    app.push_screen(ConfigScreen(model="zhangtaolab/plant-dnabert-BPE-promoter"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from .. import copy

if TYPE_CHECKING:
    from ...preflight import GeneratedConfig, SectionError
    from ..app import DNALLMPreflightApp

__all__ = ["ConfigScreen"]

_TITLE_ID = "config-title"
_STATUS_ID = "config-status"
_ERRORS_ID = "config-errors"
_PREVIEW_ID = "config-preview"
_BODY_ID = "config-body"
_FORM_ID = "config-form"
_SIDE_ID = "config-side"

# The five sections the structured form covers (D-11). A section present in
# the loaded YAML but absent here is preserved verbatim and shown read-only.
_CORE_SECTIONS = ("task", "model", "finetune", "inference", "lora")


@dataclass(frozen=True)
class _FieldSpec:
    """One form field: the raw key it edits and how its text is typed.

    Attributes:
        name: The raw key inside the section (the file's own spelling).
        kind: How the input text is turned back into a YAML value —
            ``text``, ``int``, ``float`` or ``labels`` (a comma-separated
            list of strings).
        readonly: True for the architectural facts the checkpoint owns
            (D-12); the input renders disabled and never patches the mapping.
    """

    name: str
    kind: str = "text"
    readonly: bool = False


_FORM_SECTIONS: tuple[tuple[str, str, tuple[_FieldSpec, ...]], ...] = (
    (
        "task",
        copy.CONFIG_SECTION_TASK,
        (
            _FieldSpec("task_type"),
            _FieldSpec("num_labels", "int", readonly=True),
            _FieldSpec("label_names", "labels", readonly=True),
            _FieldSpec("threshold", "float", readonly=True),
        ),
    ),
    ("model", copy.CONFIG_SECTION_MODEL, (_FieldSpec("name"),)),
    (
        "finetune",
        copy.CONFIG_SECTION_FINETUNE,
        (
            _FieldSpec("output_dir"),
            _FieldSpec("num_train_epochs", "int"),
            _FieldSpec("per_device_train_batch_size", "int"),
            _FieldSpec("learning_rate", "float"),
            _FieldSpec("weight_decay", "float"),
            _FieldSpec("warmup_ratio", "float"),
            _FieldSpec("seed", "int"),
        ),
    ),
    (
        "inference",
        copy.CONFIG_SECTION_INFERENCE,
        (
            _FieldSpec("max_length", "int"),
            _FieldSpec("batch_size", "int"),
            _FieldSpec("device"),
        ),
    ),
    (
        "lora",
        copy.CONFIG_SECTION_LORA,
        (
            _FieldSpec("r", "int"),
            _FieldSpec("lora_alpha", "int"),
            _FieldSpec("lora_dropout", "float"),
            _FieldSpec("target_modules", "labels"),
        ),
    ),
)

_SECTION_FIELDS = {name: fields for name, _title, fields in _FORM_SECTIONS}


def _field_id(section: str, name: str) -> str:
    """The widget id for one form field.

    Args:
        section: The config section the field belongs to.
        name: The raw key inside that section.
    """
    return f"config-field-{section}-{name}"


def _typed_text(value: object) -> str:
    """Render a raw YAML value as the text a form field shows.

    Args:
        value: The raw value (a scalar, a list, or None when absent).

    Returns:
        The value as editable text; an absent value is an empty field.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _parse(kind: str, text: str) -> object:
    """Turn one input's text back into the YAML value it stands for.

    A text that does not parse is returned unchanged, so the core reports a
    field-level error in its own words instead of the panel guessing (D-13).

    Args:
        kind: The field's declared kind (``text``/``int``/``float``/``labels``).
        text: The exact text in the input.

    Returns:
        The typed value, or the raw text when it cannot be parsed.
    """
    stripped = text.strip()
    if kind == "int":
        try:
            return int(stripped)
        except ValueError:
            return text
    if kind == "float":
        try:
            return float(stripped)
        except ValueError:
            return text
    if kind == "labels":
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return text


class ConfigScreen(Screen):
    """Edit the five core config sections; never compute a verdict or write a file."""

    #: Escape goes back one step; the working mapping stays on the wizard state.
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "go_back", "Back", show=True)]

    def __init__(
        self,
        *,
        model: str | None = None,
        task_type: str | None = None,
        config_path: str | None = None,
    ) -> None:
        """Seed the panel with a model, an existing file, or the wizard's mapping.

        Args:
            model: The model chosen in the suitability panel; the core builds a
                config for it.
            task_type: The task type for a base model, passed through verbatim.
            config_path: An existing YAML to edit. Loading goes through the
                core's ``edit_config`` — never a screen-side YAML read.
        """
        super().__init__()
        self._model = model
        self._task_type = task_type
        self._config_path = config_path
        self._raw: dict = {}
        self._errors: list[SectionError] = []
        self._bound = False
        self._populating = False

    @property
    def _preflight_app(self) -> DNALLMPreflightApp:
        """The typed app, for the worker seam and the markup-safe renderer."""
        return cast("DNALLMPreflightApp", self.app)

    def compose(self) -> ComposeResult:
        """Lay out the form, the error pane, and the read-only YAML preview."""
        yield Header()
        with Horizontal(id=_BODY_ID):
            with VerticalScroll(id=_FORM_ID):
                yield Static(copy.CONFIG_PANEL_TITLE, id=_TITLE_ID)
                yield Static("", markup=False, id=_STATUS_ID)
                for section, title, fields in _FORM_SECTIONS:
                    with Vertical(id=f"config-section-{section}"):
                        yield Static(title, id=f"config-heading-{section}")
                        for spec in fields:
                            yield Static(
                                copy.CONFIG_FIELD_LABELS.get((section, spec.name), spec.name),
                                id=f"config-label-{section}-{spec.name}",
                            )
                            yield Input(
                                id=_field_id(section, spec.name),
                                disabled=spec.readonly,
                            )
                        if section == "task":
                            yield Static(copy.CONFIG_READONLY_FACTS_NOTE, id="config-facts-note")
            with VerticalScroll(id=_SIDE_ID):
                yield Static(copy.CONFIG_ERRORS_HEADING, id="config-errors-heading")
                yield Static("", markup=False, id=_ERRORS_ID)
                yield Static(copy.CONFIG_PREVIEW_HEADING, id="config-preview-heading")
                yield Static("", markup=False, id=_PREVIEW_ID)
        yield Static(copy.QUIT_HINT, id="config-quit-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Ask the core for the config this panel was seeded with."""
        self._start()

    def _start(self) -> None:
        """Dispatch the one core call this panel starts with."""
        if self._model is not None:
            self._set_status(copy.CONFIG_BUILDING)
            self._preflight_app.generate_config_worker(self._model, task_type=self._task_type)

    async def apply_config_ready(
        self, generated: GeneratedConfig, output_path: str | None = None
    ) -> None:
        """Adopt one validated result and repaint the pane.

        Args:
            generated: The frozen result the core returned.
            output_path: Where a save wrote, or None for a validate-only run
                (accepted here so the app's routing hook has one signature).
        """
        self._raw = dict(generated.raw)
        self._errors = list(generated.errors)
        self._preflight_app.state.generated_config = generated
        if not self._bound:
            self._bind_fields()
            self._bound = True
        self._render_errors()
        self._render_preview()
        self._set_status(copy.CONFIG_READY if not self._errors else copy.CONFIG_HAS_ERRORS)

    async def apply_worker_failure(self, message: str) -> None:
        """Show a failed core call as one plain-language row.

        Args:
            message: The already-plain-language sentence the seam produced.
        """
        self._set_status(message)

    def action_go_back(self) -> None:
        """Return to the previous step; the working mapping stays on the state."""
        self._preflight_app.pop_screen()

    def _bind_fields(self) -> None:
        """Fill every input from the working mapping, once, on the first result."""
        self._populating = True
        try:
            for section, fields in _SECTION_FIELDS.items():
                values = self._raw.get(section)
                values = values if isinstance(values, dict) else {}
                for spec in fields:
                    widget = self.query_one(f"#{_field_id(section, spec.name)}", Input)
                    widget.value = _typed_text(values.get(spec.name))
        finally:
            self._populating = False

    def _render_errors(self) -> None:
        """Print every error in the core's own order, one line each."""
        lines = [
            copy.CONFIG_ERROR_LINE.format(field=error.field, message=error.message)
            for error in self._errors
        ]
        self.query_one(f"#{_ERRORS_ID}", Static).update("\n".join(lines))

    def _render_preview(self) -> None:
        """Show exactly the document a save would write (the core's serializer)."""
        # Function-local: a screen module never imports the core at module scope.
        from ...preflight import render_yaml

        self.query_one(f"#{_PREVIEW_ID}", Static).update(render_yaml(self._raw))

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one(f"#{_STATUS_ID}", Static).update(text)
