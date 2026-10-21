"""The config panel (TUI-06): a validated config, and never an invalid file.

Presentation only. The panel asks the core for a config through the app's
thread-worker seam — ``generate_config`` when a model was chosen, ``edit_config``
when an existing file (or a mapping already in the wizard) is being edited — and
renders the frozen :class:`~dnallm.preflight.report.GeneratedConfig` that comes
back.

Four rules the panel exists to keep:

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
   imports yaml and never writes a file itself: the only write is the core's
   ``edit_config(..., output_path=...)``, reached through the seam, and reached
   only while the error list is empty.
4. **Architectural facts are read-only, policy fields are editable (D-12).**
   ``num_labels`` / ``label_names`` / ``threshold`` come from the model
   checkpoint and are rendered disabled; loading uses the checkpoint's value
   regardless, so an editable field would be silently overridden. ``max_length``
   and every training hyperparameter are the user's own policy and stay open.

Example:
    app.push_screen(ConfigScreen(model="zhangtaolab/plant-dnabert-BPE-promoter"))
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static

from .. import copy

if TYPE_CHECKING:
    from ...preflight import GeneratedConfig, SectionError
    from ..app import DNALLMPreflightApp

__all__ = ["ConfigScreen"]

_TITLE_ID = "config-title"
_STATUS_ID = "config-status"
_ERRORS_ID = "config-errors"
_DETAILS_ID = "config-error-details"
_PREVIEW_ID = "config-preview"
_BODY_ID = "config-body"
_FORM_ID = "config-form"
_SIDE_ID = "config-side"
_KEPT_ID = "config-kept-as-is"
_KEPT_HEADING_ID = "config-kept-as-is-heading"
_OUTPUT_PATH_ID = "config-output-path"
_SAVE_ID = "config-save"
_SAVE_REASON_ID = "config-save-reason"
_CONTINUE_ID = "config-continue"

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

_FIELD_BY_ID: dict[str, tuple[str, _FieldSpec]] = {
    f"config-field-{section}-{spec.name}": (section, spec)
    for section, fields in _SECTION_FIELDS.items()
    for spec in fields
}


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


def _unchanged(kind: str, parsed: object, existing: object) -> bool:
    """True when a field's text already says what the mapping already holds.

    The guard matters because assembling the form sets every input's value,
    which queues the same ``Changed`` message a keystroke produces. Comparing
    the parsed text with the working mapping makes such a message a no-op, so
    merely opening the panel never invents a patch.

    Args:
        kind: The field's declared kind.
        parsed: The value the input's text parses to.
        existing: The value the working mapping holds for that key.

    Returns:
        True when revalidating this field would change nothing.
    """
    if existing is None and parsed in ("", []):
        return True
    if isinstance(existing, float) and isinstance(parsed, (int, float)):
        return float(existing) == float(parsed)
    return bool(parsed == existing)


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
        self._show_details = False
        self._bound = False
        self._kept: tuple[str, ...] = ()

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
                            hint = copy.CONFIG_FIELD_HINTS.get((section, spec.name))
                            if hint is not None:
                                yield Static(
                                    hint,
                                    markup=False,
                                    id=f"config-hint-{section}-{spec.name}",
                                    classes="config-hint",
                                )
                        if section == "task":
                            yield Static(copy.CONFIG_READONLY_FACTS_NOTE, id="config-facts-note")
                yield Static(copy.CONFIG_KEPT_AS_IS_HEADING, id=_KEPT_HEADING_ID)
                yield Container(id=_KEPT_ID)
            with VerticalScroll(id=_SIDE_ID):
                yield Static(copy.CONFIG_ERRORS_HEADING, id="config-errors-heading")
                yield Static("", markup=False, id=_ERRORS_ID)
                yield Button(copy.CONFIG_ERROR_DETAILS_SHOW, id=_DETAILS_ID, disabled=True)
                yield Static(copy.CONFIG_PREVIEW_HEADING, id="config-preview-heading")
                yield Static("", markup=False, id=_PREVIEW_ID)
        with Horizontal(id="config-output-row"):
            yield Static(copy.CONFIG_OUTPUT_PATH_LABEL, id="config-output-label")
            yield Input(copy.CONFIG_DEFAULT_EXPORT_NAME, id=_OUTPUT_PATH_ID)
        yield Button(copy.CONFIG_SAVE_ACTION, id=_SAVE_ID, variant="primary", disabled=True)
        yield Static(copy.CONFIG_SAVE_DISABLED_REASON, markup=False, id=_SAVE_REASON_ID)
        yield Button(copy.CONFIG_CONTINUE_TO_DOWNLOAD, id=_CONTINUE_ID, disabled=True)
        yield Static(copy.QUIT_HINT, id="config-quit-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Ask the core for the config this panel was seeded with."""
        self._start()
        self._update_continue_state()

    def _start(self) -> None:
        """Dispatch the one core call this panel starts with.

        Precedence: an explicit model, then an explicit file, then whatever the
        wizard already holds (back navigation resumes the working mapping
        through the core rather than re-reading a file), then ``--config``.
        """
        state = self._preflight_app.state
        if self._model is not None:
            self._set_status(copy.CONFIG_BUILDING)
            self._preflight_app.generate_config_worker(self._model, task_type=self._task_type)
            return
        if self._config_path is not None:
            self._set_status(copy.CONFIG_EDITING)
            self._preflight_app.edit_config_worker(self._config_path, {})
            return
        if state.generated_config is not None:
            self._set_status(copy.CONFIG_EDITING)
            self._preflight_app.edit_config_worker(state.generated_config.raw, {})
            return
        if state.config_path is not None:
            self._set_status(copy.CONFIG_EDITING)
            self._preflight_app.edit_config_worker(state.config_path, {})
            return
        self._set_status(copy.CONFIG_NO_SOURCE)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle this panel's three actions: save, error detail, continue.

        Args:
            event: The button press; only this panel's buttons are handled.
        """
        if event.button.id == _SAVE_ID:
            self._save()
        elif event.button.id == _DETAILS_ID:
            self._show_details = not self._show_details
            self._render_errors()
        elif event.button.id == _CONTINUE_ID:
            self._preflight_app.show_download()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Revalidate the working mapping on every field change (D-13).

        The working mapping is patched locally at the same instant the value
        is typed, so two edits made inside one worker round-trip both survive;
        the core then revalidates the whole mapping through the shared
        ``validate_sections`` path and returns every remaining problem.

        Args:
            event: The input change; only this panel's editable fields matter.
        """
        field = _FIELD_BY_ID.get(event.input.id or "")
        if field is None or not self._raw:
            return
        section, spec = field
        if spec.readonly:
            return
        parsed = _parse(spec.kind, event.value)
        current = self._raw.get(section)
        existing = current.get(spec.name) if isinstance(current, dict) else None
        if _unchanged(spec.kind, parsed, existing):
            return
        self._record(section, spec.name, parsed)
        # Snapshot on this thread: the worker deep-copies its input, and the
        # next keystroke keeps writing into these same nested dicts. A key
        # added mid-copy would raise "dictionary changed size during iteration"
        # inside the worker (WR-01).
        self._preflight_app.edit_config_worker(deepcopy(self._raw), {section: {spec.name: parsed}})

    def _record(self, section: str, name: str, value: object) -> None:
        """Write one edited value into the working mapping.

        Args:
            section: The config section being edited.
            name: The raw key inside it.
            value: The typed value the input now stands for.
        """
        current = self._raw.get(section)
        if not isinstance(current, dict):
            current = {}
        current[name] = value
        self._raw[section] = current

    def _save(self) -> None:
        """Write the config — only while clean, and only through the core."""
        if not self._raw or self._errors:
            return
        path = self.query_one(f"#{_OUTPUT_PATH_ID}", Input).value.strip()
        if not path:
            self._set_status(copy.CONFIG_OUTPUT_PATH_PROMPT)
            return
        self._set_status(copy.CONFIG_SAVING)
        # Same snapshot rule as on_input_changed: the worker must not read a
        # mapping this thread can still mutate (WR-01).
        self._preflight_app.edit_config_worker(deepcopy(self._raw), {}, output_path=path)

    async def apply_config_ready(
        self, generated: GeneratedConfig, output_path: str | None = None
    ) -> None:
        """Adopt one validated result and repaint the form, errors, and preview.

        Args:
            generated: The frozen result the core returned.
            output_path: Where a save wrote, or None for a validate-only run.
                The core writes only on a clean run, so a path here with no
                errors means the file exists (WR-02).
        """
        self._raw = deepcopy(generated.raw)
        self._errors = list(generated.errors)
        self._preflight_app.state.generated_config = generated
        if self._bound:
            # Only the checkpoint-owned facts are refreshed; rewriting an
            # editable field would move the caret out from under the user.
            self._bind_fields(readonly_only=True)
        else:
            self._bind_fields()
            self._bound = True
        self._render_errors()
        self._render_preview()
        await self._render_kept_as_is()
        self._update_save_state()
        if output_path is not None and not generated.errors:
            self._preflight_app.state.export_path = output_path
            self._set_status(copy.CONFIG_SAVED.format(path=output_path))
            self._update_continue_state()
            return
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

    def _bind_fields(self, *, readonly_only: bool = False) -> None:
        """Fill the inputs from the working mapping.

        Args:
            readonly_only: Refresh only the disabled, checkpoint-owned facts —
                the refresh that is safe while the user is typing elsewhere.
        """
        for section, fields in _SECTION_FIELDS.items():
            values = self._raw.get(section)
            values = values if isinstance(values, dict) else {}
            for spec in fields:
                if readonly_only and not spec.readonly:
                    continue
                widget = self.query_one(f"#{_field_id(section, spec.name)}", Input)
                widget.value = _typed_text(values.get(spec.name))

    def _render_errors(self) -> None:
        """Print every error as one plain sentence, in the core's own order.

        The plain sentence comes from :func:`copy.config_error_message`; the
        core's exact message is appended only while the detail toggle is on and
        only when it differs from that sentence, so nothing is printed twice
        and the raw text is never silently dropped (GAP A).
        """
        if not self._errors:
            self._show_details = False
        lines: list[str] = []
        for error in self._errors:
            plain = copy.config_error_message(error.section, error.field, error.message)
            lines.append(copy.CONFIG_ERROR_LINE.format(field=error.field, message=plain))
            if self._show_details and plain != error.message:
                lines.append(copy.CONFIG_ERROR_DETAIL_LINE.format(message=error.message))
        self.query_one(f"#{_ERRORS_ID}", Static).update("\n".join(lines))
        details = self.query_one(f"#{_DETAILS_ID}", Button)
        details.disabled = not self._errors
        details.label = (
            copy.CONFIG_ERROR_DETAILS_HIDE if self._show_details else copy.CONFIG_ERROR_DETAILS_SHOW
        )

    def _render_preview(self) -> None:
        """Show exactly the document a save would write (the core's serializer)."""
        # Function-local: a screen module never imports the core at module scope.
        from ...preflight import render_yaml

        self.query_one(f"#{_PREVIEW_ID}", Static).update(render_yaml(self._raw))

    async def _render_kept_as_is(self) -> None:
        """List the sections this form does not touch, so none is lost unseen."""
        kept = tuple(section for section in self._raw if section not in _CORE_SECTIONS)
        heading = self.query_one(f"#{_KEPT_HEADING_ID}", Static)
        container = self.query_one(f"#{_KEPT_ID}", Container)
        heading.display = bool(kept)
        container.display = bool(kept)
        if kept == self._kept:
            return
        self._kept = kept
        await container.remove_children()
        for section in kept:
            container.mount(
                self._preflight_app.render_text(copy.CONFIG_KEPT_AS_IS_LINE.format(section=section))
            )

    def _update_save_state(self) -> None:
        """Switch saving off while anything needs fixing (D-13, SC3)."""
        button = self.query_one(f"#{_SAVE_ID}", Button)
        button.disabled = bool(self._errors) or not self._raw
        reason = "" if not button.disabled else copy.CONFIG_SAVE_DISABLED_REASON
        self.query_one(f"#{_SAVE_REASON_ID}", Static).update(reason)

    def _update_continue_state(self) -> None:
        """Allow the optional closing step only once a config file exists (D-09).

        The download step starts from the validated YAML, so the way on stays
        switched off until a save has actually written that file; the path is
        the one the core reported, never one this panel builds.
        """
        written = self._preflight_app.state.export_path is not None
        self.query_one(f"#{_CONTINUE_ID}", Button).disabled = not written

    def _set_status(self, text: str) -> None:
        """Replace the status line with *text*, rendered literally."""
        self.query_one(f"#{_STATUS_ID}", Static).update(text)
