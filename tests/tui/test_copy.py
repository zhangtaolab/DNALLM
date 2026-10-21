"""Unit tests for the copy layer: the config-error mapper and the gloss tables.

The mapper is the one place a core ``SectionError`` becomes a sentence a
biologist can act on, so it is tested as a table: every classifier kind, every
exact entry, both field conventions the core emits, and — most important — the
case where nothing matches and the core's own message is returned unchanged.
Three assertions are built from the live core (``generate_config``) rather
than a hand-written string, so a future change of field convention fails here
instead of leaking raw jargon to the screen (GAP A).
"""

from __future__ import annotations

import pytest

from dnallm.configuration.configs import TaskConfig
from dnallm.preflight.config_gen import generate_config
from dnallm.tui import copy

# The reviewed eight task types and one plain gloss each (G-25-4). Pinned here
# test-locally so the bytes below are checked against an independent record of
# the reviewed wording, never re-derived from the copy module's own mapping.
_TASK_TYPE_GLOSSES = {
    "embedding": "a list of numbers for a whole sequence, to compare or search",
    "mask": "fill in DNA letters that were hidden from the model",
    "generation": "write out new DNA letters",
    "binary": "one yes/no answer for the whole sequence",
    "multiclass": "one answer chosen from several classes",
    "multilabel": "several yes/no answers at once",
    "regression": "a number, such as a measured value",
    "token": "one label for each DNA letter",
}
_TASK_TYPE_CHOICES = tuple(f"{value} ({gloss})" for value, gloss in _TASK_TYPE_GLOSSES.items())
_GLOSS_PROS = f"{', '.join(_TASK_TYPE_CHOICES[:-1])} or {_TASK_TYPE_CHOICES[-1]}"

# The four locked sentences (D-10). Pinned byte-for-byte: a rewording here is a
# deliberate copy decision, never a side effect of a refactor.
A1 = (
    "Choose one of: "
    + _GLOSS_PROS
    + ". The longer spellings such as binary_classification also work."
)
A2 = "This model has no built-in task, so choose one: " + _GLOSS_PROS + "."
A3 = (
    "Multiclass and multilabel training need at least 2 outputs, but this config "
    "sets fewer. Change Number of outputs in the file."
)
A4 = (
    "This model name is not in the model list. Check the spelling, or go back and "
    "choose a model there."
)

# The core's real message shapes, one per classifier kind. The messages mirror
# what dnallm/preflight/config_gen.py and Pydantic actually produce; each case
# also names the kind it must classify as, so the table covers every fragment
# in copy._CONFIG_ERROR_KINDS exactly once.
_ERROR_CASES = (
    (
        "task_type_required",
        "task",
        "task_type",
        "TASK_TYPE_REQUIRED: zhangtaolab/plant-dnabert-BPE is a base model and ships "
        "no task; pass task_type explicitly",
        A2,
    ),
    (
        "model_not_in_registry",
        "model",
        "model",
        "MODEL_NOT_IN_REGISTRY_CFG: no/such-model-xyz not found in the model registry",
        A4,
    ),
    (
        "unknown_section",
        "bogus",
        "bogus",
        "UNKNOWN_SECTION: 'bogus' is not a recognized config section",
        'The section "bogus" is not one this file understands. Remove it from the file.',
    ),
    (
        "no_sections",
        "config",
        "sections",
        "NO_SECTIONS: no recognized config sections found",
        "This file has no config sections this wizard understands. Check that it is a "
        "DNALLM config file.",
    ),
    (
        "too_few_outputs",
        "task",
        "task",
        "Value error, num_labels must be at least 2 for multiclass classification",
        A3,
    ),
    (
        "not_a_section",
        "lora",
        "lora",
        "section 'lora' patch must be a mapping, got str",
        "lora: this part of the file should be a block of settings, not a single value.",
    ),
    (
        "not_an_allowed_value",
        "task",
        "task.task_type",
        "String should match pattern '^(embedding|mask|generation|binary|"
        "binary_classification|multiclass|multi_class_classification|multilabel|"
        "multi_label_classification|regression|token|token_classification)$'",
        A1,
    ),
    (
        "not_a_whole_number",
        "finetune",
        "finetune.num_train_epochs",
        "Input should be a valid integer, unable to parse string as an integer",
        "Epochs needs a whole number, for example 3.",
    ),
    (
        "not_a_number",
        "finetune",
        "finetune.learning_rate",
        "Input should be a valid number, unable to parse string as a number",
        "Learning rate needs a number, for example 0.001.",
    ),
    (
        "not_a_list",
        "task",
        "task.label_names",
        "Input should be a valid list",
        "Output names needs a list separated by commas, for example query, value.",
    ),
    (
        "not_yes_or_no",
        "finetune",
        "finetune.some_flag",
        "Input should be a valid boolean",
        "finetune.some_flag needs yes or no.",
    ),
    (
        "required",
        "task",
        "task.task_type",
        "Field required",
        "Task type cannot be empty.",
    ),
    (
        "too_small",
        "finetune",
        "finetune.learning_rate",
        "Input should be greater than 0",
        "Learning rate is below the smallest value allowed. Raise it until this line disappears.",
    ),
    (
        "too_large",
        "finetune",
        "finetune.learning_rate",
        "Input should be less than 1",
        "Learning rate is above the largest value allowed. Lower it until this line disappears.",
    ),
)


@pytest.mark.parametrize(
    ("section", "field", "message", "expected"),
    [case[1:] for case in _ERROR_CASES],
    ids=[case[0] for case in _ERROR_CASES],
)
def test_every_kind_maps_to_its_locked_sentence(
    section: str, field: str, message: str, expected: str
) -> None:
    """Each classifier kind produces the sentence the copy table locks."""
    assert copy.config_error_message(section, field, message) == expected


def test_the_case_table_covers_every_classifier_kind() -> None:
    """No fragment can shadow an untested kind: the table is the whole list."""
    assert {kind for _fragment, kind in copy._CONFIG_ERROR_KINDS} == {
        case[0] for case in _ERROR_CASES
    }


@pytest.mark.parametrize(
    ("section", "dotted", "bare", "message", "expected"),
    [
        (
            "task",
            "task.task_type",
            "task_type",
            "String should match pattern '^(embedding|mask)$'",
            A1,
        ),
        (
            "task",
            "task.task_type",
            "task_type",
            "TASK_TYPE_REQUIRED: org/base is a base model and ships no task",
            A2,
        ),
        (
            "model",
            "model.model",
            "model",
            "MODEL_NOT_IN_REGISTRY_CFG: org/nope not found in the model registry",
            A4,
        ),
    ],
    ids=["pattern", "task_type_required", "model_not_in_registry"],
)
def test_the_dotted_and_bare_field_forms_map_identically(
    section: str, dotted: str, bare: str, message: str, expected: str
) -> None:
    """Both conventions the core emits reach one key — the regression tripwire.

    The core's Pydantic path reports ``task.task_type`` while its hand-built
    errors report ``task_type`` and ``model``. Keying the table on either raw
    form alone would silently leak raw jargon for the other.
    """
    assert copy.config_error_message(section, dotted, message) == expected
    assert copy.config_error_message(section, bare, message) == expected


def test_no_exact_entry_stores_a_dotted_field() -> None:
    """Keys are section-relative, so a dotted duplicate can never half-win."""
    dotted = [key for key in copy.CONFIG_ERROR_MESSAGES if key[1].startswith(f"{key[0]}.")]
    assert dotted == []


def test_a_labelled_field_uses_its_friendly_label() -> None:
    """A generic kind speaks in the label the form shows, not the raw key."""
    message = "Input should be a valid number, unable to parse string as a number"
    plain = copy.config_error_message("finetune", "finetune.learning_rate", message)
    assert plain == "Learning rate needs a number, for example 0.001."
    assert "learning_rate" not in plain


def test_an_unlabelled_field_uses_its_reported_path() -> None:
    """A field with no friendly label is named by the path the core reported."""
    message = "Input should be a valid integer, unable to parse string as an integer"
    assert copy.config_error_message("finetune", "finetune.max_grad_norm", message) == (
        "finetune.max_grad_norm needs a whole number, for example 3."
    )


def test_an_unrecognised_message_is_returned_verbatim() -> None:
    """The mapper falls back, it never invents: unknown wording passes through."""
    message = "something the core invented"
    assert copy.config_error_message("task", "task.unknown", message) == message


def test_a_known_kind_with_no_exact_entry_and_no_template_is_verbatim() -> None:
    """A reason code on an unexpected field keeps the core's own wording."""
    message = "MODEL_NOT_IN_REGISTRY_CFG: other/thing not found in the model registry"
    assert copy.config_error_message("model", "other", message) == message


def test_a_field_simply_named_like_its_section_is_not_stripped() -> None:
    """Only a ``"<section>."`` prefix is stripped, never a bare section name."""
    message = "String should match pattern '^(all|none)$'"
    plain = copy.config_error_message("lora", "lora_alpha", message)
    assert plain == (
        "LoRA alpha is not one of the allowed values. Choose one of the values this "
        "setting accepts."
    )


def test_the_detail_line_keeps_the_core_message_verbatim() -> None:
    """The on-demand detail line embeds the raw message and nothing else."""
    raw = "String should match pattern '^(embedding|mask)$'"
    detail = copy.CONFIG_ERROR_DETAIL_LINE.format(message=raw)
    assert "detail from the check:" in detail
    assert detail.endswith(raw)


# ── Real-path assertions (the core builds the message, not this file) ───────


def test_a_base_model_without_a_task_reports_and_maps_its_bare_field() -> None:
    """The hand-built TASK_TYPE_REQUIRED error arrives bare and maps to A2."""
    generated = generate_config("zhangtaolab/plant-dnabert-BPE")
    assert generated.errors
    error = generated.errors[0]
    assert (error.section, error.field) == ("task", "task_type")
    assert copy.config_error_message(error.section, error.field, error.message) == A2


def test_an_unknown_model_reports_and_maps_the_model_section() -> None:
    """The MODEL_NOT_IN_REGISTRY_CFG error arrives as ("model", "model") → A4."""
    generated = generate_config("no/such-model-xyz")
    assert generated.errors
    error = generated.errors[0]
    assert (error.section, error.field) == ("model", "model")
    assert copy.config_error_message(error.section, error.field, error.message) == A4


def test_a_bad_task_type_reports_a_dotted_field_and_maps_to_a1() -> None:
    """The Pydantic path arrives dotted (task.task_type) and maps to A1."""
    generated = generate_config("zhangtaolab/plant-dnabert-BPE", task_type="not-a-task")
    assert generated.errors
    error = generated.errors[0]
    assert (error.section, error.field) == ("task", "task.task_type")
    assert copy.config_error_message(error.section, error.field, error.message) == A1


# ── Gloss inventory (GAP B, D-10) ──────────────────────────────────────────

# Every term the UAT named as unexplained. The inventory is exact: adding a
# hint means adding it here, so the phase-end copy review stays one read.
_HINTED_FIELDS = (
    ("task", "task_type"),
    ("task", "threshold"),
    ("finetune", "num_train_epochs"),
    ("finetune", "per_device_train_batch_size"),
    ("finetune", "learning_rate"),
    ("finetune", "weight_decay"),
    ("finetune", "warmup_ratio"),
    ("inference", "device"),
    ("lora", "r"),
    ("lora", "lora_alpha"),
    ("lora", "lora_dropout"),
    ("lora", "target_modules"),
)


def test_every_glossed_field_has_a_plain_hint() -> None:
    """The hint table is exactly the inventory, and every sentence is non-empty."""
    assert set(copy.CONFIG_FIELD_HINTS) == set(_HINTED_FIELDS)
    assert all(copy.CONFIG_FIELD_HINTS[key].strip() for key in _HINTED_FIELDS)


def test_the_glossed_headings_and_the_inference_batch_label() -> None:
    """The two headings the UAT named carry their gloss; inference batch does too."""
    assert copy.CONFIG_SECTION_INFERENCE.endswith("(making predictions)")
    assert copy.CONFIG_SECTION_LORA.endswith("(lightweight fine-tuning)")
    assert copy.CONFIG_FIELD_LABELS["inference", "batch_size"] == (
        "Sequences read at once when predicting"
    )


def test_the_task_type_vocabulary_is_the_ordered_eight() -> None:
    """One ordered mapping, in the core's pattern order, every gloss non-empty."""
    assert list(copy.TASK_TYPE_GLOSSES) == list(_TASK_TYPE_GLOSSES)
    assert all(gloss.strip() for gloss in copy.TASK_TYPE_GLOSSES.values())


def test_the_vocabulary_never_drifts_from_the_core_pattern() -> None:
    """The copy's eight equal the live core's canonical values (T-25-21).

    The canonical eight exist only as the regex on ``TaskConfig.task_type``
    (``dnallm/configuration/configs.py:109``); the four long spellings
    (binary_classification, multi_class_classification,
    multi_label_classification, token_classification) are canonicalised in
    ``model_post_init``. Reading the live JSON-schema pattern keeps the copy in
    step with the core without exporting a constant from it, so a core-side
    addition fails here instead of silently misleading a reader.
    """
    pattern = TaskConfig.model_json_schema()["properties"]["task_type"]["pattern"]
    allowed = pattern.removeprefix("^(").removesuffix(")$").split("|")
    canonical = {spelling for spelling in allowed if not spelling.endswith("classification")}
    assert canonical == set(copy.TASK_TYPE_GLOSSES)


def _task_type_string(site: str) -> str:
    """One of the four places a task type is offered, by its test-site name.

    The static legend is this plan's new rendering surface; before it exists,
    the completeness claim is held against the placeholder that carried the
    vocabulary (the UAT's "vanishes on typing" site), so every site is judged
    on the sentences that ship today.
    """
    if site == "legend":
        return getattr(copy, "SUITABILITY_TASK_LEGEND", copy.SUITABILITY_TASK_PROMPT)
    if site == "hint":
        return copy.CONFIG_FIELD_HINTS["task", "task_type"]
    return copy.CONFIG_ERROR_MESSAGES[("task", "task_type", site)]


@pytest.mark.parametrize("site", ["legend", "hint", "not_an_allowed_value", "task_type_required"])
def test_every_task_type_string_names_and_explains_all_eight(site: str) -> None:
    """Every site names all eight values and explains all eight — no partial list."""
    sentence = _task_type_string(site)
    missing_values = [value for value in _TASK_TYPE_GLOSSES if value not in sentence]
    missing_glosses = [gloss for gloss in _TASK_TYPE_GLOSSES.values() if gloss not in sentence]
    assert missing_values == []
    assert missing_glosses == []


def test_the_task_type_hint_is_the_locked_sentence() -> None:
    """The field hint names what the model should do and then all eight choices."""
    assert copy.CONFIG_FIELD_HINTS["task", "task_type"] == (
        "What the model should do. Choose one of: " + _GLOSS_PROS + "."
    )


def test_the_legend_is_composed_from_the_same_vocabulary() -> None:
    """The legend's body is the mapping's eight lines; header and closer are fixed."""
    lines = copy.SUITABILITY_TASK_LEGEND.splitlines()
    assert lines[1:-1] == [
        f"  {value} — {gloss}" for value, gloss in copy.TASK_TYPE_GLOSSES.items()
    ]
    assert lines[0] == "The eight task types:"
    assert lines[-1] == "Leave it empty if you are not sure."
    # 74 is the width #suitability-body gives the legend at 80x24 once its
    # scrollbar shows (measured), so a longer gloss would wrap and silently add
    # rows to the panel's smallest-screen layout.
    assert max(len(line) for line in lines) <= 74
