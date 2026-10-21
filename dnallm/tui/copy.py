"""Every user-facing string the pre-flight wizard shows (D-10).

Copy lives in one module for two reasons: the phase-end plain-language review
is a single-file read, and tests can assert on the exact sentence a user sees
instead of re-deriving it.

Every sentence is written for a biologist: no exception class names, no
reason-code spelling, no jargon. Technical identifiers (a version string, a
model name, a folder path) are shown verbatim.
"""

__all__ = [
    "APP_TITLE",
    "AVAILABILITY_CACHED_HUB",
    "AVAILABILITY_CACHED_LOCAL",
    "AVAILABILITY_DETAIL_LABEL",
    "AVAILABILITY_REMOTE",
    "CHECKING_PLACEHOLDER",
    "CONFIG_BUILDING",
    "CONFIG_DEFAULT_EXPORT_NAME",
    "CONFIG_EDITING",
    "CONFIG_ERRORS_HEADING",
    "CONFIG_ERROR_LINE",
    "CONFIG_FIELD_LABELS",
    "CONFIG_HAS_ERRORS",
    "CONFIG_KEPT_AS_IS_HEADING",
    "CONFIG_KEPT_AS_IS_LINE",
    "CONFIG_NO_SOURCE",
    "CONFIG_OUTPUT_PATH_LABEL",
    "CONFIG_OUTPUT_PATH_PROMPT",
    "CONFIG_PANEL_TITLE",
    "CONFIG_PREVIEW_HEADING",
    "CONFIG_READONLY_FACTS_NOTE",
    "CONFIG_READY",
    "CONFIG_SAVED",
    "CONFIG_SAVE_ACTION",
    "CONFIG_SAVE_DISABLED_REASON",
    "CONFIG_SAVING",
    "CONFIG_SECTION_FINETUNE",
    "CONFIG_SECTION_INFERENCE",
    "CONFIG_SECTION_LORA",
    "CONFIG_SECTION_MODEL",
    "CONFIG_SECTION_TASK",
    "DATA_CHARSET_HEADING",
    "DATA_CHARSET_ROW",
    "DATA_CHECK_ACTION",
    "DATA_CONTINUE_ACTION",
    "DATA_COUNT_NOT_APPLICABLE",
    "DATA_FINDINGS_COLUMNS",
    "DATA_FINDINGS_HEADING",
    "DATA_PANEL_TITLE",
    "DATA_PATH_PROMPT",
    "DATA_READY",
    "DATA_SKIP_ACTION",
    "DATA_SKIP_NOTE",
    "DATA_SPLIT_ROW",
    "DATA_SPLIT_ROW_EMPTY",
    "DATA_SUMMARY_HEADING",
    "EMPTY_PATH_PROMPT",
    "ENV_CHECK_FAILED",
    "ENV_CONTINUE_ACTION",
    "ENV_DEVICE_SECTION",
    "ENV_DISK_SECTION",
    "ENV_PANEL_TITLE",
    "ENV_ROW_LABELS",
    "ENV_VERSIONS_SECTION",
    "HOSTING_NOT_VERIFIED",
    "HUB_NAMES",
    "INSTALL_HINT",
    "LAB_ROW_FLAG",
    "NO_DATA_PROVIDED",
    "NO_DATA_YET",
    "NO_FINDINGS",
    "NO_HEAD_LABELS",
    "PROBING_PLACEHOLDER",
    "QUIT_HINT",
    "SELECT_MODEL_HINT",
    "SUITABILITY_CHECK_ACTION",
    "SUITABILITY_FAILED",
    "SUITABILITY_INTENT_FINETUNE",
    "SUITABILITY_INTENT_PREDICT",
    "SUITABILITY_LISTING_COLUMNS",
    "SUITABILITY_LISTING_HEADING",
    "SUITABILITY_MODELS_PROMPT",
    "SUITABILITY_PANEL_TITLE",
    "SUITABILITY_READY",
    "SUITABILITY_TASK_PROMPT",
    "SUITABILITY_VERDICTS_HEADING",
    "SUITABILITY_VERDICT_COLUMNS",
    "THIRD_PARTY_ROW_FLAG",
    "UNKNOWN_STILL_DOWNLOADABLE",
    "USE_MODEL_ACTION",
    "USE_MODEL_CONFIRMED",
]

APP_TITLE = "DNALLM pre-flight check"

INSTALL_HINT = (
    "The terminal wizard needs an optional package that is not installed yet. "
    'Install it with: pip install "dnallm[tui]"'
)

QUIT_HINT = "Press q at any time to quit."

CHECKING_PLACEHOLDER = "checking…"

ENV_PANEL_TITLE = "What this computer can do"

ENV_VERSIONS_SECTION = "Software versions"

ENV_DEVICE_SECTION = "Graphics card and memory"

ENV_DISK_SECTION = "Free disk space for model downloads"

# Labels for the two placeholder rows the env panel paints before its worker
# reports. The real rows carry their own labels from dnallm.utils.env_info.
ENV_ROW_LABELS = {
    "device": ENV_DEVICE_SECTION,
    "disk": ENV_DISK_SECTION,
}

ENV_CHECK_FAILED = (
    "We could not finish checking the graphics card and the free disk space. "
    "The rest of this page still works and you can keep going."
)

ENV_CONTINUE_ACTION = "Continue to your data"

# ── Data panel (TUI-04) ──────────────────────────────────────────────────────
# Plain language for a biologist; the technical identifiers the panel shows
# beside these sentences (reason codes, file paths, characters) stay verbatim.

DATA_PANEL_TITLE = "Your data"

DATA_PATH_PROMPT = "Paste the full path to your data file (CSV, TSV, JSON, FASTA or TXT)."

DATA_CHECK_ACTION = "Check this data"

# The step is optional (D-08), but the way forward must always be one click away.
DATA_CONTINUE_ACTION = "Continue to models"

DATA_SKIP_ACTION = "I don't have data yet"

# D-08: the step is optional, and skipping states the consequence plainly.
DATA_SKIP_NOTE = (
    "No data is fine. The next page will judge models from the model and your "
    "task alone. The length and character checks stay empty until you add data."
)

NO_DATA_PROVIDED = "No data provided yet."

EMPTY_PATH_PROMPT = 'Type the full path to your data file first, then press "Check this data".'

DATA_READY = "We read your data. Here is what we found."

DATA_SUMMARY_HEADING = "What is in your data"

DATA_CHARSET_HEADING = "Characters used"

DATA_FINDINGS_HEADING = "Problems and notes"

# Shown in place of the findings table when the core reported nothing at all.
NO_FINDINGS = "No problems found in this data."

# Column labels for the findings table; the reason code column stays verbatim.
DATA_FINDINGS_COLUMNS = ("How serious", "Reason code", "What this means", "Rows affected")

# Shown when a finding has no exact count (the core reports None, not zero).
DATA_COUNT_NOT_APPLICABLE = "not counted"

# One row per split. "typical" is the median length the core computed (p50).
DATA_SPLIT_ROW = (
    'Split "{name}": {row_count} sequences; '
    "shortest {shortest}, typical {typical}, longest {longest}"
)

DATA_SPLIT_ROW_EMPTY = 'Split "{name}": 0 sequences; nothing to measure.'

# One row per character the core counted, with the split it belongs to.
DATA_CHARSET_ROW = 'Split "{split}": the character "{char}" appears {count} times'

# ── Suitability panel (TUI-05) ───────────────────────────────────────────────
# The two routes over the base<->finetuned axis. These are the WORDS the user
# reads; the core's own vocabulary ("finetune" / "predict") never reaches the
# screen, and these sentences never reach the core.

SUITABILITY_PANEL_TITLE = "Which model should do the job?"

SUITABILITY_INTENT_FINETUNE = "Fine-tune a base model on my data"

SUITABILITY_INTENT_PREDICT = "Predict now with a ready-made model"

SUITABILITY_TASK_PROMPT = (
    "Task type, when you already know it (for example binary, multiclass or token)."
)

SUITABILITY_MODELS_PROMPT = (
    "Optional: type model names separated by commas to check them one by one."
)

SUITABILITY_CHECK_ACTION = "Check these models"

SUITABILITY_VERDICTS_HEADING = "Models you asked about"

SUITABILITY_LISTING_HEADING = "Models that fit this route"

# Column labels. Reason codes, task types and model names stay verbatim.
SUITABILITY_VERDICT_COLUMNS = ("Model", "Check", "Result", "Reason code")

SUITABILITY_LISTING_COLUMNS = (
    "Model",
    "Who made it",
    "Task type",
    "Trained outputs",
    "Availability",
)

# Shown while the core is working out the listing and its first probes.
PROBING_PLACEHOLDER = "probing…"

SUITABILITY_READY = (
    "Here is what fits. Use the arrow keys to pick a model, then choose "
    '"Use this model". Press p on a row to check where that model is hosted.'
)

SUITABILITY_FAILED = (
    "We could not work out which models fit this data. Your data is safe and "
    "you can try again or keep going with what is already on this page."
)

# D-05: an unprobed row says exactly what is known — nothing more.
HOSTING_NOT_VERIFIED = "hosting not verified"

NO_HEAD_LABELS = "no trained outputs"

LAB_ROW_FLAG = "lab model"

THIRD_PARTY_ROW_FLAG = "third party"

# Availability sentences. The hub name is inserted verbatim.
AVAILABILITY_CACHED_LOCAL = "already on this computer"

AVAILABILITY_CACHED_HUB = "already downloaded from {hub}"

AVAILABILITY_REMOTE = "available on {hub}"

# How each hub is named to a reader. Keys are the core's Hub values.
HUB_NAMES = {
    "modelscope": "ModelScope",
    "huggingface": "Hugging Face",
    "local": "this computer",
    "none": "no hub",
}

# Prefix for the detail line that shows a probe's own reason, verbatim.
AVAILABILITY_DETAIL_LABEL = "Note from the check:"

# D-06: an unverified row is never an unavailable row. The sentence says so,
# because a blank column would let a user read UNKNOWN as "cannot download".
UNKNOWN_STILL_DOWNLOADABLE = (
    "We have not confirmed where this model is hosted. You can still choose "
    "it: the download tries ModelScope first, then Hugging Face."
)

# D-08: the length and character checks need data; without it the state is
# stated plainly and never shown as a failure.
NO_DATA_YET = "no data provided yet"

SELECT_MODEL_HINT = "Select a model with the arrow keys, then choose 'Use this model'."

USE_MODEL_ACTION = "Use this model"

USE_MODEL_CONFIRMED = 'You chose "{model}". The next step builds its config.'

# ── Config panel (TUI-06) ───────────────────────────────────────────────────
# D-13: the form builds a validated config, and no invalid file is ever
# written. Every sentence a user reads is here; the identifiers the core
# reports (section paths, field values) stay verbatim.

CONFIG_PANEL_TITLE = "Your config"

CONFIG_BUILDING = "building your config…"

CONFIG_READY = "This config is ready. You can save it when you are happy with it."

CONFIG_HAS_ERRORS = (
    "This config needs a fix first. The list on the right says what to change; "
    "saving stays switched off until the list is empty."
)

CONFIG_ERRORS_HEADING = "What needs fixing"

# One error line: the exact field path, then the core's own plain message.
CONFIG_ERROR_LINE = "{field} → {message}"

CONFIG_PREVIEW_HEADING = "The file this would save"

# D-12: the checkpoint decides these; an editable field would be silently
# overridden at load time anyway (registry always wins).
CONFIG_READONLY_FACTS_NOTE = "from the model checkpoint; loading uses this value"

# D-11: a section this form does not cover rides along untouched.
CONFIG_KEPT_AS_IS_HEADING = "Kept exactly as they are"

CONFIG_KEPT_AS_IS_LINE = (
    '"{section}" is kept exactly as it is — this form does not change it, '
    "and saving keeps every value in it."
)

# D-13: the write is the one irreversible act in the wizard, so it is gated.
CONFIG_SAVE_ACTION = "Save this config"

CONFIG_SAVE_DISABLED_REASON = (
    'Saving stays off until the list under "What needs fixing" is empty, '
    "so a broken config can never be written."
)

CONFIG_OUTPUT_PATH_LABEL = "Save as:"

CONFIG_DEFAULT_EXPORT_NAME = "dnallm_config.yaml"

CONFIG_OUTPUT_PATH_PROMPT = 'Type the file name to save as, then press "Save this config".'

CONFIG_SAVING = "saving…"

CONFIG_SAVED = 'Saved "{path}". That file is ready to use.'

CONFIG_EDITING = "reading your config…"

CONFIG_NO_SOURCE = (
    "There is nothing to configure yet. Go back a step and choose a model, "
    "or open an existing config file."
)

CONFIG_SECTION_TASK = "Task"

CONFIG_SECTION_MODEL = "Model"

CONFIG_SECTION_FINETUNE = "Training"

CONFIG_SECTION_INFERENCE = "Inference"

CONFIG_SECTION_LORA = "LoRA"

# Field labels, keyed by (section, raw key). Keys not listed show their own
# name verbatim — the core's spelling is the file's spelling.
CONFIG_FIELD_LABELS = {
    ("task", "task_type"): "Task type",
    ("task", "num_labels"): "Number of outputs",
    ("task", "label_names"): "Output names",
    ("task", "threshold"): "Decision threshold",
    ("model", "name"): "Model name",
    ("finetune", "output_dir"): "Where to save the trained model",
    ("finetune", "num_train_epochs"): "Epochs",
    ("finetune", "per_device_train_batch_size"): "Batch size",
    ("finetune", "learning_rate"): "Learning rate",
    ("finetune", "weight_decay"): "Weight decay",
    ("finetune", "warmup_ratio"): "Warm-up share",
    ("finetune", "seed"): "Random seed",
    ("inference", "max_length"): "Longest sequence the model reads",
    ("inference", "batch_size"): "Inference batch size",
    ("inference", "device"): "Device",
    ("lora", "r"): "LoRA rank",
    ("lora", "lora_alpha"): "LoRA alpha",
    ("lora", "lora_dropout"): "LoRA dropout",
    ("lora", "target_modules"): "LoRA target modules",
}
