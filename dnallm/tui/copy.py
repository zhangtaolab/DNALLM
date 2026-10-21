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
    "CHECKING_PLACEHOLDER",
    "DATA_CHARSET_HEADING",
    "DATA_CHARSET_ROW",
    "DATA_CHECK_ACTION",
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
    "INSTALL_HINT",
    "NO_DATA_PROVIDED",
    "QUIT_HINT",
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
