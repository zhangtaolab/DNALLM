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
    "ENV_CHECK_FAILED",
    "ENV_DEVICE_SECTION",
    "ENV_DISK_SECTION",
    "ENV_PANEL_TITLE",
    "ENV_ROW_LABELS",
    "ENV_VERSIONS_SECTION",
    "INSTALL_HINT",
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
