"""The wizard's screens — one screen per panel.

Each screen renders the report objects the core produced and never computes a
verdict of its own.
"""

from .config import ConfigScreen
from .data import DataScreen
from .download import DownloadScreen
from .entry import EntryScreen
from .env import EnvScreen
from .suitability import SuitabilityScreen
from .summary import SummaryScreen

__all__ = [
    "ConfigScreen",
    "DataScreen",
    "DownloadScreen",
    "EntryScreen",
    "EnvScreen",
    "SuitabilityScreen",
    "SummaryScreen",
]
