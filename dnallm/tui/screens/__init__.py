"""The wizard's screens — one screen per panel.

Each screen renders the report objects the core produced and never computes a
verdict of its own.
"""

from .config import ConfigScreen
from .data import DataScreen
from .entry import EntryScreen
from .env import EnvScreen
from .suitability import SuitabilityScreen

__all__ = ["ConfigScreen", "DataScreen", "EntryScreen", "EnvScreen", "SuitabilityScreen"]
