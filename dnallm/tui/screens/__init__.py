"""The wizard's screens — one screen per panel.

Each screen renders the report objects the core produced and never computes a
verdict of its own.
"""

from .data import DataScreen
from .env import EnvScreen

__all__ = ["DataScreen", "EnvScreen"]
