"""Version-source contract: dnallm.version must track pyproject.toml.

The 2026-10-19 v1.3.1/v1.3.2 bumps missed dnallm/version.py (the version
lives in THREE places: pyproject.toml, dnallm/version.py, and the editable
install's dist-info), so ``dnallm.__version__`` kept reporting 1.3.0 after
two releases. This gate fails the suite the moment the sources drift apart.
"""

import tomllib
from pathlib import Path

import dnallm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dunder_version_matches_pyproject():
    """dnallm.__version__ must equal the pyproject project.version."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject_version = tomllib.load(fh)["project"]["version"]
    assert dnallm.__version__ == pyproject_version, (
        f"dnallm/version.py ({dnallm.__version__}) drifted from "
        f"pyproject.toml ({pyproject_version}) — bump both together"
    )
