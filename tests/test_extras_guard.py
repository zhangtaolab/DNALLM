"""Bracket-member guard tests for the pyproject ``mcp`` extra (REPAIR-04).

The example lane's isolated ``dnallm-mcp-langchain`` kernelspec installs
``langchain-ollama`` inside its throwaway venv, and the langchain mcp
notebook's verified output ran against 1.1.0 -- declaring it in the
project-surface ``mcp`` extra is the REPAIR-04 statement of that fact
(08-RESEARCH "Package Legitimacy Audit": PyPI-verified official LangChain
integration package, notebook-output-matched version).

These tests fail when a future edit drops the member or silently loses one
of the five pre-existing members. The declaration supplements the venv
isolation in ``tests/examples/_execution.py`` -- it never replaces it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The five pre-existing mcp-extra members (pyproject.toml before
# REPAIR-04); none may be lost or reordered away by the declaration edit.
EXPECTED_MCP_MEMBERS = frozenset(
    {
        "mcp>=1.0.0,<2",
        "langchain>=1.3.6",
        "langchain_mcp_adapters>=0.2.1",
        "nest-asyncio>=1.5.9",
        "pydantic-ai<3",
    }
)


def _load_mcp_extra() -> list[str]:
    """Parse pyproject.toml with tomllib and return the mcp extra members."""
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data["project"]["optional-dependencies"]["mcp"])


class TestMcpExtraMembers:
    """Bracket-member assertions over project.optional-dependencies["mcp"]."""

    def test_langchain_ollama_declared_in_mcp_extra(self) -> None:
        """langchain-ollama>=1.1.0 is an exact member of the mcp extra."""
        mcp = _load_mcp_extra()
        assert "langchain-ollama>=1.1.0" in mcp, (
            f"langchain-ollama>=1.1.0 missing from the mcp extra (REPAIR-04): {mcp}"
        )

    def test_pre_existing_members_preserved(self) -> None:
        """No pre-existing mcp member was lost by the declaration edit."""
        mcp = _load_mcp_extra()
        missing = EXPECTED_MCP_MEMBERS - set(mcp)
        assert not missing, f"mcp extra lost pre-existing members: {sorted(missing)}"
