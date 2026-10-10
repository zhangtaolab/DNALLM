"""Bracket-member guard tests for the pyproject extras (REPAIR-04, 08-02).

The example lane's isolated ``dnallm-mcp-langchain`` kernelspec installs
``langchain-ollama`` inside its throwaway venv, and the langchain mcp
notebook's verified output ran against 1.1.0 -- declaring it in the
project-surface ``mcp`` extra is the REPAIR-04 statement of that fact
(08-RESEARCH "Package Legitimacy Audit": PyPI-verified official LangChain
integration package, notebook-output-matched version).

The ``notebook``-extra guards (08-02, REPAIR-01) pin the kernel-plot
compatibility pair: ``pygenometracks>=3.9`` caps ``matplotlib<3.9``, whose
``install_repl_displayhook`` imports
``IPython.core.pylabtools.backend2gui`` unguarded -- a name IPython 9
removed, killing the first ``plt.subplots`` in every notebook kernel
(regression: embedding_attention.ipynb cell 9, 2026-10-04, after the pgt
install downgraded matplotlib 3.11.2 -> 3.8.4 under IPython 9.17.1).

These tests fail when a future edit drops a member, silently loses a
pre-existing one, or resolves the installed pair back into the broken
half. Declarations supplement -- never replace -- the venv isolation in
``tests/examples/_execution.py``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import IPython
import IPython.core.pylabtools
import matplotlib
import matplotlib.pyplot as plt
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The five pre-existing mcp-extra members (pyproject.toml before
# REPAIR-04); none may be lost or reordered away by the declaration edit.
EXPECTED_MCP_MEMBERS = frozenset({
    "mcp>=1.0.0,<2",
    "langchain>=1.3.6",
    "langchain_mcp_adapters>=0.2.1",
    "nest-asyncio>=1.5.9",
    # 2026-10-10: tightened from <3 to the 1.107 series (silent backtrack to
    # 1.22.0 broke fresh CI resolves); the guard tracks the declared surface.
    "pydantic-ai>=1.107.0,<2",
})

# The notebook-extra members before the 08-02 ipython pin; none may be
# lost by the compatibility edit.
EXPECTED_NOTEBOOK_MEMBERS = frozenset({
    "jupyter>=1.1.1",
    "marimo>=0.16.3",
    "nbclient>=0.10",
    "pygenometracks>=3.9; platform_system != 'Windows'",
})


def _load_mcp_extra() -> list[str]:
    """Parse pyproject.toml with tomllib and return the mcp extra members."""
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data["project"]["optional-dependencies"]["mcp"])


def _load_notebook_extra() -> list[str]:
    """Parse pyproject.toml with tomllib and return the notebook extra members."""
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data["project"]["optional-dependencies"]["notebook"])


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


class TestNotebookExtraMembers:
    """Bracket-member assertions over project.optional-dependencies["notebook"]."""

    def test_ipython_compat_pin_declared_in_notebook_extra(self) -> None:
        """The notebook extra pins ipython>=8.31,<9 (kernel-plot pair, 08-02)."""
        notebook = _load_notebook_extra()
        pins = [member for member in notebook if member.startswith("ipython")]
        assert any("<9" in member for member in pins), (
            f"notebook extra lost the ipython<9 compatibility pin (08-02 REPAIR-01): "
            f"{notebook} -- pygenometracks>=3.9 caps matplotlib<3.9, whose "
            f"install_repl_displayhook imports IPython backend2gui unguarded, "
            f"removed in IPython 9"
        )

    def test_pre_existing_members_preserved(self) -> None:
        """No pre-existing notebook member was lost by the ipython pin edit."""
        notebook = _load_notebook_extra()
        missing = EXPECTED_NOTEBOOK_MEMBERS - set(notebook)
        assert not missing, f"notebook extra lost pre-existing members: {sorted(missing)}"


class TestNotebookKernelPlotCompat:
    """Installed-pair guard for plt figures inside notebook kernels (08-02).

    Static extras guards cannot see what the resolver actually installed,
    so this probes the live environment: the pair (IPython, matplotlib)
    must let ``install_repl_displayhook`` survive its first figure. It
    fails red on the broken half (IPython 9.x + matplotlib <3.9) and stays
    green on either healthy pairing (IPython 8.x, or matplotlib >=3.9
    guarding/dropping the backend2gui import).
    """

    def test_installed_ipython_matplotlib_pair_is_kernel_plot_safe(self) -> None:
        """backend2gui must exist or matplotlib must not import it unguarded."""
        if hasattr(IPython.core.pylabtools, "backend2gui"):
            return  # IPython < 9: the name matplotlib imports still exists

        hook = getattr(plt, "install_repl_displayhook", None)
        if hook is None:
            return  # newer matplotlib dropped the REPL display hook entirely

        source = inspect.getsource(hook)
        if "backend2gui" not in source:
            return  # this matplotlib never touches the removed name
        assert "except ImportError" in source, (
            f"broken kernel-plot pair: IPython {IPython.__version__} removed "
            f"IPython.core.pylabtools.backend2gui while matplotlib "
            f"{matplotlib.__version__} imports it unguarded in "
            f"install_repl_displayhook -- the first plt.subplots in every "
            f"notebook kernel dies with ImportError (08-02 REPAIR-01; pin "
            f"ipython<9 in the notebook extra or move matplotlib past 3.9)"
        )
