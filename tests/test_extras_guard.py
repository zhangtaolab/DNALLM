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

# The tui extra's exact members (TUI-01, v1.4): the terminal pre-flight wizard
# layer. Exact-string equality both directions -- the extra must never be
# empty, diluted, or gain an unpinned member, and the platformdirs band must
# stay textual's own declared >=3.6,<5 range so the resolver has one answer.
EXPECTED_TUI_MEMBERS = frozenset({
    "textual>=8.2,<9",
    "platformdirs>=3.6,<5",
})

# The full extra's exact members after the 18-01 atomic rename (base -> full,
# owner decision 2026-10-14): the rename must carry the former base surface
# verbatim -- nothing lost, nothing added.
EXPECTED_FULL_MEMBERS = frozenset({
    "dnallm[dev,test,notebook,mcp]",
    "isort>=6.0.1",
})  # types-transformers removed 2026-10-17: mypy-only stub (ty bundles its
# own typeshed and never reads installed stub packages) — D-09 flip

# WPAR-07 declaration: the exact marker-gated triton-windows spec in the fla
# extra (exact-string assertion; any drift in the band or marker fails).
EXPECTED_TRITON_WINDOWS_SPEC = "triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'"


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


def _load_extras() -> dict[str, list[str]]:
    """Parse pyproject.toml with tomllib and return the whole extras table."""
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return dict(data["project"]["optional-dependencies"])


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


class TestTuiExtraMembers:
    """Bracket-member assertions over project.optional-dependencies["tui"] (TUI-01).

    The ``tui`` extra is the only install gate for the terminal pre-flight
    wizard; ``test`` carries it so every pytest CI leg (which installs a
    ``.[full...]`` variant) resolves textual transitively, and ``all`` names
    it explicitly. Membership is asserted, never position -- reordering any
    of these lists can neither fake a pass nor drop a member.
    """

    def test_tui_extra_members_exact(self) -> None:
        """tui is exactly the two owner-pinned members -- nothing lost, nothing extra."""
        tui = set(_load_extras().get("tui", []))
        missing = EXPECTED_TUI_MEMBERS - tui
        assert not missing, (
            f"tui extra lost pinned members (TUI-01): {sorted(missing)} -- "
            f"textual drives the wizard and platformdirs is its own "
            f">=3.6,<5 constraint made explicit"
        )
        extra_members = tui - EXPECTED_TUI_MEMBERS
        assert not extra_members, (
            f"tui extra gained members beyond the pinned pair (TUI-01): "
            f"{sorted(extra_members)} -- every tui dependency must be an "
            f"owner-reviewed exact range"
        )

    def test_test_extra_rides_tui(self) -> None:
        """dnallm[tui] is a member of the test extra, so every CI pytest leg installs textual."""
        test_extra = _load_extras().get("test", [])
        assert "dnallm[tui]" in test_extra, (
            f"the test extra must carry dnallm[tui] (TUI-01): measured {test_extra} "
            f"-- without it, a .[full] install reaches textual only by accident, "
            f"and the TUI pytest lane fails on a clean runner"
        )

    def test_all_extra_includes_tui(self) -> None:
        """tui appears in the comma-split bracket list parsed out of all[0]."""
        all_extra = _load_extras().get("all", [])
        assert all_extra, "the all extra must declare its bracket list"
        bracket = all_extra[0]
        assert bracket.startswith("dnallm[") and bracket.endswith("]"), (
            f"the all extra's first member must stay a dnallm[...] bracket "
            f"list (measured: {bracket!r})"
        )
        members = [item.strip() for item in bracket[len("dnallm[") : -1].split(",")]
        assert "tui" in members, (
            f"the all extra must include tui (TUI-01): measured members {members} "
            f"from {bracket!r}"
        )


class TestFullExtraAndBaseAlias:
    """18-01 atomic extras surgery guards: full rename, alias retirement, retirements.

    The rename base -> full (owner decision 2026-10-14) landed with the full
    extra carrying the former base members verbatim. The one-cycle alias
    lived through v1.3 and is retired at v1.4 (PKG-01, pre-committed at v1.3
    close): the extras table must carry no ``base`` key at all, so the
    deprecation cannot silently outlive its announced removal. The
    triton-windows declaration stays marker-gated to Windows in fla
    (WPAR-07), and cuda121/rocm stay absent from both the extras table and
    the uv index/sources machinery.
    """

    def test_full_extra_members_preserved(self) -> None:
        """The rename carried the former base member set verbatim."""
        full = _load_extras().get("full", [])
        missing = EXPECTED_FULL_MEMBERS - set(full)
        assert not missing, f"full extra lost former-base members: {sorted(missing)}"
        extra_members = set(full) - EXPECTED_FULL_MEMBERS
        assert not extra_members, (
            f"full extra gained members beyond the former base set: {sorted(extra_members)}"
        )

    def test_base_alias_is_removed(self) -> None:
        """The extras table carries no `base` key (PKG-01, the v1.4 cycle).

        The alias's announced lifetime ended at v1.4. Keeping it would keep
        masking the missing-extra case: without it pip/uv warn that the extra
        does not exist and exit 0 on a core-only under-install, which is the
        honest (if unhelpful) outcome the alias was created to defer, never
        to hide permanently.
        """
        extras = _load_extras()
        assert "base" not in extras, (
            "the deprecated one-cycle base alias reappeared in "
            f"optional-dependencies (PKG-01: removed at v1.4, pre-committed "
            f"at v1.3 close); measured {extras.get('base')!r} -- the alias "
            "silently resolves the full set and so masks the core-only "
            "under-install that a missing extra produces"
        )

    def test_triton_windows_marker_declared_in_fla(self) -> None:
        """fla carries the exact marker-gated triton-windows spec (WPAR-07)."""
        fla = _load_extras().get("fla", [])
        assert EXPECTED_TRITON_WINDOWS_SPEC in fla, (
            f"fla extra lost the exact triton-windows Windows-marker declaration "
            f"(WPAR-07): expected {EXPECTED_TRITON_WINDOWS_SPEC!r} among {fla} -- "
            f"exact band, no floating, Windows-only marker"
        )

    def test_cuda121_and_rocm_extras_absent(self) -> None:
        """cuda121/rocm cannot regress into extras or the uv index machinery."""
        with PYPROJECT.open("rb") as fh:
            data = tomllib.load(fh)
        extras = data["project"]["optional-dependencies"]
        for retired in ("cuda121", "rocm"):
            assert retired not in extras, (
                f"the retired extra {retired!r} reappeared in "
                f"optional-dependencies (18-01 retirement regression)"
            )
        uv_table = data.get("tool", {}).get("uv", {})
        index_names = [entry.get("name") for entry in uv_table.get("index", [])]
        for retired_index in ("torch-cuda121", "torch-rocm"):
            assert retired_index not in index_names, (
                f"the retired uv index {retired_index!r} reappeared in "
                f"[[tool.uv.index]] (18-01 retirement regression)"
            )
        sources = uv_table.get("sources", {}).get("torch", [])
        for source in sources:
            assert source.get("index") not in ("torch-cuda121", "torch-rocm"), (
                f"the retired uv source entry reappeared in [tool.uv.sources]: {source}"
            )


class TestExtrasSurgeryResidueGates:
    """Repo-scan residue gates over the 18-01 surgery surface (phase-13 verbatim discipline).

    The base->full rename, cuda121 retirement, and rocm removal landed as one
    atomic commit; these gates keep the repo at exactly the sanctioned
    residue. Scan scope: pyproject.toml, .github/, docs/, example/,
    README.md, scripts/ -- tests/ excluded (these literals would self-match),
    .planning/ excluded (immutable history). Edge discipline: an EMPTY or
    short hit list is deletion creep (fail), any extra line is a straggler
    (fail), any missing sanctioned line (fail).
    """

    # Pattern 1: bracket install forms of base (.[base...] / dnallm[base...]).
    # The alias declaration itself (base = ["dnallm[full]"]) never matched.
    # Sanctioned: exactly ONE row -- the post-v1.4-removal row of the
    # cross-installer honesty matrix page, which documents the missing-extra
    # outcome honestly. The README/installation deprecation notes and the
    # during-alias-cycle honesty row were deleted with the alias (PKG-01).
    BASE_BRACKET_PATTERN = r"\.\[base|dnallm\[base"
    SANCTIONED_BASE_BRACKET_RESIDUE = frozenset({
        (
            "docs/getting_started/install_honesty.md",
            "| `dnallm[base]` — since the v1.4 alias removal | `warning: The "
            "package ... does not have an extra named 'base'`; installs "
            "**core only**; exit 0 | `WARNING: dnallm X.Y.Z does not provide "
            "the extra 'base'`; installs **core only**; exit 0 | live probe "
            "2026-10-14 (missing-extra case) |",
        ),
    })

    # Pattern 2: cuda121 (any form), torch-rocm (uv index name), bracket-rocm.
    # Sanctioned: exactly the three flash-linear-attention UPSTREAM-vocabulary
    # lines (Pitfall 4: fla's own [cuda]/[rocm] backend extras, never dnallm's
    # retired rocm extra) -- README:255, docs/faq/models_troubleshooting.md:13,
    # and the pyproject fla comment kept verbatim per the same rule.
    CUDA121_ROCM_PATTERN = r"cuda121|torch-rocm|\[rocm\]"
    SANCTIONED_CUDA121_ROCM_RESIDUE = frozenset({
        (
            "pyproject.toml",
            "# ([cuda]/[rocm]), they pin their own torch and would downgrade this environment.",
        ),
        (
            "README.md",
            "Three install notes: install the package **bare** "
            "(`flash-linear-attention`, not its `[cuda]`/`[rocm]` extras) — "
            "since v0.5 the backend extras pin their own torch and would "
            "downgrade your environment; keep the version within `0.5.x` "
            "(`>=0.5.2,<0.6`) — `chunk_kda` semantics are not guaranteed "
            "stable across minor versions; and on **Windows** the `fla` extra "
            "declares `triton-windows` behind a `platform_system == 'Windows'` "
            "marker, so the Triton kernels work there without a separate "
            "Triton install.",
        ),
        (
            "docs/faq/models_troubleshooting.md",
            "With the kernels installed the same checkpoints separate cleanly "
            "(p(CRE) 0.77 in-DHS vs 0.22 non-DHS on the probe set). Install "
            "the package **bare** — the `[cuda]`/`[rocm]` extras it shipped "
            "since v0.5 are legacy names no longer on the current install "
            "surface (they pin their own torch and would downgrade your "
            "environment) — and keep the version within `0.5.x` "
            "(`chunk_kda` semantics are not stable across minor versions). "
            "For current GPU installation lanes see the "
            "[GPU Support Matrix](../getting_started/gpu_support.md).",
        ),
    })

    SCAN_ROOTS = ("pyproject.toml", ".github", "docs", "example", "README.md", "scripts")
    SKIP_DIR_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints", ".git", "logs", "outputs"})

    def _scan_repo(self, pattern: str) -> set[tuple[str, str]]:
        """Yield (relative path, line) pairs whose line matches the pattern."""
        import re

        regex = re.compile(pattern)
        hits: set[tuple[str, str]] = set()
        for root_name in self.SCAN_ROOTS:
            root = REPO_ROOT / root_name
            if root.is_file():
                candidates = [root]
            else:
                candidates = [
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and not any(part in self.SKIP_DIR_NAMES for part in path.parts)
                ]
            for path in candidates:
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue  # binary / unreadable artifacts are out of scope
                for line in text.splitlines():
                    if regex.search(line):
                        # as_posix: the sanctioned frozensets store forward-slash
                        # relative paths; str(Path) on Windows yields backslashes,
                        # which unsanctioned every subdirectory row on the Windows
                        # leg (first exposed 2026-10-15, run 38676354432).
                        hits.add((path.relative_to(REPO_ROOT).as_posix(), line))
        return hits

    def test_base_bracket_residue_equals_sanctioned_list(self) -> None:
        """No unsanctioned .[base...] / dnallm[base...] install form survives."""
        measured = self._scan_repo(self.BASE_BRACKET_PATTERN)
        sanctioned = self.SANCTIONED_BASE_BRACKET_RESIDUE
        extra = measured - sanctioned
        missing = sanctioned - measured
        assert not extra, (
            f"unsanctioned base bracket-install residue (18-01 gate, PKG-01): "
            f"{sorted(extra)} -- every install command must use .[full]; only "
            f"the post-removal honesty-matrix row may carry the old spelling"
        )
        assert not missing, (
            f"sanctioned base deprecation-note residue missing (18-01 gate, "
            f"deletion creep): {sorted(missing)}"
        )

    def test_cuda121_rocm_residue_equals_sanctioned_list(self) -> None:
        """cuda121/torch-rocm/bracket-rocm survive only as fla-upstream vocabulary."""
        measured = self._scan_repo(self.CUDA121_ROCM_PATTERN)
        sanctioned = self.SANCTIONED_CUDA121_ROCM_RESIDUE
        extra = measured - sanctioned
        missing = sanctioned - measured
        assert not extra, (
            f"unsanctioned cuda121/rocm residue (18-01 gate): {sorted(extra)} "
            f"-- the retired extras/index/matrix/doc rows must stay deleted"
        )
        assert not missing, (
            f"sanctioned fla-upstream cuda121/rocm residue missing (18-01 "
            f"gate, deletion creep -- Pitfall 4 protects these): "
            f"{sorted(missing)}"
        )


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
