"""Tests for the environment fact collector (dnallm.utils.env_info).

The collector backs the wizard's first panel (D-02/D-03), so these tests pin
the contract the panel relies on: the exact row set with plain-language
values, and "degrade, never raise" for a broken environment -- a package whose
metadata is missing, and a cache folder that cannot be read.

The torch-free fast path is asserted in a subprocess on purpose:
``tests/conftest.py`` imports torch at session start, so an in-process
``sys.modules`` check could never prove the module is import-light.
"""

import importlib.metadata as metadata
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

from dnallm.utils import env_info


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh isolated interpreter (no cwd, no PYTHONPATH)."""
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv only
        [sys.executable, "-I", "-c", code], capture_output=True, text=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Import discipline (the D-02 invariant, checked at the source level)
# ─────────────────────────────────────────────────────────────────────────────


class TestImportDiscipline:
    """Module-level imports must stay stdlib plus the relative logger."""

    def test_module_level_imports_are_stdlib_only(self) -> None:
        """No torch, textual, or platformdirs at module scope (D-02)."""
        import ast

        tree = ast.parse(Path(env_info.__file__).read_text(encoding="utf-8"))
        stdlib = {"dataclasses", "importlib", "os", "pathlib", "platform", "shutil"}
        offenders: set[str] = set()
        for node in tree.body:  # module scope only; function bodies may import torch
            if isinstance(node, ast.Import):
                offenders.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    if node.module != "logger":
                        offenders.add(f".{node.module}")
                elif node.module:
                    offenders.add(node.module.split(".")[0])

        assert offenders <= stdlib, (
            f"dnallm/utils/env_info.py imports beyond the stdlib at module scope: "
            f"{sorted(offenders - stdlib)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Version rows
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectVersions:
    """The instant, library-free half of the report."""

    def test_dnallm_and_python_rows_match_the_runtime(self) -> None:
        """The two rows a user reads first carry the real runtime values."""
        rows = {row.key: row for row in env_info.collect_versions()}

        assert rows["dnallm_version"].value == metadata.version("dnallm")
        assert rows["python_version"].value == platform.python_version()

    def test_row_set_matches_the_locked_enumeration(self) -> None:
        """Exactly the D-03 version rows, in a stable order."""
        keys = [row.key for row in env_info.collect_versions()]

        assert keys == [
            "dnallm_version",
            "python_version",
            "transformers_version",
            "numpy_version",
            "datasets_version",
        ]

    def test_missing_metadata_degrades_to_a_plain_language_row(self, monkeypatch) -> None:
        """An absent distribution yields an unknown row instead of raising."""

        def _missing(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "version", _missing)

        rows = env_info.collect_versions()

        assert len(rows) == 5
        assert rows[0].value.startswith("unknown")
        assert "could not be read" in rows[0].value
        for row in rows:
            assert row.key
            assert row.label
            assert row.value


# ─────────────────────────────────────────────────────────────────────────────
# Cache-root resolution (stdlib only, huggingface_hub precedence)
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheRoots:
    """The free-disk rows must point at the folders the downloads actually use."""

    def test_hf_hub_cache_wins_over_hf_home(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))

        roots = env_info._cache_roots()

        assert roots[0] == tmp_path / "hub_cache"

    def test_hf_home_supplies_the_hub_subfolder(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))

        roots = env_info._cache_roots()

        assert roots[0] == tmp_path / "hf_home" / "hub"

    def test_modelscope_cache_env_is_honored(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms_cache"))

        roots = env_info._cache_roots()

        assert roots[1] == tmp_path / "ms_cache"

    def test_defaults_when_nothing_is_set(self, monkeypatch) -> None:
        for variable in ("HF_HUB_CACHE", "HF_HOME", "MODELSCOPE_CACHE"):
            monkeypatch.delenv(variable, raising=False)

        roots = env_info._cache_roots()

        assert roots == [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "modelscope",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Disk rows
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectDisk:
    """One plain-language free-space row per cache root."""

    def test_one_row_per_resolved_root(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "hub_cache").mkdir()
        (tmp_path / "ms_cache").mkdir()
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms_cache"))

        rows = env_info.collect_disk()

        assert len(rows) == 2
        assert all("GB free" in row.value for row in rows)
        assert all("Free disk space for model downloads" in row.label for row in rows)

    def test_unreadable_root_yields_an_unknown_row(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms_cache"))

        def _unreadable(path: str):
            raise OSError("Input/output error")

        monkeypatch.setattr(env_info.shutil, "disk_usage", _unreadable)

        rows = env_info.collect_disk()

        assert len(rows) == 2
        assert all(row.value == "unknown — this folder could not be checked" for row in rows)


# ─────────────────────────────────────────────────────────────────────────────
# The torch-free fast path (subprocess: conftest imports torch in-process)
# ─────────────────────────────────────────────────────────────────────────────


class TestTorchFreeFastPath:
    """Version and disk rows must be collectable before torch is loaded."""

    def test_versions_and_disk_do_not_import_torch(self) -> None:
        code = (
            "import sys;"
            "from dnallm.utils.env_info import collect_disk, collect_versions;"
            "collect_versions();"
            "collect_disk();"
            "print('TORCH=' + ('present' if 'torch' in sys.modules else 'absent'))"
        )

        result = _run_isolated(code)

        assert result.returncode == 0, result.stderr
        # The package logger also writes to stdout; match the marker line.
        assert "TORCH=absent" in result.stdout
