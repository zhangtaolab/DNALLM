"""Tests for the environment fact collector (dnallm.utils.env_info).

The collector backs the wizard's first panel (D-02/D-03), so these tests pin
the contract the panel relies on: the exact row set with plain-language
values, and "degrade, never raise" for a broken environment -- a package whose
metadata is missing, a machine with no GPU, a torch that is not installed, and
a cache folder that cannot be read.

The torch-free fast path is asserted in a subprocess on purpose:
``tests/conftest.py`` imports torch at session start, so an in-process
``sys.modules`` check could never prove the module is import-light.
"""

import importlib.metadata as metadata
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from dataclasses import dataclass
from pathlib import Path

from dnallm.utils import env_info

GIB = 1024**3


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh isolated interpreter (no cwd, no PYTHONPATH)."""
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv only
        [sys.executable, "-I", "-c", code], capture_output=True, text=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fakes for the torch seam (no real hardware is touched)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeCuda:
    """A stand-in for ``torch.cuda`` with configurable answers."""

    def __init__(
        self,
        *,
        available: bool = True,
        name: str = "NVIDIA GeForce RTX 4090",
        total_gb: float = 24,
        free_gb: float = 21,
    ) -> None:
        self._available = available
        self._name = name
        self._total = int(total_gb * GIB)
        self._free = int(free_gb * GIB)

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int = 0) -> str:
        return self._name

    def mem_get_info(self, index: int = 0) -> tuple[int, int]:
        return (self._free, self._total)


@dataclass
class _FakeTorch:
    """A stand-in for the torch module holding a :class:`_FakeCuda`."""

    cuda: object


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


# ─────────────────────────────────────────────────────────────────────────────
# Device and VRAM rows
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectDevice:
    """The GPU branches, driven through the replaceable torch seam."""

    def test_cuda_device_reports_name_and_memory(self, monkeypatch) -> None:
        fake = _FakeTorch(_FakeCuda(available=True, total_gb=24, free_gb=21))
        monkeypatch.setattr(env_info, "_torch", lambda: fake)

        values = {row.key: row.value for row in env_info.collect_device()}

        assert values["torch_cuda"].startswith("available")
        assert values["torch_device"] == "NVIDIA GeForce RTX 4090"
        assert values["torch_vram"] == "24 GB total, 21 GB free"

    def test_cpu_only_reports_a_single_row_and_no_vram(self, monkeypatch) -> None:
        monkeypatch.setattr(env_info, "_torch", lambda: _FakeTorch(_FakeCuda(available=False)))

        rows = env_info.collect_device()

        assert len(rows) == 1
        assert rows[0].value == "CPU only — no GPU detected"

    def test_torch_absent_reports_plain_language_row(self, monkeypatch) -> None:
        def _missing():
            raise ImportError("No module named 'torch'")

        monkeypatch.setattr(env_info, "_torch", _missing)

        rows = env_info.collect_device()

        assert len(rows) == 1
        assert "PyTorch is not installed" in rows[0].value

    def test_unanswerable_cuda_query_degrades(self, monkeypatch) -> None:
        class _ExplodingCuda:
            def is_available(self) -> bool:
                raise RuntimeError("driver query failed")

        monkeypatch.setattr(env_info, "_torch", lambda: _FakeTorch(_ExplodingCuda()))

        rows = env_info.collect_device()

        assert len(rows) == 1
        assert rows[0].value.startswith("unknown")

    def test_unreadable_driver_degrades_every_gpu_row(self, monkeypatch) -> None:
        class _BrokenDriver(_FakeCuda):
            def get_device_name(self, index: int = 0) -> str:
                raise RuntimeError("no driver")

            def mem_get_info(self, index: int = 0) -> tuple[int, int]:
                raise RuntimeError("no driver")

        monkeypatch.setattr(env_info, "_torch", lambda: _FakeTorch(_BrokenDriver(available=True)))

        values = {row.key: row.value for row in env_info.collect_device()}

        assert values["torch_cuda"].startswith("available")
        assert values["torch_device"].startswith("unknown")
        assert values["torch_vram"].startswith("unknown")


# ─────────────────────────────────────────────────────────────────────────────
# The composed report
# ─────────────────────────────────────────────────────────────────────────────


class TestCollect:
    """``collect()`` composes versions, then device, then disk -- always."""

    def test_row_order_is_versions_device_disk(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(env_info, "_torch", lambda: _FakeTorch(_FakeCuda(available=False)))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms_cache"))

        report = env_info.collect()

        keys = [row.key for row in report.rows]
        assert keys[:5] == [
            "dnallm_version",
            "python_version",
            "transformers_version",
            "numpy_version",
            "datasets_version",
        ]
        assert keys[5] == "torch_cuda"
        assert keys[6:] == [
            f"disk:{tmp_path / 'hub_cache'}",
            f"disk:{tmp_path / 'ms_cache'}",
        ]

    def test_every_row_is_renderable(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(env_info, "_torch", lambda: _FakeTorch(_FakeCuda(available=True)))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub_cache"))
        monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "ms_cache"))

        report = env_info.collect()

        assert report.rows
        for row in report.rows:
            assert row.key.strip()
            assert row.label.strip()
            assert row.value.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Import purity of the module itself, and the torch entry point
# ─────────────────────────────────────────────────────────────────────────────


class TestImportPurity:
    """``import dnallm.utils.env_info`` must not load torch (D-02)."""

    def test_importing_the_module_leaves_torch_absent(self) -> None:
        code = (
            "import sys;"
            "import dnallm.utils.env_info;"
            "print('TORCH=' + ('present' if 'torch' in sys.modules else 'absent'))"
        )

        result = _run_isolated(code)

        assert result.returncode == 0, result.stderr
        assert "TORCH=absent" in result.stdout

    def test_collect_device_is_the_only_torch_entry_point(self) -> None:
        code = (
            "import sys;"
            "from dnallm.utils.env_info import collect_device;"
            "before = 'present' if 'torch' in sys.modules else 'absent';"
            "collect_device();"
            "after = 'present' if 'torch' in sys.modules else 'absent';"
            "print('BEFORE=' + before);"
            "print('AFTER=' + after)"
        )

        result = _run_isolated(code)

        assert result.returncode == 0, result.stderr
        assert "BEFORE=absent" in result.stdout
        assert "AFTER=present" in result.stdout
