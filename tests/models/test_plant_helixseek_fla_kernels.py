"""Kernel-dependency contract for PlantHelixSeek remote-code models.

PlantHelixSeek-CRE/-Anno import ``fla.ops.kda.chunk.chunk_kda`` from
flash-linear-attention for their HelixSeekDelta (KDA) layers. Without the package
the remote code silently falls back to a pure-PyTorch path that is not KDA math
and produces positionally-uninformative outputs (proven 2026-10-03: probe-set
p(CRE) 0.0073 in-DHS vs 0.0087 non-DHS on the fallback; 0.7673 vs 0.2230 with
fla 0.5.2). These tests keep the dependency declared, wired into ``all``, and
the import path stable across fla upgrades.
"""

import tomllib
from pathlib import Path

import pytest


def _load_pyproject() -> dict:
    """Load the repo pyproject.toml as a dict."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


class TestFlaExtraDeclared:
    """The fla kernel extra must stay declared and reachable from ``all``."""

    def test_fla_extra_declared_with_bounded_range(self):
        extras = _load_pyproject()["project"]["optional-dependencies"]
        assert "fla" in extras, "the fla kernel extra must stay declared in pyproject.toml"
        assert any(
            "flash-linear-attention>=0.5.2" in spec and "<0.6" in spec for spec in extras["fla"]
        ), "fla spec must carry the 0.5.x floor and minor bound (chunk_kda stability)"

    def test_fla_reachable_from_all(self):
        extras = _load_pyproject()["project"]["optional-dependencies"]
        assert any("fla" in spec for spec in extras["all"]), "all must include the fla extra"


class TestChunkKdaImportPath:
    """When fla is installed, the exact import the remote code needs must resolve."""

    def test_chunk_kda_importable_when_fla_installed(self):
        pytest.importorskip(
            "fla",
            reason="environment-unavailable: flash-linear-attention not installed "
            "(PlantHelixSeek KDA kernels absent — models silently degrade, see docs/faq)",
        )
        from fla.ops.kda.chunk import chunk_kda

        assert callable(chunk_kda), "fla.ops.kda.chunk.chunk_kda must stay callable (0.5.x API)"
