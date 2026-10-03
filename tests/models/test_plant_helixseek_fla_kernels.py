"""Kernel-dependency contract for PlantHelixSeek remote-code models.

PlantHelixSeek-CRE/-Anno import ``fla.ops.kda.chunk.chunk_kda`` from
flash-linear-attention for their HelixSeekDelta (KDA) layers. Without the package
the remote code silently falls back to a pure-PyTorch path that is not KDA math
and produces positionally-uninformative outputs (proven 2026-10-03: probe-set
p(CRE) 0.0073 in-DHS vs 0.0087 non-DHS on the fallback; 0.7673 vs 0.2230 with
fla 0.5.2). These tests keep the dependency declared, wired into ``all``, and
the import path stable across fla upgrades.
"""

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib is stdlib-only from 3.11; pyproject-declaration tests skip on 3.10
    tomllib = None

_META_EXTRA_RE = re.compile(r"^dnallm\[([^\]]*)\]$")


def _meta_extra_names(spec: str) -> list[str]:
    """Parse a self-referential ``dnallm[...]`` extra spec into its member names.

    Members are comma-split and whitespace-stripped; empty members are dropped.
    A spec that is not the ``dnallm[...]`` shape (e.g. a plain dependency
    constraint or a bare extra name) returns an empty list.
    """
    match = _META_EXTRA_RE.match(spec.strip())
    if match is None:
        return []
    return [member.strip() for member in match.group(1).split(",") if member.strip()]


def _load_pyproject() -> dict:
    """Load the repo pyproject.toml as a dict."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


class TestMetaExtraParser:
    """The bracket-spec parser must match members exactly, never by substring."""

    def test_parses_members_with_whitespace_tolerance(self):
        assert _meta_extra_names("dnallm[base,dev,test,notebook,docs,ui,mcp,fla]") == [
            "base",
            "dev",
            "test",
            "notebook",
            "docs",
            "ui",
            "mcp",
            "fla",
        ]
        # Whitespace inside the brackets is cosmetic, not semantic
        assert "fla" in _meta_extra_names("dnallm[base, fla ]")

    def test_substring_colliding_names_are_not_members(self):
        # IN-05 regression: a future "flash-attn"-style spec must not satisfy
        # a fla-membership check the way it satisfied the old substring match
        assert "fla" not in _meta_extra_names("dnallm[base,flash-attn]")
        assert "fla" not in _meta_extra_names("dnallm[base,fla-core,mamba-fla]")

    def test_non_meta_specs_return_empty_list(self):
        assert _meta_extra_names("fla") == []
        assert _meta_extra_names("torch>=2.4.0,<2.12") == []


class TestFlaExtraDeclared:
    """The fla kernel extra must stay declared and reachable from ``all``."""

    @pytest.mark.skipif(
        tomllib is None,
        reason="environment-unavailable: tomllib requires Python >= 3.11 "
        "(pyproject declaration tests need it)",
    )
    def test_fla_extra_declared_with_bounded_range(self):
        extras = _load_pyproject()["project"]["optional-dependencies"]
        assert "fla" in extras, "the fla kernel extra must stay declared in pyproject.toml"
        assert any(
            "flash-linear-attention>=0.5.2" in spec and "<0.6" in spec for spec in extras["fla"]
        ), "fla spec must carry the 0.5.x floor and minor bound (chunk_kda stability)"

    @pytest.mark.skipif(
        tomllib is None,
        reason="environment-unavailable: tomllib requires Python >= 3.11 "
        "(pyproject declaration tests need it)",
    )
    def test_fla_reachable_from_all(self):
        extras = _load_pyproject()["project"]["optional-dependencies"]
        assert any("fla" in _meta_extra_names(spec) for spec in extras["all"]), (
            "all must include the fla extra as an exact bracket member"
        )


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
