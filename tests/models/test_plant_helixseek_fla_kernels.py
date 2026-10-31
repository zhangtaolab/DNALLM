"""Kernel-dependency contract for PlantHelixSeek remote-code models.

PlantHelixSeek-CRE/-Anno import ``fla.ops.kda.chunk.chunk_kda`` from
flash-linear-attention for their HelixSeekDelta (KDA) layers. Without the package
the remote code silently falls back to a pure-PyTorch path that is not KDA math
and produces positionally-uninformative outputs (proven 2026-10-03: probe-set
p(CRE) 0.0073 in-DHS vs 0.0087 non-DHS on the fallback; 0.7673 vs 0.2230 with
fla 0.5.2). These tests keep the dependency in the base install (D-09, 28-01:
the former ``fla`` extra is retired so PlantHelixSeek works out-of-the-box),
the retired extra absent from the extras table and ``all``, and the import
path stable across fla upgrades.
"""

import re
import tomllib
from pathlib import Path

import pytest

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
        assert _meta_extra_names("dnallm[full,dev,test,notebook,docs,ui,mcp,fla]") == [
            "full",
            "dev",
            "test",
            "notebook",
            "docs",
            "ui",
            "mcp",
            "fla",
        ]
        # Whitespace inside the brackets is cosmetic, not semantic
        assert "fla" in _meta_extra_names("dnallm[full, fla ]")

    def test_substring_colliding_names_are_not_members(self):
        # IN-05 regression: a future "flash-attn"-style spec must not satisfy
        # a fla-membership check the way it satisfied the old substring match
        assert "fla" not in _meta_extra_names("dnallm[full,flash-attn]")
        assert "fla" not in _meta_extra_names("dnallm[full,fla-core,mamba-fla]")

    def test_non_meta_specs_return_empty_list(self):
        assert _meta_extra_names("fla") == []
        assert _meta_extra_names("torch>=2.4.0,<2.12") == []


class TestFlaBaseDependency:
    """fla must ride the base install (D-09, 28-01): no extra may stand
    between a default ``pip install dnallm`` and the PlantHelixSeek KDA
    kernels. The retired ``fla`` extra must stay deleted — ``dnallm[fla]``
    syntax is gone by design, and ``all`` must not carry the member."""

    def test_fla_declared_as_base_dependency(self):
        deps = _load_pyproject()["project"]["dependencies"]
        assert "flash-linear-attention>=0.5.2,<0.6" in deps, (
            "flash-linear-attention>=0.5.2,<0.6 must be a base dependency "
            "(D-09: PlantHelixSeek KDA kernels ship with every default install)"
        )

    def test_retired_fla_extra_absent(self):
        extras = _load_pyproject()["project"]["optional-dependencies"]
        assert "fla" not in extras, (
            "the retired fla extra must stay deleted (D-09): fla rides the "
            "base install; dnallm[fla] syntax disappeared with the promotion"
        )

    def test_all_meta_extra_has_no_fla_member(self):
        extras = _load_pyproject()["project"]["optional-dependencies"]
        for spec in extras.get("all", []):
            assert "fla" not in _meta_extra_names(spec), (
                f"all must not carry the retired fla member (D-09/D-13): {spec!r}"
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
