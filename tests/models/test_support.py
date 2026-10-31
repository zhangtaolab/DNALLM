"""Offline tests for the shared plain-language support vocabulary (D-16, 27.2-04).

One registry-derived word per entry: the docs generator, the TUI listing,
and the MCP listing must show the same support vocabulary, with finetuned
entries inheriting their base model's status and exact external-package
overrides for the four repos whose need a status word cannot express.
Branch-complete: every status, every override, actual and synthetic base
inheritance, missing bases, cycles, unknown values, and the four verbatim
generated-page strings.

Every ``support`` attribute is accessed inside a test body, never at module
level: the RED run of this file collects cleanly against the docstring-only
stub and each target test fails on the missing symbol it asserts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnallm.models import registry, support
from dnallm.models.registry import RegistryEntry, TaskInfo

# Expected vocabulary, pinned as literals in this file (checker independence:
# the gate never imports the writer's mapping to verify it).
EXPECTED_SUPPORT_WORDS = {
    "verified-tf5": "works",
    "import-verified-tf5": "family-tested",
    "shim-required-tf5": "works with built-in fixes",
    # 28-02 D-01/D-14: a model measured green only after installing a curated
    # optional extra (owner ruling A: CrossDNA is the one committed row).
    "extra-required-tf5": "works with an optional extra",
    "remote-code-broken": "not working with transformers 5",
    "census-blocked": "untested",
}

# 28-02 ruling A note: BigBird KEEPS "needs deepspeed" (its forward is
# measured structurally dead even with the extra - a "works with the
# deepspeed extra" word would overclaim); CrossDNA gains its exact word.
EXPECTED_OVERRIDES = {
    "songlab/gpn-brassicales": "needs pip install gpn",
    "zehui127/Omni-DNA-116M": "needs pip install ai2-olmo",
    "AIRI-Institute/gena-lm-bigbird-base-sparse": "needs deepspeed",
    "chengCCC/CrossDNA_pretrain": "works with the flash-attn extra",
}

# The four generated-page strings the census generator emits; the shared
# module must preserve them verbatim (byte-identical regeneration).
EXPECTED_DOCS_CELLS = {
    "verified-tf5": "Tested working",
    "import-verified-tf5": "Tested loading its code; full run inferred from the family model",
    "shim-required-tf5": "Works with DNALLM's built-in compatibility fixes (tested)",
    "extra-required-tf5": "Works after installing an optional extra (tested)",
}

EXPECTED_UNTESTED = "untested"

# Actual finetuned entries over the committed registry: their bases carry the
# statuses asserted below (see docs/user_guide/model_compatibility.md).
ACTUAL_INHERITING = {
    "zhangtaolab/plant-dnabert-BPE-promoter": "works",  # base verified-tf5
    "zhangtaolab/dnabert2-promoter": "works with built-in fixes",  # base shim-required-tf5
}


def _entry(**overrides) -> RegistryEntry:
    """Return a minimal validated RegistryEntry with overrides applied."""
    payload: dict = {
        "name": "Synth Support",
        "model": "synth/support",
        "task": {"describe": "d", "task_type": "mask"},
        "kind": "pretrained",
    }
    payload.update(overrides)
    return RegistryEntry.model_validate(payload)


class TestSupportWords:
    """Every census status maps to its exact plain-language word."""

    @pytest.mark.parametrize(
        ("status", "word"),
        [pytest.param(status, word, id=status) for status, word in EXPECTED_SUPPORT_WORDS.items()],
    )
    def test_status_maps_to_exact_word(self, status: str, word: str) -> None:
        assert (
            support.support_word(_entry(model=f"synth/{status}", tf_compat_status=status)) == word
        )

    def test_missing_status_maps_to_untested(self) -> None:
        assert support.support_word(_entry()) == EXPECTED_UNTESTED

    def test_unknown_status_maps_to_untested_never_raises(self) -> None:
        """A future/unvalidated status value degrades to untested, not an error."""
        entry = RegistryEntry.model_construct(
            name="Synth Future",
            model="synth/future",
            task=TaskInfo.model_validate({"describe": "d", "task_type": "mask"}),
            kind="pretrained",
            tf_compat_status="some-future-status",
        )
        assert support.support_word(entry) == EXPECTED_UNTESTED

    def test_vocabulary_covers_exactly_the_six_census_statuses(self) -> None:
        assert set(support.SUPPORT_WORDS) == set(EXPECTED_SUPPORT_WORDS)


class TestRepoOverrides:
    """The four external-package repos show their exact need, not a status word."""

    @pytest.mark.parametrize(
        ("repo", "word"),
        [pytest.param(repo, word, id=repo) for repo, word in EXPECTED_OVERRIDES.items()],
    )
    def test_override_repo_shows_exact_package_need(self, repo: str, word: str) -> None:
        entry = _entry(model=repo, tf_compat_status="remote-code-broken")
        assert support.support_word(entry) == word

    def test_override_wins_over_general_status_mapping(self) -> None:
        """Even a green status on an override repo still shows the package need."""
        entry = _entry(model="songlab/gpn-brassicales", tf_compat_status="verified-tf5")
        assert support.support_word(entry) == "needs pip install gpn"

    def test_real_registry_override_rows_word_correctly(self) -> None:
        """Over the committed registry: all four override repos resolve by name."""
        for repo, word in EXPECTED_OVERRIDES.items():
            assert support.support_word(registry.resolve(repo)) == word, repo


class TestBaseInheritance:
    """A finetuned entry follows base_model to the pretrained base's verdict."""

    @pytest.mark.parametrize(
        ("repo", "word"),
        [pytest.param(repo, word, id=repo) for repo, word in ACTUAL_INHERITING.items()],
    )
    def test_actual_finetuned_entry_inherits_base_word(self, repo: str, word: str) -> None:
        assert support.support_word(registry.resolve(repo)) == word

    def test_synthetic_child_of_override_repo_inherits_the_override(self) -> None:
        child = _entry(
            kind="finetuned",
            model="synth/gpn-child",
            base_model="songlab/gpn-brassicales",
        )
        assert support.support_word(child) == "needs pip install gpn"

    def test_synthetic_chain_through_statusless_intermediate(self, monkeypatch) -> None:
        """child -> mid (no status) -> base (verified): the word comes from base."""
        base = _entry(model="synth/base", tf_compat_status="verified-tf5")
        mid = _entry(kind="finetuned", model="synth/mid", base_model="synth/base")
        child = _entry(kind="finetuned", model="synth/child", base_model="synth/mid")
        table = {"synth/mid": mid, "synth/base": base}
        monkeypatch.setattr(registry, "resolve", lambda repo_id: table[repo_id])
        assert support.support_word(child) == "works"

    def test_synthetic_chain_to_statusless_root_is_untested(self, monkeypatch) -> None:
        """An ancestor with no status and no base ends the chain as untested."""
        bare = _entry(model="synth/bare")
        mid = _entry(kind="finetuned", model="synth/mid2", base_model="synth/bare")
        child = _entry(kind="finetuned", model="synth/child2", base_model="synth/mid2")
        table = {"synth/mid2": mid, "synth/bare": bare}
        monkeypatch.setattr(registry, "resolve", lambda repo_id: table[repo_id])
        assert support.support_word(child) == EXPECTED_UNTESTED

    def test_missing_base_returns_untested_never_raises(self) -> None:
        child = _entry(kind="finetuned", model="synth/ghost-child", base_model="org/ghost-base")
        assert support.support_word(child) == EXPECTED_UNTESTED

    def test_two_node_cycle_returns_untested(self, monkeypatch) -> None:
        node_x = _entry(kind="finetuned", model="synth/x", base_model="synth/y")
        node_y = _entry(kind="finetuned", model="synth/y", base_model="synth/x")
        table = {"synth/y": node_y}
        monkeypatch.setattr(registry, "resolve", lambda repo_id: table[repo_id])
        assert support.support_word(node_x) == EXPECTED_UNTESTED

    def test_self_cycle_returns_untested(self) -> None:
        """A base_model pointing at the entry's own repo is caught by the seen-set."""
        self_ref = _entry(kind="finetuned", model="synth/self", base_model="synth/self")
        assert support.support_word(self_ref) == EXPECTED_UNTESTED

    def test_finetuned_without_base_or_status_is_untested(self) -> None:
        assert support.support_word(_entry(kind="finetuned")) == EXPECTED_UNTESTED


class TestDocsStatusCells:
    """The generated-page strings live here verbatim, shared with the generator."""

    def test_four_cells_verbatim(self) -> None:
        assert support.DOCS_STATUS_CELLS == EXPECTED_DOCS_CELLS

    def test_docs_cells_subset_of_status_vocabulary(self) -> None:
        assert set(support.DOCS_STATUS_CELLS) <= set(support.SUPPORT_WORDS)


class TestPublicSurface:
    """The module exports exactly the four shared-vocabulary names."""

    def test_all_exports_present_and_typed(self) -> None:
        assert set(support.__all__) == {
            "SUPPORT_WORDS",
            "SUPPORT_WORD_BY_REPO",
            "DOCS_STATUS_CELLS",
            "support_word",
        }
        assert isinstance(support.SUPPORT_WORDS, dict)
        assert isinstance(support.SUPPORT_WORD_BY_REPO, dict)
        assert isinstance(support.DOCS_STATUS_CELLS, dict)
        assert callable(support.support_word)

    def test_every_real_pretrained_entry_maps_to_a_known_word(self) -> None:
        """End-to-end over the committed registry: all 49 pretrained rows word cleanly."""
        known = set(EXPECTED_SUPPORT_WORDS.values()) | set(EXPECTED_OVERRIDES.values())
        pretrained = [e for e in registry.all_entries() if e.kind == "pretrained"]
        assert len(pretrained) == 49
        for entry in pretrained:
            assert support.support_word(entry) in known, entry.model


class TestModulePurity:
    """Source-level guards: no YAML parsing, no network, no registry-file mention."""

    def test_module_never_parses_yaml_or_names_the_registry_file(self) -> None:
        source = Path(support.__file__).read_text(encoding="utf-8")
        # REG-01 sole-mentioner: naming the packaged file here fails the
        # registry mentioner gate; parsing it here would be a second parser.
        assert "model_info.yaml" not in source
        assert "import yaml" not in source

    def test_module_imports_no_network_or_torch_clients(self) -> None:
        source = Path(support.__file__).read_text(encoding="utf-8")
        for token in ("requests", "urllib", "socket", "httpx", "http.client", "import torch"):
            assert token not in source, token
