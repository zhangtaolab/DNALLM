"""Loader unit/behavior suite for the runtime model registry (REG-01, Phase 22-01).

Fast-leg, no network: these tests exercise ``dnallm.models.registry`` — the
sole runtime parser of the packaged ``model_info.yaml`` — end to end. Task 1
covers the single cached parse path (typed frozen entries, exact lookups,
loud validation, ``reload()`` semantics). Later sections extend this file with
the query surface, inheritance resolution, and the REG-03 read-only guards.
"""

import io

import pytest
from pydantic import ValidationError

from dnallm.models import registry

# Pinned end-to-end fixture: the first finetuned entry in the packaged file.
# Every field asserted here was measured against the real registry YAML at
# planning time (quick 261020-kwt data layer + 22-01 plan verification).
PINNED_FINETUNED_NAME = "Plant DNABERT BPE promoter"
PINNED_FINETUNED_REPO = "zhangtaolab/plant-dnabert-BPE-promoter"
PINNED_FINETUNED_BASE = "zhangtaolab/plant-dnabert-BPE"
PINNED_PRETRAINED_NAME = "Plant DNABERT"

# Minimal well-formed sections for _build_registry failure-path tests.
VALID_PRETRAINED = [
    {"name": "Base", "model": "org/base", "task": {"describe": "d", "task_type": "mask"}},
]
VALID_FINETUNED = [
    {
        "name": "Ft",
        "model": "org/ft",
        "base_model": "org/base",
        "task": {"describe": "d", "task_type": "binary"},
    },
]


class _BrokenYamlResource:
    """Minimal importlib.resources fake whose text is invalid YAML."""

    def joinpath(self, name):
        return self

    def open(self, mode="r", encoding=None):
        return io.StringIO("pretrained: [unclosed")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end typed lookup
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_by_entry_name_returns_typed_entry() -> None:
    """resolve() by entry name returns the fully typed frozen entry (REG-01)."""
    entry = registry.resolve(PINNED_FINETUNED_NAME)
    assert entry.name == PINNED_FINETUNED_NAME
    assert entry.model == PINNED_FINETUNED_REPO
    assert entry.kind == "finetuned"
    assert entry.base_model == PINNED_FINETUNED_BASE
    assert entry.task.describe
    assert entry.task.task_type == "binary"
    assert entry.task.num_labels == 2
    assert entry.task.label_names == ["Not promoter", "Core promoter"]
    assert entry.task.threshold == 0.5


def test_resolve_by_repo_id_returns_same_entry() -> None:
    """resolve() by repo id returns the same cached entry object as by name."""
    by_name = registry.resolve(PINNED_FINETUNED_NAME)
    by_repo = registry.resolve(PINNED_FINETUNED_REPO)
    assert by_repo is by_name


def test_resolve_unknown_raises_matchable_not_found_value_error() -> None:
    """An unknown lookup raises the matchable not-found ValueError."""
    with pytest.raises(ValueError, match=r"not found in registry"):
        registry.resolve("no-such-model")


def test_resolve_empty_string_raises_not_found() -> None:
    """Empty-string input raises the same not-found ValueError, never a KeyError."""
    with pytest.raises(ValueError, match=r"not found in registry"):
        registry.resolve("")


def test_lookup_is_exact_case_sensitive() -> None:
    """Matching is exact str equality: no casefolding or trimming (probe: encoding)."""
    with pytest.raises(ValueError, match=r"not found in registry"):
        registry.resolve("plant dnabert bpe promoter")
    with pytest.raises(ValueError, match=r"not found in registry"):
        registry.resolve("Zhangtaolab/plant-dnabert-BPE-promoter")
    assert registry.resolve(PINNED_FINETUNED_NAME).model == PINNED_FINETUNED_REPO


def test_try_resolve_returns_none_on_miss() -> None:
    """try_resolve() is the fall-through seam: None on miss, entry on hit."""
    assert registry.try_resolve("no-such-model") is None
    assert registry.try_resolve("") is None
    hit = registry.try_resolve(PINNED_FINETUNED_NAME)
    assert hit is not None
    assert hit.model == PINNED_FINETUNED_REPO


def test_frozen_entry_assignment_rejected() -> None:
    """RegistryEntry is frozen: attribute assignment raises ValidationError."""
    entry = registry.resolve(PINNED_FINETUNED_NAME)
    with pytest.raises(ValidationError):
        entry.name = "mutated"
    with pytest.raises(ValidationError):
        entry.task.threshold = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Single cached parse / reload
# ─────────────────────────────────────────────────────────────────────────────


def test_single_cached_parse_returns_identical_objects() -> None:
    """The registry parses once per process: repeated access returns identical objects."""
    assert registry._load_registry() is registry._load_registry()
    first = registry.resolve(PINNED_PRETRAINED_NAME)
    second = registry.resolve(PINNED_PRETRAINED_NAME)
    assert first is second


def test_reload_forces_reparse() -> None:
    """reload() clears the cache: the next access re-parses into fresh objects."""
    before = registry.resolve(PINNED_FINETUNED_NAME)
    registry.reload()
    after = registry.resolve(PINNED_FINETUNED_NAME)
    assert after == before
    assert after is not before
    # The re-established cache is again single-parse.
    assert registry.resolve(PINNED_FINETUNED_NAME) is after


# ─────────────────────────────────────────────────────────────────────────────
# Loud validation on malformed registry data (D-01b)
# ─────────────────────────────────────────────────────────────────────────────


def test_malformed_entry_fails_loudly_naming_offender() -> None:
    """A schema-violating entry raises ValueError naming the offending repo id."""
    bad = {
        "pretrained": [
            {
                "name": "Bad",
                "model": "org/bad-model",
                "task": {"describe": "d", "task_type": "mask"},
                "max_length": 0,
            }
        ],
        "finetuned": VALID_FINETUNED,
    }
    with pytest.raises(ValueError, match=r"org/bad-model"):
        registry._build_registry(bad)


def test_missing_sections_rejected() -> None:
    """A missing, empty, or non-list section raises ValueError naming the section."""
    with pytest.raises(ValueError, match=r"'pretrained'"):
        registry._build_registry({})
    with pytest.raises(ValueError, match=r"'pretrained'"):
        registry._build_registry({"pretrained": [], "finetuned": VALID_FINETUNED})
    with pytest.raises(ValueError, match=r"'pretrained'"):
        registry._build_registry({"pretrained": "nope", "finetuned": VALID_FINETUNED})
    with pytest.raises(ValueError, match=r"'finetuned'"):
        registry._build_registry({"pretrained": VALID_PRETRAINED, "finetuned": []})


def test_duplicate_entry_names_rejected() -> None:
    """Duplicate entry names across both sections raise ValueError naming the name."""
    clash = [{**VALID_FINETUNED[0], "name": "Base"}]
    with pytest.raises(ValueError, match=r"Base"):
        registry._build_registry({"pretrained": VALID_PRETRAINED, "finetuned": clash})


def test_duplicate_repo_ids_rejected() -> None:
    """Duplicate repo ids across both sections raise ValueError naming the repo."""
    clash = [{**VALID_FINETUNED[0], "model": "org/base", "name": "Other"}]
    with pytest.raises(ValueError, match=r"org/base"):
        registry._build_registry({"pretrained": VALID_PRETRAINED, "finetuned": clash})


def test_non_dict_entry_rejected() -> None:
    """A non-mapping entry raises a descriptive ValueError."""
    with pytest.raises(ValueError, match=r"must be a mapping"):
        registry._build_registry({"pretrained": ["nope"], "finetuned": VALID_FINETUNED})


def test_non_dict_registry_rejected() -> None:
    """A non-mapping top-level registry document raises a descriptive ValueError."""
    with pytest.raises(ValueError, match=r"mapping"):
        registry._build_registry(["nope"])


def test_missing_resource_raises_value_error() -> None:
    """A missing packaged resource surfaces as ValueError, not FileNotFoundError."""

    def _raise_module_not_found(*args, **kwargs):
        raise ModuleNotFoundError("dnallm.models")

    registry.reload()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(registry.resources, "files", _raise_module_not_found)
    try:
        with pytest.raises(ValueError, match=r"Failed to load the packaged model registry"):
            registry.resolve(PINNED_FINETUNED_NAME)
    finally:
        monkeypatch.undo()


def test_yaml_parse_error_raises_value_error() -> None:
    """Unparseable registry YAML surfaces as ValueError, not yaml.YAMLError."""
    registry.reload()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(registry.resources, "files", lambda *a, **k: _BrokenYamlResource())
    try:
        with pytest.raises(ValueError, match=r"Failed to load the packaged model registry"):
            registry.resolve(PINNED_PRETRAINED_NAME)
    finally:
        monkeypatch.undo()
