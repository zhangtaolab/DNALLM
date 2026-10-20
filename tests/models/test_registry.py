"""Loader unit/behavior suite for the runtime model registry (REG-01, Phase 22-01).

Fast-leg, no network: these tests exercise ``dnallm.models.registry`` — the
sole runtime parser of the packaged ``model_info.yaml`` — end to end. Task 1
covers the single cached parse path (typed frozen entries, exact lookups,
loud validation, ``reload()`` semantics). Task 2 extends with the query
surface, inheritance resolution, and the facade. Task 3 enforces the three
REG-03 read-only guarantees permanently: the loader is the SOLE parser of
``model_info.yaml`` (mentioner allowlist), it exposes NO write/serialize
API of any kind (verb + emitter-token scans), and entries stay FROZEN.
"""

import ast
import io
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import dnallm
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


def test_regression_bare_string_label_names_wrapped_to_list() -> None:
    """Regression entries carry a bare-string label_names (target description).

    The packaged file stores free text there on the promoter-strength
    regression entries; the loader accepts it and wraps it into a
    one-element list so every consumer sees ``list[str] | None`` (matching
    the TaskConfig regression convention), without mutating the file.
    """
    entry = registry.resolve("zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf")
    assert entry.task.task_type == "regression"
    assert entry.task.label_names == ["promoter strength in tobacco leaves"]


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


def test_max_length_accepts_one_and_rejects_nonpositive() -> None:
    """max_length takes any positive int: 1 is valid; 0 and negatives are rejected (gt=0)."""
    minimal = {
        "pretrained": [
            {
                "name": "Boundary",
                "model": "org/boundary",
                "task": {"describe": "d", "task_type": "mask"},
                "max_length": 1,
            }
        ],
        "finetuned": VALID_FINETUNED,
    }
    built = registry._build_registry(minimal)
    assert built.by_repo["org/boundary"].max_length == 1

    for nonpositive in (0, -5):
        with pytest.raises(ValueError, match=r"org/boundary"):
            registry._build_registry({
                "pretrained": [
                    {
                        "name": "Boundary",
                        "model": "org/boundary",
                        "task": {"describe": "d", "task_type": "mask"},
                        "max_length": nonpositive,
                    }
                ],
                "finetuned": VALID_FINETUNED,
            })


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


# ─────────────────────────────────────────────────────────────────────────────
# Query surface: task_type / base_model / all entries / lab-first helpers
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TOTAL_ENTRIES = 236
EXPECTED_LAB_ENTRIES = 189
THIRD_PARTY_REPO = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"


def test_get_by_task_type_spans_both_sections() -> None:
    """task_type queries return matching entries from both packaged sections."""
    mask = registry.get_by_task_type("mask")
    assert "zhangtaolab/plant-dnabert-6mer" in [e.model for e in mask]
    assert all(e.task.task_type == "mask" for e in mask)
    assert any(e.kind == "pretrained" for e in mask)
    binary = registry.get_by_task_type("binary")
    assert PINNED_FINETUNED_REPO in [e.model for e in binary]
    assert all(e.kind == "finetuned" for e in binary)


def test_empty_or_unknown_list_queries_return_empty_lists() -> None:
    """Empty-string and unknown inputs to list queries yield [], never a KeyError."""
    assert registry.get_by_task_type("") == []
    assert registry.get_by_base_model("") == []
    assert registry.get_by_task_type("no-such-task-type") == []
    assert registry.get_by_base_model("org/no-such-base") == []


def test_get_by_base_model_lists_finetuned_children() -> None:
    """base_model queries return the finetuned children of a base repo id."""
    children = registry.get_by_base_model(PINNED_FINETUNED_BASE)
    assert PINNED_FINETUNED_REPO in [e.model for e in children]
    assert all(e.kind == "finetuned" for e in children)


def test_all_entries_count_and_file_order() -> None:
    """all_entries() returns all 236 entries in file order (pretrained first)."""
    entries = registry.all_entries()
    assert len(entries) == EXPECTED_TOTAL_ENTRIES
    assert entries[0].name == "Plant DNABERT"
    assert entries[0].kind == "pretrained"
    assert entries[-1].kind == "finetuned"


def test_is_lab_model_and_sort_key_partition_and_order() -> None:
    """is_lab_model partitions 189/47; sort_key_lab_first ranks lab first, name-stable."""
    lab = registry.resolve(PINNED_FINETUNED_BASE)
    third_party = registry.resolve(THIRD_PARTY_REPO)
    assert registry.is_lab_model(lab) is True
    assert registry.is_lab_model(third_party) is False

    entries = registry.all_entries()
    lab_count = sum(1 for e in entries if registry.is_lab_model(e))
    assert lab_count == EXPECTED_LAB_ENTRIES
    assert len(entries) - lab_count == EXPECTED_TOTAL_ENTRIES - EXPECTED_LAB_ENTRIES

    ranked = sorted(entries, key=registry.sort_key_lab_first)
    assert registry.is_lab_model(ranked[0]) is True
    assert registry.sort_key_lab_first(ranked[0]) < (1, "")
    lab_names = [e.name for e in ranked if registry.is_lab_model(e)]
    third_names = [e.name for e in ranked if not registry.is_lab_model(e)]
    assert lab_names == sorted(lab_names)
    assert third_names == sorted(third_names)


def test_subpackage_facade_exports_registry_api() -> None:
    """All eleven public registry names import from the dnallm.models facade."""
    from dnallm.models import (
        RegistryEntry,
        all_entries,
        effective_max_length,
        effective_tokenizer_type,
        get_by_base_model,
        get_by_task_type,
        is_lab_model,
        reload,
        resolve,
        sort_key_lab_first,
        try_resolve,
    )

    assert callable(all_entries)
    assert callable(effective_max_length)
    assert callable(effective_tokenizer_type)
    assert callable(get_by_base_model)
    assert callable(get_by_task_type)
    assert callable(is_lab_model)
    assert callable(reload)
    assert callable(resolve)
    assert callable(sort_key_lab_first)
    assert callable(try_resolve)
    assert issubclass(RegistryEntry, __import__("pydantic").BaseModel)


# ─────────────────────────────────────────────────────────────────────────────
# Effective-value inheritance over base_model lineage (D-02)
# ─────────────────────────────────────────────────────────────────────────────


def _synthetic_registry():
    """Crafted two-section registry exercising the inheritance axes of D-02."""
    mask_task = {"describe": "d", "task_type": "mask"}
    binary_task = {"describe": "d", "task_type": "binary"}
    data = {
        "pretrained": [
            {
                "name": "Synth Base",
                "model": "synth/base",
                "task": mask_task,
                "max_length": 1000,
                "tokenizer_type": "bpe",
            },
            {"name": "Synth Bare Base", "model": "synth/bare-base", "task": mask_task},
        ],
        "finetuned": [
            {
                "name": "Synth Own Value",
                "model": "synth/own",
                "base_model": "synth/base",
                "task": binary_task,
                "max_length": 512,
            },
            {
                "name": "Synth Inherit",
                "model": "synth/inherit",
                "base_model": "synth/base",
                "task": binary_task,
            },
            {
                "name": "Synth Child of Bare",
                "model": "synth/child-of-bare",
                "base_model": "synth/bare-base",
                "task": binary_task,
            },
            {
                "name": "Synth Ghost Child",
                "model": "synth/ghost-child",
                "base_model": "synth/ghost",
                "task": binary_task,
            },
            {
                "name": "Synth Cycle X",
                "model": "synth/x",
                "base_model": "synth/y",
                "task": binary_task,
            },
            {
                "name": "Synth Cycle Y",
                "model": "synth/y",
                "base_model": "synth/x",
                "task": binary_task,
            },
        ],
    }
    return registry._build_registry(data)


def test_effective_value_own_override_wins() -> None:
    """An entry's own value wins over the base chain (explicit override)."""
    data = _synthetic_registry()
    own = data.by_repo["synth/own"]
    assert registry._effective_value(own, "max_length", data) == 512
    assert registry._effective_value(own, "tokenizer_type", data) == "bpe"


def test_effective_value_inherits_from_base_chain() -> None:
    """A missing own value resolves from the base_model chain."""
    data = _synthetic_registry()
    inherit = data.by_repo["synth/inherit"]
    assert registry._effective_value(inherit, "max_length", data) == 1000
    assert registry._effective_value(inherit, "tokenizer_type", data) == "bpe"


def test_effective_value_none_when_absent_everywhere() -> None:
    """None propagates when neither the entry nor its chain carries a value."""
    data = _synthetic_registry()
    bare_base = data.by_repo["synth/bare-base"]
    child = data.by_repo["synth/child-of-bare"]
    assert registry._effective_value(bare_base, "max_length", data) is None
    assert registry._effective_value(child, "max_length", data) is None
    assert registry._effective_value(child, "tokenizer_type", data) is None


def test_effective_value_unresolvable_base_is_none() -> None:
    """A base_model that resolves to no registry entry surfaces as None."""
    data = _synthetic_registry()
    ghost_child = data.by_repo["synth/ghost-child"]
    assert registry._effective_value(ghost_child, "max_length", data) is None


def test_effective_value_cycle_returns_none() -> None:
    """A base_model cycle terminates via the visited-set guard, returning None."""
    data = _synthetic_registry()
    x = data.by_repo["synth/x"]
    y = data.by_repo["synth/y"]
    assert registry._effective_value(x, "max_length", data) is None
    assert registry._effective_value(y, "tokenizer_type", data) is None


def test_effective_max_length_delegates_against_live_cache(monkeypatch) -> None:
    """Public wrappers delegate to _effective_value over the live cached registry."""
    data = _synthetic_registry()
    monkeypatch.setattr(registry, "_load_registry", lambda: data)
    own = data.by_repo["synth/own"]
    assert registry.effective_max_length(own) == 512
    assert registry.effective_tokenizer_type(own) == "bpe"
    inherit = data.by_repo["synth/inherit"]
    assert registry.effective_max_length(inherit) == 1000


def test_effective_value_live_registry_pre_enrichment_is_none() -> None:
    """Over the real packaged file (22-01, pre-enrichment) missing values are None.

    22-02 lands tokenizer_type/max_length enrichment; this pins the 22-01
    None-surfacing contract over the live registry (no runtime probing).
    """
    entry = registry.resolve(PINNED_FINETUNED_NAME)
    assert registry.effective_max_length(entry) is None
    assert registry.effective_tokenizer_type(entry) is None


# ─────────────────────────────────────────────────────────────────────────────
# REG-03 read-only integrity guards (sole parser, no write API, frozen)
# ─────────────────────────────────────────────────────────────────────────────

# The only dnallm/**/*.py files allowed to contain the literal
# "model_info.yaml". The server.py hit is the stale integration comment at
# line ~821 ("# This would integrate with model_info.yaml") — the one
# pre-existing mentioner outside the loader, allowlisted here and scheduled
# for the Phase 26 MCP-consistency rewire. Any new mentioner in dnallm/ (a
# second parser, a would-be writer) fails the guard below.
EXPECTED_REGISTRY_MENTIONERS = {
    "dnallm/models/registry.py",
    "dnallm/mcp/server.py",
}

# REG-03: the loader is born read-only (Pitfall 3 — provenance destruction
# via a YAML round-trip). None of these verb substrings may appear in any
# public name of the module, so a write/serialize path can never grow
# without turning this suite red.
WRITE_API_VERB_SUBSTRINGS = (
    "write",
    "save",
    "serial",
    "dump",
    "set_",
    "update",
    "delete",
    "add_",
)

# File-open mode shape: one or more of r/w/x, optionally with b/t/+ (covers
# "r", "rb", "r+", "w", "wb", "a+", "x", ...). The loader may only ever open
# the packaged file in plain read mode.
FILE_MODE_PATTERN = re.compile(r"^[rwx][rwx+bt]*$")


def test_registry_module_is_sole_parser_of_model_info_yaml() -> None:
    """REG-01/REG-03: exactly registry.py plus the allowlisted stale comment mention the file."""
    package_root = Path(dnallm.__file__).parent
    mentioners = {
        # as_posix: forward-slash relative paths stay stable on Windows legs.
        path.relative_to(package_root.parent).as_posix()
        for path in package_root.rglob("*.py")
        if "model_info.yaml" in path.read_text(encoding="utf-8")
    }
    assert mentioners == EXPECTED_REGISTRY_MENTIONERS, (
        f"model_info.yaml mentioners drifted beyond the sole-parser allowlist "
        f"{sorted(EXPECTED_REGISTRY_MENTIONERS)}: unexpected "
        f"{sorted(mentioners - EXPECTED_REGISTRY_MENTIONERS)}, missing "
        f"{sorted(EXPECTED_REGISTRY_MENTIONERS - mentioners)} — registry.py is "
        f"the single parse owner; a second parser or writer belongs behind it, "
        f"never beside it"
    )


def test_registry_module_exposes_no_write_api() -> None:
    """REG-03 born-read-only: no public write/save/serialize/dump/update verb exists."""
    public_names = [name for name in dir(registry) if not name.startswith("_")]
    offenders = [
        name
        for name in public_names
        if any(verb in name.lower() for verb in WRITE_API_VERB_SUBSTRINGS)
    ]
    assert offenders == [], (
        f"dnallm.models.registry grew public names with write-side verbs "
        f"{offenders} — the registry is read-only forever (Pitfall 3); "
        f"registry maintenance is scripted text insertion, never a runtime "
        f"mutation API"
    )


def test_registry_module_source_has_no_yaml_emitter() -> None:
    """REG-03: the module source contains no pyyaml emitter token and opens only in read mode."""
    source = Path(registry.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    for token in ("dump", "dumper", "representer"):
        assert token not in lowered, (
            f"registry.py contains pyyaml emitter token {token!r} — the loader "
            f"can never emit YAML (Pitfall 3 round-trip destroys provenance)"
        )
    string_constants = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    mode_literals = {s for s in string_constants if FILE_MODE_PATTERN.match(s)}
    assert mode_literals <= {"r"}, (
        f"registry.py carries file-open mode literal(s) beyond read mode: "
        f"{sorted(mode_literals)} — the packaged registry file is never written"
    )


def test_registry_entries_are_frozen() -> None:
    """REG-03: entries are frozen — attribute assignment raises ValidationError."""
    entry = registry.resolve(PINNED_FINETUNED_NAME)
    with pytest.raises(ValidationError):
        entry.name = "mutated"
