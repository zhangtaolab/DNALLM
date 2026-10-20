"""Runtime model registry: the sole parser of the packaged ``model_info.yaml``.

This module turns the packaged registry YAML (236 entries: 49 pretrained +
187 finetuned) into typed, frozen, runtime-consumable data (REG-01). It is
the single parse owner of ``model_info.yaml`` inside ``dnallm/`` — a grep
guard in ``tests/models/test_registry.py`` fails CI if any second mentioner
appears — and it is born read-only (REG-03): there is no write, save,
serialize, or normalize-back API of any kind, and entries are frozen Pydantic
models. Registry maintenance is scripted text insertion, never a YAML
round-trip (see the 261020-kwt discipline and PITFALLS.md Pitfall 3).

The file is parsed exactly once per process: access goes through the
``functools.lru_cache``-decorated :func:`_load_registry`, and repeated
accessor calls return identical entry objects. ``reload()`` clears that
cache so the next access re-parses; it exists as a single-threaded
test/consistency hook only — no thread-safety contract is offered or needed
(entries are immutable and the parse is a pure function of the packaged
file). Importing this module never opens the file: parsing is lazy, on first
accessor call.

Validation is hard at first access (D-01b): a malformed entry, duplicate
names or repo ids, or a missing section fails loudly as a descriptive
``ValueError`` naming the offender instead of corrupting downstream
behavior.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..utils import get_logger

logger = get_logger("dnallm.models.registry")

__all__ = [
    "RegistryEntry",
    "all_entries",
    "effective_max_length",
    "effective_tokenizer_type",
    "get_by_base_model",
    "get_by_task_type",
    "is_lab_model",
    "reload",
    "resolve",
    "sort_key_lab_first",
    "try_resolve",
]


class TaskInfo(BaseModel):
    """Typed task metadata block of a registry entry.

    Attributes:
        describe: Human-readable task description (non-empty).
        task_type: Task class such as ``mask``, ``generation``, or ``binary``.
        num_labels: Number of output labels when the entry ships a head.
        label_names: Ordered label names when the entry ships a head. The
            packaged file also carries bare-string values on regression
            entries (a free-text target description); those are accepted and
            wrapped into a one-element list, matching the downstream
            ``TaskConfig`` regression convention (``label_names == ["value"]``).
        threshold: Decision threshold when the entry ships a head.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    describe: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    num_labels: int | None = Field(default=None, gt=0)
    label_names: list[str] | None = None
    threshold: float | None = None

    @field_validator("label_names", mode="before")
    @classmethod
    def _wrap_bare_string_label_names(cls, value):
        """Accept the regression-entry convention of a bare-string label_names.

        The packaged registry stores a free-text target description (for
        example ``"promoter strength in tobacco leaves"``) in ``label_names``
        on regression entries. Wrapping it into a one-element list keeps the
        published contract a plain ``list[str] | None`` for every consumer
        without mutating the packaged file.
        """
        if isinstance(value, str):
            return [value]
        return value


class RegistryEntry(BaseModel):
    """One frozen, typed entry of the packaged model registry.

    Attributes:
        name: Unique human-readable entry name.
        model: Repo id (Hugging Face / ModelScope), unique across sections.
        task: Typed task metadata block.
        kind: Which packaged section the entry came from.
        tokenizer_type: Tokenizer flavor of the entry, when enriched.
        max_length: Maximum context length in tokens, when enriched;
            finetuned entries may inherit the base value via ``base_model``.
        base_model: Repo id of the pretrained base of a finetuned entry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    task: TaskInfo
    kind: Literal["pretrained", "finetuned"]
    tokenizer_type: str | None = Field(default=None, min_length=1)
    max_length: int | None = Field(default=None, gt=0)
    base_model: str | None = Field(default=None, min_length=1)


@dataclass(frozen=True)
class _RegistryData:
    """Internal immutable parse result: ordered entries plus lookup indexes."""

    entries: tuple[RegistryEntry, ...]
    by_name: dict[str, RegistryEntry] = field(compare=False)
    by_repo: dict[str, RegistryEntry] = field(compare=False)


def _build_registry(data: object) -> _RegistryData:
    """Validate raw parsed registry data into indexed, typed entries.

    Args:
        data: The parsed YAML document (expected mapping with ``pretrained``
            and ``finetuned`` list sections).

    Returns:
        The immutable parse result: entries in file order (pretrained section
        first) plus name and repo-id lookup indexes.

    Raises:
        ValueError: If the document is not a mapping, a section is missing /
            empty / not a list, an entry is not a mapping, an entry violates
            the schema (naming the offending repo id), or an entry name or
            repo id is duplicated across the two sections.
    """
    if not isinstance(data, dict):
        raise ValueError(
            "The packaged model registry is malformed: the top-level document "
            f"must be a mapping, got {type(data).__name__}."
        )
    sections: dict[str, list[dict]] = {}
    for section in ("pretrained", "finetuned"):
        raw = data.get(section)
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                f"The packaged model registry is malformed: the {section!r} section "
                "is missing or is not a non-empty list."
            )
        sections[section] = raw

    entries: list[RegistryEntry] = []
    for section, raw_entries in sections.items():
        for position, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    f"The packaged model registry is malformed: the entry at "
                    f"position {position} of the {section!r} section must be a "
                    f"mapping, got {type(raw_entry).__name__}."
                )
            try:
                entries.append(RegistryEntry.model_validate({**raw_entry, "kind": section}))
            except ValidationError as e:
                model_id = raw_entry.get("model", f"<missing model id, position {position}>")
                raise ValueError(
                    f"The packaged model registry is malformed: entry "
                    f"{model_id!r} failed schema validation: {e}"
                ) from e

    by_name: dict[str, RegistryEntry] = {}
    by_repo: dict[str, RegistryEntry] = {}
    for entry in entries:
        if entry.name in by_name:
            raise ValueError(
                f"The packaged model registry is malformed: duplicate entry name {entry.name!r}."
            )
        if entry.model in by_repo:
            raise ValueError(
                f"The packaged model registry is malformed: duplicate repo id {entry.model!r}."
            )
        by_name[entry.name] = entry
        by_repo[entry.model] = entry

    logger.debug(f"Parsed {len(entries)} registry entries from the packaged file.")
    return _RegistryData(entries=tuple(entries), by_name=by_name, by_repo=by_repo)


@lru_cache(maxsize=1)
def _load_registry() -> _RegistryData:
    """Parse the packaged registry YAML exactly once per process.

    Opens ``model_info.yaml`` from the ``dnallm.models`` package data via
    importlib.resources (wheel-safe, never CWD-relative) with UTF-8 encoding
    and hard-validates it through :func:`_build_registry`.

    Returns:
        The immutable parse result.

    Raises:
        ValueError: If the packaged resource is missing, unparseable, or
            malformed (see :func:`_build_registry`).
    """
    import yaml

    try:
        resource = resources.files("dnallm.models").joinpath("model_info.yaml")
        with resource.open("r", encoding="utf-8") as f:
            # CSafeLoader is the C implementation of SafeLoader (12x faster on
            # this file); SafeLoader is the fallback where libyaml is absent.
            # Both are safe loaders — S506 cannot resolve the getattr and
            # fires on any yaml.load call regardless of the Loader argument.
            data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))  # ruff: ignore[unsafe-yaml-load]
    except (FileNotFoundError, ModuleNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load the packaged model registry: {e}") from e
    return _build_registry(data)


def resolve(name_or_repo_id: str) -> RegistryEntry:
    """Look up a registry entry by exact entry name or repo id.

    Args:
        name_or_repo_id: The entry name or repo id to look up. Matching is
            exact UTF-8 string equality — no casefolding, trimming, or
            Unicode normalization.

    Returns:
        The typed frozen entry.

    Raises:
        ValueError: If no entry matches.
    """
    registry_data = _load_registry()
    entry = registry_data.by_name.get(name_or_repo_id)
    if entry is None:
        entry = registry_data.by_repo.get(name_or_repo_id)
    if entry is None:
        raise ValueError(f"Model {name_or_repo_id} not found in registry.")
    return entry


def try_resolve(name_or_repo_id: str) -> RegistryEntry | None:
    """Look up a registry entry, returning ``None`` on a miss.

    The fall-through seam for callers (Phase 24 preset resolution) that
    degrade to a legacy path instead of raising.

    Args:
        name_or_repo_id: The entry name or repo id to look up (exact match).

    Returns:
        The typed frozen entry, or ``None`` if no entry matches.
    """
    registry_data = _load_registry()
    entry = registry_data.by_name.get(name_or_repo_id)
    if entry is None:
        entry = registry_data.by_repo.get(name_or_repo_id)
    return entry


def reload() -> None:
    """Clear the parse cache so the next accessor call re-parses the file.

    Single-threaded test/consistency hook: entries are immutable and the
    parse is idempotent, so concurrent access is safe by construction and
    this function is not needed at runtime.
    """
    _load_registry.cache_clear()


def get_by_task_type(task_type: str) -> list[RegistryEntry]:
    """Return every entry whose task_type matches, in file order.

    Queries span both packaged sections (pretrained and finetuned).

    Args:
        task_type: Task class such as ``mask``, ``generation``, or ``binary``
            (exact match).

    Returns:
        All matching entries in file order; an empty list when nothing
        matches (never an entry, never a KeyError).
    """
    return [e for e in _load_registry().entries if e.task.task_type == task_type]


def get_by_base_model(repo_id: str) -> list[RegistryEntry]:
    """Return every finetuned entry whose base_model equals the repo id.

    The base<->finetuned axis Phase 23's intent routing consumes ("fine-tune"
    lists bases, "predict now" lists task models).

    Args:
        repo_id: Repo id of the base model (exact match).

    Returns:
        All matching finetuned entries in file order; an empty list when
        nothing matches.
    """
    return [e for e in _load_registry().entries if e.base_model == repo_id]


def all_entries() -> list[RegistryEntry]:
    """Return all registry entries in file order (pretrained section first).

    Returns:
        A fresh list of the 236 typed frozen entries.
    """
    return list(_load_registry().entries)


def is_lab_model(entry: RegistryEntry) -> bool:
    """Return whether the entry is a lab model (``zhangtaolab/`` repo prefix).

    The PFC-07 contract anchor: lab models rank first in every listing the
    product surfaces.

    Args:
        entry: The registry entry to classify.

    Returns:
        True iff the entry's repo id starts with ``zhangtaolab/``.
    """
    return entry.model.startswith("zhangtaolab/")


def sort_key_lab_first(entry: RegistryEntry) -> tuple[int, str]:
    """Lab-first sort key: lab models before third-party, name-stable within each.

    Args:
        entry: The registry entry to rank.

    Returns:
        A ``(partition, name)`` tuple such that ``sorted(entries,
        key=sort_key_lab_first)`` lists ``zhangtaolab/*`` entries first with
        ascending-name order inside each partition.
    """
    return (0 if is_lab_model(entry) else 1, entry.name)


def _effective_value(entry: RegistryEntry, attr: str, data: _RegistryData):
    """Resolve an entry's effective value along the base_model chain (D-02).

    The entry's own value wins; otherwise the chain of base models is
    followed (resolved by repo id against the given parse result) with a
    visited-set cycle guard, returning the first ancestor carrying a value.

    Args:
        entry: The starting entry.
        attr: Attribute name to resolve (``max_length`` or
            ``tokenizer_type``).
        data: The registry parse result to resolve base models against.

    Returns:
        The effective value, or None when neither the entry nor its chain
        carries one (including unresolvable bases and cycles — never a loop).
    """
    visited: set[str] = set()
    current: RegistryEntry | None = entry
    while current is not None:
        if current.model in visited:
            return None
        visited.add(current.model)
        value = getattr(current, attr)
        if value is not None:
            return value
        if current.base_model is None:
            return None
        current = data.by_repo.get(current.base_model)
    return None


def effective_max_length(entry: RegistryEntry) -> int | None:
    """Return the entry's effective max context length in tokens (D-02).

    The entry's own ``max_length`` when set, otherwise the first value found
    along its ``base_model`` chain; None when the chain carries none. No
    runtime probing — missing enrichment surfaces as None.

    Args:
        entry: The registry entry.

    Returns:
        The effective positive int, or None.
    """
    return _effective_value(entry, "max_length", _load_registry())


def effective_tokenizer_type(entry: RegistryEntry) -> str | None:
    """Return the entry's effective tokenizer flavor (D-02).

    The entry's own ``tokenizer_type`` when set, otherwise the first value
    found along its ``base_model`` chain; None when the chain carries none.

    Args:
        entry: The registry entry.

    Returns:
        The effective tokenizer type, or None.
    """
    return _effective_value(entry, "tokenizer_type", _load_registry())
