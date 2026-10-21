"""Validated config generation and round-trip editing (PFC-04, plan 23-04).

1. Zero-traceback contract: every user-input error path returns a
   :class:`~dnallm.preflight.report.GeneratedConfig` carrying field-level
   :class:`~dnallm.preflight.report.SectionError` records — an unknown
   model, a missing task type, an invalid section value, or an unknown
   patch key is a record in ``errors``, never a raised exception. The one
   sanctioned exception: a nonexistent/unreadable ``existing`` path raises
   a matchable ``ValueError`` (WR-01), mirroring ``validate_data``'s
   not-found contract.
2. Collect-all: multiple broken sections each contribute their own error
   records in the fixed section order (``_SECTION_ORDER``); one section's
   failure never hides another's. A fully valid mapping validates through
   the shared :func:`dnallm.configuration.configs.validate_sections` path
   (which keeps the raise-first behavior every existing ``load_config``
   caller depends on — research Pitfall 7); the collect-all wrapper is
   used by this module only.
3. Registry-as-truth-source: model-derived defaults (task prefill for
   finetuned entries, the effective context length) come from
   ``dnallm.models.registry`` via the non-raising ``try_resolve`` lookup
   (research Pitfall 8) and ``effective_max_length``.
4. Import-light at the SOURCE level: no torch/transformers/datasets and
   no hub clients anywhere in this module (AST structure-test enforced).
   Runtime import purity is closed: a shell generating a config without
   the training stack already in ``sys.modules`` works, because the
   ``dnallm.utils`` / ``dnallm.models`` package facades resolve their
   heavy members lazily (subprocess marker test in
   ``tests/test_package_facade.py``).
5. Safe YAML only: round-trip loading uses the safe-loader family;
   output is written with ``yaml.safe_dump`` to exactly the user-named
   ``output_path`` and never otherwise (D-15 — no surprise writes, no
   unsafe constructors). There is exactly ONE serializer: the public
   :func:`render_yaml` renders a mapping to the document text and
   ``_dump_yaml`` writes that text, so a shell previewing ``render_yaml``
   can never diverge from the bytes on disk. Both :func:`generate_config`
   and :func:`edit_config` carry the mapping they would write on
   ``GeneratedConfig.raw``.

The registry-derived context length is emitted into both the ``inference``
section (where ``InferenceConfig.max_length`` validates it and the runtime
consumes it) and the raw ``finetune`` section (matching the packaged
example-config shape).
"""

import copy
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..configuration.configs import (
    DNALLMConfig,
    BenchmarkConfig,
    Ia3Config,
    InferenceConfig,
    LoraConfig,
    SweepConfig,
    TaskConfig,
    TrainingConfig,
    VepConfig,
    validate_sections,
)
from ..models import registry
from ..utils import get_logger
from .report import GeneratedConfig, SectionError

logger = get_logger("dnallm.preflight.config_gen")

__all__ = ["edit_config", "generate_config", "render_yaml"]

# The fixed, deterministic error-ordering contract: error records are
# ordered by this section order, then by each section's own field order.
_SECTION_ORDER = (
    "task",
    "model",
    "inference",
    "finetune",
    "lora",
    "ia3",
    "vep",
    "sweep",
    "benchmark",
)

# The validated sections and their classes — the same classes
# validate_sections constructs, applied per section by the collect-all
# wrapper. ``model`` is a passthrough dict; ``benchmark`` consumes the
# full raw dict.
_SECTION_CLASSES = {
    "task": TaskConfig,
    "inference": InferenceConfig,
    "finetune": TrainingConfig,
    "lora": LoraConfig,
    "ia3": Ia3Config,
    "vep": VepConfig,
    "sweep": SweepConfig,
}


def _validation_errors(section: str, exc: ValidationError) -> list[SectionError]:
    """Map a Pydantic ValidationError to ordered field-level records.

    Args:
        section: Config section the exception came from.
        exc: The raised ValidationError.

    Returns:
        One :class:`SectionError` per ``exc.errors()`` entry, in the
        exception's own field order; the field path is ``section.loc``
        joined with dots, or the section itself for model-level errors
        (empty loc).
    """
    records: list[SectionError] = []
    for entry in exc.errors():
        loc = ".".join(str(part) for part in entry.get("loc", ()))
        records.append(
            SectionError(
                section=section,
                field=f"{section}.{loc}" if loc else section,
                message=str(entry.get("msg", "")),
            )
        )
    return records


def _validate_collect_all(config_dict: dict) -> tuple[DNALLMConfig, list[SectionError]]:
    """Validate every recognized section, collecting all field-level errors.

    A fully valid mapping is validated through the shared
    :func:`dnallm.configuration.configs.validate_sections` path; on any
    validation failure the same section classes are applied per section so
    every broken section contributes its own error records (one failure
    never hides another). A section whose value is null is treated as
    absent; a section holding ``{}`` constructs with defaults.

    Args:
        config_dict: The raw config mapping (unknown top-level keys are
            ignored, as in ``load_config``).

    Returns:
        A ``(configs, errors)`` tuple: ``configs`` holds every
        successfully constructed section, ``errors`` the ordered
        field-level records (fixed section order, then field order).
    """
    present = [name for name in _SECTION_ORDER if config_dict.get(name) is not None]
    if not present:
        return {}, [
            SectionError(
                section="config",
                field="sections",
                message="NO_SECTIONS: no recognized config sections found",
            )
        ]

    # Fast path: the shared validate_sections call validates a fully valid
    # mapping exactly as load_config would.
    try:
        return validate_sections(config_dict), []
    except (ValidationError, TypeError):
        pass

    configs: DNALLMConfig = {}
    errors: list[SectionError] = []
    for section in _SECTION_ORDER:
        if section not in config_dict or config_dict[section] is None:
            continue
        value = config_dict[section]
        if section == "model":
            # Plain-dict passthrough, exactly as validate_sections does.
            configs["model"] = value
            continue
        if section == "benchmark":
            # Consumes the FULL raw dict, exactly as validate_sections does.
            try:
                configs["benchmark"] = BenchmarkConfig(**config_dict)
            except ValidationError as exc:
                errors.extend(_validation_errors("benchmark", exc))
            continue
        if not isinstance(value, dict):
            errors.append(
                SectionError(
                    section=section,
                    field=section,
                    message=f"section '{section}' must be a mapping, got {type(value).__name__}",
                )
            )
            continue
        try:
            configs[section] = _SECTION_CLASSES[section](**value)
        except ValidationError as exc:
            errors.extend(_validation_errors(section, exc))
    return configs, errors


def render_yaml(config: dict) -> str:
    """Render a config mapping to the exact text the writer emits.

    One serializer for the whole package: :func:`_dump_yaml` writes the
    string this function returns, so a preview built from it can never
    diverge from the bytes that land on disk. Nothing is written here —
    the returned string is the caller's to display or discard.

    Args:
        config: The config mapping to serialize.

    Returns:
        The YAML document text, with keys in the mapping's own order
        (``sort_keys=False``) and block style (``default_flow_style=False``).
    """
    return yaml.safe_dump(config, default_flow_style=False, sort_keys=False)


def _dump_yaml(config: dict, output_path: str | Path) -> None:
    """Write a config mapping to exactly the given path with ``yaml.safe_dump``.

    The only write in this module, reached exclusively when a caller
    passes ``output_path`` explicitly; nothing is derived beyond
    ``os.path.expanduser`` and no other file is ever touched. The document
    itself comes from :func:`render_yaml`, so the file is byte-identical to
    what a caller can preview.

    The document is serialized BEFORE the destination is opened, and written
    through a sibling temporary file that replaces the target atomically.
    Opening the destination with ``"w"`` first would truncate the user's
    config before a serialization error could be raised, destroying the file
    the failed run was supposed to leave untouched (WR-04).

    Args:
        config: The config mapping to serialize.
        output_path: The exact destination path.
    """
    # May raise (e.g. RepresenterError); nothing has been touched yet.
    text = render_yaml(config)
    path = os.path.expanduser(str(output_path))
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        # Never leave a scratch file beside the destination on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def generate_config(
    model: str,
    *,
    task_type: str | None = None,
    output_path: str | Path | None = None,
) -> GeneratedConfig:
    """Emit a complete validated config for a registry model (D-13).

    For a finetuned entry the task section is prefilled from the registry
    entry (``task_type`` / ``num_labels`` / ``label_names`` /
    ``threshold``); a base (pretrained) entry requires the ``task_type``
    argument, otherwise a ``TASK_TYPE_REQUIRED`` field-level error is
    returned. The registry-derived context length
    (``effective_max_length``) is emitted into the ``inference`` section
    (validated and runtime-consumed) and the raw ``finetune`` section
    (packaged example-config shape). An unknown model yields a
    ``MODEL_NOT_IN_REGISTRY_CFG`` field-level error — never an exception.

    Args:
        model: Model name or repo id to look up in the registry.
        task_type: Task type for base models (ignored for finetuned
            entries, whose registry task wins).
        output_path: When given and the run is error-free, the assembled
            config mapping is written to exactly this path
            (``yaml.safe_dump``); when omitted, or when the result carries
            any error records, nothing is written anywhere (WR-02).

    Returns:
        The :class:`GeneratedConfig` with every successfully validated
        section, the ordered field-level error records, and the assembled
        raw mapping (``raw``) that would be written on a clean run.
    """
    entry = registry.try_resolve(model)
    if entry is None:
        return GeneratedConfig(
            config={},
            errors=[
                SectionError(
                    section="model",
                    field="model",
                    message=f"MODEL_NOT_IN_REGISTRY_CFG: {model} not found in the model registry",
                )
            ],
        )

    errors: list[SectionError] = []
    config_dict: dict = {}

    if entry.kind == "finetuned":
        task_section = {
            key: value
            for key, value in (
                ("task_type", entry.task.task_type),
                ("num_labels", entry.task.num_labels),
                ("label_names", entry.task.label_names),
                ("threshold", entry.task.threshold),
            )
            if value is not None
        }
        config_dict["task"] = task_section
    elif task_type is not None:
        config_dict["task"] = {"task_type": task_type}
    else:
        errors.append(
            SectionError(
                section="task",
                field="task_type",
                message=(
                    f"TASK_TYPE_REQUIRED: {entry.model} is a base model and ships no task; "
                    "pass task_type explicitly"
                ),
            )
        )

    config_dict["model"] = {"name": entry.model}

    effective_length = registry.effective_max_length(entry)
    if isinstance(effective_length, int) and effective_length > 0:
        config_dict["finetune"] = {"max_length": effective_length}
        config_dict["inference"] = {"max_length": effective_length}
    else:
        config_dict["finetune"] = {}
        config_dict["inference"] = {}
    config_dict["lora"] = {}

    configs, validation_errors = _validate_collect_all(config_dict)
    if output_path is not None and not (errors + validation_errors):
        # WR-02: an error-carrying result is never materialized on disk —
        # callers must not mistake file existence for validity.
        _dump_yaml(config_dict, output_path)
    logger.debug(
        f"Generated config for {entry.model}: {len(configs)} sections, "
        f"{len(errors) + len(validation_errors)} errors."
    )
    return GeneratedConfig(config=configs, errors=errors + validation_errors, raw=config_dict)


def _load_existing(existing: str | Path) -> dict:
    """Load the round-trip input mapping from a YAML path.

    Uses the safe-loader family only (CSafeLoader with a SafeLoader
    fallback — both are safe loaders), so a document carrying a
    ``python/object``-style tag is refused with a ``yaml.YAMLError``
    instead of constructing arbitrary objects (T-23-08).

    Args:
        existing: Path to the YAML file.

    Returns:
        The parsed mapping (an empty dict for an empty document).

    Raises:
        ValueError: If the path does not exist or cannot be opened — a
            matchable message mirroring ``validate_data``'s not-found
            contract (WR-01), never a raw ``OSError`` traceback.
        yaml.YAMLError: If the document is not safely constructible.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        with open(existing, encoding="utf-8") as f:
            data = yaml.load(f, Loader=loader)  # ruff: ignore[unsafe-yaml-load]
    except FileNotFoundError as e:
        raise ValueError(f"Existing config not found: {existing}.") from e
    except OSError as e:
        raise ValueError(f"Existing config not readable: {existing}: {e}") from e
    return data if isinstance(data, dict) else {}


def _apply_patch(config_dict: dict, patch: dict) -> list[SectionError]:
    """Apply a patch dict per-field at section level (no recursive merge).

    Each patch key naming a known section overwrites exactly the named
    fields inside that section's dict (a missing or non-mapping section
    dict is created fresh). The ``model`` section is the one passthrough:
    any non-mapping patch value replaces it wholesale. A patch key outside
    ``_SECTION_ORDER`` yields an ``UNKNOWN_SECTION`` field-level error and
    never enters the config.

    Args:
        config_dict: The config mapping to patch (mutated in place).
        patch: The patch mapping (section -> field overrides).

    Returns:
        The patch-application errors (empty on a fully applicable patch).
    """
    errors: list[SectionError] = []
    for key, value in patch.items():
        if key not in _SECTION_ORDER:
            errors.append(
                SectionError(
                    section=key,
                    field=key,
                    message=f"UNKNOWN_SECTION: {key!r} is not a recognized config section",
                )
            )
            continue
        if key == "model":
            # Passthrough section: full replacement with any value shape.
            config_dict["model"] = value
            continue
        if not isinstance(value, dict):
            errors.append(
                SectionError(
                    section=key,
                    field=key,
                    message=f"section '{key}' patch must be a mapping, got {type(value).__name__}",
                )
            )
            continue
        base = config_dict.get(key)
        if not isinstance(base, dict):
            base = {}
        base.update(value)
        config_dict[key] = base
    return errors


def edit_config(
    existing: str | Path | dict,
    patch: dict | None = None,
    *,
    output_path: str | Path | None = None,
) -> GeneratedConfig:
    """Load an existing config, apply a patch, and revalidate (D-14).

    The input is a YAML path (loaded with the safe-loader family) or an
    already-parsed mapping; the patch overwrites fields per section (no
    recursive merge) and the patched mapping revalidates through the
    shared validation path with collect-all field-level errors — an
    unknown patch section, an invalid value, or a document with no
    recognized sections is a record in ``errors``, never a traceback.

    Nothing is written unless ``output_path`` is explicitly given AND the
    run produced no error records, in which case the patched mapping is
    dumped (``yaml.safe_dump``) to exactly that path (WR-02: file
    existence always implies a valid config — an error-carrying result is
    never silently materialized on disk); the input file is never modified
    and the caller's dict is never mutated.

    Args:
        existing: YAML path or parsed config mapping to start from.
        patch: Field overrides per section (``{}`` / None revalidates the
            input unchanged).
        output_path: Exact destination path for the patched mapping;
            omitted means nothing is written anywhere.

    Returns:
        The :class:`GeneratedConfig` with every successfully validated
        section, the ordered field-level error records, and the patched raw
        mapping (``raw``) — the exact mapping the next write would emit,
        unpatched sections included verbatim (D-11).

    Raises:
        ValueError: If ``existing`` is a path that does not exist or
            cannot be read — a matchable message consistent with
            ``validate_data``'s not-found contract (WR-01).
    """
    if isinstance(existing, dict):
        # Deep copy: patching overwrites fields inside section dicts, and
        # the caller's mapping (nested dicts included) must stay untouched.
        config_dict: dict = copy.deepcopy(existing)
    else:
        config_dict = _load_existing(existing)

    patch_errors = _apply_patch(config_dict, patch or {})
    configs, validation_errors = _validate_collect_all(config_dict)

    if output_path is not None and not (patch_errors + validation_errors):
        # WR-02 (same rule as generate_config): only an error-free run is
        # dumped — file existence must imply a valid config.
        _dump_yaml(config_dict, output_path)
    logger.debug(
        f"Edited config: {len(configs)} sections, "
        f"{len(patch_errors) + len(validation_errors)} errors."
    )
    return GeneratedConfig(
        config=configs,
        errors=patch_errors + validation_errors,
        raw=config_dict,
    )
