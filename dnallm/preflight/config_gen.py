"""Validated config generation and round-trip editing (PFC-04, plan 23-04).

1. Zero-traceback contract: every user-input error path returns a
   :class:`~dnallm.preflight.report.GeneratedConfig` carrying field-level
   :class:`~dnallm.preflight.report.SectionError` records — an unknown
   model, a missing task type, an invalid section value, or an unknown
   patch key is a record in ``errors``, never a raised exception.
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
4. Import-light: no torch/transformers/datasets and no hub clients
   anywhere in this module (structure-test enforced), so any shell can
   generate a config without the training stack.
5. Safe YAML only: round-trip loading uses the safe-loader family;
   output is written with ``yaml.safe_dump`` to exactly the user-named
   ``output_path`` and never otherwise (D-15 — no surprise writes, no
   unsafe constructors).

The registry-derived context length is emitted into both the ``inference``
section (where ``InferenceConfig.max_length`` validates it and the runtime
consumes it) and the raw ``finetune`` section (matching the packaged
example-config shape).
"""

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


def _dump_yaml(config: dict, output_path: str | Path) -> None:
    """Write a config mapping to exactly the given path with ``yaml.safe_dump``.

    The only write in this module, reached exclusively when a caller
    passes ``output_path`` explicitly; nothing is derived beyond
    ``os.path.expanduser`` and no other file is ever touched.

    Args:
        config: The config mapping to serialize.
        output_path: The exact destination path.
    """
    path = os.path.expanduser(str(output_path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


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
        output_path: When given, the assembled config mapping is written
            to exactly this path (``yaml.safe_dump``); when omitted,
            nothing is written anywhere.

    Returns:
        The :class:`GeneratedConfig` with every successfully validated
        section and the ordered field-level error records.
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
    if output_path is not None:
        _dump_yaml(config_dict, output_path)
    logger.debug(
        f"Generated config for {entry.model}: {len(configs)} sections, "
        f"{len(errors) + len(validation_errors)} errors."
    )
    return GeneratedConfig(config=configs, errors=errors + validation_errors)
