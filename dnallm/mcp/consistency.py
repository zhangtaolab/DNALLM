"""Read-only consistency checker for MCP model configs against the packaged registry.

Compares every ``*_inference_config.yaml`` in a config directory with the
Phase-22 model registry and reports every divergence as a pre-flight
:class:`~dnallm.preflight.report.Finding`, so config drift fails CI instead
of silently shipping confidently-wrong predictions — a swapped
``label_names`` order inverts every prediction's meaning without raising an
error. Every ``mcp_server_config*.yaml`` in the directory additionally has
its ``models[*].config_path`` references checked for integrity.

1. Repo-id keying: each config is matched to its registry entry via
   ``registry.try_resolve(config["model"]["path"])`` — the same seam the
   preset loader uses. The registry YAML itself is never opened here
   (REG-03: ``dnallm/models/registry.py`` is the sole parser).
2. Tier table: the four hard fields (``task_type``, ``num_labels``,
   ``label_names`` order, ``threshold``) produce WARN findings that flip the
   strict pass signal; a derived source mismatch (lab entries only), a
   ``max_length`` divergence, and an unknown repo id produce WARN findings
   that never do; description divergence is INFO; an unreadable or invalid
   config is ERROR (fatal in both modes).
3. Collect-all: every mismatch in every config is reported in one pass —
   the scan never stops at the first failure and never skips a file.
4. Never writes: this module is a read path. It opens config files for
   reading only and contains no YAML-dump or write primitive of any kind.
5. Finding reuse: findings are the frozen pre-flight ``Finding`` objects
   (severity + reason code + plain-language message), so shells route on
   codes exactly like the pre-flight core.

Example:
    >>> from dnallm.mcp import check_config_registry_consistency
    >>> report = check_config_registry_consistency("dnallm/mcp/configs", strict_registry=True)
    >>> report.passed
    True
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..models.registry import RegistryEntry, effective_max_length, is_lab_model, try_resolve
from ..preflight.report import CompatReasonCode, Finding, Severity
from ..utils import get_logger

logger = get_logger("dnallm.mcp.consistency")

__all__ = [
    "CONFIG_INVALID",
    "CONFIG_UNREADABLE",
    "DANGLING_CONFIG_REFERENCE",
    "DESCRIPTION_DIVERGED",
    "LABEL_ORDER_MISMATCH",
    "MAX_LENGTH_MISMATCH",
    "MODEL_NOT_IN_REGISTRY",
    "NUM_LABELS_MISMATCH",
    "SOURCE_MISMATCH",
    "STRICT_FATAL_REASON_CODES",
    "TASK_TYPE_MISMATCH",
    "THRESHOLD_MISMATCH",
    "ConsistencyReport",
    "check_config_registry_consistency",
]

# Reason codes. MODEL_NOT_IN_REGISTRY and TASK_TYPE_MISMATCH reuse the
# CompatReasonCode vocabulary verbatim; the rest follow the same convention.
MODEL_NOT_IN_REGISTRY = CompatReasonCode.MODEL_NOT_IN_REGISTRY.value
TASK_TYPE_MISMATCH = CompatReasonCode.TASK_TYPE_MISMATCH.value
NUM_LABELS_MISMATCH = "NUM_LABELS_MISMATCH"
LABEL_ORDER_MISMATCH = "LABEL_ORDER_MISMATCH"
THRESHOLD_MISMATCH = "THRESHOLD_MISMATCH"
SOURCE_MISMATCH = "SOURCE_MISMATCH"
MAX_LENGTH_MISMATCH = "MAX_LENGTH_MISMATCH"
DESCRIPTION_DIVERGED = "DESCRIPTION_DIVERGED"
CONFIG_UNREADABLE = "CONFIG_UNREADABLE"
CONFIG_INVALID = "CONFIG_INVALID"
DANGLING_CONFIG_REFERENCE = "DANGLING_CONFIG_REFERENCE"

# The D-04 hard fields: drift on exactly these flips the strict pass signal.
STRICT_FATAL_REASON_CODES: frozenset[str] = frozenset({
    TASK_TYPE_MISMATCH,
    NUM_LABELS_MISMATCH,
    LABEL_ORDER_MISMATCH,
    THRESHOLD_MISMATCH,
})


class ConsistencyReport(BaseModel):
    """Frozen result of a config-vs-registry consistency scan.

    Attributes:
        configs_checked: Sorted names of every config file the scan covered.
        findings: Every mismatch found, in scan order (collect-all).
        strict_registry: Whether the strict pass signal was requested.
        passed: The machine-readable signal: False when any ERROR-severity
            finding exists, or (strict mode) any hard-field mismatch.
    """

    model_config = ConfigDict(frozen=True)

    configs_checked: list[str]
    findings: list[Finding] = Field(default_factory=list)
    strict_registry: bool = False
    passed: bool = True


def check_config_registry_consistency(
    config_dir: str | Path, *, strict_registry: bool = False
) -> ConsistencyReport:
    """Check every MCP config in a directory against the registry.

    Args:
        config_dir: Directory holding the ``*_inference_config.yaml`` files
            and ``mcp_server_config*.yaml`` server configs to check
            (read-only; nothing is ever written back).
        strict_registry: When True, hard-field drift (the four D-04 reason
            codes) flips :attr:`ConsistencyReport.passed` to False — the CI
            gate mode. Defaults to warn-first.

    Returns:
        The frozen report listing every file checked and every finding —
        all mismatches from all files in one pass, never first-fail.
    """
    directory = Path(config_dir)
    configs_checked: list[str] = []
    findings: list[Finding] = []
    for path in sorted(directory.glob("*_inference_config.yaml")):
        configs_checked.append(path.name)
        findings.extend(_check_one_model_config(path))
    for path in sorted(directory.glob("mcp_server_config*.yaml")):
        configs_checked.append(path.name)
        findings.extend(_check_one_server_config(path, directory))
    passed = not any(
        finding.severity == Severity.ERROR
        or (strict_registry and finding.reason_code in STRICT_FATAL_REASON_CODES)
        for finding in findings
    )
    return ConsistencyReport(
        configs_checked=configs_checked,
        findings=findings,
        strict_registry=strict_registry,
        passed=passed,
    )


def _check_one_model_config(path: Path) -> list[Finding]:
    """Compare one per-model inference config against its registry entry.

    Args:
        path: The ``*_inference_config.yaml`` file to check (read-only).

    Returns:
        Every finding for this file (collect-all); a single ERROR finding
        when the file cannot be parsed or lacks its task/model sections —
        never an exception.
    """
    import yaml

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Config {path.name} could not be read or parsed: {e}")
        return [
            Finding(
                severity=Severity.ERROR,
                reason_code=CONFIG_UNREADABLE,
                message=(
                    f"{path.name} could not be read or parsed as YAML, so it cannot be "
                    f"checked against the model registry ({e.__class__.__name__})."
                ),
            )
        ]
    if (
        not isinstance(doc, dict)
        or not isinstance(doc.get("task"), dict)
        or not isinstance(doc.get("model"), dict)
    ):
        return [
            Finding(
                severity=Severity.ERROR,
                reason_code=CONFIG_INVALID,
                message=(
                    f"{path.name} is missing its 'task' or 'model' section, so it cannot "
                    "be checked against the model registry."
                ),
            )
        ]

    task_doc: dict = doc["task"]
    raw_path = doc["model"].get("path")
    entry = try_resolve(raw_path) if isinstance(raw_path, str) else None
    if entry is None:
        return [
            Finding(
                severity=Severity.WARN,
                reason_code=MODEL_NOT_IN_REGISTRY,
                message=(
                    f"{path.name} points at repo '{raw_path}', which the packaged model "
                    "registry does not know — the config may work, but it is not catalogued."
                ),
            )
        ]
    return _compare_task_fields(path.name, entry, task_doc, doc)


def _compare_task_fields(
    file_name: str, entry: RegistryEntry, task_doc: dict, doc: dict
) -> list[Finding]:
    """Compare one resolved config's fields against its registry entry.

    Args:
        file_name: Name of the config file (for finding messages).
        entry: The resolved registry entry (source of truth).
        task_doc: The config's parsed ``task`` mapping.
        doc: The full parsed config document (for source/max_length lookups).

    Returns:
        One finding per mismatched field, in fixed field-walk order.
    """
    repo_id = entry.model
    task = entry.task
    findings: list[Finding] = []

    if task_doc.get("task_type") != task.task_type:
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=TASK_TYPE_MISMATCH,
                message=(
                    f"{file_name} says the model does '{task_doc.get('task_type')}' tasks, "
                    f"but the registry records '{task.task_type}' for {repo_id}."
                ),
            )
        )

    if task_doc.get("num_labels") != task.num_labels:
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=NUM_LABELS_MISMATCH,
                message=(
                    f"{file_name} says the model has {task_doc.get('num_labels')} output "
                    f"labels, but the registry records {task.num_labels} for {repo_id}."
                ),
            )
        )

    config_labels = task_doc.get("label_names")
    if (config_labels or []) != (task.label_names or []):
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=LABEL_ORDER_MISMATCH,
                message=(
                    f"{file_name} and the registry disagree on the labels for {repo_id}: "
                    f"{config_labels} in the config vs {task.label_names} in the registry. "
                    "Label order is load-bearing — a swapped order silently flips what "
                    "each prediction means."
                ),
            )
        )

    if task.threshold is not None and task_doc.get("threshold") != task.threshold:
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=THRESHOLD_MISMATCH,
                message=(
                    f"{file_name} uses a decision threshold of "
                    f"{task_doc.get('threshold')}, but the registry records {task.threshold} "
                    f"for {repo_id}."
                ),
            )
        )

    source = doc["model"].get("source")
    if is_lab_model(entry) and source != "modelscope":
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=SOURCE_MISMATCH,
                message=(
                    f"{file_name} fetches {repo_id} from '{source}'. This is a lab model "
                    "(zhangtaolab), normally fetched from ModelScope — downloads still "
                    "work via fallback, but ModelScope is the preferred source."
                ),
            )
        )

    effective_length = effective_max_length(entry)
    inference_doc = doc.get("inference")
    if (
        effective_length is not None
        and isinstance(inference_doc, dict)
        and "max_length" in inference_doc
        and inference_doc["max_length"] != effective_length
    ):
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=MAX_LENGTH_MISMATCH,
                message=(
                    f"{file_name} sets max_length to {inference_doc['max_length']}, but the "
                    f"registry's effective limit for {repo_id} is {effective_length} tokens. "
                    "Truncation length is a user policy, so this is informational."
                ),
            )
        )

    description = task_doc.get("description")
    if description is not None and (
        not isinstance(description, str) or not task.describe.startswith(description)
    ):
        findings.append(
            Finding(
                severity=Severity.INFO,
                reason_code=DESCRIPTION_DIVERGED,
                message=(
                    f"{file_name} describes the task differently from the registry entry "
                    f"for {repo_id} — wording only, no behavioral impact."
                ),
            )
        )

    return findings


def _check_one_server_config(path: Path, config_dir: Path) -> list[Finding]:
    """Check a server config's ``models[*].config_path`` references (D-03).

    Args:
        path: The ``mcp_server_config*.yaml`` file to check (read-only).
        config_dir: Directory the server config lives in — relative
            ``config_path`` entries resolve against it, exactly the way
            ``config_manager`` resolves them (absolute stays absolute).

    Returns:
        One DANGLING_CONFIG_REFERENCE ERROR finding per reference whose
        target file does not exist; a single CONFIG_UNREADABLE ERROR
        finding when the file cannot be parsed. ``multi_model`` groups are
        not walked — the MCPServerConfig validator owns those.
    """
    import yaml

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Server config {path.name} could not be read or parsed: {e}")
        return [
            Finding(
                severity=Severity.ERROR,
                reason_code=CONFIG_UNREADABLE,
                message=(
                    f"{path.name} could not be read or parsed as YAML, so its model "
                    f"references cannot be checked ({e.__class__.__name__})."
                ),
            )
        ]
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, dict):
        return []
    findings: list[Finding] = []
    for key, entry in models.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("config_path"), str):
            continue
        raw_reference = entry["config_path"]
        reference = Path(raw_reference)
        if not reference.is_absolute():
            reference = config_dir / reference
        if not reference.is_file():
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    reason_code=DANGLING_CONFIG_REFERENCE,
                    message=(
                        f"{path.name} entry '{key}' points at '{raw_reference}', but no "
                        "such config file exists in the config directory."
                    ),
                )
            )
    return findings
