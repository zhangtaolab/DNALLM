#!/usr/bin/env python3
"""Fail-closed transformers compatibility census over the pretrained model registry."""

import argparse
import json
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] - fixed argv, no shell
import sys
from datetime import UTC, datetime
from pathlib import Path

# Hard-set the mirror BEFORE any hub client import (Pitfall 6): the dispatch
# runner cannot reach huggingface.co, and the census must never call the
# default endpoint from either route. ModelScope is preferred first everywhere.
HF_MIRROR = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_MIRROR

DEFAULT_ARTIFACT = "tests/models/compat_census.json"

# Code-only fetch patterns: remote python, config, and tokenizer metadata.
# Weights are NEVER fetched here (the import census stays KB-scale per repo).
CODE_PATTERNS = ("*.py", "config.json", "tokenizer.json", "tokenizer_config.json", "*.model")

# Pre-disposition evidence vocabulary (asserted by record validation).
STATUSES = frozenset({"arch-ok", "import-ok", "import-fail", "census-blocked"})
CLASSES = frozenset({"native", "remote"})
HUBS = frozenset({"ms", "hf"})

REQUIRED_RECORD_KEYS = (
    "repo",
    "name",
    "class",
    "hub",
    "revision",
    "transformers",
    "status",
    "failing_import",
    "traceback_tail",
    "family",
    "is_family_rep",
    "load",
    "fetched_at",
)

# Cross-repo auto_map form (Pitfall 5): ``owner/sibling--file.Class``.
CROSS_REPO_SEPARATOR = "--"
UNREACHABLE_MARKER = "unreachable on both hubs"
PROBE_TIMEOUT_S = 120
TRACEBACK_TAIL_LINES = 15

# Probe subprocesses execute untrusted trust_remote_code fetched from
# third-party repos; they must never see runner credentials or CI state
# (WR-03). Everything under these prefixes is stripped from the child env;
# interpreter/venv, hub cache, and CUDA plumbing pass through untouched.
BLOCKED_PROBE_ENV_PREFIXES = ("GITHUB_", "RUNNER_", "ACTIONS_")

_SYMBOL_PATTERNS = (
    re.compile(r"cannot import name '([^']+)'"),
    # Require a module receiver: ``'NoneType' object has no attribute 'loader'``
    # is importlib plumbing, not a transformers symbol.
    re.compile(r"module '[^']+' has no attribute '([^']+)'"),
    re.compile(r"name '([^']+)' is not defined"),
    re.compile(r"No module named '([^']+)'"),
)


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 stamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────────────────
# Pure classification / parsing helpers (offline unit-tested)
# ─────────────────────────────────────────────────────────────────────────────


def classify_config(config: dict) -> str:
    """Return the entry's code class: ``remote`` only for a non-empty ``auto_map``.

    Args:
        config: The fetched ``config.json`` mapping.

    Returns:
        ``"remote"`` when ``auto_map`` is a mapping carrying at least one
        non-empty value, otherwise ``"native"``.
    """
    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict) and any(auto_map.values()):
        return "remote"
    return "native"


def is_cross_repo_value(value: str) -> bool:
    """Return True when an auto_map value references code in a sibling repo."""
    return isinstance(value, str) and CROSS_REPO_SEPARATOR in value


def parse_cross_repo_value(value: str) -> tuple[str, str, str] | None:
    """Split ``owner/sibling--file.Class`` into (sibling repo, file stem, class).

    Args:
        value: An ``auto_map`` value such as
            ``neuralbioinfo/nbrg-transformers--models.ProkBertConfig``.

    Returns:
        The ``(sibling_repo, file_stem, class_name)`` tuple, or None when the
        value is not a well-formed cross-repo reference.
    """
    if not is_cross_repo_value(value):
        return None
    sibling, _, target = value.partition(CROSS_REPO_SEPARATOR)
    if not sibling or "." not in target:
        return None
    file_stem, _, class_name = target.rpartition(".")
    if not file_stem or not class_name:
        return None
    return sibling, file_stem, class_name


def parse_local_value(value: str) -> tuple[str, str] | None:
    """Split a local ``file.Class`` auto_map value into (file stem, class name).

    Args:
        value: An ``auto_map`` value such as ``esm_config.EsmConfig``.

    Returns:
        The ``(file_stem, class_name)`` tuple, or None for cross-repo,
        module-dotted, or otherwise non-local values.
    """
    if not isinstance(value, str) or not value or is_cross_repo_value(value):
        return None
    file_stem, _, class_name = value.rpartition(".")
    if not file_stem or not class_name or "/" in file_stem:
        return None
    return file_stem, class_name


def extract_failing_import(traceback_text: str) -> str | None:
    """Return the last ``transformers`` import frame line in a traceback, or None.

    Args:
        traceback_text: The captured ``traceback.format_exc()`` text.

    Returns:
        The last stripped source line beginning a transformers import, or None
        when the traceback carries no such frame.
    """
    if not traceback_text:
        return None
    last: str | None = None
    for raw_line in traceback_text.splitlines():
        line = raw_line.strip()
        if line.startswith("from transformers") or line.startswith("import transformers"):
            last = line
    return last


def _import_symbol_detail(traceback_text: str) -> str | None:
    """Return the missing/undefined symbol named by the final exception line."""
    if not traceback_text:
        return None
    for line in reversed(traceback_text.strip().splitlines()):
        for pattern in _SYMBOL_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1)
    return None


def describe_failing_import(traceback_text: str) -> str | None:
    """Return the exact failing import, naming both its module and symbol.

    Args:
        traceback_text: The captured ``traceback.format_exc()`` text.

    Returns:
        The last transformers import frame line, with the missing symbol in
        brackets when the exception names one; None when the traceback has no
        import frame and no recognizable symbol.
    """
    frame = extract_failing_import(traceback_text)
    symbol = _import_symbol_detail(traceback_text)
    if frame and symbol:
        return f"{frame} [{symbol}]"
    if frame:
        return frame
    return symbol


def traceback_tail(traceback_text: str, keep: int = TRACEBACK_TAIL_LINES) -> str | None:
    """Return the last *keep* lines of a traceback, or None for empty input."""
    if not traceback_text:
        return None
    return "\n".join(traceback_text.strip().splitlines()[-keep:])


def transformers_version() -> str:
    """Return the installed transformers version (the census subject)."""
    import transformers

    return transformers.__version__


def validate_record(record: dict) -> list[str]:
    """Return the validation problems of an evidence record (empty list when valid).

    Args:
        record: One census evidence record.

    Returns:
        A list of human-readable problems. An empty list means the record is
        well-formed: every required key is present, the status sits in the
        four-value vocabulary, class/hub vocabulary holds (when set), and the
        provenance stamps are non-empty.
    """
    if not isinstance(record, dict):
        return [f"record must be a mapping, got {type(record).__name__}"]
    problems = [f"missing key: {key}" for key in REQUIRED_RECORD_KEYS if key not in record]
    status = record.get("status")
    if status not in STATUSES:
        problems.append(f"status {status!r} not in {sorted(STATUSES)}")
    code_class = record.get("class")
    if code_class is not None and code_class not in CLASSES:
        problems.append(f"class {code_class!r} not in {sorted(CLASSES)}")
    hub = record.get("hub")
    if hub not in HUBS:
        problems.append(f"hub {hub!r} not in {sorted(HUBS)}")
    revision = record.get("revision")
    if not isinstance(revision, str) or not revision:
        problems.append("revision must be a non-empty string")
    transformers = record.get("transformers")
    if not isinstance(transformers, str) or not transformers:
        problems.append("transformers must be a non-empty version stamp")
    fetched_at = record.get("fetched_at")
    if not isinstance(fetched_at, str) or not fetched_at:
        problems.append("fetched_at must be a non-empty ISO 8601 stamp")
    return problems


def write_artifact(path: str | Path, records: list[dict]) -> None:
    """Write the census evidence artifact, rejecting invalid records.

    Args:
        path: Artifact destination (JSON object with an ``entries`` list).
        records: Evidence records in registry file order.

    Raises:
        ValueError: When any record fails :func:`validate_record`.
    """
    problems: list[str] = []
    for index, record in enumerate(records):
        problems.extend(f"record {index}: {problem}" for problem in validate_record(record))
    if problems:
        raise ValueError("refusing to write invalid census evidence: " + "; ".join(problems))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": records}
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_artifact(path: str | Path) -> list[dict]:
    """Read a census evidence artifact, failing closed on absent/unparseable input.

    Args:
        path: Artifact produced by :func:`write_artifact`.

    Returns:
        The evidence records in their stored order.

    Raises:
        ValueError: When the file is absent, unparseable, or lacks an
            ``entries`` list — an interrupted or tampered artifact must never
            read as "no evidence".
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read census artifact {path}: {error}") from error
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"census artifact {path} has no 'entries' list")
    return entries


def family_key(repo: str, config: dict | None) -> str:
    """Return the deterministic family key for a registry entry.

    Rule (planner-fixed): the entry's ``architectures[0]`` from its fetched
    config; when ``architectures`` is absent or empty, the repo namespace
    prefix (text before the first ``/``).

    Args:
        repo: The entry's repo id.
        config: The fetched ``config.json`` mapping, or None when unreachable.

    Returns:
        The family key string.
    """
    if isinstance(config, dict):
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            first = architectures[0]
            if isinstance(first, str) and first:
                return first
    return repo.split("/", 1)[0]


def assign_families(records: list[dict]) -> None:
    """Mark exactly one lab-first representative per family, in registry order.

    Representative = the first ``zhangtaolab/`` entry in the family when one
    exists, otherwise the first entry in registry order. Single-entry families
    are represented by themselves. Mutates the records in place.

    Args:
        records: Evidence records in registry file order.
    """
    representatives: dict[str, str] = {}
    for record in records:
        family = record.get("family") or family_key(record["repo"], None)
        record["family"] = family
        if family not in representatives:
            representatives[family] = record["repo"]
        elif record["repo"].startswith("zhangtaolab/") and not representatives[family].startswith(
            "zhangtaolab/"
        ):
            representatives[family] = record["repo"]
    for record in records:
        record["is_family_rep"] = record["repo"] == representatives[record["family"]]


# ─────────────────────────────────────────────────────────────────────────────
# Roster, fetch, and provenance
# ─────────────────────────────────────────────────────────────────────────────


def pretrained_roster() -> list:
    """Return the 49 pretrained registry entries in file order.

    The typed registry loader is the single source of truth (REG-01): this tool
    never opens ``model_info.yaml`` itself.
    """
    from dnallm.models.registry import all_entries

    return [entry for entry in all_entries() if entry.kind == "pretrained"]


def _hub_revision(hub: str, repo: str, fetched_at: str) -> str:
    """Return the serving hub's revision stamp, never an empty value.

    Args:
        hub: ``"ms"`` or ``"hf"``.
        repo: The repo id.
        fetched_at: The record's fetch timestamp (used for the unpinned form).

    Returns:
        The hub's revision sha, or the literal ``unpinned@<timestamp>`` when the
        hub exposes no sha.
    """
    sha: str | None = None
    try:
        if hub == "ms":
            from modelscope.hub.api import HubApi

            sha = getattr(HubApi().model_info(repo), "sha", None)
        else:
            from huggingface_hub import HfApi

            sha = HfApi(endpoint=HF_MIRROR).model_info(repo).sha
    except Exception:  # provenance is best-effort; the stamp stays honest
        sha = None
    return sha or f"unpinned@{fetched_at}"


def fetch_code_only(repo: str) -> tuple[str, str, str] | None:
    """Fetch code-only artifacts for a repo: ModelScope first, then hf-mirror.

    Args:
        repo: The repo id to fetch.

    Returns:
        ``(snapshot_dir, hub, revision_stamp)``, or None when both routes fail
        (the caller records a loud census-blocked verdict — never a skip).
    """
    fetched_at = utc_now()
    ms_error: Exception | None = None
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

        path = ms_snapshot_download(model_id=repo, allow_file_pattern=list(CODE_PATTERNS))
        return str(path), "ms", _hub_revision("ms", repo, fetched_at)
    except Exception as error:  # route failure must fall through to the mirror
        ms_error = error
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        path = hf_snapshot_download(
            repo_id=repo,
            allow_patterns=list(CODE_PATTERNS),
            endpoint=HF_MIRROR,
        )
        return str(path), "hf", _hub_revision("hf", repo, fetched_at)
    except Exception as hf_error:  # both routes exhausted: loud verdict below
        print(f"UNREACHABLE {repo} (ModelScope: {ms_error!r}; hf-mirror: {hf_error!r})")
        return None


def read_config(snapshot_dir: str) -> dict:
    """Read a fetched ``config.json`` mapping.

    Args:
        snapshot_dir: The code-only snapshot directory.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: When the file is absent, unparseable, or not a JSON object.
    """
    path = Path(snapshot_dir) / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fetched config.json at {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"fetched config.json at {path} is not a JSON object")
    return data


def describe_config_problem(snapshot_dir: str, error: ValueError) -> str:
    """Return a self-contained reason when the fetched config.json is unusable.

    Names the snapshot's top-level layout, so a blocked verdict explains itself
    (for example a repo that stores checkpoints under size-named
    subdirectories and ships no root config.json).
    """
    snapshot = Path(snapshot_dir)
    if (snapshot / "config.json").is_file():
        return str(error)
    try:
        entries = sorted(item.name for item in snapshot.iterdir())[:10]
    except OSError:
        return str(error)
    return f"{error}; snapshot top-level entries: {entries}"


def unresolved_architecture(architectures: list) -> str | None:
    """Return the first architecture name not resolvable in installed transformers.

    Args:
        architectures: The config's ``architectures`` list.

    Returns:
        The first unresolved (dotted) class name, or None when every name
        resolves by attribute walk from the ``transformers`` module.
    """
    import transformers

    for name in architectures:
        target = transformers
        for part in str(name).split("."):
            if not hasattr(target, part):
                return str(name)
            target = getattr(target, part)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Isolated import probe (one subprocess per remote entry)
# ─────────────────────────────────────────────────────────────────────────────


def _auto_map_items(config: dict) -> list[tuple[str, str]]:
    """Return the config's non-empty auto_map items in config/tokenizer/modeling order."""
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return []
    items = [
        (key, value)
        for key, value in auto_map.items()
        if isinstance(key, str) and isinstance(value, str) and value
    ]

    def order_key(item: tuple[str, str]) -> tuple[int, str]:
        key = item[0]
        if key == "AutoConfig":
            return (0, key)
        if key == "AutoTokenizer":
            return (1, key)
        return (2, key)

    return sorted(items, key=order_key)


def _job_for(file_stem: str, base_dir: str) -> tuple[str, str] | None:
    """Return a (module name, absolute path) probe job for a fetched file, or None.

    The path is made absolute WITHOUT resolving symlinks: HF-hub snapshots are
    symlinks into an extensionless ``blobs/<sha>`` store, and resolving them
    would strip the ``.py`` suffix the import machinery keys its loader on.
    """
    path = Path(base_dir) / f"{file_stem}.py"
    if not path.is_file():
        return None
    return file_stem, str(path.absolute())


def _dedupe_jobs(jobs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop duplicate jobs (several auto_map keys may share one modeling file)."""
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for job in jobs:
        if job[1] in seen:
            continue
        seen.add(job[1])
        deduped.append(job)
    return deduped


def resolve_probe_jobs(config: dict, snapshot_dir: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Resolve local auto_map values into probe jobs plus referenced sibling repos.

    Args:
        config: The fetched ``config.json`` mapping.
        snapshot_dir: The entry's code-only snapshot directory.

    Returns:
        A ``(jobs, sibling_repos)`` tuple: jobs are (module name, absolute
        path) in config -> tokenizer -> modeling order; ``sibling_repos`` are
        the sibling repo ids referenced by ``--``-style cross-repo values.
    """
    jobs: list[tuple[str, str]] = []
    sibling_repos: list[str] = []
    for _key, value in _auto_map_items(config):
        cross = parse_cross_repo_value(value)
        if cross is not None:
            if cross[0] not in sibling_repos:
                sibling_repos.append(cross[0])
            continue
        local = parse_local_value(value)
        if local is None:
            continue
        job = _job_for(local[0], snapshot_dir)
        if job is not None:
            jobs.append(job)
    return _dedupe_jobs(jobs), sibling_repos


def sibling_probe_jobs(config: dict, sibling: str, sibling_dir: str) -> list[tuple[str, str]]:
    """Resolve one sibling repo's cross-repo auto_map values into probe jobs."""
    jobs: list[tuple[str, str]] = []
    for _key, value in _auto_map_items(config):
        cross = parse_cross_repo_value(value)
        if cross is None or cross[0] != sibling:
            continue
        job = _job_for(cross[1], sibling_dir)
        if job is not None:
            jobs.append(job)
    return _dedupe_jobs(jobs)


def build_probe_program(jobs: list[tuple[str, str]]) -> str:
    """Build the isolated child program that imports each fetched remote module.

    The ordering is the shim guarantee (Pitfall 3): ``import dnallm`` strictly
    precedes any remote module execution, so transformers_compat's patches are
    armed before 4.x-era remote code resolves its transformers imports. The
    child depends on no CWD: every job path is absolute.

    Each snapshot directory is executed as a synthetic package (mirroring the
    transformers dynamic-module machinery): remote files import their siblings
    both relatively (``from .esm_config import EsmConfig``) and absolutely
    (``from esm_config import EsmConfig``), so each directory is registered as
    a package and appended to ``sys.path`` before its files execute.

    Args:
        jobs: (module name, absolute path) pairs in config/tokenizer/modeling order.

    Returns:
        The ``python -c`` program source.
    """
    return "\n".join([
        "import os",
        f"os.environ['HF_ENDPOINT'] = {HF_MIRROR!r}",
        "import importlib.util",
        "import json",
        "import sys",
        "import traceback",
        "import types",
        "import dnallm  # arms transformers_compat.apply_patches() before any remote import",
        f"jobs = {jobs!r}",
        "try:",
        "    directories = []",
        "    for _module_name, _path in jobs:",
        "        directory = os.path.dirname(_path)",
        "        if directory not in directories:",
        "            directories.append(directory)",
        "    for index, directory in enumerate(directories):",
        "        package = '_census_remote_%d' % index",
        "        package_module = types.ModuleType(package)",
        "        package_module.__path__ = [directory]",
        "        sys.modules[package] = package_module",
        "        sys.path.append(directory)",
        "    for module_name, path in jobs:",
        "        directory = os.path.dirname(path)",
        "        package = '_census_remote_%d' % directories.index(directory)",
        "        qualified = package + '.' + module_name",
        "        spec = importlib.util.spec_from_file_location(qualified, path)",
        "        if spec is None or spec.loader is None:",
        "            raise RuntimeError('cannot build an import spec for %s' % path)",
        "        module = importlib.util.module_from_spec(spec)",
        "        sys.modules[qualified] = module",
        "        spec.loader.exec_module(module)",
        "    print(json.dumps({'ok': True}))",
        "except Exception:",
        "    print(json.dumps({'ok': False, 'traceback': traceback.format_exc()}))",
    ])


def sanitized_probe_env() -> dict[str, str]:
    """Return the probe subprocess env with runner credentials stripped (WR-03).

    The probe children execute arbitrary ``trust_remote_code`` fetched from
    third-party repos; on the dispatch lane the parent is a GitHub runner
    process whose environment carries ``GITHUB_TOKEN`` (scoped
    ``contents: read``), ``RUNNER_*`` paths, and other CI state. Everything
    under :data:`BLOCKED_PROBE_ENV_PREFIXES` is removed; the children keep
    their interpreter/venv, hub cache, and CUDA plumbing untouched.

    Returns:
        A fresh mapping safe to pass as ``subprocess.run(env=...)``.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(BLOCKED_PROBE_ENV_PREFIXES)
    }


def run_import_probe(jobs: list[tuple[str, str]], timeout: int = PROBE_TIMEOUT_S) -> dict:
    """Run one isolated import probe and return its JSON verdict.

    Args:
        jobs: (module name, absolute path) pairs.
        timeout: Subprocess timeout in seconds.

    Returns:
        The child's verdict mapping: ``{"ok": True}`` or
        ``{"ok": False, "traceback": ...}``. A crash, timeout, or missing
        verdict becomes an ``ok: False`` payload with an explanatory traceback.
    """
    program = build_probe_program(jobs)
    try:
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv, no shell
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=sanitized_probe_env(),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "traceback": f"import probe timed out after {timeout}s"}
    for line in reversed(completed.stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    tail = "\n".join(
        (completed.stderr or completed.stdout).strip().splitlines()[-TRACEBACK_TAIL_LINES:]
    )
    return {
        "ok": False,
        "traceback": f"import probe produced no JSON verdict (exit {completed.returncode})\n{tail}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-entry census
# ─────────────────────────────────────────────────────────────────────────────


def build_record(
    entry,
    hub: str,
    revision: str,
    code_class: str | None,
    status: str,
    *,
    fetched_at: str,
    family: str | None = None,
    failing_import: str | None = None,
    traceback_text: str | None = None,
) -> dict:
    """Build one evidence record with the exact committed key set."""
    return {
        "repo": entry.model,
        "name": entry.name,
        "class": code_class,
        "hub": hub,
        "revision": revision,
        "transformers": transformers_version(),
        "status": status,
        "failing_import": failing_import,
        "traceback_tail": traceback_tail(traceback_text),
        "family": family,
        "is_family_rep": False,
        "load": None,
        "fetched_at": fetched_at,
    }


def _native_record(entry, hub, revision, config, family, fetched_at) -> dict:
    """Build the evidence record for a native (no remote code) entry."""
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        return build_record(
            entry,
            hub,
            revision,
            "native",
            "census-blocked",
            fetched_at=fetched_at,
            family=family,
            failing_import="config.json carries no architectures list",
        )
    unresolved = unresolved_architecture(architectures)
    if unresolved is not None:
        return build_record(
            entry,
            hub,
            revision,
            "native",
            "import-fail",
            fetched_at=fetched_at,
            family=family,
            failing_import=f"{unresolved} not found in transformers {transformers_version()}",
        )
    return build_record(
        entry, hub, revision, "native", "arch-ok", fetched_at=fetched_at, family=family
    )


def _remote_record(entry, hub, revision, config, family, snapshot_dir, fetched_at) -> dict:
    """Build the evidence record for a remote-code entry (subprocess import probe)."""
    jobs, sibling_repos = resolve_probe_jobs(config, snapshot_dir)
    for sibling in sibling_repos:
        sibling_fetched = fetch_code_only(sibling)
        if sibling_fetched is None:
            return build_record(
                entry,
                hub,
                revision,
                "remote",
                "census-blocked",
                fetched_at=fetched_at,
                family=family,
                failing_import=f"cross-repo auto_map sibling {sibling} {UNREACHABLE_MARKER}",
            )
        jobs.extend(sibling_probe_jobs(config, sibling, sibling_fetched[0]))
    jobs = _dedupe_jobs(jobs)
    if not jobs:
        return build_record(
            entry,
            hub,
            revision,
            "remote",
            "census-blocked",
            fetched_at=fetched_at,
            family=family,
            failing_import="auto_map values do not resolve to fetched .py files",
        )
    payload = run_import_probe(jobs)
    if payload.get("ok") is True:
        return build_record(
            entry, hub, revision, "remote", "import-ok", fetched_at=fetched_at, family=family
        )
    traceback_text = payload.get("traceback")
    if not isinstance(traceback_text, str):
        traceback_text = ""
    return build_record(
        entry,
        hub,
        revision,
        "remote",
        "import-fail",
        fetched_at=fetched_at,
        family=family,
        failing_import=describe_failing_import(traceback_text)
        or "import failed (see traceback_tail)",
        traceback_text=traceback_text,
    )


def census_entry(entry) -> dict:
    """Census one registry entry: fetch, classify, probe, and record evidence.

    Args:
        entry: A pretrained :class:`dnallm.models.registry.RegistryEntry`.

    Returns:
        The entry's validated-shaped evidence record. Unreachable repos produce
        a loud ``census-blocked`` verdict, never a silent skip.
    """
    fetched_at = utc_now()
    fetched = fetch_code_only(entry.model)
    if fetched is None:
        return build_record(
            entry,
            "ms",
            f"unpinned@{fetched_at}",
            None,
            "census-blocked",
            fetched_at=fetched_at,
            failing_import=f"{UNREACHABLE_MARKER} (ModelScope then {HF_MIRROR})",
        )
    snapshot_dir, hub, revision = fetched
    try:
        config = read_config(snapshot_dir)
    except ValueError as error:
        return build_record(
            entry,
            hub,
            revision,
            None,
            "census-blocked",
            fetched_at=fetched_at,
            failing_import=describe_config_problem(snapshot_dir, error),
        )
    code_class = classify_config(config)
    family = family_key(entry.model, config)
    if code_class == "native":
        return _native_record(entry, hub, revision, config, family, fetched_at)
    return _remote_record(entry, hub, revision, config, family, snapshot_dir, fetched_at)


# ─────────────────────────────────────────────────────────────────────────────
# Family-representative load sampling (D-02b / D-03)
# ─────────────────────────────────────────────────────────────────────────────

LOAD_TIMEOUT_S = 1800
LOAD_SEED = 27
LOAD_SEQUENCE_LENGTH = 32

# Task-appropriate Auto* routing (27-02 fix): a remote checkpoint's auto_map
# keys its task-specific classes, so the load layer must dispatch through the
# matching Auto* class. Loading a remote checkpoint through the generic
# AutoModel route when its auto_map carries only task keys resolves the
# NATIVE architecture instead (measured live: zhangtaolab/
# plant-nucleotide-transformer-6mer and kuleshov-group/PlantCAD2-Small
# misroutes of the 27-01 load layer).
LOAD_CLASS_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ForCausalLM", "AutoModelForCausalLM"),
    ("ForMaskedLM", "AutoModelForMaskedLM"),
    ("ForSequenceClassification", "AutoModelForSequenceClassification"),
    ("ForTokenClassification", "AutoModelForTokenClassification"),
)


def resolve_load_class_name(architecture: str | None) -> str:
    """Return the task-appropriate Auto* class name for an architecture.

    Mirrors dnallm's own task-type dispatch. The generic ``AutoModel``
    remains the fallback for bare backbones whose auto_map maps AutoModel
    (e.g. gena-lm's ``BertForPreTraining`` and the bare ``Caduceus``
    backbone, which deliberately route through their auto_map AutoModel).

    Args:
        architecture: The checkpoint config's ``architectures[0]`` (or None).

    Returns:
        The transformers Auto* class name to load through.
    """
    name = architecture or ""
    for suffix, auto_name in LOAD_CLASS_SUFFIXES:
        if name.endswith(suffix):
            return auto_name
    return "AutoModel"


def build_load_program(repo: str, hub: str, architectures: list, revision: str) -> str:
    """Build the isolated child program for one load + short real forward.

    The pass bar is D-03: construction and weight loading alone do not count —
    one short real forward must run. Any load or forward failure becomes an
    evidence payload, never a child crash.

    Args:
        repo: The repo id.
        hub: The census-stamped serving hub (``ms`` or ``hf``).
        architectures: The fetched config's ``architectures`` list.
        revision: The stamped revision (a sha pins the HF fetch; ModelScope
            accepts branch/tag names only, so it fetches the default revision
            and the census stamp stays the provenance).

    Returns:
        The ``python -c`` program source.
    """
    pinned = revision if revision and not str(revision).startswith("unpinned") else None
    auto_class_name = resolve_load_class_name(architectures[0] if architectures else None)
    return "\n".join([
        "import os",
        f"os.environ['HF_ENDPOINT'] = {HF_MIRROR!r}",
        "import json",
        "import time",
        "import traceback",
        "import dnallm  # arms transformers_compat.apply_patches() before any remote import",
        "import torch",
        "import transformers",
        "from transformers import AutoConfig",
        f"repo = {repo!r}",
        f"hub = {hub!r}",
        f"architectures = {architectures!r}",
        f"revision = {pinned!r}",
        f"model_class_name = {auto_class_name!r}",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        # No successful forward has run yet; a load failure therefore reports
        # forward_ok False (the record's ok/error fields name the phase).
        "result = {'ok': False, 'error': None, 'forward_ok': False, "
        "'output_shape': None, 'device': device, 'duration_s': 0.0}",
        "start = time.time()",
        "try:",
        "    if hub == 'ms':",
        "        from modelscope.hub.snapshot_download import snapshot_download",
        "        # ModelScope accepts branch/tag names only; the census sha is",
        "        # provenance, not a fetchable revision (commit shas raise",
        "        # NotExistError live). The census stamp remains on the record.",
        "        source = snapshot_download(model_id=repo)",
        "    else:",
        "        source = repo",
        "    config = AutoConfig.from_pretrained(source, trust_remote_code=True)",
        "    model_class = getattr(transformers, model_class_name)",
        # torch_dtype (not dtype): 4.x-era remote code defines its own
        # from_pretrained and forwards unknown kwargs to __init__, where
        # ``dtype`` breaks (verified live on prokbert-mini). transformers 5
        # still honors torch_dtype, warning once.
        "    load_kwargs = {'trust_remote_code': True, 'torch_dtype': torch.float32}",
        "    if hub != 'ms' and revision:",
        "        load_kwargs['revision'] = revision",
        "    try:",
        "        model = model_class.from_pretrained(source, **load_kwargs)",
        "    except TypeError as kwarg_error:",
        "        # Some 4.x-era remote classes define their own from_pretrained and",
        "        # forward unknown kwargs to __init__ (prokbert-mini). A real user",
        "        # load without the dtype kwarg works; retry kwarg-free and cast.",
        "        if 'torch_dtype' not in str(kwarg_error):",
        "            raise",
        "        fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != 'torch_dtype'}",
        "        model = model_class.from_pretrained(source, **fallback_kwargs)",
        "        model = model.to(torch.float32)",
        "    model.to(device)",
        "    model.eval()",
        "    result['ok'] = True",
        "    try:",
        "        vocab_size = int(getattr(config, 'vocab_size', 0) "
        "or getattr(model.config, 'vocab_size', 0))",
        f"        generator = torch.Generator().manual_seed({LOAD_SEED})",
        f"        input_ids = torch.randint(1, vocab_size, (1, {LOAD_SEQUENCE_LENGTH}), "
        "generator=generator).to(device)",
        "        with torch.no_grad():",
        "            output = model(input_ids)",
        "        if isinstance(output, (tuple, list)):",
        "            tensor = output[0]",
        "        elif hasattr(output, 'logits'):",
        "            tensor = output.logits",
        "        elif hasattr(output, 'last_hidden_state'):",
        "            tensor = output.last_hidden_state",
        "        else:",
        "            tensor = output",
        "        result['output_shape'] = str(tuple(tensor.shape))",
        "        result['forward_ok'] = True",
        "    except Exception:",
        "        result['forward_ok'] = False",
        "        result['error'] = traceback.format_exc()[-2000:]",
        "except Exception:",
        "    result['error'] = traceback.format_exc()[-2000:]",
        "result['duration_s'] = round(time.time() - start, 3)",
        "print(json.dumps(result))",
    ])


def _current_device() -> str:
    """Return the box's census device label (``cuda`` when available, else ``cpu``)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of this environment
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _empty_load(error: str) -> dict:
    """Return a failed load record carrying the given reason."""
    return {
        "ok": False,
        "error": error,
        "forward_ok": False,
        "output_shape": None,
        "device": _current_device(),
        "duration_s": 0.0,
    }


def run_load_probe(record: dict) -> dict:
    """Load one family representative and run one short real forward.

    Args:
        record: The representative's evidence record (repo/hub/revision).

    Returns:
        The D-02b load record (``ok``, ``error``, ``forward_ok``,
        ``output_shape``, ``device``, ``duration_s``). Load and forward
        failures are evidence, never a crash of the census run.
    """
    architectures: list = []
    fetched = fetch_code_only(record["repo"])
    if fetched is not None:
        try:
            architectures = read_config(fetched[0]).get("architectures") or []
        except ValueError:
            architectures = []
    program = build_load_program(record["repo"], record["hub"], architectures, record["revision"])
    try:
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv, no shell
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=LOAD_TIMEOUT_S,
            check=False,
            env=sanitized_probe_env(),
        )
    except subprocess.TimeoutExpired:
        return _empty_load(f"load probe timed out after {LOAD_TIMEOUT_S}s")
    for line in reversed(completed.stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            forward_ok = payload.get("forward_ok")
            return {
                "ok": bool(payload.get("ok")),
                "error": payload.get("error"),
                "forward_ok": forward_ok if isinstance(forward_ok, bool) else False,
                "output_shape": payload.get("output_shape"),
                "device": payload.get("device") or _current_device(),
                "duration_s": float(payload.get("duration_s") or 0.0),
            }
    tail = "\n".join(
        (completed.stderr or completed.stdout).strip().splitlines()[-TRACEBACK_TAIL_LINES:]
    )
    return _empty_load(f"load probe produced no JSON verdict (exit {completed.returncode})\n{tail}")


def _previous_load_passed(previous: dict | None) -> bool:
    """Return True when a prior record's load already cleared the D-03 bar."""
    if not isinstance(previous, dict):
        return False
    load = previous.get("load")
    return isinstance(load, dict) and load.get("ok") is True and load.get("forward_ok") is True


def apply_loads(records: list[dict], previous: dict[str, dict] | None = None) -> None:
    """Add the D-02b load layer: load + short forward per family representative.

    Non-representatives carry ``load: null`` plus an honest ``load_note``
    naming their representative; ``import-fail`` entries stay parked
    ("pending shim" on a first run, "final after shim pass" once prior
    evidence exists); ``census-blocked`` entries carry no load evidence.

    With a *previous* artifact (the 27-02 shim pass): an entry whose previous
    status was ``import-fail`` but whose fresh status imports clean receives
    its D-02a load in this run even when it is not a family representative;
    a representative whose previous load already passed keeps that evidence
    (the merge carries it forward) instead of being re-fetched; a
    representative whose previous load or forward failed is re-probed.

    Args:
        records: Evidence records in registry file order (after
            :func:`assign_families`).
        previous: Optional ``{repo: prior record}`` map from the existing
            artifact.
    """
    representatives = {r["family"]: r["repo"] for r in records if r["is_family_rep"]}
    for record in records:
        rep = representatives.get(record["family"])
        suffix = (
            ""
            if record["is_family_rep"] or rep is None
            else f" (inferred from family representative {rep})"
        )
        prior = (previous or {}).get(record["repo"])
        if record["status"] == "import-fail":
            record["load"] = None
            note = (
                "import-fail final after shim pass"
                if previous is not None
                else ("import-fail pending shim")
            )
            record["load_note"] = note + suffix
            continue
        if record["status"] == "census-blocked":
            record["load"] = None
            record["load_note"] = "census-blocked: no load evidence (see failing_import)" + suffix
            continue
        # D-02a: an import recovered by a shim rung must carry its own
        # load+forward evidence in the run that recovered it.
        recovered_import = (
            previous is not None
            and isinstance(prior, dict)
            and prior.get("status") == "import-fail"
            and record["status"] == "import-ok"
        )
        if not recovered_import and not record["is_family_rep"]:
            record["load"] = None
            record["load_note"] = f"inferred from family representative {rep}"
            continue
        if (
            not recovered_import
            and previous is not None
            and isinstance(prior, dict)
            and prior.get("status") == record["status"]
            and _previous_load_passed(prior)
        ):
            # Prior evidence already cleared the D-03 bar under the same
            # verdict; the merge carries it forward untouched.
            record["load"] = None
            continue
        print(f"loading {record['repo']} [{record['family']}] ...", flush=True)
        record["load"] = run_load_probe(record)
        load = record["load"]
        print(
            f"  ok={load['ok']} forward_ok={load['forward_ok']} "
            f"shape={load['output_shape']} device={load['device']} {load['duration_s']}s",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run orchestration
# ─────────────────────────────────────────────────────────────────────────────


def merge_records(existing: list[dict], fresh: list[dict], roster_order: list[str]) -> list[dict]:
    """Merge fresh records into an existing artifact, preserving registry order.

    A fresh record replaces its repo's existing record. A verdict change
    stamps ``previous_status`` on the fresh record so the pre-fix evidence
    stays addressable (the 27-02 shim-pass artifact contract). Previously
    measured load evidence is carried over only when the status is unchanged
    and the fresh run measured no loads (an import-only re-run must not erase
    the D-02b layer, and a representative skipped because its prior load
    passed keeps that evidence). An *unchanged* verdict also carries the
    hand-authored ``disposition`` (the frozen 27-02 skip/fix judgments) and
    any prior ``previous_status`` stamp forward — the judgment still
    describes the same verdict, so a re-run against the default artifact
    path must not destroy it; a verdict *change* drops the stale disposition
    (the judgment no longer describes the record) and stamps the immediate
    prior status instead.

    Args:
        existing: Records read from the existing artifact.
        fresh: Records censused this run.
        roster_order: Pretrained repo ids in registry file order.

    Returns:
        The merged records in registry order.
    """
    by_repo: dict[str, dict] = {
        record["repo"]: record
        for record in existing
        if isinstance(record, dict) and isinstance(record.get("repo"), str)
    }
    for record in fresh:
        previous = by_repo.get(record["repo"])
        if (
            isinstance(previous, dict)
            and previous.get("status") != record.get("status")
            and record.get("status") not in (None,)
        ):
            record["previous_status"] = previous["status"]
        if isinstance(previous, dict) and previous.get("status") == record.get("status"):
            # Unchanged verdict: the disposition and its evidence trail
            # still describe this record. Carried independently of the load
            # carry below — a re-probed failed load replaces the load
            # evidence while the skip/fix judgment itself still holds.
            if "disposition" not in record and "disposition" in previous:
                record["disposition"] = previous["disposition"]
            if "previous_status" not in record and "previous_status" in previous:
                record["previous_status"] = previous["previous_status"]
            if record.get("load") is None and previous.get("load") is not None:
                record["load"] = previous["load"]
                if "load_note" in previous:
                    record["load_note"] = previous["load_note"]
        by_repo[record["repo"]] = record
    ordered: list[dict] = []
    seen: set[str] = set()
    for repo in roster_order:
        if repo in by_repo:
            ordered.append(by_repo[repo])
            seen.add(repo)
    ordered.extend(record for repo, record in by_repo.items() if repo not in seen)
    return ordered


def run_census(artifact_path: Path, selected: list[str] | None, with_loads: bool = False) -> int:
    """Census the (optionally filtered) pretrained roster and write the artifact.

    Args:
        artifact_path: Destination artifact path.
        selected: Repo ids to census, or None for the full roster.
        with_loads: Also run the D-02b load+forward layer for family
            representatives (the only path allowed to fetch weights).

    Returns:
        0 when the census ran to completion. Entries unreachable on both
        hubs are recorded census evidence, not a run failure: each stays a
        ``census-blocked`` record carrying the unreachable marker and is
        reported loudly, and the verdict gate (``--verify`` against the
        committed registry, or the ``--write-registry`` mapping) decides
        whether a census-blocked verdict is committed-known or drift. 1 for
        artifact read/write errors, 2 for a usage error.
    """
    roster = pretrained_roster()
    roster_order = [entry.model for entry in roster]
    if selected is not None:
        known = set(roster_order)
        unknown = [repo for repo in selected if repo not in known]
        if unknown:
            print(f"ERROR: --entries ids outside the pretrained roster: {unknown}")
            return 2
        wanted = set(selected)
        roster = [entry for entry in roster if entry.model in wanted]

    records: list[dict] = []
    unreachable: list[str] = []
    for entry in roster:
        record = census_entry(entry)
        records.append(record)
        print(f"{record['status']:<14} {record['hub']:<2} {record['repo']}")
        if record["status"] == "census-blocked" and UNREACHABLE_MARKER in (
            record["failing_import"] or ""
        ):
            unreachable.append(record["repo"])

    assign_families(records)
    existing_records: list[dict] | None = None
    if artifact_path.exists():
        try:
            existing_records = read_artifact(artifact_path)
        except ValueError as error:
            print(f"ERROR: {error}")
            return 1
    if with_loads:
        previous = (
            {r["repo"]: r for r in existing_records if isinstance(r, dict)}
            if existing_records is not None
            else None
        )
        apply_loads(records, previous=previous)

    merged = records
    if existing_records is not None:
        merged = merge_records(existing_records, records, roster_order)
    try:
        write_artifact(artifact_path, merged)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Census complete: {len(records)} entry/entries censused -> {artifact_path}")
    if unreachable:
        # Recorded evidence, not a run failure (CR-01): the unreachable
        # entries stay census-blocked records above, and the verdict gate
        # decides their significance — a committed-known unreachable entry
        # matches its committed census-blocked field, a newly unreachable
        # one disagrees with its committed verdict. Reported loudly either
        # way, never a silent pass: the exit code must not conflate
        # committed-known unreachability with infrastructure failure.
        print(f"CENSUS BLOCKED ({len(unreachable)} unreachable on both hubs): {unreachable}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 27-03 registry write-back: artifact -> tf_compat_* fields (anchored insertion)
# ─────────────────────────────────────────────────────────────────────────────

# The five-value registry verdict vocabulary (mirrors RegistryEntry's Literal).
TF_COMPAT_STATUSES = frozenset({
    "verified-tf5",
    "import-verified-tf5",
    "shim-required-tf5",
    "remote-code-broken",
    "census-blocked",
})

# Entries whose census-green load+forward passes ONLY through a compat rung
# (27-02 evidence). The artifact records the green load but does not stamp
# which rung that load needed, so the dependency lives here: the bare Caduceus
# backbone's remote __init__ skips post_init entirely, and rung 13
# (_patch_tied_weights_bookkeeping_on_load) materializes the tied-weights
# bookkeeping at the from_pretrained boundary (verified live in 27-02 and
# pinned by tests/utils/test_transformers_compat_census.py). Entries whose
# import a rung restored but whose load/forward still fails carry skip
# dispositions in the artifact and map to remote-code-broken — the D-03
# verdict (load + one real forward) is what the registry consumes.
SHIM_RESTORED_LOADS: dict[str, str] = {
    "kuleshov-group/PlantCaduceus_l32": "_patch_tied_weights_bookkeeping_on_load",
}

# First line of the write-back provenance block; the block is appended at end
# of the registry yaml exactly once (idempotency guard for repeat runs).
COMPAT_PROVENANCE_SENTINEL = (
    "# Registry compatibility census: tf_compat_status / tf_compat_evidence /"
    " tf_compat_tested (Phase 27-03, 2026-10-27)."
)


def compat_verdict(record: dict, family_reps: dict[str, str]) -> tuple[str, str]:
    """Map one census evidence record to its registry (status, evidence) pair.

    Objective mapping table (D-07): ``census-blocked`` stays ``census-blocked``;
    a ``skip`` disposition (import-fail or structurally failed load/forward)
    maps to ``remote-code-broken`` carrying the exact failing import/reason; a
    ``fix`` disposition or a shim-restored load with a green D-03 verdict maps
    to ``shim-required-tf5`` naming the rung; any other green load+forward maps
    to ``verified-tf5``; a non-representative import/arch-ok entry maps to
    ``import-verified-tf5`` naming its family representative.

    Args:
        record: One census evidence record.
        family_reps: ``{family: representative repo}`` map over the artifact.

    Returns:
        The ``(tf_compat_status, tf_compat_evidence)`` pair.

    Raises:
        ValueError: When the record carries neither a green load nor a skip
            disposition (an unclassifiable shape must never silently pass).
    """
    repo = record["repo"]
    stamp = f"{record['hub']} {repo}@{record['revision']}"
    status = record.get("status")
    disposition = record.get("disposition")
    disposition = disposition if isinstance(disposition, dict) else {}
    load = record.get("load")
    load = load if isinstance(load, dict) else {}
    load_passed = load.get("ok") is True and load.get("forward_ok") is True
    if status == "census-blocked":
        reason = record.get("failing_import") or "census blocked (see artifact)"
        return "census-blocked", f"{stamp}; {reason}"
    skip_reason = disposition.get("skip")
    if isinstance(skip_reason, str) and skip_reason:
        failing_import = record.get("failing_import")
        detail = (
            f"failing import: {failing_import}; {skip_reason}" if failing_import else skip_reason
        )
        return "remote-code-broken", f"{stamp}; {detail}"
    fix_rung = disposition.get("fix")
    if isinstance(fix_rung, str) and fix_rung and load_passed:
        return "shim-required-tf5", f"{stamp}; load+forward verified via rung {fix_rung}"
    if load_passed:
        rung = SHIM_RESTORED_LOADS.get(repo)
        if rung:
            return "shim-required-tf5", f"{stamp}; load+forward verified via rung {rung}"
        return "verified-tf5", stamp
    if not record.get("is_family_rep"):
        representative = family_reps.get(record.get("family"))
        if representative:
            return (
                "import-verified-tf5",
                f"{stamp}; load inferred from family representative {representative}",
            )
    raise ValueError(
        f"census record for {repo} (status {status!r}) carries no green load and "
        "no skip disposition - cannot map it to a registry status"
    )


def _compat_field_lines(status: str, evidence: str, tested: str) -> list[str]:
    """Return the three indented tf_compat_* lines to insert after max_length.

    json.dumps yields a YAML-double-quoted scalar (escaping is shared), so
    evidence text with quotes, brackets, or colons round-trips safely.
    """
    return [
        f"    tf_compat_status: {status}\n",
        f"    tf_compat_evidence: {json.dumps(evidence)}\n",
        f"    tf_compat_tested: {json.dumps(tested)}\n",
    ]


def _ends_entry_block(line: str) -> bool:
    """Return True when a stripped line ends the current registry entry block."""
    return (
        line.startswith("  - ")
        or line.startswith("#")
        or not line.strip()
        or (line and not line.startswith(" ") and line.endswith(":"))
    )


def plan_compat_insertions(records: list[dict], yaml_text: str, roster: list[str]) -> dict:
    """Plan the anchored tf_compat_* insertions for the pretrained roster.

    Each roster entry is located by its exact ``    model: "<repo>"`` line
    (repo ids are unique across sections, so base_model lines never match);
    the three field lines are planned immediately after that block's
    ``max_length`` line. An entry already carrying a ``tf_compat_status`` line
    is reported as already written, never duplicated.

    Args:
        records: Census evidence records (the committed artifact).
        yaml_text: The registry yaml file's exact text.
        roster: Pretrained repo ids in registry file order.

    Returns:
        ``{"plans": [...], "already": [...]}`` where every plan carries
        ``repo``, ``name``, ``status``, ``evidence``, ``tested``, ``anchor``
        (0-based index of the block's max_length line) and ``lines``.

    Raises:
        ValueError: Fail-closed on roster/artifact disagreement, an unlocatable
            entry block, a missing max_length anchor, or an unmappable verdict.
    """
    by_repo = {r.get("repo"): r for r in records if isinstance(r, dict)}
    missing = [repo for repo in roster if repo not in by_repo]
    if missing:
        raise ValueError(f"artifact is missing census records for: {missing}")
    extra = sorted(repo for repo in by_repo if repo not in set(roster))
    if extra:
        raise ValueError(f"artifact carries records outside the pretrained roster: {extra}")
    family_reps = {
        record["family"]: record["repo"]
        for record in records
        if record.get("is_family_rep") and record.get("family")
    }
    stripped = [line.rstrip("\n") for line in yaml_text.splitlines(keepends=True)]
    plans: list[dict] = []
    already: list[dict] = []
    for repo in roster:
        record = by_repo[repo]
        model_line = f'    model: "{repo}"'
        hits = [i for i, text in enumerate(stripped) if text == model_line]
        if len(hits) != 1:
            raise ValueError(
                f"expected exactly one registry line {model_line!r}, found {len(hits)}"
            )
        start = hits[0]
        end = start + 1
        while end < len(stripped) and not _ends_entry_block(stripped[end]):
            end += 1
        block = stripped[start:end]
        if any("tf_compat_status:" in line for line in block):
            already.append({"repo": repo, "name": record.get("name")})
            continue
        anchor_offsets = [i for i, line in enumerate(block) if line.startswith("    max_length:")]
        if not anchor_offsets:
            raise ValueError(f"entry block for {repo} carries no max_length anchor line")
        status, evidence = compat_verdict(record, family_reps)
        plans.append({
            "repo": repo,
            "name": record.get("name"),
            "status": status,
            "evidence": evidence,
            "tested": record.get("transformers"),
            "anchor": start + anchor_offsets[0],
            "lines": _compat_field_lines(status, evidence, record.get("transformers")),
        })
    return {"plans": plans, "already": already}


def compat_provenance_block() -> str:
    """Return the dated write-back provenance comment block (mirrors 22-02)."""
    return (
        f"{COMPAT_PROVENANCE_SENTINEL}\n"
        "# Method: import census plus sampled load/forward evidence under transformers\n"
        "#   5.17.0 over all 49 pretrained entries; frozen evidence artifact:\n"
        "#   tests/models/compat_census.json (27-02, dispositions complete).\n"
        "# Coverage rule: all 49 pretrained entries carry the fields; finetuned entries\n"
        "#   inherit from their bases and carry no tf_compat_* keys.\n"
        "# tf_compat_tested records what was ACTUALLY tested: transformers 5.17.0 only.\n"
        "#   The transformers 4 side is inferred (model code written in the transformers 4\n"
        "#   era), never separately tested (D-05); inference wording is a docs-layer\n"
        "#   concern (27-04).\n"
        "# Tooling: scripts/census_compat.py --write-registry --apply (committed,\n"
        "#   re-runnable; anchored text insertion only - never a YAML round-trip).\n"
    )


def write_registry(
    artifact_path: str | Path,
    yaml_path: str | Path,
    *,
    apply: bool = False,
    roster: list[str] | None = None,
) -> int:
    """Write the census verdicts into the registry yaml (anchored insertion only).

    Dry-run by default: without ``apply`` the planned lines are printed and
    nothing is written. Fail-closed: an absent/unparseable artifact, a
    roster/artifact disagreement, or an unmappable verdict exits 1 without
    touching the file — never a partial silent write.

    Args:
        artifact_path: The frozen census evidence artifact.
        yaml_path: The registry yaml to write (the packaged file by default).
        apply: Write the planned insertions plus the provenance block.
        roster: Pretrained repo ids in registry order (defaults to the typed
            registry loader — the sole parser of the packaged file).

    Returns:
        0 on success, 1 on a fail-closed condition.
    """
    try:
        records = read_artifact(artifact_path)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    if roster is None:
        roster = [entry.model for entry in pretrained_roster()]
    yaml_path = Path(yaml_path)
    try:
        yaml_text = yaml_path.read_text(encoding="utf-8")
    except OSError as error:
        print(f"ERROR: cannot read registry yaml {yaml_path}: {error}")
        return 1
    try:
        planned = plan_compat_insertions(records, yaml_text, roster)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    plans = planned["plans"]
    for plan in planned["already"]:
        print(f"already written: {plan['repo']} (skipped, idempotent)")
    if not apply:
        for plan in plans:
            print(f"planned {plan['repo']} ({plan['name']}) -> {plan['status']}")
            for line in plan["lines"]:
                print(f"  + {line.rstrip()}")
        print(
            f"{len(plans)} planned insertion(s), {len(planned['already'])} already "
            "written (dry-run; pass --apply to write)"
        )
        return 0
    if not plans:
        print("nothing to write: every roster entry already carries its tf_compat_* fields")
        return 0
    lines = yaml_text.splitlines(keepends=True)
    for plan in sorted(plans, key=lambda item: item["anchor"], reverse=True):
        lines[plan["anchor"] + 1 : plan["anchor"] + 1] = plan["lines"]
    new_text = "".join(lines)
    appended_provenance = COMPAT_PROVENANCE_SENTINEL not in new_text
    if appended_provenance:
        new_text += compat_provenance_block()
    yaml_path.write_text(new_text, encoding="utf-8")
    print(f"wrote {len(plans)} tf_compat_* insertion(s) into {yaml_path}")
    if appended_provenance:
        print("provenance block appended at end of file")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 27-04 docs layer: registry -> generated plain-language compatibility page
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DOCS_PAGE = "docs/user_guide/model_compatibility.md"

GENERATED_HEADER = (
    "<!-- GENERATED by scripts/census_compat.py --emit-docs - DO NOT EDIT BY HAND -->"
)

# Data date of the census the page reflects. Pinned on purpose: a wall-clock
# stamp would break the byte-identical regeneration gate (the same registry
# must produce the same page on any later day). Bump this constant alongside a
# census re-run + --write-registry --apply; the drift gate then re-pins it.
DOCS_DATA_DATE = "2026-10-27"

# The transformers-4-side honesty wording (D-05): inference is labeled
# inference. Pinned verbatim by the Task-2 gate (including the Chinese
# original from the owner-approved label semantics).
DOCS_FOOTNOTE_INFERENCE = (
    "Transformers 4 compatibility is inferred - the model code was written in "
    "the transformers 4 era; no separate transformers 4 environment was tested. "
    # Fullwidth parens are verbatim owner-approved label text (D-05), not typos.
    "(推断兼容（模型代码写于 transformers 4 时代）)"  # ruff: ignore[ambiguous-unicode-character-string]
)

# The five pinned "What we found" wordings for statuses that need no
# evidence-derived continuation (the broken/blocked cells append their cause).
DOCS_STATUS_CELLS: dict[str, str] = {
    "verified-tf5": "Tested working",
    "import-verified-tf5": "Tested loading its code; full run inferred from the family model",
    "shim-required-tf5": "Works with DNALLM's built-in compatibility fixes (tested)",
}

DOCS_TABLE_HEADER = ("Model", "What we found", "Details")

# An unqualified both-version claim is the one wording this page must never
# carry (Pitfall 8 / D-05): the transformers 4 side is inferred, not tested.
DOCS_FORBIDDEN_CLAIM = "compatible with transformers 4 and 5"


def _docs_cell(value: str) -> str:
    """Return a markdown-table-safe cell value, failing closed on breakers.

    Args:
        value: The raw cell text.

    Returns:
        The value unchanged.

    Raises:
        ValueError: When the value carries a pipe or newline (either would
            silently corrupt the generated table's column layout).
    """
    if "|" in value or "\n" in value:
        raise ValueError(f"docs table cell carries a table-breaking character: {value!r}")
    return value


def _docs_evidence_detail(evidence: str | None) -> str:
    """Return the evidence text after its ``hub repo@revision; `` stamp."""
    if not evidence:
        return ""
    head, sep, detail = evidence.partition("; ")
    return detail.strip() if sep else head.strip()


def docs_broken_cause(evidence: str | None) -> str:
    """Return the one-breath cause of a remote-code-broken verdict.

    Rule: the first ``;``-clause of the evidence detail, with the
    ``failing import: `` bookkeeping prefix stripped so the cell reads
    ``Not usable with the current transformers - fails at <cause>``. A bare
    single-symbol failing import (for example the external ``olmo`` package)
    says nothing on its own, so the following clause is preferred up to its
    first `` - `` boundary (it names what imports the missing symbol).

    Args:
        evidence: The entry's ``tf_compat_evidence`` string.

    Returns:
        The cause clause (empty when the evidence is empty).
    """
    detail = _docs_evidence_detail(evidence)
    clauses = [clause.strip() for clause in detail.split(";")]
    cause = clauses[0]
    prefix = "failing import: "
    if cause.startswith(prefix):
        cause = cause[len(prefix) :].strip()
        if cause and " " not in cause and len(clauses) > 1:
            cause = clauses[1].split(" - ")[0].strip()
    return cause


def docs_blocked_reason(evidence: str | None) -> str:
    """Return the plain-language reason of a census-blocked verdict.

    Rule: map the two committed blocking shapes (unreachable repo; unusable
    repository layout) to biologist-readable reasons; any other shape falls
    back to its first ``;``-clause. Raw cache paths never reach the page.

    Args:
        evidence: The entry's ``tf_compat_evidence`` string.

    Returns:
        The reason text (empty when the evidence is empty).
    """
    detail = _docs_evidence_detail(evidence)
    if "unreachable on both hubs" in detail:
        return "its repository could not be reached on either download site"
    if "cannot read fetched config.json" in detail:
        return "its repository does not provide a usable configuration file at the top level"
    return detail.split(";")[0].strip()


def docs_status_cell(entry) -> str:
    """Return the pinned "What we found" cell for one pretrained entry.

    Args:
        entry: A pretrained :class:`dnallm.models.registry.RegistryEntry`.

    Returns:
        The exact status-cell wording (one of the five pinned strings,
        with the evidence-derived cause/reason appended for the
        remote-code-broken and census-blocked families).

    Raises:
        ValueError: When the entry carries no census verdict or an unknown
            one, or a broken/blocked entry carries no evidence to derive
            its cause from.
    """
    status = entry.tf_compat_status
    if status is None:
        raise ValueError(f"registry entry {entry.model} carries no tf_compat_status")
    if status in DOCS_STATUS_CELLS:
        return DOCS_STATUS_CELLS[status]
    if status == "remote-code-broken":
        cause = docs_broken_cause(entry.tf_compat_evidence)
        if not cause:
            raise ValueError(f"broken entry {entry.model} carries no cause evidence")
        return f"Not usable with the current transformers - fails at {cause}"
    if status == "census-blocked":
        reason = docs_blocked_reason(entry.tf_compat_evidence)
        if not reason:
            raise ValueError(f"blocked entry {entry.model} carries no reason evidence")
        return f"Could not be checked ({reason})"
    raise ValueError(f"unknown tf_compat_status {status!r} on {entry.model}")


def docs_details_cell(entry) -> str:
    """Return the Details cell: repo id, plus the family representative repo
    for import-verified entries (D-05 honest sibling labeling)."""
    import re

    details = f"`{entry.model}`"
    if entry.tf_compat_status == "import-verified-tf5" and entry.tf_compat_evidence:
        match = re.search(r"family representative (\S+)", entry.tf_compat_evidence)
        if match:
            details += f" (full run tested on `{match.group(1)}`)"
    return details


def docs_page_text() -> str:
    """Render the plain-language compatibility page from the typed registry.

    Every value is derived from ``dnallm.models.registry`` accessors (the
    generator never parses the registry yaml) and the output is
    deterministic: the same registry produces a byte-identical page (the
    committed date is a pinned data-date constant, never a wall-clock stamp).

    Returns:
        The full markdown page text.

    Raises:
        ValueError: When any pretrained entry lacks a verdict, a verdict is
            outside the five-value vocabulary, the tested-version stamps are
            not a single shared value, or a cell value would break the table.
    """
    from collections import Counter

    from dnallm.models.registry import all_entries, sort_key_lab_first

    pretrained = [entry for entry in all_entries() if entry.kind == "pretrained"]
    missing = [entry.model for entry in pretrained if entry.tf_compat_status is None]
    if missing:
        raise ValueError(f"pretrained entries without a census verdict: {missing}")
    tested = {entry.tf_compat_tested for entry in pretrained if entry.tf_compat_tested}
    if len(tested) != 1:
        raise ValueError(
            f"the page footnote interpolates ONE tested transformers version, got {sorted(tested)}"
        )
    tested_version = next(iter(tested))
    counts = Counter(entry.tf_compat_status for entry in pretrained)
    total = len(pretrained)
    working = counts["verified-tf5"]
    code_verified = counts["import-verified-tf5"]
    shim = counts["shim-required-tf5"]
    broken = counts["remote-code-broken"]
    blocked = counts["census-blocked"]

    lines = [
        GENERATED_HEADER,
        f"<!-- Regeneration date: {DOCS_DATA_DATE} (data date of the census run, not a wall-clock stamp) -->",
        "",
        "# Model Compatibility",
        "",
        "DNALLM depends on a widely used AI library called *transformers*, which "
        "is the software that loads and runs the models. That library has "
        "released a new major version (version 5), and the team tested every "
        "preset model that ships with DNALLM against it.",
        "",
        f"Of the {total} preset models, the team found {working} working as-is, "
        f"{code_verified} with their program code loading verified (the full run "
        "is expected to match a tested close relative in the same model family), "
        f"{shim} working through DNALLM's built-in compatibility fixes, "
        f"{broken} not working with the new version, and {blocked} that could "
        "not be checked.",
        "",
        f"| {DOCS_TABLE_HEADER[0]} | {DOCS_TABLE_HEADER[1]} | {DOCS_TABLE_HEADER[2]} |",
        "| --- | --- | --- |",
    ]
    for entry in sorted(pretrained, key=sort_key_lab_first):
        lines.append(
            f"| {_docs_cell(entry.name)} | {_docs_cell(docs_status_cell(entry))} "
            f"| {_docs_cell(docs_details_cell(entry))} |"
        )
    lines += [
        "",
        f"1. Tested on transformers {tested_version}. (在 transformers {tested_version} 上实测)",
        f"2. {DOCS_FOOTNOTE_INFERENCE}",
        "",
        "This page is generated from the model registry (the single source of "
        "truth) by `scripts/census_compat.py --emit-docs`; manual edits are "
        "overwritten on regeneration.",
        "",
    ]
    page = "\n".join(lines)
    if DOCS_FORBIDDEN_CLAIM in page:
        raise ValueError("the generated page carries an unqualified both-version claim")
    return page


def emit_docs(path: str | Path) -> int:
    """Write the generated compatibility page (no-op when already current).

    Args:
        path: Destination markdown path (the committed docs page by default).

    Returns:
        0 on success, 1 on a fail-closed condition (a verdict missing, an
        unknown status, or an unrenderable cell).
    """
    try:
        text = docs_page_text()
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    destination = Path(path)
    if destination.exists() and destination.read_text(encoding="utf-8") == text:
        print(f"docs page already current: {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print(f"wrote docs page: {destination}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 27-04 re-verification: fresh census vs committed registry fields (fail-closed)
# ─────────────────────────────────────────────────────────────────────────────

# Loud marker for the version-stamp check (research A8: a transformers bump
# invalidates version-stamped verdicts — exactly what this lane exists to catch).
VERSION_DRIFT_MARKER = "VERSION DRIFT"


def verify_registry(artifact_path: str | Path, roster: list | None = None) -> int:
    """Diff census verdicts against the committed registry fields, fail-closed.

    The fresh evidence (a just-written ``--run`` artifact, or the committed
    artifact for a dry offline verification) is mapped through
    :func:`compat_verdict` and compared, per pretrained entry, against the
    committed ``tf_compat_status`` read through the typed registry loader.
    Every disagreement, every roster/artifact gap, an absent or unparseable
    artifact, or a transformers version drift fails the run — never a silent
    green (audit_skips posture).

    Args:
        artifact_path: The census evidence artifact to verify from.
        roster: Pretrained registry entries (defaults to the typed loader —
            the sole parser of the packaged file).

    Returns:
        0 only when every entry matches AND the installed transformers
        version equals the committed tested stamp; 1 otherwise.
    """
    try:
        records = read_artifact(artifact_path)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    if roster is None:
        roster = pretrained_roster()
    installed = transformers_version()
    committed_tested = sorted({
        entry.tf_compat_tested for entry in roster if entry.tf_compat_tested
    })
    version_drift = installed not in committed_tested
    if version_drift:
        print(
            f"{VERSION_DRIFT_MARKER}: installed transformers {installed} vs committed "
            f"tf_compat_tested {committed_tested}"
        )
        print(
            f"{VERSION_DRIFT_MARKER}: version-stamped verdicts are invalidated by a "
            "transformers change - re-run the census and refresh the registry fields"
        )

    by_repo = {
        record.get("repo"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("repo"), str)
    }
    family_reps = {
        record["family"]: record["repo"]
        for record in records
        if record.get("is_family_rep") and record.get("family")
    }
    missing_from_artifact = [entry.model for entry in roster if entry.model not in by_repo]
    missing_from_registry = sorted(
        repo for repo in by_repo if repo not in {entry.model for entry in roster}
    )
    trail: list[str] = []
    unexpected = 0
    for entry in roster:
        record = by_repo.get(entry.model)
        if record is None:
            continue  # already recorded in missing_from_artifact below
        try:
            computed = compat_verdict(record, family_reps)[0]
        except ValueError as error:
            print(f"UNEXPECTED {entry.model} unmappable: {error}")
            unexpected += 1
            continue
        committed = entry.tf_compat_status
        if computed == committed:
            trail.append(f"matched {entry.model} {committed}")
        else:
            trail.append(f"UNEXPECTED {entry.model} committed={committed} fresh={computed}")
            unexpected += 1

    print(f"Registry verification: {len(trail)} entry/entries compared against {artifact_path}")
    for line in trail:
        print(f"  {line}")
    failures = False
    if missing_from_artifact:
        print(
            f"MISSING FROM ARTIFACT ({len(missing_from_artifact)} registry entries "
            f"absent from the census evidence): {missing_from_artifact}"
        )
        failures = True
    if missing_from_registry:
        print(
            f"MISSING FROM REGISTRY ({len(missing_from_registry)} artifact records "
            f"outside the pretrained roster): {missing_from_registry}"
        )
        failures = True
    if unexpected:
        print(f"UNEXPECTED ({unexpected} entr(ies) disagree with the committed registry fields)")
        failures = True
    if version_drift:
        print(
            f"{VERSION_DRIFT_MARKER}: failing the run - the committed verdicts were "
            f"stamped under {committed_tested}, not the installed {installed}"
        )
        failures = True
    if failures:
        return 1
    print("OK: census matches committed registry fields")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the census CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fail-closed transformers compatibility census over the pretrained registry."
    )
    parser.add_argument("--run", action="store_true", help="execute the census")
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "diff the census verdicts against the committed registry fields "
            "(fail-closed: any disagreement, roster gap, or transformers version "
            "drift exits 1; composes with --run or runs dry on the existing artifact)"
        ),
    )
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="write the census verdicts into the registry yaml as tf_compat_* fields (dry-run default)",
    )
    parser.add_argument(
        "--emit-docs",
        nargs="?",
        const=DEFAULT_DOCS_PAGE,
        default=None,
        metavar="PATH",
        help=(
            "write the generated plain-language compatibility page derived from the "
            f"typed registry (default path: {DEFAULT_DOCS_PAGE})"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="with --write-registry: write the planned insertions (default: dry-run)",
    )
    parser.add_argument(
        "--registry-yaml",
        default=None,
        help="registry yaml to write (default: the packaged dnallm/models/model_info.yaml)",
    )
    parser.add_argument(
        "--entries",
        default=None,
        help="comma-separated repo ids to census (default: the full pretrained roster)",
    )
    parser.add_argument(
        "--with-loads",
        action="store_true",
        help="also load and forward-test one representative per model family",
    )
    parser.add_argument(
        "--artifact",
        default=DEFAULT_ARTIFACT,
        help=f"evidence artifact path (default: {DEFAULT_ARTIFACT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the census CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 clean, 1 for a census artifact error or a
        failed ``--verify`` verdict diff, 2 for a usage error. With
        ``--run --verify`` composed, a hard census error and a verdict
        disagreement each fail the run independently (``run_rc or
        verify_rc``) — the verdict diff is never masked by the run's exit
        code, and the run's hard errors are never masked by a green diff.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.emit_docs is not None:
        return emit_docs(Path(args.emit_docs))
    if args.write_registry:
        if args.registry_yaml is not None:
            yaml_path = Path(args.registry_yaml)
        else:
            import dnallm

            yaml_path = Path(dnallm.__file__).parent / "models" / "model_info.yaml"
        return write_registry(Path(args.artifact), yaml_path, apply=args.apply)
    if not args.run and not args.verify:
        print(
            "usage: census_compat.py --run [--entries repo,...] [--with-loads] [--artifact PATH] "
            "OR --write-registry [--apply] [--registry-yaml PATH] "
            "OR --emit-docs [PATH] "
            "OR --verify [--artifact PATH]",
            file=sys.stderr,
        )
        return 2
    selected = (
        [item.strip() for item in args.entries.split(",") if item.strip()] if args.entries else None
    )
    run_rc = 0
    if args.run:
        run_rc = run_census(Path(args.artifact), selected, args.with_loads)
    if args.verify:
        verify_rc = verify_registry(Path(args.artifact))
        # CR-01: both failure classes surface independently. The previous
        # ``run_rc if run_rc != 0 else verify_rc`` discarded the verification
        # result whenever the run exited nonzero — which it always did for a
        # committed-known unreachable repo — so the composed dispatch lane
        # could never exit 0 and "red" could not distinguish real drift from
        # baseline.
        return run_rc or verify_rc
    return run_rc


if __name__ == "__main__":
    sys.exit(main())
