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
    return "\n".join([
        "import os",
        f"os.environ['HF_ENDPOINT'] = {HF_MIRROR!r}",
        "import json",
        "import time",
        "import traceback",
        "import dnallm  # arms transformers_compat.apply_patches() before any remote import",
        "import torch",
        "from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM",
        f"repo = {repo!r}",
        f"hub = {hub!r}",
        f"architectures = {architectures!r}",
        f"revision = {pinned!r}",
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
        "    architecture = architectures[0] if architectures else ''",
        "    if architecture.endswith('ForCausalLM'):",
        "        model_class = AutoModelForCausalLM",
        "    elif architecture.endswith('ForMaskedLM'):",
        "        model_class = AutoModelForMaskedLM",
        "    else:",
        "        model_class = AutoModel",
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


def apply_loads(records: list[dict]) -> None:
    """Add the D-02b load layer: load + short forward per family representative.

    Non-representatives carry ``load: null`` plus an honest ``load_note``
    naming their representative; ``import-fail`` entries stay parked as
    "pending shim" (their loads happen after the 27-02 fix); ``census-blocked``
    entries carry no load evidence. Mutates records in place.

    Args:
        records: Evidence records in registry file order (after
            :func:`assign_families`).
    """
    representatives = {r["family"]: r["repo"] for r in records if r["is_family_rep"]}
    for record in records:
        rep = representatives.get(record["family"])
        suffix = (
            ""
            if record["is_family_rep"] or rep is None
            else f" (inferred from family representative {rep})"
        )
        if record["status"] == "import-fail":
            record["load"] = None
            record["load_note"] = "import-fail pending shim" + suffix
            continue
        if record["status"] == "census-blocked":
            record["load"] = None
            record["load_note"] = "census-blocked: no load evidence (see failing_import)" + suffix
            continue
        if not record["is_family_rep"]:
            record["load"] = None
            record["load_note"] = f"inferred from family representative {rep}"
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

    A fresh record replaces its repo's existing record. Previously measured
    load evidence is carried over only when the status is unchanged and the
    fresh run measured no loads (an import-only re-run must not erase the
    D-02b layer).

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
            record.get("load") is None
            and previous is not None
            and previous.get("load") is not None
            and previous.get("status") == record.get("status")
        ):
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
        0 on a clean run, 1 when any entry is census-blocked by unreachability
        (loud failure), 2 for a usage error.
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
    if with_loads:
        apply_loads(records)

    merged = records
    if artifact_path.exists():
        try:
            existing = read_artifact(artifact_path)
        except ValueError as error:
            print(f"ERROR: {error}")
            return 1
        merged = merge_records(existing, records, roster_order)
    try:
        write_artifact(artifact_path, merged)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Census complete: {len(records)} entry/entries censused -> {artifact_path}")
    if unreachable:
        print(f"CENSUS BLOCKED ({len(unreachable)} unreachable on both hubs): {unreachable}")
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the census CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fail-closed transformers compatibility census over the pretrained registry."
    )
    parser.add_argument("--run", action="store_true", help="execute the census")
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
        Process exit code: 0 clean, 1 census-blocked evidence, 2 usage error.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.run:
        print(
            "usage: census_compat.py --run [--entries repo,...] [--with-loads] [--artifact PATH]",
            file=sys.stderr,
        )
        return 2
    selected = (
        [item.strip() for item in args.entries.split(",") if item.strip()] if args.entries else None
    )
    return run_census(Path(args.artifact), selected, args.with_loads)


if __name__ == "__main__":
    sys.exit(main())
