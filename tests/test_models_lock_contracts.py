"""CI-08 guard: every remote model id referenced by example content is a models.lock row.

Provenance (Phase 9, criterion 4 / 09-RESEARCH "Pattern 6: Lock-consistency
guard"): ``models.lock`` is the provenance contract for every remote artifact
the nightly census touches, yet nothing cross-checked it against example
content -- a notebook silently repointed at an unlocked model would pass
review.  This file closes that hole on the fast leg: it parses the lock
(fail-closed, the ``scripts/audit_skips.py`` discipline), scans example
artifacts (notebook code cells, marimo apps, example YAML configs, the NER
helper script) for remote model-id literals, and fails listing any referenced
id that is neither a lock row nor a documented non-model exclusion.

Allowlist discipline (T-09-08): ``_NON_MODEL_ALLOWLIST`` is the in-file
exclusion record -- every entry carries a one-line reason and must still
match live example content (a stale entry fails
``test_allowlist_entries_carry_reasons_and_match_live_content``).  A
genuinely fetched-but-unlocked model is a REAL catch: add the lock row
(prefix aligned to the notebook's ACTIVE ``source=`` route, purpose comment
included), never an allowlist entry.  The guard never writes to
``models.lock`` at test runtime -- lock additions are deliberate review
acts.

Route alignment (best-effort extension of the
``TestMegadnaSiblingContentContracts`` ACTIVE-line precedent): a scanned
code file that references exactly ONE lock model id and carries exactly one
distinct ACTIVE ``source="<route>"`` literal must satisfy
lock-prefix == route (``ms`` <=> modelscope, ``hf`` <=> huggingface; the
models.lock header rule).  Covered on the live tree (15 files):

- notebooks/data_prepare/finetune/finetune_data.ipynb (plant-dnabert-BPE, ms)
- notebooks/finetune_NER_task/data_generation_and_inference.ipynb (dnagpt-6mer, ms)
- notebooks/finetune_NER_task/finetune_NER_task.ipynb (plant-nucleotide-transformer-BPE, ms)
- notebooks/finetune_NER_task/generate_bpe_dataset.py (plant-nucleotide-transformer-BPE, ms)
- notebooks/finetune_binary/finetune_binary.ipynb (plant-dnabert-BPE, ms)
- notebooks/finetune_multi_labels/finetune_multi_labels.ipynb (plant-dnagpt-BPE, ms)
- notebooks/generation/inference.ipynb (plant-dnagpt-BPE, ms)
- notebooks/generation_megaDNA/inference.ipynb (megaDNA_updated, hf)
- notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb (promoter_strength_protoplast, ms)
- notebooks/inference/inference.ipynb (plant-dnagpt-BPE-promoter, ms)
- notebooks/interpretation/interpretation.ipynb (promoter_strength_leaf, ms)
- notebooks/lora_finetune_inference/lora_finetune.ipynb (PlantCAD2-Small, hf)
- notebooks/lora_finetune_inference/lora_inference.ipynb (PlantCAD2-Small, hf)
- notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb (PlantHelixSeek-Anno, ms)
- notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb (PlantHelixSeek-CRE, ms)

Skipped by the route check (ambiguous or route-less, by design):
finetune_custom_head + finetune_generation (two model ids and two routes
each), generation_evo_models (two model ids), inference_for_tRNA (two model
ids across two call sites), plant_helixseek_combined (two model ids),
embedding_attention.ipynb (raw AutoModel route -- no ACTIVE source= literal),
the marimo apps (route chosen at runtime via mo.ui.dropdown), every YAML
config (model paths but no source= lines), and the mcp_example pair (local
ollama models only).

The guard is kernel-free and network-free, runs unmarked on every push/PR
fast leg, and adds zero new skips (CI-03).  The failure mode is proven by
drift injection on synthetic fixtures (``TestDriftInjection`` /
``TestRouteAlignment``), so a vacuous pass on the live tree cannot hide
(T-09-05).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "models.lock"
EXAMPLE_DIR = REPO_ROOT / "example"

# Non-model strings that survive the org/name shape filter; each entry is a
# (token, one-line reason) pair (T-09-08: undocumented or stale entries fail
# test_allowlist_entries_carry_reasons_and_match_live_content below).
_NON_MODEL_ALLOWLIST: tuple[tuple[str, str], ...] = (
    (
        "plantcad/cross_species_acr_train_on_arabidopsis_plantcad2_small",
        "local LoRA adapter directory name (lora_finetune saves under ./outputs, "
        "lora_inference passes it as lora_adapter=) -- a local path, not a remote artifact",
    ),
)

# Lock row shapes: "<hf|ms>  <repo-id>[@<sha>]  # purpose" plus one "dataset:" row.
_ORG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Quoted org/name literals in code text. The fullmatch shape alone already
# rejects path-like tokens (leading ./ ../ / -- none can start with an
# alphanumeric -- and any second slash), so those plan filters bind here.
_QUOTED_TOKEN_RE = re.compile(
    r"""["']([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)["']"""
)

# YAML scalar value position ("key: org/name" or "- org/name"): the example
# configs carry UNQUOTED model paths (e.g. benchmark_config.yaml "path:" rows),
# which the quoted-token regex cannot see.
_YAML_VALUE_RE = re.compile(
    r"""(?:^|[ \t])(?:-[ \t]+|[A-Za-z0-9_.-]+:[ \t]*)"""
    r"""["']?([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)["']?[ \t]*$""",
    re.M,
)

# Data/config file extensions: a token ending in one of these is a file
# reference, never a model id (e.g. data/TAIR10_GFF3_chr1_....gff3 fetches).
_FILE_EXTENSION_SUFFIXES: tuple[str, ...] = (
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".py",
    ".md",
    ".ipynb",
    ".json",
    ".txt",
    ".gz",
    ".fa",
    ".fas",
    ".fasta",
    ".fna",
    ".pt",
    ".safetensors",
    ".gff",
    ".gff3",
    ".gtf",
    ".bed",
    ".bedgraph",
    ".pkl",
    ".parquet",
    ".toml",
    ".log",
    ".html",
    ".svg",
    ".png",
    ".jpg",
)

# MIME top-level types: display-mime dict keys ("image/png", "text/plain")
# in the showcase notebooks are org/name-shaped but not model artifacts.
_MIME_TOP_LEVEL_TYPES = frozenset({
    "application",
    "audio",
    "font",
    "image",
    "message",
    "multipart",
    "text",
    "video",
})

_SOURCE_ROUTE_RE = re.compile(r"""\bsource\s*=\s*["'](\w+)["']""")
_ROUTE_TO_LOCK_PREFIX = {"modelscope": "ms", "huggingface": "hf"}

_SCAN_SUFFIXES = frozenset({".ipynb", ".py", ".yaml"})
_PRUNED_DIR_NAMES = frozenset({"__pycache__"})


def _parse_lock_rows(
    path: Path = LOCK_PATH,
) -> tuple[list[tuple[str, str, str | None]], str]:
    """Parse models.lock into model rows plus the dataset row id.

    Fail-closed (the audit_skips.py discipline): trailing ``#`` comments are
    stripped, blank lines skipped, and any remaining row that is not a
    well-formed ``hf``/``ms`` model row or the single ``dataset:`` row raises
    ValueError naming the line -- an unparseable lock fails the guard instead
    of widening it.

    Args:
        path: lock file to parse (defaults to the repository models.lock).

    Returns:
        ``(model_rows, dataset_id)`` where each model row is a
        ``(prefix, repo_id, pinned_sha_or_None)`` tuple.

    Raises:
        ValueError: on any malformed or unrecognized non-comment row.
    """
    model_rows: list[tuple[str, str, str | None]] = []
    dataset_id = ""
    lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] in ("hf", "ms"):
            if len(parts) != 2 or not _ORG_NAME_RE.fullmatch(parts[1].split("@", 1)[0]):
                raise ValueError(
                    f"{path}:{lineno}: malformed model row (expected "
                    f"'<hf|ms>  <org/name>[@<sha>]'): {raw!r}"
                )
            repo_id, _, sha = parts[1].partition("@")
            model_rows.append((parts[0], repo_id, sha or None))
        elif parts[0] == "dataset:":
            if len(parts) != 2 or not _ORG_NAME_RE.fullmatch(parts[1].split("@", 1)[0]):
                raise ValueError(
                    f"{path}:{lineno}: malformed dataset row (expected "
                    f"'dataset:  <org/name>'): {raw!r}"
                )
            dataset_id = parts[1].split("@", 1)[0]
        else:
            raise ValueError(
                f"{path}:{lineno}: unrecognized lock row (expected 'hf'/'ms' model "
                f"row or 'dataset:'): {raw!r}"
            )
    return model_rows, dataset_id


def _iter_scan_files(root: Path) -> Iterator[Path]:
    """Yield .ipynb/.py/.yaml files under root, pruning dot-dirs and __pycache__."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
            continue
        rel_parts = path.relative_to(root).parts
        if any(
            part.startswith(".") or part in _PRUNED_DIR_NAMES for part in rel_parts[:-1]
        ) or path.name.startswith("."):
            continue
        yield path


def _code_text(path: Path) -> str:
    """Return code-bearing text: notebook code cells only (the _code_cells idiom)."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in nb["cells"]
            if cell["cell_type"] == "code"
        )
    if path.suffix == ".yaml":
        # Strip per-line comments so value-position matching never reads a
        # commented-out model path as a live reference.
        return "\n".join(
            line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    return path.read_text(encoding="utf-8")


def _is_model_id_candidate(token: str) -> bool:
    """True when a quoted/value org/name token looks like a remote model id.

    Rejects URL/www fragments, ``*.git`` clone targets, file-extension
    carriers, and MIME-type strings, on top of the org/name fullmatch shape.
    """
    if not _ORG_NAME_RE.fullmatch(token):
        return False
    if "http" in token or "www" in token:
        return False
    if token.endswith(".git"):
        return False
    if token.endswith(_FILE_EXTENSION_SUFFIXES):
        return False
    return token.split("/", 1)[0] not in _MIME_TOP_LEVEL_TYPES


def _example_model_id_candidates(root: Path = EXAMPLE_DIR) -> dict[str, set[Path]]:
    """Scan example content and map each model-id candidate to its sourcing files.

    Notebooks contribute ONLY code-cell source (list-or-string joined); .py
    and .yaml files contribute raw text; YAML files additionally match
    unquoted scalar-value positions with per-line comment stripping.
    """
    candidates: dict[str, set[Path]] = {}
    for path in _iter_scan_files(root):
        text = _code_text(path)
        tokens = {match.group(1) for match in _QUOTED_TOKEN_RE.finditer(text)}
        if path.suffix == ".yaml":
            tokens.update(match.group(1) for match in _YAML_VALUE_RE.finditer(text))
        for token in tokens:
            if _is_model_id_candidate(token):
                candidates.setdefault(token, set()).add(path)
    return candidates


def _find_unlocked_ids(
    model_rows: list[tuple[str, str, str | None]],
    dataset_id: str,
    candidates: dict[str, set[Path]],
) -> dict[str, set[Path]]:
    """Return candidates that are neither lock rows, the dataset row, nor allowlisted."""
    locked = {repo_id for _, repo_id, _ in model_rows} | {dataset_id}
    allowed = {token for token, _ in _NON_MODEL_ALLOWLIST}
    return {
        token: files
        for token, files in candidates.items()
        if token not in locked and token not in allowed
    }


def _active_text(text: str) -> str:
    """Drop comment lines (the _active_lines idiom): alternative source= routes
    are documented as comments; only the ACTIVE route is contract-relevant."""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def _route_alignment_violations(
    root: Path,
    model_rows: list[tuple[str, str, str | None]],
) -> tuple[list[str], int]:
    """Check single-id/single-route code files against lock prefixes.

    A scanned .ipynb/.py file is covered iff its ACTIVE text references
    exactly one lock model id and carries exactly one distinct ACTIVE
    ``source="<route>"`` literal; covered files must satisfy
    lock-prefix == route (covered/skipped sets documented in the module
    docstring). Returns ``(violations, covered_file_count)``.
    """
    prefix_by_repo = {repo_id: prefix for prefix, repo_id, _ in model_rows}
    violations: list[str] = []
    covered = 0
    for path in _iter_scan_files(root):
        if path.suffix not in (".ipynb", ".py"):
            continue  # YAML configs carry model paths but no source= routes
        active = _active_text(_code_text(path))
        routes = set(_SOURCE_ROUTE_RE.findall(active))
        referenced = {
            token
            for token in (match.group(1) for match in _QUOTED_TOKEN_RE.finditer(active))
            if token in prefix_by_repo
        }
        if len(referenced) != 1 or len(routes) != 1:
            continue  # ambiguous or route-less: the documented skipped set
        repo_id = next(iter(referenced))
        route = next(iter(routes))
        expected = _ROUTE_TO_LOCK_PREFIX.get(route)
        if expected is None:
            violations.append(
                f"{path}: ACTIVE source={route!r} maps to no lock prefix "
                f"(known routes: {sorted(_ROUTE_TO_LOCK_PREFIX)})"
            )
            continue
        covered += 1
        if prefix_by_repo[repo_id] != expected:
            violations.append(
                f"{path}: ACTIVE source={route!r} implies lock prefix {expected!r} "
                f"but models.lock prefixes {repo_id!r} as {prefix_by_repo[repo_id]!r} "
                f"(the models.lock header alignment rule)"
            )
    return violations, covered


def _write_notebook(path: Path, *code_cells: str) -> None:
    """Write a minimal single-kernel notebook JSON with the given code cells."""
    path.write_text(
        json.dumps({
            "cells": [
                {"cell_type": "code", "metadata": {}, "source": cell.splitlines(keepends=True)}
                for cell in code_cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )


class TestLockParsing:
    """Behavior 1 (CI-08): the lock parses fail-closed with its full row set."""

    def test_real_lock_parses_with_at_least_24_model_rows_and_the_dataset_row(self) -> None:
        """The live lock yields >=24 (prefix, repo_id, sha-or-None) rows + dataset id."""
        model_rows, dataset_id = _parse_lock_rows()
        assert len(model_rows) >= 24, f"expected >=24 model rows, got {len(model_rows)}"
        assert dataset_id == "zhangtaolab/plant-multi-species-core-promoters"
        assert {prefix for prefix, _, _ in model_rows} <= {"hf", "ms"}
        shas = [sha for _, _, sha in model_rows]
        # Both documented forms exist: pinned rows (@sha) and legacy unpinned rows.
        assert any(sha is not None for sha in shas), (
            "no pinned rows found -- parse lost the @sha form"
        )
        assert any(sha is None for sha in shas), (
            "no unpinned rows found -- header says ten legacy rows stay unpinned"
        )

    def test_malformed_model_row_raises_value_error(self, tmp_path: Path) -> None:
        """Unparseable model-shaped rows fail the guard instead of passing silently."""
        bad_lines = [
            "hf",  # bare prefix, no repo id
            "xx  zhangtaolab/unknown-prefix-row",  # unrecognized row prefix
            "hf  zhangtaolab/extra-field row-with-extra",  # three fields after comment strip
            "hf  no-slash-token",  # not org/name shaped
        ]
        for i, bad in enumerate(bad_lines):
            lock = tmp_path / f"bad-{i}.lock"
            lock.write_text("# header\n" + bad + "\n", encoding="utf-8")
            with pytest.raises(ValueError, match=r"line 2|:2:"):
                _parse_lock_rows(lock)

    def test_trailing_comment_and_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        """Header comments, trailing purpose comments, and blanks never become rows."""
        lock = tmp_path / "comments.lock"
        lock.write_text(
            "# header line\n"
            "\n"
            "ms  zhangtaolab/mini-model  # purpose comment with spaces\n"
            "dataset:  zhangtaolab/mini-dataset  # dataset purpose\n",
            encoding="utf-8",
        )
        model_rows, dataset_id = _parse_lock_rows(lock)
        assert model_rows == [("ms", "zhangtaolab/mini-model", None)]
        assert dataset_id == "zhangtaolab/mini-dataset"


class TestDriftInjection:
    """Behavior 2 (CI-08): the failure-mode proof -- drift IS reported."""

    def test_unlocked_synthetic_model_id_is_reported_exactly(self, tmp_path: Path) -> None:
        """A notebook referencing an id absent from the mini lock is reported,
        and the reported set equals the injected id (no over-reporting)."""
        lock = tmp_path / "mini.lock"
        lock.write_text(
            "# mini lock\nms  zhangtaolab/locked-model  # synthetic fixture\n",
            encoding="utf-8",
        )
        nb_dir = tmp_path / "example"
        nb_dir.mkdir()
        _write_notebook(
            nb_dir / "drift.ipynb",
            'model_name = "zhangtaolab/drifted-model"\n'
            'model, tok = load_model_and_tokenizer(model_name, source="modelscope")\n',
        )
        model_rows, dataset_id = _parse_lock_rows(lock)
        candidates = _example_model_id_candidates(nb_dir)
        unlocked = _find_unlocked_ids(model_rows, dataset_id, candidates)
        assert set(unlocked) == {"zhangtaolab/drifted-model"}, (
            f"drift injection must report exactly the injected id; got {sorted(unlocked)} "
            f"from candidates {sorted(candidates)}"
        )


class TestLiveTreeMembership:
    """Behavior 3 (CI-08): the live example tree is green, exclusions documented."""

    def test_every_example_referenced_remote_id_is_locked_or_documented(self) -> None:
        """Membership guard: zero unlocked ids on the live tree (CI-08)."""
        model_rows, dataset_id = _parse_lock_rows()
        candidates = _example_model_id_candidates()
        unlocked = _find_unlocked_ids(model_rows, dataset_id, candidates)
        assert not unlocked, (
            "example content references remote model ids missing from models.lock "
            "(CI-08): triage each below -- a fetched model adds a LOCK ROW (prefix "
            "aligned to its ACTIVE source= route, purpose comment), never an "
            f"allowlist entry; only non-model strings join _NON_MODEL_ALLOWLIST: "
            f"{ {tok: sorted(str(p) for p in files) for tok, files in unlocked.items()} }"
        )

    def test_allowlist_entries_carry_reasons_and_match_live_content(self) -> None:
        """Every _NON_MODEL_ALLOWLIST entry has a one-line reason and is live (T-09-08)."""
        candidates = _example_model_id_candidates()
        seen: set[str] = set()
        for token, reason in _NON_MODEL_ALLOWLIST:
            assert token not in seen, f"duplicate allowlist entry: {token}"
            seen.add(token)
            assert reason.strip(), f"allowlist entry without a reason: {token}"
            assert token in candidates, (
                f"stale allowlist entry {token} no longer appears in example content "
                "-- remove it so the record stays honest"
            )


class TestRouteAlignment:
    """Behavior 4 (CI-08 best-effort): ACTIVE source= route matches lock prefix."""

    def test_synthetic_route_prefix_mismatch_is_reported(self, tmp_path: Path) -> None:
        """An ACTIVE source="modelscope" load of an hf-prefixed id is a violation."""
        model_rows = [("hf", "zhangtaolab/mini-model", None)]
        nb_dir = tmp_path / "example"
        nb_dir.mkdir()
        _write_notebook(
            nb_dir / "mismatch.ipynb",
            'name = "zhangtaolab/mini-model"\n'
            'model, tok = load_model_and_tokenizer(name, source="modelscope")\n',
        )
        violations, covered = _route_alignment_violations(nb_dir, model_rows)
        assert covered == 1, "synthetic mismatch fixture must be covered, not skipped"
        assert len(violations) == 1, violations
        assert "mismatch.ipynb" in violations[0], violations

    def test_synthetic_aligned_route_is_not_reported(self, tmp_path: Path) -> None:
        """Positive control: modelscope route + ms prefix passes and is covered."""
        model_rows = [("ms", "zhangtaolab/mini-model", None)]
        nb_dir = tmp_path / "example"
        nb_dir.mkdir()
        _write_notebook(
            nb_dir / "aligned.ipynb",
            'name = "zhangtaolab/mini-model"\n'
            'model, tok = load_model_and_tokenizer(name, source="modelscope")\n',
        )
        violations, covered = _route_alignment_violations(nb_dir, model_rows)
        assert violations == []
        assert covered == 1

    def test_commented_alternative_routes_do_not_count(self, tmp_path: Path) -> None:
        """Alternative source= routes documented as comments stay out of the check."""
        model_rows = [("ms", "zhangtaolab/mini-model", None)]
        nb_dir = tmp_path / "example"
        nb_dir.mkdir()
        _write_notebook(
            nb_dir / "commented.ipynb",
            'name = "zhangtaolab/mini-model"\n'
            '# model, tok = load_model_and_tokenizer(name, source="huggingface")\n'
            'model, tok = load_model_and_tokenizer(name, source="modelscope")\n',
        )
        violations, covered = _route_alignment_violations(nb_dir, model_rows)
        assert violations == []
        assert covered == 1

    def test_live_tree_unambiguous_files_align_and_coverage_is_real(self) -> None:
        """Every covered live file agrees with its lock prefix; coverage is
        non-vacuous (>=10 files) so the check cannot silently cover nothing."""
        model_rows, _ = _parse_lock_rows()
        violations, covered = _route_alignment_violations(EXAMPLE_DIR, model_rows)
        assert not violations, violations
        assert covered >= 10, (
            f"route-alignment check covered only {covered} files -- the covered set "
            "shrank; update the module docstring and this floor deliberately"
        )


class TestScanFilters:
    """Behavior 5 (CI-08): the extractor ignores non-model org/name lookalikes."""

    def test_urls_git_targets_paths_extensions_and_mime_are_not_candidates(
        self, tmp_path: Path
    ) -> None:
        """URL strings, .git clone targets, path-like tokens, file extensions,
        and MIME keys produce no candidates (the megaDNA clone false positive)."""
        nb_dir = tmp_path / "example"
        nb_dir.mkdir()
        _write_notebook(
            nb_dir / "filters.ipynb",
            "!git clone https://github.com/lingxusb/megaDNA.git\n"
            'repo = "lingxusb/megaDNA.git"\n'
            'mirror = "https://modelscope.cn/models/zhangtaolab/plant-dnagpt-BPE"\n'
            'data = "data/TAIR10_DHSs_chr1_5100001_5300000.gff"\n'
            'rel = "./outputs/adapter"\n'
            'deep = "a/b/c"\n'
            'mime = "image/png"\n'
            'name = "zhangtaolab/real-model"\n',
        )
        candidates = _example_model_id_candidates(nb_dir)
        assert set(candidates) == {"zhangtaolab/real-model"}, (
            f"extractor leaked non-model tokens: {sorted(candidates)}"
        )

    def test_yaml_unquoted_value_positions_are_scanned(self, tmp_path: Path) -> None:
        """Unquoted YAML scalar model paths (benchmark_config.yaml shape) are
        candidates, and YAML comments are ignored."""
        nb_dir = tmp_path / "example"
        nb_dir.mkdir()
        (nb_dir / "cfg.yaml").write_text(
            "models:\n"
            "  - zhangtaolab/unquoted-yaml-model\n"
            "path: zhangtaolab/another-yaml-model\n"
            "# path: zhangtaolab/commented-out-model\n",
            encoding="utf-8",
        )
        candidates = _example_model_id_candidates(nb_dir)
        assert set(candidates) == {
            "zhangtaolab/unquoted-yaml-model",
            "zhangtaolab/another-yaml-model",
        }, sorted(candidates)
