# Phase 6: Model Registry & Showcase Data Curation - Pattern Map

**Mapped:** 2026-10-02
**Files analyzed:** 9 (3 new code, 3 new tests, 1 modified data, 2 new data/config artifact groups)
**Analogs found:** 9 / 9 (every file has an in-repo analog; 2 also carry a "domain reference" analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/utils/genomic_coords.py` | utility | transform | `dnallm/utils/sequence.py` | exact (pure-function utils module) |
| `tests/utils/test_genomic_coords.py` | test | transform | `tests/utils/test_sequence.py` | exact |
| `dnallm/models/model_info.yaml` (append 2 entries) | config (packaged data) | static data | itself — `finetuned:` entries at lines 164-172, 1443-1448, tail 1617-1624 | exact (self-precedent) |
| `scripts/showcase/freeze_registry.py` | script | file-I/O | `scripts/feasibility/spike_families.py` | role-match |
| `scripts/showcase/select_loci.py` | script | file-I/O + batch | `scripts/feasibility/spike_families.py` (skeleton) + `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` (domain: pyfastx/GFF3) | role-match |
| `tests/models/test_plant_helixseek_registry.py` | test | file-I/O (parse-only, no network) | `tests/examples/test_examples.py` `test_yaml_validity` (lines 280-293) | role-match |
| `tests/models/test_plant_helixseek_smoke.py` | test | request-response (real download + forward) | `tests/models/test_model.py` lines 178-202 + `tests/inference/test_inference_real_model.py` lines 22-80 | exact |
| `example/notebooks/plant_helixseek_{cre,anno,shared}/data/*` | data | file-I/O | (data artifacts — no code analog; gitignore mechanics below) | n/a |
| `example/notebooks/plant_helixseek_*/.gitignore` | config | static | `example/notebooks/finetune_NER_task/.gitignore` | exact |

## Pattern Assignments

### `dnallm/utils/genomic_coords.py` (utility, transform)

**Analog:** `dnallm/utils/sequence.py` (git-tracked, verified)

Pure-stdlib module of small typed functions with Google-style docstrings. Match this module's shape: module docstring with feature list, no heavy imports at module level, `snake_case` functions, PEP 604 unions, `ValueError` for invalid input. NOTE: unlike sequence.py (which imports `tqdm` at top), pyfastx is a **dev extra** — import it lazily inside `fetch_sequence` only (see Shared Patterns: Optional-dep guard).

**Module docstring pattern** (`dnallm/utils/sequence.py` lines 1-13):
```python
"""
Sequence utility functions for DNA sequence analysis and generation.

This module provides functions for:
- Calculating GC content
- Generating reverse complements
...

All functions are designed for use in DNA language modeling and
bioinformatics pipelines.
"""
```

**Function pattern** (lines 19-34 — summary + `Args:`/`Returns:` with untyped-param style):
```python
def calc_gc_content(seq: str) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        seq (str): DNA sequence (A/C/G/T/U/N, case-insensitive).

    Returns:
        float: GC content (0.0 ~ 1.0). Returns 0.0 if sequence is empty.
    """
    seq = seq.upper().replace("U", "T").replace("N", "")
    if len(seq) == 0:
        gc = 0.0
    ...
```

**Wiring:** re-export new public functions in `dnallm/utils/__init__.py` (follow the `from .sequence import (...)` block, lines 10-15, and extend `__all__`). Keep the module import-safe without pyfastx installed — `__init__.py` imports eagerly at package import.

RESEARCH.md Pattern 6 (lines 375-398) is the authoritative API sketch (`normalize_chrom`, `gff1_to_half_open`, `half_open_to_gff1`, `fetch_sequence`, `parse_gff_attributes`, `slice_gff_rows`) — copy that contract; the codebase analog supplies only style.

---

### `tests/utils/test_genomic_coords.py` (test, transform)

**Analog:** `tests/utils/test_sequence.py` (git-tracked, verified)

Plain module-level `test_*` functions (no class needed at this size), absolute imports from `dnallm.utils`, inline comments per assertion group. `pytest.raises(ValueError, match=...)` for the empty-result/wrong-chrom guards.

**Imports + test pattern** (`tests/utils/test_sequence.py` lines 1-29):
```python
import pytest

from dnallm.utils.sequence import (
    calc_gc_content,
    check_sequence,
    ...
)


def test_calc_gc_content():
    # Test normal DNA sequence
    assert calc_gc_content("ATGC") == 0.5
    ...
    # Test empty sequence
    assert calc_gc_content("") == 0.0
```

**Error-assert pattern** (`tests/models/test_model.py` lines 160-162):
```python
with pytest.raises(ValueError, match=r"Model test-model download failed."):
    download_model("test-model", mock_downloader, max_try=3)
```

**Fixtures:** tiny inline FASTA string + inline GFF3 rows (trailing-`;` CDS row included — RESEARCH Pitfall 4); if files are preferred, put them under `tests/utils/fixtures/` — `.fas`/`.gff`/`.gff3` are NOT gitignored (root `.gitignore` only ignores `*.fa`/`*.fasta`/`*.gz`/etc., verified lines 68-84).

---

### `dnallm/models/model_info.yaml` — append 2 finetuned entries (config, static data)

**Analog:** itself — three precedents, all git-tracked:

**Binary entry shape** (lines 164-172):
```yaml
finetuned:
  - name: "Plant DNABERT BPE promoter"
    model: "zhangtaolab/plant-dnabert-BPE-promoter"
    task:
      describe: "Predict whether a DNA sequence is a core promoter in plants by using Plant DNABERT model with BPE tokenizer."
      task_type: "binary"
      num_labels: 2
      label_names: ["Not promoter", "Core promoter"]
      threshold: 0.5
```

**Token-task entry with BILOU label list** (lines 1442-1449 — single-quoted inline list):
```yaml
  - name: "Plant NT singlebase tRNAPointer"
    model: "zhangtaolab/tRNAPointer"
    task:
      describe: "Predict of tRNA start and end sites in DNA sequence by using Plant NT model with singlebase tokenizer."
      task_type: "token"
      num_labels: 7
      label_names: ['O','B-Intron', 'I-Intron', 'B-tRNA', 'I-tRNA', 'B-anti', 'I-anti']
      threshold: 0.5
```

**`base_model:` field precedent** (lines 1449-1451, PlantCAD2 entries):
```yaml
  - name: "PlantCAD2 cell type specific acr small"
    model: "plantcad/cell_type_specific_acr_plantcad2_small"
    base_model: "kuleshov-group/PlantCAD2-Small-l24-d0768"
```

**Edit mechanics (verified this session):**
- The `finetuned:` section runs to EOF (last entry "PlantCAD2 ... translation large", line 1624). Append surgically as text — never parse-modify-dump (would reformat all 1,624 lines).
- **The file has NO trailing newline at EOF** (verified `tail -c | xxd`: ends `["off", "on"]` with no `\n`). The append must add the missing newline first or the first new line will fuse onto the last entry.
- The exact new-entry YAML bodies are pre-drafted in 06-RESEARCH.md "Code Examples" (lines 481-501) — use verbatim, with provenance comment per RESEARCH Pattern 2 step 4.
- Use `task_type: "binary"` / `"token"` (valid TaskConfig values) — some existing entries use `"classification"` which is NOT a valid enum value; do not copy that.
- This file is packaged data (`[tool.setuptools.package-data]`) with no runtime consumer (RESEARCH Pattern 1) — the edit cannot break runtime; tests are the validator.

---

### `scripts/showcase/freeze_registry.py` (script, file-I/O)

**Analog:** `scripts/feasibility/spike_families.py` (git-tracked, verified; CONTEXT names it "repo script skeleton precedent")

**Script skeleton** (lines 1-2, 36, 808+, 873-874):
```python
#!/usr/bin/env python3
"""Run GB10 feasibility spikes for the environment-gated example model families (FEAS-01)."""

import argparse
...
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
```
```python
def main() -> int:
    """Parse args, guard the GPU, run the requested families fail-closed.

    Returns:
        0 when ... 1 when ... 2 when the GPU guard failed.
    """
    parser = argparse.ArgumentParser(description=(...))
    ...
    args = parser.parse_args()
    ...
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
```

Copy from this analog: shebang, one-line requirement-ID docstring, `REPO_ROOT` anchor, argparse `main() -> int`, `sys.exit(main())`, distinct exit codes per failure class, machine-greppable `key=value` print lines (`_emit_evidence`, lines 52-88). For the checkpoint-config constraint asserts (placeholder `LABEL_i` pattern, `[17,512]` head) reuse the `_forward_pass`-style local-import + `hasattr` defensive probing (lines 182-220). No `sys.path.insert` hack — dnallm comes from the installed venv (contrast the anti-pattern at `example/notebooks/finetune_NER_task/generate_bpe_dataset.py:14`).

Freeze contract (what to write and from which source) is fully specified in 06-RESEARCH.md Pattern 2 (lines 273-277) + Code Examples (lines 466-501): upstream `LABEL_NAMES` is the semantic source; checkpoint config read proves constraints only.

---

### `scripts/showcase/select_loci.py` (script, file-I/O + batch)

**Primary analog (skeleton):** `scripts/feasibility/spike_families.py` — same shebang / `REPO_ROOT` / argparse `main() -> int` / `sys.exit` / evidence-line prints / `_gpu_guard()` fail-closed pattern (lines 91-112) / subprocess-with-`ruff: ignore` comments (lines 259-277) for the bedtools calls / `time.perf_counter()` timings.

**Domain analog (pyfastx + GFF3 parsing):** `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` (git-tracked, verified) — the only in-repo pyfastx + GFF3 consumer. Use as *domain reference only*, NOT as style model (it hardcodes `sys.path.insert(0, "/home/forrest/Github/DNALLM")` at line 14 — do not copy).

pyfastx + GFF3 reading (`generate_bpe_dataset.py` lines 44-80):
```python
from pyfastx import Fasta
...
genome = Fasta(genome_file)
...
with gzip.open("osa1_r7.all_models.gff3.gz", "rt") as infile:
    for line in tqdm(infile):
        if line.startswith("#") or line.startswith("\n"):
            continue
        info = line.strip().split("\t")
        chrom = info[0]
        datatype = info[2]
        start = int(info[3]) - 1   # GFF3 1-based -> 0-based
        end = int(info[4])
        strand = info[6]
        description = info[8].split(";")
```

Caveat: that attribute handling (`item[7:].split(',')[0]`) is exactly the naive split RESEARCH Pitfall 4 warns about — TAIR10's 197,160 trailing-`;` CDS rows need the tolerant `parse_gff_attributes` from `dnallm/utils/genomic_coords.py`. All coordinate math in select_loci.py must go through the SHOW-02 helper, not inline `-1`s.

Model verification stage goes through the dnallm public route only (`load_model_and_tokenizer` + tokenizer + forward) — import pattern from `spike_families.py` lines 330-338 (function-local `from dnallm.models import load_model_and_tokenizer`). Batch discipline: CRE bs=4 / Anno bs=1 (eager-attention ceiling, RESEARCH Pitfall 2).

Download stage (probe `~/Downloads/` first, then plantdhs.org zip with browser-UA retry, magic-byte validation, manual-placement instruction + non-zero exit on failure) has no direct in-repo analog — nearest is the retry-with-reason-classification loop in Shared Patterns below.

---

### `tests/models/test_plant_helixseek_registry.py` (test, file-I/O parse-only, fast leg)

**Analog:** `tests/examples/test_examples.py` — `test_yaml_validity` (lines 280-293):
```python
    @pytest.mark.skipif(not YAML_FILES, reason="No YAML files found")
    @pytest.mark.parametrize(
        "yaml_file",
        YAML_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_yaml_validity(self, yaml_file: Path):
        """Test that YAML config files parse correctly."""
        content = yaml_file.read_text(encoding="utf-8")
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(f"YAML error in {yaml_file.name}: {e}")
```

Copy: `yaml.safe_load` (NEVER `yaml.load`), `Path` + `read_text(encoding="utf-8")`, `pytest.fail` with filename context. The dir-anchor convention (`ids=lambda p: str(p.relative_to(EXAMPLE_DIR))`, from `tests/examples/_execution.py`'s `EXAMPLE_DIR`) applies if parametrizing over data files. Structure assertions beyond parse (fields present, 17-label order, no-network guarantee) per RESEARCH "Fast-leg registry structure test skeleton" (lines 538-558); locate the yaml via `Path(dnallm.__file__).parent / "models" / "model_info.yaml"` or `Path(__file__).parents[2] / ...` — both idioms exist in the repo (`tests/examples/_execution.py` uses the parents[N] form).

---

### `tests/models/test_plant_helixseek_smoke.py` (test, request-response, slow leg)

**Analogs:** `tests/models/test_model.py` lines 178-202 (slow + timeout real-download tests) and `tests/inference/test_inference_real_model.py` lines 22-80 (module-level marks + setUpClass skip).

**Slow + timeout marker pattern** (`tests/models/test_model.py` lines 178-187):
```python
    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_download_real_huggingface_connection(self):
        """Test real HuggingFace connection (requires network)."""
        from huggingface_hub import snapshot_download

        # Try to download a small test model
        result = download_model("microsoft/DialoGPT-small", snapshot_download, max_try=1)
        assert result is not None
        assert os.path.exists(result)
```

**Class-level marks + setup-skip pattern** (`tests/inference/test_inference_real_model.py` lines 22-24, 34-36, 70-73):
```python
@pytest.mark.slow
@pytest.mark.timeout(1800)
class TestRealModelInference(unittest.TestCase):
    ...
    @classmethod
    def setUpClass(cls):
        """Set up test class - load model and tokenizer once."""
        try:
            ...
        except ImportError as e:
            ...
            raise unittest.SkipTest(f"Required packages not available: {e}") from e
```

Prefer the plain-function + decorator form (test_model.py) over the unittest class for new code, but reuse "load once in setup, share across tests" if the 1.9 GB downloads serve multiple assertions. Timeout ladder verified in-repo: 900 (downloads) / 1800 (inference class) — RESEARCH recommends 1800 here (lines 339). Test bodies are pre-drafted in 06-RESEARCH.md "Slow-leg smoke skeleton" (lines 503-536): `TaskConfig(task_type=..., num_labels=..., label_names=..., threshold=0.5)` → `load_model_and_tokenizer(repo_id, cfg, source="modelscope")` → `assert model.config.id2label == dict(enumerate(LABELS))` → forward shape assert. Do not assert on stderr (benign runtime warnings documented, RESEARCH Pattern 3).

**Failure contract:** on transformers-5 failure the CONTEXT mandates one 4.57 fallback attempt then a typed skip — use the `environment-unavailable:` helper (Shared Patterns) and register the prefix if not already covered by `tests/expected_skips.yaml`.

---

### `example/notebooks/plant_helixseek_*/.gitignore` + `data/` artifacts (config + data)

**Analog:** `example/notebooks/finetune_NER_task/.gitignore` (git-tracked, verified, entire file):
```
*.gz
rice_annotation.bed
*.bed
*.fxi
rice_gene_ner.pkl
rice_gene_ner.token_sizes
```
and `example/marimo/benchmark/.gitignore`: `results/`

Copy the shape: a per-dir `.gitignore` listing generated/derived artifacts next to the notebook. For the showcase dirs the required entries differ (RESEARCH Pitfall 3, verified via `git check-ignore`):
- `.scratch/` must be listed explicitly — root patterns do NOT cover `.fas`/`.gff` copies inside it.
- Committed fragments use `.fas` (NOT `.fa`/`.fasta` — root `.gitignore` lines 69-70 ignore those globally; also `*.gz`, `*.zip`, `*.pkl`, `*.csv`, `*.tsv`, `*.fxi` are root-ignored, so nothing with those extensions can be committed without a negation).
- `.gff`, `.gff3`, `.md` are safe to commit as-is.

Layout: `example/notebooks/plant_helixseek_cre/data/`, `plant_helixseek_anno/data/`, plus a shared dir for the negative control + `selection.md` (RESEARCH Open Question 3, recommendation a — one source of truth).

## Shared Patterns

### Slow/fast leg split (slow marker + timeout)
**Source:** `tests/models/test_model.py:178-202`, `tests/inference/test_inference_real_model.py:22-24`, `tests/examples/test_notebook_execution.py:78-80`
**Apply to:** `tests/models/test_plant_helixseek_smoke.py` (slow leg) — every network-touching test gets `@pytest.mark.slow` + `@pytest.mark.timeout(1800)`; the registry structure test stays fast/unmarked.

### Typed-skip prefixes + expected_skips registration
**Source:** `tests/examples/_execution.py` lines 204-234:
```python
def environment_unavailable_skip(action: str, evidence: str) -> None:
    """Skip with the stable ``environment-unavailable:`` prefix. ..."""
    pytest.skip(f"environment-unavailable: {action} ({evidence})")


def optional_dep_skip(action: str, evidence: str) -> None:
    """Skip with the stable ``optional-dep:`` prefix. ..."""
    pytest.skip(f"optional-dep: {action} ({evidence})")
```
Registered in `tests/expected_skips.yaml` (lines 24-40: `prefix: "network-unavailable:"` / `"environment-unavailable:"` / `"optional-dep:"` entries with `category:`; unmatched skips fail CI via `scripts/audit_skips.py`).
**Apply to:** smoke test failure paths (REG-03 typed skip) and any pyfastx/GPU-missing guards in tests. Any NEW prefix must get an `expected_skips.yaml` entry in the same commit.

### Optional-dep lazy import (pyfastx is a dev extra)
**Source:** `dnallm/finetune/trainer.py` lines 48-50 (optuna guard):
```python
try:
    import optuna
```
**Apply to:** `dnallm/utils/genomic_coords.py` — import pyfastx inside `fetch_sequence` (function-local), never at module level; the utils package imports eagerly via `dnallm/utils/__init__.py`, and pyfastx is not a runtime dependency.

### YAML safety
**Source:** `tests/examples/test_examples.py` lines 288-293 (`yaml.safe_load` + `pytest.fail(f"YAML error in {file}: {e}")`).
**Apply to:** fast-leg registry test, freeze script's post-append verification, any config reads in select_loci.py. Never `yaml.load`.

### ValueError with matchable message
**Source:** `dnallm/models/model.py:375` (`raise ValueError(f"Model {model_name} download failed.")`); test side `tests/models/test_model.py:161` (`pytest.raises(ValueError, match=r"...")`).
**Apply to:** `genomic_coords.py` empty-result/unknown-chrom guards, select_loci.py bounds checks (locus coords vs Chr1 length 30,427,671), zip validation failure.

### Retry with reason classification + sleep between attempts
**Source:** `dnallm/models/model.py` lines 317-375 (`download_model`: `while True` / `cnt >= max_try` break / classify reason from exception text / `time.sleep(1)` / final `raise ValueError`), tested with `patch("time.sleep")` at `tests/models/test_model.py:160-176`.
**Apply to:** the plantdhs.org zip fetch in select_loci.py — browser-UA header + retry loop; on final failure print the manual-placement instruction into `.scratch/` and exit non-zero (CONTEXT contract). Distinguish exit codes like spike_families.py does.

### Script skeleton conventions
**Source:** `scripts/feasibility/spike_families.py` lines 1-2, 36, 808-874.
**Apply to:** both `scripts/showcase/*.py`: shebang, requirement-ID docstring, `REPO_ROOT = Path(__file__).resolve().parents[2]`, argparse, `main() -> int`, `sys.exit(main())`, `key=value` evidence prints. No `sys.path.insert` (anti-pattern at `example/notebooks/finetune_NER_task/generate_bpe_dataset.py:14`).

### Tree-clean discipline for the curation run
**Source:** `tests/examples/_execution.py` lines 172-199 (`assert_tree_clean`: scoped `git status --porcelain -- <paths>`, fails when the guard itself cannot run).
**Apply to:** select_loci.py — all intermediates under gitignored `.scratch/`; committed outputs land only in the designated `data/` dirs (RESEARCH Security V12).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (partial) plantdhs.org zip fetch + `~/Downloads/` probe-and-copy in `select_loci.py` | script | file-I/O | No in-repo HTTP-download-with-fallback-placement precedent; nearest patterns are the retry loop (`dnallm/models/model.py:317-375`) and subprocess handling (`spike_families.py`) — combine per RESEARCH Pattern 4 / Pitfall 6 |
| (partial) CRE bin-track / Anno BILOU decode math in `select_loci.py` | script | batch/transform | No in-repo analog (new domain logic); transcribe verbatim from upstream contracts quoted in 06-RESEARCH.md Pattern 5 + Code Examples lines 560-576 |

Everything else has a concrete in-repo analog above.

## Metadata

**Analog search scope:** `dnallm/utils/`, `dnallm/models/`, `dnallm/finetune/`, `tests/utils/`, `tests/models/`, `tests/inference/`, `tests/examples/`, `scripts/`, `scripts/feasibility/`, `example/notebooks/`, `example/marimo/`, root `.gitignore`, `tests/expected_skips.yaml`
**Tracked-source gate:** every analog path above verified via `git ls-files -- <path>` (all non-empty) on branch `phs`, 2026-10-02. No `.gsd/` capability-mirror or gitignored paths referenced.
**Verified load-bearing details:** `model_info.yaml` = 1,624 lines, `finetuned:` section runs to EOF, file lacks trailing newline (xxd-verified); root `.gitignore` ignores `*.fa`/`*.fasta`/`*.gz`/`*.zip`/`*.pkl`/`*.csv`/`*.tsv`/`*.fxi` (lines 68-84); `.fas`/`.gff`/`.gff3`/`.md` committable.
**Pattern extraction date:** 2026-10-02

---
*Phase: 6-Model Registry & Showcase Data Curation*
