---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 04
subsystem: inference
tags: [vep, rev-08, zero-shot, kernels, testing]
requires:
  - tests/conftest.py fixtures (simple_dna_tokenizer, tiny_real_model/tiny_model_factory)
provides:
  - dnallm.inference.vep.align_variant (same-slot evaluability rule — R1-3e-1 protocol answer)
  - dnallm.inference.vep.VariantAlignment (frozen result dataclass with skip_reason)
  - dnallm.inference.vep.clm_log_likelihood (full-sequence causal kernel, formula in docstring)
  - dnallm.inference.vep.mlm_slot_log_prob (masked-slot kernel, formula in docstring)
  - dnallm.inference.vep.get_model_device (module helper)
affects: []
tech-stack:
  added: []
  patterns:
    - mutagenesis kernel adaptation-with-attribution (clm_evaluate/mlm_evaluate math mirrored into pure functions)
    - skip-as-data (structured VariantAlignment records instead of exceptions)
key-files:
  created:
    - dnallm/inference/vep.py
    - tests/inference/test_vep.py
  modified: []
decisions:
  - Kernels adapted self-contained inside vep.py with docstring attribution to the mutagenesis kernels (planner discretion resolved per no-refactors constraint)
  - Multi-slot skip path proven via a checksum stub tokenizer — char-level vocabularies can never produce a multi-slot difference, so the countermeasure test needed a context-sensitive tokenization
  - Module-scope coverage proof run via `coverage run --include` after the plan's literal `--cov=dnallm.inference.vep` form tripped a pre-existing venv-wide numpy/coverage interaction (see Deviations)
metrics:
  duration: 15 min (2026-10-09T10:50:39Z to ~11:05Z)
  completed: 2026-10-09
  tests-added: 19 (all fast-lane, zero skips)
  coverage: 100% (62/62 statements, dnallm/inference/vep.py)
status: complete
actuals:
  tokens: 5733      # chars/4 over the realized diff of dnallm/inference/vep.py + tests/inference/test_vep.py
  tasks: 3
  commits: 3        # plan-attributable: 93bd52f, d782c7c, 65ba867 (+ this docs commit)
  commits_note: single-tree wave concurrency — `git rev-list --count d3097d6..HEAD` measures 9 because sibling agents' commits interleave on the shared branch
plan_head_before: d3097d67ca5cda52183f7a1ece6894ef9ecdf9bc
plan_head_after: 65ba86787c7202e57c08d504ea5ff52c46e52f49
---

# Phase 10 Plan 04: VEP Core Kernels Summary

Same-slot variant evaluability rule (`align_variant`) plus CLM/MLM scoring kernels with formula docstrings — the REV-08 protocol core that Phase 11 B5 assembles `evaluate_vcf`/CLI/ClinVar on top of.

## What Was Done

### Task 1 — align_variant same-slot rule (tracer, commit 93bd52f)

- `dnallm/inference/vep.py` created: module docstring declaring the zero-shot VEP protocol (same-slot rule as the R1-3e-1 answer; `evaluate_vcf`/CLI explicitly deferred to the next phase), `VariantAlignment` frozen dataclass, `align_variant`.
- Input contract: `pos` is 0-based; `sequence[pos:pos+len(ref)] != ref` raises matchable `ValueError("... does not match ...")` — distinct from skips.
- Skip-as-data: `length-changing allele` / `no change` / `multi-slot token difference` return `evaluatable=False` + reason + `None` fields, never raise.
- Shared tokenizer convention `tokenizer(s, return_tensors="pt", add_special_tokens=True)` in a `_tokenize_ids` helper (also accepts list-returning stubs).
- `TestAlignVariant` (9 tests): mid-sequence SNP, the PITFALLS exactly-one-differing-slot guard comparing FULL id lists (not a spot-check), all three skip paths, ref-mismatch ValueError, boundary positions (pos 0 and last), frozen-dataclass immutability, multi-slot skip via a `_ChecksumTokenizer` stub (char vocabs can never produce multi-slot diffs).
- Tracer feedback gate re-ran the verify end-to-end post-commit: 9 passed.

### Task 2 — CLM/MLM kernels (commit d782c7c)

- `get_model_device` mirroring `mutagenesis.py:241-255` exactly (`.device` attr → `parameters()` → CPU fallback).
- `clm_log_likelihood`: `@torch.no_grad()`, one forward, shifted log_softmax gather sum — same math as `Mutagenesis.clm_evaluate` (mutagenesis.py:311-347). Docstring carries the formula `log P(sequence) = sum_t log P(token_t | tokens_<t)` and the delta-log-likelihood paradigm statement.
- `mlm_slot_log_prob`: `@torch.no_grad()`, clone input_ids, mask one slot, log_softmax at that slot — same math as `Mutagenesis.mlm_evaluate` (mutagenesis.py:257-309). Docstring carries `log P(token_id | masked context)` and the log-odds paradigm statement (ids from `align_variant`).
- `TestClmLogLikelihood` (3) + `TestMlmSlotLogProb` (4) on the real tiny per-position model: finite `<= 0` floats, bit-identical determinism, ref/alt discrimination (competing token ids score differently), masked-input clone check via forward spy (only the target slot becomes MASK).

### Task 3 — Coverage completion + boundary proof (commit 65ba867)

- `TestGetModelDevice` closes the two residual branches (`.device`-attr object, plain-object CPU fallback).
- Module coverage: **100%** (62/62 statements) from fast-lane tests only — bar was >=96%.
- Boundary audit green: zero old-terminology hits in vep.py (born with "DNA large language models" per D-08), `dnallm/inference/__init__.py` untouched, public surface exactly `VariantAlignment` / `align_variant` / `clm_log_likelihood` / `mlm_slot_log_prob` (+ `get_model_device`); no `evaluate_vcf`, no CLI, no VCF parsing (grep hits are docstring boundary declarations only).

## Verification

- `uv run --no-sync pytest tests/inference/test_vep.py -q` — 19 passed, 0 skipped, no network.
- `uv run --no-sync pytest tests/inference/ -q` — 367 passed (no regressions in the inference dir).
- `uv run --no-sync python scripts/check_code.py` — all required checks passed (ruff format/check clean on both files; mypy emitted one pre-existing informational error inside numpy's own stub, unrelated to this plan).
- Coverage row: `dnallm/inference/vep.py 62 0 100%` (proof command variant below).
- Plan-level file boundary: only `dnallm/inference/vep.py` and `tests/inference/test_vep.py` touched by this plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Off-by-one in a hand-written test constant**
- **Found during:** Task 1 verify (first run)
- **Issue:** The exactly-one-slot test hardcoded alt sequence `"ACGTAGCA"` for pos=3 of `"ACGTTGCA"` — that replaces the T at index 4, not 3 (correct alt is `"ACGATGCA"`). The kernel was right; the test constant was wrong.
- **Fix:** Derive the alt sequence from the pos arithmetic (`sequence[:3] + "A" + sequence[4:]`) with an explicit expected-value assert. Verify then passed 9/9.
- **Files modified:** tests/inference/test_vep.py
- **Commit:** 93bd52f (fixed before the task commit)

### Plan-Execution Deviations

**2. Coverage proof command substituted (environmental, pre-existing)**
- The plan's literal proof `pytest tests/inference/test_vep.py --cov=dnallm.inference.vep --cov-report=term -q` fails in the current shared venv during conftest import: coverage's `source_pkgs` resolution imports `dnallm` under an active tracer, and that chain (utils shims → torch → numpy 2.5.3) trips numpy's C-extension "cannot load module more than once per process" guard. Proven pre-existing and repo-wide: the identical failure reproduces on the untouched `tests/inference/test_mutagenesis.py --cov=dnallm.inference.mutagenesis`, and a minimal `coverage.Coverage(source=["dnallm"]).start(); import numpy` reproduces it. Nothing was installed or changed by this plan.
- **Substitution:** `uv run --no-sync coverage run --include="dnallm/inference/vep.py" -m pytest tests/inference/test_vep.py -q` + `coverage report --include=...` — same measurement target, same instrument (coverage 7.16.2), no package import at coverage start. Result: 62/62 statements, 100%.
- Recorded in the Windows ledger as a deviation (CI legs pin numpy 1.26.4/2.2.0 and are unaffected).

**3. Minor implementation cleanups (no semantic change)**
- `mlm_slot_log_prob` uses a single defensive `.clone()` (the mutagenesis original double-clones; one clone already guarantees the source encoding is untouched — proven by the clone-spy test).
- The plan cites "PITFALLS #5" for slot misalignment; the PITFALLS.md heading is Pitfall 6 (REV-08). Substance identical; the guard test cites the pitfall by name.

## Commits

| Task | Commit | Subject |
| ---- | ------ | ------- |
| 1 (tracer) | 93bd52f | feat(10-04): add align_variant same-slot evaluability rule |
| 2 | d782c7c | feat(10-04): add CLM/MLM scoring kernels with formula docstrings |
| 3 | 65ba867 | test(10-04): complete vep.py coverage to 100% with device-resolution tests |

Auth gates: none. Authentication was never required (offline fast-lane only).

## Known Stubs

None. No placeholder logic, no unwired data sources, no skips added.

## Self-Check: PASSED

- dnallm/inference/vep.py — FOUND
- tests/inference/test_vep.py — FOUND
- 93bd52f, d782c7c, 65ba867 — all ancestors of HEAD
