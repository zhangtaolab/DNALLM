---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: "02"
plan_id: 11-02-random-init-baselines
subsystem: models
tags: [random-init, baselines, model-loading, from-config, reproducibility]
requires:
  - "transformers AutoConfig.from_pretrained + Auto*.from_config (installed 5.17.0, span >=4.49,<6)"
  - "modelscope Auto* wrappers for the ms route (existing core dep)"
  - "models.lock rows: zhangtaolab/plant-dnabert-BPE, zhangtaolab/plant-dnamamba-BPE-open_chromatin (both pre-pinned, delta none)"
provides:
  - "load_model_and_tokenizer(..., random_init: bool = False, random_init_seed: int = 42)"
  - "RANDOM_INIT_SUPPORTED_FAMILIES frozenset (D-07) + _gate_random_init / _detect_special_family"
  - "_load_random_init_model (from_config-only path), _get_auto_modules_for_source (download-free Auto* bundle), _tensor_digest (sha256[:10] raw bytes), _log_random_init_fingerprint (D-05 banner + per-tensor table)"
  - "TestRandomInit: 38 fast-lane + 3 slow-lane tests (BASE-01 proofs)"
affects:
  - "dnallm/models/model.py (sole-owner lane file; random branch slots before the _get_model_path_and_imports weight-download seam)"
  - "README.md Supported Models section (from-scratch baselines paragraph)"
  - "CHANGELOG.md [Unreleased] Added (REV-06, R2-5)"
tech-stack:
  added: []  # zero new dependencies (plan invariant held)
  patterns:
    - "AutoConfig.from_pretrusted + Auto*.from_config no-checkpoint instantiation (verified transformers 5.17.0 auto_factory.py:206-233)"
    - "per-tensor sha256[:10] fingerprint over detach->cpu->contiguous->numpy->tobytes with bf16 upcast"
    - "storage-identity (data_ptr) tied-alias detection instead of digest-collision heuristics"
key-files:
  created:
    - .planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-02-SUMMARY.md
  modified:
    - dnallm/models/model.py
    - tests/models/test_model.py
    - README.md
    - CHANGELOG.md
decisions:
  - "D-05 honored: banner + per-tensor hash table via get_logger INFO lines (model.py convention), not trainer print style and no sidecar file"
  - "D-06 honored: two proven slow-lane architectures — generic BERT family (plant-dnabert-BPE, ms) + mamba allowlist member (plant-dnamamba-BPE-open_chromatin, trust_remote_code from_config branch)"
  - "D-07 honored: module-level RANDOM_INIT_SUPPORTED_FAMILIES frozenset seeded with 'mamba' (documents the sanctioned remote-code member; gates special families only, generic Auto* always allowed)"
  - "no-download proof scopes to the weight-fetch seam (_get_model_path_and_imports) — the AutoConfig config.json + tokenizer fetches are explicitly allowed and documented in code + tests"
metrics:
  duration: 18.1 minutes
  completed: 2026-10-09
  tests_added: 41
actuals:
  tokens: 13900   # chars/4 over realized diff: ~53.5k on exclusive files + README paragraph + CHANGELOG bullet
  tasks: 3
  commits: 3      # lane-attributed: 87f9299, 919a603, 295a970 (ledger..HEAD range also contains sibling-lane commits by design in the shared tree)
plan_head_before: 0cb07808c8793522908121bace90924727bd273c
plan_head_after: 295a970
status: complete
---

# Phase 11 Plan 02: Random-Init Baselines Summary

One-liner: `load_model_and_tokenizer(..., random_init=True)` builds provably
from-scratch models via AutoConfig + Auto*.from_config — loud "randomly
initialized" banner, per-tensor sha256 hash table, CPU-canonical seeding,
zero weight downloads, and a special-family allowlist (BASE-01 / REV-06).

## What Was Built

**dnallm/models/model.py** (sole-owner lane):
- `random_init: bool = False` + `random_init_seed: int = 42` kwargs on
  `load_model_and_tokenizer` (documented Args/Raises per Google style).
- `RANDOM_INIT_SUPPORTED_FAMILIES: frozenset[str] = frozenset({"mamba"})`
  (D-07) + `_SPECIAL_FAMILY_MARKERS` name detection mirroring the handlers'
  own matching; `_gate_random_init` raises a matchable ValueError naming the
  family and the sorted allowlist BEFORE any `_handle_*` runs (D-06).
- `_load_random_init_model`: config.json-only fetch
  (`AutoConfig.from_pretrained(trust_remote_code=True)`), head shaping via
  config attributes (num_labels/id2label/label2id/problem_type — A4),
  `torch.manual_seed(seed)` BEFORE construction, `Auto*.from_config` only
  (the same class the task-type path selects), model stays on CPU through
  hashing and moves to device only afterwards (Pitfall 4b).
- `_get_auto_modules_for_source`: the import half of
  `_get_model_path_and_imports` with NO snapshot download — transformers
  classes for local/hf, modelscope wrappers for the ms route.
- `_tensor_digest` (sha256[:10] over detach->cpu->contiguous bytes, bf16
  upcast) + `_log_random_init_fingerprint` (banner + one INFO line per
  parameter tensor with `remove_duplicate=False` so tied aliases are visible,
  plus non-parameter buffer rows) — D-05 via get_logger.
- Tokenizer parity preserved on the random path: normal
  `load_tokenizer_with_fallback` plus the mutbert/basenji2 post-processing,
  then the shared padding/device tail.

**tests/models/test_model.py** — `TestRandomInit`, 38 fast-lane tests:
tracer end-to-end on a mocked boundary, 11-case off-list family battery,
quantization/head_config rejections, task-type Auto* selection (8 cases),
config-attribute head shaping, no-download side_effect guard, same-seed
full-table reproducibility + different-seed divergence, seed-before-init
ordering (manual_seed recorder vs from_config call order), digest byte-level
semantics (manual hashlib check, bf16, int/bool, contiguous views),
tied-alias/buffer table contents, param-shadowing buffer skip, empty-config
edge on a REAL tiny local BertForMaskedLM (network-free),
`_get_auto_modules_for_source` bundles. Plus 3 slow-lane tests:
- generic BERT member `zhangtaolab/plant-dnabert-BPE` (ms): banner,
  >=100-tensor table, same-seed reload identity, tokenizer, CPU forward.
- mamba member `zhangtaolab/plant-dnamamba-BPE-open_chromatin` (ms):
  trust_remote_code from_config branch, same assertions.
- per-tensor pretrained-vs-random difference: every float parameter digest
  differs; exceptions enumerated with counts via storage-identity detection —
  exactly two BERT tie pairs (decoder.weight<->word_embeddings.weight,
  decoder.bias<->predictions.bias) and exactly two matching non-float
  buffers (position_ids, token_type_ids); zero non-float parameters.

**Docs**: README from-scratch baselines paragraph inside `## 🧬 Supported
Models` (contains `RANDOM_INIT_SUPPORTED_FAMILIES`); CHANGELOG bullet
`(REV-06, R2-5)` under `## [Unreleased]` / `### Added` (D-09 append
discipline, sibling entries untouched).

## Verification Results

| Check | Command | Result |
|---|---|---|
| Task 1+2 fast lane | `uv run --no-sync pytest tests/models/test_model.py -q -k "random_init" -m "not slow"` | 32 passed |
| Full file fast lane | `... -m "not slow"` | 205 passed, 0 failed |
| Task 3 slow lane | `... -k "random_init" -m slow` | 3 passed (real modelscope loads, both models cached) |
| Coverage (cov-crash workaround) | `coverage run -m pytest tests/models/test_model.py -q` + `coverage report --include="dnallm/models/model.py"` | 97% (>= 96% bar) |
| Docs greps | `grep -c RANDOM_INIT_SUPPORTED_FAMILIES README.md` / `grep -c "(REV-06," CHANGELOG.md` | 1 / 1, B2-DOCS-OK |
| Invariants | `git diff <base>..<my-head> -- pyproject.toml dnallm/__init__.py` | empty; no facade re-exports; no new deps |
| models.lock | both slow-lane models pre-pinned | delta none, as planned |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added `random_init_seed` kwarg (default 42)**
- Found during: Task 1. The must-have "two random_init loads with the
  identical seed ... identical per-tensor hash tables" requires a seed
  control knob on the public surface; TaskConfig carries no seed field.
- Fix: `random_init_seed: int = RANDOM_INIT_DEFAULT_SEED (=42)` kwarg,
  documented in the docstring ("documented default" per plan wording).
- Files: dnallm/models/model.py. Commit: 87f9299.

**2. [Rule 2 - Correctness guard] random_init x quantization_config and x head_config rejections**
- Found during: Task 1. from_config has no quantization/head_config
  equivalent; letting transformers raise would leak foreign exception
  strings (version-span rule) or silently ignore the options.
- Fix: both combinations raise dnallm's own matchable ValueError inside
  `_gate_random_init`, with tests.
- Files: dnallm/models/model.py, tests/models/test_model.py. Commit: 87f9299.

**3. [Shared-tree race - documented, no content harm] README paragraph committed via lane 11-05's commit 39eac44**
- During Task 3, lane 11-05's pathspec commit (`feat(11-05): ... + README
  protocol`) landed while my README paragraph was uncommitted in the shared
  working tree, sweeping my hunk into their commit. Content verified
  correct and confined to `## 🧬 Supported Models` at HEAD
  (`grep -c RANDOM_INIT_SUPPORTED_FAMILIES README.md` == 1); README
  working-tree diff vs HEAD is empty. Rewriting their commit is prohibited;
  attribution-only impact, recorded here for the verifier.

### Test-side corrections during development (not plan deviations)

- Classification task types need num_labels in the Auto*-selection
  parametrization (`_safe_num_labels` contract).
- Tied-alias detection initially used digest collisions; fresh-init
  LayerNorm zero/one tensors legitimately collide across DIFFERENT tensors,
  so the proof uses storage identity (`data_ptr`) — BERT ties exactly two
  pairs, enumerated and asserted.
- BERT's matching non-float buffers are exactly two (position_ids,
  token_type_ids), counted in the difference test.

## Auth Gates

None.

## Known Stubs

None — all functionality is real and wired; the slow-lane typed
`network-unavailable:` skips follow the repo's sanctioned pattern (fires
only when a model is uncached AND modelscope.cn is unreachable).

## Threat Flags

None — no security-relevant surface beyond the plan's threat model; both
registered threats are mitigated as planned (T-11-04 no-download proof via
the weight-fetch seam; T-11-05 allowlist gate before any handler runs).

## Self-Check: PASSED

- dnallm/models/model.py modified, RANDOM_INIT_SUPPORTED_FAMILIES defined and used (grep count 6).
- tests/models/test_model.py TestRandomInit present (41 tests: 38 fast + 3 slow).
- Commits 87f9299, 919a603, 295a970 all ancestors of HEAD.
- README paragraph + CHANGELOG bullet verified at HEAD via greps.
