---
phase: 03-test-authoring-to-90-coverage
plan: 02
subsystem: testing
tags: [pytest, coverage, torch, models, fault-injection, sys-modules-stubs]

requires:
  - phase: 03-test-authoring-to-90-coverage
    provides: wave-1 inference closeout (176/250), shared conftest fixtures, coverage-wave1-missing.txt re-ranked worklist
  - phase: 01-harness-integrity-measured-baseline
    provides: measured baseline + coverage tooling of record
provides:
  - Models area closed from 1,209 missing to 88 (gate ≤ 240 — passed with 152 lines of slack; research residual estimate was ~185)
  - 317 new behavior tests (dispatch/retry/fallback fault-injection matrices, real-torch heads/losses/CrossDNA, sys.modules-stubbed evo/borzoi/lucaone handler bodies)
  - coverage-wave2-missing.txt — re-ranked worklist input for wave 3
  - Suite 79.90% (5,912 / 7,399) at wave-2 census, up from 64.75% post-wave-1
affects: [03-test-authoring-to-90-coverage, 04-coverage-gate-ci]

actuals:
  tokens: 43700  # chars/4 over the realized diff (174,770 chars / 4)
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Sentinel dispatch matrix: patch the ONE handler under test to a (.to()-self-returning) sentinel pair, decline the rest, guard skipped stages with side_effect=AssertionError"
    - "Real-torch wrapper testing: tiny nn.Module backbones behind patched AutoModel.from_config so autograd/pooling/loss semantics are real"
    - "sys.modules stubbing via monkeypatch.setitem only (18 stub sites, zero direct assignment — grep-gated) for absent special-family deps"
    - "Retry classification pinned by downloader call_count AND sleep call_count with time.sleep patched on every retry-path test (9 occurrences)"

key-files:
  created:
    - tests/models/test_tokenizer.py
    - tests/models/test_head.py
    - tests/models/test_losses.py
    - tests/models/test_special/test_crossdna.py
    - tests/models/test_special/test_evo.py
    - tests/models/test_special/test_family_handlers.py
    - .planning/phases/03-test-authoring-to-90-coverage/coverage-wave2-missing.txt
  modified:
    - tests/models/test_model.py

key-decisions:
  - "DNALLMforSequenceClassification covered in test_model.py with tiny real torch backbones behind patched AutoModel.from_config — the class is ~200 of model.py's 233 missing lines and the wave gate is unreachable without it"
  - "Evo timebox did NOT fire: EvoTokenizerWrapper needs no stubs at all, and the evo2/evo1 handler stub shapes (evo2, vortex.model.tokenizer, evo, stripedhyena.*) were satisfiable within one task — evo.py lands at 99% (1 missing line)"
  - "Latent cosine_similarity loss crash documented with a pytest.raises(TypeError) test and recorded in the windows ledger instead of fixed — the intended semantics are ambiguous and this phase's bug-fix scope is coverage-limited"

patterns-established:
  - "ExitStack patcher lists for wide dispatch tests (patch families + gates + loader guards in one block)"
  - "TinyCrossDNAForMaskedLM stand-in: a real nn.Module base class for testing generated subclasses of remote dynamic classes"

requirements-completed: [TEST-01]

coverage:
  - id: D1
    description: "tests/models/test_model.py extended 51 -> 163 collected tests: six early-return family selection rows, gpn/omnidna import-gate rows (ImportError + stubbed fall-through), generic fall-through, mutbert/basenji2 post-processing, quantization device-skip, plus the completed retry matrix (404 / revision-reset asserting revision=None on call 2 / incomplete-loop zero-sleep / exhaustion call+sleep counts)"
    requirement: TEST-01
    verification:
      - kind: unit
        ref: "tests/models/test_model.py (163 collected, all passing; 9 time.sleep-patched retry tests)"
        status: pass
      - kind: command
        ref: ".venv/bin/python -m pytest tests/models/test_model.py -q (163 passed) and fast leg 1010 passed"
        status: pass
    human_judgment: false
  - id: D2
    description: "tests/models/test_tokenizer.py (30), test_head.py (32), test_losses.py (10): staged-failure fallback matrix asserting which tier served, DNAOneHotTokenizer real surface, seven real-torch head classes with forward+gradient assertions, FocalLoss hand-computed values"
    requirement: TEST-01
    verification:
      - kind: unit
        ref: "tests/models/test_tokenizer.py + test_head.py + test_losses.py (72 collected, all passing)"
        status: pass
    human_judgment: false
  - id: D3
    description: "tests/models/test_special/{test_crossdna.py (66), test_evo.py (30), test_family_handlers.py (37)}: real-torch CrossDNA generated classifier, stub-reached evo2/evo1 bodies (evo timebox verdict: did not fire), grouped family coverage for the remaining ten handlers"
    requirement: TEST-01
    verification:
      - kind: unit
        ref: "tests/models/test_special/ (133 collected, all passing; monkeypatch.setitem only — grep gate clean)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Wave-2 gate: full census 1236 passed / 7 allowlisted skips / audit exit 0; models-area missing sum 88 <= 240; coverage-wave2-missing.txt committed; twice-run idempotency + tree-clean verified"
    requirement: TEST-01
    verification:
      - kind: command
        ref: "pytest --junitxml --cov full census (exit 0) -> coverage json -> sum(missing_lines over dnallm/models/) = 88"
        status: pass
      - kind: command
        ref: "scripts/audit_skips.py /tmp/p3-02-junit.xml tests/expected_skips.yaml (exit 0)"
        status: pass
    human_judgment: false

duration: 46 min
completed: 2026-09-30
status: complete
commits: 3
plan_head_before: ec5987831a11e5423df2687b96c2901574b43f26
plan_head_after: 1f984148545a4df97ee8ddedabffa9c7f9fbfb7c
---

# Phase 3 Plan 2: Models Wave — Test Authoring Summary

**317 fault-injection and real-torch behavior tests closing the models area from 1,209 to 88 missing lines (gate ≤ 240) via sentinel dispatch matrices, staged-failure tokenizer fallbacks, seven real head forwards, and sys.modules-stubbed special-family handler bodies**

## Performance

- **Duration:** 46 min (incl. 14-min full census)
- **Started:** 2026-09-30T09:50:22Z
- **Completed:** 2026-09-30T10:36:50Z
- **Tasks:** 3/3
- **Files modified:** 8 (7 new test files, 1 extended, plus the wave artifact)

## Post-Wave Measurement (full census, both roots, slow included)

- **Census:** 1236 passed / 7 skipped (all allowlisted) / 0 failed / exit 0 — 856s
- **Suite coverage:** **79.90%** (5,912 covered / 2,487 missing on 7,399 stmts) — was 64.75% after wave 1
- **Models-area missing sum: 88 (gate ≤ 240 — passed with 152 lines of slack; research residual estimate ~185)**
  - borzoi 42 · crossdna 14 · model.py 10 · megadna 10 · dnabert2 4 · head 3 · evo/lucaone/omnidna/space/tokenizer 1 each · losses/enformer/gpn/mutbert/basenji2/modeling_auto/__init__ 0
- `scripts/audit_skips.py` exit 0 on the census junit; zero new skips introduced
- Fast leg at every task boundary: 898 → 1010 → 1010 → 1010 passed, exit 0 each time (plus 368 models tests twice-run green for idempotency)
- Pragma budget: still exactly 3 occurrences under `dnallm/`; `pyproject.toml` coverage/pytest config untouched; tree clean of pdf/log artifacts after every run

## Accomplishments

- Dispatch sentinel matrix over `load_model_and_tokenizer`: per-family selection for all six early-return handlers, gpn/omnidna import-availability gates proven both ways (raw ImportError without the dep; stubbed str + generic fall-through with it), guarded first-resolved-wins fall-through, mutbert/basenji2 tokenizer post-processing identity, and the quantization path that skips `.to(device)` in favor of the bnb fix
- Retry/reason-classification matrix completed with call-count AND sleep-count assertions on every row: 404 branch (single call, zero sleeps), no-revision reset (call 2 receives revision=None), incomplete-status loop (max_try exhaustion with ZERO sleeps), connection exhaustion (3 calls / 3 sleeps)
- Tokenizer three-tier fallback chain pinned per tier (sentinel identity for tier 1/2, isinstance + warning for tier 3) plus the full real DNAOneHotTokenizer surface (vocab, padding sides, embeds, persistence round-trips)
- Seven head classes run real torch forwards with differentiability asserts; FocalLoss verified against independently recomputed terms for every reduction
- DNALLMforSequenceClassification covered end-to-end on tiny real backbones: all init branches (default/megadna/evo/lucaone), from_base_model weight diffusion, classifier + pooling selection, all five sentence-embedding strategies, and the full loss-selection matrix including every named loss_function string
- CrossDNA covered real-torch: generated classifier on an in-repo MLM stand-in (id-remap window, four pooling modes with masked-row zeros, three problem-type losses, tuple outputs), checkpoint scanner matrix (8.1M subdir, single/multiple children), registration contract, handler dispatch
- Special families: evo2/evo1 handler bodies reached with shaped sys.modules stubs (capability-suffix config selection, revision 1.1_fix branch, head_config wrapping); megadna checkpoint-file selection; borzoi both task branches; enformer/space dispatch against the vendored (coverage-omitted) models; dnabert2 triton-file restore verified by content

## Task Commits

1. **Task 1: dispatch sentinel matrix + retry classification (tracer)** — `ecefa71` (test)
2. **Task 2: tokenizer fallback + heads + losses** — `94ad06f` (test)
3. **Task 3: special families + wave-2 re-measure** — `1f98414` (test + artifact)

**Plan metadata:** this commit (docs)

## Files Created/Modified

- `tests/models/test_model.py` — 51 → 163 collected tests (extended)
- `tests/models/test_tokenizer.py` — NEW, 30 tests
- `tests/models/test_head.py` — NEW, 32 tests
- `tests/models/test_losses.py` — NEW, 10 tests
- `tests/models/test_special/test_crossdna.py` — NEW, 66 tests
- `tests/models/test_special/test_evo.py` — NEW, 30 tests
- `tests/models/test_special/test_family_handlers.py` — NEW, 37 tests
- `.planning/phases/03-test-authoring-to-90-coverage/coverage-wave2-missing.txt` — re-ranked worklist for wave 3

## Decisions Made

- DNALLMforSequenceClassification tests live in test_model.py (the class lives there) even though the plan's Task 1 action text focused on dispatch/retry — the wave gate is arithmetically unreachable without covering the wrapper's ~200 missing lines, so this is objective-required work, not scope creep
- Evo stubbing stayed inside its timebox: EvoTokenizerWrapper (no absent deps) carries most of evo.py's lines, and the four-handler stub shapes were satisfiable directly — no residual needed banking into models-core
- Latent bugs recorded rather than fixed (bug-fix scope): cosine_similarity loss constructs CosineEmbeddingLoss but calls it with two args (TypeError for every user selecting it — covered by an explicit pytest.raises test and the windows ledger)
- No source files were modified this wave; all observed behavior was correct except the two documented latent quirks

## Deviations from Plan

### Verify-command adjustment (no code impact)

Same as wave 1: pytest 9.1.1's `--collect-only -q` emits `<Function ...>` lines without `::` separators, so the plan's `grep -c "::"` tripwires always return 0. Equivalent criteria enforced via the "N tests collected" summary lines and cross-checked with per-file runs: test_model.py 163 ≥ 70; test_tokenizer/test_head/test_losses 30/32/10 ≥ 10 each.

### Latent-bug documentation (no auto-fix, scope discipline)

**1. [Recorded - Latent bug] `cosine_similarity` loss selection crashes at call time**
- **Found during:** Task 1 (loss_function matrix)
- **Issue:** `loss_fct = nn.CosineEmbeddingLoss(**kwargs)` is then called as `loss_fct(logits, labels)` — the loss needs a third `target` argument, so every user configuring loss_function="cosine_similarity" hits TypeError
- **Handling:** Covered with `test_forward_cosine_similarity_loss_crashes` (pytest.raises(TypeError)); NOT fixed because the intended target semantics (labels as ±1 targets vs embeddings) are ambiguous and this phase's bug-fix scope is coverage-limited; recorded in the windows ledger
- **Files:** tests/models/test_model.py

**2. [Recorded - Latent quirk] `DNAOneHotTokenizer.decode` crashes on 2D tensors**
- **Found during:** Task 2
- **Issue:** decode(tensor([[...]])) subscripts id_to_token with a list (unhashable) — only 1D tensors/lists decode; batch_decode handles 2D correctly
- **Handling:** Test uses the supported 1D form; quirk documented here

---

**Total deviations:** 1 verify-command adjustment + 2 latent-bug records (0 source fixes required)
**Impact on plan:** None — no scope creep, no new dependencies, pragma budget intact at 3, allowlist untouched.

## Evo Timebox Verdict (Task 3 requirement)

**The timebox did NOT fire.** EvoTokenizerWrapper imports nothing exotic and was covered directly; the evo2 (evo2, vortex.model.tokenizer) and evo1 (evo, stripedhyena.{utils,model,tokenizer}) stub shapes satisfied every attribute read the handlers perform within a fraction of one task's effort. The evo-specific residual ledger is therefore **explicitly empty** — evo.py: 194 statements, 1 missing line (86, the redundant else-arm in padding target selection).

## Accepted-Uncovered Residual Ledger (models area, documented — never pragma'd)

| File | Missing | Lines | Justification |
|------|---------|-------|---------------|
| dnallm/models/special/borzoi.py | 42 | 30, 59-68, 72, 76-77, 89-140 | BorzoiForSequenceClassification `__init__`/`forward` bodies execute only against a real Borzoi base (from_pretrained is the stub boundary); the remaining lines ARE the class body |
| dnallm/models/special/crossdna.py | 14 | 75-76, 169, 382, 520-521, 547-557, 571-572 | OSError listdir guard, config.dropout default fallback, bnb kwargs paths, and the `if not hasattr(config, ...)` default stanzas (every real CrossDNA config already ships these attrs) |
| dnallm/models/model.py | 10 | 138, 207, 229, 291-297, 768, 951-952 | custom_head dict-branch (dead: head_config is always a dict), megadna long-cast guard, dict-output elif (unreachable behind the isinstance check), encoder/decoder hidden-state fallbacks, TaskConfig head_config else, bnb weight-copy failure warn |
| dnallm/models/special/megadna.py | 10 | 26, 80-105, 136-137 | extra kwarg, DNATokenizer tokenizer-API methods (save_vocabulary etc.), ImportError wrap (torch.load failures are not ImportError) |
| dnallm/models/special/dnabert2.py | 4 | 29-31, 33 | triton TypeError arm — installed triton's probe returns True on this box, so the disable branch's inner False path never runs |
| dnallm/models/head.py | 3 | 353-354, 400 | UNet skip-size mismatch padding (needs non-power-of-2 lengths that survive pooling), MegaDNA batch>1 repeat corner |
| dnallm/models/special/{evo,lucaone,omnidna,space}.py | 4 | 86 / 18 / 15 / 19 | one line each: redundant padding else-arm, extra-kwarg arms |
| dnallm/models/tokenizer.py | 1 | 141 | return_dict=False + return_inputs_embeds without dict return |

## Known Stubs

None — every new test asserts observable behavior (identities, call counts, shapes, gradients, values); no placeholder logic introduced.

## Issues Encountered

None beyond the deviations above. The full census ran clean on the first post-authoring attempt (1236/0/7). All sys.modules stubs restored between tests (18 monkeypatch.setitem sites, grep-gated; twice-run idempotency green).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Wave 2 (models) complete at 88/240; wave 3 executor should start from `coverage-wave2-missing.txt` — the mcp area (server.py 250 missing at wave-1 census, now ~247 after suite-wide growth) is next by rank
- Suite at 79.90%; gap to the 90.5% target = 770 lines (2,487 missing − 717 allowed at the 7,399-statement denominator)
- Suite runtime: fast leg ~82s (was 78s), census 856s (was 914s — the models tests are cheap and the trainer leg ran slightly faster); no CI timeout risk
- Windows ledger: one new latent-bug residual this wave (cosine_similarity loss TypeError), joining wave-1's two

---
*Phase: 03-test-authoring-to-90-coverage*
*Completed: 2026-09-30*

## Self-Check: PASSED

All 8 created files exist on disk; all three task commits (ecefa71, 94ad06f, 1f98414) present in history; commits measured from the plan ledger (ec59878 → 1f98414 = 3).
