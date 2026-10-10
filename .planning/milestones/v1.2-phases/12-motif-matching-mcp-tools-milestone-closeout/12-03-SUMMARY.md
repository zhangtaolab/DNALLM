---
phase: 12-motif-matching-mcp-tools-milestone-closeout
plan: "03"
subsystem: docs + changelog (milestone closeout)
tags: [docs, ia3, peft, changelog, evidence-chain, coverage-expectation, census, closeout, rev-backfill]
requires:
  - phase: 12-motif-matching-mcp-tools-milestone-closeout
    provides: 12-01 (dnallm/interpret/motifs.py + tests, REV-10 entry) and
      12-02 (mcp/server.py tools + host/port fix, REV-11 entry) — verified by
      commit ancestry (07bd537, fb60af3 are ancestors of HEAD) before Task 1
provides:
  - Completed IA³ chapter at docs/user_guide/fine_tuning/peft_adapters.md
    (DOCS-01 split delivery closed; forward pointer gone)
  - Honest post-Phase-12 coverage expectation — measured 96.72% (2026-10-10)
    in docs/user_guide/continuous_integration.md + the pyproject
    [tool.coverage.report] comment (comment-only; fail_under=90 unchanged)
  - Finalized CHANGELOG ## [Unreleased] evidence chain — REV-01..REV-11 each
    SHA-linked to zhangtaolab/DNALLM commit URLs
  - The milestone-closeout verification record (this SUMMARY)
affects: [milestone-v1.2-closeout, rebuttal-letter-evidence-chain]
key-files:
  created:
    - .planning/phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md
  modified:
    - docs/user_guide/fine_tuning/peft_adapters.md
    - docs/user_guide/continuous_integration.md
    - pyproject.toml
    - CHANGELOG.md
decisions:
  - "REV->SHA links located by the plan's git log --grep mechanism: the lane
    closing commit that appended each CHANGELOG entry per D-09 (phases 10/12:
    also the fix commit; phase 11: the lanes batched entries into their
    slow-lane acceptance commits). REV-11 uses 0a4c7f5 per the 12-02 SUMMARY's
    explicit instruction — it overrides the grep's race-skewed hit (bd165d8
    absorbed the REV-11 line via the documented shared-append race). REV-10
    uses bd165d8 per the 12-01 SUMMARY."
  - "Coverage-expectation history preserved as dated provenance, not deleted:
    the docs now read 'measured 96.72% at the v1.2 closeout (2026-10-10; was
    96.30% at Phase 3)' so the number's movement is auditable (D-07 honesty)."
  - "The IA³ intro paragraph (line 5) also updated: 'IA³ adapters arrive with
    the next release' contradicted the shipped trainer branch — the Pitfall-8
    class error the task exists to remove. Minimal one-clause fix, recorded
    as a deviation (technically outside the section, inside the file's truth)."
metrics:
  duration: 34m
  completed: 2026-10-10
  tasks: 3
status: complete
actuals:
  tokens: 7059   # chars/4 over the realized diff f261bd2..HEAD (28,239 chars;
                 # estimate was 14000 — the docs lane needed no code spin-up)
  tasks: 3
  commits: 3
plan_head_before: f261bd275f72df807307874a29d1633ca1704354
plan_head_after: 32b28e2
commits: 3
---

# Phase 12 Plan 03: Milestone Closeout — Docs Chapter, CHANGELOG Evidence Chain, Coverage Honesty Summary

Milestone closeout executed in full: the IA³ chapter is real usage documentation written from the shipped Phase-11 trainer branch, the coverage-expectation docs carry the number measured in this plan (96.72%, not the stale 96.30%), every REV-01..REV-11 CHANGELOG entry is SHA-linked, the census pin verified exact, and the closing full fast lane is green — milestone v1.2 is closable fully green pending phase verification.

## Milestone-Closeout Verification Record

### Measured coverage (Task 2 — measured in this plan, 2026-10-10)

- **Command:** `uv run --no-sync coverage run -m pytest tests/ dnallm/mcp/tests -q -m "not slow"` then `uv run --no-sync coverage report`
- **Suite:** 2456 passed, 1 skipped, 75 deselected, 0 failed (exit 0)
- **Global total: 96.72%** (9586 stmts, 314 miss; exact 96.7244%, 2dp) — was 96.30%
- **Per-module rows (the >=96% milestone standard):**

| Module | Stmts | Miss | Cover |
|---|---|---|---|
| dnallm/interpret/__init__.py | 2 | 0 | 100% |
| dnallm/interpret/motifs.py | 374 | 2 | 99% (lines 386, 406) |
| dnallm/mcp/server.py | 796 | 19 | 98% |

- **Gate:** `uv run --no-sync coverage report --include="dnallm/interpret/*,dnallm/mcp/server.py" --fail-under=96` — exit 0. No per-module deficit; no escalation to lanes 12-01/12-02 needed (the plan's hard-stop rule never triggered).

### Census verification (D-07)

- ci.yml Stage 0.5 selector (`pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"`): **208/217 tests collected (9 deselected) — exactly the pin**. Re-pin NOT needed (as research predicted; the phase touched no example/ artifacts).
- Plan-literal note: `pytest tests/ --collect-only -q` reads `2473 tests collected` (whole suite) — the same not-literally-satisfiable wording 12-01 documented; the authoritative selector above is the actual CI gate and it matches exactly.

### Closing full fast lane (Task 3)

- **Command:** `uv run --no-sync pytest tests/ -q -m "not slow"` — **2404 passed, 1 skipped, 68 deselected, 0 failed** (exit 0)
- **Delta vs pre-phase 2253 baseline: +151** fast-lane tests (12-01's 91 interpret tests + 12-02's mcp tool tests minus marimo-side effects) — expected, per the plan.

### CHANGELOG evidence chain (REV-01..REV-11 — D-08/D-09)

| REV | Linked commit | Locating authority |
|---|---|---|
| REV-01 | [dae194a](https://github.com/zhangtaolab/DNALLM/commit/dae194ae112701c8d51a5a5e1a0c1f9c9ecf9791) | git log --grep (fix + entry same commit) |
| REV-02 | [58bbf41](https://github.com/zhangtaolab/DNALLM/commit/58bbf41e59b9a48834f4e6c28f760f277d53b3ae) | git log --grep (fix + entry same commit) |
| REV-03 | [36d0c74](https://github.com/zhangtaolab/DNALLM/commit/36d0c747ee136a7ff3871630974804964e86ce62) | git log --grep (fix + entry same commit) |
| REV-04 | [d4e9b68](https://github.com/zhangtaolab/DNALLM/commit/d4e9b6847d65ea589cdacb3ecf824ab652b7f1a5) | git log --grep (11-01 closing commit: entry + slow-lane acceptance; fast-lane IA³ branch itself is 3b644bd) |
| REV-05 | [d4e9b68](https://github.com/zhangtaolab/DNALLM/commit/d4e9b6847d65ea589cdacb3ecf824ab652b7f1a5) | git log --grep (same 11-01 closing commit; presets feature commit is e957e2c) |
| REV-06 | [295a970](https://github.com/zhangtaolab/DNALLM/commit/295a970ac26f8b7bb68af9094afa11dafdbd7a76) | git log --grep (11-02 closing commit; feature commits 87f9299/919a603) |
| REV-07 | [2b6f55e](https://github.com/zhangtaolab/DNALLM/commit/2b6f55e565f3bc64d5060351d8ffa95e20c517fb) | git log --grep (11-03 closing commit; probing feature commit 0cb0780) |
| REV-08 | [25862b4](https://github.com/zhangtaolab/DNALLM/commit/25862b4eac3bf2e23211c3502db49a48476ee0df) | git log --grep (11-05 closing commit; vep feature commits 9a0ddef/39eac44) |
| REV-09 | [3a719a7](https://github.com/zhangtaolab/DNALLM/commit/3a719a79493c5c8831231b5ecd98bea51991246f) | git log --grep (11-04 closing commit; sweep feature commits 9e3a427/99866ed) |
| REV-10 | [bd165d8](https://github.com/zhangtaolab/DNALLM/commit/bd165d8a2cdd48ab5c4c5add6505803688c22486) | 12-01 SUMMARY (= the grep-introducing commit; JASPAR client + entry same commit) |
| REV-11 | [0a4c7f5](https://github.com/zhangtaolab/DNALLM/commit/0a4c7f5b4005d853339c48c9dbf32143b2e080b8) | 12-02 SUMMARY explicit instruction (overrides the grep hit bd165d8 — the shared-append race put the REV-11 line into bd165d8's diff) |

Machine-verified: eleven tags present exactly once; every link resolves to an
ancestor of HEAD; append-only discipline proven (stripping the added links
from the diff leaves the eleven entry bodies byte-identical to HEAD). The
feature-commit column of alternatives is recorded above for rebuttal-letter
completeness — the links target the commits that carry each REV tag and each
CHANGELOG entry, per the plan's prescribed mechanism.

### Phase artifact rollup (what v1.2 closes with)

- `dnallm/interpret/` — FIMO-convention scanner + JASPAR client (12-01; motifs.py 99%)
- `dnallm/mcp/server.py` + `dnallm/mcp/start_server.py` — 3 new tools + host/port CLI-precedence fix (12-02; server.py 98%)
- `docs/user_guide/fine_tuning/peft_adapters.md` — completed IA³ chapter (12-03)
- `docs/user_guide/continuous_integration.md` + pyproject comment — honest 96.72% (12-03)
- `CHANGELOG.md` — complete REV-01..REV-11 SHA-linked evidence chain (appends by 12-01/12-02, backfill by 12-03)
- Suite: 2404 fast-lane passes (tests/) / 2456 (with packaged mcp root); census pin 208/217 intact; coverage 96.72% vs the 90 floor

## Task Summary

| Task | Name | Commit | Result |
|---|---|---|---|
| 1 | IA³ chapter — replace the forward-pointer stub with real usage docs | 8332913 | phrase gone; fields/rejections/log-line/save-reload verified against trainer.py + configs.py source; all 4 python fences ruff-format clean (line 100); docs snippet validation green; mkdocs zero warning delta |
| 2 | Measured coverage + honest docs update + census verify | 7460030 | 96.72% written to docs + pyproject comment (value 90 untouched); per-module gate exit 0; census exact 208/217 (no re-pin) |
| 3 | CHANGELOG SHA backfill REV-01..REV-11 + closing fast lane | 32b28e2 | eleven links, machine-verified append-only; fast lane 2404 passed / 0 failed |

## Pending Owner Input (carried from 12-01 — unchanged, still open)

The HBG1/BCL11A golden-test fixture remains owner-input-gated
(`tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml` `pending: true`).
Needed: Fig 4a window coordinates (locus/flank/assembly, verbatim), motif ID
(MA2324.1 vs MA2504.1 vs CIS-BP PWM), JASPAR release (2024 vs 2026), and the
tolerance policy (0 bp verbatim vs +-2 bp figure-derived). Activation is
fixture-files-only — the harness is committed, green, and generic. This is
the only acceptance item of the milestone that cannot close without owner
action; everything else is green.

## Deviations from Plan

### Auto-fixed / interpreted

**1. [Rule 3 - Blocking] Plan's gate command flag typo `--fail_under`**
- Found during: Task 2 verify. coverage 7.x accepts `--fail-under` (hyphen) only; the plan's `--fail_under` errors with "no such option".
- Fix: re-ran the gate verbatim otherwise: `coverage report --include="dnallm/interpret/*,dnallm/mcp/server.py" --fail-under=96` — exit 0.

**2. [Scope boundary - documented] `mkdocs build --strict` fails on 15 PRE-EXISTING warnings**
- Found during: Task 1 verify (the plan's fails_when includes "the strict build reports a ... warning").
- Issue: 14 broken relative links inside `docs/example/marimo/*` mirrors + one `user_guide/benchmark/advanced_techniques.md` link to `../../../CONTRIBUTING.md`. Proven pre-existing: rebuilt with the HEAD version of peft_adapters.md — identical 15 warnings. Zero delta from this plan's change; the page builds and renders.
- Resolution per the scope-boundary rule: NOT fixed here (marimo/benchmark files are outside this lane); logged to `deferred-items.md` in the phase directory. Task 1 verification evidence = zero warning delta + `validate_docs_snippets.py` green (all 352 blocks) + `check_docs_sync.py` green + all 4 python fences ruff-format clean — the same checks the docs-validation CI workflow runs (it has no mkdocs --strict step, so CI is not red today).

**3. [Rule 2 - Correctness] Stale intro clause updated beyond the IA³ section**
- Found during: Task 1. Line 5 of peft_adapters.md still said "IA³ adapters arrive with the next release" — the exact Pitfall-8 contradiction the task exists to remove; the acceptance criterion bars the phrase "coming in the next release" from the FILE.
- Fix: one-clause rewrite ("DNALLM supports LoRA, QLoRA, and IA³."). No LoRA/QLoRA chapter content touched (diff hunks: line 5 + the IA³ section only).

**4. [Plan-text interpretation] Census command shape**
- The plan-literal `pytest tests/ --collect-only` reads 2473 (whole suite); the actual ci.yml Stage 0.5 gate selector (`tests/examples` + `-m "not giants" -k "not mcp_example"`) reads exactly 208/217 (9 deselected). Same interpretation 12-01/12-02 recorded; the authoritative selector is the gate and it matches the pin.

### Escalations

None. No per-module coverage deficit (the Task 2 hard-stop never triggered), no red lane (fast lane green), no census re-pin cause.

## Auth Gates

None.

## Known Stubs

None — docs/changelog-only plan; no code paths introduced. (The owner-input-gated golden-fixture VALUES are 12-01's tracked pending item, reported above, not a stub of this plan.)

## Threat Surface

No surface beyond the plan's threat model. T-12-09 (Repudiation) mitigated as specified: every published number (96.72%, per-module rows, fast-lane counts, census triple) comes from the commands recorded above run in this plan; every CHANGELOG link machine-verified to resolve to a grep/SUMMARY-located commit; changes are append/update-only with the append-only discipline proven by link-stripped diff identity. T-12-SC honored: zero package installs.

## Commits

- 8332913 docs(12-03): complete the IA³ chapter from the shipped Phase-11 trainer branch
- 7460030 docs(12-03): honest post-Phase-12 coverage expectation — measured 96.72%
- 32b28e2 docs(12-03): CHANGELOG evidence-chain finalization — SHA backfill REV-01..REV-11

## Self-Check: PASSED

- Files: docs/user_guide/fine_tuning/peft_adapters.md, docs/user_guide/continuous_integration.md, pyproject.toml, CHANGELOG.md, deferred-items.md — all on disk post-commit.
- Commits: 8332913, 7460030, 32b28e2 all ancestors of HEAD (rev-list count from plan_head_before = 3, matching the pathspec-verified lane commits).
- Milestone invariants: `git diff f261bd2..32b28e2 -- pyproject.toml` = comment-only (fail_under=90 value unchanged, verified via tomllib parse); dnallm/__init__.py untouched across the plan; no dependency-list changes.
