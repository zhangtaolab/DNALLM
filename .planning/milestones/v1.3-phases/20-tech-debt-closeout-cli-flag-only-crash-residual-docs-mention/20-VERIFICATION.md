---
phase: 20-tech-debt-closeout-cli-flag-only-crash-residual-docs-mention
verified: 2026-10-16T00:19:17Z
status: passed
score: 6/6 must-haves verified
covered_files:
  - .planning/phases/20-tech-debt-closeout-cli-flag-only-crash-residual-docs-mention/20-01-PLAN.md
  - .planning/phases/20-tech-debt-closeout-cli-flag-only-crash-residual-docs-mention/20-01-SUMMARY.md
  - dnallm/cli/cli.py
  - dnallm/cli/inference.py
  - dnallm/cli/train.py
  - docs/faq/models_troubleshooting.md
  - docs/user_guide/cli/usage.md
  - docs/user_guide/fine_tuning/task_guides.md
  - tests/cli/test_cli.py
  - tests/cli/test_config_required.py
  - tests/test_extras_guard.py
covered_digest: "v3:sha256:c2f198da15091ba93244333c17bba002d5e43dc15fdfd07724390ac177d9000c"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 20: Tech-debt closeout — CLI flag-only crash + residual docs mentions Verification Report

**Phase Goal:** The three audited v1.3 debt items closed in-repo — flag-only train/inference CLI invocations exit with an actionable --config-required usage error (no KeyError crash) with regression coverage and census discipline held; the faq [cuda]/[rocm] legacy mention marked not-current with a GPU-matrix pointer (guard re-pinned same-change); task_guides.md hf-route sites aligned to lock-pinned routes
**Verified:** 2026-10-16T00:19:17Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

All executor-reported facts were re-measured independently in this pass; no SUMMARY claim was taken as evidence. Every measurement below was run against the working tree at HEAD 53a1531 by the verifier itself.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All four train/inference CLI entry points (dnallm-train, dnallm-inference console scripts + `dnallm train` / `dnallm inference` subcommands) exit 2 with "--config is required" on any flag-only invocation; no KeyError 'task' traceback possible (D-01) | ✓ VERIFIED | Source read at all four sites (`dnallm/cli/train.py:37-45`, `dnallm/cli/inference.py:37-45`, `dnallm/cli/cli.py:47-55` train, `:82-90` inference): static f-string-free `click.UsageError` raised BEFORE the lazy core imports; old flag-only else branches fully deleted (0 `minimal_config`, 0 KeyError in `dnallm/cli/`). Behavioral: `tests/cli -q --no-cov` → **71 passed** (includes the 10-test contract); verifier's own CliRunner four-site probe → exit 2 + substring at all four, extracted messages byte-identical; real-argv probes `python -m dnallm.cli.train -m … -d … -o …` and `...cli.inference -m … -i …` → exit 2, full message, no traceback |
| 2 | `tests/cli/test_config_required.py` pins the behavior for all four entry points including core-never-constructed (D-02) | ✓ VERIFIED | File exists, 211 lines, 10 `def test_` across 5 classes; full-flags tests patch `dnallm.finetune.DNATrainer` / `dnallm.inference.DNAInference` and assert `assert_not_called()`; `_assert_no_traceback` pins no escaped exception; `TestConfigRequiredMessageConsistency` pins one byte-identical message across all four sites; suite green in the 71-passed run |
| 3 | Census triple (ci.yml:892-893) equals the freshly measured collection line — expected UNCHANGED at 216/217 (1 deselected) (D-02) | ✓ VERIFIED | Verifier re-measured: `pytest tests/examples --collect-only -q -m "not giants"` → `216/217 tests collected (1 deselected) in 0.87s`; ci.yml:892 grep pattern + :893 FAIL-message triple pin the same value; `git diff e08ed6d..HEAD -- .github/workflows/ci.yml` → empty (no edit needed, exactly as the plan's conditional required) |
| 4 | faq keeps v0.5 legacy context legible, no longer presents [cuda]/[rocm] as current installables, carries page-relative GPU Support Matrix pointer, guard sanctioned line re-pinned verbatim same change (D-03) | ✓ VERIFIED | `docs/faq/models_troubleshooting.md:13` (still single line, only pattern-matching line in the file): extras marked "legacy names no longer on the current install surface", v0.5 context retained, links `[GPU Support Matrix](../getting_started/gpu_support.md)`; AST `literal_eval` of the `SANCTIONED_CUDA121_ROCM_RESIDUE` frozenset entry vs file line 13 → **byte-for-byte identical: True**; `tests/test_extras_guard.py -q --no-cov` → **11 passed**; both files changed in the same commit 0655411 |
| 5 | task_guides.md: zero `source="huggingface"` sites; seven runnable-example sites flipped to `source="modelscope"` (D-04) | ✓ VERIFIED | `grep -n 'source="'` → exactly lines 65/147/222/296/369/445/522, all `modelscope`; `grep -c 'source="huggingface"'` → 0; phase diff shows the 14-line change is exactly 7 value-only `huggingface`→`modelscope` flips, no other prose touched |
| 6 | usage.md no longer claims the CLI builds a minimal dict / is "currently broken"; describes the --config-required exit (D-01 docs side) | ✓ VERIFIED | `grep -c 'currently broken'` → 0; `grep -ci 'minimal dict'` → 0; note at usage.md:30-35 now states flag-only invocation "exits with a usage error — `--config is required`" and points at `dnallm-model-config-generator` |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

### Decision Coverage

Gate result: 4/4 trackable CONTEXT.md decisions honored by shipped artifacts (D-01 through D-04; `not_honored: []`). Non-blocking by design.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/cli/test_config_required.py` | New regression test file, four-entry-point contract | ✓ VERIFIED | 211 lines, 10 tests, substantive, collected and passing |
| `dnallm/cli/cli.py`, `train.py`, `inference.py` | Flag-only branches replaced by usage error | ✓ VERIFIED | All four command functions restructured; raise precedes lazy imports; old branches deleted |
| `tests/cli/test_cli.py` | Superseded flag-only tests rewritten | ✓ VERIFIED | 10 tests across TestTrainCommand/TestInferenceCommand/TestStandaloneTrainModule/TestStandaloneInferenceModule renamed to exit-2 contract with core-not-called pins; boundary classes untouched |
| `docs/user_guide/cli/usage.md`, `docs/faq/models_troubleshooting.md`, `docs/user_guide/fine_tuning/task_guides.md` | Docs residue closed | ✓ VERIFIED | Truths 4-6 evidence |
| `tests/test_extras_guard.py` | SANCTIONED entry re-pinned, same change as faq edit | ✓ VERIFIED | Byte-identical pin proven by AST comparison; same commit 0655411 |
| `.github/workflows/ci.yml` | Edited ONLY IF measured census differs | ✓ VERIFIED (conditional not triggered) | Measured line equals the pinned triple; zero diff — the plan's own condition makes the non-edit the compliant outcome |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject.toml:266-267 console scripts | `dnallm.cli.train:main`, `dnallm.cli.inference:main` (+ group subcommands) | all four entry points carry the identical exit | ✓ WIRED | Entry points confirmed in pyproject; consistency test + verifier probe show byte-identical extracted message at all four |
| Usage-error message docs pointer + faq GPU-matrix pointer | Real docs pages | mkdocs --strict zero-warning gate | ✓ WIRED | `/tmp/mkdocs-venv19` mkdocs build --strict → exit 0, 0 WARNING lines |
| faq line 13 | `SANCTIONED_CUDA121_ROCM_RESIDUE` pin | byte-identity in same commit | ✓ WIRED | AST literal_eval comparison → identical; commit 0655411 contains both files |

### Data-Flow Trace (Level 4)

Not applicable — no dynamically rendered data. The CLI error path is static by design (threat model T-20-01 requires a non-interpolated message; verified f-string-free at all four sites), and the docs edits are static content whose resolution is proven by the mkdocs strict gate.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| tests/cli regression suite | `.venv/bin/python -m pytest tests/cli -q --no-cov` | 71 passed in 3.50s | ✓ PASS |
| Four-site exit-2 + substring + no KeyError | CliRunner probe of (cli train, cli inference, inf.main, tr.main) | all exit 2, substring present, exception is SystemExit (not KeyError), extracted messages identical | ✓ PASS |
| Real-argv full-flags standalone train | `.venv/bin/python -m dnallm.cli.train -m … -d … -o …` | exit 2, usage error, no traceback | ✓ PASS |
| Real-argv full-flags standalone inference | `.venv/bin/python -m dnallm.cli.inference -m … -i …` | exit 2, full message, no traceback | ✓ PASS |
| Census measurement | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" \| tail -2` | `216/217 tests collected (1 deselected) in 0.87s` | ✓ PASS |
| Extras guard | `.venv/bin/python -m pytest tests/test_extras_guard.py -q --no-cov` | 11 passed | ✓ PASS |
| faq↔guard byte identity | AST literal_eval of frozenset entry vs file line 13 | True | ✓ PASS |
| Docs sync gates | `python scripts/check_docs_sync.py && python scripts/check_notebook_md_sync.py` | both OK, DOCS-SYNC-GREEN | ✓ PASS |
| mkdocs strict | `/tmp/mkdocs-venv19` mkdocs build --strict | exit 0, 0 WARNING lines | ✓ PASS |
| Ruff | `.venv/bin/ruff check dnallm/cli/ tests/cli/` | All checks passed | ✓ PASS |
| click version (deviation-1 factual basis) | `import click` | 8.5.0 installed; pyproject pins `click<8.5.1` | ✓ CONFIRMED |

### Probe Execution

No phase-declared `probe-*.sh` scripts and no `scripts/*/tests/probe-*.sh` exist; this phase's probes are the pytest/CliRunner/grep commands above, all executed directly by the verifier.

### Requirements Coverage

None mapped — by design. ROADMAP Phase 20 carries "Requirements: None mapped — debt closeout scoped by the v1.3 milestone audit (option B)" and REQUIREMENTS.md contains no Phase 20 entries; PLAN frontmatter declares `requirements: []`. No orphaned requirements. Per the phase contract, missing REQ-IDs are not a gap.

### Test Quality Audit

| Test File | Linked Item | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|------------|--------|---------|----------|-----------------|---------|
| tests/cli/test_config_required.py | D-01/D-02 | 10 | 0 | no | Value/behavioral (exit codes, substrings, mock-not-called, message identity) | OK |
| tests/cli/test_cli.py (rewritten) | D-01/D-02 | 10 rewritten | 0 | no | Value/behavioral | OK |
| tests/test_extras_guard.py | D-03 | 11 | 0 | no | Value (byte-set equality) | OK |

Disabled-test scan (pytest.mark.skip / unittest.skip / skipif / xit / todo): zero matches. No circular expected-value generation (the faq pin is compared against the live file, not against itself).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none found | — | TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER/empty-impl scans returned zero matches across all nine phase files |

### Deviation Assessment (SUMMARY "Deviations from Plan")

**Deviation 1 — click ≥8.2 SystemExit adaptation of the no-traceback pin: LEGITIMATE.** Factual basis independently confirmed: on the installed click 8.5.0 (pyproject pins `click<8.5.1`, so CI resolves the same), the verifier's own probe shows `result.exception` is `SystemExit`, never `None`, for this handled exit — the plan's literal `result.exception is None` assertion is unsatisfiable. The adaptation (`is None or isinstance(SystemExit)`) preserves the pinned invariant exactly: the pre-fix `KeyError('task')` would surface as a KeyError in `result.exception` and fail the pin on any click version. Documented, minimal, intent-preserving, and version-robust.

**Deviation 2 — Task 2 acceptance-criterion grep over-match resolved in favor of the scope-fence prohibition: LEGITIMATE.** AC3's literal text (`grep -c "are required when not using --config"` returns 0 for all three files) is unreachable without editing the benchmark (cli.py:138) and mutagenesis (cli.py:182) guards, which the plan's own must-have prohibition and task action text forbid touching. The verifier confirmed: 0 occurrences in train.py and inference.py; in cli.py the only two occurrences are the two out-of-scope boundary commands (untouched per the diff-hunk audit — zero changed lines under TestBenchmarkCommand/TestMutagenesisSubcommand and no benchmark/mutagenesis lines in the cli.py diff). The criterion's intent — superseded train/inference guard text gone — is satisfied. Resolving a plan-level acceptance grep in favor of the stronger goal-level prohibition is the correct precedence, not a gap.

### Human Verification Required

None — infrastructure/debt-closeout phase. Every acceptance criterion is command-verifiable and was verified programmatically in this pass (exit codes, message substrings, census count, byte identity, grep counts, gate scripts, strict build). No truth is left present-but-behavior-unverified: each behavior-dependent truth (CLI exit contract, core-never-constructed) has a passing behavioral test executed by the verifier, plus direct real-argv probes.

### Gaps Summary

No gaps. All six must-have truths verified with independently re-measured evidence; all artifacts present, substantive, and wired; all three key links verified; all five prohibitions held (expected_skips.yaml and ci.yml zero-diff across e08ed6d..HEAD; benchmark/mutagenesis code and tests untouched; both sync scripts green; no owner-gate/push-filter/config surfaces in the phase diff — exactly the 9 planned files plus the SUMMARY changed). The two documented deviations were assessed and both preserve the plan's intent. Census discipline held exactly as the plan's conditional required. Milestone closeout items (owner human-gates #25/#26/#27, push-filter removal) remain with the owner by explicit scope-fence design — out of scope for this phase, not gaps.

---

_Verified: 2026-10-16T00:19:17Z_
_Verifier: Claude (gsd-verifier)_
