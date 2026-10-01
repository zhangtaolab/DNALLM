# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1 — Test Suite Audit & Coverage Hardening

**Shipped:** 2026-10-01
**Phases:** 4 | **Plans:** 13 | **Sessions:** ~6 (2026-09-29 → 2026-10-01)

### What Was Built
- An honest test harness: single pytest config (`pyproject.toml` only), exit-code mask removed with a permanent CI canary, agreed 7-entry-omit coverage denominator
- A measured audit: 625-test census by skip reason, baseline **45.92%**, 43-file ranked gap worklist, cold/warm slow timings
- Suite hygiene: multiclass AUROC + CrossDNA dispatch bugs fixed and unskipped; every skip typed (`httpx.TransportError`) behind a fail-closed allowlist audit; PDF tests leave the tree clean
- ~1,000 new behavior tests across 5 ranked waves → coverage **96.30%** (7,133/7,407), pragma held at 3, 8 latent source bugs fixed en route
- A CI hard gate live in both directions: `fail_under=90` via pyproject, fast PR leg + self-hosted GPU nightly census, red-proven via probe PR #39 (78.91% < floor, 1,261 tests otherwise green), required-check branch protection on dev+main

### What Worked
- Strict dependency-order waves (harness → audit → fixes → tests → gate LAST) — the gate's first blocking day was its first green day, exactly as the research consensus prescribed
- Measured-baseline-driven authoring: the ranked gap worklist ordered Phase 3's five waves biggest-first (inference 1,505 → models 1,210 → mcp 449 → datahandling 441 → cli/compat), and each wave re-censused before the next was planned
- Behavior-first test authoring ("assert what it does" + latent-bug ledger instead of in-wave fixes) kept waves focused on coverage while still recording every real defect found
- The stale-digest verifier gate: every post-verification commit that touched covered source re-staled the phase digest and forced re-verification — it caught real drift (#4682 root cause) and kept all 4 phases verified at true HEAD
- GATE-04's synthetic-drop probe with zero residue gave the gate a falsifiable red proof rather than trust-me green

### What Was Inefficient
- Verification digests went stale repeatedly (fix commits after each code-review round re-staled already-passed phases) — phases 01/02 were re-verified twice; fix loops should be planned as first-class re-verification triggers, not discovered reactively
- CR-03 (nightly mamba leg deterministically red from a wrong extras install) was masked by `continue-on-error` and only caught at Phase 04 code review — a rehearsal run of changed CI jobs before merge would have caught it a day earlier
- The quick task that moved `test-mamba` to the self-hosted runner shipped and verified its work but never wrote its SUMMARY — the record gap surfaced only at milestone close, forcing an evidence hunt (live `gh run view`) after the fact
- ~31 info-tier review findings accumulated open across four dispositions; a periodic small-batch triage would have kept the ledger thinner than the end-of-milestone pile

### Patterns Established
- Typed, allowlisted, out-of-process skip enforcement (junit artifact → YAML allowlist → fail-closed audit script → CI step)
- Gate ratchets go live green: `fail_under` enabled only after the suite crosses the floor, never before
- Latent-bug pinning: tests pin current (buggy) behavior with an explicit deferred-items ledger entry, rather than blocking coverage waves on out-of-scope fixes
- Heavy GPU work rides nightly cadence (schedule/dispatch-only) on the self-hosted runner; PR/push stays the fast hosted leg; protected jobs are deep-compared byte-stable on every ci.yml touch
- Coverage numbers are cited against the config (denominator), not absolute counts, which wobble ±2 as source commits land

### Key Lessons
1. A coverage gate is only trustworthy with a red proof — probe PR #39 (all 1,261 tests green, only the floor red) is the artifact that makes "enforced" a true statement
2. `continue-on-error` in CI converts deterministic failure into silent green; treat every masked step as a suspected lie and rehearse changed jobs before relying on them
3. Regenerate verification digests after ANY commit touching covered source — post-verification "small" fixes are the top cause of stale trust
4. Convert guesses into ranked worklists before mass authoring: the measured baseline turned "write tests" into five ordered, individually-gated waves that crossed the target one wave early
5. Close out quick tasks when the work closes — SUMMARY-at-milestone-end means reconstructing evidence from live CI state instead of recording it while fresh

### Cost Observations
- Model mix: not tracked this milestone
- Sessions: ~6 working sessions over 3 calendar days (2026-09-29 → 2026-10-01)
- Notable: per-plan durations ran 6–80 min (median ~44 min); the two slowest plans (03-01 inference 80 min, 04-03 probe 52 min) were the ones touching real torch paths and live CI respectively

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1 | ~6 | 4 | First GSD milestone on this repo: audit-first, gate-last wave discipline; behavior-first authoring rule |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1 | 1,657 (7 allowlisted skips) | 96.30% | 0 — no new test frameworks (constraint held) |

### Top Lessons (Verified Across Milestones)

1. *(seeded from v1, pending cross-validation)* A gate without a red proof is a claim, not a gate
2. *(seeded from v1, pending cross-validation)* Measured baseline → ranked worklist beats estimated planning for coverage/QA work
