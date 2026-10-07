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

## Milestone: v1.1 — Example Execution Testing & Repair

**Shipped:** 2026-10-07
**Phases:** 5 | **Plans:** 24 | **Sessions:** ~7 (2026-10-01 → 2026-10-07)

### What Was Built
- A private nbclient execution harness (tmp-sandbox cwd isolation, kernel-kill proof, partial-failure artifacts) executing the ENTIRE `example/` tree for real — final census **196 passed / 1 benign skip / 0 failed** on the nightly GPU runner
- A repair loop with teeth: 13+ repair classes (DNATokenizer unknown-char, np.fromstring binary-mode shim, allow_patterns passthrough, evo-1 giants safetensors-only fetch, MCP single-flight deadlock, …) each landing with a same-change regression test
- The PlantHelixSeek showcase: generic-registry loads, committed ≤200kb Arabidopsis loci with a selection.md frozen contract, and two executed notebooks reproducing every frozen metric exactly (jaccard=0.3247, exon_f1=0.7522, neg fractions), written back byte-identically into the docs mirror
- example-nightly CI: staged-serial job with a hard census collection gate (197/206 pin), ≥35Gi hygiene floors, ollama loopback systemd infra, a 24-row revision-pinned models.lock with a drift-injection-proven consistency guard
- Both v1 false-green CI gates closed (docs-validation masking removed, branch protection naming both contexts) and a written GB10 feasibility matrix with real-forward evidence

### What Worked
- Census before repair: full-tree real execution with class-tagged tracebacks ranked the Phase-8 queue by evidence, not anecdote
- Family-order rollout with per-repair full-census reconciliation — same root cause fixed the whole family, and regressions surfaced immediately (final census identical across the last two plans)
- Probe-then-execute gated lanes carrying live probe results in skip messages, proven in both directions — an ever-green skip is the same dishonesty class as a false-green gate
- Selection-time calibration: floors and tolerance bands frozen in selection.md and parsed by tests at startup — zero numeric literals in assertions
- The owner decision ledger (D-01..D-21) kept mid-flight policy changes (giants exit, model swap, num_ctx deferral) auditable instead of folkloric

### What Was Inefficient
- The stale-verification cascade repeated (v1 lesson 3): post-close review fixes re-staled four passed phases; convergence needed three parallel verifier regenerations — the fixpoint rule ("land all fixes, then regenerate verifiers with zero code changes between") was discovered mid-milestone rather than practiced from the start
- CR-01: a dropped `load_model_and_tokenizer` call let evo2 outputs ship as evo1 evidence — caught only at code review; content-contract tests now pin the load cell, but the class (committed outputs asserting an execution that did not happen) deserved a structure-test family from day one
- The census ratchet pin needed a manual re-pin after four post-close test additions (nightly went red exactly as designed, but the bump was manual friction)
- Mid-flight owner decisions (num_ctx deferral, qwen3.5:4b swap) left the in-repo ollama unit pin inert and the live-bind drift open — documented, but closing them requires owner sudo that has not happened yet

### Patterns Established
- Committed executed-notebook outputs are standing evidence (the owner's evidence model); re-execution proves them, structure tests pin them
- Giants tier: safetensors-only `allow_patterns` fetch outside every quota cache; environment-available lanes deselect (never skip) in CI
- Staged-serial nightly coexistence with hard hygiene floors between torch / MCP :8000 / ollama stages
- models.lock as the single id/prefix/revision contract, enforced against notebook `source=` routes by a fast-leg guard

### Key Lessons
1. Real execution surfaces an order of magnitude more truth than static suite testing — one milestone of execute-for-real produced 13 library/example repair classes the green suite never saw
2. Freeze metric contracts at calibration time (observed values + tolerance bands) and parse them at assert time — showcase claims become both honest and regression-proof
3. A hard census pin is a ratchet with documented bump points; let it fail loudly (it did) and treat the re-pin as a feature
4. Honesty is asymmetric by design: a gated lane must carry live evidence both when it skips and when it executes

### Cost Observations
- Model mix: not tracked this milestone
- Sessions: ~7 daily sessions over 6 calendar days (2026-10-01 → 2026-10-07)
- Notable: execution costs dominated (example-nightly stage-1 2:02–2:59 h; evo giants lane ~106 s); the largest non-execution cost was verification-regeneration cascades after review-fix cycles

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1 | ~6 | 4 | First GSD milestone on this repo: audit-first, gate-last wave discipline; behavior-first authoring rule |
| v1.1 | ~7 | 5 | Execute-for-real milestone: census→repair→gate loop; frozen-contract metrics; family rollout with per-repair census reconciliation |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1 | 1,657 (7 allowlisted skips) | 96.30% | 0 — no new test frameworks (constraint held) |
| v1.1 | fast lane 1,933 (1 allowlisted skip) + nightly example census 196/1S/0F | 96.42% | 0 — nbclient/marimo ride existing dev/test extras (constraint held) |

### Top Lessons (Verified Across Milestones)

1. *(cross-validated v1.1)* A gate without a red proof is a claim, not a gate — v1.1's census pin proved it again (nightly red on a stale pin → deliberate re-pin → green)
2. *(cross-validated v1.1)* Measured baseline → ranked worklist beats estimated planning — the execute-everything census ranked Phase 8's repair queue the same way the coverage worklist ranked Phase 3
3. *(seeded from v1.1, pending cross-validation)* Real execution is the only honest test of example code — green static suites hide an order of magnitude more defects than they expose
4. *(seeded from v1.1, pending cross-validation)* Freeze metric contracts (observed values + tolerance bands) at calibration time; parse them at assert time — never literals
5. *(seeded from v1.1, pending cross-validation)* Land all fixes first, then regenerate verifiers with zero code changes between — reactive stale-digest chasing cost more than the fixes themselves
