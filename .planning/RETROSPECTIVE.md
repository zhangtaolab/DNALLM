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

## Milestone: v1.2 — Paper Revision Suite Support

**Shipped:** 2026-10-10
**Phases:** 3 | **Plans:** 12 | **Sessions:** ~3 (2026-10-09 → 2026-10-10)

### What Was Built
- The eleven reviewer capabilities (REV-01..REV-11) behind the paper revision: eval-semantics leak guard, metric-name registry, IA³ + 35-family PEFT presets, random-init baselines with per-tensor hash proof, frozen-embedding probing, zero-shot VEP from ClinVar VCFs (scikit-allel, the milestone's only sanctioned dep), n-guarded multi-seed aggregation, FIMO-convention motif scanning with a stdlib JASPAR client, and three MCP analysis tools closing the v1.1 `--host/--port` audit item
- A milestone-long honesty chain: CHANGELOG REV-ID→SHA evidence links (integration-checker verified all), two owner-adjudicated overrides recorded in VERIFICATION frontmatter rather than silently scoped down, and a GitHub-tracked golden-fixture deferral (#44) whose activation is fixture-files-only
- 145 commits / 168 files (+26.5k/−1.4k) in 2 calendar days via three sequential waves of file-disjoint parallel agents, all phases verified at a zero-code-change fixpoint

### What Worked
- The owner-fixed wave strategy compressed a 3-phase milestone to ~2 days without a single cross-agent file collision (pathspec commits after the one Wave-1 shared-index lesson, c751df6)
- Frontmatter overrides as the owner-decision channel: both adjudications (VEP magnitude, Fig 4a deferral) carried reasons, timestamps, and a reopen-free activation path — the audit consumed them mechanically
- Trace-origin forensics in test failures (notebook-import test surfacing `raised in <file>:<line>`) turned an 8-iteration CI dependency-drift hunt (pyarrow<26, pydantic-ai silent backtrack to 1.22.0) into a same-day fix with the diagnostic left in the tree
- Fixpoint verification regeneration (the v1.1 lesson, practiced this time): re-verify stale phases only after ALL code lands, zero changes between

### What Was Inefficient
- The extras-guard same-change rule was violated once (guard expectation drifted behind two pin changes) — caught by self-audit, fixed same-commit-after; a pre-commit-time guard sync check would have caught it at edit time
- Two attribution-only commit races (README riding a sibling's commit) despite pathspec discipline on content — `git log --follow` archaeology was needed at close; a post-wave attribution audit is now cheap enough to run per-wave
- The audit acknowledge writer cannot anchor GFM table rows in deferred-items.md (reader/writer shape drift) — the Phase 10 terminology table needed a content-preserving bullet reshape at close; an upstream fix or a house "never tables in deferred-items" convention would remove the friction
- Golden-fixture inputs were requested too late in the run (first surfaced at verification); asking at plan time would have given the owner a week of slack instead of a close-time decision

### Patterns Established
- REV-ID→commit SHA links in CHANGELOG as the standing rebuttal-letter evidence chain
- Owner-input-gated acceptances ship as pending-manifest fixtures (harness green on synthetic stand-ins, activation fixture-files-only, GitHub-tracked) instead of blocking phase close
- Metric emissions validate names against the registry at every emit (`validate_emission`) — name-authority contract, compute stays native

### Key Lessons
1. Owner-input-gated acceptances are a scheduling problem, not a verification problem — sequence the input request at plan time, ship the harness, and the deferral becomes a one-issue footnote
2. Dependency floors need the same review as ceilings: a silent resolver backtrack (pydantic-ai <3 → broken 1.22.0) reds CI exactly like an upper-bound break, and only fails LOUD once you floor at the dev-green series
3. Small-model evaluation honesty beats anchor-matching: recording `within_anchor_range: false` with the same-scale anchor comparison preserved both the criterion and the truth

### Cost Observations
- Model mix: not tracked this milestone
- Sessions: ~3 daily sessions over 2 calendar days (2026-10-09 → 2026-10-10); 145 commits
- Notable: verification-regeneration cascades from v1.1 avoided by practicing the fixpoint rule; the dominant unplanned cost was CI dependency-drift diagnosis (~8 iterations) before the trace-origin forensics tool existed

---

## Milestone: v1.3 — Platform Reach & Packaging Honesty

**Shipped:** 2026-10-16
**Phases:** 8 (13–20, incl. post-audit option-B phase 20) | **Plans:** 22

### What Was Built
Ollama surface clean exit with census re-pin; GAP-1 zero-rung disposition; Windows numpy parity core + dispatch gates + frozen pairs; never-uploading publish chain (event gate + OIDC + smoke); atomic extras surgery (full alias / cuda121 retired / rocm removed / triton-windows marker); 149-file docs accuracy ledger with cuda130 default + strict zero warnings; first-ever cuda130 CI leg; nightly interpreter-honesty fix; CLI --config-required closeout.

### What Worked
- Async-job manifest discipline (16-04 precedent) carried two nightly cycles and a fix cycle without losing state; A.5 classify-before-retry turned a confusing red into a root-caused fix (bdf6ba3) instead of a flaky retry loop
- One-census-move-per-phase discipline held across all 8 phases — the triple never drifted once
- The verifier gate repeatedly earned its keep: numeric_claim_authority caught a 6-vs-7 site miscount before execution; moved_files pinpointed exactly which phase-20 edits re-staled 18/19
- Owner human-gates halted cleanly (17-03/18-03 pattern) — nothing improvised past a supply-chain or hardware gate

### What Was Inefficient
- Nightly wall-clock dominated the close (two serialized trios ≈ 34 h for one plan); single-runner serialization is the structural cost
- phase.complete mechanically flipped owner-deferred requirements to Complete (caught + reverted by hand) — a tooling gap worth an upstream fix
- The unpinned-uv-venv drift class was latent for 8 days before firing — a pin-from-day-one rule on persistent runners would have avoided the red entirely

### Patterns Established
- Interpreter-label honesty on persistent self-hosted runners: pin `uv venv --python "$pythonLocation/bin/python"`; never key a wheelhouse cache without the cpXY tag
- moved_files-driven incremental re-verification (stale ≠ redo: re-verify the moved set, re-issue the fingerprint)
- Deferred-items Resolutions prose (scanner-safe resolution records)

### Key Lessons
- A repo-caused red that dies in 15 ms with no output is still fully diagnosable from per-step timings + the artifacts on the box — classify before retry, always
- Fresh goal-backward re-verification is cheap (minutes) versus override bookkeeping; prefer re-verify at every fixpoint

### Cost Observations
- ~2 days wall (10-11 → 10-16) for 8 phases incl. two ~15 h nightly cycles; execution heavily subagent-parallel with a lean orchestrator
- Notable: the milestone closed with zero Co-Authored-By trailers (owner directive, verified per commit)

## Milestone: v1.4 — Pre-flight UX & Registry-as-Truth-Source

**Shipped:** 2026-10-22
**Phases:** 5 (22–26) | **Plans:** 23 (+6 quick tasks)

### What Was Built
Registry-as-truth-source: cached read-only typed loader as sole parser of `model_info.yaml` (236 entries after the kwt +15 fix and the tta task-type audit), tokenizer/max-length enrichment from measured hub metadata, `load_preset_model` registry rewire. Pure pre-flight core (`validate_data` / `check_compatibility` / `probe_availability` / `generate_config`/`edit_config`) — offline-safe, probes-never-raise. The `dnallm preflight` Textual wizard (env → data → suitability → validated config → optional ModelScope-first download) behind a new `tui` extra, with the `base` alias retired. MCP consistency checker (warn-first, strict CI gate over committed configs) + honest 236-entry lab-first listing.

### What Worked
- Loader-before-consumers / core-before-shells dependency order: all 7 integration seams and 3 E2E flows verified live at audit with zero seam rework
- Pure-core-first made the TUI a thin consumer: copy gaps (25-11..13) closed without touching verdict logic; structure guards pin the "presentation only" contract two-way
- Honest delegated verification: 25-UAT recorded the copy read-through as model-verified-not-human-verified; the audit carried it as the standing owner item instead of a gap
- The pre-close artifact audit earned its keep: 4 open items fixed for real in one commit (aed99ce) → verified_closeout instead of acknowledge-and-carry
- Biologist-copy discipline (glossaries for all 12 ML terms + all 8 task types, verbatim YAML keys) turned UAT failures into three small gap plans, not a rewrite

### What Was Inefficient
- Phase 25 needed 13 plans incl. 3 gap closures — the 25-11 contract was narrower than its UAT truth (3 task types glossed of 8); a completeness test in the original plan would have saved a debug session + 2 gap plans
- kwt's SUMMARY frontmatter carried unquoted `model: "<repo>"` — invalid YAML that scanners silently degraded to "unknown" for 2 days; parse-verify planning artifacts when writing them
- WINDOWS row 28 was hand-appended to the table without the fenced-JSON half, blocking `windows append` from 2026-10-20 until the close

### Patterns Established
- Read-only-forever registries: maintenance = scripted insertion, never a YAML round-trip (structure guards CI-red on write APIs / second parsers)
- Unavailability is a state, not an exception: probes degrade to UNKNOWN with machine-readable reasons; WARN never blocks; lab-first (`zhangtaolab/*`) ordering everywhere
- PEP 562 lazy facades for instant CLI/TUI start on CPU-only machines (3.71 s → 0.29 s)
- Thread-worker seam lands before panels; Pilot-functional tests over snapshot/golden infra

### Key Lessons
- A phase can implement its plan verbatim and still fail UAT when the contract is narrower than the truth — assert completeness (all eight values), not presence (the three written)
- Planning-artifact frontmatter that fails to parse fails silently into "unknown" status — the audit scanner is only as honest as the YAML it can read

### Cost Observations
- 3 days wall (10-20 → 10-22), 228 commits, 170 files (+38,911/−328) — fastest plan-per-day rate to date despite a full TUI and the UI safety gate
- Notable: heaviest single phase (25: 13 plans) absorbed its own gap closures plus a retroactive UI-SPEC without reopening

## Milestone: v1.4 reopen — Compatibility Census & Remediation (Phases 27–28)

**Shipped:** 2026-10-30
**Phases:** 4 (27, 27.1, 27.2, 28) | **Plans:** 19

### What Was Built
- Per-model transformers-4/5 compatibility census over the 49 pretrained entries, import-level evidence, typed registry verdicts, generated plain-language docs behind a two-way-proven drift gate
- Transformers-5 remediation: four measured compat rungs (GENERanno/DNABERT-2/GENA-LM/MutBERT), exact pip guidance for version-locked externals, honest split evidence for structurally-broken forwards
- One support vocabulary module feeding docs, TUI, and MCP listings; curated deepspeed/flash-attn extras with guided interception and a dedicated nightly extras CI lane
- IN-01..IN-06 tech-debt closure; UAT cross-milestone sweep closing 9 stale/deferred items + numpy `<2.6` ceiling

### What Worked
- Census discipline ("a disposition flips only on a measured green probe") held across three phases — zero unverified registry claims at close
- The reopen mechanism: archive untouched, inserted phases carried the line to a clean second close with the audit re-run covering all four phases
- Integration re-audit as close gate: 11/11 seams verified with live byte-identical regeneration proof

### What Was Inefficient
- DISP-01 churn: three owner touchpoints (option-a push → GH006 rejection → defer-dispatch) where the default-branch registration constraint could have been surfaced at plan time
- The close-time CLI under-counted the reopen (saw only phases holding unchecked boxes) — post-close manual corrections needed for phase dirs, MILESTONES numbers, and closeout facts (a close-after-reopen edge case worth a GSD fix)
- The 2026-10-30 UAT audit misjudged one item (mkdocs strict "partially fixed") on grep evidence alone — planner-level live verification corrected it; lesson: grep is not a verdict

### Patterns Established
- Owner-deferred operational steps ride the acknowledge mechanism with self-invalidation (artifact change resurfaces them) — no silent debt
- Support words derive from one module; surfaces never hard-code them
- Extras follow the mamba precedent: curated, version-banded, measured-green-only flips

### Key Lessons
- Dispatchable workflows register from the default branch only — plan the registration step relative to branch protection at design time, not at dispatch time
- A cross-phase UAT audit pays for itself: 14 outstanding items → 9 closable in one day, several already-fixed-but-unrecorded
- Replanning halted on evidence (the marimo mirror case) prevents destructive execution of a stale decision

### Cost Observations
- 3 days reopen wall (10-27 → 10-30), 313 commits over the original close
- Model mix: opus planners / sonnet executors; 5 quick runs + 2 audit orchestrations same-day at close

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1 | ~6 | 4 | First GSD milestone on this repo: audit-first, gate-last wave discipline; behavior-first authoring rule |
| v1.1 | ~7 | 5 | Execute-for-real milestone: census→repair→gate loop; frozen-contract metrics; family rollout with per-repair census reconciliation |
| v1.2 | ~3 | 3 | Paper-revision suite: file-disjoint parallel waves; owner overrides in VERIFICATION frontmatter; fixpoint verification regeneration practiced from the start |
| v1.4 | ~5 | 5 | Registry-as-truth-source + first UX surface: pure-core-first with structure-guarded thin shells; probes-never-raise; delegated-verification honesty in UAT records |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1 | 1,657 (7 allowlisted skips) | 96.30% | 0 — no new test frameworks (constraint held) |
| v1.1 | fast lane 1,933 (1 allowlisted skip) + nightly example census 196/1S/0F | 96.42% | 0 — nbclient/marimo ride existing dev/test extras (constraint held) |
| v1.2 | fast lane 2,404 (0 failed) | 96.72% | 1 — scikit-allel only (owner-approved, empirically verified); everything else on the existing scipy/sklearn/stdlib footprint |
| v1.4 | per-phase gates green; 828-test cross-phase regression gate at audit | per-module ≥96% on all new code (preflight 99% pkg · TUI ≥97.4% · registry 100% · consistency 98%) | +2 optional (textual, platformdirs) behind the `tui` extra; 0 new core deps |

### Top Lessons (Verified Across Milestones)

1. *(cross-validated v1.1)* A gate without a red proof is a claim, not a gate — v1.1's census pin proved it again (nightly red on a stale pin → deliberate re-pin → green)
2. *(cross-validated v1.1)* Measured baseline → ranked worklist beats estimated planning — the execute-everything census ranked Phase 8's repair queue the same way the coverage worklist ranked Phase 3
3. *(seeded from v1.1, pending cross-validation)* Real execution is the only honest test of example code — green static suites hide an order of magnitude more defects than they expose
4. *(seeded from v1.1, pending cross-validation)* Freeze metric contracts (observed values + tolerance bands) at calibration time; parse them at assert time — never literals
5. *(seeded from v1.1, pending cross-validation)* Land all fixes first, then regenerate verifiers with zero code changes between — reactive stale-digest chasing cost more than the fixes themselves
6. *(seeded from v1.4, pending cross-validation)* A plan implemented verbatim can still fail its UAT when the contract is narrower than the truth — assert completeness over the full allowed set, not presence of the examples written
7. *(seeded from v1.4, pending cross-validation)* Unparseable planning-artifact frontmatter degrades silently to "unknown" — the honesty chain (audit scanner → close gate) is only as good as the YAML it can read
