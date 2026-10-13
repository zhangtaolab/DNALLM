# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Paper Revision Suite Support** — Phases 10–12 (shipped 2026-10-10) — [archive](milestones/v1.2-ROADMAP.md)
- 🚧 **v1.3 Platform Reach & Packaging Honesty** — Phases 13–19 (in progress)

## Overview — v1.3 Platform Reach & Packaging Honesty

v1.3 is an honesty milestone, not a feature milestone: make the example tree execute natively on Windows (where the pybedtools/bedtools ecosystem offers nothing — its own docs say "Windows is not supported") via pure-numpy interval evaluators behind acquisition-seam platform dispatch, make the published extras do what their names claim, rust-proof a publish chain that has already failed twice consecutively, disposition the ollama pytest surface per the owner's 2026-10-11 directive, and sweep docs accuracy against the final code state. Both numpy parity spikes landed VERDICT PARITY (261010-wx2 jaccard, 261011-0a3 loj-intersect), so #19's ships-only-under-parity precondition is already met; the GPU support matrix boundaries were pre-verified 2026-10-11.

Seven phases, ordered by five hard constraints rather than category convenience: **subtraction before addition** (Phase 13's ollama sweep re-pins the census triple exactly once and shrinks every later edit surface); **shared-file sequencing** (Phase 14's GAP-1 ladder lands before Phase 16's gate fork — both edit `test_script_execution.py`); **library before dispatch** (Phase 15's evaluators exist before Phase 16's rollout imports them); **smoke before extras** (Phase 17's publish smoke exists to catch exactly the install-lane rot class Phase 18's atomic extras surgery risks — and its upload guard must be its first act, before any scheduled trigger exists); **docs last** (Phase 19's deliverable — accuracy — is a function of the final code state).

Cross-cutting invariants for every phase: the suite, >90% coverage gate, and full CI matrix stay honest-green throughout; every test-set change re-pins the D-03 census triple, `expected_skips.yaml`, and ci.yml example-nightly stages in the same change (one census-moving edit per phase); the Linux truth chain stays bedtools-computed with selection.md the single truth source parsed on both platforms (tolerance lives in the comparison layer only — never a second truth source); no ever-green skips anywhere (fail-loud Windows dispatch, per-assertion ollama disposition); every example edit lands with its docs mirror byte-identical (`check_docs_sync` green); new library modules meet the ≥96% per-module same-change coverage standard; the py≥3.11 / numpy≥2.0 floors hold; commits carry no Co-Authored-By trailers (repo convention).

## Phases

<details>
<summary>✅ v1.2 Paper Revision Suite Support (Phases 10–12) — SHIPPED 2026-10-10</summary>

- [x] Phase 10: Evaluation Contract Layer & Shared Scaffolding (4/4 plans) — completed 2026-10-09
- [x] Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities (5/5 plans) — completed 2026-10-09
- [x] Phase 12: Motif Matching, MCP Tools & Milestone Closeout (3/3 plans) — completed 2026-10-10

Full phase details, requirements mapping, and success criteria: [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)

</details>

<details>
<summary>✅ v1.1 Example Execution Testing & Repair (Phases 5–9) — SHIPPED 2026-10-07</summary>

- [x] Phase 5: Execution Harness, Honest Gates & Runner Feasibility (6/6 plans) — completed 2026-10-03
- [x] Phase 6: Model Registry & Showcase Data Curation (3/3 plans) — completed 2026-10-03
- [x] Phase 7: PlantHelixSeek Showcase Notebooks (2/2 plans) — completed 2026-10-04
- [x] Phase 8: Full Execution Rollout & Repair Loop (9/9 plans) — completed 2026-10-05
- [x] Phase 9: CI Wiring & Census Verification (4/4 plans) — completed 2026-10-06

Full phase details, requirements mapping, and success criteria: [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1 Test Suite Audit & Coverage Hardening (Phases 1–4) — SHIPPED 2026-10-01</summary>

- [x] Phase 1: Harness Integrity & Measured Baseline (2/2 plans) — completed 2026-09-30
- [x] Phase 2: Suite Hygiene & Known-Bug Fixes (3/3 plans) — completed 2026-09-30
- [x] Phase 3: Test Authoring to >90% Coverage (5/5 plans) — completed 2026-09-30
- [x] Phase 4: CI Gate Enforcement (3/3 plans) — completed 2026-10-01

Full phase details, requirements mapping, and success criteria: [milestones/v1-ROADMAP.md](milestones/v1-ROADMAP.md)

</details>

**Phase Numbering:**
- Integer phases (13–19): planned v1.3 milestone work (continues v1.2's Phase 12 — numbering never restarts)
- Decimal phases (13.1, 13.2): urgent insertions (marked INSERTED)

- [x] **Phase 13: Ollama Surface Disposition** - Per-assertion remove-or-skip over the ~59 ollama pytest refs, with census triple, skip allowlist, and nightly stages reconciled same-change (completed 2026-10-13)
- [x] **Phase 14: GAP-1 Transformers 5.x Disposition** - Evidence-sized absence-gated shim ladder or honest removal — WINDOWS #11/#12 leave the owner-pending state (completed 2026-10-13)
- [x] **Phase 15: Windows Parity Core — Numpy Interval Evaluators** - `dnallm/utils/genomic_intervals.py` carrying the oracle-pinned spike contracts, locked by platform-neutral unit tests on every CI leg (completed 2026-10-13)
- [ ] **Phase 16: Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity** - Acquisition-fork platform dispatch, platform-aware test gates, test-windows numpy executor with tolerance pins, and the dispatch-leak structure test
- [ ] **Phase 17: Publish Chain Robustness** - Upload guard first, never-uploading smoke on dispatch + cron, OIDC Trusted Publishing migration, WINDOWS #20 fix, mypy cap
- [ ] **Phase 18: Packaging & Install Honesty — Atomic Extras Surgery** - `base` rename, cuda121 retirement, rocm disposition, triton-windows declaration, GPU support matrix, uv/pip installer lanes
- [ ] **Phase 19: Docs Accuracy Sweep & Milestone Closeout** - Per-file docs/ + example/ accuracy review against the final state, cuda130 default everywhere, mkdocs strict-clean

## Phase Details

**Milestone goal (v1.3):** make platform coverage and the distribution chain honest end-to-end — Windows-native execution parity, accurate packaging/install guidance, a rust-proof publish chain, and a docs surface that matches the code — with the suite, coverage gate, and CI honest-green throughout.

### Phase 13: Ollama Surface Disposition

**Goal**: The ollama pytest surface is deliberately gone or honestly skip-marked — no blanket skip, no deleted neighbor contracts — with the census and nightly gates reconciled in the same change
**Depends on**: Nothing (first phase of v1.3 — v1.2 shipped 2026-10-10)
**Requirements**: OLLM-01, OLLM-02, OLLM-03
**Success Criteria** (what must be TRUE):
  1. Every one of the ~59 ollama refs across the 5 inventoried test files carries a per-assertion remove-or-skip disposition with recorded rationale (ledger in the plan) — removed tests are deleted outright (no zombie skips), kept tests use the existing typed-skip prefixes
  2. Non-ollama contracts sharing the swept surface — kernel-plot compat pins, extras brackets, systemd/docs-mirror contracts — still exist and pass after the sweep
  3. Post-sweep `grep ollama tests/` residue equals exactly the sanctioned list recorded in the plan; the two mcp_example notebooks remain in-repo per WINDOWS #13 with `langchain-ollama` still declared in the mcp extra
  4. The D-03 census triple is re-pinned, `expected_skips.yaml` updated, and ci.yml example-nightly stages reconciled in the same change — the nightly run stays green (no pytest exit 5, no unexpected-skip audit failure)

**Plans**: 1/1 plans complete

Plans:
- [x] 13-01-PLAN.md — Atomic D-04 disposition: delete nightly stage 3 (gate fires the allowlisted network-unavailable: typed skip), re-pin census triple 208/217→216/217, drop the -k mcp deselect from stages 0.5/1, reconcile stage-4 audit list, comment contracts, allowlist comment, and both READMEs; 59-ref per-assertion ledger + 56-line sanctioned residue recorded in the plan

### Phase 14: GAP-1 Transformers 5.x Disposition

**Goal**: WINDOWS #11/#12 leave the owner-pending state — the NT promoter checkpoint either loads for real on transformers 5.17 through an evidence-sized absence-gated shim ladder, or the third benchmark model is honestly removed by recorded decision
**Depends on**: Phase 13 (subtraction first — the swept test files shrink this phase's edit surface and the census re-pins once, before this phase's un-skip moves counts again)
**Requirements**: GAPC-01, GAPC-02
**Success Criteria** (what must be TRUE):
  1. An empirical load run of the NT promoter checkpoint on transformers 5.17 under the current shim set is on record, sizing the remaining rung ladder (the live probe pins exactly `invert_attention_mask` missing; the ladder extends rung-by-rung only on real tracebacks)
  2. The disposition is executed on that evidence — either the vendored absence-gated rung extension lands and the NT tests un-skip to real execution (closing #11/#12, with #15 closure riding along on per-item probe evidence), or the third benchmark model + BPE pkl leg is removed with the decision documented (closing #11/#12 as removed-by-decision); no owner-pending or silently-broken state remains at phase close
  3. Non-NT model families are provably unaffected — the shim is version-keyed and absence-gated, and the 35-family regression gate stays green
  4. The fast lane is green with the census triple and `expected_skips.yaml` reconciled in the same change as any un-skip or removal

**Plans**: 2/2 plans complete

Plans:
**Wave 1**
- [x] 14-01-PLAN.md — GAPC-01 evidence on record (recorded promoter load run, BPE-leg probe with task_type token, invert_attention_mask reachability proof, nightly junit forensics, census baseline → 14-EVIDENCE.md sections A-F), then the locked owner disposition checkpoint (blocking-human decision: shim-zero-rung vs removal, D-04)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 14-02-PLAN.md — Post-decision execution: comment-only GAP-1 prose resolution in the two test files (pruned-AST gate, markers byte-identical, census/matcher stability proofs), local D-06 gates (fast lane + 35-family + skip audit), WINDOWS #11/#12/#15 closure via gsd-tools, GAPC requirements complete, CI coverage-nightly dispatch re-verification on ref v1.3

### Phase 15: Windows Parity Core — Numpy Interval Evaluators

**Goal**: The spike-validated numpy interval evaluators exist as covered, platform-neutral library code whose oracle-pinned contracts are locked by unit tests on every CI leg — the bedrock Phase 16's Windows dispatch imports
**Depends on**: Phase 14 (sequencing/census discipline only — the module itself is pure addition; landing after the last count-moving sweep keeps one census change per phase)
**Requirements**: WPAR-01, WPAR-02
**Success Criteria** (what must be TRUE):
  1. `dnallm/utils/genomic_intervals.py` delivers `parse_bed`, `jaccard`, `loj_intersect` carrying every oracle-pinned spike contract — the 6-column null-B fill including the `-1` score slot, A-file-order emission accepting unsorted input, join-not-merge multiplicity, bookended-adjacency merge, `n_intersections` excluded — import-pure numpy-only and re-exported on the `dnallm.utils` surface (genomic_coords precedent)
  2. The spike corpora (seeds 261010-wx2 / 261011-0a3) plus handcrafted edge cases (boundary touching, chrom isolation, duplicates, empty B) run as fast, platform-neutral unit tests with baked oracle literals — green on every CI leg including test-windows, with no bedtools binary required anywhere in the unit layer
  3. The module sits provably outside the 7-entry coverage omit list and meets the ≥96% per-module same-change coverage standard (owner rule for library changes)

**Plans**: 1/1 plans complete

Plans:
- [x] 15-01-PLAN.md — Tracer-first port: parse_bed + jaccard wired end-to-end through the dnallm.utils re-export (Task 1), loj_intersect expansion carrying the oracle-pinned null-B fill table incl. the 6-column NER vector (Task 2), then the full oracle battery (8+11 spike cases), seeded corpora with property invariants, import-purity test, and the ≥96% coverage-CLI proof with omit-list visibility (Task 3) — three files, zero CI/census/expected_skips edits

### Phase 16: Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity

**Goal**: The example tree acquires its intervals natively on Windows — platform dispatch at acquisition seams only, platform-aware test gates, test-windows executing the numpy evaluators against pinned tolerances — while the Linux bedtools truth chain is structurally protected from dispatch leakage
**Depends on**: Phase 14 (`test_script_execution.py` is edited by both — the GAP-1 ladder lands before the gate fork), Phase 15 (the dispatch imports the numpy evaluators)
**Requirements**: WPAR-03, WPAR-04, WPAR-05, WPAR-06, WPAR-08
**Success Criteria** (what must be TRUE):
  1. `generate_bpe_dataset.py` and `data_generation_and_inference.ipynb` fork at acquisition only — Linux keeps `pybedtools.BedTool.intersect(loj=True)`, Windows dispatches to the numpy equivalent through the uniform row adapter (`.chrom/.start/.end/.name/.fields`) so the downstream consumer is byte-identical on both paths; the notebook's install cell skips pybedtools on Windows
  2. The three inventoried gates (`test_script_execution.py:128`, `test_notebook_execution.py:1397`, `test_plant_helixseek_showcase.py:523`) are platform-aware — Linux probe-then-execute unchanged, Windows executes the numpy path fail-loud with no ever-green skip — with Windows-branch contract tests exercising the dispatch via monkeypatched platform
  3. test-windows executes the numpy evaluators over committed frozen interval pairs with pinned comparison tolerances (trailing-float deltas); the frozen metrics hold in band on BOTH paths — jaccard=0.324727 and exon_f1=0.7522 with its band/tolerance newly pinned as part of this work — and selection.md stays the single truth source parsed on both paths
  4. A same-change dispatch-direction structure test proves the numpy path cannot silently execute on Linux (fail-loud dispatch, no except-fallback ladder), and the Linux nightly re-execution holds the parsed selection.md bands in the same change
  5. Every example edit lands with its docs mirror byte-identical (`check_docs_sync` green)

**Plans**: 4 plans

Plans:
- [ ] 16-01-PLAN.md — Tracer: CRE cell-11 platform dispatch + showcase gate platform-aware + tests/test_dispatch_direction.py born (structure + win32/POSIX-fail-loud runtime contracts) — WPAR-04/06/08
- [ ] 16-02-PLAN.md — NER rollout: script + notebook cells 0/1/2/8 acquisition-only fork (inline LojRow + _loj_rows), script/notebook gates platform-aware, dispatch-direction suite extended with live pybedtools parity + import fail-loud contracts — WPAR-03/04/06/08
- [ ] 16-03-PLAN.md — Frozen pairs: same-session regeneration (GPU pred sides + dual bedtools/numpy oracles) committed under tests/fixtures/, selection.md exon_f1 tolerance band row (+ mirror), tests/test_frozen_pairs.py platform-agnostic at tests/ root — WPAR-05/08
- [ ] 16-04-PLAN.md — SC4 evidence: manual nightly workflow_dispatch over the complete phase change + 5/5 SC proof matrix + phase-wide guard sweep (census/skips/deps/mirrors) — WPAR-06/08 close-out

### Phase 17: Publish Chain Robustness

**Goal**: The publish chain can be exercised end-to-end — build, check, install-from-dist, import — on demand and on schedule with zero possibility of uploading, the release path migrates to OIDC Trusted Publishing, and the known dev-tooling failure classes (mypy drift, the #20 cov crash) are capped or fixed
**Depends on**: Phase 16 (position-flexible per research, but sequenced here so its smoke dispatches validate Phase 18's extras surgery; the upload guard must be this phase's first act regardless of position)
**Requirements**: PUBS-01, PUBS-02, PUBS-03, PUBS-04, PUBS-05, PKGH-08
**Success Criteria** (what must be TRUE):
  1. A dry-run dispatch can never upload — the unguarded `workflow_dispatch`→`twine upload` path is event-gated or split BEFORE any smoke trigger exists (PyPI filenames are immutable), and the existing verify step no longer pip-installs from PyPI on non-publish runs
  2. The publish smoke runs the never-uploading chain on workflow_dispatch AND a periodic cron schedule (schedule-aware): `uv build --no-sources` → `twine check` (twine pinned `>=7,<8`, redundant `build` dep dropped) → clean-venv install-from-dist → import probe — verification reads `dist/`, never PyPI
  3. OIDC Trusted Publishing is live via `pypa/gh-action-pypi-publish@release/v1` — job-level `id-token: write`, publisher registered for the exact owner/repo/workflow-filename/environment, no reusable-workflow factoring — and `PYPI_API_TOKEN` is deleted only after the first observed green OIDC release
  4. WINDOWS #20 is remediated — the local-venv `pytest --cov` dotted-target crash is fixed via a local numpy pin or the documented coverage-CLI workaround (not by a coverage version bump)
  5. mypy is capped in the dev extra (`>=1.15.0,<2.5`) so CI can never resolve an unvalidated stricter version

**Plans**: TBD

### Phase 18: Packaging & Install Honesty — Atomic Extras Surgery

**Goal**: The extras surface does exactly what its names claim after one atomic, grep-guarded surgery — `base` renamed, cuda121 retired, rocm dispositioned, triton-windows declared — with the GPU support reality documented from the pre-verified boundaries and every changed extra resolving clean in a fresh venv
**Depends on**: Phase 17 (the smoke exists to catch exactly the install-lane rot class this surgery risks — each extras change is validated by a smoke dispatch against the post-surgery tree)
**Requirements**: PKGH-01, PKGH-02, PKGH-03, PKGH-04, PKGH-06, PKGH-07, WPAR-07
**Success Criteria** (what must be TRUE):
  1. The `base` extra rename/re-scope lands atomically in one change across pyproject (self-referential `all` chain), the 5 CI/publish call sites, extras guard tests, and docs — with a one-cycle deprecated alias so pip's warn-and-exit-0 cannot silently under-install downstream, and a grep gate holding stale `.[base]` residue to a sanctioned list
  2. cuda121 is retired in the same atomic discipline (extra + uv index/sources/conflicts entries + docs references in one change, index frozen at torch 2.5.1) and the rocm disposition is executed on verified facts — remove with the restore path documented, or restore (no rocm wheel exists for torch>=2.10)
  3. WINDOWS #18 is decided and executed — triton-windows declared behind a `platform_system == 'Windows'` marker in the `fla` extra (research-recommended `>=3.7.1.post27,<3.8`) or the manual step documented — and the fla KDA smoke is re-verified on Windows under the resolved torch
  4. Install docs split into explicit uv vs pip lanes stating the `[tool.uv.sources]` asymmetry (plain pip never sees the cuda index pins and gets the default torch build), the GPU support matrix is published from the pre-verified boundaries (Pascal cpu-only with the minimum-GPU-line stated, 20xx→cu128, 30/40xx all extras, 50xx cu128+), and the cross-installer honesty matrix page (command × installer → what you actually get) is live
  5. Every changed extra resolves clean in a fresh venv, proven by Phase 17's smoke dispatch running against the post-surgery tree

**Plans**: TBD

### Phase 19: Docs Accuracy Sweep & Milestone Closeout

**Goal**: Every docs/ and example/ page is verified accurate against the final v1.3 code state — install commands describe the final extras surface with cuda130 as the default recommendation, and mkdocs builds strict-clean — closing the milestone's honesty backstop
**Depends on**: Phases 16, 17, 18 (docs must describe the final code state — the sweep lands after every surface it describes has stopped moving)
**Requirements**: PKGH-05, DOCR-01, DOCR-02, DOCR-03
**Success Criteria** (what must be TRUE):
  1. Every file under docs/ and example/ carries a per-file accuracy review against current code and post-numpy2/py311 staleness, with fixes landing in the same pass and a disposition record per file (beyond the automated sync/masking gates)
  2. Install commands across all surfaces are re-verified against the final extras surface — cuda130 is the recommended default everywhere (overview.md's 3 stale `.[base,notebook,cuda124]`-class mentions + docs + examples), lane-labeled per the Phase 18 uv/pip split
  3. `mkdocs --strict` builds with zero warnings — the 15 ledgered pre-existing warnings cleared, with the nav/mirror interplay respected
  4. Every example edit in the sweep lands with its docs mirror byte-identical and the automated sync/masking gates green; the milestone closes honest-green (fast lane, coverage gate, full CI matrix)

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 13 → 14 → 15 → 16 → 17 → 18 → 19

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 13. Ollama Surface Disposition | 1/1 | Complete    | 2026-10-13 |
| 14. GAP-1 Transformers 5.x Disposition | 2/2 | Complete    | 2026-10-13 |
| 15. Windows Parity Core — Numpy Interval Evaluators | 1/1 | Complete    | 2026-10-13 |
| 16. Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity | 0/TBD | Not started | - |
| 17. Publish Chain Robustness | 0/TBD | Not started | - |
| 18. Packaging & Install Honesty — Atomic Extras Surgery | 0/TBD | Not started | - |
| 19. Docs Accuracy Sweep & Milestone Closeout | 0/TBD | Not started | - |
