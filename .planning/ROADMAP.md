# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- 🚧 **v1.1 Example Execution Testing & Repair** — Phases 5–9 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (5–9): planned v1.1 work — numbering continues from v1, which ended at Phase 4
- Decimal phases (5.1, 5.2): urgent insertions (marked with INSERTED), execute between surrounding integers

<details>
<summary>✅ v1 Test Suite Audit & Coverage Hardening (Phases 1–4) — SHIPPED 2026-10-01</summary>

- [x] Phase 1: Harness Integrity & Measured Baseline (2/2 plans) — completed 2026-09-30
- [x] Phase 2: Suite Hygiene & Known-Bug Fixes (3/3 plans) — completed 2026-09-30
- [x] Phase 3: Test Authoring to >90% Coverage (5/5 plans) — completed 2026-09-30
- [x] Phase 4: CI Gate Enforcement (3/3 plans) — completed 2026-10-01

Full phase details, requirements mapping, and success criteria: [milestones/v1-ROADMAP.md](milestones/v1-ROADMAP.md)

</details>

### 🚧 v1.1 Example Execution Testing & Repair (In Progress)

**Milestone Goal:** Real-model execution testing for everything under `example/` with every surfaced error fixed; PlantHelixSeek-CRE/-Anno showcase notebooks over committed Arabidopsis loci whose predictions are substantially consistent with experimental truth; and the CI gate false-green (WR-08/WR-09) repaired so example tests run under formal nightly gating.

- [x] **Phase 5: Execution Harness, Honest Gates & Runner Feasibility** - Private nbclient execution harness proven on a pilot, WR-08/09 closed together with the docs-mirror drift they hid, and GB10 feasibility verdicts for the environment-gated model families (completed 2026-10-02; **REOPENED 2026-10-02** for post-closure gap closure — GAP-1 NT x transformers-5.17 compat shim + GAP-2 full example/ census bar, plans 05-04..06)
- [x] **Phase 6: Model Registry & Showcase Data Curation** - PlantHelixSeek-CRE/-Anno load through the generic registry route (labels frozen, transformers-5 verified) and the committed ≤200kb Arabidopsis loci, truth slices, rationale doc, and shared coordinate normalization helper exist (completed 2026-10-03)
- [x] **Phase 7: PlantHelixSeek Showcase Notebooks** - CRE and Anno notebooks running real sliding-window inference with prediction-vs-truth presentation, calibrated agreement floors, and rendered-figure write-back to the docs mirror (completed 2026-10-04)
- [ ] **Phase 8: Full Execution Rollout & Repair Loop** - All notebooks, marimo apps, the helper script, and every YAML execute for real on the nightly GPU runner; every surfaced error fixed with regression tests; models.lock, giant-model cache tiers, and ollama infrastructure in place
- [ ] **Phase 9: CI Wiring & Census Verification** - Nightly census formally gates the execution-test layer end to end: collection, skip audit, runtime budget, hygiene steps, consistency guard, documented coverage expectation

## Phase Details

### Phase 5: Execution Harness, Honest Gates & Runner Feasibility

**Goal**: A trustworthy private execution harness exists and is proven (including kernel-kill on hang); both false-green CI gates are closed together with the docs-mirror drift they were hiding; and the runner's real capabilities for the environment-gated model families are settled in writing before execution tests are written against them
**Depends on**: Nothing (first phase of v1.1; builds on the shipped v1 CI gate)
**Requirements**: EXEC-01, EXEC-06, CI-01, CI-02, REPAIR-02, FEAS-01 — gap closure (05-04+, reopened 2026-10-02): EXEC-01 (full-tree census), EXEC-03/EXEC-04 (dev-box legs), REPAIR-03 (partial: NT remote-code shim)
**Success Criteria** (what must be TRUE):
  1. The pilot execution tests run 1–2 already-healthy notebooks end-to-end via nbclient in tmp-sandbox cwd isolation (kernel cwd = sandbox copy), with per-cell timeout firing inside a per-test timeout mark, context-managed kernel shutdown, and partial-notebook failure artifacts captured on error — and the git tree is clean after the run
  2. A deliberate-hang test proves the harness kills a hung kernel and leaves no `ipykernel_launcher` process behind
  3. `scripts/check_docs_sync.py` exits 0 (mirror drift closed: wrapper-`.md` handling fixed, byte-identical resync, missing script mirrored) and the docs-validation workflow runs honestly — `continue-on-error` removed, `mcp` extra installed, README "Local Testing" line corrected — without blocking unrelated PRs
  4. A written verdict matrix exists for evo-1 / evo2 / megaDNA / pyBigWig (and the marimo execution flavor) on the aarch64 GB10 runner; smallest viable real variants are enabled wherever feasible, and every `environment-unavailable:` typed skip carries recorded infeasibility evidence
  5. *(Gap closure, D-07/D-08)* `zhangtaolab/nucleotide-transformer-v2-100m-promoter` loads and forwards on transformers 5.17.0 through a gated compat shim with a real-model smoke regression test; and the committed full census inventory (`05-CENSUS.md`) gives every item under `example/` (21 notebooks, 3 marimo apps, 1 script) either a real-execution result or an evidence-backed typed skip — nothing silently omitted

**Plans**: 6/6 plans complete (3 complete + 3 gap closure, reopened 2026-10-02)

Plans:
**Wave 1**
- [x] 05-01-PLAN.md — Execution harness tracer: nbclient harness + pilot notebook + deliberate-hang kill test + typed-skip prefixes (EXEC-01, EXEC-06)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 05-02-PLAN.md — Honest gates: docs-mirror drift closure + all five masked docs-validation steps flipped + mcp extra/README + branch-protection hand-off (CI-01, CI-02, REPAIR-02)
- [x] 05-03-PLAN.md — GB10 feasibility spike: per-family spike runner + verdict matrix + dispatch-only runner confirmation + conditional pyBigWig (FEAS-01)

**Gap closure (reopened 2026-10-02, `gap_closure: true`)**
**Wave 1** *(parallel — disjoint files)*
- [x] 05-04-PLAN.md — GAP-1: gated pruning-helper shim in transformers_compat.py + real-model load+FORWARD smoke on transformers 5.17 (EXEC-01, REPAIR-03 partial)
- [x] 05-05-PLAN.md — GAP-2 foundations: marimo/script execution lanes + all-21 NOTEBOOK_EXEC_SPECS + committed 05-CENSUS.md skeleton + .scratch/ ignore (EXEC-01)

**Wave 2** *(blocked on 05-04 + 05-05)*
- [x] 05-06-PLAN.md — GAP-2 census campaign: full example/ execution on the dev box + per-item verdicts + durable wiring + owner overlap hand-off (EXEC-01, EXEC-03/04 dev-box legs)

### Phase 6: Model Registry & Showcase Data Curation

**Goal**: Both PlantHelixSeek checkpoints load through the existing generic dnallm route (label order frozen to the checkpoint, transformers-5 compat proven), and the showcase's committed Arabidopsis loci — truth slices, selection rationale, negative control — exist in-repo alongside a shared, unit-tested coordinate/chrom-name normalization helper
**Depends on**: Nothing hard — independent of Phase 5 (disjoint file sets; the two tracks are parallelizable)
**Requirements**: REG-01, REG-02, REG-03, SHOW-01, SHOW-02
**Success Criteria** (what must be TRUE):
  1. `load_model_and_tokenizer` smoke-loads both `PlantHelixSeek-CRE` (binary, num_labels 2) and `PlantHelixSeek-Anno` (token, num_labels 17) through the generic task-type route on the transformers 5.x dev environment — no special handler — and the Anno execution test asserts `model.config.id2label` exactly equals the frozen 17-BILOU `label_names` order
  2. Committed in-repo Arabidopsis fragments ≤200kb per region exist with truth slices (CRE ↔ PlantDHS `TAIR10_DHSs.gff`; Anno ↔ TAIR10 GFF3), a selection-rationale doc, and one negative-control locus; the selected loci were verified for substantial prediction-truth agreement, and download intermediates stay gitignored (clean tree)
  3. The shared coordinate/chrom-name normalization helper (0-based half-open ↔ 1-based closed; `Chr1` ↔ `1`) passes unit tests on tiny fixtures, and every genomics code path uses it with non-emptiness assertions against silent-empty results

**Plans**: 3/3 plans complete

Plans:
**Wave 1** *(parallel — disjoint files)*
- [x] 06-01-PLAN.md — Registry tracer: one-shot scratch freeze run (constraint proof + provenance) + two model_info.yaml entries + fast-leg structure test + slow-leg ModelScope smoke with id2label equality (REG-01, REG-02, REG-03)
- [x] 06-02-PLAN.md — SHOW-02 helper: dnallm/utils/genomic_coords.py (coordinate/chrom normalization, non-emptiness guards) + unit tests + package re-export (SHOW-02)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 06-03-PLAN.md — Showcase curation run: scratch selection tool (acquire/rank/verify/audit) + committed ≤200kb loci, truth slices, negative controls, selection.md rationale (SHOW-01)

### Phase 7: PlantHelixSeek Showcase Notebooks

**Goal**: The two flagship showcase notebooks run real sliding-window inference on the committed loci, present prediction-vs-truth honestly (illustrative-loci framing), assert calibrated agreement floors, and land in the docs mirror with rendered figures
**Depends on**: Phase 6 (registry entries, committed loci, normalization helper), Phase 5 (harness for executed write-back)
**Requirements**: SHOW-03, SHOW-04, SHOW-05, SHOW-06, SHOW-07
**Success Criteria** (what must be TRUE):
  1. The CRE notebook executes end-to-end: 500bp window / 50bp stride / 50bp bin sliding scan via the dnallm API, altair side-by-side prediction track vs PlantDHS truth, and in-notebook peak calling (mean±1.5σ → BED/narrowPeak) with Jaccard computed on called peaks
  2. The Anno notebook executes end-to-end: 8192/4096 both-strand scan, BILOU span decode to structurally valid GFF3, nucleotide/exon-level sensitivity/precision/F1 vs TAIR10, and exon/intron gene-model diagrams (altair)
  3. The example tests assert truth-agreement floors with thresholds calibrated at loci-selection time — recorded observed values and tolerance bands, never exact outputs
  4. Both executed notebooks with rendered figures are written back into the docs mirror (these two only), presenting "illustrative loci + selection criteria" framing with no genome-wide accuracy claims

**Plans**: 2/2 plans complete

Plans:
**Wave 1**
- [x] 07-01-PLAN.md — CRE showcase vertical slice (tracer): executed CRE notebook on GB10 + harness/floors-parsing nightly test + fast structure tests + CRE docs write-back incl. shared-dir mirror (SHOW-03, SHOW-05, SHOW-06, SHOW-07)

**Wave 2** *(blocked on 07-01 — extends its test module, harness specs, wrapper pattern, and Showcase nav group)*
- [x] 07-02-PLAN.md — Anno showcase slice: executed Anno notebook (both-strand scan, BILOU decode, GFF3, exon-F1) + Anno test lane + Anno docs write-back completing SHOW-06 (SHOW-04, SHOW-05, SHOW-06, SHOW-07)

### Phase 8: Full Execution Rollout & Repair Loop

**Goal**: Every example artifact executes for real on the nightly GPU runner — all notebooks, the marimo apps, the helper script, every YAML, and the two ollama-backed mcp_example notebooks — and every error that surfaces is fixed with regression tests across example code, the docs mirror, and the dnallm library
**Depends on**: Phase 5 (proven harness + honest gates); the Phase 7 showcase notebooks join the rollout when present (no hard dependency)
**Requirements**: EXEC-02, EXEC-03, EXEC-04, EXEC-05, REPAIR-01, REPAIR-03, REPAIR-04, CI-04, CI-05, MCP-01, MCP-02
**Success Criteria** (what must be TRUE):
  1. All 21 example notebooks execute all code cells end-to-end with real models on the nightly GPU runner (fail-at-first-error per notebook, fail-soft across notebooks); the 3 marimo apps execute headlessly via subprocess with default-value assertions and exit codes asserted; `generate_bpe_dataset.py` produces its dataset artifact in-sandbox
  2. Every example YAML config passes real `load_config()` Pydantic validation on the fast leg, with zero new fast-leg skips introduced
  3. Every error surfaced by real execution is fixed with a regression test — notebook/app/script code, dnallm library bugs (v1 precedent: AUROC, CrossDNA), and the langchain notebook's `!uv pip install langchain-ollama` cell with its dependency declared in the `mcp` extra — with harness-bug vs content-bug triage explicit (no cwd false-repairs) and the docs mirror regenerated as part of each notebook repair
  4. `models.lock` carries all newly-executed model ids (~8+) with ModelScope-first prefixes aligned to each notebook's actual `source=` route and revision pins; evo-1 is fetched safetensors-only via `allow_patterns` (~12.9GB, not 29.7GB) with giants tiered outside the 10GB-quota cache so the existing warm cache is never evicted
  5. Both mcp_example notebooks execute end-to-end against loopback-only ollama on the nightly runner (systemd service, pre-pulled small model, readiness probe), with port/VRAM coexistence planned against the 6 MCP live-server probes on :8000 and the heavy torch tests; a typed `network-unavailable:` skip with evidence is the documented fallback only

**Plans**: 9 plans

Plans:
**Wave 1** *(parallel — disjoint files)*
- [ ] 08-01-PLAN.md — example-nightly job tracer (staggered/staged/fail-soft/shared cache) + langchain-ollama in mcp extra + marimo D-18 deepening (EXEC-02, EXEC-03, REPAIR-04, MCP-02)
- [ ] 08-02-PLAN.md — NT heal: pristine snapshot + shim-only proof + D-17 ledger + script lane real-green + YAML re-verify + D-03 baseline census (EXEC-04, EXEC-05, REPAIR-01, REPAIR-03)
- [ ] 08-03-PLAN.md — evo library enablers: allow_patterns passthrough + np.fromstring shim + harness kernel env override, all with same-change tests (CI-05, EXEC-02)

**Wave 2** *(family order per D-01; blocked on 08-02/08-03)*
- [ ] 08-04-PLAN.md — megaDNA tracer slice: DNATokenizer library repair + finetune_generation content repair (ordering + pinned clone) + isolated-kernelspec first execution (EXEC-02, REPAIR-01, REPAIR-03)

**Wave 3** *(blocked on 08-04 — shared test wiring + library fix)*
- [ ] 08-05-PLAN.md — megaDNA expansion slice: finetune_custom_head + generation_megaDNA repairs + family execution + D-03 census reconciliation (EXEC-02, REPAIR-01)

**Wave 4** *(blocked on 08-03 enablers + 08-05 census baseline)*
- [ ] 08-06-PLAN.md — evo tracer slice: dev-box giants tier + isolated dnallm-evo kernelspec + 8k/noFP8 notebook repair + first real execution/rung discovery (EXEC-02, CI-05, REPAIR-01)

**Wave 5** *(blocked on 08-06)*
- [ ] 08-07-PLAN.md — evo expansion slice: residual rung closure to full green + A4 verification record + evo census reconciliation (EXEC-02, CI-05, REPAIR-01)

**Wave 6** *(blocked on evo completion + shared census rollup)*
- [ ] 08-08-PLAN.md — mamba lora pair + ollama loopback systemd infra + D-13 readiness probe + mcp pair both-up + D-07 stage contract (MCP-01, MCP-02, EXEC-02)

**Wave 7** *(blocked on all families + job skeleton)*
- [ ] 08-09-PLAN.md — models.lock growth with revision pins + cache-quota decision + example-nightly completion (prereqs incl. bedtools rootless step, giants prefetch, stages 2/3) + final D-03 census (CI-04, CI-05, EXEC-02, REPAIR-01)

### Phase 9: CI Wiring & Census Verification

**Goal**: The nightly census formally gates the finished execution-test layer — verified end to end on the real runner for collection, skip audit, runtime budget, hygiene steps, lock consistency, and the documented coverage expectation
**Depends on**: Phase 8 (census validates the finished execution-test set, including the Phase 7 showcase truth-agreement tests)
**Requirements**: CI-03, CI-06, CI-07, CI-08, CI-09
**Success Criteria** (what must be TRUE):
  1. A nightly census run collects and passes all new `slow`-marked execution tests; `audit_skips.py` exits green with the new typed categories present (`network-unavailable:` reuse, `environment-unavailable:`/`optional-dep:` additions registered in `expected_skips.yaml`), and the fast leg has zero new skips
  2. Measured runtime budgets are recorded; total execution fits the 900-min nightly job, or a separate example-execution nightly job is split out (owner pre-authorized)
  3. Nightly hygiene steps are observable in the workflow: kernel `pkill` + VRAM assertion, timeout-arithmetic sum-of-ceilings review, `if: always()` artifact uploads
  4. The fast-leg models.lock consistency guard fails on drift between model id literals inside notebooks/apps and lock entries
  5. The coverage expectation is documented: example execution runs in kernel subprocesses and by design does not move the 96.30% coverage gate (AUDIT-04 precedent)

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 5 → 6 → 7 → 8 → 9 (Phases 5 and 6 are parallelizable — disjoint file sets)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 5. Execution Harness, Honest Gates & Runner Feasibility | v1.1 | 6/6 | Complete    | 2026-10-03 |
| 6. Model Registry & Showcase Data Curation | v1.1 | 3/3 | Complete    | 2026-10-03 |
| 7. PlantHelixSeek Showcase Notebooks | v1.1 | 2/2 | Complete    | 2026-10-04 |
| 8. Full Execution Rollout & Repair Loop | v1.1 | 0/TBD | Not started | - |
| 9. CI Wiring & Census Verification | v1.1 | 0/TBD | Not started | - |

---

*Coverage: 32/32 v1.1 requirements mapped (EXEC 6, REPAIR 4, CI 9, FEAS 1, REG 3, SHOW 7, MCP 2) — no orphans, no duplicates.*
*v1.1 roadmap created 2026-10-01.*
