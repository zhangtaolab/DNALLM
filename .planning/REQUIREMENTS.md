# Requirements: DNALLM

**Defined:** 2026-10-01
**Core Value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## v1.1 Requirements

Requirements for milestone v1.1 "Example Execution Testing & Repair". Each maps to roadmap phases (traceability filled at roadmap creation).

### Execution Testing

- [x] **EXEC-01**: Private execution harness (`tests/examples/_execution.py` + locally-scoped `conftest.py`) runs notebooks via nbclient-as-library with per-cell timeout inside a per-test timeout mark, tmp-sandbox cwd isolation (kernel cwd = sandbox copy), context-managed kernel shutdown, and partial-notebook failure artifacts captured on error
- [ ] **EXEC-02**: All 21 example Jupyter notebooks execute all code cells end-to-end with real models on the nightly GPU runner (slow-marked; `allow_errors=False` fail-at-first-error per notebook, fail-soft across notebooks)
- [x] **EXEC-03**: All 3 marimo apps execute headlessly via subprocess (flavor standardized in the Phase-1 pilot) with UI elements yielding defaults and exit codes asserted
- [ ] **EXEC-04**: `generate_bpe_dataset.py` executes against its committed inputs producing its dataset artifact in-sandbox
- [ ] **EXEC-05**: Every example YAML config passes real `load_config()` Pydantic validation on the fast leg (new showcase YAMLs valid from their first commit)
- [x] **EXEC-06**: A deliberate-hang test proves the harness kills a hung kernel and leaves no `ipykernel_launcher` process behind

### Repair

- [ ] **REPAIR-01**: Every error surfaced by real execution is fixed — notebook/app/script code — each with a regression test; harness-bug vs content-bug triaged explicitly (no cwd false-repairs)
- [x] **REPAIR-02**: The already-broken docs/example mirror is closed (sync-script wrapper-`.md` handling fixed, byte-identical resync, missing script mirrored) and regenerated as part of every subsequent notebook repair
- [x] **REPAIR-03**: dnallm library bugs exposed by execution are fixed with regression tests (v1 precedent: AUROC, CrossDNA)
- [x] **REPAIR-04**: The langchain notebook's `!uv pip install langchain-ollama` shell-magic cell is repaired — dependency declared in the `mcp` extra

### CI Gating

- [x] **CI-01**: WR-08 closed — docs-validation `continue-on-error: true` removed in the same reviewable unit as the mirror-drift closure, so the gate is honest without blocking unrelated PRs
- [x] **CI-02**: WR-09 closed — docs-validation installs the `mcp` extra; README "Local Testing" install line corrected
- [ ] **CI-03**: Execution tests are `slow`-marked into the nightly census with zero new fast-leg skips; typed skip prefixes (`network-unavailable:` reuse, `environment-unavailable:`/`optional-dep:` additions) registered in `expected_skips.yaml`; skip audit green with the new categories
- [ ] **CI-04**: `models.lock` extended with all newly-executed model ids (~8+) with **ModelScope-first prefixes** — `ms` wherever the model exists on ModelScope (owner decision; zhangtaolab models are mirrored there), `hf` only as fallback — each notebook's `source=` route aligned with its lock prefix, revision-pinned (`trust_remote_code` provenance)
- [ ] **CI-05**: Cache strategy survives the giants — evo-1 fetched via `allow_patterns` (safetensors only, ~12.9GB not 29.7GB), tiered so giant models persist outside the 10GB-quota cache and never evict the existing warm cache
- [ ] **CI-06**: Measured runtime budgets recorded; if total execution exceeds the 900-min nightly job, a separate example-execution nightly job is split out (pre-authorized by owner)
- [ ] **CI-07**: Nightly hygiene steps land: kernel `pkill` + VRAM assertion, timeout-arithmetic sum-of-ceilings review, `if: always()` artifact uploads
- [ ] **CI-08**: models.lock consistency guard — a fast-leg test cross-checks model id literals inside notebooks/apps against lock entries, failing on drift
- [ ] **CI-09**: Coverage expectation documented: example execution runs in kernel subprocesses and by design does not move the 96.30% coverage gate (AUDIT-04 precedent)

### Runner Feasibility

- [x] **FEAS-01**: Phase-1 spike produces a written verdict matrix for evo-1 / evo2 / megaDNA / pyBigWig on the aarch64 GB10 runner; smallest viable real variants are enabled wherever feasible (owner: spike first, run real models), and `environment-unavailable:` typed skips are used only with recorded infeasibility evidence

### PlantHelixSeek Registry

- [x] **REG-01**: `model_info.yaml` finetuned-section entries for `PlantHelixSeek-CRE` (binary, num_labels 2) and `PlantHelixSeek-Anno` (token, num_labels 17) loadable through the existing generic task-type route (no special handler)
- [x] **REG-02**: Anno `label_names` frozen to the checkpoint's exact `config.id2label` order; the execution test asserts `model.config.id2label` equality after load (silent-permutation guard)
- [x] **REG-03**: Smoke-load of both checkpoints via the dnallm route succeeds on the transformers 5.x dev environment (compat risk gate before any showcase work freezes)

### Showcase

- [x] **SHOW-01**: Loci selection produces committed in-repo Arabidopsis fragments ≤200kb per region where predictions are substantially consistent with experimental truth (CRE ↔ PlantDHS `TAIR10_DHSs.gff`; Anno ↔ TAIR10 GFF3), plus truth slices, a selection-rationale doc, and one negative-control locus; all download intermediates gitignored
- [x] **SHOW-02**: A shared, unit-tested coordinate/chrom-name normalization helper (0-based half-open ↔ 1-based closed; `Chr1` ↔ `1`) is used by all genomics code paths, with non-emptiness assertions against silent-empty results
- [x] **SHOW-03**: CRE notebook — 500bp window/50bp stride/50bp bin sliding scan via dnallm API, altair side-by-side prediction track vs PlantDHS truth, in-notebook peak calling (mean±1.5σ → BED/narrowPeak) with Jaccard computed on called peaks
- [x] **SHOW-04**: Anno notebook — 8192/4096 both-strand scan, BILOU span decode to structurally valid GFF3, nucleotide/exon-level sensitivity/precision/F1 vs TAIR10, exon/intron gene-model diagrams (altair)
- [x] **SHOW-05**: Truth-agreement floors are asserted by the example tests — thresholds calibrated at loci-selection time with recorded observed values and tolerance bands, never exact outputs
- [x] **SHOW-06**: Executed showcase notebooks with rendered figures are written back into the docs mirror (these two only — the credibility artifact)
- [x] **SHOW-07**: Both notebooks present "illustrative loci + selection criteria" framing — no genome-wide accuracy claims

### MCP / ollama

- [ ] **MCP-01**: ollama runs on the nightly GPU runner as loopback-only systemd infrastructure with a pre-pulled small model and readiness probe; both mcp_example notebooks execute end-to-end against it; a typed `network-unavailable:` skip (with evidence) is the documented fallback only
- [ ] **MCP-02**: Port and VRAM coexistence is planned against the existing 6 MCP live-server probes (:8000) and heavy torch tests (execution ordering documented)

## v1.2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Deferred Enhancements

- **NOTEBOOK-PARALLELISM**: xdist / parallel notebook execution (only if a second GPU runner appears)
- **OUTPUT-REGRESSION**: nbval-style output-regression testing of notebooks
- **IGV-APPENDIX**: optional interactive igv-notebook appendix cells (non-gated) for live exploration
- **MORE-SPECIES**: additional species' showcase loci (rice, Brachypodium via PlantDHS)
- **PER-CELL-TIMING**: per-cell timing diagnostics in failure artifacts

## Out of Scope

| Feature | Reason |
|---------|--------|
| pyGenomeTracks for track display | GPL-3.0 in an MIT project, matplotlib<3.9 pin conflict, external bedtools binary, GFF3 not a documented track type (research-verified) |
| jbrowse-anywidget | Not on PyPI (git-only install), Prototype status — CI-non-reproducible |
| igv-notebook in gated cells | Widget output does not survive nbconvert→HTML docs mirror; allowed only as future non-gated appendix |
| Executed-notebook write-back for all 21 notebooks | Only the 2 showcase notebooks get write-back; committing 21 executed copies churns the repo |
| Full-genome inference in notebooks | Hours-scale CUDA-mandatory compute; violates the ≤200kb showcase constraint by design |
| IDR scoring | Replicate-concordance tool — wrong question for prediction-vs-truth |
| Exact-output assertions | Guaranteed-flaky across the transformers 4.49–5.x span; tolerance bands instead |
| A `special/` handler for PlantHelixSeek | Generic route verified sufficient; dispatch-chain edits re-introduce CrossDNA-class risk |
| Enabling evo-class giants beyond smallest viable variants | GB10 cache/toolchain economics; smallest viable real variants only |
| `allow_errors=True` notebook execution | Cascading noise hides the first-error repair signal |
| xdist on the single GPU runner | VRAM flake on exactly the run that must be trustworthy |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXEC-01 | Phase 5 | Complete |
| EXEC-02 | Phase 8 | Pending |
| EXEC-03 | Phase 8 | Complete |
| EXEC-04 | Phase 8 | Pending |
| EXEC-05 | Phase 8 | Pending |
| EXEC-06 | Phase 5 | Complete |
| REPAIR-01 | Phase 8 | Pending |
| REPAIR-02 | Phase 5 | Complete |
| REPAIR-03 | Phase 8 | Complete |
| REPAIR-04 | Phase 8 | Complete |
| CI-01 | Phase 5 | Complete |
| CI-02 | Phase 5 | Complete |
| CI-03 | Phase 9 | Pending |
| CI-04 | Phase 8 | Pending |
| CI-05 | Phase 8 | Pending |
| CI-06 | Phase 9 | Pending |
| CI-07 | Phase 9 | Pending |
| CI-08 | Phase 9 | Pending |
| CI-09 | Phase 9 | Pending |
| FEAS-01 | Phase 5 | Complete |
| REG-01 | Phase 6 | Complete |
| REG-02 | Phase 6 | Complete |
| REG-03 | Phase 6 | Complete |
| SHOW-01 | Phase 6 | Complete |
| SHOW-02 | Phase 6 | Complete |
| SHOW-03 | Phase 7 | Complete |
| SHOW-04 | Phase 7 | Complete |
| SHOW-05 | Phase 7 | Complete |
| SHOW-06 | Phase 7 | Complete |
| SHOW-07 | Phase 7 | Complete |
| MCP-01 | Phase 8 | Pending |
| MCP-02 | Phase 8 | Pending |

**Coverage:**
- v1.1 requirements: 32 total
- Mapped to phases: 32 (Phase 5: 6, Phase 6: 5, Phase 7: 5, Phase 8: 11, Phase 9: 5)
- Unmapped: 0
- Duplicated: 0

---
*Requirements defined: 2026-10-01*
*Last updated: 2026-10-01 after roadmap creation (Phases 5–9)*
