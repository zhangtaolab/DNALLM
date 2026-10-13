# Requirements: DNALLM — Test Suite Audit & Coverage Hardening

**Defined:** 2026-10-11
**Core Value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## v1.3 Requirements

Requirements for milestone **v1.3 Platform Reach & Packaging Honesty**. Each maps to exactly one roadmap phase (traceability below, filled at roadmap creation). Grounding: `.planning/research/SUMMARY.md` (committed be31aad).

### Windows Platform Parity (WPAR)

- [x] **WPAR-01**: Pure-numpy interval module `dnallm/utils/genomic_intervals.py` (`parse_bed`, `jaccard`, `loj_intersect`) carries the oracle-pinned spike contracts (6-column null-B fill incl. the `-1` score slot, A-file-order emission accepting unsorted input, join-not-merge multiplicity, bookended-adjacency merge; `n_intersections` excluded), sits provably outside the coverage omit list, import-pure numpy-only
- [x] **WPAR-02**: Spike corpora (261010-wx2 / 261011-0a3 seeds) + handcrafted edge cases run as fast, platform-neutral unit tests with baked oracle literals on every CI leg — no bedtools binary required
- [ ] **WPAR-03**: `generate_bpe_dataset.py` and `data_generation_and_inference.ipynb` fork at acquisition only (runtime platform detection): Linux keeps `pybedtools.BedTool.intersect(loj=True)`, Windows dispatches to the numpy equivalent through a uniform row adapter (`.chrom/.start/.end/.name/.fields`) so the consumer stays byte-identical; the notebook install cell skips pybedtools on Windows
- [ ] **WPAR-04**: The three inventoried test gates (`test_script_execution.py:128`, `test_notebook_execution.py:1397`, `test_plant_helixseek_showcase.py:523`) become platform-aware — Linux probe-then-execute unchanged, Windows executes the numpy path (fail-loud, no ever-green skip)
- [ ] **WPAR-05**: test-windows executes the numpy evaluators over committed frozen interval pairs with pinned comparison tolerances (trailing-float deltas); frozen metrics hold in band on BOTH paths — jaccard=0.324727 and exon_f1=0.7522 (its band/tolerance newly pinned as part of this work); selection.md stays the single truth source parsed on both paths
- [ ] **WPAR-06**: A same-change dispatch-direction structure test proves the numpy path cannot silently execute on Linux (fail-loud dispatch, no except-fallback ladder)
- [ ] **WPAR-07**: WINDOWS #18 decided and executed — triton-windows declared behind a Windows platform marker in the `fla` extra (research-recommended: `>=3.7.1.post27,<3.8`) or the manual step documented; fla KDA smoke re-verified on Windows under the resolved torch
- [ ] **WPAR-08**: Every example edit lands with its docs mirror byte-identical (`check_docs_sync` green) and the Linux nightly re-execution holds the parsed selection.md bands

### Packaging & Install Honesty (PKGH)

- [ ] **PKGH-01**: The `base` extra is renamed/re-scoped per owner decision with a one-cycle deprecated alias; landed atomically across pyproject (self-referential `all` chain), the 5 CI/publish call sites, extras guard tests, and docs; a grep gate holds stale `.[base]` residue to a sanctioned list
- [ ] **PKGH-02**: The cuda121 extra is retired — extra, uv index/sources/conflicts entries, and docs references removed in one atomic change (index frozen at torch 2.5.1, dead lane under the `torch>=2.4,<2.12` floor)
- [ ] **PKGH-03**: The rocm disposition is executed on verified facts (research-recommended: remove — no rocm wheel exists for torch>=2.10; restore path documented if removed)
- [ ] **PKGH-04**: Install docs split into explicit uv vs pip lanes stating the `[tool.uv.sources]` asymmetry (plain pip never sees the cuda index pins and gets the default torch build)
- [ ] **PKGH-05**: cuda130 is the recommended default install across all surfaces (overview.md's 3 stale `.[base,notebook,cuda124]`-class mentions + docs + examples)
- [ ] **PKGH-06**: The GPU support matrix is published from the pre-verified boundaries (Pascal cpu-only stated explicitly with the minimum-GPU-line; 20xx→cu128; 30/40xx all extras; 50xx cu128+; optional driver-based chooser note)
- [ ] **PKGH-07**: A cross-installer honesty matrix page is published — command × installer → what you actually get
- [ ] **PKGH-08**: mypy is capped (research: `>=1.15.0,<2.5`) so CI cannot resolve an unvalidated stricter version

### Publish Chain Robustness (PUBS)

- [ ] **PUBS-01**: The unguarded `workflow_dispatch`→`twine upload` path is event-gated or split BEFORE any smoke trigger exists — a dry-run dispatch can never upload (PyPI filenames are immutable)
- [ ] **PUBS-02**: A publish smoke runs the never-uploading chain on dispatch: `uv build --no-sources` → `twine check` (twine pinned `>=7,<8`; redundant `build` dep dropped) → clean-venv install-from-dist → import probe — verification reads `dist/`, never PyPI
- [ ] **PUBS-03**: The smoke also runs on a periodic cron schedule, schedule-aware, still never uploading
- [ ] **PUBS-04**: OIDC Trusted Publishing migration via `pypa/gh-action-pypi-publish@release/v1` (job-level `id-token: write`; publisher registered for the exact owner/repo/workflow-filename/environment; no reusable-workflow factoring); `PYPI_API_TOKEN` deleted only after the first observed green OIDC release
- [ ] **PUBS-05**: WINDOWS #20 remediated — local-venv `pytest --cov` dotted-target crash fixed via a local numpy pin or the documented coverage-CLI workaround (not fixable by a coverage version bump)

### Ollama Surface (OLLM)

- [x] **OLLM-01**: All ~59 ollama refs across the 5 inventoried test files (`test_notebook_execution.py`, `test_runner_infra_contracts.py`, `test_extras_guard.py`, `tests/examples/_execution.py`, `test_models_lock_contracts.py`) are dispositioned per-assertion as remove-or-skip with recorded rationale; non-ollama contracts sharing the surface (kernel-plot compat pins, extras brackets, systemd/docs-mirror contracts) preserved
- [x] **OLLM-02**: The census triple is re-pinned, `expected_skips.yaml` updated, and ci.yml example-nightly stages reconciled same-change — nightly stays green (no pytest exit 5, no unexpected-skip audit failure)
- [x] **OLLM-03**: Post-sweep `grep ollama tests/` residue equals a sanctioned list recorded in the plan; the two mcp_example notebooks remain in-repo per WINDOWS #13

### GAP-1 Transformers Compat (GAPC)

- [x] **GAPC-01**: An empirical load run of the NT promoter checkpoint on transformers 5.17 with the current shim set sizes the remaining rung ladder (live probe: exactly `invert_attention_mask` missing)
- [x] **GAPC-02**: Shim-vs-removal is executed on that evidence — vendored absence-gated rung extension if tractable (closes WINDOWS #11/#12/#15); honest removal of the third benchmark model + BPE pkl leg if disproportionate (closes #11/#12 as removed-by-decision)

### Docs Accuracy Review (DOCR)

- [ ] **DOCR-01**: Every file under docs/ and example/ is reviewed per-file for accuracy vs current code and post-numpy2/py311 staleness, with fixes in the same pass (beyond the automated sync/masking gates)
- [ ] **DOCR-02**: Install commands are re-verified against the final extras surface — the sweep lands after the packaging phase so reviewed docs describe the final state
- [ ] **DOCR-03**: `mkdocs --strict` builds with zero warnings (the 15 ledgered pre-existing warnings cleared; nav/mirror interplay respected)

## Future Requirements (post-v1.3)

Deferred, tracked, not in the current roadmap:

- **FUTR-01**: Runtime lane/architecture import-time warning (torch 2.8-style)
- **FUTR-02**: TestPyPI staged-publishing rehearsal with promotion semantics
- **FUTR-03**: pyranges/ruranges re-evaluation if win_amd64 wheels ever ship (trigger already ledgered)
- **FUTR-04**: MOTIF-01 golden fixture activation (issue #44 — fixture-files-only until manuscript values arrive)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Re-freezing the truth chain on numpy | Owner directive: Linux truth stays bedtools-computed, no re-freeze |
| Platform-split logic inside `dnallm/` library code | Dispatch lives at example/test acquisition seams only; library stays platform-uniform |
| Any third interval engine (pyranges/rurunes/bioframe) | Rejected with evidence (no win_amd64 wheels); numpy evaluators suffice for jaccard + loj |
| Global `PretrainedConfig` monkeypatch for GAP-1 | Anti-feature per research; absence-gated rung ladder only |
| Pinning transformers `<5` | Compatibility shim is the sanctioned direction |
| Blanket skip-everything on ollama | Per-assertion disposition required — ever-green-skip is the dishonesty class this project bans |
| Making cudaXXX extras pip-routable via dependency tricks | uv-only index pins documented instead (two-lane honesty) |
| Docs re-organization (e.g. Diátaxis) inside the accuracy pass | Accuracy only; structure is separate work |
| Exact float equality across platform paths | Tolerance pins are the contract (trailing-float deltas by design) |
| Ollama MCP client notebook execution (WINDOWS #13) | Owner exclusion 2026-10-11; window remains open in the ledger |
| MOTIF-01 golden fixture values | Waiting on manuscript inputs (issue #44) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| WPAR-01 | Phase 15 | Complete |
| WPAR-02 | Phase 15 | Complete |
| WPAR-03 | Phase 16 | Pending |
| WPAR-04 | Phase 16 | Pending |
| WPAR-05 | Phase 16 | Pending |
| WPAR-06 | Phase 16 | Pending |
| WPAR-07 | Phase 18 | Pending |
| WPAR-08 | Phase 16 | Pending |
| PKGH-01 | Phase 18 | Pending |
| PKGH-02 | Phase 18 | Pending |
| PKGH-03 | Phase 18 | Pending |
| PKGH-04 | Phase 18 | Pending |
| PKGH-05 | Phase 19 | Pending |
| PKGH-06 | Phase 18 | Pending |
| PKGH-07 | Phase 18 | Pending |
| PKGH-08 | Phase 17 | Pending |
| PUBS-01 | Phase 17 | Pending |
| PUBS-02 | Phase 17 | Pending |
| PUBS-03 | Phase 17 | Pending |
| PUBS-04 | Phase 17 | Pending |
| PUBS-05 | Phase 17 | Pending |
| OLLM-01 | Phase 13 | Complete |
| OLLM-02 | Phase 13 | Complete |
| OLLM-03 | Phase 13 | Complete |
| GAPC-01 | Phase 14 | Complete |
| GAPC-02 | Phase 14 | Complete |
| DOCR-01 | Phase 19 | Pending |
| DOCR-02 | Phase 19 | Pending |
| DOCR-03 | Phase 19 | Pending |

**Coverage:**
- v1.3 requirements: 29 total
- Mapped to phases: 29
- Unmapped: 0 ✓

**Note:** the traceability table above was pre-seeded from the research's phase ordering and validated unchanged at roadmap creation (2026-10-11) — 29/29 requirements mapped to Phases 13–19, no orphans, no duplicates. Cross-category mappings are deliberate: PKGH-08 rides Phase 17 (publish-chain mypy failure class), WPAR-07 rides Phase 18 (atomic extras surgery), PKGH-05 rides Phase 19 (docs describe the final extras surface). See ROADMAP.md.

---
*Requirements defined: 2026-10-11*
*Last updated: 2026-10-11 after roadmap creation (traceability validated against ROADMAP.md Phases 13–19)*
