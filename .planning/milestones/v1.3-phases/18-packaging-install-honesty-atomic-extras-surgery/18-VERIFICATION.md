---
phase: 18-packaging-install-honesty-atomic-extras-surgery
verified: 2026-10-16T00:29:01Z
status: passed
score: 11/11 must-haves verified
covered_files: [".github/workflows/README.md", ".github/workflows/ci.yml", ".github/workflows/feasibility.yml", ".github/workflows/publish.yml", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-01-PLAN.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-01-SUMMARY.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-02-PLAN.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-02-SUMMARY.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-03-PLAN.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-03-SUMMARY.md", "README.md", "docs/example/marimo/benchmark/benchmark_demo.md", "docs/example/marimo/finetune/finetune_demo.md", "docs/example/marimo/inference/inference_demo.md", "docs/example/notebooks/benchmark.md", "docs/example/notebooks/data_prepare_finetune.md", "docs/example/notebooks/data_prepare_predict.md", "docs/example/notebooks/embedding_attention.md", "docs/example/notebooks/finetune_NER_data_generation.md", "docs/example/notebooks/finetune_NER_task.md", "docs/example/notebooks/finetune_binary.md", "docs/example/notebooks/finetune_custom_head.md", "docs/example/notebooks/finetune_generation.md", "docs/example/notebooks/finetune_multi_labels.md", "docs/example/notebooks/inference.md", "docs/example/notebooks/inference_evo_models.md", "docs/example/notebooks/inference_generation.md", "docs/example/notebooks/inference_megaDNA.md", "docs/example/notebooks/inference_trna.md", "docs/example/notebooks/interpretation.md", "docs/example/notebooks/lora_finetune.md", "docs/example/notebooks/lora_inference.md", "docs/example/notebooks/mutagenesis.md", "docs/example/notebooks/overview.md", "docs/example/notebooks/plant_helixseek_anno.md", "docs/example/notebooks/plant_helixseek_combined.md", "docs/example/notebooks/plant_helixseek_cre.md", "docs/faq/models_troubleshooting.md", "docs/getting_started/gpu_support.md", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "docs/getting_started/quick_start.md", "docs/index.md", "docs/user_guide/getting_started.md", "docs/user_guide/tutorial/end_to_end_workflow.md", "example/notebooks/overview.md", "mkdocs.yml", "pyproject.toml", "tests/models/test_plant_helixseek_fla_kernels.py", "tests/test_extras_guard.py"]
covered_digest: "v3:sha256:bdba0588f36bf98087f8304f522ddfaba99007728ff49a49ceb9db3d4da48323"
behavior_unverified: 0
overrides_applied: 2
overrides:
  - must_have: "Installing the base alias yields the IDENTICAL full member set under uv AND pip, exit 0, no under-install (pip-side full-closure resolve edge of SC1/PKGH-01)"
    reason: "Owner-deferred unrun-verify: pip's urllib3 stack stalled against this box's network across 5 attempts (direct PyPI x2, tuna mirror bulk+resolver stall, dual-index metadata crawl) while uv/gh/curl were unaffected — a box-network artifact, not a repo defect. Mechanism live-probed in 18-RESEARCH (pip 26.2.1 resolves wheel[base]->pkg[full] full set, exit 0, synthetic wheel); built-wheel METADATA alias line 'dnallm[full]; extra == \"base\"' asserted; pure-self-reference enforced by test_base_alias_is_pure_self_reference; uv-side parity green on the identical member set (isort reachable through [base]). Exact re-run command recorded in WINDOWS ledger #26."
    accepted_by: "owner (Tao Zhang) — deferral recorded in .planning/WINDOWS.md #26 and 18-01-SUMMARY.md (D5)"
    accepted_at: "2026-10-14T14:13:29Z"
  - must_have: "The fla KDA smoke is re-verified on Windows under the resolved torch (WPAR-07 acceptance half of SC3)"
    reason: "Owner explicitly DEFERRED the Windows hardware checkpoint (稍后再做, coordinator-relayed 2026-10-14 ~14:5xZ): requires the owner's RTX 5080 box, which no Linux verifier can produce. Carried as WINDOWS ledger #27 with the six-step resume checklist; 18-03 halted with zero content past the gate; WPAR-07 honestly left unchecked in REQUIREMENTS.md; WINDOWS #18 left open. The in-repo declaration contract is fully delivered and guarded (exact marker string in fla, METADATA asserted exactly once in normalized form, Linux marker-false proven in a fresh venv). Adjacent real-hardware fact stands: a KDA chunk kernel compiled+executed at triton-windows 3.7.1.post27 on RTX 5080 sm_120, ledgered 2026-10-08 (WINDOWS #18) — only the marker-pull re-proof under the post-surgery tree is pending."
    accepted_by: "owner (Tao Zhang) — deferral recorded in .planning/WINDOWS.md #27 and 18-03-SUMMARY.md (halted state)"
    accepted_at: "2026-10-14T15:49:00Z"
re_verification:
  previous_status: passed
  previous_score: 11/11
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 18: Packaging & Install Honesty — Atomic Extras Surgery Verification Report

**Phase Goal:** The extras surface does exactly what its names claim after one atomic, grep-guarded surgery — `base` renamed, cuda121 retired, rocm dispositioned, triton-windows declared — with the GPU support reality documented from the pre-verified boundaries and every changed extra resolving clean in a fresh venv (validated by Phase 17's smoke dispatch against the post-surgery tree)
**Verified:** 2026-10-16T00:29:01Z
**Status:** passed
**Re-verification:** Re-baseline #2 — the prior pass (2026-10-15T15:42:29Z, passed 11/11 @ tree 6b397d1) went fingerprint-stale when phase-20's sanctioned debt closeout (commit 0655411) touched exactly two phase-18 covered files: `docs/faq/models_troubleshooting.md` (line-13 legacy `[cuda]`/`[rocm]` mention re-worded to explicitly not-current + GPU Support Matrix pointer) and `tests/test_extras_guard.py` (`SANCTIONED_CUDA121_ROCM_RESIDUE` frozenset entry re-pinned verbatim in the same commit). Every check below was re-executed goal-backward against the CURRENT tree (v1.3 @ 1a61356). Delta since the prior pass, per `git log --since`: `.planning` docs + phase-20's c3c65cd/48f9905 (dnallm/cli/* + tests/cli/* — CLI exit-code fixes, NOT phase-18 covered) + 0655411 (faq, usage.md, task_guides.md, guard — only the first and last are phase-18 covered). No other phase-18 covered file moved.

## Goal Achievement

Merged must-haves: ROADMAP Success Criteria 1-5 (the contract) + plan truths from 18-01/18-02/18-03 frontmatter, deduplicated (plan truths restating an SC fold into it). All checks executed goal-backward against the current tree; SUMMARY claims were leads, never evidence.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1/PKGH-01: base→full rename lands atomically — pyproject `full` carries the former base member set verbatim, self-referential `all` chain swapped, one-cycle deprecated alias `base = ["dnallm[full]"]` (v1.4 removal named), CI/publish install lines renamed, guard tests extended, docs swept, ONE commit, stale `.[base]` residue held to a sanctioned list | ✓ VERIFIED | Re-measured live on HEAD: pyproject.toml:148-162 — `full = ["dnallm[dev,test,notebook,mcp]", "isort>=6.0.1", "types-transformers>=0.1.0"]`, `base = ["dnallm[full]"]` with the v1.4-removal + post-removal warn-and-exit-0 comment, `all = ["dnallm[full,dev,test,notebook,docs,ui,mcp,fla]"]`. Zero `.[base]` install forms across .github/workflows (grep, 0 hits). Pickaxe 3-way re-run live this pass: `-S '"dnallm[full]"'` (pyproject) = `-S '.[full,fla]'` (ci.yml) = `-S 'EXPECTED_FULL_MEMBERS'` (guard) = **399f9f15ab1790860ac175a87a6eab4905a6849d** — ATOMIC. Pattern-1 residue gate green live (part of the 11-passed run): measured set == 4-entry sanctioned frozenset (README deprecation note, installation.md note, 2 honesty-page rows). Phase 20 did not touch this surface (pyproject.toml, workflows, README all unchanged since the prior verified pass) |
| 2 | SC1 alias clause: the alias installs the IDENTICAL full set, exit 0 — no silent under-install (uv AND pip) | ✓ PASSED (override) | uv side live-probed in 18-01 (fresh 3.12 venv from built wheel: `[base]` exit 0, `import dnallm, isort`); pip 26.2.1 mechanism live-probed in 18-RESEARCH (synthetic wheel `base=["pkg[full]"]` resolves the full set, exit 0); METADATA alias line asserted; guard enforces pure self-reference (re-run green this pass). Pip-side full-closure resolve on the real closure remains the owner-deferred edge — **override carried forward unchanged** (WINDOWS #26 still `open`, re-checked this pass) |
| 3 | SC2/PKGH-02: cuda121 retired in the same atomic discipline — extra + `[[tool.uv.index]]` + `[tool.uv.sources]` + conflicts rows + CI matrix + docs in one change; index frozen at torch 2.5.1 recorded | ✓ VERIFIED | Re-measured live on HEAD: zero cuda121/torch-cuda121 in pyproject (grep — the ONLY cuda121/rocm-adjacent hit is the sanctioned fla-upstream comment at :185, which is itself an entry of SANCTIONED_CUDA121_ROCM_RESIDUE); extras keys = full/base/all/cpu/cuda124/cuda126/cuda128/cuda130/mamba/fla (+ dev/test/notebook/docs/ui/mcp); uv indexes exactly 6 (pypi, torch-cpu, cuda124/126/128/130); sources exactly 5 torch rows; conflicts exactly the 10 pairwise rows. Frozen-index fact stands in installation.md/README/gpu_support.md (all unchanged since the prior pass). Phase 20 did not touch this surface |
| 4 | SC2/PKGH-03: rocm removed with the restore path documented (no stale facts) | ✓ VERIFIED | No rocm extra/index/source anywhere in pyproject (grep). Bracket-rocm residue = exactly the 3 sanctioned fla-UPSTREAM lines — gate-asserted byte-exact and green live this pass (part of the 11-passed run), now including the phase-20 re-pinned faq entry (see truth 11 note). Restore path documented at installation.md:270 (file unchanged since the prior pass — git filter empty). Stale-fact sweep re-run: `grep -rn "no rocm wheel"` over docs/README/example/pyproject/scripts → 0 hits |
| 5 | SC3 declaration half/WPAR-07: triton-windows declared behind `platform_system == 'Windows'` in the fla extra, exact band `>=3.7.1.post27,<3.8`; Linux resolution unaffected | ✓ VERIFIED | Re-read live on HEAD: pyproject.toml fla member is the exact string `triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'` with the WPAR-07 rationale comment. Guard `test_triton_windows_marker_declared_in_fla` asserts the exact string (green live this pass); Linux marker-false proven in a fresh venv (18-01). Phase 20 did not touch this surface |
| 6 | SC3 acceptance half/WPAR-07: fla KDA smoke re-verified on Windows under the resolved torch | ✓ PASSED (override) | Owner DEFERRED the RTX 5080 checkpoint (稍后再做, 2026-10-14); 18-03 halted with zero content past the gate; WINDOWS #27 still `open` with the six-step resume checklist (re-checked this pass); WINDOWS #18 open; REQUIREMENTS.md honestly shows WPAR-07 `[ ]` Pending (restored 2026-10-15 by 30dde10 after phase.complete over-flipped it — re-checked on HEAD). Adjacent ledgered hardware fact (2026-10-08): KDA chunk kernel compiled+executed at triton-windows 3.7.1.post27 on RTX 5080 sm_120. **Override carried forward unchanged — owner standing item, not a repo gap** |
| 7 | SC4/PKGH-04: installation.md split into explicit uv/pip lanes with the `[tool.uv.sources]` asymmetry stated before any install table | ✓ VERIFIED | File byte-unchanged since the prior verified pass (git filter `15bf646..HEAD -- docs/getting_started/` → empty apart from nothing; no commit since the prior pass touched it). The lane-split structure and 15+7 lane labels were verified line-level in that pass; the verbatim deprecation note (a Pattern-1 sanctioned entry) is re-proven executably green by this pass's live guard run |
| 8 | SC4/PKGH-06: GPU support matrix published from the pre-verified boundaries — Pascal cpu-only with the Turing minimum-GPU line explicit, 20xx cuda124-cuda128 (cu121-retirement adjustment), 30/40xx every extra, 50xx cuda128+, cuda130 default = Ampere+ | ✓ VERIFIED | docs/getting_started/gpu_support.md present (3077 bytes, unchanged since the prior pass); nav-wired at mkdocs.yml:60 (re-checked). Content verified against the boundaries in the prior pass; additionally, this file is now the TARGET of the phase-20 faq pointer (../getting_started/gpu_support.md) — resolves to a real page |
| 9 | SC4/PKGH-07: cross-installer honesty matrix page live — command × installer → what you actually get, alias-cycle rows (full set, NO warning) and post-removal rows (warn + exit 0, core-only), nav/README-wired | ✓ VERIFIED | docs/getting_started/install_honesty.md present (unchanged since the prior pass); nav-wired at mkdocs.yml:61 + README pointer (re-checked). The two `dnallm[base]` rows are Pattern-1 sanctioned entries — byte-re-proven green by this pass's live guard run |
| 10 | SC5: every changed extra resolves clean in a fresh venv, proven by Phase 17's smoke dispatch against the post-surgery tree | ✓ VERIFIED | Both citations re-queried live via `gh run view` this pass: **38578071738** (headSha 399f9f15 = the surgery commit) — conclusion success, smoke=success, publish=skipped; **38678155010** (headSha 15bf646) — conclusion success, smoke=success, publish=skipped. Citations remain valid for the current tree: `git log 15bf646..HEAD -- pyproject.toml` is EMPTY (the extras surface the smoke built+resolved is byte-identical), publish.yml unchanged, and the smoke job's steps are build-wheel → fresh-venv bare install → `import dnallm` (no CLI invocation — read at publish.yml:81-127), so phase-20's post-15bf646 CLI exit-code fixes cannot affect it. Honest scope (as disclosed in 18-01-SUMMARY, unchanged): the smoke's install step is extras-free — the fresh-venv per-extra resolves are the extras proof, per the 18-CONTEXT-locked validation depth |
| 11 | 18-01 plan truth: D-03 census triple stays 216/217 and the extended guards are green in the fast lane | ✓ VERIFIED | Verifier-executed live on HEAD this pass: `pytest tests/test_extras_guard.py -q` → **11 passed** in 1.73s (4 alias/rename guards + 2 residue repo-scan gates + 5 pre-existing); census `pytest tests/examples --collect-only -q -m "not giants"` → **216/217 tests collected (1 deselected)**. **Phase-20 delta on this truth (the re-baseline's focus):** commit 0655411 re-worded faq line 13 to mark the v0.5-era `[cuda]`/`[rocm]` extras as "legacy names no longer on the current install surface" + a page-relative `[GPU Support Matrix](../getting_started/gpu_support.md)` pointer (target exists), and re-pinned the SANCTIONED_CUDA121_ROCM_RESIDUE frozenset entry verbatim in the SAME commit — the byte-coupling discipline held; the gate's measured==sanctioned assertion (both directions) is green live, proving the current faq line IS the sanctioned line. The edit STRENGTHENS the phase-18 honesty contract: the old phrasing could read as if `[cuda]`/`[rocm]` were current installables; the new one explicitly denies currency and points to the current lanes |

**Score:** 11/11 truths verified (0 present, behavior-unverified; 2 PASSED via owner-accepted overrides — see Owner-Deferred Standing Items)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | full + alias + all-chain swap; cuda121/rocm deleted; fla marker; uv machinery cleaned | ✓ VERIFIED | Re-read live on HEAD (truths 1-5); unchanged by phase 20; parsed by the guard tests via tomllib (wired), consumed by every workflow install line |
| `tests/test_extras_guard.py` | member/alias/marker/absence guards + repo-scan residue gates with verbatim sanctioned lists | ✓ VERIFIED | 404 lines substantive after the phase-20 re-pin (+11/-4); EXPECTED_FULL_MEMBERS / EXPECTED_BASE_ALIAS / EXPECTED_TRITON_WINDOWS_SPEC constants + TestFullExtraAndBaseAlias (4) + TestExtrasSurgeryResidueGates (2, byte-exact (path,line) frozensets incl. the re-pinned faq entry); runs in the CI fast lane; **11 passed live this pass** |
| `docs/faq/models_troubleshooting.md` | sanctioned legacy-extras mention, byte-coupled to the guard frozenset | ✓ VERIFIED | Line 13 single-line mention now explicitly not-current with the GPU Support Matrix pointer; verbatim member of SANCTIONED_CUDA121_ROCM_RESIDUE (gate green live, both directions); same-commit re-pin in 0655411 |
| `tests/models/test_plant_helixseek_fla_kernels.py` | base→full literal flips; KDA import contract intact | ✓ VERIFIED | Unchanged since the prior pass (git filter); :45/:56/:61-62 literals use `dnallm[full,...]`; chunk_kda importorskip+callable contract present |
| `.github/workflows/ci.yml` / `publish.yml` / `feasibility.yml` / `README.md` | install lines renamed; cuda matrix collapsed; comments flipped | ✓ VERIFIED | Unchanged since the prior verified pass (only pre-prior-pass bdf6ba3 touched ci.yml — nightly venv pins, already measured then); 11 install sites all `.[full...]`; zero cu121/rocm (grep) |
| `docs/getting_started/installation.md` | lane-split restructure with asymmetry before tables | ✓ VERIFIED | Unchanged since the prior pass (truth 7) |
| `docs/getting_started/gpu_support.md` | NEW (PKGH-06) | ✓ VERIFIED | Present, nav-wired (truth 8); now also the faq pointer target |
| `docs/getting_started/install_honesty.md` | NEW (PKGH-07) | ✓ VERIFIED | Present, nav-wired (truth 9) |
| `mkdocs.yml` | nav entries for both pages | ✓ VERIFIED | :59-61 re-checked (Installation / GPU Support Matrix / Install Honesty Matrix); faq page at :235 |
| `README.md` | deprecation note + retired-lane note + page pointers | ✓ VERIFIED | Unchanged since the prior pass; the deprecation note is a Pattern-1 sanctioned entry re-proven verbatim green by the live guard run |
| docs sweep surface (27 wrappers + 5 pages + example mirror) | base→full flips, cuda121/rocm row deletions | ✓ VERIFIED | Executably enforced by the residue gates (green live) over pyproject/.github/docs/example/README/scripts; no covered sweep file moved since the prior pass |
| `18-01/02/03-SUMMARY.md` | execution records | ✓ VERIFIED | Present (auto-covered by the fingerprint); 18-03 status: halted (honest), requirements-completed deliberately empty |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject alias | `test_base_alias_is_pure_self_reference` | tomllib parse + exact-list assert | ✓ WIRED | Guard green live this pass; the silent-under-install prevention chain intact |
| pyproject extra names | the 11 uv-install lines (ci/publish/feasibility) | atomicity: identical introducing commit | ✓ WIRED | Pickaxe 3-way re-run this pass = 399f9f1 |
| faq legacy-extras line | `SANCTIONED_CUDA121_ROCM_RESIDUE` frozenset | byte-coupled (path,line) pin, same-commit re-pin | ✓ WIRED | Gate's measured==sanctioned assertion green live in BOTH directions — the current faq line IS the pinned line; both edits landed in commit 0655411 (re-confirmed via `git show`) |
| faq GPU Support Matrix pointer | `docs/getting_started/gpu_support.md` | page-relative markdown link | ✓ WIRED | `../getting_started/gpu_support.md` resolves to the existing 3077-byte page (PKGH-06 artifact, nav-wired) |
| fla marker | built-wheel METADATA ↔ Windows resolution | 18-03 checkpoint | ✓ WIRED (declaration chain) | Marker declared + guarded (green live); the Windows execution edge remains the owner-deferred override (#6) |
| installation.md lane labels | honesty-matrix page | cross-links + shared anchor | ✓ WIRED | Both files unchanged since the prior pass; slug confirmed resolving in phase-19's `mkdocs build --strict` and phase-20's re-run (0 warnings) |
| honesty-page `dnallm[base]` rows | residue-gate sanctioned list | same-change frozenset extension | ✓ WIRED | Pattern-1 frozenset carries the 2 page rows verbatim; gate green live |

### Data-Flow Trace (Level 4)

N/A — packaging/config/docs phase: no runtime data rendering. The dynamic surfaces are (a) pyproject extras consumed by real installers (proven by the fresh-venv resolves + smoke dispatch, re-validated via gh this pass) and (b) the residue gates reading the live repo tree (executed green this pass).

### Behavioral Spot-Checks (verifier-executed, live on HEAD 1a61356)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Extras guard suite (the phase's own gate — the sanctioned-residue gate is the direct check on the phase-20 faq/guard delta) | `.venv/bin/python -m pytest tests/test_extras_guard.py -q` | 11 passed in 1.73s | ✓ PASS |
| D-03 census triple | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"` | 216/217 tests collected (1 deselected) in 0.85s | ✓ PASS |
| Pickaxe atomicity 3-way | `git log -S` over alias body / ci literal / guard constant | A=B=C=399f9f15ab1790860ac175a87a6eab4905a6849d | ✓ PASS |
| Smoke dispatch on surgery commit | `gh run view 38578071738` (zhangtaolab/DNALLM) | conclusion success @ headSha 399f9f15; jobs: smoke=success, publish=skipped | ✓ PASS |
| Smoke re-proof on the extras-identical code tree | `gh run view 38678155010` | conclusion success @ headSha 15bf646; smoke=success, publish=skipped; pyproject.toml byte-identical 15bf646..HEAD | ✓ PASS |
| faq↔guard byte-coupling after the phase-20 re-pin | guard Pattern-2 residue assertion (part of the 11-passed run) | measured set == sanctioned frozenset, both directions (extra and missing empty) | ✓ PASS |
| Stale rocm-fact sweep | `grep -rn "no rocm wheel"` over docs/README/example/pyproject/scripts | 0 hits | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes are declared for this phase; the validation vehicle is the phase-17 GH smoke dispatch — both runs re-verified live above. No MISSING_PROBE.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PKGH-01 | 18-01 | base renamed with one-cycle alias, atomic across all surfaces, residue grep gate | ✓ SATISFIED | Truths 1-2 (re-measured on HEAD) |
| PKGH-02 | 18-01 | cuda121 retired atomically (extra + uv machinery + docs; index frozen 2.5.1) | ✓ SATISFIED | Truth 3 (re-measured on HEAD) |
| PKGH-03 | 18-01 | rocm dispositioned on verified facts; restore path documented | ✓ SATISFIED | Truth 4 (re-measured on HEAD; faq mention now explicitly not-current — honesty strengthened) |
| PKGH-04 | 18-02 | Install docs uv/pip lane split stating the [tool.uv.sources] asymmetry | ✓ SATISFIED | Truth 7 (file unchanged; verbatim note gate-green) |
| PKGH-06 | 18-02 | GPU support matrix from pre-verified boundaries | ✓ SATISFIED | Truth 8 (unchanged; now the faq pointer target) |
| PKGH-07 | 18-02 | Cross-installer honesty matrix page live | ✓ SATISFIED | Truth 9 (unchanged; rows verbatim gate-green) |
| WPAR-07 | 18-01 + 18-03 | triton-windows declared behind Windows marker in fla; fla KDA smoke re-verified on Windows | ◐ PARTIAL — declaration SATISFIED; acceptance owner-deferred (override) | Truth 5 verified on HEAD; truth 6 override (WINDOWS #27 still open — re-checked). REQUIREMENTS.md honestly shows WPAR-07 `[ ]` Pending (traceability row: "Pending (owner-deferred #27)") — the repo does not claim the acceptance half |

Orphaned requirements: none — PKGH-05 rides Phase 19 and PKGH-08 rides Phase 17 per the REQUIREMENTS.md traceability note; every phase-18-mapped ID is claimed by a plan. REQUIREMENTS.md statuses re-read on HEAD: PKGH-01..04/06/07 `[x]` Complete, WPAR-07 `[ ]` Pending.

### Decision Coverage

Gate result: skipped — "no trackable decisions" (the CONTEXT `<decisions>` block uses prose headings the tracker does not enumerate; re-run this pass, non-blocking by design). Manual check: the three decision clusters all landed — base→full with alias (truth 1), rocm removal with rocm7.1 restore path (truth 4), WPAR-07 declaration + deferred acceptance (truths 5-6, honestly halted).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| docs/getting_started/installation.md | 595 | `2.x.x+cuXXX` matches the XXX debt-marker regex | ℹ️ Info (false positive) | Torch version-suffix FORMAT example in user-facing prose (e.g. `2.5.1+cu124`), not a debt marker — line re-checked, unchanged |

No TBD/FIXME/XXX debt markers, no TODO/HACK/PLACEHOLDER, no stub implementations in any phase-18 covered file. The two files moved since the prior pass (faq, guard) were re-scanned clean this pass. The residue discipline is executably enforced (gates green live), which is stronger than a one-shot grep.

### Advisory (New Scope, Unevidenced)

Re-verification ran; the Step 7 scan surfaced no new-scope findings.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | No new-scope, unevidenced findings this pass |

### Human Verification Required

None emitted — infrastructure/packaging phase with no user-facing runtime to hand-test. The two items that need owner-side execution (Windows hardware; healthy-network host) are already tracked as owner standing items and are represented as accepted overrides below rather than new UAT items.

### Owner-Deferred Standing Items (disclosed, not gaps)

Both are recorded in `.planning/WINDOWS.md` (re-checked `open` on HEAD this pass) with resume paths; neither is closable from this Linux box, and the in-repo contract each depends on is fully delivered and guarded:

1. **WINDOWS #26 — pip-side alias full-closure resolve (unrun-verify).** pip's urllib3 stack stalls against this box's network (5 attempts); mechanism probe-proven (pip 26.2.1, synthetic wheel, 18-RESEARCH), METADATA alias line asserted, uv-side parity green, pure-self-reference guard enforced (re-run green this pass). Re-run on a healthy-network host: `uv venv --seed /tmp/xp --python 3.12 && /tmp/xp/bin/python -m pip install "$(ls dist/*.whl)[base]" && /tmp/xp/bin/python -c "import dnallm, isort"`.
2. **WINDOWS #27 — Windows fla KDA acceptance (owner 稍后再做, 2026-10-14).** Six-step RTX 5080 checklist carried in 18-03-SUMMARY.md Task 1 Record and the ledger row; resume via `/gsd-execute-phase 18` with the captured values. WPAR-07 stays Pending in REQUIREMENTS.md and WINDOWS #18 stays open until then — the record is honest, nothing silently dropped.

Note (info-level, carried from the prior pass, still accurate): install_honesty.md row 5's provenance cell cites WINDOWS ledger #18's 2026-10-08 manual install + kernel execution at triton-windows 3.7.1.post27; the marker-PULL path under the post-surgery tree is precisely what #27 defers.

### Gaps Summary

None. All five ROADMAP success criteria are delivered and verified against the current tree (v1.3 @ 1a61356). The phase-20 delta that invalidated the prior fingerprint — commit 0655411's faq line-13 honesty strengthening plus the same-commit verbatim re-pin of the guard's sanctioned frozenset — preserves and strengthens the phase-18 contract: the legacy `[cuda]`/`[rocm]` mention is now explicitly not-current with a pointer to the current GPU lanes, the byte-coupling discipline held in one commit, the residue gates are green live (11 passed), the census holds at 216/217, and both smoke dispatch citations re-validate (the extras surface is byte-identical to the smoke-verified tree). The pyproject extras surgery surface is untouched by phase 20 (full + alias + all-chain, no cuda121/rocm, triton-windows Windows-marker in fla). The two open edges remain owner-side standing items with ledger tracking and re-run paths, accepted as overrides, not repo gaps.

---

_Verified: 2026-10-16T00:29:01Z_
_Verifier: Claude (gsd-verifier)_
