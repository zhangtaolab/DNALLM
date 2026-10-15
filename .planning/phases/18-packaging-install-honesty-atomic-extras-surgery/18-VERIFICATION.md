---
phase: 18-packaging-install-honesty-atomic-extras-surgery
verified: 2026-10-15T15:42:29Z
status: passed
score: 11/11 must-haves verified
covered_files: [".github/workflows/README.md", ".github/workflows/ci.yml", ".github/workflows/feasibility.yml", ".github/workflows/publish.yml", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-01-PLAN.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-01-SUMMARY.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-02-PLAN.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-02-SUMMARY.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-03-PLAN.md", ".planning/phases/18-packaging-install-honesty-atomic-extras-surgery/18-03-SUMMARY.md", "README.md", "docs/example/marimo/benchmark/benchmark_demo.md", "docs/example/marimo/finetune/finetune_demo.md", "docs/example/marimo/inference/inference_demo.md", "docs/example/notebooks/benchmark.md", "docs/example/notebooks/data_prepare_finetune.md", "docs/example/notebooks/data_prepare_predict.md", "docs/example/notebooks/embedding_attention.md", "docs/example/notebooks/finetune_NER_data_generation.md", "docs/example/notebooks/finetune_NER_task.md", "docs/example/notebooks/finetune_binary.md", "docs/example/notebooks/finetune_custom_head.md", "docs/example/notebooks/finetune_generation.md", "docs/example/notebooks/finetune_multi_labels.md", "docs/example/notebooks/inference.md", "docs/example/notebooks/inference_evo_models.md", "docs/example/notebooks/inference_generation.md", "docs/example/notebooks/inference_megaDNA.md", "docs/example/notebooks/inference_trna.md", "docs/example/notebooks/interpretation.md", "docs/example/notebooks/lora_finetune.md", "docs/example/notebooks/lora_inference.md", "docs/example/notebooks/mutagenesis.md", "docs/example/notebooks/overview.md", "docs/example/notebooks/plant_helixseek_anno.md", "docs/example/notebooks/plant_helixseek_combined.md", "docs/example/notebooks/plant_helixseek_cre.md", "docs/faq/models_troubleshooting.md", "docs/getting_started/gpu_support.md", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "docs/getting_started/quick_start.md", "docs/index.md", "docs/user_guide/getting_started.md", "docs/user_guide/tutorial/end_to_end_workflow.md", "example/notebooks/overview.md", "mkdocs.yml", "pyproject.toml", "tests/models/test_plant_helixseek_fla_kernels.py", "tests/test_extras_guard.py"]
covered_digest: "v3:sha256:62eb619b69407c1309c1a821f0b87fe043735bb1288070e9a8356f514c1c8d0a"
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
---

# Phase 18: Packaging & Install Honesty — Atomic Extras Surgery Verification Report

**Phase Goal:** The extras surface does exactly what its names claim after one atomic, grep-guarded surgery — `base` renamed, cuda121 retired, rocm dispositioned, triton-windows declared — with the GPU support reality documented from the pre-verified boundaries and every changed extra resolving clean in a fresh venv (validated by Phase 17's smoke dispatch against the post-surgery tree)
**Verified:** 2026-10-15T15:42:29Z
**Status:** passed
**Re-verification:** Re-baseline — the prior verification pass was invalidated by phase-19 commits touching covered files (09f9d91, 3a13419, 4693417, 15bf646, 165800d); no previous 18-VERIFICATION.md existed in the tree or git history, so this is a fresh goal-backward verification against the CURRENT tree (v1.3 @ 6b397d1, code surface = 15bf646)

## Goal Achievement

Merged must-haves: ROADMAP Success Criteria 1-5 (the contract) + plan truths from 18-01/18-02/18-03 frontmatter, deduplicated (plan truths restating an SC fold into it). All checks executed goal-backward against the current tree; SUMMARY claims were leads, never evidence.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1/PKGH-01: base→full rename lands atomically — pyproject `full` carries the former base member set verbatim, self-referential `all` chain swapped, one-cycle deprecated alias `base = ["dnallm[full]"]` (v1.4 removal named), CI/publish install lines renamed, guard tests extended, docs swept, ONE commit, stale `.[base]` residue held to a sanctioned list | ✓ VERIFIED | pyproject.toml:148-162: `full = ["dnallm[dev,test,notebook,mcp]", "isort>=6.0.1", "types-transformers>=0.1.0"]` — byte-identical member set to pre-surgery `base` (verified via `git show 399f9f1^:pyproject.toml`); alias comment names the v1.4 removal and the post-removal warn-and-exit-0 semantics; `all = ["dnallm[full,dev,test,notebook,docs,ui,mcp,fla]"]` (pre-surgery: `dnallm[base,...]`). Workflows: all 11 install lines use `.[full...]` (ci.yml ×8, publish.yml:59, feasibility.yml:67) — zero `.[base]` install forms. Pickaxe 3-way re-run live: `-S '"dnallm[full]"'` (pyproject) = `-S '.[full,fla]'` (ci.yml) = `-S 'EXPECTED_FULL_MEMBERS'` (guard test) = **399f9f15ab1790860ac175a87a6eab4905a6849d** — ATOMIC. Residue gate green live (see spot-checks): Pattern-1 measured set == 4-entry sanctioned frozenset (README deprecation note, installation.md note, 2 honesty-page rows) |
| 2 | SC1 alias clause: the alias installs the IDENTICAL full set, exit 0 — no silent under-install (uv AND pip) | ✓ PASSED (override) | uv side live-probed in 18-01 (fresh 3.12 venv from built wheel: `[base]` exit 0, `import dnallm, isort` — full-only member reaches through the alias); pip 26.2.1 mechanism live-probed in 18-RESEARCH (synthetic wheel `base=["pkg[full]"]` resolves the full set, exit 0); METADATA alias line asserted; guard enforces pure self-reference (empty-extra is the forbidden, live-probed under-install shape). Pip-side full-closure resolve on the real closure is the owner-deferred edge (WINDOWS #26, box-network urllib3 stalls) — **override, disclosed** |
| 3 | SC2/PKGH-02: cuda121 retired in the same atomic discipline — extra + `[[tool.uv.index]]` + `[tool.uv.sources]` + conflicts rows + CI matrix + docs in one change; index frozen at torch 2.5.1 recorded | ✓ VERIFIED | Pre-surgery tree (399f9f1^) carried cuda121 extra (:159), torch-cuda121 index (:206), source row (:238); current tree: zero cuda121/torch-cuda121 in pyproject (extras keys = dev/test/notebook/docs/mcp/full/base/all/cpu/cuda124/126/128/130/mamba/fla), uv indexes = pypi/torch-cpu/cuda124/126/128/130 only, sources 5 rows (no cu121), conflicts exactly the 10 pairwise rows. Repo-wide `cuda121` grep outside .planning: only tests/test_extras_guard.py's own gate literal (scan scope excludes tests by design). CI matrix: was `['12.1','12.4']` pre-surgery → single 12.4 leg at 399f9f1 (collapse verified in history) → re-pointed to `['13.0']` by phase-19 commit 165800d (PKGH-05 scope; single-leg structure intact). Frozen-index fact documented: installation.md:268 (FROZEN at torch 2.5.1, re-verified 2026-10-14, dead-lane reasoning), README.md:207, gpu_support.md:15/:42 |
| 4 | SC2/PKGH-03: rocm removed with the restore path documented (no stale facts) | ✓ VERIFIED | No rocm extra/index/source anywhere (pre-surgery had rocm extra :174 + torch-rocm index :231); bracket-rocm residue = exactly the 3 sanctioned fla-UPSTREAM lines (pyproject fla comment, README:256, docs/faq/models_troubleshooting.md:13 — gate-asserted byte-exact, green live). Restore path: installation.md:270 — rocm7.1 (torch 2.10/2.11, Linux x86_64, resolution-tested-only), rocm6.4 tops at 2.9.1, ROCm 7.2+ needs torch ≥2.12 > cap, manual lane, no AMD claim; README:207 points to it. Stale-fact sweep: `grep -rn "no rocm wheel"` over docs/README/example/pyproject/scripts → 0 hits |
| 5 | SC3 declaration half/WPAR-07: triton-windows declared behind `platform_system == 'Windows'` in the fla extra, exact band `>=3.7.1.post27,<3.8`; Linux resolution unaffected | ✓ VERIFIED | pyproject.toml fla member (exact string): `triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'` with the WPAR-07 rationale comment. Guard `test_triton_windows_marker_declared_in_fla` asserts the exact string (green live). Built-wheel METADATA asserted exactly once in setuptools-normalized form (18-01: MARKER-EXACTLY-ONCE-NORMALIZED-OK). Linux marker-false proven in a fresh venv (18-01: triton-windows ABSENT from `uv pip list`) |
| 6 | SC3 acceptance half/WPAR-07: fla KDA smoke re-verified on Windows under the resolved torch | ✓ PASSED (override) | Owner DEFERRED the RTX 5080 checkpoint (稍后再做, 2026-10-14); 18-03 halted with zero content past the gate; WINDOWS #27 carries the six-step resume checklist; WINDOWS #18 open; WPAR-07 honestly unchecked in REQUIREMENTS.md. Adjacent ledgered hardware fact (2026-10-08): KDA chunk kernel compiled+executed at triton-windows 3.7.1.post27 on RTX 5080 sm_120 — only the marker-pull re-proof under the post-surgery tree is pending. **Override, disclosed — owner standing item, not a repo gap** |
| 7 | SC4/PKGH-04: installation.md split into explicit uv/pip lanes with the `[tool.uv.sources]` asymmetry stated before any install table | ✓ VERIFIED | Structure verified live: `## Dependency Groups` (:205) opens with `### Two install lanes: uv and pip — the [tool.uv.sources] asymmetry` (:211) BEFORE `### Core Dependency Groups` (:238, the first table); no table precedes :238. Lane labels counted live: 15 `# [uv lane]` + 7 `# [pip lane]` across Methods 1-3, GPU Support, plain-pip, Windows cu130, Scenarios, troubleshooting, mamba; the asymmetry consequence stated (uv: pinned torch-cuda indexes; pip: default PyPI torch — Linux cu130-bundled, Windows CPU-only). Phase-19 commits (3a13419/4693417/09f9d91) touched this file but preserved every phase-18 structure (residue gate green over the verbatim deprecation note) |
| 8 | SC4/PKGH-06: GPU support matrix published from the pre-verified boundaries — Pascal cpu-only with the Turing minimum-GPU line explicit, 20xx cuda124-cuda128 (cu121-retirement adjustment), 30/40xx every extra, 50xx cuda128+, cuda130 default = Ampere+ | ✓ VERIFIED | docs/getting_started/gpu_support.md (51 lines, substantive): row 1 Pascal "**cpu-only — no GPU extra**… **The minimum GPU line is Turing (20xx)**"; row 2 Turing `cuda124`–`cuda128` with the sm_75-in-cu128 / not-in-cu130 facts; rows 3-4 Ampere/Ada "every hardware extra" (sm_80 same-major compat); row 5 Blackwell `cuda128`+; default-PyPI cu130-bundled = Ampere+ stated with the Windows CPU-only caveat. Verification provenance section names the 0542a6b record recovered verbatim, the local basis (torch 2.11.0+cu130, arch list sm_80/90/100/110/120), and the v1.3 20xx-row narrowing |
| 9 | SC4/PKGH-07: cross-installer honesty matrix page live — command × installer → what you actually get, alias-cycle rows (full set, NO warning) and post-removal rows (warn + exit 0, core-only), nav/README-wired | ✓ VERIFIED | docs/getting_started/install_honesty.md (52 lines, substantive): the five rows with per-row Provenance (`dnallm[full]`; `dnallm[base]` during cycle — full set, exit 0, **no warning**; `dnallm[base]` after v1.4 — pip `WARNING: … does not provide the extra 'base'` / uv `warning: … no extra named 'base'`, core-only, exit 0; `dnallm[cuda130]` uv-pinned vs pip-default; `dnallm[fla]` on Windows marker row). No during-cycle warning claim (Pitfall 2 honored — every "warning" mention post-removal-attributed). Wired: mkdocs.yml:60-61 nav entries + README.md:180 pointer paragraph; row-5 anchor slug confirmed resolving by phase-19's strict build. Same-change sanctioned-list extension green (Pattern-1 = 4 entries, gate-verified live) |
| 10 | SC5: every changed extra resolves clean in a fresh venv, proven by Phase 17's smoke dispatch against the post-surgery tree | ✓ VERIFIED | Live via `gh run view`: **38578071738** (Publish to PyPI, headSha 399f9f15 = the surgery commit, created 2026-10-14T14:09:54Z) — job `smoke` = success, job `publish` = skipped (release gate held; never-uploading). Fresh-venv resolve battery recorded in 18-01-SUMMARY ([full]/[base]-uv/[all]/[fla]/[cuda124] all exit 0; triton-windows absent on Linux; isort reachable through the alias). Re-proof on the FINAL code tree: **38678155010** (headSha 15bf646, 2026-10-15) — smoke success / publish skipped; all commits after 15bf646 are .planning docs, so the current tree's code surface is identical. Honest scope (as disclosed in the SUMMARY): the smoke's install step is extras-free — the fresh-venv resolves are the extras proof, per the 18-CONTEXT-locked validation depth |
| 11 | 18-01 plan truth: D-03 census triple stays 216/217 and the extended guards are green in the fast lane | ✓ VERIFIED | Verifier-executed live on the current tree: `pytest tests/test_extras_guard.py -q` → **11 passed** in 2.75s (4 alias/rename guards + 2 residue repo-scan gates + 5 pre-existing); census `pytest tests/examples --collect-only -q -m "not giants"` → **216/217 tests collected (1 deselected)** — the exact triple. (The guard file's as_posix Windows-path fix, commit 15bf646, post-dates 18-01 and strengthens the same gates; suite green at the phase-19 closeout battery on the identical code surface) |

**Score:** 11/11 truths verified (0 present, behavior-unverified; 2 PASSED via owner-accepted overrides — see Owner-Deferred Standing Items)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | full + alias + all-chain swap; cuda121/rocm deleted; fla marker; uv machinery cleaned | ✓ VERIFIED | All verified line-level (truths 1-5); parsed by the guard tests via tomllib (wired), consumed by every workflow install line |
| `tests/test_extras_guard.py` | member/alias/marker/absence guards + repo-scan residue gates with verbatim sanctioned lists | ✓ VERIFIED | 393 lines substantive; EXPECTED_FULL_MEMBERS / EXPECTED_BASE_ALIAS / EXPECTED_TRITON_WINDOWS_SPEC constants + TestFullExtraAndBaseAlias (4) + TestExtrasSurgeryResidueGates (2, byte-exact (path,line) frozensets); runs in the CI fast lane; **11 passed live** |
| `tests/models/test_plant_helixseek_fla_kernels.py` | base→full literal flips; KDA import contract intact | ✓ VERIFIED | :45/:56/:61-62 literals use `dnallm[full,...]`; chunk_kda importorskip+callable contract present (the 18-03 checkpoint's gating test) |
| `.github/workflows/ci.yml` / `publish.yml` / `feasibility.yml` / `README.md` | install lines renamed; cuda matrix collapsed; comments flipped | ✓ VERIFIED | 11 install sites all `.[full...]`; matrix single leg (12.4 at surgery → 13.0 by phase-19 165800d); zero cu121/rocm |
| `docs/getting_started/installation.md` | lane-split restructure with asymmetry before tables | ✓ VERIFIED | Truth 7 |
| `docs/getting_started/gpu_support.md` | NEW (PKGH-06) | ✓ VERIFIED | Truth 8; nav-wired |
| `docs/getting_started/install_honesty.md` | NEW (PKGH-07) | ✓ VERIFIED | Truth 9; nav-wired |
| `mkdocs.yml` | nav entries for both pages | ✓ VERIFIED | :59-61 (Installation / GPU Support Matrix / Install Honesty Matrix) |
| `README.md` | deprecation note + retired-lane note + page pointers | ✓ VERIFIED | :179 deprecation (v1.4, gate-asserted verbatim), :180 pointers, :207 retired-lanes + restore-path pointer |
| docs sweep surface (27 wrappers + 5 pages + example mirror) | base→full flips, cuda121/rocm row deletions | ✓ VERIFIED | Executably enforced by the residue gates (green live) over pyproject/.github/docs/example/README/scripts; overview.md mirror pair byte-identical (phase-19 truth 2 re-verified) |
| `18-01/02/03-SUMMARY.md` | execution records | ✓ VERIFIED | Present; 18-03 status: halted (honest), requirements-completed deliberately empty |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject alias | `test_base_alias_is_pure_self_reference` | tomllib parse + exact-list assert | ✓ WIRED | Guard green live; the silent-under-install prevention chain (guard + METADATA + uv live proof; pip closure = override) |
| pyproject extra names | the 11 uv-install lines (ci/publish/feasibility) | atomicity: identical introducing commit | ✓ WIRED | Pickaxe 3-way = 399f9f1 (verifier re-run) |
| fla marker | built-wheel METADATA ↔ Windows resolution | 18-03 checkpoint | ✓ WIRED (declaration chain) | Marker declared + guarded + METADATA asserted once; the Windows execution edge is the owner-deferred override (#6) |
| installation.md lane labels | honesty-matrix page | cross-links + shared anchor | ✓ WIRED | gpu_support.md:24 / install_honesty.md:7 link `installation.md#two-install-lanes-uv-and-pip-the-tooluvsources-asymmetry`; slug confirmed resolving in phase-19's `mkdocs build --strict` |
| GPU matrix rows | 0542a6b verified record | provenance section | ✓ WIRED | Page names the record, commit, local basis, and the single v1.3 narrowing; no row re-derived from vendor charts |
| honesty-page `dnallm[base]` rows | residue-gate sanctioned list | same-change frozenset extension | ✓ WIRED | Pattern-1 frozenset carries the 2 page rows verbatim; gate green live (measured == sanctioned, both directions) |

### Data-Flow Trace (Level 4)

N/A — packaging/config/docs phase: no runtime data rendering. The dynamic surfaces are (a) pyproject extras consumed by real installers (proven by the fresh-venv resolves + smoke dispatch) and (b) the residue gates reading the live repo tree (executed green).

### Behavioral Spot-Checks (verifier-executed)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Extras guard suite (the phase's own gate) | `.venv/bin/python -m pytest tests/test_extras_guard.py -q` | 11 passed in 2.75s | ✓ PASS |
| D-03 census triple | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"` | 216/217 tests collected (1 deselected) | ✓ PASS |
| Pickaxe atomicity 3-way | `git log -S` over alias body / ci literal / guard constant | A=B=C=399f9f15ab1790860ac175a87a6eab4905a6849d | ✓ PASS |
| Smoke dispatch on surgery commit | `gh run view 38578071738` (zhangtaolab/DNALLM) | conclusion success @ headSha 399f9f15; jobs: smoke=success, publish=skipped | ✓ PASS |
| Smoke re-proof on final code tree | `gh run view 38678155010` | conclusion success @ headSha 15bf646; smoke=success, publish=skipped | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes are declared for this phase; the validation vehicle is the phase-17 GH smoke dispatch — verified live above (both runs). No MISSING_PROBE.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PKGH-01 | 18-01 | base renamed with one-cycle alias, atomic across all surfaces, residue grep gate | ✓ SATISFIED | Truths 1-2 |
| PKGH-02 | 18-01 | cuda121 retired atomically (extra + uv machinery + docs; index frozen 2.5.1) | ✓ SATISFIED | Truth 3 |
| PKGH-03 | 18-01 | rocm dispositioned on verified facts; restore path documented | ✓ SATISFIED | Truth 4 |
| PKGH-04 | 18-02 | Install docs uv/pip lane split stating the [tool.uv.sources] asymmetry | ✓ SATISFIED | Truth 7 |
| PKGH-06 | 18-02 | GPU support matrix from pre-verified boundaries | ✓ SATISFIED | Truth 8 |
| PKGH-07 | 18-02 | Cross-installer honesty matrix page live | ✓ SATISFIED | Truth 9 |
| WPAR-07 | 18-01 + 18-03 | triton-windows declared behind Windows marker in fla; fla KDA smoke re-verified on Windows | ◐ PARTIAL — declaration SATISFIED; acceptance owner-deferred (override) | Truth 5 verified; truth 6 override (WINDOWS #27). REQUIREMENTS.md honestly shows WPAR-07 `[ ]` Pending — the repo does not claim the acceptance half |

Orphaned requirements: none — PKGH-05 rides Phase 19 and PKGH-08 rides Phase 17 per the REQUIREMENTS.md traceability note; every phase-18-mapped ID is claimed by a plan.

### Decision Coverage

Gate result: skipped — "no trackable decisions" (the CONTEXT `<decisions>` block uses prose headings the tracker does not enumerate; non-blocking by design). Manual check: the three decision clusters all landed — base→full with alias (truth 1), rocm removal with rocm7.1 restore path (truth 4), WPAR-07 declaration + deferred acceptance (truths 5-6, honestly halted).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| docs/getting_started/installation.md | 595 | `2.x.x+cuXXX` matches the XXX debt-marker regex | ℹ️ Info (false positive) | Torch version-suffix FORMAT example in user-facing prose (e.g. `2.5.1+cu124`), not a debt marker |

No TBD/FIXME/XXX debt markers, no TODO/HACK/PLACEHOLDER, no stub implementations, no console.log-only paths in any phase-18 file. The residue discipline is executably enforced (gates green live), which is stronger than a one-shot grep.

### Human Verification Required

None emitted — infrastructure/packaging phase with no user-facing runtime to hand-test. The two items that need owner-side execution (Windows hardware; healthy-network host) are already tracked as owner standing items and are represented as accepted overrides below rather than new UAT items.

### Owner-Deferred Standing Items (disclosed, not gaps)

Both are recorded in `.planning/WINDOWS.md` with resume paths; neither is closable from this Linux box, and the in-repo contract each depends on is fully delivered and guarded:

1. **WINDOWS #26 — pip-side alias full-closure resolve (unrun-verify).** pip's urllib3 stack stalls against this box's network (5 attempts); mechanism probe-proven (pip 26.2.1, synthetic wheel, 18-RESEARCH), METADATA alias line asserted, uv-side parity green, pure-self-reference guard enforced. Re-run on a healthy-network host: `uv venv --seed /tmp/xp --python 3.12 && /tmp/xp/bin/python -m pip install "$(ls dist/*.whl)[base]" && /tmp/xp/bin/python -c "import dnallm, isort"`.
2. **WINDOWS #27 — Windows fla KDA acceptance (owner 稍后再做, 2026-10-14).** Six-step RTX 5080 checklist carried in 18-03-SUMMARY.md Task 1 Record and the ledger row; resume via `/gsd-execute-phase 18` with the captured values. WPAR-07 stays Pending in REQUIREMENTS.md and WINDOWS #18 stays open until then — the record is honest, nothing silently dropped.

Note (info-level): install_honesty.md row 5's provenance cell says "the Windows pull was empirically executed at triton-windows 3.7.1.post27 on an RTX 5080 (WINDOWS ledger #18)" — the cited ledger row records the 2026-10-08 MANUAL install + kernel execution at that exact version; the marker-PULL path under the post-surgery tree is precisely what #27 defers. The row cites the ledger openly, but a future edit could sharpen "the pull" to "the version" for exactness.

### Gaps Summary

None. All five ROADMAP success criteria are delivered and verified against the current tree (the phase-19 doc commits that invalidated the prior pass preserved every phase-18 structure — re-proven executably by the live guard/census runs and the green smoke dispatches). The phase goal — an extras surface that does exactly what its names claim, atomically with honest guidance — holds on the current tree. The two open edges are owner-side standing items with ledger tracking and re-run paths, accepted as overrides, not repo gaps.

---

_Verified: 2026-10-15T15:42:29Z_
_Verifier: Claude (gsd-verifier)_
