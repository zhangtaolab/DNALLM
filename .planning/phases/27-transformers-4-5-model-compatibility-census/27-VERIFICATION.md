---
phase: 27-transformers-4-5-model-compatibility-census
verified: 2026-10-28T10:04:43Z
status: passed
score: 6/6 must-haves verified
covered_files:
  - .github/workflows/census_compat.yml
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-01-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-01-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-02-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-02-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-03-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-03-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-04-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-04-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-05-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-05-SUMMARY.md
  - docs/user_guide/model_compatibility.md
  - scripts/census_compat.py
  - tests/models/test_compat_docs_consistency.py
  - tests/test_runner_infra_contracts.py
covered_digest: "v3:sha256:1086b8c62444950d702f0db83667f3defcf2f63e95941f50e32ae00bcd11c2e8"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 25/25
  gaps_closed:
    - "G-27-1 (blocker): census lane green-capable — sanctioned mamba-kernel install (pip bootstrap + sha256-pinned scripts/install_mamba.sh) committed in the workflow, pinned offline by TestCensusLaneInstallSequence (13 passed incl. the WR-01 continue-on-error pin), and the lane's exact command proven green on the runner-box replica (49 matched / 0 UNEXPECTED / agreement line / exit 0 in .scratch/p27-uat/lane-sim/census-green.log)"
    - "G-27-2 (minor): all six remote-code-broken rows lead with honest per-class plain-language sentences, technical cause demoted to a bounded (<=100-char) parenthetical, every status cell <=300 chars (max 285), drift gate extended to 68 tests with independent re-derivation, page regenerates byte-identically"
    - "Post-close review fixes verified: CR-01 (53d0017 — post-checkout mkdir -p census-artifacts before the seeded cp, structurally asserted) and WR-01 (44b0cf6 — test_no_continue_on_error_anywhere pin)"
    - "Prior human item 2 (biologist readability): resolved by owner triage recorded in 27-UAT.md and the 27-UI-REVIEW.md re-audit — warnings (a)/(c) CLOSED, (b) explicitly accepted as-is"
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "One live gh workflow run census_compat.yml dispatch executes the full lane end-to-end on the self-hosted runner (checkout-clean -> mkdir -> seeded cp -> census -> verify) and finishes green"
    addressed_in: "Post v1.4->main merge — a documented operational act, not a phase (27-05-PLAN user_setup_note; 27-05-SUMMARY User Setup section)"
    evidence: "Structurally impossible during the v1.4 cycle: GitHub registers dispatchable workflows from the default branch only and main is frozen (live dispatch attempt returned HTTP 404, recorded in 27-UAT.md test 1). Every component is proven by other means: the census command by the runner-box replica green run, the install sequence + posture by committed contract tests, and the CR-01 mkdir ordering by deterministic actions/checkout v4.2.2 behavior plus the feasibility.yml:96 precedent on the same box. 27-REVIEW-FIX.md records the owner follow-up: dispatch once after merge to confirm end-to-end green."
---

# Phase 27: Transformers 4/5 Model Compatibility Census Verification Report (Re-verification)

**Phase Goal:** Per-model transformers 4/5 compatibility matrix for the registry's 49 pretrained entries with import-level evidence; every failure dispositioned (compat shim or typed-skip); verdicts as typed registry fields; generated plain-language docs table locked against drift.
**Verified:** 2026-10-28T10:04:43Z
**Status:** passed
**Re-verification:** Yes — after gap closure (prior status human_needed -> UAT found G-27-1/G-27-2 -> plan 27-05 -> post-close review fixes 53d0017/44b0cf6 -> this verification)

## Re-verification Mode and Method

Prior 27-VERIFICATION.md (2026-10-27) verified 25/25 truths with status human_needed (2 human items). The human UAT cycle then ran: item 1 (live dispatch) was executed by delegated verification — the dispatch itself returned HTTP 404 (proving the structural constraint: dispatchable workflows register from the default branch only, and main is frozen for v1.4), and the lane's exact command was replicated on the runner box instead, exposing the real gaps G-27-1 (lane deterministically red: missing mamba kernels) and G-27-2 (jargon-led failure cells). Plan 27-05 closed both; an incremental code review (27-REVIEW.md) then found CR-01/WR-01, both fixed (53d0017/44b0cf6); a UI re-audit (27-UI-REVIEW.md) confirmed the plain-language closures.

This verification re-ran the gap-closure must-haves at full depth (every command re-executed by the verifier, no SUMMARY claims taken on trust) and regression-checked the prior 25 truths.

## Goal Achievement

### Observable Truths (gap-closure must-haves, full-depth verification)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | G-27-1: on unchanged inputs, the exact census lane command (seeded copy + `--run --with-loads --verify`) exits 0 with all 49 entries matched, with the mamba kernels installed via the sanctioned pipeline (runner-box replica evidence) | ✓ VERIFIED | `.scratch/p27-uat/lane-sim/census-green.log` re-read by verifier: `grep -c "^  matched "` = **49**, `grep -c UNEXPECTED` = **0**, `OK: census matches committed registry fields` present, `LANE-EXIT-CODE: 0` present, wall clock 302 s. All five previously-disagreeing mamba-family entries now matched (caduceus-ph `verified-tf5`, caduceus-ps `import-verified-tf5`, PlantCaduceus_l32 `shim-required-tf5`, PlantCAD2 `verified-tf5`, PlantBiMoE `verified-tf5`) — direct contrast with the red baseline `census-run.log` where the same five show `import-fail` / `UNEXPECTED ... unmappable` / exit 1. This is recorded execution evidence from the actual runner box (same hardware/network/venv shape as the lane), not a presence inference |
| 2 | G-27-1: the census workflow installs the mamba kernels between the `.[full]` install and the census run via pip bootstrap + scripts/install_mamba.sh (sha256-pinned, TUNA index); an editorial revert or masking fails a committed offline contract test in seconds | ✓ VERIFIED | Workflow lines 89-90: `uv pip install --python .venv/bin/python pip` then `bash scripts/install_mamba.sh --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`, ordered between `uv pip install -e ".[full]"` (line 70) and the census run (line 133); invocation matches the proven ci.yml:392 precedent byte-for-byte; install_mamba.sh exists (13.7 KB, sha256-pinned sdist + c++20 patches per its header). `TestCensusLaneInstallSequence` (5 tests) pins ordering, exact invocation, no `|| true` masking on executed lines, no `continue-on-error` anywhere (WR-01), and the D-08 posture — verifier re-ran `pytest tests/test_runner_infra_contracts.py -q`: **13 passed** (8 pre-existing + 5 census pins). Deliberate-red vs pre-change workflow text recorded in 27-05-SUMMARY |
| 3 | G-27-1: the lane's dispatch-only posture is preserved exactly — workflow_dispatch only, contents: read, event-name guard, 240-minute timeout; only the install sequence changed | ✓ VERIFIED | Verifier re-ran the plan's structural assertion (`yaml.safe_load`, pyyaml `True`-key gotcha handled): triggers == {workflow_dispatch}, permissions == {contents: read}, job guard/runner/timeout intact, no continue-on-error at job or step level. Commit a0c7d5f is pure addition (+20 workflow / +114 test, 0 deletions); CR-01 (53d0017) added +10 lines (post-checkout `mkdir -p census-artifacts` at line 131, before the seeded `cp` at line 132 — ordering structurally asserted by the verifier) |
| 4 | G-27-2: every remote-code-broken row's What-we-found cell leads with one plain-language sentence a biologist can act on (honest per failure class); technical cause only as a short parenthetical suffix | ✓ VERIFIED | All six broken rows inspected on the committed page: DNABERT-2 + GPN → library-change lead ("the new library version removed or changed something its code depends on..."), GENA-LM → damaged-files lead ("the published model files themselves are damaged..."), MutBERT → own-defect lead ("the model's own code has a defect..."), GENA-LM-BigBird + Omni-DNA → external-package lead ("its code needs an extra software package that DNALLM does not install..."). A version-independent model defect is never worded as a library removal (D-05 honesty). Generator side: `DOCS_BROKEN_LEADS` + `DOCS_BROKEN_LEAD_CLASSES` (most-specific-first, 6 keys grounded in the committed evidence) at scripts/census_compat.py:1617-1649; `docs_broken_lead_class` raises ValueError on unmapped shapes — verifier exercised this live: unmapped evidence -> `ValueError: broken evidence matches no plain-language lead class`, all four classes map correctly, `docs_bound_technical_fragment` truncates 150->101 chars at a word boundary with a single trailing ellipsis |
| 5 | G-27-2: no What-we-found cell exceeds 300 characters; the plain-language lead portions contain none of the cited jargon tokens | ✓ VERIFIED | Verifier measured every row programmatically: max status cell **285** chars (cap 300); six broken cells 230-285 chars; technical suffixes 51-98 chars (bound 100). Leads (text before ` (technical: `) read plainly with zero engineering tokens; the gate's `LEAD_JARGON_BLOCKLIST` (ALiBi, meta, tensor, torch, token_type_ids, deepspeed, olmo, input_shape, [:-1], __init__, ...) is enforced by a committed test |
| 6 | G-27-2: the drift gate re-derives the new wording independently (no generator imports); page regenerated by the generator only; DOCS_DATA_DATE untouched | ✓ VERIFIED | Verifier re-ran `pytest tests/models/test_compat_docs_consistency.py -q`: **68 passed** (baseline 54). Gate imports only `dnallm.models.registry` (grep-verified: no import from scripts/census_compat.py) and carries its own copies of the class map, leads, suffix bound (100), cell cap (300), and jargon blocklist. Verifier re-ran the regeneration: `--emit-docs /tmp/mc-verify27.md && cmp` -> **BYTE-IDENTICAL**. `DOCS_DATA_DATE = "2026-10-27"` (scripts/census_compat.py:1585) — presentation-only change, data date unmoved |

**Score:** 6/6 gap-closure truths verified (0 present-but-behavior-unverified)

### Prior Truths Regression Check (25 truths from initial verification)

All 25 previously-verified truths still hold — quick-regression evidence, re-executed by this verifier:

- Registry state unchanged: 236 total / 49 pretrained, statuses 19 verified-tf5 / 21 import-verified-tf5 / 1 shim-required-tf5 / 6 remote-code-broken / 2 census-blocked, single tested stamp 5.17.0, zero empty evidence (live loader assertion)
- Evidence artifact intact: `tests/models/compat_census.json` carries 49 records
- Test suites green: `tests/models/test_registry.py` + `test_model_info_registry.py` + `tests/utils/test_transformers_compat_census.py` = **88 passed**; `tests/models/test_census_compat.py` = **83 passed** (this file was also touched by the G-27-2 gate extension — all pins still green)
- Compat rungs 12-13 still defined and registered in `apply_patches()` (transformers_compat.py:1534/1604, registered 1697-1698)
- Docs nav entry intact (mkdocs.yml:68); page header/footnotes/49 rows/counts sentence unchanged apart from the six G-27-2 cells
- Workflow carries zero huggingface.co URLs, loud gpu_absent fail-closed path, `if: always()` artifact upload

No regressions.

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | One live `gh workflow run census_compat.yml` dispatch executing the full lane end-to-end (checkout-clean -> mkdir -> seeded cp -> census -> verify) on the self-hosted runner | Post v1.4->main merge — documented operational act, not a phase | Structurally impossible now (dispatch returns HTTP 404: GitHub registers dispatchable workflows from the default branch only; main frozen for v1.4 — proven in the UAT cycle and recorded in 27-UAT.md test 1 and 27-05-PLAN's user_setup_note). Component evidence all present: replica green run (truth 1), contract pins (truth 2), deterministic CR-01 ordering fix + feasibility.yml:96 precedent (truth 3). 27-REVIEW-FIX.md names the owner follow-up. Not a gap: the phase contract (27-05 verification section) designates the replica run as G-27-1's verification |

### Advisory (New Scope, Unevidenced)

Re-verification ran; no unevidenced new-scope findings. Recorded advisory items (owner-dispositioned, info-severity, non-blocking):

| # | Finding | Source | Disposition |
|---|---------|--------|-------------|
| 1 | 6 open Info findings (IN-01 mid-clause truncation unbalanced parens; IN-02 summary counts not gate-pinned; IN-03 new helpers lack direct edge tests; IN-04 kwargs-only wrapper edge; IN-05 unpinned uv installer; IN-06 docs_blocked_reason fail-open edge) | 27-REVIEW-DISPOSITION.md (6/10 open, all info) | Owner scope decision — open for a future cycle |
| 2 | UI re-audit suggestions beyond the closed warnings: no visual weight differentiation of failure rows (bold/grouping); mid-clause technical suffix endings | 27-UI-REVIEW.md Top-3 (21/24 overall; prior warnings (a)/(c) CLOSED, (b) owner-deferred) | Cosmetic improvements to a plain-Markdown deliverable; no must-have violated |
| 3 | The CR-01 post-checkout `mkdir` has no dedicated offline test pin (the WR-01 pin covers continue-on-error; the no-masking pin covers `|| true`) | This verifier's observation | Fail-closed by construction: removing the mkdir makes the seeded `cp` fail loudly (red lane), never a silent green; caught on the deferred live dispatch. Recorded for the owner; not a gap against any 27-05 must-have |

### Required Artifacts (gap-closure set)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/census_compat.yml` | Sanctioned mamba install inside the existing install step; CR-01 post-checkout mkdir; posture untouched | ✓ VERIFIED | Lines 89-90 install sequence; line 131 mkdir; structural posture assertion re-run |
| `tests/test_runner_infra_contracts.py` | TestCensusLaneInstallSequence (ordering, exact invocation, no masking, no continue-on-error, posture) | ✓ VERIFIED | 5 census tests; file total 13 passed in verifier's run |
| `scripts/census_compat.py` | Plain-language lead layer + bounded technical suffix, fail-closed | ✓ VERIFIED | DOCS_BROKEN_LEADS/CLASSES + docs_broken_lead_class (ValueError verified live) + docs_bound_technical_fragment (word-boundary truncation verified live) |
| `tests/models/test_compat_docs_consistency.py` | Independent lockstep gate: class map, suffix bound, 300-char cap, jargon blocklist | ✓ VERIFIED | 68 passed; no generator import |
| `docs/user_guide/model_compatibility.md` | Regenerated page, six failure rows lead plain | ✓ VERIFIED | Byte-identical regen re-proven by verifier; 49 rows; max cell 285 |
| `.scratch/p27-uat/lane-sim/census-green.log` | Replica green-run evidence (scratch, not committed) | ✓ VERIFIED | Present on disk with all four required markers (49 matched / 0 UNEXPECTED / agreement line / exit 0) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| Workflow install step | `scripts/install_mamba.sh` | pip bootstrap then `bash scripts/install_mamba.sh --python .venv/bin/python --index-url <TUNA>` | ✓ WIRED | Invocation identical to the proven ci.yml:392 mamba lane; sha256 pin inside the script is the integrity control |
| Contract test | Committed census_compat.yml text | offline text/YAML pins | ✓ WIRED | 13 passed; revert of any pinned property fails in seconds (deliberate-red recorded in 27-05-SUMMARY) |
| Generator `docs_status_cell` | Gate `_expected_status_cell` | independent re-derivation (duplicate constants, no shared import) | ✓ WIRED | Identical 6-key class maps in both files; 68 gate tests green; byte-identical regen proves the derivation |
| Replica venv | Lane environment | same box (edgexpert-296a = dnallm-nightly), same install pipeline, same command | ✓ WIRED | Red/green log contrast isolates the single variable (mamba kernels); green run used the lane's exact command with sanitized env |
| Registry -> docs page | `dnallm.models.registry` accessors | generator emit path | ✓ WIRED | Regression re-check: loader distribution 19/21/1/6/2 matches the page's count sentence; byte-identity |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| census-green.log | 49 matched verdicts | Real hub fetches + subprocess probes + CUDA load/forward runs in the replica venv (torch 2.14.1+cu130, transformers 5.17.0, mamba_ssm 2.3.2.post1) | Yes | ✓ FLOWING |
| model_compatibility.md | 49 rows + failure leads | Typed registry accessors via generator; leads keyed on registry evidence substrings | Yes | ✓ FLOWING |
| Census workflow | install + run sequence | Committed YAML, pinned by offline contract tests | Yes (offline-proven; live execution deferred) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Lane contract pins | `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py -q` | 13 passed in 0.26s | ✓ PASS |
| Docs drift gate | `.venv/bin/python -m pytest tests/models/test_compat_docs_consistency.py -q` | 68 passed in 0.32s | ✓ PASS |
| Census tool offline tests | `.venv/bin/python -m pytest tests/models/test_census_compat.py -q` | 83 passed in 0.33s | ✓ PASS |
| Registry + rung suites (regression) | `.venv/bin/python -m pytest tests/models/test_registry.py tests/models/test_model_info_registry.py tests/utils/test_transformers_compat_census.py -q` | 88 passed in 3.41s | ✓ PASS |
| Byte-identical docs regen | `--emit-docs /tmp && cmp` | BYTE-IDENTICAL | ✓ PASS |
| Generator fail-closed (live) | `docs_broken_lead_class` on unmapped evidence | `ValueError: broken evidence matches no plain-language lead class` | ✓ PASS |
| Generator class mapping (live) | 4 class-map probes over representative evidence strings | all 4 classes correct | ✓ PASS |
| Suffix bounding (live) | `docs_bound_technical_fragment` on 150-char input | 101 chars, word boundary, single trailing ellipsis | ✓ PASS |
| Cell-cap measurement (live) | programmatic scan of all 49 page rows | max status cell 285; tech suffixes 51-98 | ✓ PASS |
| Workflow posture + mkdir ordering (live) | yaml.safe_load structural assertion | posture OK; mkdir precedes cp; no continue-on-error | ✓ PASS |
| Replica green-log markers (live) | grep counts on census-green.log | 49 matched / 0 UNEXPECTED / agreement / LANE-EXIT-CODE: 0 | ✓ PASS |
| Red-vs-green contrast (live) | grep on census-run.log vs census-green.log | caduceus-ph: import-fail+UNEXPECTED (red) vs import-ok+matched (green) | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared. The phase's probe surface is the census tool and the replica lane run — both exercised above; the replica log re-verified line-by-line.

### Requirements Coverage

Owner-directed phase (`requirements: ["TBD"]`, no REQUIREMENTS.md mapping per the tasking — not cross-referenced). Judged against the phase goal and the 27-05 gap-closure contract: goal achieved, both UAT gaps closed with committed, revert-proofed artifacts and empirical evidence.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers across all 5 files modified in this cycle | — | Debt-marker gate clean |

Stub check: no stubs, no empty implementations, no masked failures (the no-masking and no-continue-on-error pins are themselves committed tests), no hand-edits to the generated page (byte-identity proves it).

### Human Verification Required

None remaining. Both prior human items are resolved by recorded evidence:

1. **Live dispatch** — acted on in the UAT cycle: the dispatch attempt proved the structural constraint (HTTP 404; default-branch registration + main freeze), and the delegated verification produced the replica green run instead, which is the gap's contract-designated verification (27-05-PLAN verification section). The residual one-dispatch-after-merge is recorded under Deferred Items with its component evidence; it cannot be performed by anyone during the v1.4 freeze, so routing the phase to human_needed for it would stall the milestone on an impossible act the phase contract explicitly scopes out.
2. **Biologist readability** — resolved by owner action: the UAT read the rendered page against the plain-language directive, triaged the three warnings ((a) and (c) fixed by 27-05 and confirmed CLOSED by the 27-UI-REVIEW.md re-audit; (b) footnote markers explicitly accepted as-is, page-wide footnotes carry the caveat).

### Gaps Summary

No gaps. Both UAT gaps are closed and verified at full depth by independently re-executed evidence:

- **G-27-1 (blocker):** the census lane can now exit 0 — its install sequence carries the sanctioned sha256-pinned mamba-kernel pipeline (committed, revert-proofed by 5 offline contract tests, posture byte-preserved), and the lane's exact command was proven green on the runner box itself (49 matched / 0 UNEXPECTED / exit 0, with the red baseline log isolating the single-variable root cause). The post-close CR-01 checkout-clean defect is fixed (mkdir precedes the seeded cp) and WR-01's continue-on-error masking vector is pinned.
- **G-27-2 (minor):** the six failure rows lead with honest per-class plain-language sentences a biologist can act on; the technical cause survives only as a bounded parenthetical (<=100 chars, word boundary); every status cell is <=300 chars (max 285); generator, gate (68 tests), and page are in lockstep with checker independence; the page regenerates byte-identically.

The prior 25 truths show zero regressions. Score 6/6 on the gap-closure must-haves; 25/25 prior truths hold. One deferred operational act (single live dispatch after the v1.4->main merge) is recorded in the frontmatter with its evidence trail — it is the end-to-end confirmation of components already proven individually, not a code gap.

---

_Verified: 2026-10-28T10:04:43Z_
_Verifier: Claude (gsd-verifier)_
