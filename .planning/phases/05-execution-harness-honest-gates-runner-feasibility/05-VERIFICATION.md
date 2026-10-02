---
phase: 05-execution-harness-honest-gates-runner-feasibility
verified: 2026-10-02T12:05:00Z
status: human_needed
score: 25/25 must-haves verified
covered_files:
  - .github/workflows/docs-validation.yml
  - .github/workflows/feasibility.yml
  - .gitignore
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-01-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-01-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-02-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-02-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-03-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-03-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-04-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-04-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-05-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-05-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-06-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-06-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CONTEXT.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-FIX.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
  - README.md
  - dnallm/utils/transformers_compat.py
  - example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - pyproject.toml
  - scripts/check_docs_sync.py
  - scripts/feasibility/spike_families.py
  - tests/examples/_execution.py
  - tests/examples/test_marimo_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_script_execution.py
  - tests/expected_skips.yaml
  - tests/models/test_model_remote_code.py
  - tests/utils/test_transformers_compat.py
covered_digest: "v2:sha256:cee4c73bd52aedae4abba5658b13ebcb324a356bf5ba55b847ee73cd01fbe755"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "tests/examples/conftest.py provides the locally-scoped notebook_sandbox fixture (05-01 artifact)"
    reason: "Post-wave integration fix: tests/ is not a package, so a bare tests/examples/conftest.py won the conftest module-name race and broke 'from conftest import ...' in test_trainer/test_benchmark/test_dna_dataset. The fixture was relocated module-locally to tests/examples/test_notebook_execution.py with an explanatory comment; in-phase consumption and tree-clean teardown are delivered identically. Recreating the conftest would re-break three test files."
    accepted_by: "owner (Tao Zhang) — 05-UAT item 3, acknowledged 2026-10-02 at closure"
    accepted_at: "2026-10-02T04:50:00+08:00"
re_verification:
  previous_status: gaps_found
  previous_score: 19/19 (prior closure) + Post-closure Gap Addendum GAP-1/GAP-2
  gaps_closed:
    - "GAP-1 (D-07): gated pruning-helper shim landed in dnallm/utils/transformers_compat.py (vendored v4.49.0 helpers, absence-gated attach to modeling_utils AND pytorch_utils per review WR-01, sentinel idempotence, wired into apply_patches()); live-probed on transformers 5.17.0 by this verifier; real-model smoke exists and terminated at the ladder's designed rung (typed environment-unavailable skip with exact traceback — the designed honest outcome per the addendum's residual-risk clause and the 05-04 plan done-branch); benchmark notebook carried as census FAIL row + WINDOWS ledger entry"
    - "GAP-2 (D-08): committed full-tree census 05-CENSUS.md — every Table A gate re-run green by this verifier (0 pending, 21/3/1 counts vs live tree, 25 rows, verdict tally 11 PASS / 12 FAIL / 2 deferred-owner); every verdict traces to manifest.json (28 outcomes) + 26 on-disk logs + probe records; durable layer green-or-typed-skipped (7 gated tests re-run with live probe evidence; pilot re-run passed; fast leg 1648/1 green)"
    - "D-09: plans 05-01..03 untouched (git log 8fa9e05..HEAD on the three PLAN files is empty); gap plans' REQ claims match REQUIREMENTS.md states (REPAIR-03 partial/unchecked, EXEC-04 open/unchecked); all six plans' commits present on phs; .scratch/ zero tracked files; tests/examples/conftest.py still absent (override honored)"
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "05-FEASIBILITY.md Runner-confirmation column is filled from an actual dispatch run of feasibility.yml on the self-hosted GB10 runner (verdicts become official per D-04)"
    addressed_in: "post-merge integration window (phs → dev → main) — first dispatch opportunity after feasibility.yml lands on the default branch
    evidence: "Platform constraint proven live by the prior verifier (dispatch API HTTP 404; file absent on origin/main+origin/dev). Still current: the phs range remains unpushed (git log origin/dev..phs = 82 commits at verification time, manual-push-only rule honored), so the merge event that registers the workflow has not occurred. 05-UAT item 2 records the same deferral."
human_verification:
  - test: "Decide the disposition of the local ModelScope NT snapshot left in its orchestrator-patched state, then re-run `.venv/bin/python -m pytest tests/models/test_model_remote_code.py -q`"
    expected: "The NT v2 promoter smoke returns to green-or-typed-skipped. On THIS box it currently runs RED (verified by this verifier: 1 failed in 6.67s, `AttributeError: 'EsmModel' object has no attribute 'get_extended_attention_mask'`) because the patched snapshot (config.json is_decoder/add_cross_attention + modeling_esm.py init_weights→post_init; backups at `*.dnallm-bak`) moves the load past the smoke's structural marker to the forward-stage rung — exactly the rung the census benchmark row predicted. On a clean cache (CI) the smoke skips typed as designed."
    why_human: "The root cause is a deliberately-kept out-of-band cache mutation documented in the census ('decision: keep patched — richer evidence'), not committed code. The options (restore from *.dnallm-bak to restore the typed skip; widen the structural marker to the forward-stage rung; or land the Phase-8 NT disposition) are owner dispositions among sanctioned alternatives — a verifier must not mutate the owner's model cache."
  - test: "Run the slow script lane once against the real network: `.venv/bin/python -m pytest tests/examples/test_script_execution.py -q` (or wait for the nightly)"
    expected: "Either 1 passed (rice inputs download, generate_bpe_dataset.py runs to its NT structural rung) or 1 skipped network-unavailable carrying the exact transport error. The WR-04 fix's 4xx-re-raise branch must NOT fire."
    why_human: "The WR-04 review fix (6453ece) split the except chain so permanent HTTP 4xx re-raises instead of converting to an ever-green skip; its semantics were verified only via a routing replica because the real slow-lane download was not executed (fix report flags it 'requires human verification'). This verifier probed both rice URLs (HTTP 200 both — the 4xx path is latent, not firing), but only a real run exercises the full path."
---

# Phase 5: Execution Harness, Honest Gates & Runner Feasibility Verification Report

**Phase Goal:** A trustworthy private execution harness exists and is proven (including kernel-kill on hang); both false-green CI gates are closed together with the docs-mirror drift they were hiding; and the runner's real capabilities for the environment-gated model families are settled in writing before execution tests are written against them — PLUS the reopened fifth success criterion (D-07/D-08/D-09, 2026-10-02 post-closure gap closure)
**Verified:** 2026-10-02T12:05:00Z
**Status:** human_needed (25/25 truths verified; 2 items routed to the owner — neither is a committed-code defect)
**Re-verification:** Yes — third round. Origin: the prior 05-VERIFICATION.md `gaps_found` Post-closure Gap Addendum (GAP-1/GAP-2, reopen command `/gsd-plan-phase 5 --gaps --force`); its full text is preserved in git history (last state at commit 821e184's parent) and is the closure contract verified below.

## Re-Verification Scope

GAP-1 and GAP-2 (plus D-09) received full fresh verification — code reads, live probes, and test
re-runs executed by this verifier, none taken from SUMMARY claims. The 19 prior-closure truths
received quick regression: key greps re-run green, and the two behavior-dependent harness tests
(kernel-kill, partial-failure) were **re-run fresh** because `tests/examples/_execution.py`
changed substantially during gap closure (21-spec growth, delta-zero baseline, subprocess lanes).
Branch protection was re-read live via the GitHub API by this verifier.

Standing context honored: milestone is manual-push-only (82 unpushed commits on phs expected);
the four dirty notebooks (mcp pair, benchmark, inference) are the owner-sanctioned baseline —
`check_docs_sync.py` DIFFERs are exactly those four and nothing else.

## Goal Achievement

### Observable Truths

Truths 1-19 are the prior-closure set (quick regression, evidence re-collected this round).
Truths 20-25 are the reopened fifth success criterion (full verification).

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pilot notebook executes end-to-end through run_notebook() with kernel cwd inside a tmp_path sandbox copy (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round**: `TestNotebookExecution::test_notebook_executes_end_to_end[notebooks/inference/inference.ipynb]` — 1 passed in 11.48s; zero tree delta after the run (scoped status unchanged from owner baseline) |
| 2 | Scoped tree clean after execution (SC1/EXEC-01) | ✓ VERIFIED | Delta-zero vs baseline design (05-05); verifier's own runs this round left the scoped tree at exactly the 4 owner-baseline lines |
| 3 | On cell error/timeout the harness writes partial executed notebook + exception text before re-raising (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round** (fresh, post-05-05 harness changes): `TestPartialFailureArtifacts` — passed (part of 2 passed in 4.83s) |
| 4 | Timeout layering: per-cell strictly below per-test mark (SC1/EXEC-01) | ✓ VERIFIED | Spec ladder re-read (cell timeouts 600-3600; class marks 3600/7200); WR-02 fix re-verified in code (`_TIMEOUT_7200_GATED` frozenset covers finetune_custom_head AND lora_finetune) |
| 5 | Kernel shutdown guaranteed via shutdown_kernel="immediate" + plain client.execute() (SC1/EXEC-01) | ✓ VERIFIED | Re-grepped in `_execution.py`; zero kernels leaked by this verifier's runs (the 7-9 live ipykernel processes on the box all predate/postdate verifier activity — owner's live Jupyter session) |
| 6 | Harness private to its test tree (EXEC-01) | ✓ VERIFIED | Underscore module; sole test-tree consumers; no root-conftest/dnallm importers |
| 7 | Kill test: hung kernel killed, no surviving ipykernel_launcher (SC2/EXEC-06) | ✓ VERIFIED | **Re-run this round (fresh)**: `TestKernelLifecycle` — passed (4.83s combined run); goal-level kernel-kill proof remains green on the current harness |
| 8 | check_docs_sync.py honest, wrapper-.md relaxation scoped to right_only (SC3/REPAIR-02) | ✓ VERIFIED | Re-run this round: DIFFERs are EXACTLY the 4 owner-baseline notebooks (sanctioned); all other mirror files sync-clean |
| 9 | Mirror byte-identical after resync (SC3/REPAIR-02/D-03) | ✓ VERIFIED | Non-baseline mirror content diff-clean this round; the 4 DIFFERs are owner execution-output churn outside the gates (standing context) |
| 10 | REPAIR-02 edges: fail-closed absent dirs, single-prefix reporting | ✓ VERIFIED | Carried (script unchanged in gap closure; behavior verified at prior closure) |
| 11 | docs-validation honest: zero continue-on-error, born green (SC3/CI-01/D-01) | ✓ VERIFIED | Re-grepped: `continue-on-error` count 0; job `docs-validation` present |
| 12 | docs-validation installs .[test,dev,mcp]; README documents proven install line (SC3/CI-02) | ✓ VERIFIED | Workflow line 42 + README.md:497 re-grepped this round |
| 13 | Branch protection on dev and main lists BOTH required contexts (CI-01/D-02) | ✓ VERIFIED | **Live read-only re-verification by this verifier (2026-10-02T12:0xZ)**: `gh api` on both branches returns `coverage-gate (py3.12, fast leg)` + `docs-validation` — unchanged since the prior closure's PUTs |
| 14 | Written verdict matrix, every row carries measured evidence (SC4/FEAS-01/D-05) | ✓ VERIFIED | 05-FEASIBILITY.md untouched by gap closure; all 8 spike logs re-listed on disk this round |
| 15 | Verdicts against exact notebook variants via real forward; pyBigWig real round-trip (D-05) | ✓ VERIFIED | Carried (matrix + logs unchanged; census re-recorded the fallback legs fresh: evo1-8k generate OK, evo2-noFP8 generate OK, megadna pinned forward OK) |
| 16 | Every non-FEASIBLE verdict shows both attempts with recorded failure text (D-06) | ✓ VERIFIED | Carried; census Gated-ladder table re-records the D-05/D-06 ordering walked variant-first |
| 17 | Spike ran in throwaway venv; project env untouched (D-04) | ✓ VERIFIED | Re-proven this round: stripedhyena/evo2/MEGABYTE_pytorch/pyBigWig/langchain_ollama/mamba_ssm/megaDNA ALL absent from .venv; pyproject porcelain-empty |
| 18 | Dispatch-gated runner confirmation job exists + documented + hand-off (D-04) | ✓ VERIFIED | Re-grepped: `on: workflow_dispatch` only, runs-on [self-hosted, dnallm-nightly], event gate, timeout 240, if: always() upload, permissions block; official dispatch stays post-merge (deferred item 1) |
| 19 | marimo flavor decided with evidence; pyBigWig not added to pyproject (FEAS-01) | ✓ VERIFIED | Export-html flavor proven by 3 green marimo census rows; pyBigWig count in pyproject = 0 |
| 20 | D-07 shim: vendored v4.49.0 pruning helpers, absence-gated attach (4.x no-op), wired into apply_patches() | ✓ VERIFIED | Full code read (`transformers_compat.py:74-377`); **live probe on transformers 5.17.0 by this verifier**: modeling_utils exposes BOTH helpers as the identity of the vendored functions, sentinel set, idempotent under repeat apply_patches(); pytorch_utils gained `find_pruneable_heads_and_indices` (vendored identity) while its NATIVE `prune_linear_layer` is untouched (never-overwrite contract); pruning arithmetic spot-checked ({1} / rows [0,1,4,5,6,7]) |
| 21 | D-07 smoke: real-model load+forward attempt with the ladder-terminal typed skip as the designed honest outcome | ✓ VERIFIED | `tests/models/test_model_remote_code.py` read: slow+timeout(1800), real `load_model_and_tokenizer` (source="modelscope"), forward with (1,2)-logits asserts, structural-marker gate (`'EsmConfig' object has no attribute 'is_decoder'`) — only that documented rung skips, anything else re-raises; commit-time behavior "1 skipped (typed), exit 0" recorded in 05-04; census carries the benchmark FAIL row and WINDOWS ledger entries 11-12 record the owner disposition. **Live finding this round (routed to human item 1)**: on THIS box the smoke currently runs RED (1 failed in 6.67s) because the local NT ModelScope snapshot was left patched — the load moves past the marker and dies at the forward-stage `get_extended_attention_mask` rung, exactly as the census benchmark row predicted; the designed live-probe behavior (loud failure on non-marker breakage) is functioning |
| 22 | D-08 census: every Table A item carries a verdict, nothing pending, nothing silently omitted | ✓ VERIFIED | **All Table C gates re-run by this verifier**: `grep -c '\| pending'` = 0; `pending-owner` string = 0; awk Table A ipynb rows 21 == live 21; marimo rows 3 == live 3; script rows 1; Table A data rows 25; Table B data rows 34; tracked example files 57; on-disk non-log files 57 == tracked 57; every Table A path exists on disk (existence walk: zero MISSING); verdict tally 11 PASS / 12 FAIL / 2 deferred-owner |
| 23 | D-08 evidence: every verdict traces to recorded evidence; ladder visible in gated rows | ✓ VERIFIED | manifest.json on disk: 28 item outcomes (14 pass incl. 3 spike legs / 12 fail / 2 pending-owner); 26 per-item logs under `.scratch/census-out/logs/` all present; probe records complete (ollama GREEN qwen3.8:latest with full model JSON, mcp-server-endpoint connection-refused, 5 import-absent probes, GPU GB10); Gated-ladder evidence table cites the spike fallback logs; FAIL rows carry exact terminal one-liners + file:line |
| 24 | D-08 durable wiring: green set active, gated set probe-then-skip, suite green-or-typed-skipped | ✓ VERIFIED | ACTIVE_NOTEBOOKS x8 + GATED_NOTEBOOKS x7 with gate functions read in code; **re-run this round**: TestGatedNotebookExecution — 7 skipped, each message carrying LIVE probe results (ollama GREEN / MCP endpoint refused / find_spec None); fast leg — 1648 passed, 1 pre-existing skip, 49 deselected; compat contract tests — 32 passed; campaign-time full tests/examples 107 passed / 9 skipped recorded in 05-06 (not re-run in full: ~48 min GPU campaign; component re-verification above) |
| 25 | D-09: plans 05-01..03 untouched; gap plans carry honest REQ claims | ✓ VERIFIED | `git log 8fa9e05..HEAD -- 05-0{1,2,3}-PLAN.md` empty; REQUIREMENTS.md cross-check: REPAIR-03 unchecked/Phase-8 Pending == "partial" claim, EXEC-04 unchecked/Pending == "open" claim, EXEC-01/06/CI-01/CI-02/REPAIR-02/FEAS-01 complete, EXEC-03 checked complete on the documented dev-box basis (census §4; nightly leg arrives with Phase 8 SC1); all six plans' commits present (05-01:8, 05-02:4, 05-03:12, 05-04:8 incl. RED-first ebdf482, 05-05:5, 05-06:4); `git ls-files .scratch/` = 0 |

**Score:** 25/25 truths verified (0 present-behavior-unverified; 2 items routed to human verification — an environment-state decision and a network-lane check, neither a committed-code defect)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Matrix Runner-confirmation column filled from an actual dispatch run of feasibility.yml (D-04 official-verdict step) | Post-merge integration (phs → dev → main) | Platform constraint live-proven by the prior verifier (dispatch API 404; file absent on default branch). Still current: 82-commit phs range unpushed (manual-push-only honored), so the registering merge has not occurred. Same event flips the remote's docs-validation copy honest. |
| 2 | Census FAIL repair queue (12 rows) + 2 deferred-owner rows + Phase 7-9 rescoping | Phase 8 (owner decisions, flagged in 05-CENSUS.md Hand-off §1-§5) | The census hand-off explicitly parks these as owner/Phase-8 work per D-09 — they are the designed deliverable (an honest worklist), not unmet must-haves of this phase. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/utils/transformers_compat.py` | vendored helpers + gated dual-module attach wired into apply_patches() | ✓ VERIFIED | Read in full; WR-01 dual-module structure (`_attach_remote_code_pruning_helpers` per module, per-name absence gate, per-module sentinel); live-probed on 5.17.0 |
| `tests/models/test_model_remote_code.py` | slow real-model load+forward smoke with structural-marker typed skip | ✓ VERIFIED | Read in full; GAP-1 provenance documented; current-box behavior routed to human item 1 (environment-state, not code) |
| `tests/utils/test_transformers_compat.py` | behavior-contract tests for the vendored helpers | ✓ VERIFIED | 32 passed this round (26 + 6 WR-01 tests incl. synthetic-module routing) |
| `tests/examples/_execution.py` | 21 notebook specs + 3 marimo specs + 3 runner lanes + typed-skip helpers | ✓ VERIFIED | Read; `str(EXAMPLE_DIR` literal count 24 = 21 notebooks + 3 marimo (dispatch's "22" was the 05-05 state; 05-06 grew marimo to 3 per plan); all spec keys exist on disk |
| `tests/examples/test_notebook_execution.py` | ACTIVE x8 + GATED x7 + kill + partial-failure | ✓ VERIFIED | Read + re-run (pilot 11.48s pass; gated 7 evidence-bearing skips; kill+partial 4.83s pass) |
| `tests/examples/test_marimo_execution.py` | parametrized over all 3 census-green apps | ✓ VERIFIED | Read (MARIMO_EXEC_SPECS-driven parametrize, class mark 7200) |
| `tests/examples/test_script_execution.py` | script lane with WR-04 4xx-re-raise routing | ✓ VERIFIED | Read; HTTPError clause precedes URLError, 4xx raises, 5xx/transport typed-skips; URLs live-probed HTTP 200 (human item 2) |
| `05-CENSUS.md` | full-tree census, all verdicts filled, hand-off section | ✓ VERIFIED | Every Table C gate re-run green (truth 22-23) |
| `.gitignore` | `.scratch/` entry | ✓ VERIFIED | Line 60; `git ls-files .scratch/` = 0 |
| Prior-closure artifacts (docs-validation.yml, README, feasibility.yml, spike_families.py, expected_skips.yaml, check_docs_sync.py, pyproject, 05-FEASIBILITY.md, spike logs) | unchanged-honest | ✓ VERIFIED | Quick regression greps all green (truths 8-19) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| dnallm/utils/__init__.py | transformers_compat.apply_patches() | eager import chain | ✓ WIRED | Live-probed: attachment active from `import dnallm` in a fresh process |
| _patch_remote_code_pruning_helpers | transformers.modeling_utils + pytorch_utils | per-name setattr | ✓ WIRED | Identity-verified against the vendored functions this round |
| tests/models/test_model_remote_code.py | load_model_and_tokenizer → ModelScope NT snapshot | real load attempt | ✓ WIRED | Exercised by this verifier's run (6.67s, weights 340/340 loaded, fails at the documented forward-stage rung on the patched cache) |
| census Table A rows | .scratch/census-out/ manifest + logs | evidence paths | ✓ WIRED | 26/26 logs present; manifest 28 outcomes; spot-read rows match log contents |
| TestGatedNotebookExecution | registered prefixes in expected_skips.yaml | typed-skip helpers | ✓ WIRED | 7 skips this round all matched the registered network-unavailable:/optional-dep: prefixes |
| 05-04 shim | benchmark/NER/script NT items | one root cause, one owner disposition | ✓ WIRED | Census NT-REMOTE-STRUCTURAL class covers all three; WINDOWS ledger entries 11-12 + 14 |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| NT smoke (D-07 live probe) | `pytest tests/models/test_model_remote_code.py -q` | **1 failed in 6.67s** — `AttributeError: 'EsmModel' object has no attribute 'get_extended_attention_mask'` (patched-cache forward-stage rung) | ✗ FAIL on this box → human item 1 (environment-state; designed loud-failure on non-marker breakage; typed skip on clean cache at commit time) |
| Gated notebook probes | `pytest ...::TestGatedNotebookExecution -q -rs` | 7 skipped in 1.07s, every message carrying live probe evidence | ✓ PASS |
| Pilot notebook (ACTIVE lane) | `pytest ...::test_notebook_executes_end_to_end[notebooks/inference/inference.ipynb]` | 1 passed in 11.48s | ✓ PASS |
| Kernel-kill + partial-failure (fresh, post-05-05 harness) | `pytest ...::TestKernelLifecycle ...::TestPartialFailureArtifacts -q` | 2 passed in 4.83s | ✓ PASS |
| Shim attach (both modules, identity, sentinel, idempotence) | fresh-process python probe | all True on transformers 5.17.0; native pytorch_utils.prune_linear_layer preserved | ✓ PASS |
| Compat contract tests | `pytest tests/utils/test_transformers_compat.py -q` | 32 passed in 3.47s | ✓ PASS |
| Fast leg (full suite, once) | `pytest -m "not slow" -q` | 1648 passed, 1 skipped (pre-existing allowlisted), 49 deselected, 88.75s | ✓ PASS |
| Census completeness gates | Table C command block | 0 pending; 21/3/1; 25/34 rows; 57 tracked; all paths exist | ✓ PASS |
| Environment isolation | find_spec probes + pyproject porcelain | zero spike-only packages; pyproject clean | ✓ PASS |
| Branch protection (live, read-only) | `gh api .../branches/{dev,main}/protection` | both branches: both required contexts | ✓ PASS |
| Rice input URLs (WR-04 latency) | `curl -I` both rice.uga.edu URLs | HTTP 200 / HTTP 200 | ✓ PASS (4xx path not firing; full lane = human item 2) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared by this phase; the plans' verify legs are pytest/script/awk commands, all re-executed directly above by this verifier (SUMMARY PASS claims were not used as evidence for any gate).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXEC-01 | 05-01, 05-04, 05-05, 05-06 | Private execution harness (nbclient-as-library, timeout layering, sandbox isolation, kernel shutdown, partial artifacts) | ✓ SATISFIED | Truths 1-7, 20-24; pilot/kill/partial/gated re-run green this round |
| EXEC-06 | 05-01 | Deliberate-hang kill test proves kernel cleanup | ✓ SATISFIED | Truth 7 — re-run green this round on the current harness |
| REPAIR-02 | 05-02 | Docs mirror closed + regenerated per repair | ✓ SATISFIED | Truths 8-10; sync DIFFERs == owner baseline only; gap closure repaired generate_bpe_dataset.py + resynced mirror |
| CI-01 | 05-02 | Masking removed in same unit as drift closure; docs-validation required check | ✓ SATISFIED | Truth 11 + truth 13 (live re-read this round) |
| CI-02 | 05-02 | mcp extra installed; README install line corrected | ✓ SATISFIED | Truth 12 |
| FEAS-01 | 05-03 | Written verdict matrix, real variants, evidence-backed typed skips | ✓ SATISFIED | Truths 14-19; census re-proved all three fallback legs |
| EXEC-03 (dev-box leg, claimed complete) | 05-05, 05-06 | All 3 marimo apps execute headlessly with defaults | ✓ SATISFIED (dev-box basis) | Census: 3/3 PASS (8.2s/4.9s/5.9s); MARIMO_EXEC_SPECS x3; nightly-runner leg arrives with Phase 8 SC1 — REQUIREMENTS.md check-off documents the dev-box basis via census §4 |
| EXEC-04 (dev-box leg, claimed open) | 05-05, 05-06 | generate_bpe_dataset.py produces artifact in-sandbox | ◐ PARTIAL (honest, deliberate) | REQUIREMENTS.md unchecked/Phase-8 Pending; script runs standalone (bed repair proven) but terminates at the NT structural rung — census records FAIL + self-healing typed skip; REQ stays open for Phase 8 exactly as claimed |
| REPAIR-03 (partial claim) | 05-04 | dnallm bugs exposed by execution fixed with regression tests | ◐ PARTIAL (honest, deliberate) | REQUIREMENTS.md unchecked/Phase-8 Pending; the shim + 32 contract tests + smoke = the NT remote-code slice; 12 census FAIL rows (incl. dnallm-side benchmark.py:296) stay in the Phase-8 repair queue exactly as claimed |

Orphaned requirements: none — REQUIREMENTS.md maps exactly EXEC-01/EXEC-06/REPAIR-02/CI-01/CI-02/FEAS-01 to Phase 5 (all Complete), matching the plans' `requirements` fields; gap plans' additional claims (EXEC-03/EXEC-04/REPAIR-03) are recorded honestly as complete-on-dev-box / open / partial and match the REQUIREMENTS.md states.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER matches across all 9 gap-closure code files | - | - |

Re-verification evidence gate (#3304): the census's evidence living in gitignored `.scratch/` is BY DESIGN (owner standing rule, .gitignore line 60) — the committed census rows carry the terminal one-liners and the scratch paths, and this verifier confirmed the referenced artifacts exist on disk. The two mcp deferred-owner verdicts use the sanctioned collision-free wording; the manifest's `pending-owner` outcome string appears nowhere in the census.

ℹ️ Info: EXEC-03 is checked complete in REQUIREMENTS.md on the dev-box basis while remaining mapped to Phase 8 — the nightly-runner leg is still part of Phase 8 SC1; the census §4 documents the basis. ℹ️ Info: 7-9 ipykernel processes observed on the box all belong to the owner's live Jupyter session (start times 10:28-14:25 and 20:01, none coinciding with verifier test executions); verifier runs left zero kernel debris and zero tree delta.

### Human Verification Required

### 1. NT snapshot disposition (smoke currently red on this box)

**Test:** Decide the disposition of the orchestrator-patched local ModelScope NT snapshot (backups at `*.dnallm-bak`), then re-run `.venv/bin/python -m pytest tests/models/test_model_remote_code.py -q`.
**Expected:** Smoke returns to green-or-typed-skipped. Currently 1 failed in 6.67s at the forward-stage `get_extended_attention_mask` rung — the patched config moves the load past the smoke's `config.is_decoder` structural marker, so the designed loud-failure path fires instead of the typed skip. On a clean cache (CI) the smoke skips typed as designed.
**Why human:** The root cause is a deliberately-kept out-of-band cache mutation documented in the census ("decision: keep patched — richer evidence"), not committed code. Options (restore backups / widen the marker to the forward-stage rung / land the Phase-8 NT disposition) are owner dispositions; a verifier must not mutate the owner's model cache.

### 2. WR-04 rice-URL network lane (flagged by the fix report)

**Test:** Run the slow script lane once against the real network (`.venv/bin/python -m pytest tests/examples/test_script_execution.py -q`) or accept the nightly's first run as the check.
**Expected:** 1 passed or 1 typed network-unavailable skip carrying the exact transport error — the 4xx re-raise branch must not fire (both rice URLs verified HTTP 200 this round).
**Why human:** The WR-04 except-chain routing was verified only via a replica; the real slow-lane download was never executed (fix report: "requires human verification").

### Gaps Summary

No failed truths, no missing/stub artifacts, no unwired links, no debt markers. GAP-1 and GAP-2 are closed exactly along the designed-honesty line the reopen addendum drew: the shim is landed, live-proven on transformers 5.17.0, and its smoke is a functioning live probe that terminated at the ladder's sanctioned rung at commit time; the census is complete, provably exhaustive, and every verdict traces to on-disk evidence this verifier re-checked. D-09 held (original plans untouched; honest REQ claims). The two routed items are an owner environment-state decision (the patched NT cache making the smoke loudly red on this box — the census predicted the exact rung) and a never-executed network lane — neither is a committed-code defect, and neither can be resolved by a verifier without mutating owner state. Status: **human_needed** (25/25; the ordered decision tree places rule 2 ahead of a clean pass).

---

_Verified: 2026-10-02T12:05:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification of: 05-VERIFICATION.md @ 821e184 parent state (gaps_found Post-closure Gap Addendum — the reopen origin; preserved in git history); gap-closure commits 8fa9e05..6453ece (plans 05-04..06 + review fixes 265dcec/d7493f5/eb55cff/6453ece)_
