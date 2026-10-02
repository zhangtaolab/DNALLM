---
phase: 05-execution-harness-honest-gates-runner-feasibility
verified: 2026-10-01T22:12:12Z
status: gaps_found
reopened_at: 2026-10-02T10:22:00+08:00
reopen_command: "/gsd-plan-phase 5 --gaps --force — closed-phase gate #3569 overridden by owner decision"
score: 19/19 must-haves verified (prior closure; see Post-closure Gap Addendum)
covered_files:
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-01-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-01-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-02-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-02-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-03-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-03-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/expected_skips.yaml
  - pyproject.toml
  - scripts/check_docs_sync.py
  - .github/workflows/docs-validation.yml
  - README.md
  - scripts/feasibility/spike_families.py
  - .github/workflows/feasibility.yml
  - .github/workflows/README.md
covered_digest: "v2:sha256:06e0ee0ea0961e702961640d4ce49b865796283aced59e5baa73d22dfb992791"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "tests/examples/conftest.py provides the locally-scoped notebook_sandbox fixture (05-01 artifact)"
    reason: "Post-wave integration fix: tests/ is not a package, so a bare tests/examples/conftest.py won the conftest module-name race and broke 'from conftest import ...' in test_trainer/test_benchmark/test_dna_dataset. The fixture was relocated module-locally to tests/examples/test_notebook_execution.py:41-55 with an explanatory comment; in-phase consumption and tree-clean teardown are delivered identically (behaviorally exercised again this round by the verifier's own passing runs). Recreating the conftest would re-break three test files."
    accepted_by: "owner (Tao Zhang) — 05-UAT item 3, acknowledged 2026-10-02 at closure"
    accepted_at: "2026-10-02T04:50:00+08:00"
re_verification:
  previous_status: human_needed
  previous_score: 18/19
  previous_digest: "v2:sha256:317a3b48244661e864ef354df7cc818bbb383eea429377ec5cf9ed713cee302a"
  gaps_closed:
    - "Truth 13 (D-02 branch protection): owner executed both PUTs; this verifier live-read both endpoints — dev and main each list BOTH 'coverage-gate (py3.12, fast leg)' and 'docs-validation' (widened, not narrowed). CLOSED."
    - "D-04 runner-confirmation gate: reclassified from pending-owner to a documented post-merge sequencing state after the platform constraint was proven live by this verifier (dispatch API HTTP 404 'not found on the default branch'; workflow registry lists no feasibility.yml; file ABSENT on origin/main and origin/dev, PRESENT on origin/phs). Recorded as a deferred follow-up, not a gap — the phase's deliverable (dispatch-only workflow + documented sequencing + local evidence matrix) is complete."
    - "Conftest-relocation acknowledgment (prior informational human item): owner-acknowledged 2026-10-02 per 05-UAT item 3; formalized as the override above."
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "05-FEASIBILITY.md Runner-confirmation column is filled from an actual dispatch run of feasibility.yml on the self-hosted GB10 runner (verdicts become official per D-04)"
    addressed_in: "post-merge integration window (phs → dev → main) — first dispatch opportunity after feasibility.yml lands on the default branch"
    evidence: "05-FEASIBILITY.md header: 'Runner confirmation is SEQUENCED POST-MERGE: GitHub workflow_dispatch requires the workflow file on the default branch (main); feasibility.yml currently exists only on phs'. Independently reproduced by this verifier: gh run list --workflow=feasibility.yml → HTTP 404; actions/workflows registry (default branch) lists only ci.yml, docs-validation.yml, publish.yml; git cat-file confirms feasibility.yml ABSENT on origin/main and origin/dev, PRESENT on origin/phs (tip 1168e0e). The plan's own design anticipated this ('The local matrix stays marked provisional until then'); plan truth 18 requires only the committed workflow + README documentation + recorded hand-off — all verified."
---

# Phase 5: Execution Harness, Honest Gates & Runner Feasibility Verification Report

**Phase Goal:** A trustworthy private execution harness exists and is proven (including kernel-kill on hang); both false-green CI gates are closed together with the docs-mirror drift they were hiding; and the runner's real capabilities for the environment-gated model families are settled in writing before execution tests are written against them
**Verified:** 2026-10-01T22:12:12Z
**Status:** passed
**Re-verification:** Yes — final closure round after owner validation (prior: 18/19 human_needed, digest 317a3b48)

## Re-Verification Scope

Since the prior verification (2026-10-01T21:56:28Z, digest 317a3b48) exactly two commits landed
(`1168e0e`, `59d7f30`), touching only planning artifacts: `05-FEASIBILITY.md` (4 lines — the
post-merge sequencing annotation in the Status header and the matrix column header),
`05-UAT.md`, `05-VERIFICATION.md` closure addendum, and `.planning/STATE.md`.
**Zero covered source files changed** (`git diff --name-only 894f17a..HEAD` lists only those four
planning paths), so every previously-verified truth/artifact/link carried by quick regression
(re-run this round and green). The three open human items received full fresh verification,
including live read-only GitHub API checks executed by this verifier — none of the closure
claims were taken from the UAT/SUMMARY on faith.

## Goal Achievement

### Observable Truths

Truths 1-12, 14-19 carried VERIFIED from the prior round via quick regression (all covering files byte-identical since digest 317a3b48; key greps re-run green this round). Truths 13 and 18 re-verified live below.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pilot notebook executes end-to-end through run_notebook() with kernel cwd inside a tmp_path sandbox copy (SC1/EXEC-01) | ✓ VERIFIED | Carried (test code unchanged since verifier-run pass; mechanism `resources={"metadata": {"path": str(sandbox)}}` re-grepped at `_execution.py:152`); 3 tests collected, all slow-marked, not-slow selection empty (re-checked this round) |
| 2 | Scoped `git status --porcelain -- example docs/example` empty after execution (twice-run tree-clean proof) (SC1/EXEC-01) | ✓ VERIFIED | `assert_tree_clean()` intact; re-proven this round — verifier's own test runs left the scoped tree empty (checked after running truths 3 and 7's tests) |
| 3 | On cell error/timeout the harness writes partial executed notebook + exception text under tmp_path before re-raising (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round**: `TestPartialFailureArtifacts::test_cell_error_captures_artifacts_and_reraises` — 1 passed in 1.45s; mechanism at `_execution.py:159-164` (nbformat.write + .error.txt + bare raise) |
| 4 | Timeout layering: per-cell 600 < timeout(1800) pilot; per-cell 3 < timeout(120) kill test (SC1/EXEC-01) | ✓ VERIFIED | Marks re-grepped at `test_notebook_execution.py:79,130` (plus 162 for the partial-failure test); NOTEBOOK_EXEC_SPECS cell_timeout 600; kill test client timeout=3 |
| 5 | Kernel shutdown guaranteed via `shutdown_kernel="immediate"` + plain `client.execute()` (nbclient 0.11.0 not a context manager) (SC1/EXEC-01) | ✓ VERIFIED | Re-grepped: `shutdown_kernel="immediate"` at line 151, `client.execute()` count = 1, `with NotebookClient` count = 0; supersession documented in module docstring |
| 6 | Harness private to its test tree: underscore module, no root-conftest/dnallm importers (EXEC-01) | ✓ VERIFIED | Carried; sole consumer is test_notebook_execution.py (import at line 24) |
| 7 | Kill test: 2-cell notebook sleeping 300s under per-cell timeout=3 raises CellTimeoutError, ipykernel_launcher count returns to pre-test baseline (delta-zero) (SC2/EXEC-06) | ✓ VERIFIED | **Re-run this round (goal-level proof, "proven including kernel-kill on hang")**: `TestKernelLifecycle::test_hung_kernel_is_killed_and_cleaned_up` — 1 passed in 4.41s; post-run process check found zero surviving kernels (the single pgrep hit was the verifier's own grep pipeline self-matching — exactly Pitfall 1; ps confirmed no real ipykernel_launcher process) |
| 8 | check_docs_sync.py exits 0 printing the OK line; wrapper-.md relaxation scoped to right_only only; both-sides .md strictness preserved (SC3/REPAIR-02) | ✓ VERIFIED | **Re-run this round**: exit 0, `OK: docs/example/ is in sync with example/`; direct read confirms `_is_docs_only` consulted at exactly one site, inside the `dircmp.right_only` loop; left_only/diff_files untouched; global IGNORE has no .md |
| 9 | Mirror byte-identical: 10 DIFFER resyncs + generate_bpe_dataset.py mirrored, stale outputs included (SC3/REPAIR-02/D-03) | ✓ VERIFIED | Sync green over the whole tree this round (byte-level cmpfiles shallow=False); mirrored script present |
| 10 | REPAIR-02 edges: absent input dirs → ERROR + exit 1 (fail-closed); each drifted file reported exactly once under one prefix | ✓ VERIFIED | Carried (behavioral prior round; script unchanged since — re-read this round) |
| 11 | docs-validation honest: zero continue-on-error; five enforcement steps run exact commands, born green (SC3/CI-01/D-01) | ✓ VERIFIED | Re-grepped this round: `continue-on-error` count = 0 in HEAD's workflow; install `.[test,dev,mcp]` line 42; job `docs-validation` line 13. (See Info note: remote main/dev still run the pre-phase masked copy until the unpushed range merges) |
| 12 | docs-validation installs .[test,dev,mcp]; README documents a proven install line (SC3/CI-02) | ✓ VERIFIED | Workflow line 42 + README.md:497 re-grepped this round |
| 13 | Branch protection on dev and main lists required contexts containing BOTH "coverage-gate (py3.12, fast leg)" and "docs-validation" (CI-01/D-02) | ✓ VERIFIED (closed this round) | **Live read-only verification by this verifier (2026-10-01T22:1xZ)**: `gh api repos/zhangtaolab/DNALLM/branches/dev/protection --jq '.required_status_checks.contexts[]'` → `coverage-gate (py3.12, fast leg)` + `docs-validation`; identical on main. The array was widened, not narrowed (prohibition upheld). Owner executed both PUTs per 05-UAT item 1 |
| 14 | Written verdict matrix with rows for evo-1, evo2, megaDNA, pyBigWig, marimo — every row carries measured evidence or exact failure text (SC4/FEAS-01/D-05) | ✓ VERIFIED | Re-grepped this round: 10 matrix rows; all 8 spike logs on disk; post-merge edit touched only the Status header + column header (git diff 894f17a..HEAD inspected — no verdict content changed) |
| 15 | Verdicts against the EXACT notebook variants via real forward pass through the dnallm route; pyBigWig row is import + real BigWig write/read round-trip (D-05) | ✓ VERIFIED | Carried (matrix names and log contents matched prior round; spike_families.py 874 lines, `--help` exit 0 re-run this round) |
| 16 | Every non-FEASIBLE verdict shows BOTH attempts with recorded failure text; environment-unavailable only with evidence (D-06) | ✓ VERIFIED | Carried (4-attempt evo-1 ladder, evo2 FP8 ImportError verbatim, megaDNA pinned clone cb2f5ab4, pyBigWig evidence-ref — all in committed logs, re-listed on disk this round) |
| 17 | Spike ran locally in a throwaway /tmp venv; project .venv and pyproject carry no spike-only packages (D-04) | ✓ VERIFIED | Re-grepped: pyproject pyBigWig 0 matches; nbclient>=0.10 only at lines 99 + 104 |
| 18 | Dispatch-gated runner confirmation job exists (feasibility.yml: [self-hosted, dnallm-nightly], workflow_dispatch only, if: always() upload, timeout 240), documented in workflows README, hand-off recorded (D-04) | ✓ VERIFIED | Re-grepped this round: `on: workflow_dispatch` line 13 (zero push/PR/schedule), runs-on line 26, event gate line 27, timeout 240 line 31, `if: always()` line 102, permissions block lines 18-19; README section 7 + trigger-overview line 16; hand-off verbatim in 05-03-SUMMARY. The official runner run itself is post-merge by GitHub platform constraint — live-proven this round (dispatch API 404; registry lacks feasibility.yml; file ABSENT origin/main + origin/dev, PRESENT origin/phs) — recorded as deferred follow-up, not a gap |
| 19 | marimo flavor decided with evidence; pyBigWig enters dev extra ONLY on FEASIBLE verdict + owner provenance (else pyproject untouched) (FEAS-01) | ✓ VERIFIED | Carried: spike_marimo.log flavor A/B evidence, export-html decision in matrix; pyBigWig verdict environment-unavailable → pyproject untouched (0 matches re-confirmed) |

**Score:** 19/19 truths verified (0 present-behavior-unverified; 0 pending)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Matrix Runner-confirmation column filled from an actual dispatch run of feasibility.yml (D-04 official-verdict step) | Post-merge integration (phs → dev → main) — first dispatch window after feasibility.yml reaches the default branch | Platform constraint independently reproduced by this verifier: HTTP 404 "workflow feasibility.yml not found on the default branch"; registry lists no feasibility.yml; file ABSENT on origin/main and origin/dev, PRESENT on origin/phs. Documented in 05-FEASIBILITY.md header ("a platform constraint, not an unfinished action") and 05-UAT item 2. Same merge event also flips the remote's running docs-validation copy honest (origin/main and origin/dev still carry the 5-flag masked version until the unpushed range lands) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/examples/_execution.py` | specs, seed_sandbox, run_notebook, assert_tree_clean, typed-skip helpers | ✓ VERIFIED | All 8 declared exports re-grepped this round; execute() x1, with NotebookClient x0; artifact mechanism at 159-164 |
| `tests/examples/conftest.py` | locally-scoped notebook_sandbox fixture | ✓ PASSED (override) | File deliberately absent (post-wave bare-conftest collision fix); fixture module-local at `test_notebook_execution.py:41-55`, consumed in-phase, teardown exercised — owner-accepted 2026-10-02 (05-UAT item 3); see `overrides:` frontmatter |
| `tests/examples/test_notebook_execution.py` | TestNotebookExecution + TestKernelLifecycle + TestPartialFailureArtifacts | ✓ VERIFIED | All three classes present (lines 80/126/158); kill test and partial-failure test both PASSED in fresh verifier runs this round (4.41s / 1.45s) |
| `tests/expected_skips.yaml` | prefix entries environment-unavailable: / optional-dep: | ✓ VERIFIED | Lines 35 + 40 re-grepped |
| `pyproject.toml` | nbclient>=0.10 in notebook (+test) extra; no pyBigWig; no spike deps | ✓ VERIFIED | Lines 99 + 104; pyBigWig 0 matches |
| `scripts/check_docs_sync.py` | DOCS_ONLY_SUFFIXES scoped to right_only | ✓ VERIFIED | Definition line 21, sole consult inside right_only loop (direct read this round); run green this round |
| `docs/example/` | byte-identical mirror | ✓ VERIFIED | Sync exit 0 this round (behavioral) |
| `.github/workflows/docs-validation.yml` | five honest steps, mcp extra, job name unchanged | ✓ VERIFIED | 0 continue-on-error; line 42 install; line 13 job id/name |
| `README.md` | proven install line in Testing section | ✓ VERIFIED | Line 497 |
| `scripts/feasibility/spike_families.py` | per-family spike runner, D-05 contract, D-06 fallback | ✓ VERIFIED | 874 lines; `--help` exit 0 this round |
| `.github/workflows/feasibility.yml` | dispatch-only feas-spike job | ✓ VERIFIED | All gates re-grepped this round (dispatch-only, runner label, event gate, timeout 240, if: always() upload, least-privilege permissions) |
| `05-FEASIBILITY.md` | verdict matrix + evidence + prefix assignment + runner column | ✓ VERIFIED | 10 rows; column header now "Runner confirmation (post-merge)"; post-merge edit verified additive-only by diff inspection |
| `.github/workflows/README.md` | feasibility documented by name and trigger | ✓ VERIFIED | Section 7 + trigger-overview line 16 re-grepped; last content change (bb57709, WR-09 residuals) predates the prior digest |
| `spike-logs/spike_*.log` (8 files) | raw per-family evidence | ✓ VERIFIED | All 8 re-listed on disk this round |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| tests/examples/test_notebook_execution.py | tests/examples/_execution.py | `from tests.examples._execution import ...` | ✓ WIRED | Import line 24; run_notebook/seed_sandbox/NOTEBOOK_EXEC_SPECS/assert_tree_clean used (re-exercised by this round's passing runs) |
| test_notebook_execution.py | notebook_sandbox fixture (relocated) | pilot test requests fixture, yields sandbox Path | ✓ WIRED | Fixture at lines 41-55; consumed in-phase; override records the file-level deviation |
| tests/examples/_execution.py | nbclient | resources metadata path + execute() | ✓ WIRED | Lines 152 + 157 re-grepped |
| tests/examples/_execution.py | tests/expected_skips.yaml | typed-skip prefixes matched by audit | ✓ WIRED | Prefixes at yaml lines 35/40; helpers emit literal prefixes |
| .github/workflows/docs-validation.yml | scripts/check_docs_sync.py | sync step gates the job | ✓ WIRED | 0 continue-on-error between |
| scripts/check_docs_sync.py | docs/example | right_only consults DOCS_ONLY_SUFFIXES | ✓ WIRED | Single consult site confirmed by direct read |
| .github/workflows/feasibility.yml | scripts/feasibility/spike_families.py | dispatch job runs the committed runner | ✓ WIRED | Unchanged since prior verification |
| 05-FEASIBILITY.md | tests/expected_skips.yaml | prefix assignment names registered prefixes | ✓ WIRED | Unchanged since prior verification |
| docs-validation (job) | branch protection on dev+main | required contexts entry | ✓ WIRED (closed this round) | Live read: both branches list both contexts; A5 hedge moot — the context string equals the job name |

### Data-Flow Trace (Level 4)

Not applicable in the render sense (no UI). Equivalent data-flow checks hold: verdict-matrix values trace to the 8 committed spike logs; the sync verdict traces to byte-level file comparison (re-run green this round); truth-3 artifact capture traces to a real pytest run re-executed by this verifier; truth-13 traces to live GitHub API reads (not the SUMMARY's record of them).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Kernel-kill on hang (phase-goal clause, EXEC-06) | `.venv/bin/python -m pytest ...::TestKernelLifecycle::test_hung_kernel_is_killed_and_cleaned_up` | 1 passed in 4.41s | ✓ PASS |
| Partial-failure artifact capture (truth 3) | `.venv/bin/python -m pytest ...::TestPartialFailureArtifacts::test_cell_error_captures_artifacts_and_reraises` | 1 passed in 1.45s | ✓ PASS |
| Docs sync green + OK line (REPAIR-02) | `.venv/bin/python scripts/check_docs_sync.py` | exit 0, `OK: docs/example/ is in sync with example/` | ✓ PASS |
| D-02 protection state (live, read-only) | `gh api .../branches/{dev,main}/protection --jq '.required_status_checks.contexts[]'` | both branches: `coverage-gate (py3.12, fast leg)` + `docs-validation` | ✓ PASS |
| D-04 platform constraint (live, read-only) | `gh run list --workflow=feasibility.yml` + workflow registry + per-ref cat-file | HTTP 404 "not found on the default branch"; registry lacks feasibility.yml; ABSENT origin/main+origin/dev, PRESENT origin/phs | ✓ PASS (constraint confirmed; dispatch impossible pre-merge) |
| Spike CLI contract | `.venv/bin/python scripts/feasibility/spike_families.py --help` | exit 0 | ✓ PASS |
| Slow-mark discipline | pytest --collect-only / -m 'not slow' | 3 collected; not-slow node count 0 | ✓ PASS |
| Tree clean after verifier runs | `git status --porcelain -- example docs/example` | empty | ✓ PASS |
| Kernel-leak residue after runs | pgrep/ps for ipykernel_launcher | zero real kernels (single hit = verifier's own pipeline self-match, per documented Pitfall 1) | ✓ PASS |
| Pilot slow run, injected-drift strictness, fail-closed absent dirs, venv isolation, 229-test collection | — | not re-run: covering files byte-identical to digest 317a3b48; verifier-run passes recorded in 05-VERIFICATION.md @ 894f17a | ✓ CARRIED |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared by this phase's PLAN/SUMMARY; `find scripts -path '*/tests/probe-*.sh'` returns 0. The plans' verify legs are pytest/script commands, re-executed directly above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXEC-01 | 05-01 | Private execution harness, nbclient-as-library, timeout layering, sandbox cwd isolation, kernel shutdown, partial artifacts | ✓ SATISFIED | Truths 1-6; partial-failure test re-run green this round |
| EXEC-06 | 05-01 | Deliberate-hang kill test proves kernel cleanup | ✓ SATISFIED | Truth 7 — **re-run green by this verifier this round** |
| REPAIR-02 | 05-02 | Docs mirror closed: wrapper-.md fix, byte-identical resync, missing script mirrored | ✓ SATISFIED | Truths 8-10; sync re-run green this round |
| CI-01 | 05-02 | Masking removed in same unit as drift closure; docs-validation promoted to required check | ✓ SATISFIED | Truth 11 (0 masking flags) + truth 13 (branch protection live-verified listing both contexts — D-02 now closed) |
| CI-02 | 05-02 | mcp extra installed; README install line corrected | ✓ SATISFIED | Truth 12 (re-grepped) |
| FEAS-01 | 05-03 | Written verdict matrix, real variants enabled where feasible, evidence-backed typed skips | ✓ SATISFIED | Truths 14-19; runner confirmation sequenced post-merge by platform constraint (deferred item 1) — matrix itself complete with measured evidence |

Orphaned requirements: none — REQUIREMENTS.md maps exactly EXEC-01, EXEC-06, REPAIR-02, CI-01, CI-02, FEAS-01 to Phase 5 (all marked Complete), matching the plans' `requirements` fields exactly.

### Decision Coverage

D-01 through D-06 all trace to shipped artifacts (prior round ran `check.decision-coverage-verify`: 6/6). D-02's execution half is now closed (live-verified this round). D-04's official-verdict half is the recorded post-merge follow-up.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | Zero TBD/FIXME/XXX/TODO/HACK/placeholder matches across all 8 covered code files (re-grepped this round) | - | - |

Re-verification evidence gate (#3304): the only file changed since the prior `verified:` timestamp among covered sources is `05-FEASIBILITY.md` (commit 1168e0e) — diff inspected: 4 lines, the post-merge sequencing annotation; no verdict content, no debt markers, no stubs. All other covered files byte-identical. No new-scope findings; `advisory:` is empty.

ℹ️ Info (transient remote state, resolved by the same merge): origin/main and origin/dev still carry the pre-phase docs-validation.yml (5 continue-on-error flags) because the milestone is manual-push-only and local dev is 46 commits unpushed; the honest version (0 flags) is on HEAD/phs and flips the remote at merge. Likewise the D-02 PUTs were executed ahead of the dev push (the hand-off sequenced them after it); the protection state itself — the must-have — is exactly as required, and both required checks will produce runs on the first post-merge push. Pre-existing `uv run pytest` README resolver issue remains recorded out-of-scope in deferred-items.md (unchanged).

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/examples/test_notebook_execution.py | EXEC-01, EXEC-06 | 3 | 0 | 0 | Behavioral (no error outputs; delta-zero kernel count vs captured baseline; tree-clean; artifact files + content assertions) | OK |
| tests/examples/test_examples.py (regression) | CI-01 born-green | 95 | 1 (allowlisted, pre-existing) | 0 | Value/import | OK |

No disabled tests on requirements; no circular evidence.

### Human Verification Required

None. All three prior human items are closed with independently verified evidence:
1. D-02 branch-protection PUTs — executed by the owner; **live-verified by this verifier** (both branches list both contexts).
2. D-04 runner confirmation — proven impossible pre-merge by GitHub platform constraint (dispatch requires the workflow on the default branch; live-reproduced HTTP 404); reclassified as the documented post-merge follow-up recorded in `deferred:`, not a pending human gate of this phase. The phase's own deliverables for D-04 (dispatch-only workflow, README documentation, hand-off, local evidence matrix, provisional-until-confirmed design) are all present and verified.
3. Conftest-relocation acknowledgment — owner-acknowledged (05-UAT item 3); formalized as the `overrides:` entry.

### Gaps Summary

No gaps. All 19 truths verified (truth 13 closed by live API read this round; truth 7's goal-level kernel-kill proof re-run green), all artifacts present and substantive (one owner-accepted file-level deviation recorded as an override), all links wired, zero debt markers in covered code, and both requirement coverage and decision coverage are complete with no orphans. The single follow-up — dispatching feasibility.yml and filling the matrix Runner-confirmation column — is blocked by a GitHub platform constraint (workflow_dispatch registers only from the default branch; independently reproduced: HTTP 404, file absent on origin/main), was anticipated by the phase's own design (verdicts explicitly provisional until confirmed), and is recorded as a deferred post-merge item rather than an unmet must-have. Status: **passed** (19/19; human verification section empty).

---

_Verified: 2026-10-01T22:12:12Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification of: 05-VERIFICATION.md @ 894f17a (18/19, human_needed, digest 317a3b48); closure commits 1168e0e + 59d7f30_

## Post-closure Gap Addendum (2026-10-02)

**Status flipped: `passed` → `gaps_found`.** Phase reopened by owner decision under
`/gsd-plan-phase 5 --gaps --force`. The prior 19/19 closure remains historically accurate for what
it examined; the gaps below were invisible to it by scope, not by misverification.

### GAP-1 — registry checkpoint broken on the pinned dev environment (transformers 5.17)

- **Repro:** `example/notebooks/benchmark/benchmark.ipynb` cell 4 (`benchmark.run()`) →
  `ValueError: Failed to load model: cannot import name 'find_pruneable_heads_and_indices' from
  'transformers.modeling_utils'` (raised through `dnallm/models/model.py:888`).
- **Root cause (verified 2026-10-02):** `zhangtaolab/nucleotide-transformer-v2-100m-promoter`
  (`dnallm/models/model_info.yaml:214`; third model of `example/notebooks/benchmark/benchmark_config.yaml`,
  ModelScope source) ships a transformers-4.x-era remote-code `modeling_esm.py`. transformers 5.17.0
  removed `find_pruneable_heads_and_indices` and `prune_linear_layer` from
  `transformers.modeling_utils` (hasattr-verified False) AND from `transformers.pytorch_utils`
  (import fails) — the remote module's `from transformers.modeling_utils import (...)` therefore
  crashes inside `get_class_in_module` under `trust_remote_code=True`. `dnallm/` itself references
  neither symbol (grep-verified zero hits) — this is a third-party-remote-code × transformers-4→5
  span gap, not a dnallm import.
- **Environment delta:** transformers 5.17.0 installed 2026-09-17 (dist-info mtime); the last green
  benchmark run whose outputs are committed dates 2026-05-17 (commit 936ef04) — it predates the
  upgrade and proves nothing about 5.17.
- **Why the 19/19 closure did not catch it:** zero test references to this checkpoint (grep across
  `tests/` + `dnallm/mcp/tests/`); real-model tests cover `plant-dnagpt-BPE-promoter` only; the
  Phase 5 execution-harness pilot was `example/notebooks/inference/inference.ipynb` alone
  (`tests/examples/test_notebook_execution.py:34`).
- **Fix direction (planner input, not a locked design):** compat shim in
  `dnallm/utils/transformers_compat.py` — vendor the two pruning helpers, attach them to
  `transformers.modeling_utils` when absent, no-op on transformers 4.x — plus a real-model
  load+forward smoke test covering this checkpoint. **Residual risk to plan for:** the import shim
  may expose deeper 5.x breakage inside `modeling_esm.py`; the smoke test must perform load +
  forward (import-only proof is insufficient).

### GAP-2 — example/ execution coverage closed on a pilot only (owner-directed scope expansion)

- **Owner ruling 2026-10-02:** Phase 5 gap closure must execute the ENTIRE `example/` tree — all
  notebooks, marimo apps, scripts and programs, 一项不漏. Acceptance standard (owner-selected):
  every census item carries either a real-execution result or an evidence-backed typed
  `environment-unavailable:` skip (D-05/D-06 variant rules apply). Deliverable includes a committed
  full census inventory with per-item verdict. Nothing silently omitted.
- **Overlap with Phase 7/8 charters is acknowledged and deferred:** roadmap rescoping is an owner
  action AFTER this closure, flagged at plan hand-off — the gap-closure plans must not silently
  rewrite Phases 7–9 scope.
- **Standing constraints carried into gap-closure plans:**
  - Never recreate `tests/examples/conftest.py` (frontmatter `overrides:` entry 1 — module-name race
    broke 3 test files; the fixture lives module-locally in `tests/examples/test_notebook_execution.py`).
  - One-off scripts (census generators, shim probes, anything throwaway) stay in gitignored
    `.scratch/` — never committed/pushed (owner rule, also recorded in Phase 6 context).
  - dev+main are read-only for this milestone; all work lands on `phs`; pushes are manual-only.
