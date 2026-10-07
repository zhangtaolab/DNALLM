---
phase: 05-execution-harness-honest-gates-runner-feasibility
plan: 02
subsystem: ci
tags: [docs-validation, branch-protection, continue-on-error, docs-mirror, uv-extras, github-actions]

requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: wave-1 harness landing point (05-01) — independent track, no code dependency
  - phase: 04-ci-gate-enforcement (v1)
    provides: branch-protection baseline "coverage-gate (py3.12, fast leg)" on dev+main and the v1 PUT precedent
provides:
  - Honest docs-validation gate — all five enforcement steps fail the job (zero masking flags), born green on the exact commands the workflow runs
  - Closed docs/example mirror drift — byte-identical resync (10 DIFFER files + generate_bpe_dataset.py mirrored, stale outputs included per D-03) and check_docs_sync.py exits 0
  - Wrapper-.md scoping in check_docs_sync.py — DOCS_ONLY_SUFFIXES consulted in right_only only; both-sides .md strictness injected-drift-proven
  - Repaired mcp_pydantic_ai.md block 5 (369-char string restored to single line, byte-identical to notebook cell 6)
  - mcp extra in the docs-validation install line + proven README Testing install line
  - Owner-run branch-protection PUT hand-off (dev+main) naming BOTH required contexts (D-02)
affects: [09-census-gates, 08-example-rollout, ship-gate]

actuals:
  tokens: 12400
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Scoped mirror relaxation: docs-only suffix allowlist consulted in exactly one diff branch (right_only); the strict branches (left_only/diff_files) prove strictness via an injected both-sides drift in verify"
    - "Born-green gate flip: every formerly-masked CI command is run green locally in the same unit as the flag removal, so the first honest run cannot fail on unrelated content"

key-files:
  created:
    - docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  modified:
    - scripts/check_docs_sync.py
    - docs/example/** (10 resynced mirror files)
    - docs/example/mcp_pydantic_ai.md
    - .github/workflows/docs-validation.yml
    - README.md

key-decisions:
  - "DOCS_ONLY_SUFFIXES relaxation scoped to the right_only loop only — a .md mirrored on BOTH sides (overview.md) must still match byte-for-byte; proven by injecting a both-sides drift and requiring the script to fail"
  - "docs-validation flipped honest only after all five workflow commands were verified green locally (94 passed + 1 allowlisted skip / 21 passed / 3 validators exit 0) — the gate is born green, not red (D-01)"
  - "README install line uv pip install -e '.[test,dev,mcp]' documented only after being run verbatim (exit 0); the pre-existing uv run resolver failure is logged out-of-scope in deferred-items.md, not silently fixed"

patterns-established:
  - "Injected-drift strictness proof: temporarily corrupt a both-sides file, require the sync script to fail, restore, require green — the regression test for any future ignore-vocabulary widening"

requirements-completed: [CI-01, CI-02, REPAIR-02]

coverage:
  - id: D1
    description: Docs mirror closure — wrapper-.md scoping in check_docs_sync.py, byte-identical resync of the 10 DIFFER files plus the missing generate_bpe_dataset.py mirror (stale outputs included per D-03)
    requirement: REPAIR-02
    verification:
      - kind: other
        ref: command ".venv/bin/python scripts/check_docs_sync.py -> exit 0, OK line"
        status: pass
      - kind: other
        ref: command "injected overview.md both-sides drift -> DIFFER line + exit 1, restored -> exit 0 (STRICTNESS LOST never printed)"
        status: pass
      - kind: other
        ref: command "cmp example/notebooks/finetune_NER_task/generate_bpe_dataset.py docs/example/.../generate_bpe_dataset.py -> identical"
        status: pass
    human_judgment: false
  - id: D2
    description: Honest docs-validation gate — five masking flags deleted, install line widened to .[test,dev,mcp], latent snippet failure repaired, README install line proven; all five workflow commands green locally
    requirement: CI-01
    verification:
      - kind: other
        ref: command "grep -c 'continue-on-error' .github/workflows/docs-validation.yml -> 0"
        status: pass
      - kind: other
        ref: command "validate_docs_snippets.py -> exit 0 (was 1 error at mcp_pydantic_ai.md:92 block 5)"
        status: pass
      - kind: integration
        ref: command "five-command chain green: check_docs_sync, validate_docs_snippets, validate_yaml, pytest tests/examples/test_examples.py (94 passed 1 allowlisted skip), pytest tests/configuration/test_yaml_load.py (21 passed)"
        status: pass
    human_judgment: false
  - id: D3
    description: README Testing section documents an install line proven verbatim (CI-02 README half)
    requirement: CI-02
    verification:
      - kind: other
        ref: command "uv pip install -e '.[test,dev,mcp]' -> exit 0; pytest works in the resulting env (21 passed via uv run --no-sync and .venv/bin/python -m pytest)"
        status: pass
    human_judgment: false
  - id: D4
    description: Branch-protection promotion hand-off (D-02) — owner-run PUT payloads for dev+main naming BOTH required contexts, recorded verbatim, with pre-flight read green
    verification:
      - kind: other
        ref: command "gh api repos/zhangtaolab/DNALLM/branches/{dev,main}/protection --jq '.required_status_checks.contexts[]' -> exit 0, prints coverage-gate (py3.12, fast leg)"
        status: pass
    human_judgment: true
    rationale: The PUT itself is an owner-admin action by design (v1 precedent, CONTEXT D-02); D-02 is not closed until the owner executes the recorded PUTs and both verification reads list both contexts. The blocking human check below carries the exact steps.

duration: 12 min
completed: 2026-10-02
status: complete
plan_head_before: a8f8460839759811f525113e5f10a4fc2fcfbeff
plan_head_after: 7dad6a5c55a3d887430a58745ca202936c7ba311
---

# Phase 5 Plan 02: Honest Gates & Docs-Mirror Repair Summary

**All five docs-validation masking flags removed over a byte-identical mirror resync (wrapper-.md scoping in check_docs_sync.py, D-03 outputs preserved), mcp extra installed, README install line proven, and the dev+main branch-protection PUT hand-off recorded naming both required contexts**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-01T17:55:10Z
- **Completed:** 2026-10-01T18:06:52Z
- **Tasks:** 3/3
- **Files modified:** 15 (1 created, 14 modified)

## Accomplishments
- `check_docs_sync.py` exits 0 with the OK line on the real tree for the first time since the wrapper-.md drift began: `DOCS_ONLY_SUFFIXES = (".md",)` accepts the 24 docs-only wrapper tutorials in `right_only` only, while an injected both-sides `overview.md` drift still fails the script (strictness proven, T-05-05 mitigated)
- Mirror closed byte-identically per D-03: 10 DIFFER files copied `example/` → `docs/example/` with stale outputs included, plus `notebooks/finetune_NER_task/generate_bpe_dataset.py` mirrored (cmp-identical); `git diff --stat` under docs/example/ shows exactly the 11 manifest files — nothing stripped, no wrapper regenerated
- docs-validation flipped honest (D-01): all five `continue-on-error` flags deleted, step bodies untouched, one rationale comment added ("Step failures fail the job: masked-outcome steps removed per WR-08" — worded without the flag's literal name so the zero-count check stays meaningful); job id/name `docs-validation` unchanged (it is the branch-protection context string)
- The one latent validator failure repaired in the same unit: `mcp_pydantic_ai.md` block 5's 369-char DNA string restored to a single line byte-identical to notebook cell 6 — `validate_docs_snippets.py` now exits 0 over 142 files / 328 blocks
- Install line widened to `uv pip install -e ".[test,dev,mcp]"` in the workflow (CI-02); README 🧪 Testing section documents `uv pip install -e '.[test,dev,mcp]'`, run verbatim by the executor (exit 0) before the claim; line-197 comment now mentions the mcp deps for example tests
- Born-green proof: the exact five workflow commands all green locally — sync OK, snippets OK, 21 YAML files OK, `tests/examples/test_examples.py` 94 passed + 1 allowlisted skip, `tests/configuration/test_yaml_load.py` 21 passed
- Branch-protection pre-flight read green against the live repo: both dev and main currently list exactly `coverage-gate (py3.12, fast leg)` — the baseline the owner PUT must widen, never narrow

## Task Commits

Each task was committed atomically:

1. **Task 1: Sync-script wrapper-.md fix + byte-identical mirror resync + missing script mirror** - `4decbe4` (fix)
2. **Task 2: Repair latent snippet failure, flip five masked steps honest (D-01), install mcp extra (CI-02), correct README** - `7dad6a5` (fix)
3. **Task 3: Branch-protection promotion hand-off (D-02)** - recorded in this SUMMARY (no repo mutation; the deliverable is the recorded hand-off + pre-flight read)

**Plan metadata:** (see final docs commit below)

## Files Created/Modified
- `scripts/check_docs_sync.py` - DOCS_ONLY_SUFFIXES + `_is_docs_only()` consulted in right_only loop only; IGNORE/left_only/diff_files untouched
- `docs/example/**` - 10 byte-identical resyncs (marimo demos, mcp notebooks, benchmark/finetune_data/NER/binary/multi_labels notebooks) + new `generate_bpe_dataset.py` mirror
- `docs/example/mcp_pydantic_ai.md` - block 5 single-line string repair
- `.github/workflows/docs-validation.yml` - five masking flags deleted, install `.[test,dev,mcp]`, honest-gate comment, job name unchanged
- `README.md` - proven install line in 🧪 Testing; line-197 comment mentions mcp deps

## Decisions Made
- Scoped the `.md` relaxation to `right_only` only (plan-locked); proved the boundary empirically rather than trusting the code shape
- Added the honest-gate rationale comment WITHOUT the masking flag's literal name — the verify counts that literal in the file, so a comment containing it would self-invalidate the zero-count check
- Logged the pre-existing `uv run` resolver failure as out-of-scope (deferred-items.md) instead of rewriting the five pre-existing README pytest command lines the plan did not ask to change

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- **Pre-existing (out of scope, logged):** the README's pre-existing `uv run pytest` command form fails under uv 0.12.20 universal resolution (`dnallm[mamba]` × cuda `conflicts` forks × unbounded `requires-python` → unsatisfiable py3.14-darwin fork). pyproject.toml is byte-unchanged by this plan; the documented install line itself is proven — `uv run --no-sync pytest` (21 passed) and `.venv/bin/python -m pytest` (21 passed) both work in the resulting environment. Recorded in `deferred-items.md`; fixing means restructuring the pyproject conflicts matrix (Rule 4), deliberately not attempted here.

## User Setup Required
None for services. **One owner action is pending — see the D-02 hand-off below (blocking human check).**

## D-02 Owner Hand-Off: promote docs-validation to a required check on dev+main

**Run AFTER this plan's workflow commit is pushed** (milestone is manual-push-only). The PUT **replaces** the entire `contexts` array — each payload names BOTH contexts so the promotion widens the gate and never un-requires coverage-gate.

**1. dev branch:**

```bash
gh api -X PUT repos/zhangtaolab/DNALLM/branches/dev/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["coverage-gate (py3.12, fast leg)", "docs-validation"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```

**2. main branch:**

```bash
gh api -X PUT repos/zhangtaolab/DNALLM/branches/main/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["coverage-gate (py3.12, fast leg)", "docs-validation"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```

**3. Verify both reads list BOTH strings** (D-02 is not closed until they do):

```bash
gh api repos/zhangtaolab/DNALLM/branches/dev/protection --jq '.required_status_checks.contexts[]'
gh api repos/zhangtaolab/DNALLM/branches/main/protection --jq '.required_status_checks.contexts[]'
```

Expected output of each: `coverage-gate (py3.12, fast leg)` and `docs-validation`.

**Pre-flight (already executed by the executor, read-only):** both reads exited 0 and printed `coverage-gate (py3.12, fast leg)` — auth is live and the baseline the PUT must preserve is confirmed.

**A5 hedge:** if the post-PUT verification shows GitHub reporting the docs-validation context under a different string, check a real check-run first (`gh api repos/zhangtaolab/DNALLM/commits/<sha>/check-runs`) before re-issuing the PUT with the corrected name.

### Blocking human check (gate="blocking-human")

Owner runs the hand-off above after pushing this plan's workflow commit: (1) execute both PUT commands verbatim; (2) run both verification reads and confirm each lists BOTH "coverage-gate (py3.12, fast leg)" and "docs-validation". D-02 is not closed until both reads show both contexts. No `gh api` PUT mutation was executed by the executor (owner-run by design).

## Known Stubs
None — no placeholder or unwired paths were introduced.

## Next Phase Readiness
- The docs lane is enforceable from this phase on: any later docs/example drift fails docs-validation, and once the owner PUT lands, the check is required on dev+main
- Phase 8 repairs ride an enforced lane: notebook output refreshes (D-03) will re-sync through the same byte-identity contract
- 05-03 (GB10 feasibility spike) proceeds independently; this plan's mirror closure does not touch the spike's surface
- Unpushed local range to report at phase close: `a8f8460..HEAD` (manual-push-only milestone)

## Self-Check: PASSED

- scripts/check_docs_sync.py, .github/workflows/docs-validation.yml, README.md modified on disk; docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py exists
- Commits 4decbe4, 7dad6a5 present on dev (measured 2 via `git rev-list --count a8f8460..HEAD`)
- Five-command honest chain green locally; masking-flag count 0; pre-flight gh reads green on dev+main

---
*Phase: 05-execution-harness-honest-gates-runner-feasibility*
*Completed: 2026-10-02*
