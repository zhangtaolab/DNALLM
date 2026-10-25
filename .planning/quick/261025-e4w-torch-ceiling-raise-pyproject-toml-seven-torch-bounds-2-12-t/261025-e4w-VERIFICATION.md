---
phase: quick-261025-e4w
verified: 2026-10-25T11:45:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - ".github/dependabot.yml"
  - ".planning/quick/261025-e4w-torch-ceiling-raise-pyproject-toml-seven-torch-bounds-2-12-t/261025-e4w-PLAN.md"
  - ".planning/quick/261025-e4w-torch-ceiling-raise-pyproject-toml-seven-torch-bounds-2-12-t/261025-e4w-SUMMARY.md"
  - "pyproject.toml"
covered_digest: "v3:sha256:605139db7e2dea11517b309d4b262228af850ca90d5ed55d19b044cb0e234980"
behavior_unverified: 0
overrides_applied: 0
---

# Quick 261025-e4w: torch ceiling raise <2.12 -> <2.15 Verification Report

**Item Goal:** torch ceiling raise — pyproject.toml seven torch bounds <2.12 to <2.15 as one commit citing GitHub run 40140203118; depends on item 261025-e4v landing first
**Verified:** 2026-10-25T11:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                                                                                                    | Status     | Evidence                                                                                                                                                                                                                                                             |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | pyproject.toml declares exactly seven torch specs all ending `,<2.15` (base + cpu/cuda124/cuda126/cuda128/cuda130/mamba) with floors unchanged (>=2.4.0 x3, >=2.6.0 x3, >=2.9.0 x1)      | ✓ VERIFIED | tomllib semantic extraction: `torch>=2.4.0,<2.15` (base, cpu, cuda124), `torch>=2.6.0,<2.15` (cuda126, cuda128, mamba), `torch>=2.9.0,<2.15` (cuda130); COUNT=7, ALL_END_2.15=True; grep floors 3/3/1; commit diff shows exactly 7 changed lines, floors untouched |
| 2   | The token `,<2.12` appears zero times in pyproject.toml                                                                                                                                 | ✓ VERIFIED | `grep -c ',<2.12' pyproject.toml` = 0; semantic tomllib dump confirms no spec anywhere references 2.12 (see also Truth 5 check)                                                                                                                                       |
| 3   | Dependabot torch entry still ignores semver-minor + semver-major; comment cites run 40140203118 plus the mamba-ssm sdist c++20 patch pipeline as the completed coordinated cycle          | ✓ VERIFIED | `.github/dependabot.yml:48-49` — `- dependency-name: "torch"` + `update-types: ["version-update:semver-minor", "version-update:semver-major"]` intact; comment lines 30-47 record the completed cycle citing run 40140203118, the sha256-pinned sdist c++20 pipeline in scripts/install_mamba.sh, ceiling <2.15, correct six-extra mirror set (rocm stale parenthetical corrected), repeat-cycle rule for >=2.15; `yaml.safe_load` parses |
| 4   | The whole item is ONE commit on branch v1.4 touching exactly pyproject.toml and .github/dependabot.yml, message citing run 40140203118                                                    | ✓ VERIFIED | HEAD f00e90d on branch v1.4; `git show --name-only` = exactly `.github/dependabot.yml` + `pyproject.toml` (count 2); message cites 40140203118 (subject + body); zero co-authored trailers; since e4v (ef91f8d) landed, f00e90d is the ONLY commit touching either file; both files clean in porcelain |
| 5   | No torch bound anywhere in pyproject.toml is wider than the verified 2.12-2.14 line (ceiling exactly <2.15; nothing unbounded, nothing >=2.15-admitting)                                   | ✓ VERIFIED | tomllib dump of ALL torch specs across base [project] dependencies AND every extra key (all/cpu/.../ui): exactly the 7 specs, every one upper-bounded `<2.15` — no unbounded torch, no other extras carry torch deps |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `pyproject.toml` | seven `<2.15` torch bounds, TOML-valid, floors untouched | ✓ VERIFIED | tomllib parses; 7 specs semantic-verified; floors 3/3/1; commit diff confined to the 7 bound lines (14 +/- lines total) |
| `.github/dependabot.yml` | rewritten torch-ignore comment recording the completed cycle, ignore rule intact, YAML-valid | ✓ VERIFIED | YAML parses; diff vs parent ef91f8d is comment-only (zero non-comment rule lines changed — ignore rule byte-identical); 40140203118 present (1x), install_mamba present (2x), zero `<2.12` literal |
| Single commit on v1.4 citing run 40140203118 | f00e90d | ✓ VERIFIED | On branch v1.4 (HEAD); exactly two files; message cites run twice; no attribution trailers; HEAD:pyproject.toml committed blob carries the seven `<2.15` bounds (verified via `git show HEAD:pyproject.toml` + tomllib) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| mamba-extra torch bound `<2.15` | e4v patched mamba-ssm sdist build path | scripts/install_mamba.sh + CI install lines | ✓ WIRED | e4v commit ef91f8d is an ancestor of f00e90d; scripts/install_mamba.sh carries 6x `std=c++20` + sha256 pin `104cc47e` (sdist pipeline, guessed-URL wheel path documented dead at script lines 13-17); .github/workflows/ci.yml invokes `bash scripts/install_mamba.sh` at lines 361 (test-mamba lane) and 857 (nightly wheelhouse, citing run 40140203118 blocker evidence) |
| dependabot ignore comment | pyproject ceiling | documented ceiling matches enforced bound | ✓ WIRED | Comment states "the ceiling (<2.15, mirrored across the cpu/cuda124/cuda126/cuda128/cuda130/mamba extras)" — matches the enforced `<2.15` in all seven pyproject specs |
| commit message | evidence chain | run 40140203118 recorded in research docs | ✓ WIRED | Commit body cites run 40140203118 criterion-a (2.14.1+cu130, census 3215 passed); `.planning/research/torch-214-gb10-verification.md` (2 mentions) and `.planning/async-jobs/spike-torch-214.json` (3 mentions) both exist and cite the run |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| TOML semantic gate — exactly 7 torch specs ending `,<2.15` from base + six extras | `.venv/bin/python -c "import tomllib; ..."` | COUNT=7, ALL_END_2.15=True | ✓ PASS |
| Committed HEAD blob (not working tree) carries the seven `<2.15` bounds | `git show HEAD:pyproject.toml` + tomllib | "HEAD committed blob carries 7 <2.15 bounds: OK" | ✓ PASS |
| Dependabot YAML parses | `yaml.safe_load` | yaml OK | ✓ PASS |
| Extras-guard regression suite (plan's named test) | `.venv/bin/python -m pytest tests/test_extras_guard.py -q` | 14 passed in 2.45s | ✓ PASS |

### Requirements Coverage

Quick-batch item — no REQUIREMENTS.md IDs mapped. Item goal verified directly (truths table).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| pyproject.toml | 180 | Comment "this extra stays the end-user torch<2.12 fast path" now stale against the raised ceiling | ℹ️ Info | Comment-only (no comma prefix, so correctly outside the `,<2.12`==0 gate and outside the suffix-scoped replacement); executor documented it for the docs follow-up item; not a bound, no resolver effect |
| docs/getting_started/installation.md + docs/faq/models_troubleshooting.md | — | Docs mirrors still say torch 2.12+ "must NOT be used" | ℹ️ Info | Explicitly out of scope per PLAN verification section; executor recommended a docs quick item to the orchestrator |
| PLAN Task 3 gate expression | — | `git show --stat` non-empty-line count can never equal 2 for a healthy two-file commit (trailing summary row) | ℹ️ Info | Plan-gate form defect only; intended property (exactly two files) verified with the precise instrument `--name-only` = 2. No work defect, no override needed |

### Human Verification Required

None. All five must-have truths are statically verifiable declarative properties (file contents, git history) and each was verified programmatically with certainty — no behavior-dependent truth, no state-transition invariant, no visual/external-service surface in scope.

**Owner post-land watch (informational, explicitly NOT a gate per PLAN):** nightly lanes are lockfile-free, so the first 03:00/05:30 UTC cycle implicitly jumps the fleet to torch 2.14.x and exercises the e4v patch pipeline fleet-wide. Watch one full cycle; any red attributable to the ceiling reverts single commit f00e90d. Expected torch-independent noise: the lazy-facade notebook regression (fixed by sibling 261025-e4x, already landed on this branch) and the test_sample_datasets 300s timeout singleton.

### Gaps Summary

No gaps. The goal is fully achieved in the codebase:

- Exactly seven torch bounds raised `<2.12` -> `<2.15` (base + cpu/cuda124/cuda126/cuda128/cuda130/mamba), floors byte-identical, zero old tokens, nothing wider than `<2.15` anywhere.
- Dependabot torch-ignore comment records the completed coordinated cycle citing run 40140203118 and the e4v c++20 sdist pipeline; the semver-minor/major ignore rule is byte-identical to before (comment-only diff).
- Landed as exactly ONE commit (f00e90d) on branch v1.4 touching exactly the two planned files, message citing run 40140203118, no attribution trailers.
- Dependency ordering honored: e4v (ef91f8d, the sha256-pinned c++20 patch pipeline in scripts/install_mamba.sh) is an ancestor of f00e90d, satisfying the plan's Task-3 precondition.
- Extras-guard regression suite green (14/14).

---

_Verified: 2026-10-25T11:45:00Z_
_Verifier: Claude (gsd-verifier)_
