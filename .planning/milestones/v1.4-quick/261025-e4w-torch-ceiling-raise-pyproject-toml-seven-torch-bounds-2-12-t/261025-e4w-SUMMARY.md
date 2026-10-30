---
phase: quick-261025-e4w
plan: 01
type: execute
subsystem: packaging
tags: [torch, ceiling, pyproject, dependabot, mamba-ssm, c++20]
requires:
  - "261025-e4v — mamba-ssm c++20 sdist patch pipeline in scripts/install_mamba.sh (sha256 pin 104cc47e, -std edits)"
  - "run 40140203118 evidence chain (.planning/research/torch-214-gb10-verification.md, .planning/async-jobs/spike-torch-214.json)"
provides:
  - "pyproject torch ceiling <2.15 across base + cpu/cuda124/cuda126/cuda128/cuda130/mamba — nightly fleet resolves 2.14.x on next lockfile-free cycle"
  - "dependabot torch-ignore comment records the completed coordinated kernel-rebuild + nightly-validation cycle (run 40140203118)"
affects:
  - "nightly lanes (coverage-nightly 03:00 UTC, example-nightly 05:30 UTC) — first cycle implicitly jumps to torch 2.14.x and exercises the e4v patch pipeline fleet-wide"
key-files:
  created: []
  modified:
    - pyproject.toml
    - .github/dependabot.yml
decisions:
  - "Ceiling stops exactly at <2.15 — ATen's C++20 floor is torch-2.14-specific; >=2.15 unverified and excluded (T-e4w-01 mitigation)"
  - "Dependabot semver-minor/major torch ignore retained byte-identical — the ceiling is an ABI claim Dependabot must never auto-widen"
  - "dependabot comment phrased without any literal '<2.12' token (gate requires zero) — old ceiling referenced as 'origin 5721ec3' instead"
metrics:
  duration: 4 min
  completed: 2026-10-25
status: complete
actuals:
  tasks: 3
  commits: 1
---

# Quick 261025-e4w: torch ceiling raise <2.12 -> <2.15 Summary

Raised the declared torch ceiling across all seven bound sites in pyproject.toml (base deps + cpu/cuda124/cuda126/cuda128/cuda130/mamba extras) from <2.12 to <2.15, and rewrote the .github/dependabot.yml torch-ignore comment to record the completed coordinated kernel-rebuild + nightly-validation cycle — one atomic commit (f00e90d) on v1.4 citing GitHub run 40140203118.

## What Was Done

### Task 1: pyproject.toml — seven torch bounds <2.12 -> <2.15

- Baseline confirmed at HEAD fac22a0: `,<2.12` count exactly 7 (all torch spec lines — line 65 base + lines 161/164/167/170/173/184 extras), `,<2.15` count 0, floors 3/3/1.
- Suffix-scoped replacement `,<2.12` -> `,<2.15` (replace-all exact: every comma-suffixed occurrence in the file is a torch bound). Diff is exactly 7 lines, nothing else touched — floors `>=2.4.0` x3, `>=2.6.0` x3, `>=2.9.0` x1 all unchanged; no comments, no `[[tool.uv.index]]` entries, no other bounds.
- Spike branch spike/torch-214-verification (commit 96420a0) was NOT merged or cherry-picked — the edit was typed fresh onto v1.4, as the plan requires (spike ci.yml SHA pinning must not merge).
- Gates: tomllib parse + 7 specs ending `, <2.15` (semantic extraction over base deps + six extras) green; count gates 7/0/3/3/1 green; `tests/test_extras_guard.py` 14 passed in 2.48s.

### Task 2: .github/dependabot.yml — torch-ignore comment records the completed cycle

- Rewrote ONLY the 9-line comment block above the torch ignore entry (18 new comment lines). New comment records: the cycle is COMPLETE and the ceiling is now <2.15; run 40140203118 (torch 2.14.1+cu130 resolved+installed on the GB10 runner, census 3215 passed, 2 failures torch-independent); causal-conv1d 1.7.0 source-compiles green as-is; mamba-ssm 2.3.2.post1 builds via the sha256-pinned sdist `-std=c++17 -> -std=c++20` patch pipeline in scripts/install_mamba.sh (kernel-rebuild half, landed with the mamba-extra install rework); prior ceiling origin 5721ec3 (2026-05-08) was forward-looking, never defect-driven; standing rule — ATen's C++20 floor is torch-2.14-specific, so any bump to >=2.15 repeats the same coordinated cycle.
- Corrected the stale parenthetical: mirror set is cpu/cuda124/cuda126/cuda128/cuda130/mamba (the old text claimed "every cuda/rocm/mamba extra"; there is no rocm extra).
- The ignore rule itself (`- dependency-name: "torch"` + update-types semver-minor/semver-major) is byte-identical — verified: the non-comment diff filter shows zero changed rule lines. All other entries (mcp, mamba-ssm, causal-conv1d, transformers comments, groups, labels, commit-message config) untouched.
- Gates: YAML parses; `40140203118` present; `<2.12` count 0 (the new comment deliberately avoids the literal, citing "origin 5721ec3" instead); `dependency-name: "torch"` exactly 1; `install_mamba` present; `semver-minor` present.

### Task 3: single atomic commit on v1.4

- Precondition verified before committing: `grep -c 'std=c++20' scripts/install_mamba.sh` = 6 and sha256 pin `104cc47e` present (e4v pipeline on branch); HEAD on v1.4 (main frozen per owner directive).
- Pre-commit HEAD safety assertion run: branch v1.4, not protected (gsd_run confirmed false).
- Staged exactly pyproject.toml + .github/dependabot.yml; commit f00e90d with subject "deps(torch): raise ceiling <2.12 -> <2.15 across base + six extras — verified on GB10, run 40140203118" and a four-paragraph body covering: run 40140203118 criterion-a (2.14.1+cu130 installed, census 3215 passed); the mamba-ssm c++20 blocker resolved by the e4v patch pipeline; causal-conv1d green as-is; origin 5721ec3 forward-looking; the dependabot comment update; the lockfile-free nightly fleet jump with single-commit revert as rollback. No attribution trailers.
- Gates: branch v1.4; commit contains exactly the two files; message cites 40140203118 (2x); zero co-authored; both files clean in porcelain; HEAD:pyproject.toml (committed blob, not working tree) carries the seven <2.15 bounds via tomllib.
- Post-commit deletion check: no deletions. No untracked files introduced (git status shows only pre-existing orchestrator-owned .planning changes).

## Deviations from Plan

None in work product. One plan-gate form note (not a work deviation — the underlying property holds):

**Gate-form note: Task 3's `--stat` file-count link**

- The gate link `[ "$(git show --stat --format= HEAD | grep -cv '^$')" = "2" ]` can never equal 2 for a healthy two-file commit: `--stat` output includes the trailing "N files changed" summary row, so a correct commit yields 3 non-empty lines (2 file rows + 1 summary).
- The intended property — exactly two files in the commit — was verified with the precise instrument: `git show --name-only --format= HEAD | grep -cv '^$'` = 2, listing exactly `.github/dependabot.yml` and `pyproject.toml`. All other gate links green as written.

## Threat Model Disposition

- T-e4w-01 (supply-chain surface widening): mitigated — raise bounded exactly to <2.15, the verified 2.12–2.14 line; floors untouched; Task 1 semantic gate proved exactly 7 specs ending `, <2.15` with zero `, <2.12`; dependabot minor/major ignore retained.
- T-e4w-02 (nightly fleet breakage): mitigated — hard depends_on e4v honored; Task 3 precondition grep of the c++20 pipeline markers passed pre-commit; rollback is single-commit revert.
- T-e4w-SC (package legitimacy): accepted as planned — this change installs nothing; the only new build input (mamba-ssm sdist) enters via e4v's sha256-verified pipeline.

## Verification Results

All plan gates green (Task 3's --stat link satisfied in intent via --name-only, above). Extras-guard 14/14 passed. TOML and dependabot YAML parse. Known non-gates left untouched as the plan directs: tests/models/test_plant_helixseek_fla_kernels.py:66 literal "torch>=2.4.0,<2.12" parser input stays; CHANGELOG history untouched; ty stub env (torch==2.10.0) sits inside the new range.

Post-land owner watch (not an automated gate here): nightly is lockfile-free — the first 03:00 UTC coverage-nightly and 05:30 UTC example-nightly cycle implicitly jumps the fleet to 2.14.x; watch one full cycle. Expected torch-independent noise: the lazy-facade notebook regression (already fixed by sibling 261025-e4x) and the test_sample_datasets 300s timeout singleton. Any red attributable to the ceiling: revert f00e90d alone.

## Follow-up for Orchestrator (out of scope by plan)

- Docs mirrors go stale the moment this lands (plan's verification section, recommend a docs quick item): docs/getting_started/installation.md lines 155, 174-175, 260-267, 269 and docs/faq/models_troubleshooting.md line 26 still say torch 2.12+ "must NOT be used" with the old extras table; docs/getting_started/gpu_support.md:28 states an upstream cu128-wheel fact, true regardless — leave.
- Minor prose nit found during execution (plan forbade touching it — "no comments", diff confined to seven lines): the e4v-landed mamba-extra comment in pyproject.toml (line 180, "this extra stays the end-user torch<2.12 fast path") now reads stale against the raised ceiling — the literal has no comma prefix so it correctly stayed out of the suffix-scoped replacement and out of the `, <2.12`==0 gate. Fold into the docs follow-up item if desired.

## Self-Check: PASSED

- Commit f00e90d exists on v1.4 and is HEAD: `git merge-base --is-ancestor f00e90d HEAD` OK; `git show --name-only` = exactly pyproject.toml + .github/dependabot.yml.
- Both modified files exist and carry the changes (working tree clean for both).
- Message cites 40140203118; zero attribution trailers.
