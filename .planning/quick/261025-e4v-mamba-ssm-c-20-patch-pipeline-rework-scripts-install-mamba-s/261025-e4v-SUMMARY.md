---
phase: quick-261025-e4v
plan: 01
subsystem: build-pipeline
tags: [mamba-ssm, cuda-kernels, supply-chain, ci, c++20, torch-214]
requires:
  - "torch-214-gb10-verification.md B+C verdict (4-flag patch proven; sdist sha256 104cc47e…d028)"
provides:
  - "scripts/install_mamba.sh — sha256-pinned sdist c++20 patch pipeline (--patch-only / --verify-file / --index-url / --python / --wheelhouse / --no-install)"
  - "CI test-mamba + example-nightly kernel installs routed through the pipeline with a patch-marked wheelhouse cache key"
  - "Unblocks 261025-e4w (torch ceiling raise): its depends_on artifacts all landed"
affects:
  - ".github/workflows/ci.yml (test-mamba, example-nightly stage-0, wheelhouse cache key)"
  - "README.md, docs/getting_started/installation.md (mamba install guidance)"
  - "pyproject.toml (comment only, zero bound changes)"
tech-stack:
  added: []
  patterns:
    - "fail-closed gate ordering: fetch bytes -> sha256 verify -> extract -> patch -> build; nothing from the archive executes before the hash gate"
    - "direct PEP 503 simple-page fetch (python stdlib) instead of pip resolution, with index-declared sha256 fragment cross-check pre-download"
key-files:
  created: []
  modified:
    - scripts/install_mamba.sh
    - .github/workflows/ci.yml
    - README.md
    - docs/getting_started/installation.md
    - pyproject.toml
decisions:
  - "Fetch reworked from pip download to a direct PEP 503 stdlib fetch: pip executes setup.py from the unverified archive during sdist metadata builds, violating the plan's own no-execution-before-hash ordering; the direct fetch also makes --patch-only truly torch/pip-free"
  - "pip bootstrap check moved to the build path (exact hint text kept) so --patch-only runs on pip-less uv venvs"
metrics:
  duration: 11 min
  completed: 2026-10-25
actuals:
  tokens: 7500   # chars/4 over the realized diff (30012 chars, git diff c03189e..HEAD)
  tasks: 3
  commits: 3     # measured: git rev-list --count c03189e..HEAD
  plan_head_before: c03189e
  plan_head_after: ef91f8d
status: complete
---

# Quick 261025-e4v: mamba-ssm sha256-pinned sdist c++20 patch pipeline Summary

One-liner: install_mamba.sh now fetches the sha256-pinned mamba-ssm 2.3.2.post1 sdist from any PyPI index route, flips the four `-std=c++17` -> `-std=c++20` flags with occurrence-exact assertions, and builds with `MAMBA_FORCE_BUILD=TRUE --no-build-isolation --no-deps`; both CI kernel-install sites (test-mamba, example-nightly wheelhouse) route through it via TUNA with a patch-marked cache key.

## What Was Built

### Task 1 — scripts/install_mamba.sh rework (commit 628067a)

Full pipeline rewrite keeping `set -e`, the info/warn/error helpers, and a build-path-only environment check (torch-presence guard + informational logging; `--patch-only` skips it). Deleted: interactive GitHub-proxy selection, `git ls-remote` version listing, version prompts, wheel-URL construction, URL install loop. Pinned constants at the top (`MAMBA_SSM_VERSION=2.3.2.post1`, `CAUSAL_CONV1D_VERSION=1.7.0`, `MAMBA_SDIST_SHA256=104cc47e…d028` — no override flag or env exists).

Fail-closed flow in order: (1) fetch the exact pinned sdist filename from the index's simple page; (2) sha256sum gate with expected-vs-actual output — mismatch aborts before any extraction; (3) extract into a fresh `mktemp -d`, later steps receive the path (nothing runs from inside the tree); (4) global `-std=c++17` -> `-std=c++20` substitution wrapped by `grep -o | wc -l` occurrence-exact assertions (4 before / 0 c++17 after / 4 c++20 after — any other count is a hard error); `--patch-only` prints `PATCH VERIFIED: sha256 ok, 4/4 flags flipped, 0 remain` and exits 0 here; (5) `MAMBA_FORCE_BUILD=TRUE pip wheel $SRC_DIR --no-build-isolation --no-deps --no-cache-dir` (literal string TRUE, canonical upstream gate; built from the patched directory, never the tarball); (6) causal_conv1d 1.7.0 built unpatched from the same index (`--no-build-isolation`); (7) unless `--no-install`, both wheels installed with deps resolving from the index. `TORCH_CUDA_ARCH_LIST` deliberately never set (caller-exported values pass through untouched).

Modes: `--default` (no-op, one info line), `--patch-only`, `--verify-file PATH` (pure sha256sum check, exit 0/1, needs no python), `--index-url URL` (default `https://pypi.org/simple`), `--python PATH`, `--wheelhouse DIR`, `--no-install`. Missing pip errors with the exact bootstrap hint (`uv pip install --python <venv> pip` or ensurepip).

### Task 2 — CI wiring (commit 8b6ae10)

- **test-mamba** step "Create virtual environment and install mamba dependencies": kept everything through `uv pip install -e ".[full,fla]"` (venv pinning comment, LDFLAGS export, extras rationale); the kernel-install line is replaced by `uv pip install --python .venv/bin/python pip` + `bash scripts/install_mamba.sh --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`, with the extended step comment (run 40140203118 evidence, PR #1000 unmerged, wheel matrix stops at torch 2.10, TUNA route + sha256-pin rationale, mamba extra stays the end-user torch<2.12 fast path, and the interim torch-2.11 patched-build note until 261025-e4w lands). Both stale job-header sentences refreshed to name the script pipeline; scheduling/gating/timeout untouched.
- **example-nightly** stage-0: cache-miss guard and pip bootstrap kept; the index wheel-build command replaced by `bash scripts/install_mamba.sh --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple --wheelhouse wheelhouse-mamba --no-install`; trailing uv wheel-install line unchanged; step comment updated with the evidence citation.
- **Wheelhouse cache key** gains the patch marker segment with a one-line discriminator comment — a wheel cached before the patch can never restore under an identical torch version.

### Task 3 — user-facing text sync (commit ef91f8d)

- README.md and docs/getting_started/installation.md: the proxy-selection comment on the script line replaced (checksum-verified source build + the optional `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` mirror flag); the GitHub-connectivity sentence replaced with "the script downloads from PyPI (or the mirror flag), so GitHub is not needed". Plain-language style: one sentence of what changed, one of what a user does. The troubleshooting block (uv extra command + plain `sh scripts/install_mamba.sh`) left untouched — still accurate.
- pyproject.toml: comment block above the `mamba = [` extra (fla-comment style) explaining why source installs on torch 2.11+ must go through the script. Zero dependency-string changes (bounds are 261025-e4w's single commit).

## Verification Results

All three task verify blocks green on this box:

- **Task 1**: `bash -n` clean; `--patch-only` green on BOTH routes — TUNA (exit 0, PATCH VERIFIED line, count grep = 1) and the default pypi.org route (exit 0, PATCH VERIFIED) — proving both routes serve hash-identical bytes (`104cc47e…d028` verified on each); corrupted-archive negative test: one appended `\0` byte -> `--verify-file` printed expected-vs-actual and exited 1 (FAIL-CLOSED OK); `releases/download` count in script = 0; hash constant present.
- **Task 2**: script invocation count in ci.yml = 2 (both sites); `cxx20patch` = 1 (cache key only); `pip wheel` remnants = 0; mamba-extra install command remnants = 0; YAML parses (yaml.safe_load OK); `if:` gating and schedule entries byte-identical to before.
- **Task 3**: `connect to GitHub` = 0 and `github proxy` (case-insensitive) = 0 in both docs; `install_mamba.sh` referenced in pyproject; TOML parses; pyproject diff contains comment lines only (zero dependency-string changes).

Full end-to-end build was not run on this dev box (owner directive: targeted verification only) — per the plan's verification section, the end-to-end proof re-proves automatically in the next scheduled test-mamba / example-nightly runs (and an owner workflow_dispatch of CI on the landed v1.4 commit is the recommended immediate check).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 + Rule 2 - Blocking issue + security ordering] pip-mediated fetch replaced with a direct PEP 503 stdlib fetch**
- **Found during:** Task 1 verification
- **Issue:** The plan's step-1 command (`pip download ... --no-binary :all:`) fails outright on this pip (26.2.1): the observed traceback proves pip EXECUTES setup.py from the downloaded archive in an isolated metadata build (`ModuleNotFoundError: No module named 'torch'` from `/tmp/pip-build-env-*/.../run_setup()`). That both (a) hard-fails wherever the isolated env lacks torch — breaking the plan's own "--patch-only must work on boxes without torch/CUDA" requirement — and (b) runs unverified archive code BEFORE the hash gate, violating the plan's key_link "nothing executes from the archive before the hash gate passes".
- **Fix:** `fetch_sdist()` helper: python-stdlib resolution of the exact pinned filename on the index's PEP 503 simple page, an early cross-check of the index-declared `#sha256=` fragment against the pinned constant (fails closed pre-download with the route named — a bonus control strengthening T-e4v-01), then a streamed byte download. Wheel-substitution bypass is closed by construction (the exact `.tar.gz` filename is demanded; no resolver ever sees a wheel), replacing the `--no-binary :all:` mechanism. Consequence: the pip bootstrap check moved to the build path (exact hint text kept) so `--patch-only` is now genuinely torch/pip/CUDA-free. Verified green on both routes.
- **Files modified:** scripts/install_mamba.sh
- **Commit:** 628067a

**2. [Environment adaptation] Negative-test setup command needed two flags**
- **Found during:** Task 1 verification
- **Issue:** The verify block's plain `pip download` for staging the clean sdist in /tmp/e4v-neg cannot run as written on this box: plain pip ignores the user-level uv config (env fact — explicit `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` needed) and the isolated metadata build lacks torch (`--no-build-isolation` needed; host venv has torch).
- **Fix:** Ran the staging download with both flags added; the actual gate under test (`--verify-file` on the corrupted archive) ran exactly as specified — FAIL-CLOSED OK with expected-vs-actual output.
- **Files modified:** none (verify-time adaptation only)

**3. [Environment observation] pypi.org route proved live, not dead**
- **Found during:** Task 1 verification
- **Issue:** Env facts predicted pypi.org dead (0 B/s) and instructed a plumbing-only check; the live run showed the route fully green for the 216 kB sdist (files.pythonhosted.org fetch succeeded, pinned hash verified, PATCH VERIFIED, exit 0).
- **Fix:** None needed — the plan's original dual-route green expectation was met without modification; both routes served byte-identical bytes. Recorded because it contradicts the session's env facts (the dead-state likely applies to bulk throughput, not small files, or the route recovered).

## Auth Gates

None.

## Known Stubs

None — no dnallm/ package code was touched (the plan intentionally touches none, so the same-change pytest rule never triggered).

## Self-Check: PASSED

- Commits 628067a, 8b6ae10, ef91f8d all verified as ancestors of HEAD (v1.4)
- All five modified files present on disk; worktree clean, no untracked leftovers
- 3/3 task verify blocks green (details above)
