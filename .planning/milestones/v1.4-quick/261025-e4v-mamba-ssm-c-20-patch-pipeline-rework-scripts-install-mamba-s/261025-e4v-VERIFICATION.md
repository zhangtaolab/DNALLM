---
phase: quick-261025-e4v
verified: 2026-10-25T03:04:29Z
status: passed
score: 7/7 must-haves verified
covered_files:
  - .github/workflows/ci.yml
  - .planning/quick/261025-e4v-mamba-ssm-c-20-patch-pipeline-rework-scripts-install-mamba-s/261025-e4v-PLAN.md
  - .planning/quick/261025-e4v-mamba-ssm-c-20-patch-pipeline-rework-scripts-install-mamba-s/261025-e4v-SUMMARY.md
  - README.md
  - docs/getting_started/installation.md
  - pyproject.toml
  - scripts/install_mamba.sh
covered_digest: "v3:sha256:47ed1dc9fd488feded893153952d61457463ff8905ebb6aa450931bed4f43c35"
behavior_unverified: 0
overrides_applied: 0
---

# Quick 261025-e4v: mamba-ssm sha256-pinned sdist c++20 patch pipeline Verification Report

**Item Goal:** Rework scripts/install_mamba.sh and the CI test-mamba/example-nightly install lines to build mamba-ssm 2.3.2.post1 from the sha256-verified sdist (104cc47e…d028) with the four -std=c++17 -> -std=c++20 setup.py edits (upstream PR 1000 equivalent), replacing the guessed-URL wheel path; reproducible and TUNA-mirror compatible for the GB10 runner.
**Verified:** 2026-10-25T03:04:29Z
**Status:** passed
**Re-verification:** No — initial verification
**Landed as:** commits 628067a (script), 8b6ae10 (ci.yml), ef91f8d (docs/pyproject) — all verified ancestors of HEAD (v1.4); verified against the current tree.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Script constructs zero GitHub wheel URLs; fetches the pinned sdist from a --index-url-configurable index (default https://pypi.org/simple); sha256 verified BEFORE any extraction/patch/build; fails closed (non-zero exit, no wheel) on mismatch; no flag/env overrides the hash constant | ✓ VERIFIED | grep: zero matches for github.com/codeload/releases/download/git ls-remote/git clone in scripts/install_mamba.sh; default INDEX_URL at line 79. Behavioral (stronger than the plan's own negative test): a local PEP 503 index serving corrupted bytes while DECLARING the correct #sha256= fragment was served to the full (non---patch-only) pipeline — the script's local sha256sum gate aborted with exit 1 ("sha256 MISMATCH … refusing to extract, patch, or build"), log contains 0 "Extracting"/"Patching"/"Building" lines, no wheel produced. `--verify-file` on a `\0`-corrupted copy: exit 1 with expected-vs-actual. Constant assigned exactly once (line 61), referenced only in comparisons/messages; no env-var reads exist in the script. |
| 2 | Patch applies -std=c++17 -> -std=c++20 inside the EXTRACTED setup.py tree (never the tarball path), occurrence-exact counting (grep -o \| wc -l): exactly 4 before, 0 after; wheel build from the patched directory with MAMBA_FORCE_BUILD=TRUE, --no-build-isolation, --no-deps | ✓ VERIFIED | Script lines 228-251: extract into fresh mktemp -d, counts via `grep -o -- '-std=c++17' \| wc -l` asserted 4 before / 0 remain / 4 c++20 after, any other count errors. Lines 276-277: `MAMBA_FORCE_BUILD=TRUE pip wheel "$SRC_DIR" --no-build-isolation --no-deps --no-cache-dir -w` — target is the extracted SRC_DIR, never the tarball. Counting behaviorally proven: --patch-only runs on the real sdist asserted 4/4 flips and printed "PATCH VERIFIED: sha256 ok, 4/4 flags flipped, 0 remain" on both index routes. Build command structure matches the B+C-proven invocation; the physical GB10 build re-proves via the scheduled CI runs (plan-committed scope — see Verification Scope note). |
| 3 | ci.yml test-mamba no longer installs mamba-extra kernels via uv; bootstraps pip into the uv venv and invokes the script with the TUNA index URL | ✓ VERIFIED | ci.yml lines 360-361: `uv pip install --python .venv/bin/python pip` then `bash scripts/install_mamba.sh --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`. Commit 8b6ae10 diff shows the old `uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation` line removed; grep for mamba-extra install variants across ci.yml = 0. Everything through `uv pip install -e ".[full,fla]"` (venv pin, LDFLAGS, extras rationale) kept unchanged. |
| 4 | ci.yml example-nightly builds its wheelhouse via the script (--wheelhouse wheelhouse-mamba --no-install) and the wheelhouse cache key carries a cxx20patch marker | ✓ VERIFIED | ci.yml line 842: key `mamba-conv1d-1.7.0-ssm-2.3.2.post1-cxx20patch-${{ runner.arch }}-…` with the discriminator comment; lines 854-858: cache-miss guard + pip bootstrap + script invocation with `--wheelhouse wheelhouse-mamba --no-install`; line 859 `uv pip install wheelhouse-mamba/*.whl` unchanged (context lines in the 8b6ae10 diff). grep: `cxx20patch` count = 1, script-invocation count = 2, `pip wheel` remnants = 0. |
| 5 | causal_conv1d==1.7.0 stays unpatched and builds from sdist through the same index route | ✓ VERIFIED | Script lines 279-283: `pip wheel causal_conv1d==1.7.0 --no-build-isolation --no-cache-dir --index-url $INDEX_URL`; the sed patch touches only the mamba sdist's SETUP_PY — causal_conv1d is never extracted or patched. Install step (297-298) routes both wheels through the same index. |
| 6 | Pipeline needs zero GitHub network access — no github.com, no codeload route | ✓ VERIFIED | grep over scripts/install_mamba.sh: zero matches for github.com/codeload/git commands; the fetch is a pure python-stdlib PEP 503 page parse + streamed download. (ci.yml:875 `git clone …/megaDNA.git` is the pre-existing megaDNA stage, unrelated to the kernel pipeline.) |
| 7 | --patch-only mode reproduces the B+C patch verification without CUDA and runs green on both the pypi.org and TUNA routes from the dev box | ✓ VERIFIED | Behavioral: `bash scripts/install_mamba.sh --python .venv/bin/python --patch-only` exit 0 + "PATCH VERIFIED: sha256 ok, 4/4 flags flipped, 0 remain" (pypi.org route); same with `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` (TUNA route), log shows `fetched … from https://pypi.tuna.tsinghua.edu.cn/packages/…` and `sha256 verified: 104cc47e…d028` — both routes served byte-identical archives matching the pin. --patch-only skips the torch/CUDA environment check (guard at lines 147-162) and exits before pip is ever required. |

**Score:** 7/7 truths verified (0 present, behavior-unverified)

### Verification Scope Note (plan-committed boundary)

The full GB10 kernel build (script steps 5-7: `pip wheel` of the patched tree) was intentionally not executed during verification — per the plan's own verification section, the in-repo proof is the patch-only path plus the grep/parse gates (all green above, plus the stronger corrupted-bytes gate test run by this verifier), and the end-to-end build re-proves automatically in the next scheduled test-mamba/example-nightly runs (an owner workflow_dispatch on v1.4 is the immediate option). Failure there is loud by design: test-mamba red blocks item 261025-e4w. The B+C experiment (2026-10-25, cited in plan context) already proved this exact patched build compiles (cp313 wheel) and matches reference at 5.7e-06 on the GB10 box.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| scripts/install_mamba.sh | sha256-pinned sdist c++20 patch pipeline with all 7 modes | ✓ VERIFIED | 299 lines, full fail-closed pipeline; modes --default/--patch-only/--verify-file/--index-url/--python/--wheelhouse/--no-install all present; substantive (no stubs); wired (2 CI invocation sites) |
| .github/workflows/ci.yml | both kernel-install sites routed through the script; cxx20patch cache key | ✓ VERIFIED | Lines 360-361 (test-mamba) and 854-858 (example-nightly); cache key line 842; YAML parses (yaml.safe_load OK); scheduling/gating lines byte-identical before/after 8b6ae10 (23 if:/cron lines, diff empty) |
| README.md | stale GitHub-connectivity/proxy guidance replaced | ✓ VERIFIED | "connect to GitHub" = 0, "github proxy" (ci) = 0; ef91f8d diff replaces both stale statements (proxy comment -> checksum-verified source + mirror flag; GitHub sentence -> PyPI route), plain-language style per owner directive |
| docs/getting_started/installation.md | same two fixes | ✓ VERIFIED | Same gates = 0; diff confirms both replacements; the untouched uv-extra and plain-script call sites remain accurate |
| pyproject.toml | pointer comment above the mamba extra, zero bound changes | ✓ VERIFIED | Comment block added above `mamba = [` (fla-comment style); diff grep for `^[+-]\s*"` = 0 — comment lines only; TOML parses; dependency strings byte-identical (bounds reserved for 261025-e4w) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| sha256 constant | fail-closed gate ordering | fetch -> hash verify -> extract fresh dir -> patch with count assertions -> pip wheel the patched directory; nothing executes from the archive before the hash gate | ✓ WIRED (documented deviation) | **Deviation (SUMMARY-documented, verified here):** the fetch step uses a direct python-stdlib PEP 503 simple-page fetch, not the planned `pip download --no-binary :all:` — because pip EXECUTES setup.py from the unverified archive during sdist metadata builds, violating this very link's "nothing executes before the hash gate" clause (observed traceback: isolated metadata build importing torch). The replacement strengthens the contract: exact `.tar.gz` filename demanded (no resolver can substitute a wheel — closes the wheel-bypass route by construction), plus a pre-download cross-check of the index-declared #sha256= fragment. Ordering behaviorally proven by the corrupted-bytes local-index test (gate aborts pre-extract). No execution from the archive occurs anywhere before the gate — verified in the fetch code (pure stdlib urlopen + regex). |
| ci.yml test-mamba install step | scripts/install_mamba.sh | `--python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple` with pip bootstrapped first | ✓ WIRED | Lines 360-361; old mamba-extra line verifiably removed in 8b6ae10 |
| ci.yml example-nightly stage-0 | wheelhouse cache | script `--wheelhouse wheelhouse-mamba --no-install` -> unchanged `uv pip install wheelhouse-mamba/*.whl` -> cache key cxx20patch marker | ✓ WIRED | Lines 842, 854-859; the uv wheel-install line is unchanged diff context |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| scripts/install_mamba.sh | sdist bytes | live PEP 503 index fetch (pypi.org and TUNA both exercised) | ✓ | FLOWING — real downloads, real sha256 verification against the pin, real extraction/patch of the actual upstream setup.py |
| .github/workflows/ci.yml | kernel wheels | script invocations (2 sites) | ✓ | FLOWING — both install paths route through the script; no static/unpatched kernel build command remains |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Script syntax | `bash -n scripts/install_mamba.sh` | clean | ✓ PASS |
| Patch-only, pypi.org route | `bash scripts/install_mamba.sh --python .venv/bin/python --patch-only` | exit 0; "PATCH VERIFIED: sha256 ok, 4/4 flags flipped, 0 remain" (count grep = 1) | ✓ PASS |
| Patch-only, TUNA route | same + `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` | exit 0; PATCH VERIFIED; sha256 verified = 104cc47e…d028 | ✓ PASS |
| Full-pipeline gate fail-closed (corrupted bytes, index declares CORRECT hash) | local http server serving corrupt.tar.gz with correct #sha256= fragment; run script WITHOUT --patch-only | exit 1; "sha256 MISMATCH … refusing to extract, patch, or build"; 0 Extracting/Patching/Building log lines; no wheel | ✓ PASS |
| --verify-file clean copy | `--verify-file clean.tar.gz` | exit 0, match message | ✓ PASS |
| --verify-file corrupted copy (`\0` appended) | `--verify-file corrupt.tar.gz` | exit 1, expected-vs-actual printed | ✓ PASS |
| CI grep gates | invocation count / cxx20patch / pip wheel / mamba-extra | 2 / 1 / 0 / 0 (all as required) | ✓ PASS |
| Workflow YAML parses | `.venv/bin/python -c "import yaml; yaml.safe_load(...)"` | YAML OK | ✓ PASS |
| pyproject TOML parses + comment-only diff | tomllib.load + diff grep for dependency-string changes | TOML OK; 0 dependency-string changes | ✓ PASS |
| Scheduling/gating untouched | if:/schedule:/cron lines before vs after 8b6ae10 | identical (23 lines, empty diff) | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared or conventional for this item; the plan's runnable checks are the task verify blocks, re-executed above (all PASS).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| E4V-01 | 261025-e4v-PLAN.md | Plan-local requirement ID for the mamba c++20 patch pipeline (no .planning/REQUIREMENTS.md registry entry exists for quick items) | ✓ SATISFIED | Truths 1-7 all verified; the artifacts satisfy the requirement's full scope (script, CI wiring, docs, pyproject pointer) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| docs/getting_started/installation.md | 594 | `XXX` regex hit — false positive: literal "cuXXX" CUDA build-tag notation | ℹ️ Info | Pre-existing (commit 33935c4), not modified by this item; not a debt marker |
| .github/workflows/ci.yml | 138-140 | Comment still says "the compiled mamba kernels (.[mamba]) stay on the linux dnallm-nightly box" (test-windows job) | ℹ️ Info | Claim remains true; the extra-name parenthetical is mildly imprecise now that CI installs via the script — outside the plan's declared refresh scope (it named lines 269-270, 281-283 only); no action required |

### Human Verification Required

None within the plan-committed scope. All must-have truths were verified with in-repo evidence (static gates plus behavioral runs on both index routes and both negative tests). The end-to-end GB10 kernel build is scheduled automation with loud gating (a red test-mamba blocks 261025-e4w), not a human UAT item — per the plan's verification section and the dispatching context, judge on the plan's committed proof, which is fully green.

### Gaps Summary

No gaps. All 7 must-have truths verified, all 5 artifacts present/substantive/wired with real data flow, all 3 key links wired (one documented, security-strengthening deviation on the fetch mechanism, called out above), all task verify blocks independently re-run green, scheduling/gating provably untouched, pyproject provably comment-only. Watch item for the owner: the next scheduled test-mamba/example-nightly (or a workflow_dispatch on the landed v1.4 commit) is the automatic end-to-end re-proof; an interim torch-2.11 patched-build failure there would be loud, correct gating — capture the mamba-test-logs artifact before any retry (per plan).

---

_Verified: 2026-10-25T03:04:29Z_
_Verifier: Claude (gsd-verifier)_
