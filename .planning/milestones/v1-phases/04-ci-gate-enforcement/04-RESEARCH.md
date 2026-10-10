# Phase 4: CI Gate Enforcement - Research

**Researched:** 2026-09-30
**Domain:** CI/CD infrastructure — coverage fail-under enforcement, GitHub Actions job design, HF/ModelScope model caching, synthetic-regression proof mechanics
**Confidence:** HIGH (every gate-mechanic claim probed live this session; the one open measurement — CI-side runtime — is bounded by a local CPU probe and flagged for first-run calibration)

## Summary

Phase 4 turns the existing green 96.30% suite into an enforced ratchet. The enforcement mechanism itself is a one-line config change with verified exit-code plumbing: this session probed, in an isolated `/tmp` project with the repo's own venv (pytest-cov 7.1.0 / coverage 7.16.2), that `fail_under = 90` in `[tool.coverage.report]` plus bare `pytest --cov` yields **rc=1 with `ERROR: Coverage failure: total of 80 is less than fail-under=90`** when all tests pass but coverage is low, **rc=0 at exactly 90.00%** (>= semantics), and bare `coverage report` exits 2. A second live measurement defused the biggest hidden interaction: the existing fast leg (`-m "not slow" --cov`) covers **96% (275/7405 missing)** on its own, so adding `fail_under` does not break the six matrix legs — the slow tests are worth only ~1 statement of coverage (7131 covered full-census vs ~7130 fast-only). The gated job's value is slow-suite health plus enforcement on the full denominator, not coverage points.

The models.lock question resolves into a 9-entry manifest: only **2 downloads are HF-sourced** (`microsoft/DialoGPT-small`, `zhangtaolab/plant-dnagpt-BPE-promoter` via `from_pretrained`), whose cold HF scratch grew to 2.0G in the Phase-1 audit; the other **7 are ModelScope-sourced** (~2.97G measured locally with `du`), and ModelScope ignores `HF_HOME` — so the cache must cover `~/.cache/huggingface/hub` AND `~/.cache/modelscope/hub`, keyed on `hashFiles('models.lock')`. actions/cache@v4 restores caches from the current branch, the default branch, and the PR base branch, saves only on green jobs (`post-if: "success()"` in the action's action.yml), and the repo total (~5G models + ~2-3G uv) fits the 10G per-repo cap.

The dominant risk is runtime, not correctness. A completed local CPU-only probe (`CUDA_VISIBLE_DEVICES=""`, this session) ran both heavy tests green: `test_qlora_training` + `test_complete_training_workflow` together took **2860.29s (47:40) on a 10-core CPU vs 220s with GPU — a 13.0x factor** (~4.5 min for qlora, ~43 min for complete_training at ~1018% CPU and **~15.9 GB RSS**, i.e. at the 16 GB ubuntu-latest RAM ceiling; bitsandbytes NF4 does not crash CPU-only). Extrapolated: the warm GPU slow leg (819.6s) becomes ~2.96 h on local CPU, and a 4-core runner adds a further ~1.5-2.5x core-scaling → **the gated job is realistically a 4-7.5 h job on standard ubuntu-latest — longer than GitHub's 360-minute default job timeout**. The existing `--timeout=300` in addopts would kill every long trainer test even before that. The plan must therefore: (1) add `@pytest.mark.timeout(N)` overrides (verified: the marker beats the CLI/ini value — behavioral probe killed a `sleep(4)` test at 2.02 s while `--timeout=300` was set), (2) set an explicit job `timeout-minutes` high enough to survive the first full run (480 initial, then tighten), (3) run a `workflow_dispatch` calibration run on CI before locking numbers, and (4) surface the measured runtime expectation to the owner with the explicit option menu (accept / larger runner / LANE-01 pull-forward), since the "runtime cost accepted" decision predates these numbers.

**Primary recommendation:** Add `fail_under = 90` to `[tool.coverage.report]`; create a root `models.lock` manifest (9 entries) as the actions/cache key; add a single `coverage-gate` job (py3.12, numpy 2.2.0, full suite via the census command of record) to the existing `ci.yml` with per-test timeout marks, job `timeout-minutes`, and `workflow_dispatch`; prove GATE-04 by pushing a branch that deletes `tests/models/test_model.py` (deterministic drop to ~81%), assert the CI job fails with the coverage-failure exit code via `gh`, then delete the branch; drop the codecov step (native gate replaces it) unless the owner wants v7 dashboards, which require a token secret or `id-token: write`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (pre-locked by project decisions — do not re-litigate)
- Gate ratchet: enabled only now that the suite is above it (never permanently red) — 90, not 96
- The gated run includes `slow` tests (network accepted by owner)
- Codecov is reporting-only at most, never the gate
- GATE-02's `models.lock` cache key: the repo has no lockfile committed — the plan must either create the models.lock manifest or pick the equivalent cache-key mechanism; ratchet semantics stay
- Deferred from Phase-1 review (WR-02/WR-05/WR-06 dispositions): mamba no-op GPU leg, workflows-README broad staleness, uv installer pinning are OWNER decisions visible at this phase's CI rework — surface them in the plan as explicit decision points or leave untouched with rationale; do not silently expand scope

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP goal, success criteria, GATE-01..05, and codebase conventions.

### Deferred Ideas (OUT OF SCOPE)
None — discuss skipped (infrastructure phase). Out of scope per domain: two-lane CI split, patch coverage, nightly drift (v2 backlog per REQUIREMENTS.md).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GATE-01 | `fail_under = 90` in `[tool.coverage.report]` — enabled only after the suite first crosses 90% (ratchet) | Exit-code plumbing verified live (rc=1 below, rc=0 at exactly 90.00%); fast-leg measured 96% so nothing breaks; exact pyproject edit site identified (pyproject.toml:512-514 comment block) |
| GATE-02 | Dedicated single-leg slow-inclusive coverage CI job (py3.12, full suite, HF model cache keyed on `models.lock`, per-test timeout marks + job-level `timeout-minutes` backstop) | models.lock manifest enumerated (9 entries, 2 HF + 7 ModelScope, ~5G total); cache key/paths/restore-keys designed; timeout-marker precedence verified; runtime risk measured (CPU probe) with calibration-first plan |
| GATE-03 | Fix or remove the dead `codecov-action@v3` step (→ `@v7` or drop); reporting only, never the gate | v3 status confirmed (works today, but "v3 versions and below will not have access to CLI features"); v7 requirements documented (token secret or `use_oidc` + `id-token: write`); removal recommended |
| GATE-04 | Synthetic-regression proof that the gate actually fails CI when coverage drops (end-to-end exercise of HARN-02) | Deterministic drop recipe (delete `tests/models/test_model.py` → ~81%); gh CLI auth verified (repo+workflow scopes); PR-to-dev trigger already live in ci.yml `on:`; safe probe mechanics (branch → PR → assert red check → delete) |
| GATE-05 | Gate runs on PRs to every protected branch (`dev` and `main`), not just `main` | ci.yml already triggers `pull_request: branches: [ main, master, dev ]` — the new job inherits it; GATE-04's PR doubles as the proof; NOTE: neither branch is currently protected (checked via API) — "protected" is aspirational; making checks required is an owner action |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Coverage enforcement (fail_under) | CI / test tooling | Local dev (same pyproject) | Enforcement lives in coverage config consumed by pytest exit code — identical local/CI by construction |
| Gate job orchestration | GitHub Actions (ci.yml) | — | Single new job in the existing workflow inherits triggers/permissions; no new workflow file needed |
| Model download caching | GitHub Actions cache | Runner filesystem (~/.cache/{huggingface,modelscope}/hub) | actions/cache@v4 with hashFiles key; HF/ModelScope caches are opaque blob+symlink trees, cached whole |
| Timeout bounding | pytest-timeout (per-test marks) | Job `timeout-minutes` (backstop) | Marker overrides global 300s addopts value (verified); job timeout bounds total wall clock |
| Regression proof (GATE-04) | Git/GitHub (branch + PR + gh CLI) | — | Ephemeral branch, PR to dev, assert red check, delete; no repo residue |
| Coverage reporting (codecov) | External SaaS (optional) | — | Reporting-only at most; native gate does enforcement |

## Standard Stack

### Core
| Tool | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| coverage (via pytest-cov) | 7.16.2 installed; floor `coverage[toml]>=7.10.6` | `fail_under` enforcement from `[tool.coverage.report]` | Already the suite's measurement stack (Phase 1 Route A decision: config in pyproject) — verified exit-code behavior this session |
| pytest-cov | 7.1.0 installed; floor `>=7.0` | `--cov` activation; turns coverage failure into pytest rc=1 | Existing; the census command of record already ends in `--cov` |
| pytest-timeout | 2.4.0 installed; range `>=2.3.1,<2.5` | Per-test timeout marks on slow tests | Already a dependency with `--timeout=300` in addopts; marker precedence over the global value verified live |
| actions/cache | v4 (already used in ci.yml) | HF + ModelScope cache keyed on `models.lock` | Already in the workflow for `~/.cache/uv`; same action, new key/paths |
| actions/setup-python | v7 (already used) | py3.12 for the gate leg | Existing pattern |

### Supporting
| Tool | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| codecov/codecov-action | v7 (only if owner keeps reporting) | Coverage dashboard/PR annotations | Only on the keep-codecov branch of the GATE-03 decision; requires token secret or OIDC permission |
| gh CLI | installed, authed as forrestzhang (scopes: repo, workflow) | GATE-04 proof: PR create, run watch, log grep, branch delete | Execution-time verification tooling; `workflow` scope required to push ci.yml changes — present |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `fail_under` in pyproject | `--cov-fail-under=90` CLI flag in CI only | Violates "identical command locally and in CI" success criterion; flag exists (verified in `pytest --help`) but config is the locked Route A |
| actions/cache on hub dirs | Pre-baked Docker image / self-hosted runner | Overkill for ~5G of small models; keeps PR-loop iteration cheap |
| Per-model cache paths | Whole-hub-dir caching | Whole-dir is robust to the symlink blob structure (partial caching breaks integrity); 5G restore is ~1 min |
| Job in existing ci.yml | New `coverage-gate.yml` workflow | New file must duplicate triggers/permissions; existing file's `on:` already covers dev+main PRs — adding a job inherits everything |

**Installation:** No packages installed. This phase adds zero Python dependencies (constraint: no new test frameworks). All GitHub Actions referenced already exist in the workflow except an optional codecov bump.

**Version verification:** No registry lookups needed (no new packages). Installed versions probed from `.venv`: pytest-cov 7.1.0, coverage 7.16.2, pytest-timeout 2.4.0, pytest 9.1.1.

## Package Legitimacy Audit

**No new packages installed this phase.** The only third-party action changes possible: (a) `actions/cache@v4` and `actions/setup-python@v7` — already in use in this workflow [VERIFIED: .github/workflows/ci.yml:40,48]; (b) `codecov/codecov-action` v3→v7 — an existing dependency being version-bumped or removed, not added; if kept, pin by commit SHA (currently pinned only by major tag — supply-chain note in Security Domain).

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none — no new packages) | — | — | — | — | — | Approved (N/A) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                    push/PR (main, master, dev)          workflow_dispatch (calibration)
                          │                                     │
                          ▼                                     ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │ ci.yml                                                            │
        │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐  │
        │  │ test (6x)  │  │ test-cuda  │  │ test-mamba │  │ coverage-   │  │
        │  │ py3.11-13  │  │ (2x)       │  │ (no-op)    │  │ gate (NEW)  │  │
        │  │ numpy both │  └────────────┘  └────────────┘  │ py3.12      │  │
        │  └────────────┘                                  └─────┬───────┘  │
        └────────────────────────────────────────────────────────┼──────────┘
                                                                   │
             ┌─────────────────────────────────────────────────────┼──────────────────────┐
             ▼                       ▼                             ▼                      ▼
   checkout + free-disk     actions/cache@v4 (x2)        uv venv + .[base]        pytest (census cmd)
   (14G runner disk)        key: os-uv-hash(pyproject)   + numpy==2.2.0           -ra --durations=0
                            key: os-models-hash(models.lock)                       --junitxml --cov
                            path: ~/.cache/uv            (same shape as fast leg)  (both roots, slow incl.)
                                  ~/.cache/huggingface/hub                             │
                                  ~/.cache/modelscope/hub                               ├─ rc!=0 & "Coverage failure:
                                                                                         │   total of X is less than
                                                                                         │   fail-under=90" → job RED
                                                                                         ▼
                                                                       audit_skips.py on junit
                                                                       (exit 1 on unmatched skip)
                                                                                         │
                                                                                         ▼
                                                                       pytest rc → step rc → job
                                                                       conclusion → PR check
                                                                       (GATE-04 asserts this RED)

  GATE-04 proof flow (ephemeral):
   branch gate-regression-probe (git rm tests/models/test_model.py)
     → push → gh pr create --base dev
     → coverage ~81% < 90 → pytest rc=1 → job fails
     → gh run view --log-failed | grep "fail-under"  (evidence capture)
     → gh pr close --delete-branch                    (zero residue on dev)
```

### Recommended Project Structure
```
.github/workflows/ci.yml        # + coverage-gate job, + workflow_dispatch trigger
models.lock                     # NEW: 9-entry manifest; sole cache-key input
pyproject.toml                  # [tool.coverage.report] fail_under = 90 (+ comment update)
tests/finetune/test_trainer_real_model.py   # + @pytest.mark.timeout marks on heavy slow tests
tests/inference/test_inference_real_model.py
tests/inference/test_inference.py
tests/models/test_model.py                 # timeout marks; also the GATE-04 deletion target (probe branch only)
.github/workflows/README.md     # WR-05 disposition (owner decision: fix or leave with rationale)
```

### Pattern 1: The gate is config, not script
**What:** Enforcement = `fail_under` in `[tool.coverage.report]` + the pytest exit code. No custom "compare percentage" step, no `if [ $(...) -lt 90 ]` bash.
**When to use:** Always, for this milestone (Phase-1 Route A decision).
**Example (probed verbatim this session):**
```
# /tmp/fu-probe, repo venv, pyproject [tool.coverage.report] fail_under = 90
$ python -m pytest --cov -q          # 80% coverage, all tests pass
PYTEST_RC=1
ERROR: Coverage failure: total of 80 is less than fail-under=90
1 passed in 0.01s

$ python -m pytest --cov -q          # exactly 90.00%
PYTEST_RC=0
Required test coverage of 90.0% reached. Total coverage: 90.00%
```
Source: live probe, 2026-09-30, `.venv` (pytest-cov 7.1.0 / coverage 7.16.2). Semantics per official docs: "If the total coverage measurement is under this value, then exit with a status code of 2" (bare coverage CLI; pytest-cov surfaces it as rc=1); `[report] precision` (default 0) "affects the interpretation of the fail_under setting" [CITED: coverage.readthedocs.io/en/latest/config.html].

### Pattern 2: models.lock as the cache key
**What:** A root-level manifest listing every remote artifact the slow suite downloads; `hashFiles('models.lock')` keys the cache. Editing the manifest (adding/removing/changing a model) rotates the key; restore-keys bridge to the previous warm set.
**When to use:** GATE-02. The repo has no dependency lockfile, so this manifest is the single source of truth for "what should be in the cache".
**Example (content verified against test sources + local cache sizes):**
```
# models.lock — remote artifacts fetched by the slow test suite.
# Keys the gated CI job's model cache (actions/cache hashFiles).
# Edit an entry to rotate the cache key.
hf  microsoft/DialoGPT-small                         # tests/models/test_model.py::test_download_real_huggingface_connection
hf  zhangtaolab/plant-dnagpt-BPE-promoter            # tests/inference/test_inference_real_model.py (from_pretrained, HF route)
ms  zhangtaolab/plant-dnabert-BPE                    # tests/finetune/test_trainer_real_model.py (12 call sites, source=modelscope)
ms  zhangtaolab/plant-dnabert-BPE-promoter           # dnallm/mcp/tests/configs/promoter_inference_config.yaml
ms  zhangtaolab/plant-dnabert-BPE-conservation       # dnallm/mcp/tests/configs/conservation_inference_config.yaml
ms  zhangtaolab/plant-dnamamba-BPE-open_chromatin    # dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
ms  zhangtaolab/plant-dnagpt-BPE-promoter            # tests/inference/test_inference.py::test_real_model_integration (modelscope route)
ms  ZhejiangLab-LifeScience/DNA_bert_4               # tests/models/test_model.py::test_download_real_modelscope_connection
ms  dataset:zhangtaolab/plant-multi-species-core-promoters  # DNADataset.from_modelscope (trainer tests)
```
[VERIFIED: test-source grep this session + du of local caches; exact paths/lines in the table below. NOTE: `plant-dnamamba-BPE-H3K27ac`/`H3K27me3` configs exist but are referenced by no test — excluded from the manifest.]

Measured cache payload:

| Entry | Source | Local size | Evidence |
|---|---|---|---|
| DialoGPT-small + plant-dnagpt HF copies | HF | **2.0G total** (cold scratch growth, Phase-1 audit; both models' local HF cache dirs are metadata-only ~1-2M, so the full weights only exist in the deleted scratch) | [VERIFIED: 01-AUDIT-REPORT.md leg 3] |
| plant-dnabert-BPE | ModelScope | 353M | [VERIFIED: du ~/.cache/modelscope] |
| plant-dnabert-BPE-promoter | ModelScope | 353M | [VERIFIED: du] |
| plant-dnabert-BPE-conservation | ModelScope | 353M | [VERIFIED: du] |
| plant-dnamamba-BPE-open_chromatin | ModelScope | 739M | [VERIFIED: du] |
| plant-dnagpt-BPE-promoter | ModelScope | 352M | [VERIFIED: du] |
| DNA_bert_4 | ModelScope | 332M | [VERIFIED: du] |
| dataset plant-multi-species-core-promoters | ModelScope | 487M | [VERIFIED: du ~/.cache/modelscope/hub/datasets] |
| **Total** | | **~5.0G** (2.0G HF + ~2.97G MS) | |

Cache step shape:
```yaml
- name: Restore model caches (keyed on models.lock)
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/huggingface/hub
      ~/.cache/modelscope/hub
    key: ${{ runner.os }}-models-${{ hashFiles('models.lock') }}
    restore-keys: |
      ${{ runner.os }}-models-
```
Scope rules that make this correct [CITED: docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching]: a run restores caches from its current branch, the default branch, and — for `pull_request` events — the base branch (so a PR to dev restores dev-seeded caches); caches created on a PR are tied to the merge ref; 10 GB per-repo cap with LRU eviction on overflow and 7-day unused eviction; the save post-step runs only on success (`post-if: "success()"` in actions/cache@v4 action.yml [VERIFIED: raw.githubusercontent.com/actions/cache/v4/action.yml]) — a red GATE-04 probe run cannot poison the cache. Cache whole hub dirs, never fragments (symlinked blob structure).

### Pattern 3: per-test timeout marks that beat the global 300s
**What:** `@pytest.mark.timeout(N)` on the heavy slow tests; the marker overrides the `--timeout=300` from addopts.
**When to use:** Every test expected to exceed 300s on a 4-core CPU runner (all five long trainer tests at minimum).
**Example (precedence verified two ways this session):**
```python
@pytest.mark.slow
@pytest.mark.timeout(3600)   # marker wins over --timeout=300 (verified: sleep(4) test
def test_complete_training_workflow(self):   # killed at 2.02s under mark.timeout(2) + --timeout=300)
```
[VERIFIED: behavioral probe (rc=1, `Failed: Timeout (>2.0s) from pytest-timeout.` at 2.02 s) + plugin source pytest_timeout.py:389-406 — marker settings are consumed first, config value only when no marker. Marker is plugin-registered → safe under `--strict-markers` (shown in `pytest --markers`). No existing test uses `mark.timeout` today — a clean addition.]

### Pattern 4: GATE-04 synthetic regression (safe, local-first, ephemeral)
**What:** Prove the Phase-1 exit-code fix end-to-end: a real PR whose coverage drops below 90 must produce a red check, with the coverage-failure string in the job log.
**Recipe:**
1. **Predict locally first:** current totals 7131/7405 = 96.30%; the gate needs covered < 0.90 × 7405 = 6664.5, i.e. remove ≥467 covered statements. `tests/models/test_model.py` is 2198 lines of the models wave (models area missing went 1209→88 in Phase 3) — deleting it removes far more than 467 → predicted ~81% (arithmetically safe even if only half its coverage vanishes). Optional cheap rehearsal: run the census command with `--ignore=tests/models/test_model.py` and read `coverage json` totals before pushing.
2. `git checkout -b gate-regression-probe dev && git rm tests/models/test_model.py && git commit -m "gate probe: synthetic coverage regression (temporary)"`
3. `git push -u origin gate-regression-probe && gh pr create --base dev --title "GATE-04 probe: must fail CI"`
4. `gh pr checks <n> --watch` (or `gh run watch`) → assert the coverage-gate job conclusion == failure
5. Evidence: `gh run view --job <id> --log-failed | grep "fail-under"` → expect `Coverage failure: total of ... is less than fail-under=90`
6. Cleanup: `gh pr close <n> --delete-branch` — dev history untouched (the PR never merges)
**When to use:** once, after the gate job is green on dev; the same PR run doubles as the GATE-05 proof (a PR to dev triggered it).
[VERIFIED: gh authed with repo+workflow scopes this session; `pull_request: branches: [ main, master, dev ]` already at ci.yml:9-10; coverage arithmetic from 03-VERIFICATION.md totals.]

### Anti-Patterns to Avoid
- **A custom "check coverage" bash step** comparing percentages — re-implements `fail_under` worse (rounding, exit codes, local/CI divergence). Use the config knob.
- **Caching `~/.cache/huggingface` wholesale or partial model dirs** — hub cache is a symlinked blob tree; cache the `hub` subdir whole. Same for `~/.cache/modelscope/hub`.
- **Keying the model cache on `pyproject.toml` or test files** — models change independently of deps; a test-file edit would needlessly rotate a 5G cache. models.lock is the single rotation lever.
- **Raising the global `--timeout` in addopts for CI's sake** — weakens the hang-guard for 1600+ fast tests; use per-test marks instead (verified precedence).
- **Relying on HF_HOME to make ModelScope tests warm** — Phase-1 audit boundary: `HF_HOME` isolates only the HF cache; the dominant slow tests are ModelScope-sourced.
- **Duplicating the canary/audit steps without purpose** — the exit-code canary already lives in the fast leg (every run); the gated job needs the skip audit (it produces the junit) but not a second canary.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|------|
| Coverage threshold enforcement | Bash comparing coverage output | `[tool.coverage.report] fail_under` | Exit-code plumbing verified; identical local/CI; rounding semantics owned by coverage |
| Per-test timeout overrides | Timer threads / signal code in tests | `@pytest.mark.timeout(N)` (pytest-timeout already installed) | Marker precedence verified; `--strict-markers`-safe |
| Cross-run artifact caching | tarballs in artifacts / Git LFS | actions/cache@v4 with hashFiles key | Branch scoping, LRU eviction, save-on-success all handled |
| PR check observation | Scraping the Actions web UI | `gh pr checks` / `gh run watch --exit-status` | gh already authed with the right scopes |
| Cache key rotation | Manual key bumps in ci.yml | Edit models.lock; `hashFiles` derives the key | Single source of truth; self-documenting |

**Key insight:** every mechanism this phase needs (fail_under exit codes, timeout markers, cache scoping, PR checks) already exists and is in-repo or one config line away — the phase's real work is calibration and proof, not construction.

## Runtime State Inventory

> Not a rename/refactor/migration phase (greenfield CI job + config line) — section omitted per protocol. CI-side runtime state relevant to the plan is captured here instead: **no GitHub branch protection exists on `main` or `dev` today** [VERIFIED: `gh api repos/zhangtaolab/DNALLM/branches/{main,dev}/protection` → 404 "Branch not protected"], and the repo has zero pre-existing actions/cache entries for model caches (only `os-uv-*` and mkdocs keys).

## Common Pitfalls

### Pitfall 1: The 300s global timeout will kill the gated job's slow tests
**What goes wrong:** `addopts` carries `--timeout=300` (pyproject.toml:475). On CPU-only hardware the probe measured a **13.0x GPU→CPU factor**: `test_qlora_training` 26s→~4.5 min and `test_complete_training_workflow` 194s→**~43 min (2590s)** on a 10-core CPU. On the 4-core runner (≥2.5x further), the three ~3-min-GPU trainer tests project to **~1.5-2 h each** — every one of them trips 300s.
**Why it happens:** the 300s default was calibrated on the GPU/warm local environment (longest census test: 194.2s, audit report); CI has no GPU and 4 cores.
**How to avoid:** `@pytest.mark.timeout(N)` marks on all slow tests that can exceed 300s (precedence verified). Initial N: `timeout(7200)` for the three ~3-min-GPU trainer tests (CI projection ~1.5-2 h each), `timeout(3600)` for early-stopping/no-early-stopping/qlora/prediction-class tests, unmarked for the <15s stratum — then recalibrate from the first CI run's `--durations=0` output.
**Warning signs:** gated job log shows `Failed: Timeout (>300.0s) from pytest-timeout.` on otherwise-healthy tests.

### Pitfall 2: The runner may OOM on the heaviest slow test
**What goes wrong:** local CPU probe of `test_complete_training_workflow` ran at **~15.9 GB RSS** — the ubuntu-latest standard runner has 16 GB RAM total (4 vCPU/16 GB/14 GB SSD) [CITED: docs.github.com runners reference].
**Why it happens:** full-workflow test (train + predict + infer) over the sampled ModelScope dataset; torch allocator + arrow memory-maps.
**How to avoid:** nothing in-repo cheaply reduces RSS (dataset sampling is test logic; out of bug-fix scope). Make the first gated run a calibration (`workflow_dispatch`); if the job dies with exit 137/killed, surface to the owner: options are a larger runner (paid), marking that one test with a memory note, or LANE-01 v2 pull-forward. Do NOT silently delete the test.
**Warning signs:** job log ends abruptly with "The job running on runner ... has been terminated" / exit code 137.

### Pitfall 3: Runtime expectation mismatch with the owner's accepted-cost decision
**What goes wrong:** "runtime cost accepted by owner" was recorded when the mental model was a ~15-20 min suite (local GPU census 881-934s). Measured: 13.0x GPU→CPU on 10 local cores (2860.29s for the two heaviest tests, both green); the 819.6s warm GPU slow leg projects to ~2.96 h local-CPU and **~4-7.5 h on the 4-core runner** — beyond GitHub's 360-minute default job timeout, so an unmarked job would be killed mid-run even if every test passes.
**Why it happens:** GPU→CPU factor on small-batch BERT training + core-count reduction.
**How to avoid:** set explicit `timeout-minutes: 480` initially (survives the worst projection), encode a `workflow_dispatch` calibration run plus a numeric owner-facing report (job wall time, per-test durations from `--durations=0`), then tighten. Present the option menu: accept hours-long PR gates / larger runner (8-core, paid) / pull LANE-01 (two-lane split + nightly drift) forward from v2 — an owner decision, not an executor improvisation.
**Warning signs:** timeouts tripping at `timeout-minutes`; PR feedback cycle measured in hours.

### Pitfall 4: fail_under changes EVERY local `--cov` invocation too
**What goes wrong:** a developer runs `pytest tests/utils/test_sequence.py --cov` — scoped coverage is far below 90 → rc=1 despite green tests.
**Why it happens:** Route A put enforcement in pyproject, deliberately shared local/CI.
**How to avoid:** scoped runs simply drop `--cov` (or pass `--no-cov`, verified present in pytest-cov's flags). Document in the workflows README / CONTRIBUTING touch-point of this phase.
**Warning signs:** confused "tests pass but exit 1" reports.

### Pitfall 5: Caching HF but forgetting ModelScope (or vice versa)
**What goes wrong:** only `~/.cache/huggingface` cached → every gated run re-downloads ~3G from modelscope.cn (cross-Pacific from US runners; unmeasured and potentially slow/flaky); only ModelScope cached → +85.5s of HF downloads every run.
**Why it happens:** Phase-1 audit boundary — HF_HOME isolates only the HF cache; the trainer tests (the bulk of slow wall time) are ModelScope-sourced.
**How to avoid:** cache BOTH hub dirs under one key (models.lock covers both sources).
**Warning signs:** gated job stuck minutes in `snapshot_download` progress bars.

### Pitfall 6: Disk pressure on the 14 GB runner
**What goes wrong:** restored caches (~5G) + venv with CUDA-bundled torch from PyPI (`.[base]` resolves torch without a cpu extra → nvidia wheels, several GB) + uv cache (~2-3G) exceed 14 GB.
**How to avoid:** keep the `Free disk space` step (existing legs reclaim ~20G by removing android/dotnet/ghc/chromium) in the gated job; reuse the shared `os-uv-*` cache key so install stays warm.
**Warning signs:** "No space left on device" mid-install or mid-test.

### Pitfall 7: junit filename collisions and skip-audit surprises in the full run
**What goes wrong:** the gated job reuses `pytest-junit.xml` (fine — separate job), but the full-suite junit contains 7 skips (1 content + 6 `network-unavailable:` MCP live-server probes); any NEW skip reason fails `audit_skips.py` (exit 1, fail-closed).
**How to avoid:** run the same audit step on the gated junit; the current allowlist already anticipates the Phase-4 slow leg (`tests/expected_skips.yaml` comment: "serves local/full runs and the Phase-4 slow leg").
**Warning signs:** audit step red with "unmatched skip" listing a new reason.

### Pitfall 8: codecov v7 needs credentials v3 didn't
**What goes wrong:** mechanical bump `@v3`→`@v7` starts failing: v4+ tokenless upload is restricted; v7 needs `token: ${{ secrets.CODECOV_TOKEN }}` or `use_oidc: true` + `permissions: id-token: write` (the workflow is currently `contents: read` only).
**Why it happens:** v3's legacy uploader path still works tokenless for public repos (the current step SUCCEEDS — verified in run 36617996840), but v7 wraps the Codecov CLI whose features v3 lacks [CITED: github.com/codecov/codecov-action README].
**How to avoid:** decide GATE-03 explicitly — removal (recommended; native gate replaces it) or v7-with-token (owner creates the secret / grants OIDC permission). Never keep `fail_ci_if_error` unset-and-hoped-for; if kept, pin `fail_ci_if_error: false` explicitly (reporting-only).

### Pitfall 9: "Protected branches" don't exist — a red check does not block merges today
**What goes wrong:** GATE-05 is satisfied at the workflow level (job runs on PRs to dev+main), but with no branch protection, a red gate check can be merged through anyway.
**How to avoid:** surface it: making the check required is a repo-settings/API action outside workflow YAML (owner has admin via the same gh token). Recommended as the phase's documented owner follow-up, not an executor action. [VERIFIED: both branches unprotected via API this session]

### Pitfall 10: coverage drift across environments
**What goes wrong:** the 96.30% figure was measured on py3.13.15/torch 2.11.0+cu130/transformers 5.17.0; the gate leg runs py3.12 with fresh-resolved unpinned deps (`transformers>=4.49.0,<6`) — version-guarded lines (transformers_compat shims, platform guards) can shift totals by tenths of a point.
**How to avoid:** 6.3 points of headroom over 90 absorbs it by design ("90, not 96" locked decision). If a future upstream release drops the suite below 90, the gate going red is the mechanism working — fix forward.
**Warning signs:** gated totals differing from local by >0.5 point on identical commits.

## Code Examples

### GATE-01: the one-line config change (+ comment update)
```toml
# pyproject.toml (current state at :512-514)
[tool.coverage.report]
show_missing = true
# NO fail_under in this phase — the enforcement threshold is added in Phase 4, ratcheted
```
becomes:
```toml
[tool.coverage.report]
show_missing = true
fail_under = 90   # Phase 4 ratchet (GATE-01). Suite landed at 96.30% (Phase 3).
                  # Applies to every `--cov` invocation — use `--no-cov` for scoped runs.
```

### GATE-02: coverage-gate job skeleton (shape, not verbatim YAML)
```yaml
# ci.yml — add to `on:`:
#   workflow_dispatch:        # enables the calibration run without a PR

  coverage-gate:
    name: coverage-gate (py3.12, full suite incl. slow)
    runs-on: ubuntu-latest
    timeout-minutes: 480          # backstop — projection 4-7.5h on 4-core; tighten after calibration
    steps:
      - uses: actions/checkout@v4
      - name: Free disk space     # same step as existing legs (Pitfall 6)
        run: |
          sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc \
            /usr/local/share/powershell /usr/local/share/chromium /usr/local/.ghcup
      - uses: actions/setup-python@v7
        with: { python-version: '3.12' }
      - run: curl -LsSf https://astral.sh/uv/install.sh | sh      # WR-06 disposition pending
      - uses: actions/cache@v4                                     # shared with fast legs
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-uv-${{ hashFiles('**/pyproject.toml') }}
          restore-keys: ${{ runner.os }}-uv-
      - uses: actions/cache@v4                                     # NEW — Pattern 2
        with:
          path: |
            ~/.cache/huggingface/hub
            ~/.cache/modelscope/hub
          key: ${{ runner.os }}-models-${{ hashFiles('models.lock') }}
          restore-keys: ${{ runner.os }}-models-
      - name: Install (same shape as fast leg)
        env: { UV_HTTP_TIMEOUT: 300, UV_CONCURRENT_DOWNLOADS: 4 }
        run: |
          uv venv
          uv pip install -e ".[base]"
          uv pip install "numpy==2.2.0"          # single leg; 1.26.4 costs +2min (scipy rebuild)
      - name: Gated full-suite census (command of record; slow included)
        run: |
          source .venv/bin/activate
          .venv/bin/python -m pytest -ra --durations=0 \
            --junitxml=pytest-junit-gate.xml --cov \
            -p no:cacheprovider -p no:progress
          # rc=1 + "Coverage failure: total of X is less than fail-under=90" when below gate
      - name: Skip audit (gated junit)
        run: |
          source .venv/bin/activate
          python scripts/audit_skips.py pytest-junit-gate.xml tests/expected_skips.yaml
```
Notes: `.venv/bin/python -m pytest` vs bare `pytest` — CI legs `source .venv/bin/activate` first; either resolves the same pyproject config (HARN-01 probe matrix). `-p no:progress` matches the census command of record (pytest-progress is in the `test` extra, installed in CI via `base`).

### Timeout marks (GATE-02)
```python
# tests/finetune/test_trainer_real_model.py — initial values, recalibrate from first CI run
# CPU probe (this session): 13.0x GPU→CPU on 10 cores; CI 4-core adds ~2.5x more.
@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_complete_training_workflow(self): ...     # 194s GPU → 2590s local CPU (measured); CI est ~1.5-2h

@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_with_config_file(self): ...               # 190s GPU; CI est ~1.5-2h

@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_training(self): ...                       # 182s GPU; CI est ~1.5-2h

@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_early_stopping_stops_before_full_epochs(self): ...   # 95s GPU; CI est ~50min

@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_no_early_stopping_runs_full_epochs(self): ...        # 51s GPU; CI est ~30min

@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_qlora_training(self): ...                 # 26s GPU → ~4.5min local CPU (measured); CI est ~12min
```
(GPU numbers: Phase-1 cold/warm audit table; local CPU probe this session. Unmarked slow tests all ran <15s locally and need no mark unless CI calibration says otherwise.)

### GATE-04: evidence-capture command sequence
```bash
gh pr create --base dev --head gate-regression-probe --title "GATE-04 probe: synthetic coverage regression"
gh pr checks --watch                # or: gh run watch $(gh run list --branch gate-regression-probe --json databaseId -q '.[0].databaseId')
gh run view --job $(gh run view <run-id> --json jobs -q '.jobs[] | select(.name|startswith("coverage-gate")) | .databaseId') \
  --log-failed | grep -E "fail-under|Coverage failure"    # the Phase-1 exit-code fix, observed end to end
gh pr close --delete-branch
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Exit-code masked by `os._exit(0)` in conftest (pre-Phase-1) | `pytest_sessionfinish` propagates exitstatus (conftest.py:27-33) + CI canary | Phase 1 (HARN-02) | pytest rc reaches the Actions step — fail_under enforcement is now possible at all |
| `codecov-action@v3` bash uploader | v7 wraps the Codecov CLI; token/OIDC required | v4 (2024) onward; v7 is current | GATE-03: bump costs a secret/permission; removal is free |
| Coverage measurement only (no floor) | `fail_under` ratchet at 90 | This phase | The milestone's enforcement teeth |
| Custom CI gate scripts | Config-native `fail_under` + junit audits | Coverage has long supported it; adopted here | Less code, identical local/CI semantics |

**Deprecated/outdated:**
- `codecov-action@v3`: still functional today (step succeeds in run 36617996840) but "v3 versions and below will not have access to CLI features (e.g. global upload token, ATS)" [CITED: codecov-action README].
- `.github/workflows/README.md`: documents a matrix "3.10, 3.11, 3.12", Black/isort/Flake8 quality steps, a "develop" branch, and "Slow tests are excluded from CI" — all wrong today and the last claim flips this phase (WR-05).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | CI 4-core CPU runtime ≈ ~2.5x the measured local 10-core CPU times (core-count scaling; local factor 13.0x vs GPU is now measured, the runner scaling is the assumption) | Pitfalls 1/3, timeout marks | Marks/timeout-minutes initially too small (or too large) — mitigated by the workflow_dispatch calibration run; marks are cheap to adjust |
| A2 | HF cold downloads (~2.0G) take single-digit minutes on GH runners (good bandwidth to huggingface.co); ModelScope (~3.0G) download speed from US runners is unmeasured | Pattern 2, Pitfall 5 | First (cold) gated run slower than expected; cache makes subsequent runs warm — calibration absorbs it |
| A3 | `timeout(3600/7200)` marks are large enough for CI (extrapolated, not CI-measured) | Code Examples | A healthy test trips its mark → job red with a Timeout failure distinct from a coverage failure; recalibrate |
| A4 | Fresh CI resolve (unpinned transformers/torch) lands on versions behaving like local 5.17.0/2.11.0 for coverage purposes | Pitfall 10 | Small totals drift; 6.3-point headroom absorbs; if below 90 the gate is correctly red |
| A5 | Deleting `tests/models/test_model.py` drops coverage below 90 (predicted ~81%; arithmetic says needs only ≥467 of its covered statements removed) | Pattern 4 | Probe PR unexpectedly green — rehearsal step (local `--ignore` run or coverage json check) catches it before pushing |
| A6 | `--no-cov` guidance suffices for scoped dev runs under the new fail_under | Pitfall 4 | Developer annoyance only |
| A7 | Job name "coverage-gate" as the future required-check name | GATE-05 note | Cosmetic; owner picks the check name when configuring branch protection |

## Open Questions

1. **GATE-03 codecov: drop or bump to v7?**
   - What we know: v3 works today (verified green in run 36617996840); v7 needs `token` secret or `use_oidc` + `id-token: write`; enforcement no longer needs codecov at all.
   - What's unclear: does the owner value the codecov.io dashboard/PR annotations?
   - Recommendation: **remove** (zero maintenance, no secrets, no permission widening; coverage trends are v2 PATCH-02 territory). If the owner wants it: v7 + `fail_ci_if_error: false` + token secret, pinned by SHA.
2. **Owner-facing runtime budget**
   - What we know: measured GPU→CPU factors (Pitfalls 1-3) point to a multi-hour gated job.
   - What's unclear: the owner's actual tolerance now that numbers exist (the accept decision predates them).
   - Recommendation: plan includes a calibration run + numeric report; owner decides whether to accept, tighten via larger runner, or pull LANE-01 forward.
3. **Phase-1 deferred dispositions (WR-02/WR-05/WR-06) — encode leave-untouched vs minimal-touch**
   - WR-02 (test-mamba no-op GPU leg): leave untouched recommended — `deploy.needs` already tolerates it; touching it is behavior change beyond gate scope. Rationale: no-op cost is ~1 min of runner time.
   - WR-05 (workflows README staleness): minimal-touch recommended — this phase necessarily edits claims the README makes ("slow excluded from CI", job list); a full rewrite is out of scope but the gate-related lines should not be left lying. Owner call on depth.
   - WR-06 (unpinned `curl | sh` uv installer): leave untouched recommended — pinning (versioned installer URL or astral-sh/setup-uv) is supply-chain hygiene orthogonal to the gate; if the owner wants it, `astral-sh/setup-uv` with a pinned version is the one-line form (would also touch 4 existing jobs — scope creep).
4. **Branch protection / required checks (GATE-05 "protected")**
   - What we know: neither branch is protected; workflow triggers already cover dev+main PRs.
   - What's unclear: whether the owner wants enforcement to block merges (requires protection rules) or advisory red checks suffice this milestone.
   - Recommendation: ship the YAML; document the exact `gh api` payload to make `coverage-gate` a required check as the owner follow-up.
5. **Should `deploy` wait on the gate job?**
   - What we know: `deploy.needs: [test, test-cuda, test-mamba]`; deploy runs only on main pushes.
   - Recommendation: leave unchanged (minimal-touch) — the gate blocks at PR time; adding it to `needs` would delay docs deploys by the (multi-hour) gate runtime. Owner can flip later.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| gh CLI | GATE-04 proof, run inspection | ✓ | authed as forrestzhang, scopes repo+gist+workflow | — |
| GitHub Actions (repo zhangtaolab/DNALLM) | gate job, GATE-04 PR | ✓ | CI workflow live, runs green on dev pushes | — |
| Network to huggingface.co | slow tests | ✓ (local, direct — audit "route of record") | — | HF_ENDPOINT mirror exists in code but tests use direct route |
| Network to modelscope.cn | slow trainer tests | ✓ locally (107G warm cache); CI speed unmeasured | — | actions/cache makes it a first-run-only cost |
| actions/cache capacity | model caches | ✓ | ~5G models + ~2-3G uv vs 10G repo cap | LRU eviction; whole-hub dirs cached |
| Local `.venv` for rehearsal probes | fail_under/GATE-04 prediction | ✓ | pytest-cov 7.1.0, coverage 7.16.2, pytest-timeout 2.4.0, numpy 2.5.3, py3.13.15 | — |
| GPU (local) | none required by phase (CPU probes used CUDA_VISIBLE_DEVICES="") | ✓ present, deliberately hidden for CI emulation | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none (ModelScope CI bandwidth unknown — cache is the mitigation).

## Validation Architecture

Skipped — `workflow.nyquist_validation: false` in `.planning/config.json` [VERIFIED: config read this session]. (The phase's own verification IS behavioral: GATE-04 is the canary-of-canaries.)

## Security Domain

ASVS level 1 (`security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high` — `.planning/config.json`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surfaces in scope (CI-internal) |
| V3 Session Management | no | — |
| V4 Access Control | yes (CI permissions) | Workflow-level `permissions: contents: read` (existing, Phase-1 WR-01 fix); the new job inherits it; do NOT widen except a deliberate `id-token: write` if the codecov-v7 OIDC path is chosen |
| V5 Input Validation | yes (artifact parsing) | `scripts/audit_skips.py` parses CI junit fail-closed (existing, T-02-04 disposition); models.lock is a new trusted-input file consumed only via `hashFiles` (never executed/parsed by tests) |
| V6 Cryptography | no | — |
| V14 Config/Logging | yes (CI supply chain) | See threat table |

### Known Threat Patterns for CI-gate work

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Third-party action supply chain (codecov-action, unpinned major tag) | Tampering/Elevation | Prefer removal; if kept: v7 pinned by commit SHA (current `@v3` tag-pinning is the weakest form) |
| Cache poisoning of model artifacts | Tampering | Cache key = `hashFiles('models.lock')` (content-addressed); saves only on green jobs (`post-if: success()`); caches restore only within branch scope — a red GATE-04 probe cannot write cache |
| `curl \| sh` installer executed in CI (WR-06) | Tampering/Elevation | Open owner disposition this phase (leave-untouched recommended); the pinned-installer/`astral-sh/setup-uv` form is documented in Open Questions |
| Secrets exposure (CODECOV_TOKEN if v7) | Information Disclosure | Token referenced only via `${{ secrets.* }}` in the step env; never echoed |
| Hung-job resource burn (DoS on runner minutes) | DoS | Per-test timeout marks + job `timeout-minutes` backstop (this phase's explicit deliverable) |
| Untrusted junit/yaml inputs to audit tooling | Tampering | Existing fail-closed parsing (audit_skips.py raises on absent/unparseable/malformed allowlist) |

## Sources

### Primary (HIGH confidence)
- Live probes this session (repo `.venv`): fail_under exit codes (rc=1 below / rc=0 at exactly 90.00% / coverage CLI rc=2); fast-leg coverage 96% (275/7405 missing, 1635 passed, 113s); pytest-timeout marker-over-CLI precedence (2.02s kill under `--timeout=300`); completed CPU-only slow-test probe — both green, `2 passed in 2860.29s` (qlora ~4.5min; complete_training ~43min at 1018% CPU / ~15.9GB RSS; 13.0x GPU→CPU factor)
- `.github/workflows/ci.yml` (triggers lines 3-10; codecov step lines 92-96; canary lines 98-114; cache/uv/free-disk patterns)
- `pyproject.toml` (coverage run/report blocks lines 500-514; addopts `--timeout=300` line 475; test extra floors lines 92-99)
- `.planning/phases/01-.../01-AUDIT-REPORT.md` (cold/warm slow timings; 2.0G HF scratch; ModelScope/HF_HOME boundary)
- `.planning/phases/03-.../03-VERIFICATION.md` (final census 96.30%, 7131/7405, 881s, 1656 passed / 7 skipped)
- Test-source grep + `du`: 9-entry model manifest with per-entry sizes
- `gh api` branch-protection checks (both unprotected); `gh run view` 36617996840 step timings; `gh auth status` scopes
- actions/cache@v4 action.yml (`post-if: "success()"`) via raw.githubusercontent.com
- pytest-timeout 2.4.0 installed source (marker precedence, pytest_timeout.py:389-406)

### Secondary (MEDIUM confidence)
- [coverage.readthedocs.io/config.html](https://coverage.readthedocs.io/en/latest/config.html) — fail_under exit-2 semantics; precision interaction
- [docs.github.com dependency-caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching) — cache scope (current/default/base branch), 10GB cap, LRU + 7-day eviction
- [docs.github.com GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) — 4 vCPU / 16 GB / 14 GB SSD
- [docs.github.com workflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions) — `timeout-minutes` default 360
- [codecov-action README](https://github.com/codecov/codecov-action) — v7 current; v3 lacks CLI features; token/`use_oidc`+`id-token: write` requirements; inputs `token`, `fail_ci_if_error`, `files`, `flags`, `verbose`, `version`, `use_oidc`

### Tertiary (LOW confidence)
- None used for any recommendation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — nothing new installed; all mechanisms probed in the repo's own venv
- Architecture: HIGH — job shape mirrors proven fast-leg steps; cache scope rules from GitHub docs
- Pitfalls: HIGH — the three critical ones (timeout kill, OOM adjacency, runtime blowup) are measured, not speculated: completed probe `2 passed in 2860.29s`, 13.0x factor, ~15.9GB RSS; CI-side absolute numbers remain extrapolated (calibration run encoded in recommendations)

**Research date:** 2026-09-30
**Valid until:** 2026-10-30 (stable tooling; re-check codecov v7 and runner specs if execution slips past that)
