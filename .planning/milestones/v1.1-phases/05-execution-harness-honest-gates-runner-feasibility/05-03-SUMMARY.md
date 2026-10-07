---
phase: 05-execution-harness-honest-gates-runner-feasibility
plan: 03
subsystem: testing
tags: [gb10, feasibility-spike, evo, evo2, megadna, pybigwig, marimo, flash-attn, self-hosted-runner]

requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: typed-skip prefixes environment-unavailable:/optional-dep: registered in expected_skips.yaml (05-01)
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: honest docs-validation lane the spike docs ride on (05-02)
provides:
  - Committed spike runner scripts/feasibility/spike_families.py with the D-05 evidence contract and D-06 fallback ladder
  - Written verdict matrix 05-FEASIBILITY.md with per-family evidence logs (5 rows, all verdict vocabulary-clean)
  - Dispatch-only feas-spike runner-confirmation workflow (.github/workflows/feasibility.yml) + README documentation
  - Owner dispatch hand-off (push + gh workflow run + artifact download + matrix finalization)
affects: [08-example-rollout, 06-registry-loci, 09-census-gates]

actuals:
  tokens: 25974
  tasks: 3
  commits: 8

tech-stack:
  added: []  # nothing enters pyproject — the pyBigWig row is environment-unavailable, spike-only packages live in throwaway venvs
  patterns:
    - "Throwaway-/tmp-venv spike isolation: project .venv and pyproject provably untouched while spike-only packages install elsewhere"
    - "D-06 attempt ladders recorded in log headers: every verdict carries notebook-variant-first then smallest-viable evidence, failures verbatim"
    - "Compat-shim recipe collection for Phase 8: np.fromstring shim, noFP8-config forcing, flash-attn sm_120 source build, pinned hash-verified clone"

key-files:
  created:
    - scripts/feasibility/spike_families.py
    - .github/workflows/feasibility.yml
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_evo1.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_evo1_fallback.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_evo2.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_evo2_fallback.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_megadna.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_megadna_fallback.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_marimo.log
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/spike-logs/spike_pybigwig.log
  modified:
    - .github/workflows/README.md

key-decisions:
  - "evo-1 verdict FEASIBLE(small-variant): the 131k notebook variant's remote code cannot construct on any transformers in dnallm's span (pos_idx_in_fp32 removed by 4.49), but evo-1-8k-base's older remote code runs end-to-end — Phase 8 executes the 8k variant and updates the notebook reference per D-06"
  - "evo2 verdict FEASIBLE(notebook-variant) via the noFP8 config: the GB10 FP8 trap fired live (auto-selection demands Transformer Engine; only 7b tiers run without TE); the empty TE meta package must stay absent because its RuntimeError escapes vortex's ImportError guard"
  - "pyBigWig verdict environment-unavailable: the default sdist build fails on the stock box (curl-config present but headers off the include path; setup.py uses --libs never --cflags); a CFLAGS deviation builds and round-trips green but a dev-extra line cannot encode it — pyproject untouched"
  - "marimo flavor decision: export-html (deterministic exit + HTML artifact); script-mode also terminates cleanly and binds no port (A4 resolved)"
  - "PyPI evo-1 is an unrelated SLAM package — the notebook-documented evo-model==0.5 is the correct prerequisite; stripedhyena imports fine without flash-attn but the evo-1 remote code declares flash-attn required"
  - "HF revision correction: the handler fetches 'main' for togethercomputer/evo-1-131k-base (no dot in the id), not the research-expected 1.1_fix; observed snapshot 78c715ab is the natural models.lock pin"

patterns-established:
  - "Evidence-first spike runner: GPU guard exit 2 distinct from family FAIL exit 1; key=value greppable evidence lines; both-attempts ladders in log headers"
  - "Flash-attn GB10 build recipe: FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 (default 4-arch MAX_JOBS=16 gets OOM-killed)"
  - "HF-cache footprint measurement: inode-deduped st_size walk over the models-- root (du is unreliable on this box and snapshot entries are symlinks into blobs/)"

requirements-completed: [FEAS-01]

coverage:
  - id: D1
    description: Committed spike runner with the D-05 evidence contract (per-family functions, --fallback D-06 ladder, fail-closed main, GPU guard exit 2)
    requirement: FEAS-01
    verification:
      - kind: other
        ref: command ".venv/bin/python scripts/feasibility/spike_families.py --help (all six family choices) && ruff check + ruff format --check -> exit 0"
        status: pass
    human_judgment: false
  - id: D2
    description: Written verdict matrix for all five families with measured evidence (load_s/forward_s/peak_vram_gb/disk_gb) or exact failure text, D-05 notebook-variant-first attempts, D-06 fallback ladders, typed-skip prefix assignment, provisional runner-confirmation column
    requirement: FEAS-01
    verification:
      - kind: other
        ref: command "grep -cE '^\\| (evo-1|evo2|megaDNA|pyBigWig|marimo) ' 05-FEASIBILITY.md -> 10 rows; 8 spike logs on disk and committed; three exact variant ids grepped; both-attempts evidence on every failure verdict"
        status: pass
      - kind: other
        ref: command "git status --porcelain -- pyproject.toml empty; .venv import stripedhyena/evo2/pyBigWig/flash_attn/MEGABYTE_pytorch all ModuleNotFoundError"
        status: pass
    human_judgment: false
  - id: D3
    description: Dispatch-only runner-confirmation workflow on the self-hosted GB10 runner with unconditional artifact upload, documented in the workflows README
    requirement: FEAS-01
    verification:
      - kind: other
        ref: command "yaml.safe_load OK; dispatch event gate + runs-on [self-hosted, dnallm-nightly] + if: always() + timeout-minutes 240 grepped; zero push/PR/schedule triggers; README mentions feasibility + manual-dispatch-only"
        status: pass
    human_judgment: false
  - id: D4
    description: Local verdicts become official only after the owner's runner confirmation per D-04 — the dispatch hand-off (push range, gh workflow run, artifact download, matrix finalization) is recorded verbatim below
    verification: []
    human_judgment: true
    rationale: The owner must push (manual-push-only milestone), dispatch the workflow, and fill the Runner confirmation column from the runner artifacts; D-04's official-verdict step is owner-run by design.

duration: 94 min
completed: 2026-10-02
status: complete
plan_head_before: 97a7dcb8f3f74df7bc7ff69f67687c35d8d31ae5
plan_head_after: 7b8bf8ab16be889d1b7624d89cfdc09506c71bd1
---

# Phase 5 Plan 03: GB10 Feasibility Spike Summary

**Written verdict matrix with real-forward evidence for all five environment-gated families — evo-1 feasible on the 8k variant, evo2 feasible via the noFP8 config (FP8 trap confirmed live), megaDNA feasible via a pinned clone, pyBigWig environment-unavailable on the default build, marimo export-html decided — plus a dispatch-only runner-confirmation workflow and the owner hand-off**

## Performance

- **Duration:** 94 min (plus unattended background download/compile waits inside it)
- **Started:** 2026-10-01T18:09:09Z
- **Completed:** 2026-10-01T19:43:30Z
- **Tasks:** 3/3
- **Files modified:** 12 (4 created code/docs + 8 evidence logs + README)

## Accomplishments
- `scripts/feasibility/spike_families.py` — committed spike runner: per-family functions with exact notebook variants first and D-06 fallbacks (evo-1-8k-base, evo2 noFP8 config via a documented `is_fp8_capable` patch, pinned hash-verified megaDNA clone at `cb2f5ab4`), greppable `key=value` evidence lines, GPU guard with exit code 2 distinct from family FAIL
- `05-FEASIBILITY.md` — the FEAS-01 verdict matrix: 5 family rows + typed-skip prefix assignment + marimo flavor decision, every Evidence cell pointing at a committed spike log, box identity line and /tmp-venv isolation recorded, verdict vocabulary exactly FEASIBLE(...)/environment-unavailable:
- Verdict highlights: **evo-1 FEASIBLE(small-variant)** (131k remote code reads `rotary_emb.pos_idx_in_fp32`, absent from every transformers ≥4.49 — 4-attempt ladder incl. transformers 5.18 and 4.57.6; evo-1-8k-base runs a real forward + the notebook's exact `generate` in 53.4s, 13.88GB VRAM); **evo2 FEASIBLE(notebook-variant)** via `evo2-1b-8k-noFP8.yml` (auto-FP8 selection fails upstream-confirmed: "Only 7b models ... can run without Transformer Engine"; notebook generate OK, 2.32GB VRAM); **megaDNA FEASIBLE(notebook-variant)** via pinned clone + MEGABYTE_pytorch==0.2.1 (forward 0.4s, 0.79GB VRAM); **pyBigWig environment-unavailable** on the default build (curl headers off the include path; CFLAGS deviation builds + round-trips green — recorded as the remediation path, pyproject untouched); **marimo FEASIBLE** with the export-html flavor decision
- `.github/workflows/feasibility.yml` — dispatch-only `feas-spike` job on `[self-hosted, dnallm-nightly]` (timeout 240, one `--family all --fallback` pass, unconditional artifact upload), documented in `.github/workflows/README.md` per WR-05 discipline
- Project environment provably untouched: pyproject byte-clean, all five spike-only imports absent from `.venv`, everything installed in `/tmp/feas-venv` only

## Task Commits

Each task was committed atomically (Task 1/2 accumulated four Rule-1 fix commits, listed under Deviations):

1. **Task 1: Per-family spike runner** - `f7b5fa9` (feat)
2. **Task 2: Local spike runs + verdict matrix + evidence logs** - `e74b0ce` + `7b8bf8a` (docs; logs needed `git add -f` — the root `.gitignore`'s `*.log` rule is for runtime artifacts)
3. **Task 3: Dispatch-gated runner confirmation workflow + README** - `4f78a76` (ci)

**Plan metadata:** (final docs commit below)

## Files Created/Modified
- `scripts/feasibility/spike_families.py` - spike runner (GPU guard, evidence contract, D-06 fallbacks, np.fromstring shim)
- `.planning/phases/05-.../05-FEASIBILITY.md` - the verdict matrix (provisional until runner confirmation)
- `.planning/phases/05-.../spike-logs/spike_*.log` - 8 raw evidence logs with attempt ladders
- `.github/workflows/feasibility.yml` - dispatch-only feas-spike job
- `.github/workflows/README.md` - feas-spike section + trigger overview updated

## Decisions Made
- evo-1 recorded FEASIBLE(small-variant) rather than a typed skip: the D-06 ladder found a real execution path (8k variant + flash-attn build + fromstring shim) — the spike's job was to find it, and the notebook's model reference update moves to Phase 8
- The evo2 noFP8-config forcing is recorded as a documented deviation (`is_fp8_capable` patched False for the load); productizing it (GB10 config pin or TE support) is Phase 8 work
- pyBigWig left out of the dev extra per the Task-3 conditional: the default install path is the gate, not the deviation build; owner remediation (install system libcurl headers, then re-run the spike) is recorded in the matrix
- marimo standardizes on export-html for Phase 8; script-mode remains a viable fallback flavor (no port binding, deterministic exit)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pyBigWig `addEntries` explicit-ends form needs chroms as a list**
- **Found during:** Task 2 (first pybigwig spike run)
- **Issue:** The plan's pseudocode `bw.addEntries("Chr1", [0,100], ends=...)` raises RuntimeError — pyBigWig's explicit-ends form requires a per-start chromosome list
- **Fix:** `bw.addEntries(["Chr1", "Chr1"], [0, 100], ends=[50, 200], values=[0.5, 1.0])`; D-05 round-trip intent unchanged
- **Files modified:** scripts/feasibility/spike_families.py
- **Verification:** round-trip green (values[0]==0.5)
- **Committed in:** 53aaaae

**2. [Rule 1 - Bug] transformers BatchEncoding is a UserDict, not a dict instance**
- **Found during:** Task 2 (megaDNA fallback 2b)
- **Issue:** `isinstance(encoded, dict)` is False for BatchEncoding, so the whole encoding object was passed as ids and crashed every family's forward
- **Fix:** probe for the `input_ids` key and coerce with `torch.as_tensor`
- **Files modified:** scripts/feasibility/spike_families.py
- **Verification:** megaDNA/evo2/evo-1 forwards pass
- **Committed in:** 53aaaae

**3. [Rule 1 - Bug] `du` under-reports the HF cache on this box; snapshots are symlinks into blobs/**
- **Found during:** Task 2 (megaDNA disk evidence showed 8.0K for a 582MB checkpoint)
- **Issue:** du measured 76 bytes for a 582MB file; lstat measures symlink lengths; naive sums double-count blob+link
- **Fix:** inode-deduped `os.stat` walk over the `models--<repo>` root
- **Files modified:** scripts/feasibility/spike_families.py
- **Verification:** megaDNA 582.4MB, evo2 2.70GB, evo-1 29.73GB — all match `find -printf %s` totals
- **Committed in:** ffde0c1

**4. [Rule 1 - Bug] evo2 weights are inference-mode tensors; vortex saves activations for backward**
- **Found during:** Task 2 (evo2 fallback 2a — model loaded at 2.39GB then the forward raised "Inference tensors cannot be saved for backward")
- **Fix:** spike forwards run under `torch.no_grad()` (inference semantics)
- **Files modified:** scripts/feasibility/spike_families.py
- **Verification:** evo2 fallback forward + generate green
- **Committed in:** 41b5588

**5. [Rule 1 - Bug] stripedhyena 0.2.2 calls `np.fromstring` (removed in numpy 2)**
- **Found during:** Task 2 (evo-1-8k constructed with 12.98GB VRAM but could not tokenize)
- **Fix:** in-process `np.fromstring` → `np.frombuffer` shim in the spike, mirroring `dnallm/utils/transformers_compat.py`'s patch pattern (downgrading numpy<2 instead broke the venv's numpy-2-built scipy/ml-dtypes stack at `import transformers` — attempt recorded)
- **Files modified:** scripts/feasibility/spike_families.py
- **Verification:** evo-1-8k full pipeline green (forward 1.2s, generate 53.4s)
- **Committed in:** fc0d974

**6. [Rule 3 - Blocking] The plan's `evo-1` PyPI package is an unrelated SLAM package; flash-attn is a hard prerequisite of both evo families**
- **Found during:** Task 2 (prerequisite installation)
- **Issue:** Research A1 assumed `pip install evo-1`; that name hosts an odometry/SLAM package (ToniRV fork). The notebook's own documented line is `pip install evo-model`; additionally the evo-1 remote code declares flash-attn required (trust_remote_code check_imports), and vortex imports `flash_attn_2_cuda` unguarded
- **Fix:** installed `evo-model==0.5` + `stripedhyena --no-deps`; built flash-attn 2.8.3.post1 from source for GB10 (`FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8` — the default 4-arch MAX_JOBS=16 build was OOM-killed, both builds' logs recorded)
- **Files modified:** none (throwaway venv only)
- **Verification:** evo-1-8k and evo2 both execute
- **Committed in:** evidence in spike logs

---

**Total deviations:** 6 auto-fixed (5 Rule 1 bugs, 1 Rule 3 blocker)
**Impact on plan:** All fixes preserve plan intent; none touch the project environment. The evo-model/flash-attn correction and the four spike-side bug fixes were exactly the "spike resolves each empirically" work the plan reserved.

## Issues Encountered
- megaDNA's and evo-1's notebook-path `engine.generate` calls surface dnallm-side tokenizer issues (single-string encode for the megadna DNATokenizer; recorded as generate_note evidence) — forward feasibility is unaffected; both are Phase 8 repair candidates, not Phase 5 scope
- The box's HF cache shows cross-repo blob sharing/hardlink semantics (the 131k repo's blob dir emptied after the 8k repo downloaded identical content) — footprint numbers in the matrix are per-repo st_size walks taken at each run; Phase 8 cache-tier design should account for the dedup
- An `uv run`-era note: `pip install "transformers<5"` and `pip install "numpy<2"` inside the throwaway venv each produced resolver conflict warnings (jaxlib/scipy/ml-dtypes need numpy>=2) — recorded, resolved by restoring numpy 2 and shimming instead

## User Setup Required
None for services. **One owner action is pending — the dispatch hand-off below (blocking human check).**

## D-04 Owner Hand-Off: push, dispatch, and finalize the runner confirmation

**The milestone is manual-push-only — nothing has been pushed.** The full unpushed range (all of Phase 5 plus its planning artifacts) is reported by:

```bash
git log origin/dev..dev --oneline   # currently 26 commits, b312484..7b8bf8a (phase planning + 05-01/02/03)
```

**1. Push:**

```bash
git push origin dev
```

**2. Dispatch the runner-confirmation job (the workflow commit must be on the remote first):**

```bash
gh workflow run feasibility.yml --ref dev
```

**3. Watch and download the artifacts:**

```bash
gh run watch      # select the feasibility run
gh run download <run-id> -n feas-spike-logs -D /tmp/feas-runner-artifacts
```

**4. Finalize the matrix:** compare the runner artifacts' evidence lines against `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md` and fill each row's Runner confirmation column (e.g. `confirmed <date> — matches local verdict` or the divergent evidence). The local verdicts are provisional until this is done (D-04). Note: the runner venv installs only `.[base]`, so its evo/evo2 legs are expected to carry the handler ImportError texts (fast, no model downloads), megadna downloads its 582MB checkpoint and fails the unpickle without the pinned clone, pybigwig fails its import, and marimo executes for real on the warm ModelScope cache — the runner leg reproduces the verdict *shape*, not the full prerequisite ladder.

### Blocking human check (gate="blocking-human")

Owner runs the hand-off above: (1) push, (2) `gh workflow run feasibility.yml --ref dev`, (3) download artifacts, (4) fill the matrix Runner confirmation column. D-04's official-verdict step is not closed until the column is filled. The pyBigWig provenance gate was moot this run — no dev-extra line was added (the row's verdict is environment-unavailable); for the record, PyPI metadata for pyBigWig 0.3.26 resolves to github.com/deeptools/pyBigWig (deeptools official), verified via the PyPI JSON API during Task 2.

## Known Stubs
None — no placeholder or unwired paths were introduced.

## Next Phase Readiness
- Phase 8 can author per-family execution tests against recorded verdicts: evo-1 on the 8k variant (notebook reference update + flash-attn build + fromstring shim productized in transformers_compat.py), evo2 with the noFP8 config condition, megaDNA with the pinned clone prerequisite, pyBigWig behind an evidence-backed typed skip unless the owner remediates the box, marimo via export-html
- The runner confirmation path is committed and dispatch-only; the PR-code-never-reaches-the-runner invariant is preserved (T-05-10 mitigated)
- models.lock guidance recorded in the matrix revision note (evo-1 'main' snapshot 78c715ab, not the research-expected 1.1_fix)

## Self-Check: PASSED

- scripts/feasibility/spike_families.py, .github/workflows/feasibility.yml, 05-FEASIBILITY.md, and all 8 spike logs exist on disk and in git
- Commits f7b5fa9, 53aaaae, ffde0c1, 41b5588, fc0d974, e74b0ce, 4f78a76, 7b8bf8a present on dev (measured 8 via `git rev-list --count 97a7dcb..HEAD`)
- Task 2 verify block: 10 matrix rows, 8 logs, pyproject clean, throwaway venv executable, three exact variant ids present
- Task 3 verify block: YAML parses, dispatch-only gate, runner label, if: always() upload, README documented, pyBigWig conditional matches the verdict with no pyproject line

---
*Phase: 05-execution-harness-honest-gates-runner-feasibility*
*Completed: 2026-10-02*
