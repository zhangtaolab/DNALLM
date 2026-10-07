---
phase: 05-execution-harness-honest-gates-runner-feasibility
reviewed: 2026-10-01T20:13:56Z
depth: standard
iteration: 2
files_reviewed: 22
files_reviewed_list:
  - docs/example/marimo/finetune/finetune_demo.py
  - docs/example/marimo/inference/inference_demo.py
  - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - docs/example/mcp_pydantic_ai.md
  - docs/example/notebooks/benchmark/benchmark.ipynb
  - docs/example/notebooks/data_prepare/finetune/finetune_data.ipynb
  - docs/example/notebooks/finetune_binary/finetune_binary.ipynb
  - docs/example/notebooks/finetune_multi_labels/finetune_multi_labels.ipynb
  - docs/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
  - docs/example/notebooks/finetune_NER_task/finetune_NER_task.ipynb
  - docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - .github/workflows/docs-validation.yml
  - .github/workflows/feasibility.yml
  - .github/workflows/README.md
  - pyproject.toml
  - README.md
  - scripts/check_docs_sync.py
  - scripts/feasibility/spike_families.py
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/expected_skips.yaml
findings:
  critical: 0
  warning: 2
  info: 5
  total: 7
status: issues_found
---

# Phase 5: Code Review Report (Iteration 2)

**Reviewed:** 2026-10-01T20:13:56Z
**Depth:** standard
**Files Reviewed:** 22
**Status:** issues_found

## Summary

Iteration-2 re-review after the fix pass (commits `89b2f63..824b901`). Scope:
the same 21 files as iteration 1 plus `tests/expected_skips.yaml`. Two goals:
verify all seven iteration-1 fixes (CR-01, WR-01..06) hold on current source,
and hunt for defects the fixes introduced or that iteration 1 missed.

### Fix verification — all 7 hold (independently re-verified, not taken from the fix report)

- **CR-01 (unused noqa):** `scripts/feasibility/spike_families.py:296` carries
  no `noqa`; live `ruff check .` exits 0 ("All checks passed!") across the
  repo. **VERIFIED.**
- **WR-01 (git-status guard):** `tests/examples/_execution.py:196-198` asserts
  `returncode == 0` before interpreting stdout. Live test: `REPO_ROOT` pointed
  at a non-git dir raises `git status failed (rc=128)...`; clean repo passes.
  **VERIFIED.**
- **WR-02 (byte-level mirror check):** `scripts/check_docs_sync.py:62-64` now
  calls `filecmp.cmpfiles(..., shallow=False)`. Live test: a same-size,
  same-mtime content divergence (`AAAA` vs `BBBB`, mtimes pinned) is CAUGHT
  (`DIFFER: x.txt`) where the old `dircmp.diff_files` shallow path returned
  `[]`; current tree still exits 0 (`OK: docs/example/ is in sync`). The
  Python-3.10-compat route (`cmpfiles`, not the 3.13+-only `dircmp(shallow=)`)
  is correct, and `cmpfiles` routes unreadable/type-mismatched files into the
  `UNREADABLE:` error path instead of crashing. All 11 in-scope
  `docs/example/**` mirrors re-confirmed byte-identical via `cmp`.
  **VERIFIED.**
- **WR-03 (nbclient in test extra):** `pyproject.toml:99` — `"nbclient>=0.10"`
  inside the `test` extra; `pytest tests/examples/test_notebook_execution.py
  --collect-only` collects 2 items cleanly. **VERIFIED.**
- **WR-04 (exit-code honesty):** `|| true` gone from
  `.github/workflows/feasibility.yml:88-92` (`set -o pipefail` + `tee` kept).
  `main()` exit semantics live-tested with monkeypatched families: all-FAIL
  run exits 0 with `failed_families=` listed; a mid-run crash (evo2 raising
  pre-evidence) prints a `result=CRASH` block, continues to the remaining
  families, and exits 1; failed GPU guard exits 2 before any family runs.
  **VERIFIED.**
- **WR-05 (GPU-absent evidence):** `feasibility.yml:39-50` — on GPU absence
  writes `gpu_absent=<UTC> nvidia-smi unavailable` into
  `spike-logs/spike_runner_all.log` then `exit 1`; the marker path is correct
  (written pre-checkout so the skipped checkout cannot remove it; the upload's
  `if: always()` still runs and at least one file matches, so the artifact
  carries the reason). Subsequent steps' explicit `if:` conditions get an
  implicit `success() &&` and are skipped as intended. **VERIFIED** (logic
  level; dispatch-level confirmation still pending on the runner, as the fix
  report itself notes).
- **WR-06 (permissions):** `feasibility.yml:18-19` — file-level
  `permissions: contents: read` matching the `ci.yml` convention. **VERIFIED.**

### New findings this iteration

The fix pass itself introduced one documentation regression (WR-07: the
workflows README still describes the GPU-absent behavior that WR-05 just
changed from green-skip to hard-fail). One further misdescription of the
dispatch semantics (WR-08) predates the fix pass but was missed in iteration 1
and materially affects how the owner reads the runner evidence. One quality
defect in lint-suppression comments (IN-05) is proven inert by reproduction.
The four known-deferred Info findings (IN-01..04) are re-confirmed still
present and remain Info.

## Warnings

### WR-07: Workflows README still documents the GPU-absent "fail-safe no-op" that WR-05's fix replaced — introduced by the fix pass

**File:** `.github/workflows/README.md:120-121` (vs `.github/workflows/feasibility.yml:39-50`)
**Issue:** Section 7 (Feasibility Spike Job), step 1 reads: "**Code Checkout /
GPU Check** (same fail-safe no-op as `test-mamba` if the box loses its GPU)".
The WR-05 fix deliberately inverted that behavior for `feasibility.yml`: GPU
absence now writes a `gpu_absent` marker and **fails the job** (evidence-only
deliverable must not green with zero evidence), explicitly diverging from
`test-mamba`'s green skip. The README was not updated, so the runbook now
misdocuments the exact honest-gate behavior this phase shipped: an operator
seeing a red gpu-absent feas-spike run (or reviewing whether the workflow
needs "fixing" back to a no-op) is told the opposite of what the code does.
The step order is also stale — the workflow runs GPU Check *before* Checkout
(required so the marker is written to an un-checked-out workspace), while the
README lists Checkout first.
**Fix:** Update section 7, step 1 to match the code:

```markdown
1. **GPU Check** (runs BEFORE checkout): on GPU absence writes a
   `gpu_absent` marker into the artifact path and FAILS the job — unlike
   `test-mamba`'s green fail-safe no-op, this job's deliverable is evidence,
   so zero-evidence green is forbidden (the `if: always()` upload still
   delivers the marker)
```

### WR-08: Dispatch comments claim "notebook variant first, then D-06 fallback" — `--fallback` replaces, not sequences, so the runner never produces notebook-variant evidence

**File:** `.github/workflows/feasibility.yml:75-77` and `.github/workflows/README.md:124` (code truth: `scripts/feasibility/spike_families.py:792-805`)
**Issue:** The run-step comment says "One --family all --fallback pass per the
plan: each family runs its notebook variant first, then its D-06 fallback",
and the workflows README section 7 step 4 says "**Spike Execution**:
`--family all` (notebook variants) then the D-06 fallback legs". Neither is
what the code does: `_run_family(family, fallback=True)` runs **only** the
fallback variant — evo1 gets `togethercomputer/evo-1-8k-base` instead of the
notebook's `131k` variant (`spike_families.py:795`), evo2 gets the notebook id
with the `noFA-noFP8` config override, megadna gets the notebook id plus the
pinned clone; the notebook variant never runs on the runner (pybigwig/marimo
ignore the flag). The single fallback pass matches the plan (05-03-PLAN.md
line 171 mandates exactly `--family all --fallback`), so the *behavior* is
plan-conformant — but the comment/README promise a two-leg ladder that does
not exist. The owner fills the verdict matrix's Runner confirmation column
from the artifact believing the notebook-variant leg was runner-confirmed;
only the `variant=` lines inside the log (easy to skip) reveal otherwise.
Pre-existing (the misdescription predates the fix pass); newly surfaced this
iteration because this workflow file was re-read in full.
**Fix:** Correct both descriptions to state the actual semantics:

```yaml
        # Single fallback pass per the plan: --fallback runs each family's
        # D-06 fallback variant (evo-1-8k-base / evo2 noFA-noFP8 config /
        # pinned megaDNA clone), NOT the notebook variant — notebook-variant
        # verdicts come from the committed local evidence (spike-logs/).
```

and in the README: "4. **Spike Execution**: `--family all --fallback` (D-06
fallback variants only; notebook-variant verdicts rest on the committed local
evidence)...".

## Info

### IN-01 (carried, iteration 1 — still present): `test_timeout` / `extra_inputs` spec keys are dead config

**File:** `tests/examples/_execution.py:54-60` and `tests/examples/test_notebook_execution.py:53,77-78`
**Issue:** Unchanged since iteration 1: the spec dict documents
`test_timeout` and `extra_inputs`, but the test layer hardcodes
`@pytest.mark.timeout(1800)` and the fixture calls
`seed_sandbox(pilot_dir, tmp_path)` without `extra_inputs`. Known-deferred;
both values coincide at 1800/empty today, so nothing breaks. Wire the spec
keys or delete them when Phase 8 generalizes the fixture.
**Fix:** deferred to Phase 8 fixture generalization (dynamic
`pytest.mark.timeout(spec["test_timeout"])` parametrization, or drop the keys).

### IN-02 (carried, iteration 1 — still present): `assert_tree_clean` fails on pre-existing developer WIP under `example/`

**File:** `tests/examples/_execution.py:173-201`
**Issue:** Unchanged since iteration 1: teardown asserts absolute cleanliness
of `example`/`docs/example` rather than a pre-run baseline/delta comparison,
so unrelated uncommitted local edits fail with a harness-attributed message.
(The WR-01 returncode guard was correctly added on top without changing this.)
Known-deferred.
**Fix:** snapshot `git status --porcelain --` output in fixture setup and
assert no *new* lines in teardown.

### IN-03 (carried, iteration 1 — still present): hardcoded developer home path in the mirrored NER dataset generator

**File:** `docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py:14`
**Issue:** `sys.path.insert(0, "/home/forrest/Github/DNALLM")` still present;
byte-identical in `example/` (mirror fidelity re-verified this iteration), so
per the phase scoping (D-03) the content repair is deferred to the Phase 8
per-notebook loop.
**Fix (Phase 8):** drop the `sys.path.insert` line or derive
`Path(__file__).resolve().parents[2]`.

### IN-04 (carried, iteration 1 — still present): pinned megaDNA clone uses a fixed shared `/tmp` path

**File:** `scripts/feasibility/spike_families.py:518`
**Issue:** `/tmp/megadna-pinned-clone` is still a predictable shared location;
commit-hash verification checks `HEAD` only, not untracked on-disk content.
Spike-runner blast radius only (verdict matrix), known-deferred.
**Fix:** `clone_dir = Path(tempfile.mkdtemp(prefix="megadna-pinned-"))`.

### IN-05 (new this iteration): `# ruff: ignore[rule-name]` comments are inert — not a ruff directive, proven by reproduction

**File:** `tests/examples/_execution.py:35,185`; `tests/examples/test_notebook_execution.py:13,67`; `scripts/feasibility/spike_families.py:7,259,265,271,635,675`
**Issue:** These comments use an invented syntax — ruff has no
`# ruff: ignore[...]` directive (file-level suppression is
`# ruff: noqa: <CODES>`, line-level `# noqa: <CODES>`). Reproduced live:
`ruff check --isolated --select S404,S603,S607` fires S603+S607 on the exact
lines the comments claim to suppress (e.g. S607 on the `pgrep` call directly
under `tests/examples/test_notebook_execution.py:67`), proving they suppress
nothing. Harmless today only because the project config
(`select = [...S...]` with `preview = true`) does not fire those codes —
`ruff check .` exits 0. The risk is reliance on phantom suppression: if the
config ever starts firing S603/S607 (rule graduation, preview flip, or a
no-preview runner), these call sites fail lint with no working suppression
pattern in the file to copy from.
**Fix:** Either delete the comments (nothing currently fires) or replace with
the real syntax where suppression is genuinely wanted:

```python
    result = subprocess.run(  # noqa: S603, S607
```

or file-level `# ruff: noqa: S603, S607` at the top of the module.

---

_Reviewed: 2026-10-01T20:13:56Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Iteration: 2 (prior CR-01 + WR-01..06 verified fixed; IN-01..04 carried as known-deferred)_
