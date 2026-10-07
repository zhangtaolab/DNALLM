# Phase 5: Execution Harness, Honest Gates & Runner Feasibility - Research

**Researched:** 2026-10-02
**Domain:** nbclient-as-library pytest execution harness; GitHub Actions gate honesty + branch protection; docs-mirror sync tooling; GB10 (aarch64 Blackwell) model-family feasibility spike
**Confidence:** HIGH — every load-bearing claim was verified this session by direct file reads, empirical probes run on this exact GB10 dev box (same hardware class as the nightly runner), the PyPI JSON API, or the v1 planning archives. Environment-dependent unknowns (evo-1/evo2/megaDNA package installability on aarch64) are explicitly flagged as spike-resolved, never guessed.

## Summary

Phase 5 has three workstreams with fully-mapped mechanics. (1) **The harness**: `tests/examples/_execution.py` + `tests/examples/conftest.py`, mirroring the `dnallm/mcp/tests/_network_skip.py` private-helper seam. The single most important correction to the milestone research: **`NotebookClient` in nbclient 0.11.0 is NOT a context manager** (`hasattr(NotebookClient, "__enter__") == False`); the context manager is `setup_kernel()`, and plain `client.execute()` already wraps cells in a start→`finally _cleanup_kernel()` chain, so kernel shutdown is guaranteed on error when the client owns the kernel manager. I ran a live deliberate-hang probe on this GB10 box: per-cell `timeout=3` raised `CellTimeoutError` at 3.5s, `shutdown_kernel="immediate"` left **zero** leftover kernels, and the mutated notebook node carried completed cells' outputs (the partial artifact). EXEC-06 is therefore mechanics-proven, not speculative. (2) **The honesty repairs**: the exact 5 `continue-on-error` lines, the exact drift inventory (`check_docs_sync.py` exits 1 with 36 error lines: 24 docs-only wrapper `.md`, 10 DIFFER files, 1 missing script mirror), the one latent `validate_docs_snippets.py` failure (`mcp_pydantic_ai.md:92` unterminated string — a wrapper line-wrap artifact, source notebook is single-line), and the v1-archived branch-protection `gh api` PUT (which **replaces** the contexts array — the new payload must name both `coverage-gate (py3.12, fast leg)` and `docs-validation`). (3) **The GB10 spike**: this dev box is the same GB10 (compute capability **(12, 1)**, torch 2.11.0+cu130); the exact notebook variants are `togethercomputer/evo-1-131k-base` (dnallm pins HF revision `1.1_fix`), `arcinstitute/evo2_1b_base` (requires the `evo2`+`vortex` packages, none installed), `lingxusb/megaDNA_updated` (torch-loads a pickled checkpoint, `weights_only=False`), and pyBigWig (x86_64-wheels-only confirmed via PyPI JSON → aarch64 sdist compile is the spike question). A directly-on-point upstream source — ArcInstitute evo2 discussion #221, "Evo 2 on DGX Spark (GB10)" — documents a working source-build recipe (flash-attn 2.8.0 + TE 2.12.0 at arch 120) and the accuracy caveat that only the 7B tier is accurate on Blackwell.

**Primary recommendation:** Build the harness exactly on the empirically-probed shapes below (plain `NotebookClient(...).execute()` with `resources={"metadata": {"path": sandbox}}`, `shutdown_kernel="immediate"`, per-cell timeout strictly below the per-test mark); pilot on `notebooks/inference/inference.ipynb` (model already in local ModelScope cache, predict-only, 7KB) plus optionally `inference_for_tRNA/inference.ipynb`; land WR-08 flip + wrapper-`.md` fix + byte-identical resync + the one snippet repair as one commit; run the spike locally in a throwaway venv, record the verdict matrix, then confirm once on the runner via a dispatch-gated job cloned from the `test-mamba` pattern.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**WR-08 honesty scope**
- **D-01:** ALL five `continue-on-error` steps in `docs-validation.yml` are flipped honest in this phase (check_docs_sync, validate_docs_snippets, validate_yaml, run example tests, run YAML load tests). If the snippets/YAML validators expose latent failures once unmasked, they are repaired to green within this phase — no advisory remnants. — **Reversibility:** costly
- **D-02:** `docs-validation` is promoted to a required branch-protection check on dev+main, same tier as `coverage-gate`. — **Reversibility:** reversible

**Notebook stale outputs**
- **D-03:** The 19/21 notebooks with committed outputs keep them as-is; the docs mirror is resynced byte-identical from the current state (outputs included). Output refresh happens only when Phase 8 repairs a given notebook (and via the Phase 7 showcase write-back for the two PlantHelixSeek notebooks). No bulk output-stripping in this phase.

**GB10 feasibility spike**
- **D-04:** Spike executes locally on the dev box first (same GB10 hardware class as the runner, verified via nvidia-smi) for fast iteration, then the resulting verdict matrix is re-confirmed once on the self-hosted nightly runner via `workflow_dispatch` to become the official verdict. The runner's schedule/dispatch-only security posture is untouched — PR-authored code never reaches it.
- **D-05:** Verdict depth uses the EXACT model variants the notebooks reference (e.g. `togethercomputer/evo-1-131k-base`, not a smaller evo-1): a family is FEASIBLE only after a real forward pass with the notebook variant succeeds, with time/VRAM/disk evidence recorded in the matrix. pyBigWig's entry needs import + a small real BigWig write/read. — **Reversibility:** costly
- **D-06:** If a family's notebook variant fails on GB10, the same spike falls back to that family's smallest viable variant; if the small variant runs, Phase 8 executes with the small variant and the notebook's model reference is updated accordingly. Typed `environment-unavailable:` skip is used only when both variants fail, with recorded evidence.

**Carried-forward decisions (locked earlier, not re-discussed)**
- nbclient used as a library inside parametrized pytest tests — no nbmake, no new test frameworks
- Timeout layering: nbclient per-cell timeout as the inner guard, `@pytest.mark.timeout(N)` as the outer backstop
- Harness mechanics live in `tests/examples/_execution.py` + a locally-scoped `tests/examples/conftest.py` — never root conftest, never inside `dnallm/`
- marimo apps execute via subprocess (export-html vs script-mode flavor decided by the Phase 5 pilot itself)
- Kernel management: context-managed `NotebookClient`, `shutdown_kernel="immediate"`, plus a nightly `pkill` hygiene step
- WR-08 flip and mirror-drift closure land in the same reviewable unit
- ModelScope-first model sourcing (applies to Phase 8 lock entries; spike downloads may use either hub)

### Claude's Discretion
- Choice of the 1–2 pilot notebooks (pick already-healthy, fast, no-huge-download ones)
- Typed-skip prefix naming details (`environment-unavailable:` vs `optional-dep:` assignment per family) as long as both are registered in `expected_skips.yaml` with the audit green
- Sandbox fixture mechanics (copy strategy, artifact layout) within the locked tmp-sandbox + cwd-redirect pattern
- Verdict-matrix document format and location (suggested: `.planning/` artifact + phase docs)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-01 | Private execution harness (`tests/examples/_execution.py` + locally-scoped `conftest.py`) runs notebooks via nbclient-as-library with per-cell timeout inside a per-test timeout mark, tmp-sandbox cwd isolation (kernel cwd = sandbox copy), context-managed kernel shutdown, partial-notebook failure artifacts captured on error | Verified nbclient 0.11.0 API + empirical probe; exact `_execution.py`/`conftest.py` skeletons below; `resources={"metadata": {"path": ...}}` kernel-cwd mechanism verified at `nbclient/client.py:535`; `setup_kernel()` context-manager + `execute()` internal `finally` cleanup verified from installed source |
| EXEC-06 | A deliberate-hang test proves the harness kills a hung kernel and leaves no `ipykernel_launcher` process behind | **Live probe on this GB10 box passed**: `timeout=3` → `CellTimeoutError` at 3.5s, `shutdown_kernel="immediate"` → zero leftover kernels; pgrep self-match trap identified with the delta-zero assertion design |
| CI-01 | WR-08 closed — docs-validation `continue-on-error: true` removed in the same reviewable unit as the mirror-drift closure | The 5 exact step locations listed; full 36-line drift inventory captured; one latent snippets failure found (`mcp_pydantic_ai.md:92`) that must be repaired in-phase per D-01 |
| CI-02 | WR-09 closed — docs-validation installs the `mcp` extra; README "Local Testing" install line corrected | Workflow install line at `docs-validation.yml:42` (`.[test,dev]`); verified `mcp` extra exists in pyproject and that example tests pass with it (115 passed locally); README current text located (no "Local Testing" heading exists today — see Pitfall 6) |
| REPAIR-02 | The already-broken docs/example mirror is closed (sync-script wrapper-`.md` handling fixed, byte-identical resync, missing script mirrored) | `check_docs_sync.py` read fully; IGNORE set quoted; minimal fix = ignore docs-only `*.md` wrappers; resync = copy 10 DIFFER files + 1 missing script from `example/` → `docs/example/` |
| FEAS-01 | Phase-1 spike produces a written verdict matrix for evo-1 / evo2 / megaDNA / pyBigWig on the aarch64 GB10 runner; smallest viable real variants enabled wherever feasible; `environment-unavailable:` typed skips only with recorded infeasibility evidence | Exact notebook variants extracted; dnallm special-handler code paths read (`evo.py`, `megadna.py`, `support.py`); local cache state audited (evo/megaDNA NOT cached); upstream GB10 evidence (evo2 discussion #221) fetched; spike script shapes + runner-confirmation workflow shape provided |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Test files `test_<module>.py`; test classes `Test*`; test functions `test_*` descriptive behavior names; relative imports inside `dnallm/`, absolute imports in tests
- ruff line-length 100, `E4,E7,E9,F,W,B,C4,UP,N,S,T20,PT,Q,RUF` with `preview = true`; `fixable = ["ALL"]`; tests get per-file ignores (S101 asserts allowed, etc.)
- Type hints on new code (mypy relaxed); PEP 604 unions (`X | None`), built-in generics
- Google-style docstrings (`Args:`/`Returns:`/`Raises:`); comments in English
- No bare `print()` in library code (T20) — harness lives in `tests/` so prints are allowed but prefer `logging`/pytest output
- `ValueError` with descriptive matchable messages for invalid input; `pytest.raises(..., match=...)` convention
- GSD workflow enforcement: phase work runs under `/gsd-execute-phase`
- Pre-commit runs `ruff format` → `ruff check` → `mypy dnallm/` (advisory)
- CI matrix compat: Python 3.11/3.12/3.13, numpy 1.26.4 & 2.2.0 — harness code must be 3.10+-syntax clean (`requires-python >= 3.10`)

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Notebook kernel execution (launch, run cells, collect outputs) | Test layer — `tests/examples/_execution.py` wrapping nbclient | nbclient library (owns kernel lifecycle) | Test-only code must not ship in the wheel; nbclient already guarantees cleanup inside `execute()` |
| Kernel lifecycle / kill-on-hang | nbclient `NotebookClient.execute()` internal `finally → _async_cleanup_kernel()` | Nightly `pkill` hygiene step (Phase 9 wiring; hygiene-step design lands now) | Verified from installed source: `client.py:503-516, 655-660`; pytest-timeout kills only the pytest process, so the cell timeout must fire first |
| Timeout layering | nbclient per-cell `timeout` trait (inner) | `@pytest.mark.timeout(N)` per-test mark (outer) | Both verified: per-cell semantics at `client.py:831-841`; 9 existing mark precedents (900/1800/3600/7200) |
| Sandbox isolation (cwd, sibling inputs, artifacts) | Test layer — pytest fixtures in `tests/examples/conftest.py` | Repo gitignore (belt-and-braces) | Kernel cwd set via `resources={"metadata": {"path": sandbox}}`; v1 Phase-2 PDF `autouse` rebind is the precedent pattern |
| Docs-mirror byte-identity | Repo script `scripts/check_docs_sync.py` | CI job `docs-validation` (hosted, ubuntu-latest) | The script is the contract; the workflow is the enforcement tooth |
| Branch protection (required checks) | Owner via `gh api` (repo settings API) | Executor records exact runnable commands (v1 precedent) | Branch protection is an owner-admin action; v1 Phase 4 archived the exact PUT payload |
| Model-family feasibility verdicts | Spike script run by executor on the GB10 dev box | Runner confirmation via dispatch-gated workflow job | D-04: local first, runner confirmation second; `test-mamba` job is the workflow pattern to clone |
| Typed-skip taxonomy | `tests/expected_skips.yaml` + `scripts/audit_skips.py` | Harness helper emitting prefixed messages | Existing audit fails on any unlisted skip; new prefixes must land in the allowlist in the same unit |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **nbclient** | 0.11.0 (installed; latest) | Execute notebooks as library calls inside parametrized pytest tests | It IS the engine under nbconvert/nbmake, minus plugin semantics. Verified installed at `.venv/lib/python3.13/site-packages/nbclient/`. Currently transitive via `jupyter>=1.1.1` in the `notebook` extra — the milestone research recommends making it explicit: add `"nbclient>=0.10"` to the `notebook` extra so the harness import is a declared dependency `[VERIFIED: .venv inspection + PyPI]` |
| **nbformat** | 8.1.1 (installed, via jupyter) | Read/write notebook nodes; generate the deliberate-hang fixture notebook in-test via `nbformat.v4.new_notebook/new_code_cell` | Used by the harness for sandbox copies (`nbformat.read`/`write`) and the kill-test fixture; transitive today, same explicit-declaration argument `[VERIFIED: .venv inspection]` |
| **pytest-timeout** | 2.4.0 (installed; pinned `>=2.3.1,<2.5`) | Per-test timeout marks as the outer backstop | Existing repo ladder: 900 (downloads), 1800 (real inference), 3600/7200 (real finetune) — `[VERIFIED: tests/models/test_model.py:179, tests/inference/test_inference_real_model.py:23, tests/finetune/test_trainer_real_model.py:54]` |
| **pytest** | 8.4+ (`>=8.4` in test extra) | Parametrized execution tests, fixtures | No new frameworks (locked decision) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pyBigWig** | 0.3.26 | BigWig write/read for the FEAS-01 matrix entry (import + small real write/read per D-05) | Spike-gated ONLY: no aarch64 wheels on PyPI (x86_64 cp39–cp313 + sdist only) — add to `dev` extra only if the GB10 sdist compile succeeds `[VERIFIED: PyPI JSON API 2026-10-02]` |
| **evo2 / evo-1 / stripedhyena** (spike-only) | evo2 0.6.0 latest / evo-1 1.1.2 / stripedhyena 0.2.2 | dnallm's evo special handlers import these | NEVER committed to pyproject in this phase; install only inside a throwaway spike venv |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nbclient as library | nbmake / `nbconvert --execute` / papermill | All rejected by locked decision (new plugin semantics / subprocess opacity / no parametrization need) |
| pyBigWig sdist compile on GB10 | pybbi (ships aarch64 wheels) | pybbi is read-focused and not the milestone-locked library; legitimacy gate also flags it (no source repo). Use only if pyBigWig compile proves impossible AND owner approves a library swap — otherwise typed skip with evidence |
| Third-party aarch64 pyBigWig wheels (RISE index) | — | Rejected: non-official index = supply-chain risk on a box that runs `trust_remote_code` models |

**Installation (pyproject delta — the ONLY committed dependency changes this phase):**
```toml
[project.optional-dependencies]
notebook = [
    "jupyter>=1.1.1",
    "marimo>=0.16.3",
    "nbclient>=0.10",   # ADD — make the today-transitive dep explicit; tests import it
]
# dev gains "pyBigWig>=0.3.26; platform_system != 'Windows'" ONLY if the spike proves the aarch64 build
```
**Version verification (run this session):** `nbclient 0.11.0`, `nbformat` installed in `.venv`; PyPI JSON: pyBigWig latest 0.3.26 (files listed above); `marimo 0.25.0`, `torch 2.11.0+cu130` local.

## Package Legitimacy Audit

> Gate run via `gsd-tools query package-legitimacy check --ecosystem pypi nbclient pyBigWig evo2 evo-1 stripedhyena pybbi`. All verdicts came back `SUS` driven by `unknown-downloads` (the gate's PyPI download-stats channel returned null this session) — NOT by missing repos. Dispositions below weigh the repo-signal manually.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| nbclient | PyPI | current release 2026-06; project since 2019 | unknown (gate) | github.com/jupyter/nbclient | SUS (data-gap only) | Approved — [WARNING: gate flagged unknown-downloads.] Jupyter-org official package, already installed transitively via `jupyter>=1.1.1`; confirm at install time with `uv pip install` from default index |
| pyBigWig | PyPI | 0.3.26 released 2026-09-14 | unknown (gate) | github.com/deeptools/pyBigWig | SUS (too-new + data-gap) | Spike-gated — [WARNING: verify before adding to dev extra.] deeptools official; add only after the GB10 sdist build succeeds |
| evo2 | PyPI | 2026-06-19 | unknown (gate) | github.com/arcinstitute/evo2 | SUS | Spike-only, never committed |
| evo-1 | PyPI | PyPI page dates to 2018 (name-squat risk: repoUrl resolves to a personal fork) | unknown (gate) | github.com/ToniRV/evo-1 (unofficial mirror) | SUS | Spike-only; prefer the documented `evo-design/evo` install path; verify what `pip index versions evo-1` actually resolves to before installing |
| stripedhyena | PyPI | 2024-02-23 | unknown (gate) | github.com/togethercomputer/stripedhyena | SUS | Spike-only, never committed |
| pybbi | PyPI | 2025-10-30 | unknown (gate) | none | SUS | NOT USED — no source repo |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** all above (data-gap driven); only `nbclient` and (conditionally) `pyBigWig` can ever reach pyproject, and both are canonical-org packages. Planner adds `checkpoint:human-verify` before the pyBigWig commit if the spike green-lights it.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────── PHASE 5 WORKSTREAMS ───────────────────────────┐
                    │                                                                           │
  A. HARNESS (EXEC-01/06)                     B. HONEST GATES (CI-01/02, REPAIR-02)          C. FEASIBILITY (FEAS-01)
                    │                                                                           │
  pytest (slow leg, nightly-only)              push/PR to dev/main                           GB10 dev box (same class as runner)
   │                                           │                                             │
   ├─ tests/examples/conftest.py               docs-validation.yml (ubuntu-latest)            spike script per family:
   │    notebook_sandbox fixture                ├─ install .[test,dev,mcp]   ← WR-09          evo-1-131k-base (HF rev 1.1_fix)
   │      copy nb+siblings → tmp_path           ├─ check_docs_sync.py       ← flip honest     evo2_1b_base (evo2 pkg + .pt)
   │                                           ├─ validate_docs_snippets   ← flip honest     megaDNA_updated (torch.load .pt)
   ├─ tests/examples/_execution.py             ├─ validate_yaml            ← flip honest     pyBigWig import+write/read
   │    NotebookClient(resources={"metadata":  ├─ pytest test_examples.py  ← flip honest       │
   │      {"path": sandbox}}, timeout=cell,     └─ pytest test_yaml_load    ← flip honest      measurements: time / VRAM / disk
   │      shutdown_kernel="immediate")                │                                      │
   │      → kernel cwd = sandbox (client.py:535)      check_docs_sync.py repair                  verdict matrix (written doc)
   │      → per-cell timeout ── inner guard           ├─ ignore docs-only wrapper *.md              │
   │    @pytest.mark.timeout(N) ── outer guard        ├─ byte-identical resync (10 files)          workflow_dispatch confirmation
   │      cell_timeout < mark < job budget            └─ mirror generate_bpe_dataset.py            on [self-hosted, dnallm-nightly]
   │                                                 README Testing install line corrected      (test-mamba job pattern)
   │    on error:                                                     │
   │      CellTimeoutError/CellExecutionError                         gh api PUT branches/{dev,main}/protection
   │      + nbformat.write(nb → failure artifact)                     contexts = [coverage-gate…, docs-validation]
   │                                                                 (owner runs; PUT REPLACES the array)
   └─ kill test: time.sleep cell > cell timeout                                              
      → CellTimeoutError → SIGKILL kernel → pgrep delta-zero
```

### Recommended Project Structure

```
tests/examples/
├── test_examples.py                  # existing structural layer — UNTOUCHED
├── conftest.py                       # NEW — locally-scoped fixtures only
└── _execution.py                     # NEW — private harness mechanics, zero tests
      ├── NOTEBOOK_EXEC_SPECS: dict[str, dict]   # path → {cell_timeout, test_timeout, extra_inputs}
      ├── seed_sandbox(src_dir, tmp_path, extra_inputs) → Path
      ├── run_notebook(nb_path, sandbox, cell_timeout, artifact_dir) → NotebookNode (raises on cell error)
      └── assert_tree_clean(paths=("example", "docs/example")) → None
```
Rationale (locked): never root `tests/conftest.py` (shared by ~1,700 tests), never `dnallm/` (ships in wheel; coverage denominator). Mirrors `dnallm/mcp/tests/_network_skip.py` — a private `_`-prefixed helper module beside its consumer tests. Parametrized ids use `str(p.relative_to(EXAMPLE_DIR))` exactly like `test_examples.py:108` so junit names stay greppable for `audit_skips.py`.

### Pattern 1: run_notebook — the proven invocation shape

**What:** One function wrapping nbclient; per-cell timeout, immediate shutdown, sandbox cwd, partial-artifact capture.
**When to use:** Every notebook execution test (Phase 5 pilot; Phase 8 rollout reuses verbatim).

```python
# Verified against installed nbclient 0.11.0 + live probe on GB10, 2026-10-02.
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

def run_notebook(nb_path, sandbox, cell_timeout=600, artifact_dir=None):
    """Execute a notebook copy inside *sandbox*; return the executed node.

    Raises CellExecutionError / CellTimeoutError after writing the
    partial-notebook artifact (completed cells' outputs included).
    """
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=cell_timeout,              # PER CELL (client.py:831-841)
        allow_errors=False,                # default; fail at first error = repair signal
        kernel_name="python3",
        startup_timeout=120,               # default 60; model-heavy first cell justifies margin
        shutdown_kernel="immediate",       # SIGKILL now= True (client.py:503-516)
        resources={"metadata": {"path": str(sandbox)}},   # kernel cwd (client.py:535)
        # env tweaks for deterministic headless execution:
        # MPLBACKEND=Agg, MPLCONFIGDIR=<tmp>, WANDB_MODE=disabled, TOKENIZERS_PARALLELISM=true
    )
    try:
        client.execute()   # internally: async_setup_kernel → cells → finally _cleanup_kernel()
        return nb
    except (CellExecutionError, CellTimeoutError):
        if artifact_dir is not None:
            nbformat.write(nb, Path(artifact_dir) / f"{nb_path.stem}.executed.ipynb")
        raise
```

Key verified facts:
- `NotebookClient` is **not** a context manager (`hasattr(..., "__enter__") == False`); `setup_kernel()` is (`client.py:581-609`). Plain `execute()` wraps itself in the async setup/cleanup (`client.py:669-690` + `finally` at `:655-660`) — cleanup runs on cell error AND on cell timeout, as long as you do NOT pass `km=` (the client must own the kernel manager: `owns_km`).
- `timeout` is per-cell; `timeout_func` (callable per cell) exists if per-cell-class timeouts are ever needed (`client.py:831-841`).
- The timeout exception message embeds the failing cell source verbatim — probe output: `"A cell timed out while it was being executed, after 3 seconds. ... Here is a preview of the cell contents: ... time.sleep(300)"`.
- The mutated `nb` node is the partial artifact — probe confirmed `nb.cells[0]["outputs"] == [{'output_type': 'stream', 'name': 'stdout', 'text': 'hello from cell0\n'}]` after cell 1 timed out. Persist it AND the exception text (`str(exc)`) — the timed-out cell itself carries no error output in the node.

### Pattern 2: Sandbox seeding + tree-clean guard (v1 Phase-2 PDF pattern generalized)

**What:** Copy the artifact dir's notebook + sibling inputs into `tmp_path`; execute with kernel cwd there; assert the repo tree stays clean after.
**When to use:** Every execution test.

```python
def seed_sandbox(src_dir: Path, tmp_path: Path, extra_inputs: list[Path] | None = None) -> Path:
    """Copy the notebook + sibling files into tmp_path/<name>/ and return it."""
    sandbox = tmp_path / src_dir.name
    shutil.copytree(src_dir, sandbox, ignore=shutil.ignore_patterns(
        ".ipynb_checkpoints", "__pycache__", "outputs*", "results*", "*.gz"))
    for extra in extra_inputs or []:
        dest = sandbox / extra.name
        if not dest.exists():
            shutil.copy2(extra, dest)
    return sandbox


def assert_tree_clean(paths: tuple[str, ...] = ("example", "docs/example")) -> None:
    """Fail if an execution wrote into the repo tree (scoped — the repo carries
    untracked .planning/.gsd working dirs that are none of the harness's business)."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.stdout.strip() == "", f"execution dirtied the tree:\n{result.stdout}"
```

v1 precedent (exact shape, `[VERIFIED: tests/inference/test_plot.py:168-171]`):
```python
@pytest.fixture(autouse=True)
def pdf_output_dir(tmp_path, monkeypatch):
    """Write all PDF artifacts under tmp_path so the repo working tree stays clean."""
    monkeypatch.setattr(sys.modules[__name__], "PDF_OUTPUT_DIR", tmp_path)
```
Its recorded lesson (`[VERIFIED: .planning/milestones/v1-phases/02-suite-hygiene-known-bug-fixes/02-02-SUMMARY.md:36]`): rebind via `sys.modules[__name__]`, never a dotted path — tests/ has no `__init__.py` so pytest imports modules top-level and a dotted-path setattr rebinds a second module object. For notebooks the equivalent trap is avoided entirely by cwd: notebooks use `load_config("./inference_config.yaml")` relative paths, and the kernel cwd (`resources.metadata.path`) resolves them inside the sandbox. The twice-run tree-clean proof from v1 (`pytest -m pdf` ×2 → `git status --porcelain tests/inference/` empty both times) is the verification pattern to repeat for the pilot.

### Pattern 3: The deliberate-hang kill test (EXEC-06)

**What:** An in-test-generated notebook whose second cell sleeps far beyond the per-cell timeout; assert `CellTimeoutError` fires and the kernel is gone.
**When to use:** Ships with the harness in Phase 5; runs every census thereafter.

```python
import subprocess, time
import nbformat.v4 as nbf
from nbclient.exceptions import CellTimeoutError

def _kernel_count() -> int:
    """Count live ipykernel_launcher processes. In-pytest use is self-match-safe:
    the pytest cmdline never contains the pattern (the enclosing bash -c of a CI
    step WOULD — see Pitfall 1)."""
    r = subprocess.run(["pgrep", "-f", "ipykernel_launcher"], capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])

@pytest.mark.slow
@pytest.mark.timeout(120)   # outer backstop: kernel start (~5s) + 3s cell + kill + poll ≈ 15-25s
def test_hung_kernel_is_killed_and_cleaned_up(tmp_path):
    nb = nbf.new_notebook(cells=[
        nbf.new_code_cell('print("ok")'),
        nbf.new_code_cell("import time; time.sleep(300)"),
    ])
    before = _kernel_count()
    client = NotebookClient(nb, timeout=3, kernel_name="python3",
                            shutdown_kernel="immediate",
                            resources={"metadata": {"path": str(tmp_path)}})
    with pytest.raises(CellTimeoutError):
        client.execute()
    # delta-zero, polled: kernel kill is async-ish; 15s is generous (probe: gone instantly)
    deadline = time.time() + 15
    while time.time() < deadline and _kernel_count() > before:
        time.sleep(0.5)
    assert _kernel_count() == before, "hung kernel survived the harness"
```

**Probe-verified on this GB10 box (2026-10-02):** `CellTimeoutError` raised after 3.5s with `timeout=3`; kernel count returned to baseline immediately (no polling actually needed, but keep it — CI boxes differ); `shutdown_kernel="immediate"` confirmed as the `now=True` path at `client.py:505-509`. Generate the fixture notebook in-test (as above) — a committed `hang.ipynb` under `example/` would be picked up by `test_examples.py`'s `rglob` and eventually by execution rollout.
Mark placement: `@pytest.mark.slow` (execution family joins the slow side by locked decision). It is technically fast-leg-capable (~20s, no GPU/network) — flag for the planner as an option, default slow.

### Pattern 4: Mirror repair — check_docs_sync.py minimal honest fix + resync

**What:** Teach the sync script that docs-only `*.md` wrappers are intentional; then make the trees byte-identical.
**Current behavior** `[VERIFIED: scripts/check_docs_sync.py:8-16 — ignore set quoted verbatim]`:
```python
IGNORE = {
    "__pycache__",
    "logs",
    "outputs",
    "outputs_multilabel",
    ".ipynb_checkpoints",
    ".gitignore",
}
IGNORE_SUFFIXES = (".gz", ".log")
```
The `right_only` branch (lines 33-42) reports every docs-only file — including the 24 wrapper `.md` tutorial skeletons generated by `scripts/generate_md_from_notebook.py` / `generate_md_from_marimo.py` (headers verified: "Auto-generate markdown tutorial skeletons from Jupyter notebooks").
**Minimal fix:** add `".md"` handling that ignores docs-only `*.md` files (files present in `docs/example/` but absent in `example/`) — e.g. a `DOCS_ONLY_SUFFIXES = (".md",)` consulted only in the `right_only` loop, or drop `name.endswith(".md")` there. Do NOT add `.md` to `IGNORE` globally — that would also silence a legitimately-mirrored `.md` differing or going missing on the example side (the `left_only`/`diff_files` paths must stay strict). `example/notebooks/overview.md` exists on BOTH sides today, proving the distinction matters.
**Byte-identical resync (D-03 direction: copy FROM `example/` TO `docs/example/`):** the 10 DIFFER files + 1 missing script, exact list from the run this session (`python scripts/check_docs_sync.py` → exit 1, 36 lines):
- DIFFER: `marimo/finetune/finetune_demo.py`, `marimo/inference/inference_demo.py`, `mcp_example/mcp_client_ollama_langchain_agents.ipynb`, `mcp_example/mcp_client_ollama_pydantic_ai.ipynb`, `notebooks/benchmark/benchmark.ipynb`, `notebooks/data_prepare/finetune/finetune_data.ipynb`, `notebooks/finetune_NER_task/data_generation_and_inference.ipynb`, `notebooks/finetune_NER_task/finetune_NER_task.ipynb`, `notebooks/finetune_binary/finetune_binary.ipynb`, `notebooks/finetune_multi_labels/finetune_multi_labels.ipynb`
- ONLY in example/: `notebooks/finetune_NER_task/generate_bpe_dataset.py` (copy into `docs/example/notebooks/finetune_NER_task/`)
- ONLY in docs/example/ (24 wrapper `.md`, all resolved by the ignore fix): `mcp_langchain.md`, `mcp_pydantic_ai.md`, `marimo/benchmark/benchmark_demo.md`, `marimo/finetune/finetune_demo.md`, `marimo/inference/inference_demo.md`, `notebooks/{benchmark, data_prepare_finetune, data_prepare_predict, embedding_attention, finetune_NER_data_generation, finetune_NER_task, finetune_binary, finetune_custom_head, finetune_generation, finetune_multi_labels, inference, inference_evo_models, inference_generation, inference_megaDNA, inference_trna, interpretation, lora_finetune, lora_inference, mutagenesis}.md`

### Pattern 5: Branch protection — the exact owner command (v1 precedent + Phase-5 delta)

v1 archived command `[VERIFIED: .planning/milestones/v1-phases/04-ci-gate-enforcement/04-03-SUMMARY.md:169-181]`:
```bash
gh api -X PUT repos/zhangtaolab/DNALLM/branches/dev/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["coverage-gate (py3.12, fast leg)"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```
**Phase 5 delta — CRITICAL:** the PUT **replaces** the `contexts` array. The new payload must name BOTH checks or docs-validation promotion silently un-requires coverage-gate:
```bash
"contexts": ["coverage-gate (py3.12, fast leg)", "docs-validation"]
```
The docs-validation check context is `docs-validation` — the job's `name:` field `[VERIFIED: .github/workflows/docs-validation.yml:14 — "name: docs-validation"]`; there is no matrix suffix to reproduce. Same command for `branches/main/protection`. `gh` is installed and authenticated on this box (`/home/linuxbrew/.linuxbrew/bin/gh`, account `forrestzhang`); the executor verifies with `gh api repos/zhangtaolab/DNALLM/branches/dev/protection` → `required_status_checks.contexts` contains both entries (the v1 UAT did exactly this check — `[VERIFIED: .../04-UAT.md:19-21]`).

### Pattern 6: GB10 spike — per-family script shapes and the runner confirmation job

Local-first (D-04), throwaway venv so the project venv is never polluted:
```bash
python -m venv /tmp/feas-venv && /tmp/feas-venv/bin/pip install -e ".[base]" --no-cache-dir
# per family: /tmp/feas-venv/bin/python spike_<family>.py 2>&1 | tee spike_<family>.log
```
Minimal forward shapes (all via the dnallm route — the notebooks never use evo2/evo-1 packages directly):
```python
# spike_evo1.py — notebook variant: togethercomputer/evo-1-131k-base
from dnallm.configuration.configs import load_config
from dnallm.models import load_model_and_tokenizer
import time, torch
cfg = {"task": type("T", (), {"task_type": "generation"})}  # or load_config on a copied YAML
t0 = time.time()
model, tok = load_model_and_tokenizer("togethercomputer/evo-1-131k-base", task_config=cfg["task"], source="huggingface")
print("load_s", time.time() - t0)
ids = tok(["ACGT" * 64], return_tensors="pt")          # 256nt prompt
out = model.generate(ids["input_ids"].to(model.device), max_new_tokens=32) if hasattr(model, "generate") else model(ids["input_ids"].to(model.device))
print("forward_ok", float(out.logits.shape[-1]) if hasattr(out, "logits") else "generated")
print("peak_vram_gb", torch.cuda.max_memory_allocated() / 1e9)
```
- **evo-1** prerequisite: `pip install evo-1 stripedhyena` — dnallm's handler imports `from evo import Evo` and `from stripedhyena.{utils,model,tokenizer} import ...` `[VERIFIED: dnallm/models/special/evo.py:296-301]`; without them the handler raises `ImportError("EVO-1 package is required for togethercomputer/evo-1-131k-base but not installed. ...")` — that exact message becomes the typed-skip evidence if install fails. Handler quirks that shape the spike: it downloads via `_get_model_path_and_imports(model_name, source, revision=...)` with `revision = "1.1_fix" if "." in model_name and source == "huggingface" else "main"` `[VERIFIED: evo.py:371]` → for `togethercomputer/evo-1-131k-base` from huggingface the pinned revision IS `1.1_fix` (record it in the matrix; it is also the natural `models.lock` revision pin). Weights transfer: HF `AutoModelForCausalLM` (trust_remote_code) → local `StripedHyena` → `to_bfloat16_except_poles_residues()` `[VERIFIED: evo.py:325-337]`. Config selection: flash-attn absent → `evo-1-131k-base-noFA.yml` `[VERIFIED: evo.py:357-366 + ls dnallm/configuration/evo/]`.
- **evo2** prerequisite: `pip install evo2` (pulls `vortex`) — handler imports `from evo2 import Evo2` + `from vortex.model.tokenizer import CharLevelTokenizer` `[VERIFIED: evo.py:191-192]`, then loads `<snapshot>/evo2_1b_base.pt` via `Evo2.load_evo2_model` with a dnallm-packaged config YAML `[VERIFIED: evo.py:247-253]`. **Auto-selection trap:** `is_fp8_capable()` returns True on GB10 (compute capability (12,1) ≥ (9,0)) `[VERIFIED: dnallm/utils/support.py:16-18 + live torch.cuda.get_device_capability() == (12, 1)]`, so with no flash-attn the handler picks `evo2-1b-8k-noFA.yml` — an FP8 config — while upstream evidence says FP8 accuracy recipes need Hopper. If the notebook variant fails, the smallest-viable fallback per D-06 is forcing the `-noFA-noFP8.yml` config variant (exists: `evo2-1b-8k-noFA-noFP8.yml`), which requires either source-path loading (`source="local"` with a prepared dir) or a small documented deviation — record whichever path the spike proves.
- **megaDNA** — no extra package strictly required by the handler code path: it defines `DNATokenizer` inline and does `torch.load(downloaded_model_path, weights_only=False)` on `megaDNA_phage_145M.pt` from the HF repo `lingxusb/megaDNA_updated` `[VERIFIED: dnallm/models/special/megadna.py:110-120]`. BUT `weights_only=False` unpickles a full `nn.Module` — if the pickle references classes from the cloned `github.com/lingxusb/megaDNA` `model/` package, the import is required at unpickle time (that is why the notebook's commented cell says `git clone`). Spike question #1: does the unpickle succeed without the repo? If not, the smallest honest fallback is a pinned, hash-verified clone (Pitfall 8) or typed `environment-unavailable:`/`optional-dep:` skip with the failure recorded.
- **pyBigWig** — `/tmp/feas-venv/bin/pip install pyBigWig` (expect sdist compile; needs gcc + zlib/libcurl headers — check `gcc --version`, `dpkg -l zlib1g-dev` or just let it fail and record) then per D-05: `bw = pyBigWig.open(p,"w"); bw.addHeader([("Chr1", 1000)]); bw.addEntries("Chr1", [0,100], ends=[50,200], values=[0.5,1.0]); bw.close()` + reopen and `bw.values("Chr1", 0, 50)` round-trip.
- **marimo flavor spot-check** (decides export-html vs script-mode for Phase 8): `cd <tmp-copy of example/marimo/inference> && marimo export html inference_demo.py -o /tmp/out.html` vs `python inference_demo.py` (all 3 apps end with `if __name__ == "__main__": app.run()` `[VERIFIED: example/marimo/inference/inference_demo.py:295]`); defaults verified: task `'open chromatin'`, model `'Plant DNABERT'`, tokenizer `'BPE'`, source `'modelscope'` `[VERIFIED: inference_demo.py:36-80]` → maps to `zhangtaolab/plant-dnamamba-BPE-open_chromatin`, already in the local ModelScope cache, so the spot-check is warm. Script-mode `app.run()` blocks (starts a server? verify — `marimo run` binds a port; bare `app.run()` in 0.25 runs the reactive graph and returns; the CLI `marimo run` is the server). Record which flavor yields deterministic exit codes + an artifact.

**Evidence to record per family (D-05):** install steps + versions, load seconds, forward/generate seconds, `torch.cuda.max_memory_allocated`, model disk footprint (`du -sh` of the snapshot), exact failure text if infeasible. Matrix location (Claude's discretion): `.planning/phases/05-.../05-FEASIBILITY.md` or `docs/` — suggest `.planning/` artifact + pointer from phase verification.

**Runner confirmation (D-04):** add a dispatch-gated job cloned from the `test-mamba` pattern `[VERIFIED: .github/workflows/ci.yml:262-279 — "runs-on: [self-hosted, dnallm-nightly]", "if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"]`:
```yaml
feas-spike:
  runs-on: [self-hosted, dnallm-nightly]
  if: github.event_name == 'workflow_dispatch'
  timeout-minutes: 240
  steps:
    - checkout; uv venv; uv pip install -e ".[base]"
    - run: python scripts/feasibility/spike_evo1.py   # or repo-committed spike runner with per-family subcommands
    - upload-artifact (spike logs + matrix) if: always()
```
Owner triggers with `gh workflow run CI --ref dev` (manual push of the workflow commit first — milestone is manual-push-only; report the unpushed range). Whether it lands in `ci.yml` or a dedicated `feasibility.yml` is planner/owner taste; keep it dispatch-only to preserve the PR-code-never-touches-the-runner invariant.

### Anti-Patterns to Avoid

- **Treating `with NotebookClient(...)` as the API** — it is not a context manager in 0.11.0; use plain `execute()` (cleanup guaranteed internally) or `setup_kernel()` explicitly. Milestone research's "context manager" phrasing maps to the latter.
- **Adding `.md` to the global IGNORE in check_docs_sync.py** — silences the strict side too; scope the relaxation to docs-only (`right_only`) files only.
- **PUT-ting branch protection with only the new context** — replaces the array and un-requires coverage-gate.
- **`pgrep -f ipykernel_launcher` inside a CI bash step** — the enclosing `bash -c` cmdline contains the literal pattern → self-match. In pytest (`subprocess.run(["pgrep", ...])`) it is safe. For workflow hygiene steps use the bracket trick: `ps -eo args | grep "[i]pykernel_launcher"`.
- **Committing a hang-fixture notebook under `example/`** — discovered by `rglob` collection.
- **Running the spike in the project venv** — evo2/stripedhyena/flash-attn/TE installs would mutate the dev environment that the 96.30% gate depends on.
- **Skipping a family without recorded evidence** — D-05/D-06 mandate variant-then-fallback-then-skip ordering with the failure text in the matrix.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Kernel launch/exec/cleanup | Custom subprocess+jupyter_client plumbing | nbclient `NotebookClient` | Handles ZMQ channels, per-cell timeout, interrupt/kill fallbacks, output collection — all verified |
| Notebook JSON I/O / fixture generation | Raw dict manipulation | `nbformat.read/write` + `nbformat.v4.new_notebook` | Schema validation, list-vs-string source normalization |
| Process-liveness assertion | PID tracking of kernel children | `pgrep -f` delta (in-test) | Kernel PIDs are grandchildren; pgrep by cmdline is the only portable handle |
| Skip allowlisting | Per-test skip-reason suppression | Existing `expected_skips.yaml` + `audit_skips.py` contract | The audit already fails CI on unlisted skips; new prefixes just add entries |
| Docs-mirror diffing | New sync tool | Extend `check_docs_sync.py` (12-line change) | The tool exists, runs in CI, and its IGNORE vocabulary is the natural extension point |
| GPU/VRAM measurement | Custom nvidia-smi parsing | `torch.cuda.max_memory_allocated()` + `nvidia-smi --query-gpu=` one-liners | Standard evidence the matrix needs |

**Key insight:** every sub-problem in this phase already has a repo-internal precedent (network-skip seam, PDF tmp-path autouse, timeout ladder, allowlist audit, sync script, dispatch-gated GPU job) — the work is assembly and honest wiring, not invention.

## Common Pitfalls

### Pitfall 1: pgrep self-match produces a phantom kernel
**What goes wrong:** `pgrep -f ipykernel_launcher` matches any process whose full cmdline contains the string — including the `bash -c` wrapper of the very CI step or probe shell that runs it (observed live this session: PID 769092, a Claude shell-snapshot wrapper, counted as a "kernel").
**Why it happens:** `-f` matches the whole command line, and heredoc/inline scripts echo their own text into `/proc/<pid>/cmdline`.
**How to avoid:** In pytest the pattern lives in argv of a short-lived pgrep only — safe; assert **delta-zero** (after == before) rather than absolute zero anyway (belt-and-braces for unrelated jupyter servers on the box). In workflow YAML hygiene steps use `ps -eo args | grep "[i]pykernel_launcher"` (bracket trick).
**Warning signs:** kernel count never reaching 0 despite the kill having worked.

### Pitfall 2: `NotebookClient` is not a context manager (milestone-research correction)
**What goes wrong:** Writing `with NotebookClient(nb, ...) as client:` — AttributeError; or hand-rolling try/finally cleanup around `execute()` that double-shuts-down.
**Why:** nbclient 0.10+ moved the context-manager role to `setup_kernel()`; `execute()` self-cleans via its internal async setup.
**How to avoid:** Use the `run_notebook` shape above. If kernel reuse across `execute()` calls is ever needed, that is what `setup_kernel()` is for.
**Warning signs:** `AttributeError: __enter__` in harness unit tests.

### Pitfall 3: The branch-protection PUT replaces contexts
Covered in Pattern 5 — payload must contain both context strings.

### Pitfall 4: `is_fp8_capable()` lies on GB10
**What goes wrong:** dnallm's check is `compute capability >= (9,0)`; GB10 is (12,1) → True → evo2 config auto-selection picks an FP8 config, but upstream FP8 recipes for evo2 large tiers need Hopper, and TE has no GB10 wheels (source-build required, ≤2.12.0).
**Why:** the capability probe conflates "FP8 hardware support" with "evo2's FP8 path works here".
**How to avoid:** the spike treats the dnallm-route attempt as primary evidence; on failure, fall back per D-06 to the `-noFA-noFP8` config variant before any skip. Record both attempts in the matrix.
**Warning signs:** evo2 loading fine but generation output garbage; `Failed to build transformer_engine` in the spike log.

### Pitfall 5: megaDNA unpickle may require the cloned repo
`torch.load(..., weights_only=False)` on `megaDNA_phage_145M.pt` — if the pickle references `model.megaDNA` classes, `import` fails without the repo. Spike resolves; fallback = pinned hash-verified clone or typed skip with the traceback. (Also a security note: `weights_only=False` on a remote artifact is arbitrary-code-execution-by-upstream — the revision pin that Phase 8 adds to `models.lock` is the mitigation.)

### Pitfall 6: "README Local Testing section" does not exist as a heading
**What goes wrong:** The requirement text says correct README "Local Testing" — but today's README has no such heading; the closest artifacts are (a) the install-example line `[VERIFIED: README.md:197 — "uv pip install -e '.[test,cpu]'       # Testing only, no GPU"]` inside the "Dependency Groups" examples block, and (b) the "🧪 Testing" section (line 490) which prescribes `uv run pytest` with NO install line at all. The v1 ledger entries (WR-09 phase-01, WR-07 phase-02) referenced a `.[test,dev]` line that no longer exists verbatim.
**How to avoid:** implement CI-02's README half as: add/correct an install line in the 🧪 Testing section that matches the (post-repair) docs-validation environment — `uv pip install -e '.[test,dev,mcp]'` (or `'.[base]'`, which resolves to dev+test+notebook+mcp `[VERIFIED: pyproject [base] = dnallm[dev,test,notebook,mcp]]`) — and optionally fix the line-197 example comment. Exact wording is an executor/owner checkpoint; the audit criterion is "README install line, run verbatim, makes the documented pytest commands work".

### Pitfall 7: unmasked validators must be green in the SAME unit as the flip
Current measured state (run this session): `validate_yaml.py` exit 0 (21 files pass) — no work; `validate_docs_snippets.py` exit 1 with exactly one error: `docs/example/mcp_pydantic_ai.md:92 (block 5): unterminated string literal (detected at line 5)` — caused by the wrapper generator wrapping a 369-char single-line string from the source notebook cell 6 across two lines (source verified single-line, len=369). Fix = re-emit that block unwrapped (and ideally teach `generate_md_from_notebook.py` not to wrap fenced code) — else the honest gate is born red. Example/YAML test steps: 115 passed, 1 skipped ("No import statements found" — already allowlisted) **with mcp deps present**; they fail today in docs-validation only because `.[test,dev]` lacks the `mcp` extra (the mcp_example notebooks' `langchain`/`pydantic_ai` imports).
**Warning signs:** docs-validation red on the first honest push for reasons unrelated to the mirror.

### Pitfall 8: sandbox gaps for notebooks reading siblings
The pilot notebook reads `./inference_config.yaml` and `./test.csv` from its own directory; the tRNA notebook two YAMLs; the marimo app `./plant_DNA_LLMs_finetune_list.xlsx`. `copytree` of the whole artifact dir covers all of these — do not try to copy only the `.ipynb`. `NOTEBOOK_EXEC_SPECS.extra_inputs` exists for out-of-dir references (none found in the pilot candidates).

### Pitfall 9: kernel interpreter identity
The venv kernelspec `{sys.prefix}/share/jupyter/kernels/python3` uses bare `python` from PATH. Run pytest from the project venv (`.venv/bin/python -m pytest`) or the kernel may launch a different interpreter. `kernel_name="python3"` explicit (as in the shape above) plus venv-activation in CI is the guard.

## Code Examples

### The full pilot test (ready-to-adapt skeleton)
```python
# tests/examples/test_notebook_execution.py
from pathlib import Path
import pytest
from tests.examples._execution import EXAMPLE_DIR, NOTEBOOK_EXEC_SPECS, run_notebook, seed_sandbox

PILOTS = [EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb"]

@pytest.mark.slow
@pytest.mark.parametrize("nb_path", PILOTS, ids=lambda p: str(p.relative_to(EXAMPLE_DIR)))
def test_notebook_executes_end_to_end(nb_path, tmp_path, request):
    spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
    pytest.mark.timeout(spec["test_timeout"])(test_notebook_executes_end_to_end)  # or class-level mark
    sandbox = seed_sandbox(nb_path.parent, tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    nb = run_notebook(nb_path, sandbox, cell_timeout=spec["cell_timeout"], artifact_dir=artifact_dir)
    # invariant assertions (structure, not values):
    code_cells = [c for c in nb.cells if c.cell_type == "code" and "".join(c["source"]).strip()]
    assert all(
        not any(o.get("output_type") == "error" for o in c.get("outputs", []))
        for c in code_cells
    )
    from tests.examples._execution import assert_tree_clean
    assert_tree_clean()
```
Timeout arithmetic for the pilot: cell_timeout 600 < `@pytest.mark.timeout(1800)` < nightly budget — mirrors the existing ladder.

### Verdict matrix skeleton (FEAS-01 deliverable)
```markdown
| Family | Notebook variant | Result | Evidence | Fallback variant | Result | Verdict |
|--------|------------------|--------|----------|------------------|--------|---------|
| evo-1 | togethercomputer/evo-1-131k-base (rev 1.1_fix) | load+fwd OK / FAIL | time __s, VRAM __GB, disk __GB, log: spike_evo1.log | evo-1-8k-base | … | FEASIBLE(notebook-variant) / FEASIBLE(small-variant) / environment-unavailable: <evidence> |
| evo2 | arcinstitute/evo2_1b_base | … | … | evo2_1b_base @ noFA-noFP8 config | … | … |
| megaDNA | lingxusb/megaDNA_updated | … | … | (same, pinned clone) | … | … |
| pyBigWig | 0.3.26 sdist on aarch64 | import+write/read OK / FAIL | build log | — | — | … |
| marimo | export html vs python app.py | both/subset OK | exit codes + artifacts | — | — | flavor: <choice> |
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `with NotebookClient(...)` context-manager pattern (pre-0.10 nbclient lore) | `setup_kernel()` CM + self-cleaning `execute()` | nbclient 0.10+ (installed 0.11.0) | Harness shape in this research supersedes the milestone-research phrasing |
| x86_64-runner assumption for pyBigWig wheels (`platform_system != 'Windows'` marker) | GB10 runner is aarch64 — marker does not protect | Runner class known since v1 | Spike-gated dev-extra addition only |
| evo2 "needs Hopper FP8" blanket verdict (milestone LOW confidence) | Official GB10 thread: source-build recipe works; 7B-only accurate tier; 1B untested there | discussion #221 (fetched 2026-10-02) | Spike has a concrete enabling path instead of a default skip |

**Deprecated/outdated:** none in-repo beyond the false-green being removed this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `evo-1` PyPI package (`pip install evo-1`) provides `from evo import Evo` and installs/works on aarch64 (with stripedhyena) | Pattern 6 | Low — spike resolves empirically; failure text becomes skip evidence; gate flagged the PyPI `evo-1` name as resolving to an old personal-fork project, so the install source may need to be the documented `evo-design/evo` path |
| A2 | megaDNA unpickle needs the cloned repo's classes | Pattern 6 / Pitfall 5 | Low — spike resolves; either way recorded |
| A3 | `zhangtaolab/tRNADetector` + `tRNAPointer` are small downloads (few hundred MB) | Pilot selection | Low — first pilot (`inference/inference.ipynb`) needs zero downloads; tRNA is the optional second |
| A4 | marimo script-mode `python app.py` terminates deterministically (does not start a server) when run headless | Pattern 6 | Low — the flavor spot-check is precisely designed to decide this; `marimo export html` is the verified-execution alternative |
| A5 | `docs-validation` check-run context string is exactly `docs-validation` (no suffix) | Pattern 5 | Medium — if GitHub reports it differently, protection silently waits on a check that never reports; verify with `gh api repos/.../commits/<sha>/check-runs` after the first honest run before/after the owner PUT |
| A6 | README correction shape: add install line to 🧪 Testing section matching `.[test,dev,mcp]`/`.[base]` | Pitfall 6 | Low — wording checkpoint; requirement only demands the line "corrected" |
| A7 | `infer_file(..., evaluate=True)` in the pilot notebook writes nothing outside cwd (and sandbox absorbs cwd writes) | Pilot | None — tmp-sandbox + tree-clean guard make this self-verifying |

## Open Questions

> Resolution status after planning (2026-10-02 revision): every question below carries a recorded
> resolution vehicle — either a Phase 5 plan task that settles it empirically (RESOLVED BY EXECUTION)
> or an explicit deferral to a later phase with its vehicle named.

1. **evo2-1b tier accuracy on GB10** — the upstream thread covers 7B/20B/40B but not 1B; the notebook variant IS 1b_base.
   - What we know: source-build recipe exists; FP8-on-Blackwell accuracy is tier-dependent.
   - What's unclear: whether `evo2_1b_base` (likely bf16 checkpoint) avoids the FP8 issue entirely.
   - Recommendation: spike tries the dnallm route as-is first (auto FP8 config), then the `-noFA-noFP8` config — both attempts recorded.
   - **(RESOLVED BY EXECUTION: 05-03 Task 2 — spike_evo2 runs the auto-selected config then the noFA-noFP8 fallback per D-06, both attempts' evidence recorded in 05-FEASIBILITY.md.)**
2. **Does `python app.py` (marimo) block or exit?** — flavor spot-check decides; if it blocks, script-mode needs a timeout+kill wrapper or export-html wins.
   - **(RESOLVED BY EXECUTION: 05-03 Task 2 — spike_marimo runs both flavors with subprocess timeout, recording exit codes, wall time, port binding, and artifacts in the matrix; the flavor decision is a matrix deliverable.)**
3. **Spike workflow placement** — `ci.yml` dispatch-gated job vs separate `feasibility.yml`. Owner taste; dispatch-only either way.
   - **(RESOLVED BY PLANNING: 05-03 Task 3 commits a separate dispatch-only `.github/workflows/feasibility.yml` cloned from the test-mamba pattern — keeps the runner's PR-unreachable posture independent of ci.yml edits.)**
4. **evo-1 download size** — 29.7GB repo / ~12.9GB safetensors filtered; whether `allow_patterns` filtering applies to the `1.1_fix` revision path dnallm uses is a Phase 8 (CI-05) detail; the Phase 5 spike just needs disk+time recorded (2.4TB free verified).
   - **(Phase 5 half RESOLVED BY EXECUTION: 05-03 Task 2 records measured disk_gb (du -sh) and load/download times in the matrix. The `allow_patterns`-on-`1.1_fix` sub-question is DEFERRED to Phase 8 (CI-05 models.lock work) — out of Phase 5 scope per the phase boundary in 05-CONTEXT.md.)**

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| GB10 GPU (same class as runner) | Spike, D-04 parity | ✓ | NVIDIA GB10, CC (12,1), driver 580.178.04, torch 2.11.0+cu130, CUDA available | — |
| Free disk | Spike model downloads | ✓ | 2.4 TB free on / | — |
| nbclient / nbformat / ipykernel | Harness | ✓ | 0.11.0 / 8.1.x / venv kernelspec present | — |
| pytest-timeout marks | Timeout layering | ✓ | 2.4.0 (pinned <2.5) | — |
| `pgrep`/`pkill` | Kill test + hygiene | ✓ | /usr/bin/pgrep, pkill | — |
| `gh` CLI (authed) | Branch protection verify, workflow dispatch | ✓ | github.com account forrestzhang, active | Owner runs the PUT themselves (intended) |
| Network to PyPI/HF | Spike installs + downloads | ✓ | PyPI JSON fetched this session | — |
| HF cache: evo-1/evo2/megaDNA | Spike cold-start | ✗ not cached | — | Expected: spike downloads (evo-1 ~13-30GB, evo2_1b ~2-3GB, megaDNA_updated ~600MB [ASSUMED sizes]) |
| ModelScope cache (zhangtaolab warm set) | Pilot notebook model | ✓ | plant-dnagpt-BPE-promoter present | — |
| pyBigWig aarch64 wheel | FEAS-01 pyBigWig entry | ✗ (x86_64-only wheels; sdist path) | — | sdist compile attempt = the spike itself; typed skip w/ evidence if it fails |
| gcc / zlib+libcurl headers | pyBigWig sdist build | unchecked | — | `apt`/brew install or record failure as evidence |

**Missing dependencies with no fallback:** none blocking.
**Missing dependencies with fallback:** pyBigWig aarch64 wheels (fallback = sdist build attempt, then evidence-backed skip); HF-cached spike models (fallback = download; disk verified).

## Security Domain

> `security_enforcement: true`, ASVS level 1, block_on high. This phase touches test infrastructure, CI gate honesty, and supply-chain-adjacent spikes.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surfaces added |
| V3 Session Management | no | — |
| V4 Access Control | yes (indirect) | Branch-protection required checks = merge access control on dev+main; the PUT payload must not weaken `enforce_admins`/reviews relative to v1 settings |
| V5 Input Validation | yes | `nbformat.read` validates notebook JSON; `check_docs_sync.py` fail-closed on absent dirs; `audit_skips.py` fail-closed on unparseable junit (existing) |
| V6 Cryptography | no | — |
| V14 Config | yes | `continue-on-error` removal = removing a config-level integrity bypass; spike venv isolated from the project env |

### Known Threat Patterns for this phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| False-green gate masking tampered docs mirror | Tampering/Elevation of integrity | WR-08 flip + byte-identical mirror + required-check promotion (this phase's core) |
| `trust_remote_code` model fetch executes upstream code on the runner | Elevation | Spike records the evo-1 `1.1_fix` revision; Phase 8 adds revision pins to `models.lock`; runner stays schedule/dispatch-only (PR code never reaches it) |
| megaDNA `weights_only=False` unpickle of remote artifact | Tampering/Elevation | Spike uses the pinned HF revision; any Phase 8 enablement must pin + hash-verify; never execute the notebook's `git clone && pip install -e .` unpinned |
| Spike pip installs polluting/mutating the shared dev or runner env | Tampering (integrity of test env) | Throwaway `/tmp` venv locally; dispatch job's ephemeral `.venv` on the runner; nothing enters pyproject without the legitimacy checkpoint |
| Third-party aarch64 wheel indexes (RISE) | Supply chain | Rejected in-research; official sdist or evidence-backed skip only |
| Kernel subprocesses on a persistent runner | Resource abuse / DoS | `shutdown_kernel="immediate"` + delta-zero kill test + (Phase 9) nightly pkill hygiene — designed this phase, wired later |

## Sources

### Primary (HIGH confidence — this session)
- Empirical probes on this GB10 box (2026-10-02): deliberate-hang probe (`timeout=3` → `CellTimeoutError` @3.5s, immediate shutdown, delta-zero kernels, partial-artifact outputs, exception message with cell source); `torch.cuda.get_device_capability() == (12, 1)`; `pytest tests/examples/ + test_yaml_load.py` → 115 passed 1 skipped; `python scripts/check_docs_sync.py` → exit 1, 36-line inventory; `python scripts/validate_yaml.py` → exit 0; `python scripts/validate_docs_snippets.py` → exit 1 (one error); nbclient introspection (`NotebookClient` traits, `setup_kernel` source, `_get_timeout`, `_async_cleanup_kernel`, `resources.metadata.path` at client.py:535); marimo 0.25.0 CLI help; HF/ModelScope cache listings; `nvidia-smi`; `gh auth status`
- Repo reads: `scripts/check_docs_sync.py` (full), `.github/workflows/docs-validation.yml` (full), `.github/workflows/ci.yml` (full), `tests/examples/test_examples.py` (full), `dnallm/mcp/tests/_network_skip.py` (full), `tests/expected_skips.yaml` (full), `scripts/audit_skips.py` (full), `conftest.py` (root, full), `tests/inference/test_plot.py:40-171`, `dnallm/models/special/evo.py` (full), `dnallm/models/special/megadna.py` (full), `dnallm/utils/support.py` (full), `dnallm/configuration/evo/` listing, `models.lock` (full), `pyproject.toml` (extras + pytest ini), `README.md` (headings, 190-198, 488-505), example notebook model-reference extraction (evo/megaDNA/tRNA/inference/generation/lora), marimo app defaults and `__main__` blocks
- v1 archives: `.planning/milestones/v1-phases/04-ci-gate-enforcement/04-03-SUMMARY.md:163-185` (branch-protection PUT), `04-UAT.md:19-21` (verification method), `v1-phases/02-suite-hygiene-known-bug-fixes/02-02-SUMMARY.md` + `02-VERIFICATION.md` (PDF autouse pattern, twice-run proof, sys.modules lesson), `v1-MILESTONE-AUDIT.md` (WR-08/09 lineage)
- PyPI JSON API (fetched 2026-10-02): pyBigWig 0.3.26 file list (x86_64-only wheels + sdist)

### Secondary (MEDIUM confidence)
- [ArcInstitute/evo2 discussion #221 — "Evo 2 on DGX Spark (GB10)"](https://github.com/ArcInstitute/evo2/discussions/221) — official-repo thread: source-build recipe (flash-attn 2.8.0 + TE 2.12.0, arch 120), 7B-only-accurate, 40B OOM, ~10x-vs-H100. Single-thread source; the spike independently re-verifies everything that matters
- [pyBigWig on PyPI](https://pypi.org/project/pyBigWig) + websearch corroboration (RISE aarch64 wheels exist third-party; pybbi ships aarch64 wheels) — [pybbi on PyPI](https://pypi.org/project/pybbi)
- [evo2 on PyPI](https://pypi.org/project/evo2) install-order guidance (TE + flash-attn first)
- Milestone research (`.planning/research/{SUMMARY,STACK,ARCHITECTURE,PITFALLS}.md`, 2026-10-01) — every claim re-checked this session; one correction recorded (Pattern 2)

### Tertiary (LOW confidence)
- Spike model download sizes (evo2_1b ~2-3GB, megaDNA ~600MB) — [ASSUMED], replaced by measured `du -sh` at spike time
- tRNA model sizes — [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — installed versions verified; pyproject delta minimal and gated
- Harness architecture: HIGH — API verified from installed source AND live-probed on the target hardware class
- Gate-repair mechanics: HIGH — every step's current state measured this session (exact exit codes, exact line numbers, full drift inventory)
- Feasibility spike: MEDIUM — environment facts (GB10, caches, wheels, handler code paths) HIGH; family outcomes intentionally unresolved until the spike runs (upstream GB10 thread is MEDIUM corroboration)
- Pitfalls: HIGH — pgrep self-match and FP8-trap observed/derived from read code this session

**Research date:** 2026-10-02
**Valid until:** 2026-11-01 (stack facts stable; drift inventory valid only until the resync commit lands)
