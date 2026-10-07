# Phase 5: Execution Harness, Honest Gates & Runner Feasibility - Pattern Map

**Mapped:** 2026-10-02
**Files analyzed:** 13 (6 new, 7 modified)
**Analogs found:** 12 / 13 (1 planning artifact needs no code analog)

All analog paths below verified git-tracked via `git ls-files` (repo-root tracked source; no capability-sync mirrors involved).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tests/examples/_execution.py` (NEW) | test utility (private harness helper) | process-execution + file-I/O | `dnallm/mcp/tests/_network_skip.py` | exact (research-locked seam mirror) |
| `tests/examples/conftest.py` (NEW) | test config (locally-scoped fixtures) | file-I/O (sandbox seeding) | `tests/conftest.py` + `tests/inference/test_plot.py:168-171` | role-match |
| `tests/examples/test_notebook_execution.py` (NEW) | test (parametrized execution + kill test) | batch execution | `tests/examples/test_examples.py` | exact (same dir, same parametrize shape) |
| `scripts/feasibility/spike_<family>.py` (NEW, 1-per-family or single runner) | utility script | batch (model load + forward + measure) | `scripts/check_docs_sync.py` / `scripts/audit_skips.py` (script skeleton) + `tests/inference/test_inference_real_model.py:34-54` (real-model load) | role-match |
| `.github/workflows/feasibility.yml` (NEW) or `feas-spike` job in `ci.yml` | CI config (dispatch-gated) | — | `.github/workflows/ci.yml:262-341` (`test-mamba`) | exact (research says clone it) |
| `05-FEASIBILITY.md` verdict matrix (NEW, phase dir) | planning doc | — | none (skeleton in 05-RESEARCH.md "Code Examples") | none needed |
| `.github/workflows/docs-validation.yml` (MOD) | CI config | — | itself + `ci.yml:322-332` (honest-step precedent) | exact (self) |
| `scripts/check_docs_sync.py` (MOD) | utility (tree diff) | file-I/O | itself (`:8-16` IGNORE, `:32-47` branch loops) | exact (self) |
| `docs/example/**` resync (MOD: 10 DIFFER files + 1 missing script + `mcp_pydantic_ai.md` block fix) | data mirror | file copy | itself + repo precedent commit `6ddc036` | exact (self) |
| `scripts/generate_md_from_notebook.py` (MOD, optional root-cause fix) | utility (transform) | transform | itself (`:119-133` code-fence emission) | exact (self) |
| `README.md` (MOD) | docs | — | itself (`:490-505` Testing section, `:193-198` install examples) | exact (self) |
| `pyproject.toml` (MOD) | config | — | itself (`:80-127` extras, `:464-487` pytest ini) | exact (self) |
| `tests/expected_skips.yaml` (MOD) | test config (skip allowlist) | — | itself + `scripts/audit_skips.py:49-63` (matcher contract) | exact (self) |

Non-repo deliverable (owner-run, record runnable commands in the plan): branch-protection `gh api` PUT for dev+main — exact payload in 05-RESEARCH.md Pattern 5 (contexts array must contain BOTH `coverage-gate (py3.12, fast leg)` and `docs-validation`; the PUT replaces the array).

## Pattern Assignments

### `tests/examples/_execution.py` (test utility, process-execution + file-I/O)

**Analog:** `dnallm/mcp/tests/_network_skip.py` — the research-locked private-helper seam: a `_`-prefixed module living beside its consumer tests, never imported outside its test tree, never shipped.

**Private-helper conventions** (`dnallm/mcp/tests/_network_skip.py:1-17`):
```python
"""Typed network-skip helper for the MCP live-server tests.

The MCP clients fail through httpx inside an anyio TaskGroup, so the
catchable exception is an ``ExceptionGroup`` wrapping an
...
"""

import httpx
import pytest

NETWORK_ERRORS = (httpx.TransportError,)
```
Copy: module docstring explaining WHY the module exists (several paragraphs), stdlib-then-third-party import block, module-level constants in UPPER_SNAKE_CASE, Google-style `Args:`/`Returns:`/`Raises:` docstrings on every function, type hints on all signatures.

**Typed-skip emission shape** (`_network_skip.py:58-62`) — the `NOTEBOOK_EXEC_SPECS`-driven harness must emit skip messages the same way (stable prefix, deterministic, junit-greppable):
```python
    leaves = _network_leaves(exc)
    if leaves and all(isinstance(leaf, NETWORK_ERRORS) for leaf in leaves):
        message = f"network-unavailable: {action} (no server reachable: {type(leaves[0]).__name__})"
        pytest.skip(message)
    raise exc
```
Phase-5 prefixes: `environment-unavailable:` and `optional-dep:` (D-06; both must land in `expected_skips.yaml` the same unit).

**Directory anchor convention** (`tests/examples/test_examples.py:18`):
```python
EXAMPLE_DIR = Path(__file__).parent.parent.parent / "example"
```
`_execution.py` re-exports/defines `EXAMPLE_DIR` identically so parametrize ids and `seed_sandbox` agree with the existing structural layer. Also define `REPO_ROOT = Path(__file__).parent.parent.parent` for `assert_tree_clean`'s `git status --porcelain -- example docs/example` call.

**Core API to implement** (shape verified live on GB10 — 05-RESEARCH.md Pattern 1 is the authoritative skeleton, copy it): `NOTEBOOK_EXEC_SPECS: dict[str, dict]`, `seed_sandbox(src_dir, tmp_path, extra_inputs) -> Path` (whole-dir `shutil.copytree` with `ignore_patterns(".ipynb_checkpoints", "__pycache__", "outputs*", "results*", "*.gz")` — Pitfall 8: pilot reads sibling `./inference_config.yaml` + `./test.csv`), `run_notebook(nb_path, sandbox, cell_timeout, artifact_dir) -> NotebookNode`, `assert_tree_clean(paths=("example", "docs/example")) -> None`. CRITICAL anti-pattern (RESEARCH Pitfall 2): `NotebookClient` is NOT a context manager in nbclient 0.11.0 — use plain `NotebookClient(...).execute()`; do NOT hand-roll try/finally kernel cleanup around it.

---

### `tests/examples/conftest.py` (test config, file-I/O)

**Analog 1:** `tests/conftest.py` — fixture style: plain `@pytest.fixture` functions, one-line docstring starting with "Return ..." (see `tests/conftest.py:167-182`). This file is LOCALLY scoped: only sandbox/kernel fixtures go here; the shared mock fixtures stay in `tests/conftest.py` and remain usable by `tests/examples/` via normal pytest conftest inheritance.

**Analog 2 — the autouse tmp-redirect precedent (v1 Phase-2 pattern to generalize):** `tests/inference/test_plot.py:168-171`:
```python
@pytest.fixture(autouse=True)
def pdf_output_dir(tmp_path, monkeypatch):
    """Write all PDF artifacts under tmp_path so the repo working tree stays clean."""
    monkeypatch.setattr(sys.modules[__name__], "PDF_OUTPUT_DIR", tmp_path)
```
The notebook-sandbox fixture follows the same shape but redirects via kernel cwd (`resources={"metadata": {"path": str(sandbox)}}`) instead of `sys.modules` rebind — RESEARCH notes the `sys.modules[__name__]` trap does not apply to notebooks because notebooks resolve `./` paths against the kernel cwd.

Fixture to provide (name per planner; mechanics locked): wraps `_execution.seed_sandbox` over `tmp_path`, yields the sandbox path, and (belt-and-braces) runs `assert_tree_clean` teardown. Note `tests/examples/` has NO conftest today and NO `__init__.py` anywhere under `tests/` — keep it that way.

---

### `tests/examples/test_notebook_execution.py` (test, batch execution)

**Analog:** `tests/examples/test_examples.py` — sits in the same directory, untouched by this phase; copy its collection/parametrize conventions exactly so junit ids stay uniform for `audit_skips.py`.

**Parametrize id convention** (`tests/examples/test_examples.py:158-163`):
```python
    @pytest.mark.skipif(not NOTEBOOK_FILES, reason="No notebook files found")
    @pytest.mark.parametrize(
        "nb_file",
        NOTEBOOK_FILES,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
```
Use `ids=lambda p: str(p.relative_to(EXAMPLE_DIR))` for the pilot parametrization (RESEARCH: greppable junit names).

**Test-class grouping convention** (`test_examples.py:155`): `class TestNotebookExamples:` with one class per concern — e.g. `class TestNotebookExecution:` (pilot) and `class TestKernelLifecycle:` (deliberate-hang kill test). Box-drawing section separators (`# ────...────`, `test_examples.py:152-154`) optional in larger files.

**Timeout-mark + slow-mark ladder** (outer backstop; research-locked). Precedents:
`tests/models/test_model.py:178-180`:
```python
    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_download_real_huggingface_connection(self):
```
`tests/inference/test_inference_real_model.py:22-24` (class-level mark):
```python
@pytest.mark.slow
@pytest.mark.timeout(1800)
class TestRealModelInference(unittest.TestCase):
```
Pilot arithmetic (RESEARCH): cell_timeout 600 < `@pytest.mark.timeout(1800)` < nightly job budget. Kill test: `@pytest.mark.slow` + `@pytest.mark.timeout(120)` (default slow; fast-leg placement is a planner option). Do NOT register new markers — reuse `slow` (`pyproject.toml:478`); `timeout` needs no marker registration.

**Assertions style:** plain `assert` with message strings (`test_examples.py:172-184`) and `pytest.fail(f"...")` for collected-error lists (`test_examples.py:218-219`); `pytest.raises(CellTimeoutError)` for the kill test; delta-zero kernel-count assertion with pgrep inside `subprocess.run` (NEVER inside a CI bash step — RESEARCH Pitfall 1). Kill-test body skeleton is probe-verified: 05-RESEARCH.md Pattern 3, copy it. Generate the hang-fixture notebook in-test via `nbformat.v4.new_notebook/new_code_cell` — do NOT commit a `hang.ipynb` under `example/` (rglob collection trap).

---

### `scripts/feasibility/spike_<family>.py` (utility script, batch model-execution)

**Analog 1 — repo script skeleton:** `scripts/check_docs_sync.py:1-8, 56, 77-78`:
```python
#!/usr/bin/env python3
"""Verify docs/example/ is a byte-identical mirror of example/ (except runtime artifacts)."""

import filecmp
import sys
from pathlib import Path
...
def main() -> int:
    ...
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
(Same shape in `scripts/audit_skips.py` and `scripts/validate_docs_snippets.py`.) Copy: shebang, one-line module docstring, `main() -> int` returning exit codes 0/1, `sys.exit(main())` guard, `print()` to stdout (scripts may print; T20 applies only to library code). Do NOT copy `scripts/inference_mamba2_npu.py` / `megatron.py` style — those are vendored/vendor-style and excluded from lint/mypy.

**Analog 2 — real-model load shape:** `tests/inference/test_inference_real_model.py:34-54` — function-local imports of `load_config`/`DNAInference`-adjacent loaders, load-once-then-measure structure, `zhangtaolab/plant-dnagpt-BPE-promoter` (the warm-cache model family) as the reference for "load a real model from config". The spike's per-family bodies are given verbatim in 05-RESEARCH.md Pattern 6 (`spike_evo1.py` shape: `load_model_and_tokenizer(...)` → `time.time()` deltas → `torch.cuda.max_memory_allocated()` → print evidence lines) — research is authoritative here; use analog 1 for the file skeleton and analog 2 for import placement (imports inside functions, absolute `from dnallm...`).

Evidence contract (D-05): every run prints install steps/versions, load seconds, forward seconds, peak VRAM, disk footprint (`du -sh`), or the exact failure text (failure text becomes the typed-skip evidence).

---

### `.github/workflows/feasibility.yml` (or `feas-spike` job in `ci.yml`) (CI config, dispatch-gated)

**Analog:** `.github/workflows/ci.yml:262-279` — the `test-mamba` job (research-locked pattern to clone):
```yaml
  test-mamba:
    # Self-hosted GPU box on nightly cadence per GATE-02 (amended): ...
    # schedule/dispatch-only preserves the coverage-nightly posture that PR code
    # (including forks) never executes on this runner. ...
    runs-on: [self-hosted, dnallm-nightly]
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    timeout-minutes: 180
```
Copy from `ci.yml:262-341`: `runs-on: [self-hosted, dnallm-nightly]`, event gate (Phase 5 tightens to `workflow_dispatch` only per D-04), explanatory comment block above the job (this repo documents WHY self-hosted jobs are event-gated — keep that discipline), `timeout-minutes` backstop (research says 240), `uv venv` + `uv pip install -e ".[base]"`, and `actions/upload-artifact@v4` with `if: always()` (`ci.yml:334-341` is the upload-on-failure precedent; spike uses `if: always()` unconditionally per RESEARCH Pattern 6). Trigger: `gh workflow run` (owner; manual-push-only milestone — report unpushed range).

---

### `.github/workflows/docs-validation.yml` (MODIFY — CI config)

**Analog:** itself. Five steps carry `continue-on-error: true` at `docs-validation.yml:44-72` (lines 45, 51, 57, 63, 69): Check docs/example/ sync, Validate docs code snippets, Validate YAML configs, Run example tests, Run YAML load tests. Delete each `continue-on-error: true` line; leave step bodies unchanged.

**WR-09 install fix** (`docs-validation.yml:36-42`):
```yaml
      - name: Create virtual environment and install dependencies
        env:
          UV_HTTP_TIMEOUT: 300
          UV_CONCURRENT_DOWNLOADS: 4
        run: |
          uv venv
          uv pip install -e ".[test,dev]"
```
Change `.[test,dev]` → `.[test,dev,mcp]` (research-verified: example tests need the mcp extra's langchain/pydantic-ai imports; 115 passed locally with it).

**Honest-step precedent** (why no masking comment is needed): `ci.yml:322-325` — the test-mamba test step carries the comment "No continue-on-error here: a failing mamba test must fail the job." Mirror that commenting style on the flipped steps if a rationale note is warranted.

**Check-name invariant (D-02/A5):** `docs-validation.yml:13-14` — `docs-validation: / name: docs-validation`. The job's `name:` IS the branch-protection context string; do not rename it.

---

### `scripts/check_docs_sync.py` (MODIFY — utility, file-I/O diff)

**Analog:** itself. Current ignore vocabulary (`check_docs_sync.py:8-16`):
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
The branch to relax (`check_docs_sync.py:38-44`):
```python
    for name in dcmp.right_only:
        if not _should_ignore(name):
            errors.append(
                f"ONLY in docs/example/: {path}/{name}"
                if path
                else f"ONLY in docs/example/: {name}"
            )
```
Minimal fix (research-locked scope): add a `DOCS_ONLY_SUFFIXES = (".md",)` consulted ONLY in the `right_only` loop — docs-only wrapper `.md` files (24 today) are intentional. Do NOT touch `left_only` (`:32-36`) or `diff_files` (`:46-47`) — those must stay strict (a missing/differing legitimately-mirrored `.md` must still fail; `example/notebooks/overview.md` exists on both sides). Keep the existing comment style (short inline comments; script docstring already states the byte-identity contract).

---

### `docs/example/**` resync + `mcp_pydantic_ai.md` fix (MODIFY — data mirror, file copy)

**Analog:** itself + repo precedent commit `6ddc036` ("fix(docs): align 10 markdown tutorial code blocks with source notebooks") — same class of repair as the `mcp_pydantic_ai.md:92` block fix. Neither `scripts/generate_md_from_notebook.py` nor `generate_md_from_marimo.py` contains any wrap/textwrap logic (verified by grep), so the mid-string line wrap at `mcp_pydantic_ai.md:92-93` is a historical artifact; fix by re-emitting that block from the source notebook cell (single-line 369-char string, `example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb` cell 6).

Exact copy manifest (research-measured, valid until the resync commit lands): 10 DIFFER files copied `example/` → `docs/example/` (`marimo/finetune/finetune_demo.py`, `marimo/inference/inference_demo.py`, `mcp_example/mcp_client_ollama_langchain_agents.ipynb`, `mcp_example/mcp_client_ollama_pydantic_ai.ipynb`, `notebooks/benchmark/benchmark.ipynb`, `notebooks/data_prepare/finetune/finetune_data.ipynb`, `notebooks/finetune_NER_task/data_generation_and_inference.ipynb`, `notebooks/finetune_NER_task/finetune_NER_task.ipynb`, `notebooks/finetune_binary/finetune_binary.ipynb`, `notebooks/finetune_multi_labels/finetune_multi_labels.ipynb`), 1 file copied in (`notebooks/finetune_NER_task/generate_bpe_dataset.py` → `docs/example/notebooks/finetune_NER_task/`). Byte-identical copies include stale outputs (D-03) — do not "clean" anything. Verification: `python scripts/check_docs_sync.py` exits 0.

---

### `scripts/generate_md_from_notebook.py` (MODIFY, optional root-cause fix — utility, transform)

**Analog:** itself, code-fence emission at `generate_md_from_notebook.py:119-131`:
```python
    for kind, content in merged:
        if kind == "text":
            cleaned = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
            if cleaned.strip():
                lines.append(cleaned)
                lines.append("")
        elif kind == "code":
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
```
The emission path appends code verbatim (correct). If the planner elects the optional hardening (RESEARCH Pitfall 7: "ideally teach generate_md_from_notebook.py not to wrap fenced code"), the invariant to assert is exactly this: fenced-cell content lands in the output unmodified — a regression test would regenerate from a notebook with a >100-char single-line string and `assert` the line survives unwrapped. This fix is optional; the mandatory fix is the `mcp_pydantic_ai.md` block repair above.

---

### `README.md` (MODIFY — docs)

**Analog:** itself. Target sections:
- `README.md:490-505` (🧪 Testing): currently prescribes `uv run pytest` with NO install line. Add an install line matching the post-repair docs-validation env: `uv pip install -e '.[test,dev,mcp]'` (or `'.[base]'`, which resolves to dev+test+notebook+mcp — RESEARCH Pitfall 6). Audit criterion: run the line verbatim → the documented pytest commands work.
- `README.md:193-198` (install examples block): optionally correct the `uv pip install -e '.[test,cpu]'  # Testing only, no GPU` comment line to mention mcp deps for example tests.
There is no "Local Testing" heading today (RESEARCH Pitfall 6) — do not search for one; the 🧪 Testing section is the canonical location.

---

### `pyproject.toml` (MODIFY — config)

**Analog:** itself. Extras block at `pyproject.toml:100-103`:
```toml
notebook = [
    "jupyter>=1.1.1",
    "marimo>=0.16.3",
]
```
Delta: add `"nbclient>=0.10"` to `notebook` (makes the today-transitive harness import explicit — the ONLY committed dependency change, RESEARCH "Standard Stack"). Conditional second delta: `"pyBigWig>=0.3.26; platform_system != 'Windows'"` added to `dev` (`pyproject.toml:81-91`) ONLY if the GB10 sdist build succeeds — gate behind `checkpoint:human-verify` (legitimacy audit SUS verdict). Marker caveat from RESEARCH: the platform marker does NOT encode aarch64; the spike evidence is the real gate.

Pytest ini (`pyproject.toml:464-487`): NO changes expected — `slow` marker already registered (`:478`), `--timeout=300` default already in addopts (`:475`), `--strict-markers` (`:472`) means any new marker would fail collection (there must be none).

---

### `tests/expected_skips.yaml` (MODIFY — test config, skip allowlist)

**Analog:** itself — the prefix-entry shape at `tests/expected_skips.yaml:28-30`:
```yaml
  - prefix: "network-unavailable:"
    category: network
```
Add sibling entries for the new prefixes in the same unit as any harness skip emission:
```yaml
  - prefix: "environment-unavailable:"
    category: environment
  - prefix: "optional-dep:"
    category: optional-dep
```
Matcher contract (enforced by `scripts/audit_skips.py:37-45`): every entry needs exactly one non-empty matcher key (`exact`/`prefix`/`reason_like`) plus a `category`; empty/wildcard entries fail the audit (`audit_skips.py:44-45`). `prefix` semantics = `message.startswith(...)` (`audit_skips.py:61-62`). Keep the per-entry comment style (each entry carries a comment naming its source file/line — see existing entries at `expected_skips.yaml:16-17, 26-28, 37-39`).

---

### `05-FEASIBILITY.md` verdict matrix (NEW — planning artifact)

No codebase analog. Use the skeleton in 05-RESEARCH.md "Code Examples" (family × notebook-variant × result × evidence × fallback × verdict table). Location (Claude's discretion, RESEARCH suggestion): `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md`, linked from phase verification.

## Shared Patterns

### Typed-skip discipline (apply to: `_execution.py`, `test_notebook_execution.py`, `expected_skips.yaml`)
**Source:** `dnallm/mcp/tests/_network_skip.py:43-62` + `scripts/audit_skips.py:96-112`
A skip is emitted only through a helper that builds `f"<stable-prefix>: {action} ({evidence})"` and calls `pytest.skip(message)`; the prefix lands in `expected_skips.yaml` as a `prefix:` entry in the same commit; the out-of-process audit (`audit_skips.py <junit> tests/expected_skips.yaml`, run at `ci.yml:93-96` and `:398-401`) fails CI on any unlisted skip. Non-qualifying exceptions are re-raised (honest failure), never converted to skips.

### Timeout ladder: per-cell inner, per-test-mark outer (apply to: all execution tests)
**Source:** `pyproject.toml:475` (`--timeout=300` default) + `tests/models/test_model.py:178-180`, `tests/inference/test_inference_real_model.py:22-24`, ladder 900/1800/3600/7200
Stack strictly: nbclient `timeout=` (per cell) < `@pytest.mark.timeout(N)` (per test) < job `timeout-minutes`. Existing marks override the 300s ini default; new marks follow the same decorator order `@pytest.mark.slow` then `@pytest.mark.timeout(N)`, stacked above the function/class.

### Sandbox isolation + tree-clean tripwire (apply to: `conftest.py`, `_execution.py`)
**Source:** `tests/inference/test_plot.py:168-171` (autouse tmp-redirect, v1 Phase-2 precedent)
All execution writes go under `tmp_path` (kernel cwd = sandbox copy via `resources={"metadata": {"path": ...}}`); after each test, `assert_tree_clean(("example", "docs/example"))` runs `git status --porcelain -- <paths>` and fails on any output (scoped paths only — untracked `.planning/`/`.gsd/` are none of the harness's business). Verification pattern from v1: run the pilot twice, tree clean both times.

### Repo script skeleton (apply to: spike scripts, any new `scripts/` file)
**Source:** `scripts/check_docs_sync.py`, `scripts/audit_skips.py`, `scripts/validate_docs_snippets.py`
`#!/usr/bin/env python3` + one-line docstring + stdlib/third-party imports + pure functions + `main() -> int` (fail-closed: absent inputs → return 1) + `if __name__ == "__main__": sys.exit(main())`.

### Self-hosted dispatch-only job (apply to: feasibility workflow)
**Source:** `.github/workflows/ci.yml:262-341`
`runs-on: [self-hosted, dnallm-nightly]` + `if: github.event_name == ...` gate + rationale comment block + `timeout-minutes` backstop + `uv venv`/`uv pip install -e ".[base]"` + artifact upload. Invariant: PR-authored code never reaches the runner.

### Honest CI steps (apply to: docs-validation flips, feasibility job)
**Source:** `.github/workflows/ci.yml:322-332` — no `continue-on-error` anywhere on enforcement steps; failures surface and fail the job. WR-08 is the deletion of the five violations of this pattern in `docs-validation.yml`.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `05-FEASIBILITY.md` verdict matrix | planning doc | — | Documentation artifact; skeleton supplied by 05-RESEARCH.md |

Everything else has an in-repo analog (several are self-modifications where the analog is the file's own current content, with exact line citations above). Note on nbclient itself: no repo code imports nbclient today — the invocation shape comes from the probe-verified skeleton in 05-RESEARCH.md Pattern 1 (treat RESEARCH as the source of truth for the nbclient API surface; the codebase supplies only the surrounding test/script conventions).

## Metadata

**Analog search scope:** `tests/` (incl. `tests/examples/`, `tests/conftest.py`, root `conftest.py`), `dnallm/mcp/tests/`, `scripts/`, `.github/workflows/`, `pyproject.toml`, `README.md`, `docs/example/`
**Files scanned:** 14 analog files read in full or targeted (all ≤ 550 lines except `tests/inference/test_plot.py` and `tests/models/test_model.py`, read targeted); `git ls-files` tracked-status check passed on every named analog
**Pattern extraction date:** 2026-10-02
