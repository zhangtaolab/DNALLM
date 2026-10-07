---
phase: 05-execution-harness-honest-gates-runner-feasibility
reviewed: 2026-10-06T14:51:10Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - docs/example/mcp_pydantic_ai.md
  - .github/workflows/README.md
  - .gitignore
  - pyproject.toml
  - README.md
  - scripts/check_docs_sync.py
  - tests/examples/_execution.py
  - tests/examples/test_marimo_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_script_execution.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 5: Code Review Report (incremental re-review at HEAD)

**Reviewed:** 2026-10-06T14:51:10Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Incremental re-review of the 13 phase-05-scope files that changed since the
last reviewed delta base `a943411` (changes landed by later phases / quick
tasks: rice-cache seeding, isolated megaDNA/evo kernel lanes, spec `env` /
`yaml_patch` seams, ollama retry-window probe, model-id swap to
`qwen3.5:4b`, ruff 0.16.10 bump, runner README/docs updates, transformers
absence-contract tests).

The delta is substantially sound. Verified live against the working tree:

- **Mirror sync**: the `qwen3.5:4b` swap landed in BOTH `example/mcp_example/`
  twins and their `docs/example/` mirrors (same blob hashes); `scripts/check_docs_sync.py`
  and `scripts/check_notebook_md_sync.py` both pass; the new IGNORE entries
  (`benchmark_results`, `.scratch`, `.pdf`) correspond to genuinely untracked
  paths (`git ls-files` confirms no tracked `.pdf` / `benchmark_results`), so
  no mirror drift can be masked.
- **Budget contracts**: every `cell_timeout` in `NOTEBOOK_EXEC_SPECS` stays
  strictly below its outer pytest-timeout mark (3600-cell entries — mcp pair,
  finetune_custom_head, finetune_generation, lora_finetune — all carry the
  7200s override via `_TIMEOUT_7200_GATED`; evo 1800 < class 3600; lora_inference
  1800 < 3600). The new `giants` marker is registered in `pyproject.toml`
  markers, matching the `-m "not giants"` CI deselection.
- **Typed-skip honesty**: the new gates (`_gate_evo`, `_gate_megadna_isolated`)
  use the registered `optional-dep:` prefix; the D-13 retry window returns
  verbatim per-attempt evidence and never sleeps after the last attempt.
- **pydantic-ai API claims**: `from pydantic_ai.mcp import FastMCPClient,
  MCPToolset` and attribute-style `result.usage` are valid against the
  installed pydantic-ai 1.102.0 (`usage` is a `_DeprecatedCallableProperty`;
  attribute access works, call-style is what is deprecated). The notebook and
  the `.md` wrapper agree on the new API shape.
- **Deps**: venv resolves cleanly under the new pins (`ipython 8.39.0`
  satisfies `>=8.31,<9`; `pip check` clean apart from a pre-existing cu13
  platform marker note); ruff 0.16.10 installed.
- **Tests executed**: `tests/utils/test_transformers_compat.py::TestTransformersAbsenceContract`
  (12 passed) and the fast lane of the three example modules (44 passed,
  27 deselected as slow).
- **Workflows README accuracy**: cross-checked against `.github/workflows/ci.yml`
  (dual cron + `github.event.schedule` gates, `uvx ty@0.0.84 ... || true`
  advisory step, bedtools rootless step in coverage-nightly, 900/2700-minute
  timeouts, models.lock hub-cache removal) — all claims match.

No critical issues found. Two warnings and two info items below.

## Warnings

### WR-01: Model-id swap incomplete — Prerequisites still pulls `qwen3.6:latest`

**File:** `docs/example/mcp_pydantic_ai.md:21`
**Issue:** The delta swapped the tutorial's model to `qwen3.5:4b` in the
`OpenAIChatModel` code block (line 45) and in both mirrored notebooks, but the
Prerequisites section of the same file still instructs `ollama pull
qwen3.6:latest` — a third model id that appears in no notebook. A user
following the tutorial pulls the wrong model and then runs a different one.
The automated gates cannot catch this: `check_notebook_md_sync.py` only parses
Python code blocks, so bash prerequisite blocks drift silently (verified — it
passes despite the divergence). The same stale line exists in the out-of-scope
twin `docs/example/mcp_langchain.md:23`, which should be fixed in the same
edit since it documents the swapped langchain notebook.
**Fix:**
```bash
# docs/example/mcp_pydantic_ai.md:21 and docs/example/mcp_langchain.md:23
-ollama pull qwen3.6:latest
+ollama pull qwen3.5:4b
```

### WR-02: ACTIVE-lane sandbox fixture ignores the spec `yaml_patch` key the gated lane forwards

**File:** `tests/examples/test_notebook_execution.py:139-154`
**Issue:** The delta added two parallel spec-consumption paths for
`yaml_patch`: `TestGatedNotebookExecution.gated_sandbox` (line 1293) forwards
`spec.get("yaml_patch")` to `seed_sandbox`, but the ACTIVE-lane
`notebook_sandbox` fixture seeds with `extra_inputs` only. Today no
`ACTIVE_NOTEBOOKS` entry carries a `yaml_patch`, so nothing is wrong at
runtime — but if an editor later adds a `yaml_patch` to a notebook that is
(also) in `ACTIVE_NOTEBOOKS`, it will be silently ignored and the sandbox will
run the committed uncut values. That is exactly the stale-spec-drift class
this harness's own header documents as having hidden a strictly-below
violation before (05 review WR-02/WR-03 precedent). Robustness trap, not a
current defect.
**Fix:** Forward the same key in the ACTIVE fixture (or fail loudly on an
unexpected key):
```python
    nb_path = Path(request.node.callspec.params["nb_path"])
    extras = _NOTEBOOK_EXTRA_INPUTS.get(nb_path.relative_to(EXAMPLE_DIR).as_posix(), [])
    spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
    yield seed_sandbox(
        nb_path.parent, tmp_path, extra_inputs=extras, yaml_overrides=spec.get("yaml_patch")
    )
    assert_tree_clean()
```

## Info

### IN-01: `yaml_overrides` on a non-mapping section raises `AttributeError`, not the documented `ValueError`

**File:** `tests/examples/_execution.py:443-450`
**Issue:** The fail-closed contract (docstring: "Raises ValueError ... a
patched section is absent") covers `data is None` and missing sections, but a
section that maps to a non-dict (e.g. YAML `finetune:` with null/scalar value)
reaches `data[section].update(kv)` and raises `AttributeError`. Still loud
(never a silent unpatched sandbox), just a different exception type than the
documented contract. Edge case; no current spec targets such a file.
**Fix:** Widen the guard: `if not isinstance(data, dict) or not isinstance(data.get(section), dict): raise ValueError(...)`.

### IN-02: Prerequisite probes let `subprocess.TimeoutExpired` escape instead of reporting `(False, evidence)`

**File:** `tests/examples/_execution.py:694-700, 848-854`
**Issue:** `megadna_prerequisites_installed()` / `evo_prerequisites_installed()`
run `subprocess.run(..., timeout=120)` without catching `TimeoutExpired`. A
hung venv interpreter (heavy torch-backed imports under I/O pressure) makes
the gate raise an unhandled exception (test ERROR) rather than the documented
honest typed skip with probe evidence. It fails loud — never a false green —
so this is robustness polish only.
**Fix:** Wrap the probe in `try: ... except subprocess.TimeoutExpired as exc:
return False, f"venv probe timed out after 120s ({venv_python}): {exc}"`.

---

_Reviewed: 2026-10-06T14:51:10Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
