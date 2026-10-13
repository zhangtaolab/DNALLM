# Phase 14: GAP-1 Transformers 5.x Disposition - Pattern Map

**Mapped:** 2026-10-13
**Files analyzed:** 7 (touched/produced by the shim-path disposition — the evidence-supported path)
**Analogs found:** 6 / 7 (the remaining one is a phase-dir artifact whose analog is the phase-13 SUMMARY shape)

**Key framing from RESEARCH:** the gap is already closed. This phase writes **zero new rungs and zero functional code** — every file change is comment/prose honesty, ledger bookkeeping, or evidence recording. The dominant pattern to copy is therefore **phase 13's comment-only-edit discipline with mechanical matcher-diff and residue-count gates**, not any code-generation pattern.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tests/models/test_model_remote_code.py` | test (prose-only edit) | n/a (no data flow; evidence harness) | `tests/examples/test_notebook_execution.py` D-07 comment-block rewrite in phase-13 Task 1 step 8 (`.planning/phases/13-ollama-surface-disposition/13-01-PLAN.md` lines 127, 260-262) | exact (comment-block accuracy rewrite, zero functional edits) |
| `tests/examples/test_script_execution.py` | test (prose-only edit) | n/a | same — phase-13 "Group C comment-rewritten" disposition class | exact |
| `tests/examples/test_notebook_execution.py` | test (prose-check, possibly zero-diff) | n/a | same | role-match |
| `.planning/WINDOWS.md` (via `gsd-tools windows fixed 11/12/15`) | config/ledger | bookkeeping | phase-13's WINDOWS closures via gsd-tools (see commit 7534255 / 7f18e6f history: "windows #22 fixed") | exact |
| `.planning/REQUIREMENTS.md` | config/ledger | bookkeeping | phase-13 REQUIREMENTS checkbox updates | exact |
| `.planning/STATE.md` | config/ledger | bookkeeping | phase-13 STATE.md phase decision line | exact |
| `<phase-dir>/14-01-SUMMARY.md` + verification/evidence notes | doc (new) | n/a | `.planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md` (verbatim residue record + measured census evidence shape) | exact |

No source, service, controller, or model files are modified in the recommended shim-path-zero-rung disposition. `dnallm/utils/transformers_compat.py` is a READ-ONLY reference this phase (analog only if a real traceback appears, per the locked decision).

## Pattern Assignments

### `tests/models/test_model_remote_code.py` (test, prose-only)

**Analog:** the phase-13 "Group C" comment-accuracy rewrite, executed as PLAN 13-01 Task 1 step 8 (13-01-PLAN.md:127, 260-262): rewrite ONLY the stale prose block; every functional line stays byte-identical; the diff gate asserts the touched lines are comments-only.

**What to change here:** module docstring lines 1-18 still narrate the pre-2026-10-02 D-07 terminal state as current ("the load then hit the sanctioned fallback ladder's terminal rung… the smoke records the exact traceback as a typed skip"). Rewrite to the resolved state: the `_patch_pretrained_config_legacy_defaults` rung (landed 2026-10-02, transformers_compat.py:489-510) answers `is_decoder`/`add_cross_attention`; the checkpoint loads and forwards (1, 2) on 5.17 (recorded GAPC-01 run). The marker machinery stays UNTOUCHED:

```python
# Source: tests/models/test_model_remote_code.py:33 (KEEP byte-identical — regression tripwire)
_STRUCTURAL_MARKER = "'EsmConfig' object has no attribute 'is_decoder'"
```

```python
# Source: tests/models/test_model_remote_code.py:62-76 (KEEP — the self-healing typed-skip branch)
        except ValueError as exc:
            if _STRUCTURAL_MARKER not in str(exc):
                raise
            environment_unavailable_skip(
                "nt-v2 promoter remote code on transformers 5.17",
                f"sanctioned fallback ladder terminal rung (D-07): remote "
                ...
            )
```

The comment block at lines 29-32 above the marker should be updated to state the branch is a currently-dormant tripwire (the rung answers both names, so the AttributeError text can no longer occur unless transformers regresses).

**Mechanical gate to copy (phase-13 Task 2 verify, 13-01-PLAN.md:169):**
```bash
# Adapt to this file: diff HEAD vs working tree, assert only comment/docstring lines changed
git show HEAD:tests/models/test_model_remote_code.py > /tmp/head-smoke.py
diff /tmp/head-smoke.py tests/models/test_model_remote_code.py   # every +/- line must be prose
```

### `tests/examples/test_script_execution.py` (test, prose-only)

**Analog:** same Group-C comment-rewrite class.

**What to change:** the GAP-1 comment block at lines 35-45 ("GAP-1 terminal signature (05-04)… Owner disposition pending") narrates the pending state. Rewrite to resolved: same closed-map rung answer; the script passed for real in example-nightly run 38308972028 (stage 1: `test_generate_bpe_dataset_produces_artifact` PASS). KEEP the marker and branch:

```python
# Source: tests/examples/test_script_execution.py:46 (KEEP byte-identical)
_NT_STRUCTURAL_MARKER = "Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'"
```

Branch at line 186 (`if _NT_STRUCTURAL_MARKER in str(exc):`) stays byte-identical — probe-conditioned tripwire.

**Census guard (RESEARCH Pitfall 2, ci.yml:867):** prose-only edits inside existing bodies do NOT move the `216/217 tests collected (1 deselected)` pin; verify after edit:
```bash
.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" 2>/dev/null | tail -1
```

### `tests/examples/test_notebook_execution.py` (test, prose-check)

**Analog:** same. Task: check for any GAP-1/narrating-stale-state prose on the benchmark-notebook lane (grep `GAP-1`, `is_decoder`, `terminal rung`). If none narrates the pending state (current grep shows no GAP-1 hits), record "zero-diff inspected" in the SUMMARY rather than editing.

### Ledger / planning files (`WINDOWS.md`, `REQUIREMENTS.md`, `STATE.md`)

**Analog:** phase-13 closure mechanics. Never hand-edit WINDOWS.md rows:

```bash
# Source: RESEARCH "Don't Hand-Roll" — verified this session
gsd-tools windows fixed 11   # reason carries probe evidence; then 12, 15
```

Closure semantics per locked decision: #11/#12/#15 fixed with the GAPC-01 probe evidence (shim path). For #15, cite per-item nightly-junit rows (run 38308972028: 5 skips are 1 content / 2 mcp network / 2 pygenometracks optional-dep — none #15-class; Pitfall 5 forbids blanket closure).

### `<phase-dir>/14-01-SUMMARY.md` + evidence notes (new doc)

**Analog:** `.planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md` — shape: verbatim recorded command outputs in fenced blocks + measured evidence + per-task artifacts. Record here:
- GAPC-01 captured run (RESEARCH Pattern 3, exact command):
```bash
.venv/bin/python -m pytest tests/models/test_model_remote_code.py::TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards \
  -m slow -q --junitxml=/tmp/gapc01-smoke-junit.xml 2>&1 | tee <phase-evidence-file>
```
  (note `-m slow` must re-include the `@pytest.mark.slow` test; default fast-lane markers deselect it)
- reachability analysis for `invert_attention_mask` (dead branch; remote modeling_esm.py:1028-1041 guard `self.config.is_decoder and encoder_hidden_states is not None`)
- optional BPE-leg probe evidence (task_type MUST be `"token"`, Pitfall 3)
- census reproduction lines, fast-lane + benchmark-gate results, skip-audit output

Evidence lives in the phase dir, NOT a new repo file (RESEARCH Open Question 1 recommendation).

## Shared Patterns

### Comment-only edit discipline (applies to all three test files)
**Source:** `.planning/phases/13-ollama-surface-disposition/13-01-PLAN.md` (Task 1 step 8, Task 2; prohibition P2 + matcher-diff gate at line 169)
- Only docstrings/comment blocks change; every marker constant, branch, function, and collected test id stays byte-identical.
- Prove it mechanically: diff against HEAD showing only prose lines; then run the census collect (count unchanged) and the file's tests.
- Deleting the marker branches is a documented ANTI-PATTERN (RESEARCH Anti-Patterns #2) — they are the loud re-skip tripwire.

### Rung ladder shape (READ-ONLY reference; use only on a real traceback)
**Source:** `dnallm/utils/transformers_compat.py` — rung that closed GAP-1 at lines 489-510 (`_LEGACY_PRETRAINED_CONFIG_DEFAULTS` closed map) and `apply_patches()` at 1228-1246; hypothetical-rung template `_patch_get_head_mask` at 913-940 (try-import, hasattr gate, attach, `_dnallm_*` sentinel). Unit-test home: `tests/utils/test_transformers_compat.py` (79 tests, class-per-rung). The locked decision caps any extension at 3 rungs, public-API absence only. **Expected this phase: zero rungs** — adding a defensive `invert_attention_mask` rung is Anti-Pattern #1.

### Gate commands (verified shapes, RESEARCH Code Examples)
```bash
.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"   # census: expect 216/217
.venv/bin/python -m pytest -m "not slow" -q                                    # fast lane
.venv/bin/python -m pytest tests/benchmark/test_benchmark.py -q                # 35-family gate
.venv/bin/python scripts/audit_skips.py <junit.xml> tests/expected_skips.yaml  # skip honesty
```
`tests/expected_skips.yaml` needs NO change (no un-skip, no removal); any edit to it would be matcher drift — copy phase-13's MATCHERS-UNCHANGED diff gate if in doubt.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | Every touched file has a same-class precedent in phase 13 or is an in-place prose edit of its own prior state. |

## Metadata

**Analog search scope:** `.planning/phases/13-ollama-surface-disposition/` (full 13-01-PLAN.md read), `tests/models/`, `tests/examples/`, `dnallm/utils/transformers_compat.py` (targeted reads), `.github/workflows/ci.yml` (via RESEARCH citations)
**Tracked-source check:** all named analogs verified via `git ls-files` (tests/models/test_model_remote_code.py, tests/examples/test_*.py, dnallm/utils/transformers_compat.py, scripts/audit_skips.py all tracked; phase-13 planning docs are committed per git history)
**Pattern extraction date:** 2026-10-13
