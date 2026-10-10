# Phase 9: CI Wiring & Census Verification - Pattern Map

**Mapped:** 2026-10-05
**Files analyzed:** 13 (8 modified, 3 new, 2 conditional) + 2 explicitly out-of-scope seams recorded
**Analogs found:** 11 exact (in-file precedent) / 13 total; 2 mechanisms have no in-repo analog (D-19 cron clause, docs CI page) and inherit RESEARCH.md patterns

This is a wiring phase over an already-green execution layer: nearly every edit target carries its own in-file precedent. "Analog: itself" below means the exact line ranges to copy/extend are inside the file being edited. All analog paths are git-tracked (`git ls-files` verified 2026-10-05). No gitignored/mirror paths are referenced.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `.github/workflows/ci.yml` | config (CI workflow) | event-driven batch | itself — in-file precedent at every seam (see assignments) | exact |
| `pyproject.toml` | config | static registration | itself — `markers` list at 511-521, `ruff==0.16.9` pin at 83, `[tool.ty.src]` at 395-404 | exact |
| `tests/examples/_execution.py` | test-harness utility | file-I/O (sandbox copytree + YAML patch) | itself — `seed_sandbox` 326-388; spec `env` key precedent 214-230 | exact |
| `tests/examples/test_notebook_execution.py` | test | transform (spec-derived parametrize marks) + file-I/O contracts | itself — `_TIMEOUT_7200_GATED` 1120-1130, marks comprehension 1145-1153, `TestSeedSandbox` 293-357 | exact |
| `tests/test_models_lock_contracts.py` (NEW, name = planner's call) | test (fast contract) | file-I/O parse (lock + notebook JSON scan) | `tests/test_extras_guard.py` (file structure) + `TestMegadnaSiblingContentContracts` in `tests/examples/test_notebook_execution.py:833-903` (notebook JSON scanning idiom) | role-match + exact idiom |
| `scripts/runner/ollama.service` | config (systemd unit) | request-response (server env) | itself — `Environment=` block 26-32 | exact |
| `scripts/runner/README.md` | docs (owner ops) | N/A | itself — "One-time owner setup" 9-28, loopback rationale 30-36 | exact |
| `docs/<new CI/testing page>.md` (NEW) | docs | N/A | `docs/user_guide/troubleshooting.md` (page structure) + `tests/TESTING.md` (suite content) + `pyproject.toml:546-549` (precedent wording) | partial (no CI page exists) |
| `mkdocs.yml` | config (docs nav) | N/A | itself — nav User Guide block 60-65 | exact |
| `.github/workflows/README.md` | docs (CI topology) | N/A | itself — "Workflow Triggers" 9-17 (the stale text to fix) | exact |
| `tests/TESTING.md` | docs (test suite) | N/A | itself — markers list at 98-99 and 112-124 | exact (conditional: add `giants` row) |
| `tests/expected_skips.yaml` | config (skip-audit contract) | N/A | itself — matcher semantics header 1-13 | exact (conditional: expected NO change) |
| `09-CENSUS-ROLLUP.md` (phase artifact, not code) | planning artifact | N/A | `.planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md` | exact |

**Explicitly OUT of Phase 9 scope (D-09 boundary — do NOT edit this phase):** `scripts/check_code.py`, `.pre-commit-config.yaml`, the mypy CI steps, and `[tool.mypy]`. The ty hard-gate flip + mypy retirement is one later atomic quick task. Seams recorded at the bottom so the later task inherits them.

## Pattern Assignments

### `.github/workflows/ci.yml` (config, event-driven batch)

**Analog:** itself. All seven wiring points have in-file precedents.

**D-01 — stage-1 deselect line** (current step at 792-804; the invocation to extend is line 798):

```yaml
      - name: "Stage 1: torch-heavy example execution (ACTIVE + showcase + marimo + script + YAML leg)"
        # D-08 fail-soft: each invocation records its exit code and the
        # step exits 0 so stages 1.5-4 still run; the stage-4 summary
        # carries the non-zero verdict when anything failed.
        run: |
          set +e
          .venv/bin/python -m pytest tests/examples -q --junitxml=pytest-junit-example-stage1.xml -k "not mcp_example"
          rc=$?
          echo "stage1-examples=$rc" >> stage-results.txt
```

Phase 9 adds `-m "not giants"` to that single pytest line (selectors AND together). Nothing else in the step changes.

**D-19 — event gates to make cron-aware** (three identical gates today, at lines 275, 423, 525):

```yaml
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
```

Replace with `github.event.schedule == '<cron>'` clauses per RESEARCH.md (coverage-nightly at 423 and test-mamba at 275 get `'0 3 * * *'`; example-nightly at 525 gets `'30 5 * * *'`; each keeps the `workflow_dispatch` disjunct). No in-repo `github.event.schedule` usage exists (grep verified) — the clause form comes from RESEARCH.md "D-19 gate" example. Also rewrite the now-false schedule-block comment at lines 14-17 (it claims event gates distinguish crons, which they cannot).

**D-03 — collection-count hard assertion step** (new pre-stage-1 step; hard-assert step shape precedent = the exit-code canary, lines 103-119):

```yaml
      - name: Exit-code canary (a failing test must fail the job)
        run: |
          source .venv/bin/activate
          cat > tests/test_ci_exitcode_canary.py <<'CANARY'
          def test_ci_exitcode_canary():
              assert False, "intentional canary failure"
          CANARY
          if pytest tests/test_ci_exitcode_canary.py -q -p no:cacheprovider > canary.log 2>&1; then
            echo "CANARY FAILED: pytest exited 0 despite a failing test (exit-code mask regression)"
            ...
            exit 1
          else
            echo "Canary OK: pytest exited non-zero as expected"
          ```
```

Copy this if/grep-then-`exit 1`-with-explicit-FAIL-message shape for the collect-only assertion (expected-form triple from RESEARCH.md Pattern 2; a stage-0-area failure failing the job directly matches the 550-553 contract). Capture output to `census-collect.txt` so D-14's upload includes it.

**D-04 — evo step blocks to delete** (keep neighbors intact):

| Delete | Lines | Content |
|--------|-------|---------|
| evo isolated venv step | 662-686 | `"Stage 0: evo isolated venv — FEASIBILITY-locked stack"` |
| `evo_torch=` output line | 693 only | inside the wheelkeys step (688-694) — keep the step and its `torch=` line 692 (mamba wheelhouse key at 722 consumes it) |
| flash-attn wheelhouse cache + build | 696-716 | `Cache flash-attn wheelhouse (evo venv)` + build/install step |
| giants prefetch | 760-779 | `"Stage 0: evo-1 giants prefetch (safetensors-only → ~/models-giants, D-14/CI-05)"` |

**Keep verbatim:** mamba wheelhouse (718-737), megaDNA provisioning (739-758), runner inventory probe (781-790), all stage 2/3 server steps.

**D-11 — models-cache blocks to delete** (both keyed on `hashFiles('models.lock')`):

- coverage-nightly: lines 467-475 (`"Restore model caches (keyed on models.lock)"`)
- example-nightly: lines 592-604 (same step name + the D-09 comment block 593-596)

Keep: uv dependency caches (459-465, 584-590), bedtools prefix cache (630-634), wheelhouse caches (696-722 minus the flash-attn one).

**D-13 — hygiene named steps** (expand the two existing settle steps at 806-813 and 866-873):

```yaml
      - name: "Stage 1.5: kernel cleanup + VRAM settle (D-07)"
        # Explicit VRAM/process cleanup between the torch-heavy stage and
        # any server-binding stage: stray ipykernel children are killed
        # before GPU memory is expected to settle.
        run: |
          pkill -f ipykernel_launcher || true
          sleep 5
          nvidia-smi --query-gpu=memory.used,memory.total --format=csv || echo "nvidia-smi MISSING (VRAM settle unverified)"
```

Extend into named hygiene steps with before/after `free -g` logging and the >=35Gi hard floor per RESEARCH.md "Named hygiene step" example (nvidia-smi stays telemetry-only — GB10 pitfall). Keep the existing step-name numbering convention (`"Stage 1.5: ... (D-07)"` → cite D-13 in the new name/comment).

**D-14 — full if:always() upload** (extend the existing upload at 924-931):

```yaml
      - name: "Stage 4: upload junit artifacts"
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: example-nightly-junit
          path: |
            pytest-junit-*.xml
            mcp-server-*.log
```

Add `stage-results.txt`, per-stage logs, and `census-collect.txt` to `path`. Per-stage log capture precedent (tee) is verbatim in test-mamba at 343-345: `set -o pipefail` + `pytest ... 2>&1 | tee pytest.log`.

**D-08/ty — advisory static-check step** (shape precedent = the mypy advisory step, lines 121-124; placement precedent = coverage-gate skip-audit step at 411-414):

```yaml
      - name: Run type checking
        run: |
          source .venv/bin/activate
          mypy dnallm/ --show-error-codes --pretty --exclude=dnallm/tasks/metrics/ || true
```

New ty step mirrors this exactly (`|| true`, advisory comment citing D-08/D-09) on the coverage-gate job, after the skip-audit step. Invocation: `uvx ty@0.0.84 check dnallm/ || true` (no pyproject change) or `.venv/bin/ty check dnallm/ || true` with the dev-dep pin. Do NOT touch the mypy steps (D-09).

**D-12 — budget comment rewrites** (the two comment blocks to rewrite from D-02 measured values): coverage-nightly 424-432 and example-nightly 541-563 (both quoted in full in the file; keep the per-test-marks-primary / job-kill-backstop argument structure, replace the numbers with measured actuals). Also update the "Measured dev-box actuals" block 554-563.

---

### `pyproject.toml` (config, static registration)

**Analog:** itself.

**D-01 — marker registration** (addopts carry `--strict-markers` at line 506; markers list at 511-521 — registration is MANDATORY in the same atomic change as first use):

```toml
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--strict-config",
    "--asyncio-mode=auto",
    "--timeout=300"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "pdf: marks tests that generate PDF files",
    "performance: marks performance-related tests",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "inference: marks inference-related tests",
    "utils: marks utility function tests",
    "data: marks data handling tests",
    "legacy: marks tests for legacy/deprecated features"
]
```

Append a `giants` entry in the same `"name: description"` one-line style, with the deselect hint mirroring the `slow` line's wording (e.g. `"giants: marks giant-model (evo) tests excluded from the example-nightly census by owner policy (deselect with '-m \"not giants\"')"` — exact wording is planner's call per CONTEXT discretion).

**D-08 optional — ty dev-dep pin** (precedent = the exact `ruff` pin at line 83 inside the `dev` extra, lines 81-91):

```toml
dev = [
    "dnallm[test,notebook]",
    "ruff==0.16.9",
    ...
]
```

If the dev-pin form is chosen over `uvx`, add `"ty==0.0.84"` next to `ruff==0.16.9` (research flags this for a `checkpoint:human-verify` per the SUS registry verdict; the `uvx` form sidesteps pyproject entirely). The `[tool.ty.src]` exclude block already exists at 395-404 — no change needed there.

---

### `tests/examples/_execution.py` (test-harness utility, file-I/O)

**Analog:** itself — `seed_sandbox` (lines 326-388) is the D-05 seam.

**Existing signature + copytree to extend** (326-358, abridged):

```python
def seed_sandbox(
    src_dir: Path,
    tmp_path: Path,
    extra_inputs: list[Path | tuple[Path, str]] | None = None,
) -> Path:
    """Copy an example directory into a pytest tmp sandbox for execution.
    ...
    """
    sandbox = tmp_path / src_dir.name
    shutil.copytree(
        src_dir,
        sandbox,
        ignore=shutil.ignore_patterns(
            ".ipynb_checkpoints", "__pycache__", "outputs*", "results*", "*.gz",
        ),
    )
    ...
    return sandbox
```

Add an optional `yaml_overrides: dict[str, dict] | None = None` parameter applied post-copytree, before `return sandbox` (RESEARCH.md Pattern 3 sketch). Follow this module's conventions: Google docstring with `Args:`/`Raises:`, defensive `ValueError` for bad input, comment explaining the D-05 "committed content unchanged" contract. Note: the module currently imports no yaml — adding `import yaml` at the top matches `scripts/audit_skips.py`'s plain `import yaml` usage.

**Spec-key precedent for the driver side** (a spec entry carrying a per-notebook override consulted by the test layer — lora entries at 214-230):

```python
    str(EXAMPLE_DIR / "notebooks" / "lora_finetune_inference" / "lora_finetune.ipynb"): {
        "cell_timeout": 3600,
        "extra_inputs": [],
        # 08-08 mirror endpoint: ...
        "env": {"HF_ENDPOINT": "https://hf-mirror.com"},
    },
```

A `"yaml_patch"` key on the `finetune_custom_head` spec entry (line 189-192) follows this exact pattern; the gated fixture then reads `spec.get("yaml_patch")` the way `run_notebook` reads `spec.get("env")`.

---

### `tests/examples/test_notebook_execution.py` (test, transform + file-I/O contracts)

**Analog:** itself — three in-file precedents.

**D-01 — spec-derived mark application** (the exact mechanism to extend: `_TIMEOUT_7200_GATED` at 1120-1130 + the marks comprehension at 1145-1153):

```python
_TIMEOUT_7200_GATED: frozenset[str] = frozenset({
    "notebooks/finetune_custom_head/finetune.ipynb",
    "notebooks/finetune_generation/finetune_generation.ipynb",
    "notebooks/lora_finetune_inference/lora_finetune.ipynb",
})


@pytest.mark.slow
@pytest.mark.timeout(3600)
class TestGatedNotebookExecution:
    """Probe-then-execute for environment-gated notebooks (D-05/D-06)."""

    @pytest.fixture
    def gated_sandbox(self, tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
        """Seed the gated notebook's directory and assert the tree stays clean."""
        nb_path = EXAMPLE_DIR / request.node.callspec.params["gated_id"]
        yield seed_sandbox(nb_path.parent, tmp_path)
        assert_tree_clean()

    @pytest.mark.parametrize(
        "gated_id",
        [
            pytest.param(nb_id, marks=pytest.mark.timeout(7200))
            if nb_id in _TIMEOUT_7200_GATED
            else nb_id
            for nb_id, _gate in GATED_NOTEBOOKS
        ],
        ids=str,
    )
```

Add a sibling `_GIANTS_GATED: frozenset[str]` holding `"notebooks/generation_evo_models/inference.ipynb"` (the `GATED_NOTEBOOKS` entry at line 1104) and fold `marks=pytest.mark.giants` into the same comprehension (composable marks list on the `pytest.param`, or a second conditional). Do NOT mark the fast evo contract tests (`TestEvoIsolatedLane` 726-770, `TestSpecEnvOverrides` evo rows 583-588) — they are kernel-free fast-lane tests (research A3 recommendation, planner's call inside the discretion).

**D-05/D-07 — same-change sandbox contract tests** (kernel-free unit-test pattern = `TestSeedSandbox`, 293-357):

```python
class TestSeedSandbox:
    """Kernel-free unit contract for :func:`seed_sandbox` extra inputs (261002-sl7).
    ...
    """

    @staticmethod
    def _seed_dir(tmp_path: Path) -> Path:
        """Create a tiny fake example dir holding one notebook and one sibling input."""
        src_dir = tmp_path / "fake_example"
        src_dir.mkdir()
        (src_dir / "tiny.ipynb").write_text("{}", encoding="utf-8")
        (src_dir / "sibling.csv").write_text("sequence,label\nAT,1\n", encoding="utf-8")
        return src_dir
```

Copy this `_seed_dir` fake-directory idiom for the yaml_overrides test: seed a fake dir with a `finetune_config.yaml` carrying `num_train_epochs: 3`, apply the patch, assert the SANDBOX copy reads `1` while the source still reads `3` (the D-05 honesty contract).

**CI-08 scanning idiom** (if any lock-guard checks land here instead of a new file): `TestMegadnaSiblingContentContracts._code_cells` (847-850) and `_active_lines` (881-885) + `test_source_routes_are_d15_aligned` (887-903) are the JSON-code-cell-scanning and active-route-vs-lock-direction precedents — the lock guard "extends, not duplicates" this contract.

---

### `tests/test_models_lock_contracts.py` (NEW — test, file-I/O parse)

**Analog A — new standalone fast contract file structure:** `tests/test_extras_guard.py` (140 lines). Module docstring stating the decision provenance, `from __future__ import annotations`, REPO_ROOT anchor, file loader helpers, `Test*` classes:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_mcp_extra() -> list[str]:
    """Parse pyproject.toml with tomllib and return the mcp extra members."""
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data["project"]["optional-dependencies"]["mcp"])


class TestMcpExtraMembers:
    """Bracket-member assertions over project.optional-dependencies["mcp"]."""

    def test_langchain_ollama_declared_in_mcp_extra(self) -> None:
        """langchain-ollama>=1.1.0 is an exact member of the mcp extra."""
        mcp = _load_mcp_extra()
        assert "langchain-ollama>=1.1.0" in mcp, (
            f"langchain-ollama>=1.1.0 missing from the mcp extra (REPAIR-04): {mcp}"
        )
```

Anchor `MODELS_LOCK = REPO_ROOT / "models.lock"` the same way. Runs unmarked (fast leg) — every test in this file must be kernel-free and network-free.

**Analog B — lock parsing (fail-closed YAML/text discipline):** `scripts/audit_skips.py` 29-46:

```python
    try:
        with open(allowlist_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"cannot read allowlist {allowlist_path}: {e}") from e
```

The lock is comment-annotated text (`<hf|ms>  <repo-id>[@<sha>]  # purpose`, header at `models.lock:1-9`, 24 model rows + 1 dataset row) — parse line-by-line, strip `#` comments, fail the test (not skip) on any unparseable model row.

**Analog C — notebook scanning:** `TestMegadnaSiblingContentContracts` in `tests/examples/test_notebook_execution.py:833-903` (see above). The guard scans `example/**/*.ipynb` code-cell sources + marimo apps + example YAML configs for model-id literals and fails on any example-referenced remote model id absent from the lock.

**Analog D — D-07 unit-file contract test (same new file or beside it):** pin `scripts/runner/ollama.service` content the way `TestLoraMirrorEndpoint` (test_notebook_execution.py:649-669) pins spec content — a fast assertion that the unit contains `OLLAMA_CONTEXT_LENGTH=8192` and `OLLAMA_HOST=127.0.0.1:11434`, whose docstring cites the D-06 decision and the live-drift finding, so "an editorial revert fails in seconds on the fast lane, not at the next 25-minute real execution".

---

### `scripts/runner/ollama.service` (config, request-response)

**Analog:** itself — the Environment block to extend (lines 26-32):

```ini
# Genericized PATH: standard system dirs plus /usr/local/bin (where the
# official ollama installer places the binary). No box-specific paths.
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
# D-12: loopback-only bind -- the access control for network clients.
Environment="OLLAMA_HOST=127.0.0.1:11434"
# Optional: pin model storage to a dedicated dir (uncomment to use).
# Environment="OLLAMA_MODELS=/var/lib/ollama/models"
```

Add `Environment="OLLAMA_CONTEXT_LENGTH=8192"` with a D-06 comment in the same style (decision tag + one-line rationale). Preserve the loopback pin verbatim — the LIVE unit has drifted to `0.0.0.0` (research, live-probed) and the owner's re-apply of this file restores it. Extend the header comment block (1-15) with the num_ctx rationale. File self-describes as "auditable and rebuildable by diff" — keep it that way.

---

### `scripts/runner/README.md` (docs, owner ops)

**Analog:** itself. Extend the "One-time owner setup" numbered list (9-28) with the re-apply step (`sudo cp ... / systemctl daemon-reload / sudo systemctl restart ollama` — env is read at server start) and a short "Why num_ctx 8192 (D-06)" section mirroring the existing "Why loopback-only (D-12)" section at 30-36. Flag the 0.0.0.0 drift + restore explicitly for the owner.

---

### `docs/<new CI/testing page>.md` (NEW — docs) + `mkdocs.yml`

No CI/testing docs page exists today (research CI-09; mkdocs nav verified — no coverage/nightly entry).

**Structure analog:** `docs/user_guide/troubleshooting.md` (64 lines) — H1 title, one-paragraph summary, `##` sections, relative links to sibling pages, bash/yaml fenced blocks.

**Content analog:** `tests/TESTING.md` (294 lines) — the in-repo test-suite documentation (markers table at 112-139, coverage section 141-160).

**Wording precedent for the coverage expectation (D-15/AUDIT-04):** `pyproject.toml:546-549`:

```toml
[tool.coverage.report]
show_missing = true
fail_under = 90   # Phase 4 ratchet (GATE-01). Suite landed at 96.30% (Phase 3).
                  # Applies to every `--cov` invocation — use `--no-cov` for scoped runs.
```

The new page documents: example execution runs in kernel subprocesses and by design does not move the 96.30% coverage gate (kernel subprocess coverage is not measured — that is the AUDIT-04 design note), the fail_under=90 ratchet, and the nightly topology.

**Nav registration:** `mkdocs.yml` User Guide block (lines 60-65) — add one `Page Name: user_guide/<file>.md` line in the same `Name: path` style. Placement is planner's discretion (research A5).

---

### `.github/workflows/README.md` (docs, CI topology)

**Analog:** itself. The stale "Workflow Triggers" section (lines 9-17) currently documents only the 03:00 cron:

```markdown
- **Scheduled nightly run** at 03:00 UTC — triggers the `coverage-nightly` full census and the `test-mamba` kernel-build leg (GitHub runs cron schedules only from the default branch)
```

Update with the D-19 post-fix topology (03:00 → coverage-nightly + test-mamba; 05:30 → example-nightly; cron-string gates), and fold in the stage list / cache changes from D-04/D-11 while editing (research Pitfall 8: fold the refresh into the same plan that edits the schedule gates). The "Model Caches" bullet in the coverage-nightly section (line 104) also needs the D-11 removal reflected.

---

### `tests/TESTING.md` (docs — conditional)

**Analog:** itself. If `giants` is registered, add it to both marker listings: the config list at lines 98-99 (`markers` — `slow`, `pdf`, ..., `legacy`) and the "Test Markers" table at 112-124 (one `**@pytest.mark.giants**` bullet in the same format). One-line docs change, same commit as the marker registration.

### `tests/expected_skips.yaml` (config — conditional, expected NO change)

**Analog:** itself. The evo exit is a marker deselect, NOT a skip, so no new entry is expected (research CI-03). The `optional-dep:` prefix entry (lines 39-41) continues to cover the coverage-nightly typed skip for the gated evo test (research OQ1 recommends leaving coverage-nightly unchanged). Only touch this file if the planner/owner opts to deselect giants on coverage-nightly too — in which case the audit stays green with zero changes anyway (a deselect produces no skip message).

## Shared Patterns

### Fail-soft stage recording + summary red (example-nightly contract)

**Source:** `.github/workflows/ci.yml:796-804` (record) and `933-946` (summary)
**Apply to:** any new example-nightly step that runs pytest (the D-03 census assertion is the deliberate exception — it is a HARD gate like stage 0)

```yaml
          set +e
          .venv/bin/python -m pytest ... 
          rc=$?
          echo "stage1-examples=$rc" >> stage-results.txt
          ...
          exit 0
```

```yaml
          cat stage-results.txt
          if grep -qE '=[1-9][0-9]*$' stage-results.txt; then
            echo "FAIL: at least one stage item exited non-zero (D-08: this job may never be forever-green)"
            exit 1
          fi
```

Never `continue-on-error` anywhere in example-nightly (ci.yml:550-553 prohibition).

### Hard-assert step shape

**Source:** `.github/workflows/ci.yml:103-119` (exit-code canary)
**Apply to:** the D-03 collection assertion step — explicit echo of the failure reason + `exit 1`; never fail-soft.

### Advisory static-check step shape

**Source:** `.github/workflows/ci.yml:121-124` (mypy `|| true`)
**Apply to:** the new ty advisory step on coverage-gate. Identical body: `source .venv/bin/activate`, tool invocation, `|| true`, comment naming the decision (D-08) and the flip condition (D-09).

### Fast contract-test file structure

**Source:** `tests/test_extras_guard.py`
**Apply to:** the new CI-08 lock-guard test file and the D-07 unit-file contract test. Conventions: provenance docstring citing the decision ID, `from __future__ import annotations`, `REPO_ROOT = Path(__file__).resolve().parent.parent` anchor, private `_load_*` helpers, `Test*` classes with one-line method docstrings, assertion messages that name the decision and remediation, kernel-free/network-free so the fast leg collects them.

### Spec-driven per-notebook overrides

**Source:** `tests/examples/_execution.py:214-230` (spec `env` keys) consumed via `spec.get(...)` at `tests/examples/test_notebook_execution.py:1165-1193`
**Apply to:** the D-05 `yaml_patch` spec key — same dict-shape, same `.get()` consumption, same fast-lane contract test pinning it (`TestSpecEnvOverrides` 574-647 / `TestLoraMirrorEndpoint` 649-669 precedent).

### Marker registration atomicity

**Source:** `pyproject.toml:506` (`--strict-markers` in addopts)
**Apply to:** D-01 — the pyproject registration, the first `pytest.mark.giants` application, and the ci.yml deselect MUST land in one commit; an unregistered marker breaks collection of every pytest invocation repo-wide (research Pitfall 2).

### Typed-skip audit contract (context only — no new skips this phase)

**Source:** `scripts/audit_skips.py` (fail-closed parse 83-92, matcher validation 37-46) + `tests/expected_skips.yaml` (one matcher key per entry, category required)
**Apply to:** nothing new — the marker mechanism exists precisely to avoid adding skip entries. Any plan step that would create a new skip message is wrong by D-01.

## No Analog Found

| Mechanism | Reason | Pattern Source |
|-----------|--------|----------------|
| `github.event.schedule == '<cron>'` gate clause | No in-repo usage (grep verified — all three nightly gates are event_name-only at ci.yml:275/423/525) | RESEARCH.md "D-19 gate" example; in-file partial analog = the existing event gates being edited |
| docs CI/testing page | No coverage/nightly/CI page exists; mkdocs nav has no slot for it | RESEARCH.md CI-09 + structural analogs listed in the assignment above |
| ty invocation in CI | No ty step exists anywhere yet (mypy is the advisory incumbent) | RESEARCH.md Pattern 5 (`uvx ty@0.0.84 check dnallm/ || true`); shape analog = mypy step ci.yml:121-124 |

## Out-of-Scope Seams (recorded for the later D-09 quick task — do NOT edit in Phase 9)

- `scripts/check_code.py:258-267` — the mypy informational step (`run_step(mypy_cmd, "MyPy type checking (informational)", check=False)`); the ty step replaces/joins this block only at the atomic flip, alongside `required_tools` at line 128 (`["ruff", "pytest", "mypy"]`).
- `.pre-commit-config.yaml:26-33` — the mypy hook; retires in the same atomic change as the flip.
- `.github/workflows/ci.yml:121-124` and `200-203` — the two mypy advisory steps; unchanged this phase.
- `pyproject.toml:406-425` — `[tool.mypy]` config; removed only at the flip.

## Metadata

**Analog search scope:** `.github/workflows/`, `tests/` (top level + `tests/examples/`), `scripts/` (`audit_skips.py`, `check_code.py`, `scripts/runner/`), `docs/` + `mkdocs.yml`, `pyproject.toml`, `models.lock`, `example/notebooks/finetune_custom_head/finetune_config.yaml`
**Files scanned:** 15 read in full; grep-verified absences: `github.event.schedule`, `OLLAMA_CONTEXT_LENGTH`, `models.lock` in tests, `giants` marker
**Tracked-source gate:** every analog path above passes `git ls-files` (verified 2026-10-05); no `.gsd/capabilities/` or other mirror paths referenced
**Pattern extraction date:** 2026-10-05
