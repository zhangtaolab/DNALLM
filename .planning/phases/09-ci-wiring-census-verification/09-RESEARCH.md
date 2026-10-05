# Phase 9: CI Wiring & Census Verification - Research

**Researched:** 2026-10-05
**Domain:** GitHub Actions workflow wiring, pytest census gating, sandbox-patch/num_ctx runtime cuts, ty advisory adoption
**Confidence:** HIGH (all in-repo claims read this session; environment claims live-probed on the runner's own box; external claims verified against official docs)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01..D-20 — LOCKED, research THESE, no alternatives)

- **D-01:** evo exit mechanism = **`giants` pytest marker** (registered into `pyproject.toml [tool.pytest.ini_options] markers`), example-nightly runs `-m "not giants"`. NOT a typed-skip — the runner environment is available; this is owner-policy exclusion. Marker semantics self-document; usable from any local/CI invocation.
- **D-02:** Post-exit new census baseline = **rebuilt during Phase 9 execution**: one baseline census run with ALL cuts applied (evo exit + epochs 1 + num_ctx 8k); the new authoritative count (expected ~190P±) recorded in the 09 rollup; **the same run doubles as the timeout measurement** (see D-12).
- **D-03:** Nightly job gains a **collection-count hard assertion**: `pytest --collect-only` count == expected value, else RED — prevents a mistyped marker silently dropping a whole test class (criterion 1 collection integrity, complementary to criterion 4's lock guard).
- **D-04:** ALL evo-specific example-nightly CI steps **removed** (giants prefetch, evo venv build, flash-attn wheelhouse cache). **The flash-attn build-isolation bug (run 37278002681) dissolves with deletion — fix = delete, no torch-first build rework.** Local `~/models-giants` assets stay (dispatch/manual lane still works, never cleaned per owner rule).
- **D-05:** finetune_custom_head epochs 3→1 via a **test-sandbox-only YAML patch** (harness sandbox-patch step; committed notebook content unchanged; loop body identical, executability claim unchanged). ~31→~11 min.
- **D-06:** mcp_example pair gets **per-request-equivalent `options.num_ctx` ~8k** injected at the D-13/probe layer (same model per D-11, same turns, notebook untouched; today's measured 256k kv-cache held 36GB and dominated both latency and the 14:07 VRAM trough).
- **D-07:** Both cuts land with **same-change tests** (sandbox patch affects only the sandbox copy; num_ctx injection point has a contract test).
- **D-08:** **Staged**: Phase 9 wires ty as an **advisory standalone step** on the coverage-gate fast leg (hosted, seconds, same layer as the ruff `--statistics` step); E-family 165→0 triage is a **later quick task** (inherits the 261003-0p0 list); flip to hard gate happens once at zero.
- **D-09:** **mypy retirement lands in the SAME atomic change as the ty hard-gate flip** (pre-commit + CI + `scripts/check_code.py` + `[tool.mypy]` config together) — no "double no-type-gate" window. NOT Phase 9 scope.
- **D-10:** Static checks never enter pytest (lint lane separate from behavior lane).
- **D-11:** **Model-cache layer removed from CI** (cold pulls proven: 65-min all-cold stage 1 vs 2700-min budget; lock-only 15.2GiB > 10GB quota and never successfully saved) — lands as a ci.yml edit; uv wheelhouse / bedtools prefix small caches stay. Local four-way caches (hf/ms/giants/ollama) **never cleaned** (owner rule).
- **D-12:** Post-cut timeout budgets **set from measurement**: D-02's baseline census measured values written directly back into the budgets/sum-of-ceilings comments — no proportional estimation risk.
- **D-13:** Nightly hygiene steps = **named between-stage steps** (kernel pkill + VRAM assertion + before/after value logging; the existing stage 2.5 cleanup expanded into a formal named step) — literal satisfaction of criterion 3 "observable".
- **D-14:** `if:always()` artifact uploads **complete**: all stage logs + census output + server logs (failure-scene preservation for the fail-soft job).
- **D-15:** Coverage expectation (criterion 5) **written into docs/** (CI/testing page + AUDIT-04 cross-reference: design note that kernel subprocesses don't count toward the 96.30% gate) — not only planning docs.
- **D-16:** **Per-plan dispatch**: after each Phase 9 plan lands its ci.yml portion, immediately dispatch example-nightly and watch the result (incremental verification; first-dispatch consumption precedent).
- **D-17:** **Green run as gate**: the final plan's verify gate includes **one green complete example-nightly dispatch** (formal confirmation of criterion 1).
- **D-18:** The two transient-failure legs (test-mamba / coverage-nightly) **re-dispatched within this phase** (once egress stabilizes; cheap, confirms recovery, leaves no loose ends).
- **D-19:** The example-nightly 05:30 cron double-trigger of coverage/test-mamba gets its **one-line job-gate fix** in this phase (08-06 leftover open flag, owner defaulted to in-phase handling).
- **D-20:** dependabot torch ignore rules landed (16a9ffb); PR #42 stays open as record, not merged.

### Claude's Discretion (research options, make recommendations)

- `giants` marker exact naming and placement (test-function decorators vs NOTEBOOK_EXEC_SPECS-derived marks)
- ty invocation form and version pin in CI (uvx vs dev-dep exact pin; ruff ==0.16.9 precedent)
- Collection-count assertion implementation (env-var expected value vs inline literal vs derived from rollup)
- docs page structure and exact AUDIT-04 cross-reference wording
- D-05 sandbox-patch implementation seam (seed_sandbox hook vs spec-env extension)
- D-06 num_ctx injection mechanism (environment variable vs test fixture rewriting client params)

### Deferred Ideas (OUT OF SCOPE — ignore completely)

- CONCERNS.md's 7 latent library bugs (inference.py:1645, mutagenesis.py:429, model.py:264/276, data.py:983, metrics.py:612, configs.py:118, benchmark.py:296)
- torch upper-bound relaxation (PR #42)
- WINDOWS.md stale items (#13 mcp_example, ship-triage WR-01)
- TypedDict consumer remainder (cli layer) — merges with the typing special
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description (from REQUIREMENTS.md) | Research Support |
|----|-------------------------------------|------------------|
| CI-03 | Execution tests `slow`-marked into the nightly census, zero new fast-leg skips; typed prefixes registered in `expected_skips.yaml`; skip audit green with new categories | Existing contract verified (all prefixes already registered; evo exit via D-01 marker introduces NO new skip — deselection is not a skip). Collection assertion (D-03) is the census-integrity half. Live: tests/examples collects 197 items today. |
| CI-06 | Measured runtime budgets recorded; separate example-execution nightly job split out (pre-authorized) | The split EXISTS (example-nightly, D-05/08). Phase 9 delivers the measured-budget rewrite: 08-CENSUS-ROLLUP runtime hand-off numbers + the D-02 baseline census actuals written into the sum-of-ceilings comments (ci.yml:424-432 coverage-nightly, ci.yml:541-563 example-nightly). |
| CI-07 | Nightly hygiene steps: kernel `pkill` + VRAM assertion, timeout-arithmetic sum-of-ceilings review, `if: always()` artifact uploads | D-13/D-14 wiring: existing stage 1.5/2.5 steps (ci.yml:806-813, 866-873) expanded to named steps with `free -g` assertion (GB10 pitfall: nvidia-smi reports no memory) and before/after logging; tee'd per-stage logs + full artifact path list; D-12 comment rewrites. |
| CI-08 | models.lock consistency guard — fast-leg test cross-checking model id literals inside notebooks/apps against lock entries, failing on drift | No existing test reads models.lock (verified by grep). New fast contract test; in-file precedent = TestMegadnaSiblingContentContracts.test_source_routes_are_d15_aligned (route-vs-prefix alignment) and the JSON content-contract class pattern. Lock format verified: 24 model rows + 1 dataset row, `<hf|ms>  <repo-id>[@<sha>]  # purpose`. |
| CI-09 | Coverage expectation documented: example execution runs in kernel subprocesses and by design does not move the 96.30% coverage gate (AUDIT-04 precedent) | No CI/testing docs page exists today (verified: docs/ has no coverage/nightly page; mkdocs nav lacks one). Precedent wording lives in pyproject.toml `[tool.coverage.report]` comment ("Suite landed at 96.30% (Phase 3)" / fail_under=90 ratchet) + STATE.md AUDIT-04 carry-over. |
</phase_requirements>

## Summary

Phase 9 is a CI-wiring phase over an already-green execution layer: every edit target is an existing seam with an in-file precedent. The example-nightly job in `.github/workflows/ci.yml` (lines 514-946) is a staged-serial, fail-soft job whose stage-1 pytest invocation (line 798) already deselects the mcp pair by keyword — `-m "not giants"` is a one-token addition to the same line once the marker is registered (`--strict-markers` in pyproject addopts makes registration mandatory). The evo-specific steps to delete are exactly four blocks (ci.yml:662-686, 696-716, 760-779, plus the `evo_torch=` line in the wheelkeys step) while the mamba wheelhouse (718-737) and megaDNA provisioning (739-758) stay. The model-cache layer to remove is two `actions/cache` blocks (ci.yml:467-475 coverage-nightly, 592-604 example-nightly).

The two runtime cuts have sharply different feasibility profiles. D-05 (epochs 3→1) has a clean, verified seam: the notebook loads `./finetune_config.yaml` (which carries `num_train_epochs: 3` at line 38) inside a `seed_sandbox` tmp copy, so a sandbox-only post-copy YAML patch never touches committed content. D-06 (num_ctx ~8k) was live-probed on the runner's own box (the runner shares $HOME with this dev box): ollama 0.34.1 serves `qwen3.8:latest` (17.74GB, native context_length 262144) with NO Modelfile `PARAMETER num_ctx`, so the documented `OLLAMA_CONTEXT_LENGTH` server-default env governs — and per-request `options.num_ctx` is provably infeasible for the pydantic_ai sibling (it talks OpenAI-compat via `OpenAIChatModel` + `/v1`, which has no num_ctx parameter). The probe also surfaced a security-relevant drift: the LIVE systemd unit runs `OLLAMA_HOST=0.0.0.0:11434` while the in-repo `scripts/runner/ollama.service` pins `127.0.0.1:11434` — the D-06 unit edit + owner re-apply also closes this.

The D-19 cron fix is confirmed against GitHub docs: `github.event.schedule` carries the exact cron string that fired, so `if:` gates can (and must) compare it — today all three nightly jobs fire on BOTH schedule entries, which goes live the moment `phs` merges to main. ty was live-run (`uvx ty check dnallm/` → "Found 167 diagnostics", ty 0.0.84) and has one verdict flag (package-legitimacy SUS: "too-new"/unknown-downloads — mitigated by owner-locked decision D-08, Astral repo, and prior 261003-0p0 in-repo adoption).

**Primary recommendation:** Register `giants` in pyproject markers + apply it spec-derived via the existing `_TIMEOUT_7200_GATED` parametrize-marks pattern; add the collect-count assertion as a hard pre-stage-1 step asserting the "X/Y collected (Z deselected)" triple; implement D-05 as a `seed_sandbox`-stage YAML override and D-06 as `OLLAMA_CONTEXT_LENGTH=8192` in the in-repo ollama.service unit (with a same-change contract test parsing the unit file); fix D-19 with `github.event.schedule == '<cron>'` clauses on all three nightly jobs.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Census exclusion policy (evo/giants) | Test layer (marker) + CI layer (deselect) | — | D-01: the marker is repo-wide semantics; the deselect is example-nightly policy. Typed-skip explicitly rejected by owner. |
| Collection-count hard assertion | CI layer (example-nightly step) | Test layer (optional fast contract pin) | Counts must be checked where the deselect actually runs; a fast in-repo pin is optional belt-and-braces. |
| epochs 3→1 cut | Test-harness layer (sandbox patch) | — | Must never touch committed notebook/YAML content (D-05); sandbox copy is the only honest seam. |
| num_ctx 8k cut | Runner infrastructure (systemd unit env) | Test layer (contract test on the unit file) | Server-side default is the only seam that covers BOTH notebooks without content changes; the unit file is the in-repo auditable source. |
| ty advisory step | CI layer (coverage-gate fast leg) | pyproject (`[tool.ty.src]` — already present) | D-08: hosted seconds-scale step parallel to ruff `--statistics`; no pytest integration (D-10). |
| Timeout budgets | CI layer (comments + timeout-minutes) | Phase execution (D-02 measured actuals) | D-12: measured values written back at wiring time. |
| Hygiene steps + uploads | CI layer (named steps) | — | D-13/D-14 are job-shape changes. |
| Lock-consistency guard | Test layer (fast contract test) | — | CI-08 names a fast-leg test; runs on every push/PR. |
| Coverage-expectation docs | Docs site (new CI/testing page) | — | D-15: user-visible documentation, not planning-internal. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest (in-venv) | >=8.4 (installed) | census execution, `--collect-only` assertion | Existing suite runner; addopts verified |
| GitHub Actions workflow syntax | actions/checkout@v4, actions/cache@v4, actions/upload-artifact@v4 | all CI wiring | Existing job vocabulary; no new actions needed |
| ty (astral) | 0.0.84 (live-verified via uvx) | advisory type-check step | Owner-locked D-08; `[tool.ty.src]` config already in pyproject; 261003-0p0 adopted it (570→165) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| uvx (uv tool runner) | installed on runner + dev box | pinned ty invocation without pyproject change | Alternative to dev-dep pin for the advisory step |
| PyYAML (in-venv) | >=6.0 | sandbox YAML patch + lock parsing in CI-08 test | Already a core dep; audit_skips.py precedent |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `giants` marker + `-m "not giants"` | typed `optional-dep:` skip | Owner explicitly rejected ("environment available yet skipped is dishonest") — forbidden |
| `OLLAMA_CONTEXT_LENGTH` env | per-request `options.num_ctx` | Per-request is unreachable for the pydantic_ai sibling (OpenAI-compat endpoint); only viable server-side |
| `OLLAMA_CONTEXT_LENGTH` env | Modelfile `num_ctx` re-tag | Changes the model tag → notebook content change → violates D-06 "notebook untouched" |
| ty dev-dep exact pin | `uvx ty@0.0.84` inline | uvx keeps pyproject untouched in Phase 9; dev pin matches the ruff==0.16.9 precedent and gives check_code.py a venv binary at the D-09 flip — either defensible; dev pin recommended if the step should be reproducible locally |
| Inline literal expected count | expected count from rollup file / env var | Inline literal in ci.yml is the simplest reviewable bump-point; env var adds indirection; rollup-derivation adds a parsing dependency on a planning file inside CI (wrong direction — planning is not CI input) |

**Installation (only if dev-dep pin chosen for ty):**

```bash
uv pip install "ty==0.0.84"   # or add to [project.optional-dependencies].dev next to ruff==0.16.9
```

**Version verification (this session):** `uvx ty --version` → `ty 0.0.84`; `pip index versions ty` → latest 0.0.84 (whole-version increments 0.0.x — fast-moving, exact pin warranted). ty is NOT currently installed in the project venv (PackageNotFoundError).

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| ty | PyPI | latest release 2026-09-24 (~11 days) | unknown (PyPI stats unavailable to the checker) | github.com/astral-sh/ty | SUS | Flagged — see mitigation below |

**Packages removed due to SLOP verdict:** none.
**Packages flagged suspicious [SUS]:** `ty` — reasons "too-new", "unknown-downloads". Mitigating evidence: (1) the dependency is owner-locked by D-08 (the owner decision itself is the human verification of intent); (2) publisher is Astral (the ruff vendor, already a hard-gate dependency at `ruff==0.16.9`); (3) the project already adopted ty in quick task 261003-0p0 (config block, 44 audited suppressions, E-family triage list). Per protocol the planner should still add a `checkpoint:human-verify` before any pyproject dev-dep addition, or sidestep entirely with the `uvx ty@0.0.84` form (no dependency added).

## Architecture Patterns

### System Architecture Diagram (example-nightly after Phase 9)

```
schedule "0 3 * * *" ──┬─> coverage-nightly   (gate: schedule=='0 3 * * *' || dispatch)
schedule "30 5 * * *" ─┴─> example-nightly    (gate: schedule=='30 5 * * *' || dispatch)
                              test-mamba      (gate: schedule=='0 3 * * *' || dispatch)
                                    │
        ┌───────────────────────────┘  (D-19: cron-string gates; today all 3 fire on both crons)
        v
example-nightly (timeout 2700min, HF_ENDPOINT=hf-mirror.com)
  stage 0: venv + full extras + bedtools prefix + mamba wheelhouse + megaDNA venvs
           [DELETED: evo venv / flash-attn wheelhouse / giants prefetch]
           [DELETED: models-cache restore (D-11)]
  NEW: census collection assertion (hard): collect-only == expected, else RED
  stage 1:   pytest tests/examples -k "not mcp_example" -m "not giants"   <-- giants deselected
             (sandbox YAML patch: finetune_custom_head epochs 1, test-side only)
  stage 1.5: NAMED hygiene step: pkill kernels + free -g assert (>=35Gi) + before/after log
  stage 2:   MCP server :8000 streamable probes, restart, sse probes
  stage 2.5: NAMED hygiene step (same shape as 1.5)
  stage 3:   fresh MCP server + mcp pair (ollama at num_ctx 8k via unit env)
  stage 4:   skip audits + FULL if:always() upload (junit + stage logs + census
             output + server logs) + fail-soft summary (non-zero if any stage failed)
```

### Recommended Project Structure (files this phase touches)

```
.github/workflows/ci.yml          # D-03/D-04/D-11/D-13/D-14/D-19 wiring + D-12 comment rewrites
pyproject.toml                    # giants marker registration (D-01); optional ty pin (D-08)
tests/examples/_execution.py      # D-05 sandbox YAML-patch seam (spec-driven)
tests/examples/test_notebook_execution.py  # giants mark application + D-05/D-07 contract tests
scripts/runner/ollama.service     # D-06 OLLAMA_CONTEXT_LENGTH=8192 (+ closes 0.0.0.0 drift)
scripts/runner/README.md          # D-06 owner re-apply instructions update
tests/ (new fast test file)       # CI-08 lock-consistency guard + D-07 unit-file contract
docs/ (new CI/testing page)       # D-15 coverage expectation + AUDIT-04 cross-ref; mkdocs.yml nav
09 rollup artifact                # D-02 baseline census counts + measured budgets
```

### Pattern 1: Marker registration + spec-derived mark application (D-01)

**What:** Register the marker, then apply it through the existing dynamic-parametrize-marks mechanism rather than scattering decorators.
**When to use:** Any "class of tests exits a lane by policy" need.

Registration is MANDATORY before use — addopts carries `--strict-markers` [VERIFIED: pyproject.toml:503-510, markers list at 511-521 with the existing 10 markers: `slow, pdf, performance, integration, unit, inference, utils, data, legacy`].

The in-file precedent for deriving marks per-parametrization is exactly the timeout-override comprehension [VERIFIED: tests/examples/test_notebook_execution.py:1145-1153]:

```python
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

Recommended placement: a `_GIANTS_GATED: frozenset[str]` (mirroring `_TIMEOUT_7200_GATED` at lines 1126-1130) currently holding the one evo id `"notebooks/generation_evo_models/inference.ipynb"` (GATED_NOTEBOOKS entry at line 1104), applied as `marks=[pytest.mark.giants]` in the same comprehension, OR a `"markers": ["giants"]` key in the spec consulted the same way. The fast evo contract tests (TestEvoIsolatedLane, 3 tests at lines 747-770; TestSpecEnvOverrides evo rows, lines 585-588) should stay UNMARKED — they are kernel-free monkeypatched contract tests that belong on every fast leg; marking them would shrink fast-leg coverage for no runtime win. (5-test "evo lane" vs 1-test scope is planner's call inside this discretion; the 08-07 census counted 5 items, but 4 of them are sub-second fast-lane tests.)

### Pattern 2: Collection-count hard assertion (D-03)

**What:** A hard (non-fail-soft) step asserting the collect-only counts before stage 1.
**When to use:** Wherever a deselect policy could silently shrink a census.

Live-verified summary-line shapes [VERIFIED: live probe 2026-10-05]:

```
pytest tests/examples --collect-only -q                -> "197 tests collected in 0.89s"
pytest tests/examples --collect-only -q -m "not slow"  -> "167/197 tests collected (30 deselected) in 0.89s"
pytest tests/examples --collect-only -q -k "not mcp_example" -> "189/197 tests collected (8 deselected) in 0.87s"
```

The assertion should pin the TRIPLE from the stage-1 selector set (`-m "not giants" -k "not mcp_example"`): total (catches import/collection breakage), deselected (catches marker typos and accidental over-marking), selected (the number that must run). With one giants-marked test the expected values are 188/197 (9 deselected = 8 mcp + 1 giants) — final numbers come from the D-02 baseline run. Recommended form (inline literal, comment pointing at the 09 rollup):

```bash
# Stage 0.5: census collection assertion (D-03) — hard gate, census integrity
COLLECTED=$(.venv/bin/python -m pytest tests/examples --collect-only -q \
  -m "not giants" -k "not mcp_example" 2>/dev/null | tail -1)
echo "census collection: $COLLECTED"
echo "$COLLECTED" | grep -qE "^188/197 tests collected \(9 deselected\) in " \
  || { echo "FAIL: census collection mismatch (expected 188/197, 9 deselected)"; exit 1; }
```

(A stage-0 failure fails the job directly — consistent with the existing contract that stage-0 infra failures are honest job failures, ci.yml:550-553.)

### Pattern 3: Sandbox-only YAML patch (D-05)

**What:** Test-side override applied to the seeded sandbox copy after `seed_sandbox`'s copytree, driven by a spec key.
**When to use:** Any runtime cut that must not touch committed example content.

Verified facts: the notebook reads `configs = load_config("./finetune_config.yaml")` [VERIFIED: example/notebooks/finetune_custom_head/finetune.ipynb source, cell at JSON line 53] and the committed YAML carries `num_train_epochs: 3` inside the `finetune:` section [VERIFIED: example/notebooks/finetune_custom_head/finetune_config.yaml:38]. The gated sandbox fixture seeds `nb_path.parent` (whole dir, including the YAML) [VERIFIED: tests/examples/test_notebook_execution.py:1138-1143], and `seed_sandbox` returns the sandbox path after its copies [VERIFIED: tests/examples/_execution.py:326-388].

Two seams, both honest:
1. **`seed_sandbox` hook parameter** (recommended): an optional `yaml_overrides: dict[str, dict]` (relative path → nested key patch) applied post-copy inside `seed_sandbox` — one implementation point, unit-testable kernel-free like TestSeedSandbox (lines 293-357), reusable for future cuts.
2. **Spec key + fixture application**: the gated fixture reads `spec.get("yaml_patch")` and rewrites the sandbox YAML — keeps `seed_sandbox` pure but spreads patch logic to every call site.

Same-change test (D-07): a fast test seeding a fake dir, applying the patch, asserting the sandbox copy has `num_train_epochs: 1` while the source file still reads `3`.

### Pattern 4: Server-default num_ctx via the in-repo unit (D-06)

**What:** `OLLAMA_CONTEXT_LENGTH=8192` in `scripts/runner/ollama.service`, re-applied by the owner (one systemctl op), asserted by a same-change contract test parsing the unit file.
**When to use:** The only seam that cuts BOTH mcp notebooks without touching notebook content.

Verified basis:
- Official FAQ: "to set the default context window to 8K, use: `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`" [CITED: docs.ollama.com/faq + docs.ollama.com/context-length]. Precedence: per-request `options.num_ctx` > Modelfile `PARAMETER num_ctx` > `OLLAMA_CONTEXT_LENGTH` > built-in default; env is read at server start (restart required) [CITED: docs.ollama.com/faq].
- Live probe on the runner's box [VERIFIED: live 2026-10-05]: ollama 0.34.1; `qwen3.8:latest` present (17.74GB, `context_length: 262144` — the 256k the owner measured); `ollama show qwen3.8:latest --modelfile` shows **no** `PARAMETER num_ctx` → the env default WILL govern for this model.
- Per-request refuted for the pair: pydantic_ai sibling builds `OpenAIChatModel(model_name='qwen3.8:latest', provider=OllamaProvider(base_url='http://localhost:11434/v1'))` [VERIFIED: example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb source] — the OpenAI-compat endpoint has no `num_ctx` parameter, so only a server-side default reaches it. The langchain sibling (`create_agent(model="ollama:qwen3.8:latest", ...)`) could take `num_ctx` per-request, but a mechanism that covers only one sibling of the pair is the wrong shape.
- Same-change test (D-07): a fast contract test asserting `scripts/runner/ollama.service` contains the `OLLAMA_CONTEXT_LENGTH` pin (file-parse precedent: TestLoraMirrorEndpoint-style spec pins; the unit header itself declares the file "auditable and rebuildable by diff" [VERIFIED: scripts/runner/ollama.service:1-9]).
- README update: scripts/runner/README.md documents the one-time owner install (lines 9-28) — add the re-apply step (systemctl edit / restart) and the num_ctx rationale.

### Pattern 5: ty advisory step (D-08)

**What:** A seconds-scale advisory step on the coverage-gate fast leg, same layer as ruff `--statistics`.
**When to use:** Now; the hard flip + mypy retirement is the later atomic quick task (D-09).

Precedents, all verified: ruff statistics step `ruff check . --statistics` [VERIFIED: ci.yml:85-91]; mypy advisory shape `mypy dnallm/ --show-error-codes --pretty --exclude=dnallm/tasks/metrics/ || true` [VERIFIED: ci.yml:121-124]; `[tool.ty.src]` exclude block already configured with the comment documenting flagless `uvx ty check dnallm/` [VERIFIED: pyproject.toml:395-404]. Live baseline re-measured this session: **"Found 167 diagnostics"** with ty 0.0.84 (was 165 at 261003-0p0 close — +2 drift from subsequent quick tasks; the E-family triage quick task inherits whatever count is current).

```yaml
- name: Run type checking (ty, advisory until E-family zero)
  run: |
    source .venv/bin/activate
    uvx ty@0.0.84 check dnallm/ || true   # advisory per D-08; hard flip lands with mypy retirement (D-09)
```

or with the dev-dep pin: `.venv/bin/ty check dnallm/ || true`. Placement: coverage-gate job (ci.yml:356-414), after the skip-audit step. Do NOT touch the mypy steps, pre-commit, or check_code.py in Phase 9 (D-09 boundary).

### Pattern 6: Lock-consistency guard (CI-08)

**What:** A new fast-leg test cross-checking model id literals in notebooks/apps against models.lock.
**When to use:** Always (push/PR fast legs run tests/).

Verified lock format [VERIFIED: models.lock:1-34]: header comment declares the pinned form `<hf|ms>  <repo-id>@<revision-sha>  # purpose` and the prefix-alignment rule ("ms ⇒ source=\"modelscope\", hf ⇒ source=\"huggingface\""); 24 model rows + 1 `dataset:` row. No existing test references models.lock (grep verified). The guard should (a) parse lock repo-ids, (b) scan `example/**/*.ipynb` (JSON code-cell source) + marimo apps + example YAML configs for model-id-shaped literals matching lock families (e.g. `plant-dnagpt-BPE`, `evo-1-8k-base`, `megaDNA_updated`), (c) fail on any example-referenced remote model id absent from the lock, and (d) optionally assert ACTIVE `source=` route alignment per row comment — extending, not duplicating, the existing per-notebook route contracts (TestMegadnaSiblingContentContracts.test_source_routes_are_d15_aligned, lines 887-903).

### Anti-Patterns to Avoid

- **Typed skip as the evo exit** — owner-rejected; also silently pollutes the skip census (D-01 exists precisely to avoid this).
- **Editing the committed `finetune_config.yaml` or notebooks for the runtime cuts** — violates D-05/D-06 "notebook untouched"; the executability claim stays on committed content.
- **`continue-on-error` anywhere in example-nightly** — the job's design prohibits forever-green (ci.yml:550-553); the fail-soft pattern is exit-code recording + summary red, not continue-on-error.
- **nvidia-smi as the VRAM assertion on GB10** — GB10 reports `[N/A]` for memory queries [VERIFIED: 08-CENSUS-ROLLUP.md:178-180]; the operative guard is `free -g` available-memory with the >=35Gi floor discipline [VERIFIED: 08-CENSUS-ROLLUP.md:127-129, 262-267].
- **Expecting `.scratch/` venvs to survive between jobs** — the self-hosted runner cleans the work dir between jobs; that is why stage 0 provisions per-run (and why the deleted evo steps cannot be "kept but skipped").
- **Deleting any local cache as part of this phase** — owner rule: local caches are never cleaned by default; no plan step may propose deletion without explicit owner approval.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cron disambiguation | Per-job schedule-splitting into multiple workflow files | `github.event.schedule == '<cron>'` in the job `if:` | Documented GitHub pattern [CITED: docs.github.com events-that-trigger-workflows; latchkey.dev guide]; one line per job |
| Context-length default | Wrapper scripts / client monkeypatch shims per notebook | `OLLAMA_CONTEXT_LENGTH` in the systemd unit | Officially documented server default; covers both client stacks uniformly |
| Census counting | junit parsing to count ran tests | `pytest --collect-only -q` summary triple | Deterministic (all parametrize sources are repo-content-derived, environment-independent); the junit route conflates skips/deselects |
| Skip allowlisting | Ad-hoc grep on pytest stdout | existing `scripts/audit_skips.py` + `tests/expected_skips.yaml` | Fail-closed, matcher-validated, already wired into stage 4 |
| Sandbox YAML editing | Regex-on-file string surgery | PyYAML load/dump (or targeted line edit) inside the seed seam | YAML round-trip with `yaml.safe_load` is the audit_skips.py precedent; string surgery breaks on comments/format drift — note the committed YAML has heavy comments, so prefer a minimal targeted `num_train_epochs` line replacement in the copy if comment preservation matters for artifact readability |

**Key insight:** every mechanism this phase needs already exists in the repo with an in-file precedent (deselect line, spec env, spec-derived marks, contract-test class shapes, fail-soft recording, tee'd log upload). The work is applying them, not inventing.

## Runtime State Inventory

> Included because this phase edits live-service config (ollama unit) and CI-registered state (cache store, schedules).

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | GitHub cache store: exactly 2 entries (`Linux-uv-cuda-*`, 10.37GB total, at the 10GB eviction threshold); ZERO models-cache entries ever saved [VERIFIED: 08-CENSUS-ROLLUP.md:197-204] | Code edit only — remove the cache steps (D-11); no cache-store deletion needed or proposed |
| Live service config | **ollama systemd unit on the runner: LIVE divergence from the in-repo definition — `OLLAMA_HOST=0.0.0.0:11434` + box-specific PATH vs the in-repo `127.0.0.1:11434` pin** [VERIFIED: live `systemctl show ollama -p Environment` 2026-10-05; in-repo pin at scripts/runner/ollama.service:30] | D-06 unit edit (add `OLLAMA_CONTEXT_LENGTH=8192`) + owner re-apply ALSO restores the loopback pin — flag to owner explicitly; also note qwen3.8:latest currently loaded/available, no Modelfile num_ctx |
| OS-registered state | ollama.service (systemd, owner-enabled once); dnallm-nightly actions-runner service (separate unit, untouched) | Owner op: `systemctl edit`/copy + `daemon-reload` + `restart ollama` (env is read at server start [CITED: docs.ollama.com/faq]); runner restarts must keep using sanitized env (env -i) per ops note [VERIFIED: scripts/runner/README.md:49-55] |
| Secrets/env vars | None in play — workflows default `permissions: contents: read` [VERIFIED: ci.yml:23-24]; no secrets referenced by any nightly job | None |
| Build artifacts | `.scratch/` throwaway venvs wiped by the runner between jobs (per-job stage-0 rebuilds by design); local `~/models-giants` (15GB) + `~/.cache/{huggingface,modelscope}` retained forever per owner rule | None — deletion forbidden without owner approval; D-04 keeps local giants assets for the dispatch/manual lane |
| Workflow-schedule state | Both cron entries live only on `phs` today; schedules fire from the DEFAULT branch post-merge [VERIFIED: 08-CENSUS-ROLLUP.md:139-141] | D-19 gate fix MUST land before phs→main integration, else the double-trigger goes live at first merged schedule |

## Common Pitfalls

### Pitfall 1: GB10 reports no VRAM through nvidia-smi
**What goes wrong:** A VRAM assertion built on `nvidia-smi --query-gpu=memory.used` reads `[N/A]` on GB10 and either always-fails or always-degrades to an echo.
**Why it happens:** GB10's driver doesn't expose memory totals via that query on this box [VERIFIED: 08-CENSUS-ROLLUP.md:178-180 runner inventory].
**How to avoid:** Assert on `free -g` available memory with the established >=35Gi floor (08-08/08-09 discipline); keep nvidia-smi output as best-effort telemetry only (the existing steps' `|| echo` fallback already treats it as such, ci.yml:813, 873).
**Warning signs:** A new hygiene step that is green on every runner while memory is actually exhausted.

### Pitfall 2: `--strict-markers` makes the marker registration order-load-bearing
**What goes wrong:** Applying `pytest.mark.giants` before adding it to `[tool.pytest.ini_options] markers` breaks COLLECTION of every pytest invocation repo-wide (exit code, not a warning).
**Why it happens:** addopts carry `--strict-markers` [VERIFIED: pyproject.toml:506].
**How to avoid:** Register the marker in pyproject in the same atomic change (same commit) as the first mark application and the ci.yml deselect.
**Warning signs:** Fast-lane collection errors mentioning "Unknown criterion marker" / `giants`.

### Pitfall 3: The double-trigger is invisible until merge
**What goes wrong:** Fixing D-19 "later" still ships a window where every nightly job runs twice per day (03:00 AND 05:30), stacking multi-hour jobs on the single queue-serialized runner.
**Why it happens:** Both schedule entries trigger the whole workflow; job gates compare only `github.event_name` [VERIFIED: ci.yml:275, 423, 525 — all three identical gates]. Schedules only fire from the default branch, so the defect activates at phs→main integration, not on phs dispatches.
**How to avoid:** Land the cron-string gates (`github.event.schedule == '0 3 * * *'` for coverage-nightly/test-mamba, `== '30 5 * * *'` for example-nightly, each keeping the `workflow_dispatch` disjunct) in this phase — and fix all THREE jobs, since example-nightly firing at 03:00 alongside coverage-nightly is the same defect the checker flagged from the other side.
**Warning signs:** Two example-nightly runs per day in the actions list post-merge.

### Pitfall 4: Collection counts are deterministic but only from repo content
**What goes wrong:** An expected count pinned from a stale run, or one computed with different selectors than stage 1 (`-m` and `-k` AND together), drifts and reds spuriously — or worse, the assertion uses a selector set that doesn't include the giants filter and never guards the marker at all.
**Why it happens:** The count depends on selector composition; today's verified values are 197 total / 189 with `-k "not mcp_example"` / 167 with `-m "not slow"`.
**How to avoid:** Pin the triple (total/deselected/selected) with the exact stage-1 selector flags; refresh the literal from the D-02 baseline run; bumping the literal is a deliberate review act (that is the guard working, not friction).
**Warning signs:** A census-growth PR failing the collection step with no test removed.

### Pitfall 5: Modelfile num_ctx would silently defeat the env pin
**What goes wrong:** `OLLAMA_CONTEXT_LENGTH=8192` appears not to work — kv-cache stays huge.
**Why it happens:** Modelfile `PARAMETER num_ctx` outranks the env default [CITED: docs.ollama.com/faq].
**How to avoid:** Already probed live: `qwen3.8:latest` has NO Modelfile num_ctx [VERIFIED: live `ollama show --modelfile` grep 2026-10-05], so the env governs. If the model is ever re-pulled or replaced, re-probe before assuming.
**Warning signs:** Stage-3 latency/VRAM unchanged after the unit change; `ollama ps` showing a large context while loaded.

### Pitfall 6: Flaky github egress vs the green-dispatch gate (D-16/D-17/D-18)
**What goes wrong:** Dispatch/verification steps (`gh workflow run`, `gh run watch`, artifact download) hang or fail — this session's own probe of https://github.com timed out while `gh auth status` succeeded [VERIFIED: live 2026-10-05].
**Why it happens:** Known intermittent egress on this box (phase brief); the runner pulls from mirrors fine, but control-plane calls from the dev box are the flaky path.
**How to avoid:** Make dispatch-watch steps retry-tolerant and non-load-bearing for plan-verify (the known-env-facts rule); the D-17 green dispatch is a phase gate, so schedule it as its own step with retries, and treat a hung watch as "still running", not failed. D-18's re-dispatches explicitly wait for egress stability.
**Warning signs:** gh commands hanging past their timeouts while curl to hf-mirror returns 200.

### Pitfall 7: Sum-of-ceilings comments are prose, not arithmetic
**What goes wrong:** Rewriting the timeout comments from recomputation instead of measurement reproduces the original problem (the current example-nightly comment's ~2620min figure does not transparently re-derive from the marks it names [VERIFIED: ci.yml:541-563]; coverage-nightly's 840min figure predates the showcase marks, with 07-02 noting ~970min on paper vs the 900-min cap [VERIFIED: STATE.md 07-02 hand-off; CONCERNS.md:133]).
**Why it happens:** Different comments count different mark sets (e.g. whether fast-contract 300s defaults are included).
**How to avoid:** D-12 is explicit: write MEASURED actuals (the D-02 baseline run) into both comments, and state the per-test-marks-primary / job-kill-backstop relationship; if coverage-nightly's recomputed ceiling exceeds 900 on paper, either raise its `timeout-minutes` above the new sum or document the intentional override — decide with the measurement in hand.
**Warning signs:** A rewritten comment whose numbers no reader can reproduce from the test files.

### Pitfall 8: Workflow README staleness compounds
**What goes wrong:** `.github/workflows/README.md` describes only the 03:00 nightly and omits the 05:30 schedule/staged contract [VERIFIED: CONCERNS.md:67] — editing ci.yml without it deepens the drift.
**How to avoid:** Fold a short README refresh into whichever plan edits the schedule gates (cheap, same-review-unit).
**Warning signs:** Two docs describing different nightly topologies.

## Code Examples

### Stage-1 deselect line (D-01, existing line + one flag)

```yaml
# Source: .github/workflows/ci.yml:798 (current) — Phase 9 adds -m "not giants"
- name: "Stage 1: torch-heavy example execution (ACTIVE + showcase + marimo + script + YAML leg)"
  run: |
    set +e
    .venv/bin/python -m pytest tests/examples -q --junitxml=pytest-junit-example-stage1.xml \
      -k "not mcp_example" -m "not giants"
```

### Named hygiene step with GB10-safe assertion (D-13, generalizes ci.yml:806-813/866-873)

```yaml
- name: "Stage 1.5: hygiene — kernel cleanup + memory floor assertion (D-13)"
  run: |
    echo "before: $(free -g | awk '/^Mem:/{print $7"Gi available"}')"
    pkill -f ipykernel_launcher || true
    sleep 5
    AVAIL_GI=$(free -g | awk '/^Mem:/{print $7}')
    echo "after:  ${AVAIL_GI}Gi available"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv || echo "nvidia-smi MISSING (telemetry only on GB10)"
    if [ "$AVAIL_GI" -lt 35 ]; then
      echo "FAIL: only ${AVAIL_GI}Gi available (floor 35Gi before a heavy stage — 08-08/08-09 discipline)"
      exit 1
    fi
```

(Assertion step is hard; it sits between fail-soft stages and a failure here means the box cannot safely continue — matches the >=35Gi discipline recorded in 08-CENSUS-ROLLUP.md:127-129. Whether the floor asserts at 1.5/2.5 or only before heavy stages is planner's choice; the 08-09 record asserts before heavy stages.)

### Full if:always() upload (D-14, extends ci.yml:924-931)

```yaml
- name: "Stage 4: upload junit artifacts"
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: example-nightly-junit
    path: |
      pytest-junit-*.xml
      mcp-server-*.log
      stage-results.txt
      stage*.log
      census-collect.txt
```

Per-stage logs require tee at each stage — the test-mamba precedent is verbatim in ci.yml:343-345 (`set -o pipefail` + `2>&1 | tee pytest.log`).

### D-19 gate (one line per nightly job)

```yaml
# coverage-nightly + test-mamba
if: github.event_name == 'workflow_dispatch' || (github.event_name == 'schedule' && github.event.schedule == '0 3 * * *')
# example-nightly
if: github.event_name == 'workflow_dispatch' || (github.event_name == 'schedule' && github.event.schedule == '30 5 * * *')
```

[Pattern CITED: GitHub docs events-that-trigger-workflows — `github.event.schedule` carries the fired cron string; community-pattern guide latchkey.dev]

### Sandbox YAML patch seam sketch (D-05, Pattern 3 option 1)

```python
# tests/examples/_execution.py — extend seed_sandbox (kernel-free unit-testable)
def seed_sandbox(src_dir, tmp_path, extra_inputs=None, yaml_overrides=None):
    ...  # existing copytree + extras
    for rel_path, patch in (yaml_overrides or {}).items():
        target = sandbox / rel_path
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        for section, kv in patch.items():
            data[section].update(kv)
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return sandbox
```

Caveat for the planner: `yaml.safe_dump` drops the committed file's comments in the SANDBOX COPY only (acceptable — the sandbox is tmp scratch), but if comment fidelity matters, do a targeted line replacement on the copy instead. Spec-side: `NOTEBOOK_EXEC_SPECS[str(nb_path)]["yaml_patch"] = {"finetune_config.yaml": {"finetune": {"num_train_epochs": 1}}}` consulted by the gated fixture (spec-env precedent at _execution.py:214-230).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| evo/giants inside the pytest census (per-run venv + flash-attn build + giants prefetch) | `giants` marker + deselect; dispatch/manual lane only | Owner directive 2026-10-05 | Stage 0 loses ~50-90min one-time builds + 12.9GB prefetch; flash-attn bug dissolves (D-04) |
| models-cache layer (never successfully saved; 15.2GiB lock-only > 10GB quota) | Cold pulls (65-min cold stage 1 proven vs 2700-min budget) | Owner decision 2026-10-05 | Two actions/cache blocks deleted; local caches remain the persistence layer (owner rule) |
| mypy advisory everywhere | ty advisory now; ty hard + mypy retired in one later atomic change | Owner decision 2026-10-05 (flip deferred) | Phase 9 adds only the advisory step (D-08); D-09 atomicity forbids partial retirement now |
| Single-cron nightly (03:00) | Dual cron (03:00 + 05:30 stagger) — gates not yet cron-aware | 08-09 wiring | D-19 makes gates cron-aware before the second cron goes live on main |
| Fixed num_ctx behavior (256k kv-cache, 36GB) | Server-default 8k via unit env | This phase (D-06) | Cuts mcp-pair latency + the VRAM trough; uniform across both client stacks |

**Deprecated/outdated in-repo docs to touch while editing:** `.github/workflows/README.md` (only knows 03:00); the ci.yml schedule-block comment (lines 14-17) whose claim "the example-nightly job below carries its own event gate; the other schedule-gated jobs fire per their own cron entries" is mechanically false — event gates cannot distinguish crons; `github.event.schedule` can.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The two `trainer.train()` calls in finetune_custom_head (notebook JSON lines 281, 673) both read `num_train_epochs` from the single `finetune_config.yaml` load, so one patched key cuts both trainings (~31→~11min claim) | Pattern 3 | Only one training is cut — measured baseline (D-02) exposes the true delta immediately; low risk |
| A2 | `uvx ty@0.0.84 check dnallm/` syntax (pkg@version + args) works in the CI step exactly as plain `uvx ty check dnallm/` did locally | Pattern 5 | Fall back to dev-dep exact pin (ruff==0.16.9 precedent); trivial |
| A3 | Marker scope = the 1 gated evo execution test (not the 4 fast evo contract tests) is the right default | Pattern 1 | Planner/owner may prefer the 5-test lane; either way counts are pinned by D-03 so nothing is silent |
| A4 | `free -g` "available" column is the >=35Gi discipline's metric (08-09 recorded "115Gi avail" style values matching this column) | Pitfall 1 / D-13 example | If the discipline used a different column, the assertion needs the owner's exact metric — confirm at execution; cheap to adjust |
| A5 | A new docs CI/testing page under the existing nav (e.g. User Guide or a development section) is acceptable placement for D-15 | CI-09 | Pure discretion; owner may want a different location — wording and nav entry are planner's call |
| A6 | evo execution on the local dispatch/manual lane remains runnable post-D-04 (local venv + giants tier intact per owner rule) | D-04 | If the local evo venv was on `.scratch` and wiped, the dispatch lane needs a documented rebuild path (08-06 recipe exists in-repo via ensure_evo_kernel + stage-0 history) |

## Open Questions

1. **Coverage-nightly vs the giants marker**
   - What we know: coverage-nightly runs the FULL suite (no `-m` filter, ci.yml:507) and the evo gated test currently typed-skips there (cold venv, `optional-dep:` prefix allowlisted — audit green).
   - What's unclear: should coverage-nightly ALSO deselect giants (`-m "not giants"`), or keep the honest typed skip?
   - Recommendation: leave coverage-nightly unchanged in this phase (minimal wiring; the skip is allowlisted and honest); note it in the 09 rollup. If the owner prefers census purity, it is a one-flag addition later.
2. **Exact giants-marker scope (A3) and the resulting census triple**
   - What we know: 197 total collected; stage-1 selectors already deselect 8 (mcp); 1 vs 5 giants candidates.
   - What's unclear: owner preference on including the 4 fast contract tests.
   - Recommendation: mark only the execution test; D-02 baseline fixes the literal either way.
3. **D-12 backstop arithmetic for coverage-nightly**
   - What we know: 900-min kill vs ~970-min recomputed on-paper ceiling (07-02 note); measured showcase actuals ~15 min.
   - What's unclear: raise `timeout-minutes` above the recomputed sum, or keep 900 with a documented per-test-marks-primary rationale.
   - Recommendation: decide after the D-02 measured run; the 08-CENSUS steady-state numbers (stage 1 ~3h, stages 2/3 ~45min) suggest actuals sit far below both figures, so the comment rewrite may simply record measured totals and keep both backstops.
4. **Runner cache-store eviction posture after D-11**
   - What we know: store sits at 10.37GB (threshold); removing the models-cache layer stops nightly save attempts.
   - What's unclear: whether the two uv-cuda entries should be pruned (owner action; GitHub-side).
   - Recommendation: out of scope (no deletion without owner action); mention in rollup only.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Self-hosted runner `dnallm-nightly` (GB10) | all nightly dispatches | ✓ (shared $HOME with dev box; free -g: 121 total / 114 available GiB) | — | none (single box by design) |
| ollama (loopback service) | D-06 / stage 3 | ✓ (live: 0.34.1, qwen3.8:latest loaded-listed) | 0.34.1 | typed network-unavailable skip (existing gate) |
| gh CLI (dispatch/watch) | D-16/D-17/D-18 | ✓ installed + authenticated (forrestzhang) | brew latest | retry on egress flake; owner can dispatch from UI |
| github.com egress from dev box | dispatch verification | ✗ INTERMITTENT (live probe timed out this session; gh auth worked) | — | retry-tolerant steps; never load-bearing at plan-verify time |
| uvx / uv | ty advisory step | ✓ (uvx ty 0.0.84 ran live this session) | 0.0.84 resolved | dev-dep exact pin |
| Local `~/models-giants` + hf/ms caches | dispatch/manual evo lane; nightly warm path | ✓ retained (owner rule; hub at 19GB post-owner-cleanup) | — | cold pulls (65-min stage 1 proven) |
| pytest collection on this box | D-03 numbers | ✓ verified (197 / 189 / 167 shapes) | pytest in .venv | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** github egress for dispatch-watch (retry + UI dispatch).

## Security Domain

`security_enforcement: true`, ASVS L1. This phase touches CI config and one systemd unit — no application code, no authn/authz surfaces, no user input parsing.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surfaces touched (gh auth is operator-side) |
| V3 Session Management | no | — |
| V4 Access Control | yes (network) | ollama loopback bind is the access control [VERIFIED: scripts/runner/README.md:30-36]; the LIVE unit drift (`OLLAMA_HOST=0.0.0.0:11434`) is an open exposure of an unauthenticated model server on every interface — the D-06 unit re-apply must restore `127.0.0.1:11434` |
| V5 Input Validation | yes (CI-side) | audit_skips.py parses junit XML fail-closed with no entity resolution [VERIFIED: scripts/audit_skips.py:83-92]; the new CI-08 lock test parses trusted in-repo files only; collection-assertion greps pytest's own output |
| V6 Cryptography | no | — |
| V14/Config | yes | Workflow `permissions: contents: read` least-privilege default preserved [VERIFIED: ci.yml:23-24]; no new secrets; no new third-party actions introduced |

### Known Threat Patterns for CI-wiring + local model server

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Unauthenticated model server on non-loopback bind (LIVE drift found this session) | Information Disclosure / Elevation | In-repo unit pins `OLLAMA_HOST=127.0.0.1:11434`; owner re-apply restores it; never copy the FAQ's 0.0.0.0 example [VERIFIED: scripts/runner/ollama.service:11-15] |
| Fork/PR code executing on the self-hosted GPU box | Elevation | All nightly jobs stay gated `schedule || workflow_dispatch` (never push/PR) [VERIFIED: ci.yml:275, 423, 525]; D-19 must preserve this while adding cron-string matching |
| Runaway job monopolizing the single runner | DoS | timeout-minutes backstops (180/900/2700) + per-test marks; D-12 keeps backstops above measured sums |
| Artifact leakage of credentials | Information Disclosure | Runner inventory probe echoes only command locations/disk/GPU/HTTP codes, never env dumps [VERIFIED: ci.yml:781-790]; uploads contain junit/logs only |
| Prompt-injection via notebook/model output executing in CI | Elevation | Out of scope here (notebooks are first-party, reviewed content; no untrusted-source notebooks execute) |

## Sources

### Primary (HIGH confidence)
- Repo reads this session: `.github/workflows/ci.yml` (all 993 lines), `tests/examples/_execution.py` (all 1134), `tests/examples/test_notebook_execution.py` (all 1194), `pyproject.toml`, `tests/expected_skips.yaml`, `scripts/audit_skips.py`, `scripts/check_code.py`, `.pre-commit-config.yaml`, `models.lock`, `scripts/runner/ollama.service`, `scripts/runner/README.md`, `example/notebooks/finetune_custom_head/finetune_config.yaml`, `.planning/phases/08-.../08-CENSUS-ROLLUP.md`, `.planning/codebase/TESTING.md`, `.planning/codebase/CONCERNS.md` (grep), both mcp notebooks (client-construction cells)
- Live probes on the runner's box (2026-10-05): pytest collection shapes (197/189/167); `uvx ty check dnallm/` → 167 diagnostics, ty 0.0.84; ollama 0.34.1 / qwen3.8 modelfile+tags; `systemctl show ollama` env drift; `free -g`; github.com egress timeout; pip index versions ty
- Knowledge graph `.planning/graphs/graph.json` (fresh 2h) — confirmed seam locations (seed_sandbox, assert_tree_clean, census nodes)

### Secondary (MEDIUM confidence)
- [GitHub docs — Events that trigger workflows](https://docs.github.com/en/actions/reference/events-that-trigger-workflows) — `github.event.schedule` cron-string semantics; [Latchkey guide](https://latchkey.dev) — multi-cron branching pattern
- [Ollama docs — Context length](https://docs.ollama.com/context-length) and [FAQ](https://docs.ollama.com/faq) — `OLLAMA_CONTEXT_LENGTH=8192 ollama serve` default, precedence (per-request > Modelfile > env > default), server-start read

### Tertiary (LOW confidence)
- None — no claim in this document rests on unverified training memory

## Metadata

**Confidence breakdown:**
- CI wiring facts: HIGH — every edited line range read this session with line numbers
- Runtime-cut seams (D-05/D-06): HIGH — notebook/YAML/unit/service all opened; num_ctx feasibility live-probed (Modelfile has no num_ctx); per-request route affirmatively refuted for the pydantic_ai sibling
- Census/timeout numbers: HIGH for current counts (live-collected); the post-change expected values are formulas pending the D-02 baseline run (by design, D-02/D-12)
- ty: HIGH for version/baseline (live); SUS registry verdict acknowledged with mitigation
- Security finding (ollama bind drift): HIGH — live systemctl output quoted verbatim in Runtime State Inventory context

**Research date:** 2026-10-05
**Valid until:** 2026-11-05 (repo facts are commit-bound; ollama/ty/gh environment facts ~7-day freshness — re-probe ollama Modelfile if the model is re-pulled)
