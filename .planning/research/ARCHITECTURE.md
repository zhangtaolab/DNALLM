# Architecture Research

**Domain:** Integration of v1.3 "Platform Reach & Packaging Honesty" onto the existing DNALLM architecture — Windows platform parity (numpy genomic interval evaluators), packaging/extras honesty, publish-chain robustness, docs accuracy, ollama test-surface removal, GAP-1 transformers 5.x remote-code compat
**Researched:** 2026-10-11 (milestone v1.3, branch `v1.3`)
**Confidence:** HIGH for all codebase integration points — every claim verified against the working tree (pyproject.toml, ci.yml, publish.yml, `tests/examples/*`, `example/notebooks/finetune_NER_task/*`, `example/notebooks/plant_helixseek_cre/*.ipynb`, `dnallm/utils/genomic_coords.py`, `dnallm/utils/transformers_compat.py`, `.planning/WINDOWS.md` ledger #11–#20, spike summaries 261010-wx2 / 261011-0a3). MEDIUM only for the two externally-grounded publish-chain topics (web-sourced, cross-checked) and where a planning-time decision is deliberately open (flagged inline as **decision needed**).

---

## Standard Architecture

### System Overview

The v1.3 features decompose into **five integration shapes**, none of which restructures a layer:

1. **New import-pure library module** — the numpy interval evaluators land as a sibling of `dnallm.utils.genomic_coords` (the sanctioned precedent for genomic helper modules at 100% statement coverage).
2. **Runtime dispatch at example-code acquisition seams** — platform forks inside `example/` code and its `docs/example/` byte-identical mirrors; the library itself stays platform-uniform (dispatch lives in consumers, never in `dnallm/`).
3. **Platform-aware test gates** — the three existing probe-gate sites flip from "skip when pybedtools/bedtools absent" to "pass via the numpy path on Windows"; the frozen truth chain (selection.md bands) stays single-sourced.
4. **CI/publish workflow surgery** — a new publish-smoke dry-run job, test-windows promoted to the numpy-path executor, ollama stage-3 excision from example-nightly, extras re-scope in pyproject.
5. **Compat-shim extension** — GAP-1 extends the existing absence-gated attach pattern in `dnallm/utils/transformers_compat.py` (or removes the affected surfaces).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CI / PUBLISH LAYER (workflow files)                                          │
│  ci.yml:  test-windows → numpy-path executor (frozen-value fast tests)       │
│           example-nightly: stage 3 + 2.5 excised (ollama directive),        │
│           D-03 census triple re-pinned; test-cuda matrix updated on         │
│           cuda121 retirement                                                 │
│  publish.yml: NEW publish-smoke job (periodic workflow_dispatch dry-run;    │
│           OIDC trusted-publishing migration lands here or is deferred)      │
├──────────────────────────────────────────────────────────────────────────────┤
│ EXAMPLE LAYER (example/ + byte-identical docs/example/ mirrors)              │
│  generate_bpe_dataset.py:7,249   ← platform fork (import + loj call)        │
│  data_generation_and_inference.ipynb cells 0/2/8 ← same fork + install cell │
│  plant_helixseek_cre.ipynb cell 11 ← jaccard subprocess fork                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ TEST-GATE LAYER (tests/examples/)                                            │
│  test_script_execution.py:128  _gate_script_pybedtools → platform-aware      │
│  test_notebook_execution.py:1397 _gate_pybedtools (via _ACTIVE_             │
│    NOTEBOOK_GATES) → platform-aware;  ollama gates/entries removed          │
│  test_plant_helixseek_showcase.py:523 which('bedtools') assert → fork       │
│  NEW fast frozen-value tests (not slow-marked) → run on test-windows        │
├──────────────────────────────────────────────────────────────────────────────┤
│ LIBRARY LAYER (dnallm/) — platform-uniform, no dispatch here                 │
│  NEW dnallm/utils/genomic_intervals.py:  parse_bed(BED3+), jaccard(a,b),    │
│    loj_intersect(a,b) + pinned null-fill; numpy-only, outside every         │
│    coverage omit glob; re-exported from dnallm/utils/__init__.py            │
│  dnallm/utils/transformers_compat.py: GAP-1 absence-gated attach of         │
│    PretrainedConfig defaults (is_decoder/add_cross_attention) — or removal  │
├──────────────────────────────────────────────────────────────────────────────┤
│ PACKAGING SURFACE (pyproject.toml)                                           │
│  [project.optional-dependencies]: base re-scope, cuda121 retirement,        │
│    rocm restore-or-remove, mypy cap in dev;  [tool.uv] indexes/sources/     │
│    conflicts kept in sync;  platform_system != 'Windows' markers unchanged  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Integration for v1.3 |
|-----------|----------------|---------------------|
| `dnallm/utils/genomic_intervals.py` (NEW) | Pure-numpy BED interval arithmetic: `parse_bed` (BED3+, file order preserved, never sorted), `jaccard` (merge with bookend coalescing, exact-string chrom keys), `loj_intersect` (A-file-order join, B-file-order hits, oracle-pinned null fill). Port of the two VERDICT-PARITY spike cores (261010-wx2, 261011-0a3) from spike artifacts to library code | The dependency root of the whole Windows-parity feature |
| `dnallm/utils/__init__.py` | Re-exports utils symbols | Add the new module's public names (genomic_coords precedent) |
| `tests/utils/test_genomic_intervals.py` (NEW) | Unit + parity-fixture tests: committed `.bed` pairs with frozen oracle outputs; edge cases (boundary touching, chrom isolation, duplicates, empty B, null vector at NER 6-column shape); **no bedtools binary required** | Fast, non-slow-marked → runs on every leg including test-windows |
| `tests/examples/_execution.py` | Private execution harness + typed-skip helpers + `NOTEBOOK_EXEC_SPECS` | Remove the two mcp_example spec entries + `LANGCHAIN_*` kernel constants (ollama directive); no dispatch logic added here |
| `tests/examples/test_notebook_execution.py` | Notebook execution census + gates | Remove `_gate_ollama_stack`/`GATED_NOTEBOOKS` ollama entries/`_TIMEOUT_7200_GATED` ollama rows; make `_gate_pybedtools` platform-aware |
| `tests/examples/test_script_execution.py` | Script lane | Make `_gate_script_pybedtools` platform-aware; GAP-1 disposition consumes/removes the `_NT_STRUCTURAL_MARKER` skip ladder |
| `tests/examples/test_plant_helixseek_showcase.py` | Showcase band assertions (selection.md-parsed) | Line 523 `which('bedtools')` assert becomes platform-forked; band parsing untouched |
| `.github/workflows/ci.yml` | Multi-job CI | test-windows gains the numpy-path execution surface; example-nightly stage 2.5/3 + `-k "not mcp_example"` selectors + D-03 triple re-pinned; test-cuda matrix legs updated for cuda121 retirement |
| `.github/workflows/publish.yml` | Release chain | NEW publish-smoke job; OIDC migration decision point; mypy behavior surfaced here |
| `pyproject.toml` | Package + extras + tooling config | Extras re-scope; mypy cap; coverage denominator untouched (new module counts — it must stay outside the 7-entry omit list) |
| `tests/test_extras_guard.py`, `tests/test_runner_infra_contracts.py`, `tests/test_models_lock_contracts.py` | Contract tests pinning extras membership, runner unit file, models.lock rows | Ollama refs removed/skip-marked per item; extras-guard expectations updated to the re-scoped extras |
| `example/` + `docs/example/` mirrors | Executable examples, byte-identical under `scripts/check_docs_sync.py` | Every platform-fork edit writes both copies in the same change |

## Recommended Project Structure

```
dnallm/
├── utils/
│   ├── genomic_coords.py          # EXISTING precedent: import-pure, 100% cov,
│   │                              #   validated-input, ValueError-loud helpers
│   ├── genomic_intervals.py       # NEW: numpy jaccard + loj_intersect + parse_bed
│   │                              #   (the spike cores, productionized)
│   └── transformers_compat.py     # EXISTING absence-gated attach precedent;
│                                  #   GAP-1 extends here (or removal is chosen)
tests/
├── utils/
│   └── test_genomic_intervals.py  # NEW: mirror of test_genomic_coords.py shape;
│                                  #   committed oracle fixtures, fast/non-slow
├── examples/
│   ├── _execution.py              # MODIFIED: ollama surface removal only
│   ├── test_script_execution.py   # MODIFIED: platform-aware gate; GAP-1 ladder
│   ├── test_notebook_execution.py # MODIFIED: ollama removal + platform-aware gate
│   └── test_plant_helixseek_showcase.py  # MODIFIED: platform-forked bedtools assert
example/notebooks/finetune_NER_task/
│   ├── generate_bpe_dataset.py    # MODIFIED: platform fork at :7 (import) and
│   │                              #   :249 (loj acquisition)
│   └── data_generation_and_inference.ipynb  # MODIFIED: cells 0/2/8 fork
example/notebooks/plant_helixseek_cre/
│   └── plant_helixseek_cre.ipynb  # MODIFIED: cell 11 jaccard fork (md cells
│                                  #   0/10/22 stay bedtools-worded — historical)
docs/example/...                   # MODIFIED: byte-identical mirrors of the above
.github/workflows/
│   ├── ci.yml                     # MODIFIED: windows leg, ollama excision, matrix
│   └── publish.yml                # MODIFIED: publish-smoke job (+ OIDC?)
```

### Structure Rationale

- **`dnallm/utils/genomic_intervals.py` (one module, not two):** jaccard and loj share `parse_bed` (BED3+ parsing with the loud-ValueError contract), the exact-string chromosome grouping, and the int64 half-open arithmetic; splitting them would duplicate the parser and the validation docstring discipline. `genomic_coords` is the naming/behavior precedent (silent-empty structurally impossible; every function validates). numpy is a core dependency (`numpy>=2.0.0` in base `[project.dependencies]`), so top-level `import numpy` is fine — unlike `genomic_coords`'s lazy `pyfastx` (a dev extra).
- **Must stay OUTSIDE the coverage omit globs:** the 7-entry `[tool.coverage.run]` omit list excludes vendored/unimportable code. The new module must be provably outside `*/dnallm/tasks/metrics/*` and friends — same proof duty `metric_registry.py` discharged in v1.2 (AST-proven import-light, outside the glob). It counts toward the shared `fail_under=90` floor on every leg.
- **Do NOT place it in `dnallm/tasks/` or `dnallm/inference/`:** it is not a task metric (not in the metric-registry emission chain) and not inference; it is a genomic utility consumed by example code and tests. Placement anywhere else would also entangle it with facades that stay byte-stable.

## Architectural Patterns

### Pattern 1: Acquisition-fork, consumption-uniform (platform dispatch in example code)

**What:** The platform fork happens ONLY where the bedtools/pybedtools API is acquired; everything downstream consumes a uniform row interface.
**When to use:** `generate_bpe_dataset.py:7,249`, notebook cells 2/8, CRE cell 11.
**Trade-offs:** One fork point per file instead of scatter; the `tokens_to_nerdata` traversal body stays byte-identical across platforms.

```python
# generate_bpe_dataset.py — fork at import (top)
import platform
_IS_WINDOWS = platform.system() == "Windows"
if _IS_WINDOWS:
    from dnallm.utils.genomic_intervals import loj_intersect
else:
    from pybedtools import BedTool          # line 7 today

# fork at acquisition (line 249 today) — emit a uniform row object either way
if _IS_WINDOWS:
    rows = _adapt(loj_intersect(tokens_bed, annotation_bed))   # namedtuple rows
else:
    rows = BedTool(tokens_bed).intersect(annotation_bed, loj=True)
# consumer loop (iv.chrom/iv.start/iv.end/iv.name/iv.fields[4]/iv.fields[9]/iv.fields[10])
# is UNCHANGED below this line on both platforms
```

The adapter is the load-bearing piece: the numpy path must yield objects with `.chrom/.start/.end/.name/.fields` (a namedtuple with a `fields` tuple suffices) so the `name == "-1"` intergenic branch — which the 0a3 pin proved FIRES because bedtools emits `-1` in the B score slot at the 6-column shape — keeps firing on the exact same field. A port that returns bare lists would force forking the consumer too, which is exactly what to avoid.

**Spike-derived contracts the port MUST carry (from report.txt pins):**
- Null-B fill per column count: chrom-col `.`, start/end `-1`, name `.`; score slot `-1` at B-count 6 (always), content-dependent at count 5, `.` at ≥7. Production port should assert the 6-column NER shape or port `_null_b_fields` whole.
- Plain `-loj` emits A-file order, per-A B hits in B-file order, unsorted input accepted — never sort.
- Join, not merge: duplicates multiply per copy; the consumer's `token_seen` dedup absorbs A-side duplication — do not dedup early.
- jaccard: merge coalesces bookended adjacency; exact-string chrom keys (`Chr1 != chr1`); `n_intersections` is NOT portable (bedtools counts raw A rows = 51 vs merged = 45 on the frozen pair) — exclude that field from any production contract.

### Pattern 2: Gate inversion on Windows (probe-then-execute → dispatch-then-execute)

**What:** The three existing gates probe for a Linux-only dependency and skip when absent. On Windows the numpy path makes the dependency unnecessary, so the gate inverts: no probe needed (numpy + dnallm are always importable), the test executes for real.
**When to use:** `_gate_script_pybedtools` (test_script_execution.py:128), `_gate_pybedtools` (test_notebook_execution.py:1397, wired through `_ACTIVE_NOTEBOOK_GATES`), the `which('bedtools')` assert (test_plant_helixseek_showcase.py:523).
**Trade-offs:** Keeps the Linux lanes byte-identical in behavior (gate still probes and passes on Linux where the dev extra installs pybedtools); on Windows the gate function becomes a no-op or probes `dnallm.utils.genomic_intervals` importability instead.

```python
def _gate_script_pybedtools() -> None:
    if platform.system() == "Windows":
        return  # numpy path: no pybedtools/bedtools prerequisite exists
    spec = importlib.util.find_spec("pybedtools")
    ...  # unchanged Linux probe-then-skip
```

The showcase assert (line 523) is the fail-loud variant: on Windows it should still fail loud — asserting the numpy path prerequisites — rather than becoming silent. The Windows-first-exposure contract tests for these gates (the `TestScriptGateContracts` / `TestActiveNotebookGates` classes) get sibling cases for the Windows branch (monkeypatched `platform.system`), preserving the same-change-test discipline.

### Pattern 3: Single-sourced frozen truth with pinned tolerance (no truth-chain fork)

**What:** selection.md stays the ONLY source of bands/floors, parsed at test startup on every path (`_parse_floors`/`_parse_bands`, parse-guarded). The numpy path asserts against the SAME parsed bands — never literals, never a Windows-specific copy.
**When to use:** the dual-path acceptance (`exon_f1=0.7522` in band on both paths; CRE jaccard band [0.3, 1.00] both paths).
**Trade-offs:** the tolerance pin is comparison-layer only: numpy yields trailing float deltas (e.g. `0.3247270000001`), so exact-equality comparisons against frozen values get a pinned tolerance (round-to-6dp or `abs(diff) <= 1e-9` — the wx2 evidence: full-precision `0.32472687098490116` rounds to the recorded 6-dp anchor `0.324727`). The selection.md bands themselves are wide ([0.3, 1.00]) and need no change. Note honestly: `exon_f1` is currently evidence-only (no band) in the Anno notebook; pinning its Windows-path band/tolerance is part of v1.3's frozen-value pinning work.

### Pattern 4: Absence-gated attach (GAP-1 compat shim) — existing precedent

**What:** `transformers_compat.py` already vendors removed upstream helpers and re-attaches them "per name only where the module does not already expose it." GAP-1 extends the same pattern: 4.x-era remote `modeling_esm.py` reads `config.is_decoder` / `config.add_cross_attention` — defaults removed in transformers 5.x.
**When to use:** if the shim disposition is chosen for windows #11/#12. The shim attaches the missing defaults onto `PretrainedConfig` (or a targeted subset) only when the attributes are absent, so transformers 4.49–5.x both work untouched.
**Trade-offs:** shim keeps the benchmark third model (#11) and the script-lane pkl leg (#12) alive and unskips them (closing #15 "rides along if shimming succeeds" per PROJECT.md); removal is the lighter long-term surface but costs the benchmark notebook a registry model and leaves the script lane gated. **Decision needed at planning time** (the milestone scope says so); architecturally the shim is a small, precedent-shaped extension — the removal is a subtraction from example code + models.lock.

### Pattern 5: Dry-run publish smoke (release-chain canary)

**What:** a separate job in `publish.yml` (or a sibling workflow — see Integration Points) triggered by `workflow_dispatch` (+ optionally a weekly `schedule`) that runs everything the real release runs EXCEPT the production upload: `uv build` → `twine check --strict dist/*` → install-from-dist clean-venv import smoke → optionally a TestPyPI upload via trusted publishing.
**When to use:** the two consecutive v1.2.1 release failures (uv implicit-system-env removal; `check_code.py --with-tests` flag removal) were exactly dry-run-detectable rot.
**Trade-offs:** TestPyPI upload requires registering a second trusted publisher (test.pypi.org, workflow filename + environment must match claims) and tolerating version-already-exists noise (`--skip-existing` style handling); a no-upload dry-run is cheaper but cannot catch upload-path auth regressions — which is precisely where PYPI_API_TOKEN rot would hide, arguing for the OIDC migration to land WITH the smoke job (MEDIUM confidence, web-sourced: the periodic-cron canary is an uncommon-but-sound extension of the ecosystem's workflow_dispatch dry-run norm).

## Data Flow

### Windows-parity flow (the core new data path)

```
committed inputs (showcase loci ≤200kb / NER rice inputs)
        ↓
[Linux/macOS]                          [Windows]
pybedtools.BedTool / bedtools CLI       dnallm.utils.genomic_intervals
(subprocess or library)                 (pure numpy: parse_bed → jaccard /
        ↓                                loj_intersect + pinned null fill)
        └──────────┬───────────────────────────┘
                   ↓  uniform row interface / float metric
        example-code consumer (tokens_to_nerdata traversal,
        CRE jaccard print) — byte-identical both paths
                   ↓
        selection.md bands/floors (parsed, single source)
                   ↓
  ┌────────────────────────────┬──────────────────────────────┐
  │ example-nightly (Linux)    │ test-windows (hosted)         │
  │ full notebook execution,   │ FAST non-slow frozen-value    │
  │ bedtools path, band asserts│ tests: numpy evaluator over   │
  │ (existing lanes unchanged) │ committed pairs vs frozen     │
  │                            │ values, pinned tolerance      │
  └────────────────────────────┴──────────────────────────────┘
```

Key constraint: hosted Windows CI cannot run model inference (the CRE predicted-beds come from GPU inference), so the Windows leg executes the numpy evaluators over COMMITTED interval pairs (the frozen selection pair from wx2: intersection=15872, union=48878, jaccard=0.324727; NER-shaped loj fixtures from 0a3) and asserts against the frozen values with the pinned tolerance. Full dual-path notebook execution stays on the Linux nightly + owner dev boxes.

### Publish-smoke flow

```
weekly cron / workflow_dispatch
   → checkout + uv (pinned action versions)
   → uv pip install -e '.[<re-scoped extras>'   (exercises the SAME install lane as release)
   → scripts/check_code.py                        (flags verified current)
   → uv build
   → twine check --strict dist/*
   → clean venv: pip install dist/*.whl → import dnallm smoke
   → [OIDC option] pypa/gh-action-pypi-publish → TestPyPI
        (permissions: id-token: write at JOB level; no password: — a lingering
         password silently disables OIDC; publisher registered for the exact
         workflow FILENAME + environment; old PYPI_API_TOKEN revoked)
```

### Key Data Flows

1. **Ollama removal cascade:** deleting the mcp_example pytest surface flows through `tests/examples/test_notebook_execution.py` (gates, `GATED_NOTEBOOKS`, `_TIMEOUT_7200_GATED`, spec entries) → `tests/examples/_execution.py` (LANGCHAIN kernel constants, spec entries) → `tests/test_runner_infra_contracts.py` (unit-file pins, model-swap pins) → `tests/test_extras_guard.py` (langchain-ollama declaration — **decision needed**: the notebooks stay in-repo per WINDOWS #13, so the mcp extra's `langchain-ollama` and its guard test probably stay; remove only the pytest surface) → `tests/test_models_lock_contracts.py` (ollama models.lock rows) → `ci.yml` example-nightly stage 2.5/3 + stage-1 `-k "not mcp_example"` selector + **D-03 census triple re-pin** (208/217→new value; removals change collection, the deliberate bump-point).
2. **Extras re-scope cascade:** pyproject extras → `[tool.uv]` indexes/sources/conflicts (cuda121 rows, rocm un-comment-or-delete) → ci.yml install lines on EVERY job (`.[base]` in test/test-windows/coverage-gate/coverage-nightly/example-nightly/publish) → `tests/test_extras_guard.py` expectations → docs `installation.md` + README install commands → publish-smoke install lane. The `base` rename/re-scope is the highest-fanout single decision in the milestone.
3. **Coverage flow:** the new `genomic_intervals` module enters the `source_pkgs=["dnallm"]` denominator on every `--cov` leg; its tests must land in the same change (the metric_registry precedent: high per-module coverage proven in-phase).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Current (1 OS leg, 21 notebooks) | Patterns above suffice; no new infra |
| +Windows fast executor | Frozen-value fast tests add seconds-scale cost to test-windows (60-min budget, currently mostly install); watch the leg's runtime as fixtures grow |
| +Weekly publish smoke | One extra hosted job per week; negligible quota impact vs the 2700-min nightly |
| Future (rurunes win_amd64 wheels) | Ledger #19 already records the revisit trigger: if pyranges' successor ships Windows wheels, re-evaluate; the numpy evaluators remain the lower-dependency path regardless |

### Scaling Priorities

1. **First bottleneck: the D-03 census triple + expected_skips.yaml churn.** Three features (ollama removal, GAP-1 unskip, Windows gates) all touch collection/skip surfaces in `tests/examples/`. Land them in separate phases with one deliberate re-pin each; do not interleave two collection-changing edits in one phase.
2. **Second bottleneck: example-nightly stage budget.** Removing stage 3 shortens the pipeline; the Windows-path additions add nothing to the nightly (they live on the hosted leg). The stage-0.5 census assertion and stage-results ledger need their one-time updates.

## Anti-Patterns

### Anti-Pattern 1: Forking the consumer (per-platform traversal bodies)

**What people do:** wrap the whole `tokens_to_nerdata` in `if windows: ... else: ...`.
**Why it's wrong:** doubles the surface that must reproduce exon_f1 in-band; the 0a3 pins (null encoding, ordering, multiplicity) would need re-proofing per branch; docs-mirror diffs explode.
**Do this instead:** acquisition-fork + uniform row adapter (Pattern 1); one forked line per file, consumer byte-identical.

### Anti-Pattern 2: A second truth source for Windows bands

**What people do:** hard-code Windows-expected values (e.g. `0.3247270000001`) in the Windows tests.
**Why it's wrong:** violates the D-05/D-06 decision (single source of truth: thresholds parsed from selection.md at test startup); any re-freeze would desynchronize the two truths the split explicitly exists to avoid.
**Do this instead:** parse the same selection.md bands on both paths; put ONLY the comparison tolerance (a rounding/epsilon rule) in the test layer.

### Anti-Pattern 3: Sorting inputs in the numpy loj port

**What people do:** sort A/B by coordinate "for safety" before joining.
**Why it's wrong:** the pinned order rule says plain `-loj` emits A-file order (B hits in B-file order) and the consumer's ascending traversal works only because tokens_bed is near-sorted in practice — sorting changes emission order and can silently reorder the NER label stream.
**Do this instead:** never sort; preserve file order exactly (both spikes pin this).

### Anti-Pattern 4: Extras re-scope without updating the guard tests / CI lanes in the same change

**What people do:** rename `base` in pyproject alone; fix CI next PR.
**Why it's wrong:** every ci.yml job installs `.[base]`; `test_extras_guard.py` pins membership; publish.yml installs `.[base]`. A partial rename reds every lane at once and can fake a "broken package" signal.
**Do this instead:** one atomic change: pyproject + guard-test expectations + ci.yml/publish.yml install lines + docs, validated by a publish-smoke dispatch (another reason the smoke job lands early).

### Anti-Pattern 5: Treating `n_intersections` as portable in the jaccard evaluator

**What people do:** port all four `bedtools jaccard` output fields.
**Why it's wrong:** wx2 proved bedtools counts raw A rows (51) while the natural numpy analogue counts merged intervals (45) on the frozen pair — a permanent off-by-design divergence.
**Do this instead:** the evaluator returns intersection/union/jaccard; `n_intersections` is excluded from the production contract (or ports raw-row semantics explicitly — planning-time choice, but exclusion is simpler and nothing downstream consumes it).

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| PyPI (production) | OIDC trusted publishing via `pypa/gh-action-pypi-publish` — `permissions: id-token: write` at JOB level, publisher registered for the exact workflow FILENAME + environment; OR keep `TWINE_USERNAME=__token__`/`PYPI_API_TOKEN` | **Gotchas (web-verified, MEDIUM):** a lingering `password:` silently disables OIDC; old token must be revoked after migration; repo renames break publisher registration. Default workflow `permissions: contents: read` already in ci.yml — publish.yml needs its own job-level override |
| TestPyPI | Same action with `repository-url: https://test.pypi.org/legacy/` + separate publisher registration + `testpypi` environment | Needed only if the smoke job does a real upload; `--skip-existing`-style handling for version collisions |
| rice.uga.edu (NER inputs) | Existing cache-first seeding (`_seed_rice_input`, WR-04 4xx-vs-5xx semantics) | Unchanged; Windows full-script execution stays dev-box/nightly-only (115MB flaky host, slow-marked) |
| GitHub Actions windows hosted runner | test-windows leg, 60-min cap, `PYTHONUTF8=1`, git autocrlf disabled | Gains: numpy-path frozen-value tests (fast) — the hosted cap is why the Windows executor is fixture-based, not notebook-based |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `dnallm/utils/genomic_intervals.py` ↔ example code | Direct import inside the Windows branch only | Library stays platform-uniform; no `platform.system()` inside `dnallm/` — dispatch is a consumer concern (keeps the package testable on Linux for both paths: the numpy functions are directly unit-testable everywhere) |
| numpy evaluators ↔ bedtools truth chain | One-directional parity evidence | Spikes (committed under `.planning/quick/`) are the parity record; the frozen values in selection.md/models stay bedtools-computed — no re-freeze (owner directive) |
| test gates ↔ `tests/expected_skips.yaml` | Typed-prefix allowlist via `scripts/audit_skips.py` in 4 CI jobs | Windows gate inversion REMOVES potential skip sites (good — fewer skips); GAP-1 unskip removes the `environment-unavailable:` script-lane skip (ledger #12 closes); ollama removal removes `network-unavailable:` mcp-pair skips from the nightly — all shrink the skip surface, none add new prefixes |
| `example/` ↔ `docs/example/` | Byte-identical mirrors enforced by `scripts/check_docs_sync.py` | Every platform-fork edit carries its mirror write-back in the same commit (the v1.1 repair-loop discipline) |
| publish-smoke ↔ extras re-scope | Smoke job installs the re-scoped extras exactly as release does | Sequencing dependency: smoke job usable to validate each extras change via dispatch |

## Suggested Build Order (dependency-respecting)

Phase numbering continues from 13 (PROJECT.md). Rationale inline; file-disjointness noted for parallelization.

1. **Phase A — Ollama pytest-surface removal** (first: pure subtraction, shrinks every later edit surface)
   - `tests/examples/test_notebook_execution.py`, `_execution.py`, `test_runner_infra_contracts.py`, `test_extras_guard.py`, `test_models_lock_contracts.py`, `example/mcp_example` mirrors, `ci.yml` stage 2.5/3 + selectors + D-03 triple re-pin.
   - Why first: it is the only feature that REMOVES tests; doing it first means the Windows-parity and GAP-1 phases edit smaller files and the census is re-pinned once here instead of being invalidated later. Independent of everything else (file-disjoint from GAP-1 and from the new utils module).
2. **Phase B — GAP-1 disposition** (windows #11/#12; shim-vs-removal decided at plan time)
   - `dnallm/utils/transformers_compat.py` + `tests/utils/test_transformers_compat*.py`, `example/notebooks/benchmark/benchmark.ipynb` (+ mirror), `tests/examples/test_script_execution.py` skip ladder.
   - Why before Windows parity: both B and C edit `test_script_execution.py` (the `_NT_STRUCTURAL_MARKER` ladder vs the `_gate_script_pybedtools` platform fork) — sequencing avoids reworking the same test class twice. Note: unskipping changes skip counts, not collection, so no census re-pin conflict with Phase A.
   - Can run in parallel with Phase A if plans stay file-disjoint (benchmark notebook + transformers_compat vs ollama files) — only `ci.yml` is shared, which suggests serial or carefully split ci.yml edits.
3. **Phase C — Windows parity core: the numpy evaluators as library code**
   - NEW `dnallm/utils/genomic_intervals.py` + `tests/utils/test_genomic_intervals.py` (committed oracle fixtures; port the spike cores + the three pinned contracts). Depends on nothing above; pure addition.
4. **Phase D — Platform dispatch rollout** (depends on C)
   - `generate_bpe_dataset.py` + NER notebook cells 0/2/8 + CRE cell 11 (+ all docs mirrors); the three test gates go platform-aware (+ Windows-branch contract tests); NEW fast frozen-value tests (selection.md-parsed bands + pinned tolerance) targeting the test-windows leg; `ci.yml` test-windows step set.
5. **Phase E — Publish chain robustness** (independent of B/C/D; ideally before F so it can validate F)
   - `publish.yml` publish-smoke job (+ OIDC decision — the deferred row says the implementation decision lands with this phase), mypy cap in pyproject dev extra, #20 local-venv `pytest --cov` crash fix.
6. **Phase F — Extras re-scope + install honesty** (validated by E's smoke dispatches)
   - pyproject extras (base re-scope, cuda121 retirement + uv index/sources/conflicts + test-cuda matrix, rocm decision), `test_extras_guard.py`, ci.yml/publish.yml install lines, `docs/getting_started/installation.md` uv-vs-pip lanes, cuda130 default recommendation, GPU support matrix page (pre-verified boundaries: Pascal cpu-only, 20xx→cu128, 30/40xx all extras, 50xx cu128+), #18 triton-windows declaration decision (same pyproject surface).
7. **Phase G — Docs accuracy sweep** (last: after every code/extras surface is final)
   - Per-file review of `docs/` + `example/` (accuracy vs code, post-numpy2/py311 staleness, install-command correctness) with same-pass fixes. Running G before F would leave the install docs immediately stale again; running it last means it verifies the final surface once.

**Ordering answers to the specific questions asked:**
- **GAP-1 before or after ollama removal:** either is defensible (file-disjoint), but ollama-first is cleaner because it is a pure subtraction that re-pins the census once; GAP-1 must precede the Windows platform-dispatch phase because both edit `test_script_execution.py`.
- **Docs review before or after extras:** AFTER — install docs must describe the final extras surface; the docs sweep is the milestone's verification pass over a frozen surface.
- **Publish-smoke before extras:** BEFORE (or with) — the smoke job exists to validate exactly the kind of install-lane changes the extras re-scope makes.

## Sources

- Codebase (HIGH, primary): `.planning/PROJECT.md` (v1.3 scope), `.planning/WINDOWS.md` ledger #11–#20 (esp. #19 change-surface inventory, #18 triton-windows, #20 cov crash), `.planning/STATE.md` deferred rows (extras hygiene, publish rot, mypy cap, OIDC), `pyproject.toml` (extras, uv indexes, coverage omit, pytest config), `.github/workflows/ci.yml` (job graph, stages, D-03 census gate), `.github/workflows/publish.yml`, `dnallm/utils/genomic_coords.py` (module precedent), `dnallm/utils/transformers_compat.py` (absence-gated attach precedent), `tests/examples/_execution.py`, `tests/examples/test_script_execution.py`, `tests/examples/test_notebook_execution.py`, `tests/examples/test_plant_helixseek_showcase.py`, `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` + `data_generation_and_inference.ipynb`, `tests/expected_skips.yaml`
- Spike evidence (HIGH, in-repo): `.planning/quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/` (VERDICT PARITY; frozen pair 15872/48878, jaccard delta 0.0; n_intersections non-portability), `.planning/quick/261011-0a3-spike-numpy-pandas-loj-intersect-parity-/` (VERDICT PARITY; the three empirical pins — null fill, output order, sorted requirement; NER port notes)
- Web (MEDIUM, cross-checked via search): [PyPI — Publishing with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/), [packaging.python.org — Publishing package distributions using GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/), [PyPI blog — Introducing Trusted Publishers](https://blog.pypi.org/posts/2023-04-20-introducing-trusted-publishers/), plus community publish-runbook patterns for the dry-run shape ([cognirepo publish.yml](https://github.com/ashlesh-t/cognirepo/blob/main/.github/workflows/publish.yml), [local-sdlc-kit releasing](https://senssei.github.io/local-sdlc-kit/releasing/), [ATSW PUBLISHING.md rot record](https://raw.githubusercontent.com/davidesg/art-python/refs/heads/master/docs/PUBLISHING.md))

---
*Architecture research for: DNALLM v1.3 Platform Reach & Packaging Honesty*
*Researched: 2026-10-11*
