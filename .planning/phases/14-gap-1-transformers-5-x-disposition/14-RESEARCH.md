# Phase 14: GAP-1 Transformers 5.x Disposition - Research

**Researched:** 2026-10-13
**Domain:** transformers 4.x→5.x remote-code compatibility disposition (trust_remote_code ESM checkpoints), pytest skip-ledger honesty, CI census reconciliation
**Confidence:** HIGH — the decisive claims were verified by live probes run this session on the dev venv (transformers 5.17.0), by reading every touched source file, and by the green example-nightly junit artifact (run 38308972028)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### 判定路径与阈值 (owner-accepted 2026-10-13)
- Evidence first: run the empirical ladder probe under the current shim set; extend rungs only on real tracebacks (GAPC-01), then decide on the evidence table
- Disproportionateness threshold: at most 3 additional rungs, public-API absence only; touching private transformers internals or behavior emulation flips the disposition to removal
- Removal scope (if that path): third benchmark model + BPE pkl leg removed together (#11/#12 same GAP-1 root)
- Owner checkpoint retained: ONE confirmation with the evidence table (shim vs removal options) before executing the disposition

#### 处置执行与记账边界 (owner-accepted 2026-10-13)
- Shim lands in the existing dnallm/utils/transformers_compat.py ladder (version-keyed, absence-gated rungs, matching the 1251-line established structure)
- Regression proof: 35-family gate + fast lane green locally in the same change, plus CI coverage-gate leg re-verification
- Census triple re-pin + expected_skips.yaml reconciliation land in the SAME commit as any un-skip or removal
- Windows closure semantics: shim path → #11/#12/#15 all fixed with probe evidence; removal path → #11/#12 fixed as removed-by-decision, #15 closed on per-item probe evidence

### Claude's Discretion
Implementation details of the probe harness, rung code shape, and test edits are at Claude's discretion within the accepted contracts above.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GAPC-01 | An empirical load run of the NT promoter checkpoint on transformers 5.17 with the current shim set sizes the remaining rung ladder (live probe: exactly `invert_attention_mask` missing) | **Executed this session** — load + real forward both SUCCEED under the current shim set; the remaining ladder is EMPTY. `invert_attention_mask` is provably dead code for this checkpoint (reachability analysis below). Evidence-recording mechanics documented in Architecture Patterns. |
| GAPC-02 | Shim-vs-removal is executed on that evidence — vendored absence-gated rung extension if tractable (closes WINDOWS #11/#12/#15); honest removal of the third benchmark model + BPE pkl leg if disproportionate (closes #11/#12 as removed-by-decision) | The evidence table sizes the ladder at zero rungs → the disposition is the shim path with no new rung code; both closure paths' full blast radii inventoried below for the owner checkpoint. |
</phase_requirements>

## Summary

This phase was scoped around a presumed-open engineering gap; live forensics show the gap **already closed**. The GAP-1 terminal failure that opened WINDOWS #11/#12 (remote `modeling_esm.py` reading `config.is_decoder`/`config.add_cross_attention` — 4.x `PretrainedConfig` defaults removed in 5.x) was fixed when the `_patch_pretrained_config_legacy_defaults` rung landed in `dnallm/utils/transformers_compat.py` on 2026-10-02. What remained was never re-verified end-to-end until now.

Three decisive facts, all established this session:

1. **The NT promoter checkpoint loads and forwards for real on transformers 5.17.0 under the current shim set.** A probe mirroring `tests/models/test_model_remote_code.py` exactly (`zhangtaolab/nucleotide-transformer-v2-100m-promoter`, binary task, `source="modelscope"`) completed with `EsmForSequenceClassification`, `config.is_decoder resolved: False`, `config.add_cross_attention resolved: False`, forward logits `(1, 2)` — the exact assertions of the smoke test. [VERIFIED: live probe this session, /tmp/gapc01-probe/load_probe.py on the dev venv]
2. **The BPE pkl leg's checkpoint also loads and forwards** (`zhangtaolab/plant-nucleotide-transformer-BPE`, task `token`, num_labels 7 per `ner_task_config.yaml`) — remote `EsmForTokenClassification`, `EsmTokenizer` roundtrip green, forward logits `(1, 17, 7)`. [VERIFIED: live probe this session] And the **full script already passed for real** in the green example-nightly: `tests.examples.test_script_execution.TestExampleScriptExecution::test_generate_bpe_dataset_produces_artifact` = PASS in run 38308972028 (stage 1: 211 passed / 5 skipped / 1 deselected; none of the 5 skips is a GAP-1 skip). [VERIFIED: junit artifact of run 38308972028, downloaded this session]
3. **`invert_attention_mask` — the "one known gap" from the milestone attr-inventory probe — is dead code for this checkpoint.** In the remote `modeling_esm.py` the ONLY call site sits inside `if self.config.is_decoder and encoder_hidden_states is not None:`; both `is_decoder` (False via the legacy-defaults rung) and `encoder_hidden_states` (None on every binary/token route) make the branch unreachable. The milestone probe was an import/attr inventory, not an execution trace. [VERIFIED: /home/forrest/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/snapshots/master/modeling_esm.py:1028-1041 — the guard reads `if self.config.is_decoder and encoder_hidden_states is not None:` and the call `encoder_extended_attention_mask = self.invert_attention_mask(` follows inside it]

**Primary recommendation:** Execute the shim-path disposition with **zero new rungs**. The phase's real work is evidence-recording and bookkeeping honesty: (1) put the empirical load run on record (GAPC-01) by executing the existing smoke test with captured output plus the reachability analysis; (2) update the now-false prose in the two test files (the skip-marker *branches* stay as regression tripwries — they are probe-conditioned and can never fire again unless transformers regresses); (3) close WINDOWS #11/#12/#15 with probe evidence; (4) verify the census triple stays 216/217 and `expected_skips.yaml` needs no growth; (5) run the gates (fast lane, benchmark tests, census assertion). The owner checkpoint presents the evidence table with both formal options; the evidence makes the choice trivial (removal would delete working, nightly-green functionality).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 4.x→5.x remote-code API restoration | Library (`dnallm/utils/transformers_compat.py`) | — | The established absence-gated rung ladder owns every cross-version patch; apply_patches() runs at import of dnallm |
| Empirical load evidence (GAPC-01) | Test layer (`tests/models/test_model_remote_code.py`) + recorded phase artifact | Dev venv (transformers 5.17.0) | The smoke test IS the load run; evidence = its captured green execution |
| Benchmark third-model executability | Example + nightly lane (`tests/examples/test_notebook_execution.py` over `benchmark.ipynb`) | — | Already green in nightly 38308972028 |
| BPE pkl leg executability | Example + nightly lane (`tests/examples/test_script_execution.py`) | — | Already green in nightly 38308972028 |
| Skip allowlist honesty | CI gate (`scripts/audit_skips.py` × `tests/expected_skips.yaml`) | — | Every junit skip must match one allowlisted prefix/exact |
| Census integrity | CI gate (ci.yml stage 0.5 hard pin) | — | `216/217 tests collected (1 deselected)` triple is the bump-point |
| Ledger closure | `.planning/WINDOWS.md` via `gsd-tools windows fixed <id>` | — | #11/#12/#15 closure with evidence reasons |

## Standard Stack

No new packages. This phase installs nothing.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers (installed dev venv) | 5.17.0 | The 5.x side of the compat surface; the evidence target named by GAPC-01 | [VERIFIED: live `import transformers` this session] pyproject pins `transformers>=4.49.0,<6` [VERIFIED: pyproject.toml:67] |
| dnallm/utils/transformers_compat.py | in-repo (1251 lines) | The version-keyed/absence-gated rung ladder — 10 patch families in `apply_patches()` | Every rung is absence-gated per-name, idempotent via sentinels, and no-ops on 4.x [VERIFIED: transformers_compat.py:1228-1251] |
| torch (dev venv) | 2.11.0+cu130 | Model execution | [VERIFIED: live this session] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| modelscope (route) | installed | `snapshot_download` + Auto* wrappers used by `load_model_and_tokenizer(source="modelscope")` — the exact benchmark/smoke route | Both NT checkpoints resolve through it; caches at `~/.cache/modelscope/` [VERIFIED: model.py:461-495] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Recorded local smoke execution as GAPC-01 evidence | Wait for a green coverage-nightly | The nightly is *also* required by the phase contract, but it is schedule-bound and was last cancelled (power-outage recovery); the local recorded run is immediate and uses the exact asserted route |

**Installation:** none.

**Version verification:** `transformers 5.17.0` / `torch 2.11.0+cu130` confirmed by live import in the dev venv this session. CI legs resolve the open `>=4.49.0,<6` range freshly — the ladder's `_patch_device_type_query` rung exists precisely because remote CI resolved 5.19.0 while the dev box sits on 5.17 [VERIFIED: transformers_compat.py:1112-1135 comment records the live 5.19.0 CI observation].

## Package Legitimacy Audit

No packages are installed or recommended by this phase. **Table: none.** All work is in-repo code, test prose, ledger, and CI pin verification. (The Package Legitimacy Gate protocol was not triggered — no external package names enter the plan.)

## Architecture Patterns

### System Architecture Diagram

```
                      GAPC-01 evidence flow (the decision gate)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ dev venv: transformers 5.17.0 + dnallm (import applies rung ladder)      │
 └──────────────┬───────────────────────────────────────────────────────────┘
                │ load_model_and_tokenizer("zhangtaolab/nt-v2-100m-promoter",
                │                             binary, source="modelscope")
                ▼
   modelscope snapshot ──► remote modeling_esm.py imports/calls
                │                │
                │                ├─ from transformers.modeling_utils import
                │                │    find_pruneable_heads_and_indices, ...   ── rung: _patch_remote_code_pruning_helpers
                │                ├─ self.get_extended_attention_mask(...)     ── rung: _patch_get_extended_attention_mask
                │                ├─ self.get_head_mask(...)                   ── rung: _patch_get_head_mask
                │                ├─ self.init_weights() entry                 ── rung: _patch_legacy_init_weights_bookkeeping
                │                ├─ config.is_decoder / add_cross_attention  ── rung: _patch_pretrained_config_legacy_defaults
                │                └─ self.invert_attention_mask(...)           ── DEAD BRANCH (is_decoder=False and
                │                                                                 encoder_hidden_states=None ⇒ unreachable)
                ▼
        LOAD OK + forward (1,2)  ──►  ladder sized: ZERO new rungs
                │
                ▼
   Owner checkpoint (ONE confirmation, evidence table: shim-zero-rung vs removal)
                │
                ▼
   Disposition execution ──► test-prose honesty edits + WINDOWS #11/#12/#15 fixed
                │
                ▼
   Gates: fast lane (-m "not slow") + tests/benchmark (35 tests) + census 216/217
          + expected_skips.yaml audit (stage-4 shape) + CI coverage-gate re-verify
```

### Recommended Project Structure

No new files are required by the recommended disposition. Touched files (shim path):

```
tests/models/test_model_remote_code.py      # docstring + GAP-1 comment prose -> resolved state
tests/examples/test_script_execution.py     # GAP-1 comment block prose -> resolved state
.planning/WINDOWS.md                        # #11/#12/#15 fixed with evidence reasons (via gsd-tools)
.planning/REQUIREMENTS.md                   # GAPC-01/02 checkboxes
.planning/STATE.md                          # phase decision line
```

### Pattern 1: The rung (absence-gated, version-keyed, idempotent)

**What:** every compat rung follows the same shape — try-import the target module, `hasattr`/probe gate per name, attach the vendored 4.49.0 implementation under its upstream name, set a `_dnallm_*` sentinel. **When to use:** only on a real traceback from the empirical load (the locked decision). **This phase needs zero new rungs** — documented here because the plan must reference the shape if the recorded run surprises (see Open Questions).

```python
# Source: dnallm/utils/transformers_compat.py:507-510 (verbatim, the rung that closed the GAP-1 terminal failure)
_LEGACY_PRETRAINED_CONFIG_DEFAULTS: dict[str, object] = {
    "is_decoder": False,
    "add_cross_attention": False,
}
```

```python
# Source: dnallm/utils/transformers_compat.py:1228-1246 (verbatim, apply_patches body)
def apply_patches():
    """Apply all compatibility patches. Safe to call multiple times.
    ...
    """
    _patch_device_type_query()
    _patch_get_parameter_or_buffer()
    _patch_initialize_weights_for_quantized_missing()
    _patch_remote_code_pruning_helpers()
    _patch_get_extended_attention_mask()
    _patch_pretrained_config_legacy_defaults()
    _patch_mamba_cache()
    _patch_deberta_vocab_dict()
    _patch_get_head_mask()
    _patch_legacy_init_weights_bookkeeping()
    _patch_numpy_fromstring()
```

A hypothetical new rung (NOT recommended this phase) would follow `_patch_get_head_mask` (transformers_compat.py:913-940): try-import, `hasattr(PreTrainedModel, name)` gate, attach, sentinel. Unit-test pattern: `tests/utils/test_transformers_compat.py` (79 tests, classes per rung).

### Pattern 2: The self-healing typed skip (marker branch)

**What:** the two GAP-1 test surfaces convert ONE documented structural failure into an `environment-unavailable:` typed skip carrying the exact traceback; every other failure propagates loudly. Because they are probe-conditioned, they have already self-healed into real green passes — no un-skip edit is needed.

```python
# Source: tests/models/test_model_remote_code.py:33 (verbatim)
_STRUCTURAL_MARKER = "'EsmConfig' object has no attribute 'is_decoder'"
```

```python
# Source: tests/examples/test_script_execution.py:46 (verbatim)
_NT_STRUCTURAL_MARKER = "Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'"
```

Both markers can no longer match: the legacy-defaults rung answers `is_decoder`/`add_cross_attention`, so the `AttributeError` text never occurs. **Keep the branches** — they are the regression tripwire (a future transformers default-removal regression re-skips loudly instead of failing opaquely). **Fix the prose** — module docstrings and the GAP-1 comment blocks still narrate the 05-04 terminal state as current.

### Pattern 3: Evidence-on-record for GAPC-01

The requirement text says the empirical load run must be "on record". The exact asserted route already exists as the smoke test; the recorded run is its execution with captured output:

```bash
.venv/bin/python -m pytest tests/models/test_model_remote_code.py::TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards \
  -m slow -q --junitxml=/tmp/gapc01-smoke-junit.xml 2>&1 | tee <phase-evidence-file>
```

(The slow marker must be re-included: the test is `@pytest.mark.slow`.) Complement with the reachability analysis for `invert_attention_mask` (Pitfalls → Pitfall 1) and, optionally, the second-leg probe (`plant-nucleotide-transformer-BPE`, task `token`) — its green execution is the per-item evidence for #12's script lane.

### Anti-Patterns to Avoid
- **Adding a defensive `invert_attention_mask` rung "while we're here":** violates the locked decision "extend rungs only on real tracebacks" — the branch is unreachable for every NT/ESM encoder route dnallm loads. The ladder stays evidence-sized at zero.
- **Deleting the marker branches as "dead code":** they are the loud re-skip path if transformers regresses; deleting them converts a future regression into an opaque failure (the exact erosion Pitfall 11 warns about).
- **Editing test classes/collection in tests/examples/:** any add/remove of collected ids moves the 216/217 census triple and forces a same-commit re-pin (D-03). Prose-only edits inside test bodies do not.
- **Global `PretrainedConfig` re-init or catch-all `__getattr__`:** explicitly an anti-feature in REQUIREMENTS.md Out of Scope; the closed-map rung is the sanctioned shape.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 4.x API restoration on 5.x | New ad-hoc monkeypatches in test or example code | The existing rung ladder in `dnallm/utils/transformers_compat.py` | 10 established rung families with idempotency sentinels and 79 unit tests [VERIFIED: tests/utils/test_transformers_compat.py] |
| Skip reconciliation | Custom junit parsing | `scripts/audit_skips.py <junit> tests/expected_skips.yaml` | The CI stage-4 shape; exit 1 on any unmatched skip [VERIFIED: audit_skips.py:1-40, ci.yml:1044] |
| Ledger bookkeeping | Hand-editing WINDOWS.md rows | `gsd-tools windows fixed <id>` (subcommands: status, append, waive, fixed) | Keeps frontmatter counters consistent [VERIFIED: gsd-tools windows --help this session] |
| Docs-mirror sync | Manual copy edits | `scripts/check_docs_sync.py` (docs-validation.yml gate) | If any removal-path docs mirror is touched, byte-identity is machine-checked |

**Key insight:** every mechanism this phase needs — rung ladder, typed skips, census pin, skip audit, ledger CLI — already exists and is green; the phase verifies and records, it does not build.

## Runtime State Inventory

Not a rename/refactor/migration phase — the full 5-category table is omitted. One runtime-state fact matters: the remote code that executes comes from the **modelscope snapshot cache** (`~/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/snapshots/master/`) and the **HF modules cache** (`~/.cache/huggingface/modules/transformers_modules/_81b29e…/8324cfe…/modeling_esm.py`); the two cached copies differ ONLY in the import-statement shape (HF copy: `from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer`; modelscope copy imports all three names from `transformers.modeling_utils`) — both covered by the pruning rung, which patches BOTH modules [VERIFIED: diff this session + transformers_compat.py:363-376]. No migration action required in either disposition path.

## Common Pitfalls

### Pitfall 1: Treating the attr-inventory probe as an execution trace
**What goes wrong:** the milestone research's "exactly `invert_attention_mask` missing" is an import/attribute inventory of the remote file, not a runtime trace. Planning a rung for it wastes the disproportionateness budget on unreachable code.
**Why it happens:** an inventory cannot see branch conditions.
**How to avoid:** the recorded GAPC-01 run + the reachability analysis (call site guarded by `self.config.is_decoder and encoder_hidden_states is not None`; `is_decoder` resolves False via the closed-map rung — `config.json` carries no `is_decoder`/`add_cross_attention` keys [VERIFIED: checkpoint config.json read this session]).
**Warning signs:** a plan task "vendor invert_attention_mask" without a traceback artifact.

### Pitfall 2: Census drift from careless test edits
**What goes wrong:** adding/removing collected test ids in `tests/examples/` breaks the stage-0.5 hard pin and the nightly goes red before stage 1.
**Why it happens:** the pin is exact-text: `if ! tail -1 census-collect.txt | grep -qE '^=* *216/217 tests collected \(1 deselected\) in '` [VERIFIED: ci.yml:867].
**How to avoid:** prose-only edits inside test bodies/functions; if a test id must change, re-pin the triple in the SAME commit (locked decision) and re-measure with the exact stage-0.5 flags: `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"` [VERIFIED: reproduced locally this session → `216/217 tests collected (1 deselected) in 1.03s`].

### Pitfall 3: Wrong task_type in a BPE-leg probe routes to the native EsmModel and fails on `rope_theta`
**What goes wrong:** loading `plant-nucleotide-transformer-BPE` with a task_type outside {mask, generation, binary, multiclass, multilabel, regression, token} falls into the `else` branch → plain `AutoModel` → the checkpoint's auto_map has no plain `AutoModel` key → transformers falls back to the NATIVE `EsmModel` with the REMOTE `EsmConfig` → `AttributeError: 'EsmConfig' object has no attribute 'rope_theta'` (native 5.x `compute_default_rope_parameters` reads it; the 4.x-era remote config class never sets it). Observed live this session with task_type `"token_classification"`.
**Why it happens:** the fallback ladder in `model.py:653-664` plus auto_map coverage gaps.
**How to avoid:** the script's real config is `task_type: "token"` [VERIFIED: ner_task_config.yaml] — probes must mirror it exactly. Note for the ledger: the `embedding` task type on THIS checkpoint family would hit the same native-route `rope_theta` wall (out of phase scope; the embedding notebook uses `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, which is nightly-green).
**Warning signs:** a traceback whose frames are in `site-packages/transformers/models/esm/` rather than a `transformers_modules` cache path.

### Pitfall 4: CI transformers drift vs the 5.17 evidence base
**What goes wrong:** the recorded evidence targets transformers 5.17.0 (the dev venv), but CI legs resolve the open `>=4.49.0,<6` freshly — historically 5.19.0 on remote runners [VERIFIED: transformers_compat.py:1112-1135]. A newer 5.x breakage class would surface on CI, not locally.
**How to avoid:** the phase contract already carries "CI coverage-gate leg re-verification"; keep the smoke's marker branch intact so a regression typed-skips loudly rather than failing opaquely.
**Warning signs:** nightly failures whose tracebacks mention transformers internals absent on 5.17.

### Pitfall 5: Closing #15 with a blanket "rides along"
**What goes wrong:** #15's items (evo/megaDNA optional-dep gates, megaDNA demo cell, PlantCAD lora pair / mamba_ssm) are environment gates, not GAP-1 gates; a blanket closure without per-item evidence is the dishonesty class the ledger bans.
**How to avoid:** per-item probe evidence from the green nightly junit — in run 38308972028 NONE of the #15-class skips fired (the 5 skips are: 1 content, 2 mcp network-unavailable, 2 pygenometracks optional-dep — the latter belong to the #19/showcase surface, not #15), i.e. every #15 gate executed for real [VERIFIED: junit parsed this session]. Cite those rows in the `windows fixed` reason.

### Pitfall 6: The cancelled coverage-nightly leaves the smoke test without a recent on-record CI execution
**What goes wrong:** the models-lane smoke runs only on coverage-nightly (full suite incl. slow); the last coverage-nightly was cancelled mid-run (2026-10-12, power-outage recovery) — the smoke's green exists only as this session's local probe until the next nightly completes.
**How to avoid:** the phase's own recorded local run IS the GAPC-01 evidence; note the nightly as the standing follow-up (it is already the #22-class discipline).

## Code Examples

### The empirical load run (GAPC-01) — verified green this session

```python
# Source: this session's probe (/tmp/gapc01-probe/load_probe.py), mirroring
# tests/models/test_model_remote_code.py:56-89 verbatim in its load route.
import dnallm.utils.transformers_compat  # applies the rung ladder at import
from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer

task_config = TaskConfig(
    task_type="binary",
    num_labels=2,
    label_names=["Not promoter", "Core promoter"],
)
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/nucleotide-transformer-v2-100m-promoter",
    task_config,
    source="modelscope",
)
# Observed output (2026-10-13, transformers 5.17.0, torch 2.11.0+cu130):
#   model class: EsmForSequenceClassification
#   config.is_decoder resolved: False
#   config.add_cross_attention resolved: False
#   forward logits shape: (1, 2)
# One benign deprecation warning: `use_return_dict` is deprecated! Use `return_dict` instead!
```

### The reachability proof for the one known gap

```python
# Source: /home/forrest/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/
#         snapshots/master/modeling_esm.py:1028-1041 (verbatim structure)
if self.config.is_decoder and encoder_hidden_states is not None:
    ...
    encoder_extended_attention_mask = self.invert_attention_mask(
        encoder_attention_mask
    )
else:
    encoder_extended_attention_mask = None
# checkpoint config.json: is_decoder ABSENT, add_cross_attention ABSENT →
# legacy-defaults rung answers False → branch unreachable on every
# binary/token load route (encoder_hidden_states is never passed by dnallm).
```

### Gate commands (verified shapes)

```bash
# Census assertion (stage-0.5 exact flags) — reproduces 216/217 locally
.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"

# Fast lane (CI coverage-gate shape, minus coverage for local speed)
.venv/bin/python -m pytest -m "not slow" -q

# Benchmark regression gate (35 tests)
.venv/bin/python -m pytest tests/benchmark/test_benchmark.py -q

# Skip audit over a produced junit
.venv/bin/python scripts/audit_skips.py <junit.xml> tests/expected_skips.yaml

# Ledger closure
gsd-tools windows fixed 11   # (then 12, 15) — reason carries the probe evidence
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GAP-1 treated as owner-pending terminal rung (05-04 D-07: "config-attribute dependency, not vendored pure helper") | Closed-map `PretrainedConfig.__getattr__` rung restoring READ behavior | 2026-10-02 owner instruction (rung supersedes D-07 termination) [VERIFIED: transformers_compat.py:489-503 comment] | The terminal failure is fixed; ledger/prose never caught up |
| "exactly `invert_attention_mask` missing" (milestone probe, 2026-10-11) | Dead code for encoder checkpoints (this session's reachability + execution proof) | 2026-10-13 | Ladder stays at zero new rungs |

**Deprecated/outdated:**
- The module docstring of `tests/models/test_model_remote_code.py` ("the load then hit the sanctioned fallback ladder's terminal rung…") and the GAP-1 comment block of `tests/examples/test_script_execution.py:35-45` — both narrate the pre-2026-10-02 state as current; they are the primary honesty edits of the shim path.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The next coverage-nightly will pass the models-lane smoke (prediction from the identical local route) | Pitfall 6 / Open Questions | A CI-side transformers-drift breakage would need a new evidence-sized rung — the marker branch re-skips loudly, the phase contract's re-verification catches it |
| A2 | CI legs currently resolve a transformers ≥5.19 variant (observed 5.19.0 historically; current resolution not probed this session) | Pitfall 4 | Same as A1 — the device-type rung already covers the one known 5.19 breakage class |
| A3 | rice.uga.edu stays reachable-or-typed-skipped for cold-cache nightly script-lane runs (existing network-variance absorption; not re-probed) | Environment Availability | None to the disposition — the network-unavailable typed skip is the sanctioned absorption path |

## Open Questions

1. **Where does the GAPC-01 evidence artifact live?**
   - What we know: the requirement says "on record"; no committed location is mandated.
   - What's unclear: phase-dir verification notes vs a committed doc.
   - Recommendation: record the captured output in the phase dir (verification artifact) and reference it from the `windows fixed` reasons; do NOT add a repo file — the smoke test itself is the durable, re-runnable evidence.
2. **Does the owner checkpoint still present removal as a live option?**
   - What we know: CONTEXT retains ONE confirmation with the evidence table (shim vs removal).
   - What's unclear: nothing material — the evidence table now shows zero rungs needed and nightly-green functionality on all three surfaces; removal would delete working coverage.
   - Recommendation: present the table with the measured facts; the recommendation column says shim-path/zero-rung.
3. **Should `_set_gradient_checkpointing` signature drift (flagged in milestone research as a benign-override candidate) be pre-empted?**
   - What we know: it did NOT surface in either load run.
   - Recommendation: no — same evidence-sizing rule as `invert_attention_mask`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| dev venv (.venv) | all local gates | ✓ | py3.13 | — |
| transformers | GAPC-01 evidence target | ✓ | 5.17.0 | — |
| torch | model forward | ✓ | 2.11.0+cu130 | — |
| modelscope snapshot caches (both NT checkpoints + weights) | load probes | ✓ | master snapshots | cold download (network verified up, ~56 MB/s during probe) |
| pybedtools | script-lane local rerun (optional) | ✓ (import) | installed | — |
| bedtools CLI | script-lane local rerun (optional) | ✗ (not on dev-box PATH) | — | not needed for phase gates (fast lane/benchmark/census are the required gates; the nightly box carries the rootless prefix) |
| gh CLI | nightly junit forensics | ✓ | available (runs queried this session) | — |
| network (modelscope.cn, HF mirror) | checkpoint fetch | ✓ | verified live | — |

**Missing dependencies with no fallback:** none — every phase gate is locally runnable.
**Missing dependencies with fallback:** bedtools CLI (only needed to re-execute the slow script lane locally; the recorded nightly green plus the gates suffice).

## Security Domain

`security_enforcement: true`, ASVS level 1. This phase touches no authentication, session, access-control, or cryptography surface; it modifies test prose, planning ledger state, and (in the disfavored path) example configs.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | no new inputs | existing typed-skip/junit parsing unchanged (audit_skips.py already fails closed on malformed allowlist entries [VERIFIED: audit_skips.py:26-40]) |
| V6 Cryptography | no | — |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| trust_remote_code execution of third-party checkpoint code | Tampering/Elevation | Already accepted project-wide; the rung ladder restores API *names* only, never executes remote code itself; checkpoints pinned by commit in `models.lock` (promoter @c138b8c5…, plant-BPE @e66bf253…, both `source=modelscope`) [VERIFIED: models.lock:32-33] |

## Sources

### Primary (HIGH confidence — live probes and file reads this session)
- Live load+forward probes, dev venv transformers 5.17.0: `/tmp/gapc01-probe/load_probe.py`, `/tmp/gapc01-probe/bpe_leg_probe.py` (both legs green; full BPE-leg traceback captured at `/tmp/gapc01-probe/bpe_full.log`)
- `dnallm/utils/transformers_compat.py` (full read; rung inventory, apply_patches, legacy-defaults closed map)
- `tests/models/test_model_remote_code.py` (full read; marker + route)
- `tests/examples/test_script_execution.py` (full read; marker + rice cache mechanics)
- Checkpoint artifacts: `~/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/snapshots/master/{config.json, modeling_esm.py, esm_config.py}`; plant-BPE equivalents
- Example-nightly junit of green run 38308972028 (downloaded + parsed: 211P/5S/1D; benchmark/BPE/embedding/data_generation rows all PASS)
- `.github/workflows/ci.yml` (example-nightly stages, census pin, audit call sites, fast-lane command)
- `tests/expected_skips.yaml`, `scripts/audit_skips.py`, `models.lock`, `pyproject.toml:67`
- Local census reproduction: `216/217 tests collected (1 deselected) in 1.03s`

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` §F (milestone probe table — superseded on `invert_attention_mask` by this session's execution evidence)
- `.planning/research/PITFALLS.md` Pitfall 11 (35-family blast radius framing)

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Disposition evidence (loads green, ladder sufficient): HIGH — live execution + nightly junit
- Blast-radius inventories: HIGH — every listed file read this session
- CI-side future behavior (transformers drift, nightly outcomes): MEDIUM — historical evidence only (Assumptions A1/A2)

**Research date:** 2026-10-13
**Valid until:** 2026-11-13 (stable — in-repo facts; only CI resolution drift can erode A1/A2)
