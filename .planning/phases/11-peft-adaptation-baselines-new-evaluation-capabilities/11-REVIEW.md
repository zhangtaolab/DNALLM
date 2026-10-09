---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
reviewed: 2026-10-09T15:05:00Z
depth: standard
files_reviewed: 25
files_reviewed_list:
  - CHANGELOG.md
  - dnallm/cli/vep.py
  - dnallm/configuration/configs.py
  - dnallm/configuration/presets/lora_targets.yaml
  - dnallm/finetune/sweep.py
  - dnallm/finetune/trainer.py
  - dnallm/inference/inference.py
  - dnallm/inference/probing.py
  - dnallm/inference/vep.py
  - dnallm/models/model.py
  - models.lock
  - pyproject.toml
  - README.md
  - tests/cli/test_vep_cli.py
  - tests/configuration/test_configs.py
  - tests/configuration/test_peft_presets.py
  - tests/expected_skips.yaml
  - tests/finetune/test_sweep.py
  - tests/finetune/test_trainer.py
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/data/synthetic_reference.txt
  - tests/inference/data/synthetic_variants.vcf
  - tests/inference/test_probing.py
  - tests/inference/test_vep.py
  - tests/models/test_model.py
findings:
  critical: 1
  warning: 4
  info: 9
  total: 14
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-10-09T15:05:00Z
**Depth:** standard
**Files Reviewed:** 25
**Status:** issues_found

## Summary

Phase 11's five lanes were reviewed at standard depth with the cross-lane
contracts checked explicitly. The functional core is strong: all 366 fast-lane
tests for the new modules pass locally (sweep/presets/vep-cli/probing/vep:
151 passed; random-init/configs/trainer: 215 passed), `ruff check` is clean,
and every documented deviation holds as described. The cross-lane contracts
verified:

- **B1 preset chain**: `lora_targets.yaml` structure is validated at load;
  the family set is pinned equal to `PRETRAIN_MODEL_MAPS` by test; preset
  injection writes the same section object both adapter branches consume;
  the dry-run validator and the D-04 ratio guard read the same row. peft
  kwargs are filtered through the installed `IA3Config` dataclass fields
  (verified against installed peft 0.21.2 — every dnallm `Ia3Config` field
  passes through). The dnamamba `out_proj` LoRA exclusion and the
  `use_ia3`×`use_qlora` / LoRA×IA³ rejections are implemented and tested.
- **Metric registry**: probing emits `AUROC`/`AUPRC`/`accuracy` exclusively
  through `metric_registry.resolve` plus `validate_emission`; vep emits
  `AUROC`/`AUPRC` through `resolve`; sweep aggregates only (no metric
  emission). No sklearn metric bypass found.
- **B5 VEP**: windows are always uppercased before alignment (proven by the
  fast fixture's lowercase-context row being scored, not skipped); skip
  accounting is per-allele with the three machine-readable reasons kept
  separate from the D-17 convention exclusion buckets; the fixture's 13
  considered / 11 evaluated / 2 skipped / exclusion-count assertions match a
  hand-verification of every fixture row's coordinates against the committed
  reference (all 14 rows' REF alleles verified base-exact, contig length 160
  exact). AUROC over `-delta` and the 0.45 sanity floor are documented
  deviations that hold as described.
- **B4 sweep**: `seed_result.json` carries the `timestamp`/`metrics` blocks
  consistent with the Phase-10 `eval_{split}_result.json` convention; the
  n-guard boundaries (n<3 none, 3–9 t/omit, ≥10 bootstrap) are exact and
  boundary-tested at n=2/3/9/10.
- **Packaging**: the pyproject delta is exactly
  `scikit-allel>=1.3.13,<2` + the `dnallm-vep` entry (verified via
  `git diff` — dependency lists otherwise byte-identical vs diff_base); the
  presets YAML is covered by the pre-existing
  `"dnallm.configuration" = ["presets/*.yaml"]` package-data glob;
  `dnallm/__init__.py` was not touched. models.lock's 3 new `ms` rows are
  well-formed `@sha` pins and all 5 VEP acceptance models have lock rows;
  the `clinvar-unavailable:` skip prefix was allowlisted in the same change.

One CI-gate regression (a ruff-format violation in the new random-init test
code) is the only Critical finding. Four Warnings concern robustness of the
new PEFT/dry-run plumbing and VEP error attribution; none invalidates the
phase's acceptance results.

## Critical Issues

### CR-01: New test code fails the CI `ruff format --check .` gate

**File:** `tests/models/test_model.py:936-942, 994-996, 1001-1003`
**Issue:** `ruff format --check` on the current tree reports "1 file would be
reformatted". The three offending hunks are all inside the new random-init
test helpers (`FakeAutoConfig.__init__`'s ternary, and two over-wrapped
`patch(...)` calls in `_random_init_patches`). The file at `diff_base`
(710c437) is format-clean, so this is introduced by Phase 11. CI runs
`ruff format --check .` in at least two jobs (`.github/workflows/ci.yml:99`
and `ci.yml:178`), so the required checks will go red on the next push —
this is exactly the false-green-gate class the 0.6.0 release worked to close.
**Fix:** run `ruff format tests/models/test_model.py` (or
`python scripts/check_code.py --fix`) and commit the reformat:

```bash
.venv/bin/ruff format tests/models/test_model.py
.venv/bin/ruff format --check .   # must report "N files already formatted"
```

## Warnings

### WR-01: `peft_dry_run=true` without an adapter flag silently runs full training

**File:** `dnallm/finetune/trainer.py:383-412`
**Issue:** The dry-run path only executes inside `if peft_kind is not None:`
(i.e. `use_lora=True` or `finetune.use_ia3=true`). A user who sets
`finetune.peft_dry_run=true` but neither adapter flag gets no validation, no
report, no warning — `DNATrainer` proceeds straight into `set_up_trainer()`
and `train()` runs a full fine-tune (potentially hours of GPU). The flag's
documented purpose is "validate … and exit before training"; the silent
no-op inverts that intent. The TrainingConfig field description does say
"(with use_ia3 or LoRA)", but the failure mode is silent rather than loud —
against this module's own fail-loud convention.
**Fix:** In `DNATrainer.__init__`, when `self.train_config.peft_dry_run` is
true but `peft_kind is None`, raise (or at minimum print a loud `[Warning]`):

```python
if self.train_config.peft_dry_run and peft_kind is None:
    raise ValueError(
        "finetune.peft_dry_run=true requires an adapter method: pass "
        "use_lora=True or set finetune.use_ia3=true. Refusing to start a "
        "full training run under a dry-run flag."
    )
```

### WR-02: `DNATrainer.__init__` mutates the caller's config Mapping

**File:** `dnallm/finetune/trainer.py:387-391` (and `402`)
**Issue:** The ctor parameter is typed `Mapping[str, Any]` (immutable
interface), but the new code performs `config["ia3"] = section` and then
mutates the section in place (`section.target_modules = targets`,
`section.feedforward_modules = ...`). Consequences: (a) an immutable mapping
(e.g. `MappingProxyType`, some frozen config containers) raises `TypeError`
at trainer construction; (b) a caller-owned config dict now carries injected
preset targets — a second `DNATrainer` built from the same dict silently
skips preset resolution and any downstream consumer sees trainer-internal
state. The LoRA branch has always required `config["lora"]` to exist, but
writing into the config is new.
**Fix:** Either narrow the annotation to `dict[str, Any]` / `MutableMapping`
to make the mutation honest, or avoid the write: resolve targets into local
variables and build the peft configs from the locals, registering the
default `Ia3Config` only when the key is absent (with a comment that the
config is intentionally treated as mutable), e.g.:

```python
ia3_section = config.get("ia3") or Ia3Config()
# operate on ia3_section; do not write back into config
```

### WR-03: Preset-table validation misses `match_names` and band typing — corrupted rows crash later with foreign errors

**File:** `dnallm/finetune/trainer.py:113-134` (validation) vs `166-169`, `127-131`
**Issue:** `_load_peft_presets` documents that a corrupted table "fails
loudly here instead of silently freezing a backbone mid-training", and
checks target lists, FFN subset, and band shape. But it does not check
`match_names` (non-empty list) or that ratio-band entries are numeric: (a) a
row with `match_names: []` passes validation, and `_resolve_peft_preset`
then dies on `max(len(m) for m in kv[1].get("match_names") or [])` with
`ValueError: max() arg is an empty sequence` — a foreign, unmatchable
message; (b) a band like `["1e-4", "5e-3"]` passes the `len(band) != 2`
check and raises `TypeError` on `band[0] <= band[1]`. The packaged table is
currently fine (verified: all 31 rows carry non-empty `match_names` and
numeric bands), so this only bites hand-edited/derived tables — exactly the
corruption scenario the validator exists for.
**Fix:** In the per-row validation loop add:

```python
markers = row.get("match_names")
if not isinstance(markers, list) or not markers or not all(
    isinstance(m, str) and m for m in markers
):
    raise ValueError(
        f"The PEFT preset for family '{family}' is malformed: "
        f"'match_names' must be a non-empty list of non-empty strings."
    )
...
if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in band):
    raise ValueError(f"... '{key}' must be a [lo, hi] pair of numbers.")
```

### WR-04: `evaluate_vcf` relabels every per-variant `ValueError` as a REF/reference mismatch

**File:** `dnallm/inference/vep.py:861-868`
**Issue:** The per-variant wrap is

```python
except ValueError as exc:
    raise ValueError(f"REF/reference mismatch at {chrom}:{pos1} (REF={ref}, ALT={alt}): {exc}")
```

The comment claims the paradigm guard having already run means the wrap
"only ever sees genuine REF/reference mismatches", but `score_variant` →
`align_variant` → `tokenizer(...)` can raise plain `ValueError`s from a
misbehaving tokenizer (e.g. sequence-length or type complaints), which then
surface as a confidently-wrong coordinate-mismatch diagnosis — actively
misleading during acceptance debugging. Only `align_variant`'s specific
mismatch check should be relabeled.
**Fix:** Narrow the relabel to the alignment contract, letting other errors
propagate with their own text (optionally chained with coordinate context):

```python
try:
    outcome = score_variant(model, tokenizer, window, local_pos, ref, alt, paradigm=paradigm)
except ValueError as exc:
    if "does not match sequence at position" in str(exc):
        raise ValueError(
            f"REF/reference mismatch at {chrom}:{pos1} (REF={ref}, ALT={alt}): {exc}"
        ) from exc
    raise ValueError(f"Scoring failed at {chrom}:{pos1} (REF={ref}, ALT={alt}): {exc}") from exc
```

(A cleaner variant: have `align_variant` raise a dedicated exception
subclass and catch that.)

## Info

### IN-01: `_resolve_chromosome` dead `elif`

**File:** `dnallm/inference/vep.py:505-508`
**Issue:** `elif chrom.startswith("chr")` is always true when reached (the
`if` branch already established `not chrom.startswith("chr")`). Harmless;
simplify to `else:`.
**Fix:** `else: candidates.append(chrom[3:])`.

### IN-02: Redundant mkdir in `run_seeds`

**File:** `dnallm/finetune/sweep.py:311`
**Issue:** `result_path.parent.mkdir(parents=True, exist_ok=True)` re-creates
`seed_dir`, created three lines earlier (`seed_dir.mkdir(parents=True,
exist_ok=True)` at line 308). Same for `stats_path.parent.mkdir` (line 376)
re-creating `task_root` (line 304). Harmless but noise.
**Fix:** Delete the two redundant mkdir calls.

### IN-03: models.lock carries three rows for `plant-dnagpt-BPE-promoter`

**File:** `models.lock:12,19,39`
**Issue:** The repo now appears as one unpinned row (line 12), one
`@1311a7f…` pin (line 19, example notebooks), and the new `@78184ac…` pin
(line 39, VEP acceptance). The header declares the file provenance-only
since D-11, and registry-head drift between phases explains it, but three
"validated" revisions of one repo in a lock-named file invites confusion
about which pin a lane should trust. Consider a convention note in the
header ("multiple pins per repo are expected; each row's comment names its
lane") or collapsing older pins when superseded.
**Fix:** Documentation-level clarification in the models.lock header.

### IN-04: `extract_embeddings` permanently flips `model.config.output_hidden_states`

**File:** `dnallm/inference/probing.py:482-486`
**Issue:** The best-effort `model.config.output_hidden_states = True` is
never restored after extraction, so the caller's model keeps materializing
the full hidden-state stack (memory cost) in later forwards. The code
explicitly mirrors the `DNAInference` idiom, so this is house precedent, not
a new pattern — but unlike `DNAInference`, probing receives an externally
owned frozen model and returns without the engine owning its lifecycle.
**Fix:** Save and restore the prior value in a `try/finally` when the
attribute was settable.

### IN-05: Degenerate FASTA header yields an empty-name record

**File:** `dnallm/inference/vep.py:475`
**Issue:** A header line of exactly `>` produces `name = ""` and a
`sequences[""]` entry. Nothing downstream breaks (chrom resolution will just
never match `""`), but a malformed reference is accepted silently.
**Fix:** Treat a bare `>` header as `ValueError(f"Reference FASTA '{path}'
has an unnamed record.")`.

### IN-06: POS beyond the chromosome end surfaces as "REF/reference mismatch"

**File:** `dnallm/inference/vep.py:853-868` with `_build_window` at `537-540`
**Issue:** For `pos0 >= len(ref_seq)`, `_build_window` returns an empty
window (`end < start`), and `align_variant`'s mismatch check then reports a
REF mismatch. Technically a raised ValueError, but the message points at the
alleles instead of the coordinate being out of range — a real ClinVar vs
assembly-mismatch scenario.
**Fix:** Guard in the row loop: `if pos0 >= len(ref_seq): raise
ValueError(f"VCF position {pos1} on {chrom} exceeds reference length
{len(ref_seq)} — assembly mismatch?")`.

### IN-07: `fit_probe` propagates sklearn's foreign error on single-class train splits

**File:** `dnallm/inference/probing.py:617`
**Issue:** `estimator.fit(x_train_scaled, y_train)` with one label class in
the train split raises sklearn's
`ValueError: This solver needs samples of at least 2 classes` — foreign text,
no dnallm wrap, contradicting the project's matchable-ValueError convention
for invalid input.
**Fix:** Pre-check: `if len(set(y_train.tolist())) < 2: raise
ValueError("fit_probe requires both label classes in the train split.")`.

### IN-08: Dry-run early return leaves the trainer half-constructed

**File:** `dnallm/finetune/trainer.py:408-412`
**Issue:** After the dry-run return, `self.trainer`, `self.training_args`,
`self.data_split` do not exist; `train()` is guarded, but `evaluate()`,
`infer()`, or `plot_history()` on the same object crash with bare
`AttributeError`s. Documented as "exits before training", so low severity,
but a matchable error would match the module's conventions.
**Fix:** Set `self.trainer = None` in the dry-run branch and guard the other
public methods with `if getattr(self, "_peft_dry_run", False): raise
ValueError("not available after finetune.peft_dry_run=true")`.

### IN-09: Explicit `metric_keys` entries with non-numeric values silently vanish from statistics

**File:** `dnallm/finetune/sweep.py:346-366`
**Issue:** A metric missing from a seed's result hard-raises ("did not
report"), but a metric present yet non-numeric is only INFO-logged and then
omitted from `statistics` — including when the caller explicitly requested
it via `metric_keys`. The asymmetry means a typo'd numeric-producing fn
(e.g. returning `np.int64`, which is not a Python `int`) quietly produces an
empty statistics block. The log line mitigates this; behavior is defensible
but asymmetric.
**Fix:** When `metric_keys` was explicit and the value is non-numeric, raise
instead of skipping (auto-discovery can keep the soft skip).

---

## Cross-lane contract verification record (all pass)

| Contract | Result |
|---|---|
| presets yaml ↔ trainer loader ↔ dry-run ↔ ratio guard | Consistent; family set pinned to `PRETRAIN_MODEL_MAPS` by test; both adapter branches consume the injected section; guard keyed on the same preset row |
| probing/sweep/vep metrics via `metric_registry.resolve` | Verified; no direct sklearn metric emission in any new module |
| vep uppercase-window ↔ align_variant ↔ skip accounting; D-17 | Verified end-to-end on the committed fixture; skip channel and convention exclusions never mix; `-delta` direction documented in code, README, and proven by the unit-AUROC mocked-kernel test |
| sweep `seed_result.json` ↔ Phase-10 conventions | `{model_name, task_name, seed, timestamp, metrics}`; slow test asserts `eval_test_result.json` co-locates in the seed dir |
| pyproject delta exactly `scikit-allel>=1.3.13,<2` + `dnallm-vep` | Verified byte-exact vs diff_base via git diff |
| models.lock rows well-formed; `clinvar-unavailable:` same-change | Verified (3 new `ms @sha` rows; skip prefix added in the same change; all 5 acceptance models locked) |
| Documented deviations (AUROC/-delta, 0.45 floor, HeadConfig fix, peft-kwargs filter, dnamamba `out_proj` exclusion, README hunk) | All hold as described |
| Conventions (relative imports, docstrings, matchable ValueErrors, no `__init__` re-export additions, tests absolute imports) | Clean except as noted in WR-02/WR-04/IN-07 |
| Coverage claims plausible | Yes — 366 fast-lane tests for the new modules pass locally; error-path breadth (corrupt cache, atomic-write failure, key mismatch, bytes cells, empty VCF, absent INFO, guard variants) is unusually thorough |

_Reviewed: 2026-10-09T15:05:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
