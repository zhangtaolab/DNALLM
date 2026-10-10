# Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities - Pattern Map

**Mapped:** 2026-10-09
**Files analyzed:** 12 (5 lanes B1–B5 + shared append surface)
**Analogs found:** 12 / 12 (all analogs git-tracked — verified via `git ls-files`)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/configuration/configs.py` (B1: Ia3Config refinement, `peft_dry_run`, cross-field validator) | config/model | transform | `dnallm/configuration/configs.py:263-354` (TrainingConfig validators — same file) | exact |
| `dnallm/finetune/trainer.py` (B1: IA³ branch, dry-run, ratio guard, lora×ia3 rejection) | service | batch | `dnallm/finetune/trainer.py:181-197` (LoRA/QLoRA branch) | exact |
| `dnallm/inference/inference.py:111-131` (B1: PeftModel save/reload shared path) | service | request-response | existing `lora_adapter` block (same region) | exact |
| `dnallm/configuration/presets/lora_targets.yaml` (B1: NEW) | config/data | file-I/O | `dnallm/models/model_info.yaml` (packaged YAML registry) + package-data glob in `pyproject.toml` | role-match |
| `dnallm/models/model.py` (B2: `random_init` kwarg, from_config path, allowlist frozenset) | service | request-response | `dnallm/models/model.py:753-918` (`load_model_and_tokenizer` dispatch chain) | exact |
| `dnallm/inference/probing.py` (B3: NEW) | service | batch/transform | `dnallm/inference/inference.py:611-713` (`_setup_hidden_states_config` + embedding extraction) | role-match |
| `dnallm/finetune/sweep.py` (B4: NEW) | service | batch | `dnallm/finetune/trainer.py:625-652` (result-JSON writer + evaluate loop) | role-match |
| `dnallm/inference/vep.py` (B5: `evaluate_vcf`, `score_variant` paradigm guard) | service | batch/file-I/O | `dnallm/inference/vep.py` (landed `align_variant` + kernels — same file) | exact |
| `dnallm/cli/vep.py` (B5: NEW `dnallm-vep` entry) | controller/CLI | request-response | `dnallm/cli/mutagenesis.py` (D-10 names it as the precedent) | exact |
| `pyproject.toml` (B5: scikit-allel dep + entry point) | config | — | existing `[project.scripts]` block (lines 271-277) | exact |
| `tests/**` new test files (all lanes) | test | — | `tests/finetune/test_trainer.py:567+` (TestLoraWiring) + `tests/conftest.py:160-235` fixtures | exact |
| `CHANGELOG.md` (shared append REV-04..REV-09) | docs | — | existing CHANGELOG entry format (D-09 discipline) | exact |

## Pattern Assignments

### `dnallm/finetune/trainer.py` IA³ branch (B1, service, batch)

**Analog:** the LoRA branch in the same file — `dnallm/finetune/trainer.py:181-197`

```python
# LoRA / QLoRA
if use_lora:
    from ..models.model import peft_forward_compatiable

    print("[Info] Applying LoRA to the model...")

    # QLoRA: prepare 4-bit model for k-bit training
    if self.train_config.use_qlora:
        from peft import prepare_model_for_kbit_training

        print("[Info] Preparing model for 4-bit QLoRA training...")
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**config["lora"].dict())
    model = peft_forward_compatiable(model)
    self.model = get_peft_model(model, lora_config)
    self.model.print_trainable_parameters()
```

The IA³ branch replaces the interim warn block at `trainer.py:171-179` (`use_ia3` print) with the same shape minus the kbit prep (rejected combination). Note the print style: `[Info] ...` / `[Warning] ...` via `print` in trainer.py (house convention), `get_logger` elsewhere.

**Non-TrainingArguments field popping** — `trainer.py:222-236` (extend, don't remove):
```python
training_args = self.train_config.model_dump()
...
training_args.pop("use_qlora", None)
training_args.pop("use_ia3", None)          # already popped at line 230 — keep
training_args.pop("allow_test_as_eval", None)
```
B1 adds `training_args.pop("peft_dry_run", None)` here (D-03).

**Result-JSON writer** (B4's seed-JSON mirror) — `trainer.py:639-651`:
```python
result_path = Path(self.train_config.output_dir) / f"eval_{split}_result.json"
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(
        {"split": split, "timestamp": datetime.now(timezone.utc).isoformat(),
         "metrics": metrics, "runtime": runtime},
        f, indent=2,
    )
```

**Test analog** — `tests/finetune/test_trainer.py:567-599` `TestLoraWiring`: mocked fast lane with `mock_hf_boundary` fixture + `patch("builtins.print")` for `[Warning]`/`[Info]` assertions. The two interim-warn tests (`test_use_ia3_warns_no_effect_yet`, `test_use_ia3_default_does_not_warn`) are the designated demolition site; `test_use_lora_wraps_model_via_peft` (line 601+) is the wiring-test shape to replicate for IA³ (patch `get_peft_model`, fake PeftModel with controllable `requires_grad` tensors).

---

### `dnallm/configuration/configs.py` Ia3Config + cross-field validators (B1, config, transform)

**Analog:** same file. Field-validator pattern — `configs.py:336-354`:
```python
@field_validator("report_to")
@classmethod
def validate_report_to(cls, v: list[str]) -> list[str]:
    valid = {"tensorboard", "wandb", "none", "all"}
    invalid = set(v) - valid
    if invalid:
        raise ValueError(f"Invalid report_to values: {invalid}. Valid: {valid}")
    ...
```

Existing Ia3Config stub at `configs.py:391-423` already carries `target_modules` / `feedforward_modules` / `init_ia3_weights` / `modules_to_save` — field-for-field compatible with peft 0.21.1's real `IA3Config` surface (which has `feedforward_modules`, NOT `feedforward_only` — D-18). The `use_ia3 × use_qlora` rejection is a `model_validator` on `TrainingConfig` (both fields live there, `configs.py:319-334`); the `lora × ia3` rejection cannot see the ctor kwarg — put it at trainer-init time as a matchable `ValueError` (RESEARCH Open Question 1 recommendation).

LoraConfig at `configs.py:357-388` shows the `Field(default=..., pattern=..., description=...)` idiom for any new fields (`peft_dry_run` mirrors `allow_test_as_eval` at lines 310-318: bool + description documenting semantics).

---

### `dnallm/inference/inference.py:111-131` adapter reload (B1, service, request-response)

**Analog:** the existing `lora_adapter` block (same file/region) — `dnallm/inference/inference.py:111-131`:
```python
if lora_adapter:
    from peft import PeftModel
    from ..models.model import peft_forward_compatiable

    if os.path.isdir(lora_adapter):
        source = "local"
    else:
        source = model.source if hasattr(model, "source") else "huggingface"
    try:
        lora_adapter_path, _ = _get_model_path_and_imports(lora_adapter, source)
    except Exception as e:
        raise ValueError(f"Failed to load LoRA adapter from {lora_adapter}: {e}") from e

    ...
    model = peft_forward_compatiable(model)
    self.model = PeftModel.from_pretrained(model, lora_adapter_path)
    logger.info(f"Loaded LoRA adapter from {lora_adapter}")
```
`PeftModel.from_pretrained` is already adapter-type-agnostic — the shared-path change is naming/logging, not mechanics. Error wrap idiom: `raise ValueError(...) from e` at the load boundary.

---

### `dnallm/configuration/presets/lora_targets.yaml` (B1, NEW, config data)

**Analog:** `dnallm/models/model_info.yaml` — packaged YAML registry read at import/run time. Loading precedent is `importlib.resources` (RESEARCH Standard Stack); the `pyproject.toml` package-data glob for `dnallm/configuration/presets/*.yaml` already landed in Phase 10 — no pyproject change needed (B1 must not touch pyproject; B5 owns it).

---

### `dnallm/models/model.py` random_init (B2, service, request-response)

**Analog:** `load_model_and_tokenizer` dispatch chain — `dnallm/models/model.py:753-918`:

- Signature/kwargs: `load_model_and_tokenizer(model_name, task_config, source="local", use_mirror=False, revision=None, custom_tokenizer=None, quantization_config=None)` (lines 753-761) — add `random_init: bool = False` in the same style; Google-style `Args:`/`Raises:` docstring already present (lines 770-787).
- Special-family chain: `_handle_evo2_models(...)`, `_handle_evo1_models(...)`, etc., each `if result is not None: return result` (lines 806-863) — the `RANDOM_INIT_SUPPORTED_FAMILIES` frozenset check slots at the top of this chain: off-list special families raise `ValueError` (D-06/D-07).
- Guarded merge chain at lines 898-918 (crossdna → dnabert2 → generic `_load_model_by_task_type`): note the "must NOT return early" comment — the random_init generic path must preserve the tokenizer post-processing below.
- `ValueError` idiom: `raise ValueError(f"num_labels should be at least 2 for task type '{task_type}', but got {safe_num_labels}.")` (lines 744-748) — matchable f-string messages.
- Logging: `get_logger` (NOT trainer's print style) for D-05's per-tensor hash INFO lines.
- No-download proof target: `_get_model_path_and_imports(model_name, source)` (line 868) is the weight-fetch path to patch/assert.

---

### `dnallm/inference/probing.py` (B3, NEW, service, batch/transform)

**Analog:** `dnallm/inference/inference.py:611-646` `_setup_hidden_states_config`:
```python
def _setup_hidden_states_config(
    self, output_hidden_states: bool
) -> tuple[bool, dict, dict | None]:
    ...
    embeddings: dict[str, Any] = {
        "hidden_states": None, "attention_mask": [], "labels": [],
    }
    if "output_hidden_states" in params:
        try:
            self.model.config.output_hidden_states = True
        except ValueError as e:
            warnings.warn(f"Cannot enable output_hidden_states: {e}", stacklevel=2)
            return False, {}, params
    return True, embeddings, params
```
probing.py reuses this hidden-states mechanics read-only (call `scoring(output_hidden_states=True)`-adjacent paths rather than re-implementing); adds layer/pooling selection over the returned `hidden_states` tuple and the npz cache. Module skeleton/docstring analog: `dnallm/inference/vep.py:1-44` (module docstring with numbered Features list + `Example:` block, `from __future__ import annotations`, dataclass result types). Module-level constants with docstrings per D-13 (RESEARCH Code Examples block gives the exact constant set). Metrics via `dnallm/tasks/metric_registry.py` `resolve(name)` (line 380) / `registered_names()` (line 420) — read-only.

Cache/path discipline: outputs under caller-supplied `output_dir/probe_cache/`, never CWD — mirror `trainer.py:639-640`'s `result_path.parent.mkdir(parents=True, exist_ok=True)` idiom. Window-fetch uppercase precedent: `dnallm/utils/genomic_coords.py:130` `fetch_sequence(..., uppercase=True)`.

---

### `dnallm/finetune/sweep.py` (B4, NEW, service, batch)

**Analog:** `dnallm/finetune/trainer.py:625-652` (evaluate + result-JSON writer, quoted above) for the per-seed output JSON; `DNATrainer.train()` (`trainer.py:451`) / `evaluate()` (`trainer.py:559`) are the per-seed entry points orchestrated. Structure it import-light (pure-function `aggregate_seeds` testable without torch). Consumes `SweepConfig` verbatim (`configs.py:455-498`: `seeds`, `out_root`, `n_bootstrap`, `bootstrap_seed`, `small_n_ci`). `run_seeds` output protocol `{model}/{task}/seed_{s}/` under `out_root`, mkdir via the `parents=True, exist_ok=True` idiom. n-guard / bootstrap code is given verbatim in RESEARCH Code Examples — copy that, don't redesign.

---

### `dnallm/inference/vep.py` evaluate_vcf (B5, service, batch + file-I/O)

**Analog:** the landed Phase-10 core in the same file — `dnallm/inference/vep.py:46-120`:
- `@dataclass(frozen=True) class VariantAlignment` with `Attributes:` docstring (lines 46-68) — skip-record style for the driver's per-variant output.
- `align_variant(sequence, pos, ref, alt, tokenizer)` with 0-based `pos` contract (lines 94-120): the driver owns 1-based VCF POS → `pos0 = pos - 1`, `.upper()` on windows (Pitfall 5), and identical-left-context ref/alt windows.
- Module docstring (lines 22-27) explicitly defers `evaluate_vcf` + CLI to this phase — extend the docstring's numbered feature list when adding.

The variant-loop driver shape is in RESEARCH Code Examples (quoting this file). Paradigm guard (D-11): plain `ValueError` with matchable message, same idiom as `_tokenize_ids`' callers.

---

### `dnallm/cli/vep.py` (B5, NEW, controller/CLI)

**Analog:** `dnallm/cli/mutagenesis.py` (D-10 names it the precedent). Key excerpts to copy:

Structure (lines 1-15):
```python
#!/usr/bin/env python3
"""Standalone mutagenesis CLI for DNALLM."""
...
import click
import numpy as np

from ..utils import get_logger

logger = get_logger("dnallm.cli.mutagenesis")
```
(Relative imports inside `dnallm/`; `get_logger` namespaced `dnallm.cli.vep`.)

Command shape (lines 31-37, 75-88):
```python
@click.command()
@click.option("--sequence", "-s", type=str, help="...")
...
@click.option("--model-name", "-m", type=str, required=True, help="...")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path (defaults to stdout)")
def main(...):
    """Run in-silico mutagenesis analysis on DNA sequences."""
    from ..inference.mutagenesis import Mutagenesis   # lazy import inside command body
    from ..models import load_model_and_tokenizer
```

Error handling (lines 131-141, 256-258):
```python
except Exception as e:
    click.echo(f"Error loading model '{model_name}': {e}", err=True)
    sys.exit(1)
```

Output writing (lines 247-254): `out_path.parent.mkdir(parents=True, exist_ok=True)` + `json.dump(..., indent=2)` or stdout. Add a `--config/-c` option mirroring the CLI conventions (`dnallm/cli/cli.py`) for the VepConfig YAML path, plus `--vcf`, `--reference` options.

Entry point — `pyproject.toml:271-277` `[project.scripts]`: append `dnallm-vep = "dnallm.cli.vep:main"` next to `dnallm-mutagenesis`; add `"scikit-allel>=1.3.13,<2"` to the dependency list (D-08 bounded range).

---

## Shared Patterns

### Error handling: matchable ValueError
**Source:** `dnallm/models/model.py:744-748`, `dnallm/inference/inference.py:122`, `dnallm/utils/genomic_coords.py:117-126`
**Apply to:** every lane (D-04 ratio guard, D-06 allowlist, D-11 paradigm guard, B1 cross-field rejections)
```python
raise ValueError(
    f"num_labels should be at least 2 for task type '{task_type}', "
    f"but got {safe_num_labels}."
)
# boundary wrap:
raise ValueError(f"Failed to load LoRA adapter from {lora_adapter}: {e}") from e
```
Tests use `pytest.raises(ValueError, match=r"...")` against dnallm's own messages only (Pitfall 10 — never match peft/transformers foreign error strings).

### Logging split
**Source:** trainer.py uses `print("[Info] ...")` / `print("[Warning] ...")` (lines 171-197, 249); everything else uses `logger = get_logger("dnallm.<module>")` (cli/mutagenesis.py:15, inference.py).
**Apply to:** B1's trainer branch uses `[Info]` prints; B2's hash proof (D-05), B3/B4/B5 use `get_logger`.

### Output-path discipline (never CWD)
**Source:** `dnallm/finetune/trainer.py:639-640` — `Path(output_dir) / name` + `mkdir(parents=True, exist_ok=True)`.
**Apply to:** B3 probe_cache (D-12), B4 out_root, B5 result JSON.

### Mocked fast-lane testing
**Source:** `tests/conftest.py:168-235` (`simple_dna_tokenizer`, `tiny_model_factory`, `tiny_real_model`) + `tests/finetune/test_trainer.py:567-620` (TestLoraWiring with `mock_hf_boundary`, `Mock()` model, `patch("builtins.print")`).
**Apply to:** all lanes; per-module scoped coverage ≥96% (e.g. `--cov=dnallm.inference.vep --cov-report=term-missing`). Slow-lane discipline: `slow` marker, typed network skips + `tests/expected_skips.yaml` entry, models.lock rows same-change.

### Docstring style
**Source:** `dnallm/inference/vep.py:1-36` (module: summary + numbered features + `Example:`), `:46-62` (dataclass `Attributes:`), `dnallm/utils/genomic_coords.py:130-153` (function `Args:`/`Returns:`/`Raises:` with untyped-param style).
**Apply to:** probing.py, sweep.py, cli/vep.py, all new public functions.

### CHANGELOG append
**Source:** existing CHANGELOG.md entry format; D-09 discipline — REV-ID + reviewer-comment inline, unique anchors, append-only (B1–B5 each append REV-04..REV-09 sections without touching others' entries).

## No Analog Found

None — every file has a tracked in-repo analog or extends an existing file in place. The genuinely-new behaviors (sklearn probe fitting, scipy t-interval, scikit-allel read_vcf, `Auto*.from_config`) have no codebase analog; use the verbatim code examples in 11-RESEARCH.md § Code Examples / § Architecture Patterns for those (research already pinned exact APIs against installed versions).

## Metadata

**Analog search scope:** dnallm/finetune, dnallm/inference, dnallm/models, dnallm/configuration, dnallm/cli, dnallm/utils, tests/, pyproject.toml
**Files scanned:** ~14 (targeted reads; analog line ranges cited above)
**Tracked-source gate:** all named analogs verified via `git ls-files` (non-empty for every path)
**Pattern extraction date:** 2026-10-09
