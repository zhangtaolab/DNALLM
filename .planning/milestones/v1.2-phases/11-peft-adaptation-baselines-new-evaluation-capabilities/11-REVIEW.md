---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
reviewed: 2026-10-09T15:41:00Z
depth: standard
iteration: 3
files_reviewed: 26
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
  - tests/inference/test_inference.py
  - tests/inference/test_probing.py
  - tests/inference/test_vep.py
  - tests/models/test_model.py
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 11: Code Review Report (Iteration 3 — convergence check)

**Reviewed:** 2026-10-09T15:41:00Z
**Depth:** standard
**Files Reviewed:** 26
**Status:** clean

## Summary

Iteration 3 of 3 — focused convergence check after iteration 2's single
Critical (the stale adapter-reload regex) was fixed in commit `7c06b5c`.
All four convergence criteria are green. All reviewed files meet quality
standards. No issues found.

Scope note: the evaluation-scope resolver returned `status: degraded`
(reason: `no-task-commit-rows` — resolved via phase-range instead of task
rows). The file set is usable and complete: 26 files, none missing on disk,
none outside the union. This is the full phase-11 scope at diff_base
`710c437`.

## Convergence Evidence

### (a) The `7c06b5c` fix holds at HEAD

- HEAD **is** `7c06b5c` (`fix(11): CR-01 align adapter-reload test regex
  with PEFT-agnostic message`) — `git log --oneline -1` confirms; there is
  no later commit that could have regressed it.
- Working tree is clean for every source path: `git diff HEAD --stat --
  dnallm/ tests/ pyproject.toml` is empty; only `.planning/` artifacts are
  dirty.
- `tests/inference/test_inference.py:1711` reads
  `with pytest.raises(ValueError, match=r"Failed to load PEFT adapter"):`
  and the source at `dnallm/inference/inference.py:122` raises
  `ValueError(f"Failed to load PEFT adapter from {lora_adapter}: {e}")` —
  the regex prefix-matches the actual message.

### (b) Full fast lane — CI gate shape — zero failures

```
uv run --no-sync pytest tests/ -q -m "not slow"
=> 2253 passed, 1 skipped, 66 deselected, 7 warnings in 98.87s
=> exit code 0
```

Zero failures. The 1 skip and 66 deselected are governed by
`tests/expected_skips.yaml` and the `slow` marker (the coverage-gated CI
lane re-includes slow separately). The 7 warnings are pre-existing
third-party noise (dill PicklingWarning on MagicMock, numpy divide
warnings in plot/mutagenesis test paths, one never-awaited coroutine
warning asserted by an MCP client test) — none originates from phase-11
code.

### (c) Ruff repo-green

```
ruff format --check .  => 298 files already formatted
ruff check .           => All checks passed!
```

Both gates that CI enforces (`.github/workflows/ci.yml` format + lint
jobs) are green — this also confirms iteration-1's CR-01 (format violation
in `tests/models/test_model.py`) remains fixed.

### (d) No stale matcher of any phase-11-renamed message anywhere in the tree

Method: extracted every string literal REMOVED from `dnallm/` between
diff_base `710c437` and HEAD (set-difference against strings added in the
same range). Exactly two runtime messages were renamed:

| Pre-rename (removed) | Post-rename (current) | Site |
|---|---|---|
| `Failed to load LoRA adapter from {lora_adapter}: {e}` | `Failed to load PEFT adapter from {lora_adapter}: {e}` | `dnallm/inference/inference.py:122` |
| `Loaded LoRA adapter from {lora_adapter}` | `Loaded PEFT adapter from {lora_adapter}` | `dnallm/inference/inference.py:131` |

Greps over `tests/` **and** `dnallm/` for `Failed to load LoRA`,
`Loaded LoRA adapter`, `has no effect yet`, and `LoRA adapter from`
(excluding the PEFT form) return **zero matches**. The remaining removed
strings are docstring fragments (the superseded "IA³ training support
arrives with the next release" notes from before lane 11-01 implemented
the branch) — not matchable runtime messages, no test references them.

## Prior-fixes-hold verification (all 14 iteration-1 findings + iteration-2 CR-01)

Each guard was re-confirmed present at HEAD by direct source inspection:

| Fix | Evidence at HEAD |
|---|---|
| CR-01 (format) | `ruff format --check .` green (see c) |
| WR-01 dry-run without adapter | `dnallm/finetune/trainer.py:411-413` raises "requires an adapter method" |
| WR-02 no config mutation | `tests/finetune/test_trainer.py:985` builds the trainer over `MappingProxyType(...)` |
| WR-03 preset row validation | `dnallm/finetune/trainer.py:123-129` — `match_names` non-empty + band-entries numeric |
| WR-04 honest VEP error attribution | `dnallm/inference/vep.py:81` `class RefMismatchError(ValueError)`; only that subclass is relabeled |
| IN-01 dead elif | `dnallm/inference/vep.py:521-525` plain `else` with rationale comment |
| IN-02 redundant mkdir | only `task_root` (sweep.py:306) and `seed_dir` (sweep.py:310) mkdirs remain |
| IN-03 multi-pin convention | `models.lock` header documents pin form + per-lane purpose |
| IN-04 hidden-states restore | `dnallm/inference/probing.py:394-412` — `_UNSET` sentinel, `try/finally` restore |
| IN-05 unnamed FASTA record | `dnallm/inference/vep.py:490` raises "has an unnamed record" |
| IN-06 POS beyond chromosome end | `dnallm/inference/vep.py:885` raises "exceeds the reference length" |
| IN-07 single-class train split | `dnallm/inference/probing.py:641-645` pre-check, matchable message |
| IN-08 dry-run half-constructed trainer | guarded trainer-dependent methods with matchable error |
| IN-09 explicit metric_keys non-numeric | `dnallm/finetune/sweep.py` hard-fail path for explicit keys |

## Conclusion

Convergence reached at iteration 3. The phase-11 codebase passes the full
fast lane (2253 tests), both ruff gates, carries zero stale matchers of
renamed messages, and every fix from iterations 1 and 2 is present and
test-covered at HEAD. No new findings.

---

_Reviewed: 2026-10-09T15:41:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Iteration: 3 of 3 (convergence)_
