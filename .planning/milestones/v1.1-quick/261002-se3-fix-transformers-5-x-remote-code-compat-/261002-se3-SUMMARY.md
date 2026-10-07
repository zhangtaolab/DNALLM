---
quick_task: 261002-se3-fix-transformers-5-x-remote-code-compat-
type: execute
status: complete
branch: phs
push: manual-only
commit: fdc4915
transformers_version: 5.17.0
files_changed:
  - dnallm/utils/transformers_compat.py
  - tests/utils/test_transformers_compat.py
estimate:
  tokens: 30000
actuals:
  tokens: 15500
  tasks: 2
  commits: 1
started_utc: 2026-10-02T12:35:52Z
completed_utc: 2026-10-02T12:42:07Z
---

# Quick Task 261002-se3: Restore get_extended_attention_mask for transformers 5.x remote code — Summary

Vendored the transformers v4.49.0 `ModuleUtilsMixin.get_extended_attention_mask` into
`dnallm/utils/transformers_compat.py` and re-attached it onto `PreTrainedModel`
(absence-gated, sentinel-idempotent), so 4.x-era trust_remote_code ESM checkpoints whose
`EsmModel.forward` calls `self.get_extended_attention_mask(...)` no longer crash with
`AttributeError` on transformers 5.x; 10 new contract tests lock the shape/value semantics
and the version-agnostic attachment identity.

## What Was Done

**Task 1 — Vendor + patch (tdd, RED->GREEN):**

- Provenance comment block above the vendored function, mirroring the pruning-helpers
  style: cites tag v4.49.0, `src/transformers/modeling_utils.py`
  (`ModuleUtilsMixin.get_extended_attention_mask`), names the remote-code caller
  (`zhangtaolab/nucleotide-transformer-v2-100m-promoter` modeling_esm.py
  `EsmModel.forward`), and flags the three documented deviations.
- `_get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None)`
  faithful to v4.49.0: dtype=None falls back to `self.dtype`; 3D mask unsqueezes to
  `[batch, 1, from, to]`; 2D encoder mask to `[batch, 1, 1, seq]`; other dims raise
  `ValueError` with upstream's verbatim `Wrong shape for input_ids ... or attention_mask
  ...` message; tail is upstream's exact two-step `.to(dtype=dtype)` then
  `(1.0 - mask) * torch.finfo(dtype).min`. Deviations: (1) decoder flag read as
  `getattr(self.config, "is_decoder", False)`; (2) decoder branch raises
  `NotImplementedError` (naming the method, the decoder config, and dnallm's
  transformers-5 remote-code shim) instead of delegating to the also-removed
  `create_extended_attention_mask_for_decoder`; (3) upstream's cosmetic FutureWarning on
  `device` dropped while the parameter stays in the signature for call compatibility.
- `_patch_get_extended_attention_mask()` mirrors `_patch_get_parameter_or_buffer`:
  function-local `from transformers.modeling_utils import PreTrainedModel` under a broad
  except; absence gate (`hasattr` -> return, so 4.x natives are never overwritten);
  `_dnallm_extended_mask_patch` sentinel gate; class assignment with
  `# type: ignore[method-assign]` / `# type: ignore[attr-defined]`.
- Wired as the fourth call inside `apply_patches()`; import-time activation unchanged.

**Task 2 — Contract tests + gates + single atomic commit:**

- `TestGetExtendedAttentionMask` (10 tests) appended to
  `tests/utils/test_transformers_compat.py`, using `types.SimpleNamespace` fake
  receivers (`dtype=`, `config=` with and without `is_decoder`) called as unbound
  functions: 2D (2,4)->(2,1,1,4) and 3D (2,3,4)->(2,1,3,4) expansion with 0.0 /
  `torch.finfo(dtype).min` mapping; explicit `dtype=torch.float16`; dtype=None fallback
  to receiver float64; missing-`is_decoder` encoder branch (05-04 D-07 next-rung guard);
  1D mask `ValueError` (`match="Wrong shape"`); `is_decoder=True`
  `NotImplementedError` (`match="decoder"`); live-class callable + version-agnostic
  identity (5.x IS the vendored function / 4.x is NOT — inverts, never skips);
  `apply_patches()` idempotency with sentinel state per major version; monkeypatched
  bare class receives vendored function + sentinel while a native-placeholder class is
  left untouched with no sentinel.
- ONE atomic commit `fdc4915` on `phs` containing exactly the two authorized files
  (258 insertions, 0 deletions), message
  `fix(utils): restore get_extended_attention_mask for transformers 5.x remote code`,
  no attribution trailers. Not pushed.

## Gate Evidence

Preconditions (all pass before work): `.venv/bin/python` executable; transformers+torch
importable; **transformers 5.17.0** (5.x side — Task 1 gate RED before / GREEN after);
branch `phs`; no staged changes to target files.

| Gate | Result |
|------|--------|
| Task 1 verify (RED, pre-change) | `AttributeError: type object 'PreTrainedModel' has no attribute 'get_extended_attention_mask'` — the diagnosed bug reproduces |
| Task 1 verify (GREEN, post-change) | duck-typed 2D mask -> shape (1, 1, 1, 3), float32, 0.0 and `finfo(float32).min`, config without `is_decoder` — PASS |
| `pytest tests/utils/test_transformers_compat.py -q` | **42 passed** (32 pre-existing + 10 new), 3.7s |
| `ruff format --check` (both files) | clean (one assert restructure auto-applied by `ruff format`, then re-verified) |
| `ruff check` (both files) | clean — "All checks passed!" |
| `mypy dnallm/utils/transformers_compat.py` | advisory: stock invocation dies on a pre-existing numpy-stub syntax error before reaching the file (identical on untouched `dnallm/utils/logger.py`; CI runs mypy `|| true`); with `--python-version 3.13` the module checks with **0 findings in transformers_compat.py** (21 pre-existing errors in 4 unrelated modules) |
| `git show --stat HEAD` | exactly `dnallm/utils/transformers_compat.py` (+111) and `tests/utils/test_transformers_compat.py` (+147) |
| Push state | `phs` ahead of `origin/phs`; `fdc4915` unpushed |
| Out-of-scope surface | none — no notebook edits, nothing under `~/.cache/huggingface/` touched (import-only commands), working-tree residue identical to the pre-task snapshot |

## Deviations from Plan

None material. One cosmetic note: ruff format restructured one new assert from
`assert (expr), "msg"` to `assert expr, ("msg")` style — applied via `ruff format` and
all gates re-run green afterwards. The mypy/numpy-stub failure is a pre-existing
environment condition (documented above), not introduced by this change.

## Notes for the Milestone

- The benchmark notebook's current failure rung (remote EsmModel forward AttributeError)
  is closed; later unrelated notebook failures (census KeyError etc.) remain explicitly
  out of scope per the plan.
- Transformers-span safety holds by construction: absence gate + invert-never-skip
  identity tests, so the 4.49-5.x CI matrix exercises both sides.

## Self-Check: PASSED

- Commit `fdc4915` exists on `phs` (`git log --oneline --all | grep fdc4915`).
- Both files exist and carry the changes (`git show --stat HEAD` lists exactly them).
- SUMMARY not committed (orchestrator handles docs artifacts); ROADMAP.md untouched.
