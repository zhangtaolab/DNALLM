---
phase: quick-261024-dtw
plan: 01
subsystem: inference
tags: [mamba, mps, predictor, device-guard, smoke-test, docs]
status: complete
requires:
  - "torch 2.14.1 / transformers 5.20.2 local venv (documented deviation above the declared torch ceiling)"
  - "zhangtaolab/plant-dnamamba-BPE checkpoint cached in ~/.cache/modelscope"
provides:
  - "Predictor route keeps device mps for mamba-family models (cuda/mps membership guard)"
  - "tests/inference/test_mamba_mps_smoke.py — darwin-gated real-device MPS smoke through DNAInference"
  - "reason_like census entry 'MPS mamba smoke is darwin-only' in tests/expected_skips.yaml"
affects: []
tech-stack:
  added: []
  patterns:
    - "cross-test-module import of _is_environment_error from tests.models.test_plant_helixseek_smoke (classifier shared, cannot drift)"
key-files:
  created:
    - tests/inference/test_mamba_mps_smoke.py
  modified:
    - dnallm/inference/inference.py
    - tests/inference/test_inference.py
    - tests/expected_skips.yaml
    - docs/faq/models_troubleshooting.md
decisions:
  - "Guard is a not-in membership test — device.type not in (\"cuda\", \"mps\") keeps the device; every other type demotes to cpu; fp16 force-off byte-identical (unverified on MPS, conservative keep)"
  - "Smoke forward uses batch_infer(dataloader, do_pred=False), not infer_seqs: generation task_type makes logits_to_preds raise (unsupported task type) so do_pred=True is not viable, and infer_seqs(do_pred=False) returns None; batch_infer is the forward half of infer_seqs and returns non-empty logits (plan pre-authorized the choice via 'consult batch_infer to pick the form')"
  - "MPS pin in the smoke lives on next(engine.model.parameters()).device.type, not on logits — _process_batch_outputs moves logits to cpu (inference.py:791)"
metrics:
  duration: 12 min
  completed: 2026-10-24
actuals:
  tokens: 3058   # chars/4 over the realized diff (12230 chars)
  tasks: 3
  commits: 3     # measured: git rev-list --count 247462b..HEAD
plan_head_before: 247462be9017e6057f06829ed7e830d5287a60bd
plan_head_after: 9952ddca0c8c8f75c4951a41b1b5b32de5af1711
---

# Quick Task 261024-dtw: Relax mamba Predictor device guard for MPS Summary

**One-liner:** Predictor keeps MPS for mamba models (cuda/mps membership guard, fp16 force-off unchanged), pinned by three mock tests plus a real-device plant-dnamamba-BPE MPS smoke with census-safe skip registration, and the stale macOS CPU-only docs claim replaced with the measured truth.

## What Was Done

### Task 1 — Guard relaxation + mock tests (18cd32a, TDD)

- `dnallm/inference/inference.py` mamba guard: `if self.device.type != "cuda":` → `if self.device.type not in ("cuda", "mps"):`; cuda and mps now keep their device, every other type (e.g. xla from device="tpu") still demotes via `self.device = torch.device("cpu")`. The coupled `use_fp16 → False` enforcement is byte-identical. Comment rewritten to the new semantics (MPS parity measured 2026-10-24, torch 2.14.1; fp16 conservatively off). Nothing else in the constructor touched (the pre-guard `.to(self.device)` shape intentionally preserved).
- `tests/inference/test_inference.py` (TDD RED → GREEN):
  - NEW `test_mamba_mps_kept_and_fp16_forced_off` — monkeypatches `torch.backends.mps.is_available` to True, device="mps" use_fp16=True; asserts device == mps, use_fp16 False, `to_called_with` == mps. RED against the old guard (demoted to cpu), GREEN after.
  - REPLACED `test_mamba_downgraded_on_cpu` (its cuda-unavailable→cpu shape no longer exercises the demote branch) with `test_mamba_non_cuda_non_mps_demoted_to_cpu` — device="tpu" (xla, always-True availability, no patch/warning needed) asserts cpu demote + fp16 off.
  - `test_mamba_cuda_still_forces_fp32` and `test_cuda_available_selected` untouched (still pin cuda-keeps).
  - `MambaLikeModel` docstring updated to the cuda/mps/demote semantics.
- Whole file green (124 passed); ruff format + check clean on both files.

### Task 2 — Real-device MPS smoke (60715d7)

- `tests/inference/test_mamba_mps_smoke.py`: `test_plant_dnamamba_mps_predictor_keeps_device`, `@pytest.mark.slow` + `@pytest.mark.timeout(900)` (T-02 mitigation) + static `skipif(sys.platform != "darwin", reason="MPS mamba smoke is darwin-only (Apple Silicon real device)")`. Runtime gate: MPS-unavailable darwin hosts typed-skip with the already-registered `environment-unavailable:` prefix carrying torch/transformers versions. Load: ModelScope-first with a single HF retry, environment classification via `_is_environment_error` imported from `tests.models.test_plant_helixseek_smoke` (dnallm regressions propagate). Predictor config via `inference_config_factory(task_type="generation", device="mps", use_fp16=True, max_length=512)` — use_fp16=True deliberately pins the force-off on real hardware. Asserts `engine.device.type == "mps"` (the regression pin), `use_fp16 is False`, params on MPS, and a completed forward with non-empty logits via `batch_infer(dataloader, do_pred=False)`.
- `tests/expected_skips.yaml`: new `reason_like: "MPS mamba smoke is darwin-only"` entry (category environment) beside the SONAME//proc platform-gate precedents, with the lane commentary (example-nightly full census collects it; CI fast legs deselect slow).
- Verified on this Mac: PASSED for real in 3.66 s (offline-fast ModelScope cache hit, real MPS forward). Allowlist well-formed under `scripts/audit_skips.py`'s loader; grep pair matches.

### Task 3 — Docs truth fix (9952ddc)

- `docs/faq/models_troubleshooting.md` macOS Mamba section: dropped the "(cpu only)" claim and the CPU-bound fallback framing. Now states: `MambaForCausalLM` is the accepted official mac path per the 2026-10-24 owner ruling (incl. the mambapy pscan gate, satisfied by the base dep — no checkpoint re-export); runs on CPU and MPS with the Predictor keeping mps and always forcing fp32; measured 349 ms MPS vs 2119 ms CPU (~6.1x, bitwise-identical, torch 2.14.1, 2026-10-24); honest caveat that MPS needs torch >= 2.12 (above the declared ceiling — documented local deviation) so within the declared range the path runs on CPU; still slower than mamba-ssm CUDA kernels. Install snippet and heading untouched; no other section changed.

## Plan Verification Results

- Full mamba-related surface on this Mac: **509 passed** in 114 s (`tests/inference/test_inference.py`, `tests/models/test_model.py`, `tests/utils/test_transformers_compat.py`, `tests/configuration/test_peft_presets.py`, `tests/models/test_model_info_registry.py`, `tests/mcp/test_interpret_tool.py`, `tests/inference/test_mamba_mps_smoke.py`).
- Repo gates: ruff format --check + ruff check clean on all three Python files.
- Census neutrality proven: `-m "not slow"` deselects the smoke (1 deselected, 0 collected); unfiltered collection collects 1; the registered reason_like substring appears verbatim inside the skipif reason string; darwin runtime skips reuse the registered `environment-unavailable:` prefix.
- fp16 force-off provably unchanged: forced False on cuda (existing test), mps (new mock test + real-hardware smoke), cpu/other (tpu demote test).

## Deviations from Plan

None — Rules 1-4 never triggered. Two plan-delegated/normal-compliance choices, documented for the record:

1. Smoke forward form: the plan's primary `engine.infer_seqs([...], do_pred=False)` returns None for generation tasks and `do_pred=True` raises (`logits_to_preds`: "Unsupported task type: generation"); the plan's parenthetical explicitly delegated the form choice to `batch_infer` — used `batch_infer(dataloader, do_pred=False)` (the forward half of infer_seqs) so the non-empty-return assertion targets real logits.
2. Lint compliance: ruff format collapsed one wrapped call onto a single (100-char) line and the repo's pytest-composite-assertion rule required splitting one composite assert into two — no semantic change.

## Authentication Gates

None.

## Known Stubs

None — no placeholder patterns in any changed file (grep hits were device-unavailability warning texts and the registered skip prefix, not stubs).

## Threat Flags

None — no security-relevant surface beyond the plan's threat model. Dispositions applied as planned: T-01 accepted (owner-org ModelScope-first sourcing, confined to a slow-marked test), T-02 mitigated (`@pytest.mark.timeout(900)` + slow mark), T-03 mitigated (docs carry date + torch version + owner-ruling citation).

## Self-Check: PASSED

All 5 changed files + the SUMMARY exist on disk; all 3 task commits (18cd32a, 60715d7, 9952ddc) are ancestors of HEAD (9952ddc). Working tree carries only the uncommitted docs artifact (this SUMMARY), per the orchestrator-handles-docs-commit constraint.
