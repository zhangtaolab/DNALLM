---
phase: quick-261024-dtw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/inference/inference.py
  - tests/inference/test_inference.py
  - tests/inference/test_mamba_mps_smoke.py
  - tests/expected_skips.yaml
  - docs/faq/models_troubleshooting.md
autonomous: true
requirements: [DTW-01]
estimate:
  tokens: 55000
  raw_tokens: 35000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "DNAInference constructed with a mamba-family model keeps device.type 'cuda' and 'mps' (MPS no longer demoted); every other non-cuda device type (e.g. xla from device='tpu') still demotes to cpu exactly as before"
    - "Mamba never gets fp16 through the Predictor on ANY device — pred_config.use_fp16 is forced False on cuda, mps, and cpu alike (conservative behavior unchanged)"
    - "The mock suite in tests/inference/test_inference.py pins the new semantics: an MPS-passthrough test (mps kept, fp16 off), a cuda-keeps test, and a demote test using a non-cuda non-mps device (tpu/xla) so the demote branch stays covered"
    - "On darwin with MPS available, the real-device smoke loads zhangtaolab/plant-dnamamba-BPE (ModelScope-first, HF retry) through DNAInference with device 'mps' and completes a real forward on MPS"
    - "The new smoke test cannot shift the CI census red: it is slow-marked (deselected on '-m \"not slow\"' fast legs) and its static non-darwin skipif reason is registered as a reason_like entry in tests/expected_skips.yaml; darwin-side runtime skips reuse the already-registered environment-unavailable: prefix"
    - "docs/faq/models_troubleshooting.md no longer claims the macOS transformers Mamba path is restricted to CPU; it records the measured MPS numbers (349ms vs 2119ms for a 512-token forward, ~6.1x, bitwise-identical, torch 2.14.1, 2026-10-24), the torch>=2.12 MPS requirement (above the declared ceiling — documented local-deviation), and the 2026-10-24 owner ruling accepting the transformers-official implementation incl. its mambapy pscan gate"
  artifacts:
    - dnallm/inference/inference.py
    - tests/inference/test_inference.py
    - tests/inference/test_mamba_mps_smoke.py
    - tests/expected_skips.yaml
    - docs/faq/models_troubleshooting.md
  key_links:
    - "inference.py guard condition ('cuda'/'mps' keep, else cpu demote) <-> the three MambaLikeModel mock tests in TestConstructionBranches"
    - "the skipif reason string in test_mamba_mps_smoke.py <-> the new reason_like entry in tests/expected_skips.yaml (exact substring match)"
    - "the real-device smoke's load <-> the modelscope-first + single HF-retry fallback contract with _is_environment_error classification (dnallm regressions propagate, never typed-skip)"
---

<objective>
Relax the stale mamba device guard in the Predictor (dnallm/inference/inference.py) so Apple MPS passes through instead of being demoted to CPU, with same-change mock tests, a darwin-gated real-device MPS smoke test under the repo's typed-skip discipline, and the stale macOS-Mamba docs claim corrected.

Purpose: the guard predates MPS maturity (commit e0b3494); measured 2026-10-24 the transformers-official Mamba path runs green on MPS with torch 2.14.1 (512-token forward 349ms MPS vs 2119ms CPU, ~6.1x, bitwise-identical outputs), so the forced CPU demotion now only costs Mac users a 6x slowdown. The coupled fp16 force-off stays (fp16 on MPS is unverified — conservative keep).

Output: relaxed guard; updated + new mock tests; new tests/inference/test_mamba_mps_smoke.py with skip registration; corrected docs/faq/models_troubleshooting.md; full mamba-related test surface green on this Mac.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

- dnallm/inference/inference.py:118-198 (constructor: device resolution at 185, model .to() at 186-190, the mamba guard at 191-196)
- tests/inference/test_inference.py:743-764 (MambaLikeModel double), 1621-1697 (TestDeviceSelection), 1700-1721 (TestConstructionBranches mamba tests)
- tests/conftest.py:186-226 (inference_config_factory fixture — real YAML-roundtrip configs, available to tests/inference/)
- tests/models/test_plant_helixseek_smoke.py (the _load_with_fallback + _is_environment_error pattern to mirror; module-level importable helpers)
- tests/expected_skips.yaml (typed-skip allowlist; environment-unavailable: prefix already registered at line 44; reason_like precedent entries: "SONAME check is Linux-specific", "fd accounting needs /proc")
- Background facts (verified 2026-10-24, do not re-derive): the load_model_and_tokenizer route has NO such guard and MPS already works there; only the Predictor path demotes. The plant-dnamamba remote code is a vendored copy of transformers' official modeling_mamba; the checkpoints pin use_mambapy=true and mambapy is a base dep, so the pscan gate is satisfied on all platforms (owner ruling 2026-10-24: accepted path as-is, no checkpoint re-export).
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Relax the mamba device guard in DNAInference to allow MPS (with same-change mock tests)</name>
  <files>dnallm/inference/inference.py, tests/inference/test_inference.py</files>
  <behavior>
    - Test (mps passthrough): with torch.backends.mps.is_available monkeypatched to True (pattern of test_apple_alias_mps_unavailable at tests/inference/test_inference.py:1652), config device="mps" use_fp16=True, a MambaLikeModel engine resolves device == torch.device("mps") and pred_config.use_fp16 is False
    - Test (demote branch still covered): config device="tpu" use_fp16=True (device_map maps tpu to xla with an always-True availability lambda — no patch and no warning context needed) resolves device == torch.device("cpu") and pred_config.use_fp16 is False
    - Test (cuda unchanged): existing test_mamba_cuda_still_forces_fp32 keeps passing verbatim — cuda device kept, fp16 forced off
  </behavior>
  <action>
    In dnallm/inference/inference.py, change ONLY the mamba guard at lines 191-196: the inner condition `if self.device.type != "cuda":` becomes a not-in membership test allowing both "cuda" and "mps" to keep their device, with every other device type still demoted via `self.device = torch.device("cpu")` exactly as today. The coupled `use_fp16 → False` enforcement stays byte-for-byte unchanged. Update the stale comment on line 191 ("mamba only support cuda and cpu, and only allow fp32") to state the new semantics: mamba runs fp32 on cuda and mps (MPS parity measured 2026-10-24, torch 2.14.1); other devices demote to cpu; fp16 stays conservatively off because it is unverified on MPS. Do NOT touch anything else in the constructor: the model .to(self.device) at lines 186-190 runs before the guard and the demotion does not re-move the model — that pre-existing shape is intentionally preserved (out of scope; mocks record to_called_with and never relied on a re-move).

    Same change in tests/inference/test_inference.py (the class at line 743 is named MambaLikeModel — planning context called it _MambaModelDouble, the live name is MambaLikeModel): (a) update the MambaLikeModel docstring (line 744, "fp32/CPU downgrade branch") to the new cuda/mps/demote semantics; (b) replace test_mamba_downgraded_on_cpu (line 1703) — its current shape (cuda unavailable → resolves cpu → stays cpu) no longer exercises the demote branch — with a test using device="tpu" that asserts the xla-resolved device demotes to cpu and fp16 is forced off (rename to reflect the non-cuda non-mps demote, e.g. test_mamba_non_cuda_non_mps_demoted_to_cpu); (c) leave test_mamba_cuda_still_forces_fp32 (line 1714) and test_cuda_available_selected (line 1635, also uses MambaLikeModel) untouched — they already pin cuda-keeps; (d) add the NEW mps-passthrough test to TestConstructionBranches per the behavior block, monkeypatching torch.backends.mps.is_available to lambda: True and asserting device == torch.device("mps"), pred_config.use_fp16 is False, and MambaLikeModel's recorded to_called_with equals the mps device.
  </action>
  <verify>
    <automated>.venv/bin/pytest tests/inference/test_inference.py -q && .venv/bin/ruff format --check dnallm/inference/inference.py tests/inference/test_inference.py && .venv/bin/ruff check dnallm/inference/inference.py tests/inference/test_inference.py</automated>
  </verify>
  <done>Guard keeps cuda and mps, demotes every other device type to cpu, and still forces use_fp16 False for mamba on all devices; the three mock behaviors (mps kept, tpu/xla demoted, cuda kept) pass; the whole tests/inference/test_inference.py file stays green; ruff format and lint clean on both files.</done>
</task>

<task type="auto">
  <name>Task 2: Real-device MPS smoke test for mamba through the Predictor (typed-skip discipline + census-safe registration)</name>
  <files>tests/inference/test_mamba_mps_smoke.py, tests/expected_skips.yaml</files>
  <precondition>The zhangtaolab/plant-dnamamba-BPE checkpoint is cached under ~/.cache/modelscope on this machine and the venv runs torch 2.14.1 + transformers 5.20.2 (verified 2026-10-24); assert before running — if the cache is absent the load falls back to HF and finally typed-skips rather than failing.</precondition>
  <action>
    Create tests/inference/test_mamba_mps_smoke.py modeled on tests/models/test_plant_helixseek_smoke.py (read it first — mirror its module docstring contract, evidence emission, and fallback structure): a single test function test_plant_dnamamba_mps_predictor_keeps_device, decorated @pytest.mark.slow and @pytest.mark.timeout(900), plus a static gate @pytest.mark.skipif(sys.platform != "darwin", reason="MPS mamba smoke is darwin-only (Apple Silicon real device)"). Inside: (1) runtime gate — if not torch.backends.mps.is_available(), pytest.skip with an "environment-unavailable:" prefixed reason carrying torch and transformers versions (that prefix is already registered in tests/expected_skips.yaml:44 — do not invent a new prefix for this); (2) load zhangtaolab/plant-dnamamba-BPE with TaskConfig(task_type="generation") via load_model_and_tokenizer with source="modelscope" first and a single source="huggingface" retry — import _is_environment_error from tests.models.test_plant_helixseek_smoke (cross-test-module import precedent: that module itself imports from tests.models.test_plant_helixseek_registry) so environment-class failures on both routes skip with an "environment-unavailable:" evidence message while a dnallm-side regression propagates and fails; (3) build the Predictor config with the inference_config_factory fixture from tests/conftest.py using task_type="generation", device="mps", use_fp16=True, max_length=512 (use_fp16=True deliberately, to pin the conservative force-off on real hardware); (4) construct DNAInference(model=..., tokenizer=..., config=...) and assert engine.device.type == "mps" (the relaxed guard — this is the regression the test exists to pin), engine.pred_config.use_fp16 is False, and next(engine.model.parameters()).device.type == "mps"; (5) one real forward through the engine via engine.infer_seqs(["ACGT" * 32], do_pred=False) and assert the call completes with a non-empty return (if generation-task prediction semantics make do_pred=True the natural form, use it — the assertion target is the completed MPS forward, not the prediction format; consult batch_infer at dnallm/inference/inference.py:974 to pick the form and assert accordingly).

    Register the static gate in tests/expected_skips.yaml as a new reason_like entry, category environment, with a comment citing the test file and the lanes where it can fire (non-darwin collection — the example-nightly full census runs without the not-slow filter; CI fast legs deselect slow and never collect it), placed beside the existing "SONAME check is Linux-specific" / "fd accounting needs /proc" platform-gate precedents. The reason_like substring must appear verbatim inside the skipif reason string in the test source.
  </action>
  <verify>
    <automated>.venv/bin/pytest tests/inference/test_mamba_mps_smoke.py -v --timeout=900 && grep -q "MPS mamba smoke is darwin-only" tests/inference/test_mamba_mps_smoke.py && grep -q "MPS mamba smoke is darwin-only" tests/expected_skips.yaml && .venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from audit_skips import load_allowlist; load_allowlist('tests/expected_skips.yaml'); print('allowlist well-formed')" && .venv/bin/ruff format --check tests/inference/test_mamba_mps_smoke.py && .venv/bin/ruff check tests/inference/test_mamba_mps_smoke.py</automated>
  </verify>
  <done>On this Mac (darwin, MPS available, cached checkpoint) the smoke runs for real: plant-dnamamba-BPE loads ModelScope-first, DNAInference keeps device.type "mps", fp16 is forced off, and a forward completes through the Predictor on MPS. The skipif reason string matches the registered reason_like entry in tests/expected_skips.yaml; the allowlist stays well-formed under scripts/audit_skips.py's loader; the file is slow-marked so CI fast legs never collect it.</done>
</task>

<task type="auto">
  <name>Task 3: Fix the stale macOS Mamba docs claim in models_troubleshooting.md</name>
  <files>docs/faq/models_troubleshooting.md</files>
  <action>
    Rewrite the "Mamba models on macOS (Apple Silicon)" section (docs/faq/models_troubleshooting.md lines 15-26). Keep what is still true: mamba-ssm depends on Triton, which ships Linux-only wheels, so the [mamba] extra cannot be installed on macOS, and the transformers path remains slower than the mamba-ssm CUDA kernels (Linux + CUDA). Replace the stale CPU-restriction parenthetical at line 26 and the surrounding "fallback is CPU-bound" framing with the measured facts: the HuggingFace transformers pure-PyTorch Mamba implementation (MambaForCausalLM) is the accepted official path on macOS per the 2026-10-24 owner ruling — including its mambapy pscan gate, which the base mambapy dependency satisfies on every platform, so no checkpoint re-export is needed; it runs on CPU and on MPS (Apple Silicon), with the Predictor keeping device mps automatically and always forcing fp32 (fp16 on MPS is unverified and stays conservatively off); measured 2026-10-24 on torch 2.14.1, a 512-token forward runs 349 ms on MPS vs 2119 ms on CPU (~6.1x) with bitwise-identical outputs. State the requirement honestly: MPS for this path needs torch &gt;= 2.12, which sits above dnallm's currently declared torch ceiling, so MPS means a locally raised torch (a documented local deviation) — within the declared dependency range the path runs on CPU. Keep the install snippet (uv pip install -e '.[full]') and the section's heading style. Do not touch any other section of the file.
  </action>
  <verify>
    <automated>! grep -q "cpu only" docs/faq/models_troubleshooting.md && grep -q "torch >= 2.12\|torch>=2.12" docs/faq/models_troubleshooting.md && grep -q "6.1" docs/faq/models_troubleshooting.md</automated>
  </verify>
  <done>The section no longer claims the macOS transformers Mamba path is restricted to CPU (the phrase grep'd above is gone from this file — baseline: it exists at line 26 today); the measured MPS numbers, the torch 2.12 requirement with the declared-ceiling caveat, the fp32 force-off, and the owner-ruling acceptance of the transformers-official path (incl. the mambapy gate) are all stated; no other section changed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| hub/cache → process | The smoke test loads third-party-sourced model weights + remote code (zhangtaolab/plant-dnamamba-BPE) which execute inside the pytest process |
| docs → user | The troubleshooting doc tells users which device/path to trust for results |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261024-dtw-01 | Tampering / Elevation | tests/inference/test_mamba_mps_smoke.py (remote-code model load) | medium | accept | Established repo pattern: owner-org zhangtaolab lab-models-first sourcing decision, ModelScope-first with single HF retry, load confined to a slow-marked test; no new install or untrusted-package step is introduced |
| T-261024-dtw-02 | DoS | tests/inference/test_mamba_mps_smoke.py (real-device forward) | low | mitigate | @pytest.mark.timeout(900) bounds a hung MPS forward; slow mark keeps CI legs clear |
| T-261024-dtw-03 | Repudiation | docs/faq/models_troubleshooting.md | low | mitigate | Measured claims carry date + torch version (2026-10-24, torch 2.14.1) and cite the owner ruling, so the performance claim is auditable against STATE.md session continuity |

No package-manager installs occur in this plan, so no T-SC row applies.
</threat_model>

<verification>
Full mamba-related surface on this Mac (cwd repo root, venv torch 2.14.1 / transformers 5.20.2 / py3.13; expect all green, no new skips on darwin):

    .venv/bin/pytest tests/inference/test_inference.py tests/models/test_model.py tests/utils/test_transformers_compat.py tests/configuration/test_peft_presets.py tests/models/test_model_info_registry.py tests/mcp/test_interpret_tool.py tests/inference/test_mamba_mps_smoke.py -q

Plus repo gates on every changed file:

    .venv/bin/ruff format --check dnallm/inference/inference.py tests/inference/test_inference.py tests/inference/test_mamba_mps_smoke.py && .venv/bin/ruff check dnallm/inference/inference.py tests/inference/test_inference.py tests/inference/test_mamba_mps_smoke.py

Census neutrality proof for CI (linux + windows): the only new collected-on-non-darwin skip is the static skipif; confirm the registered reason_like substring matches the skipif reason string exactly (the Task 2 grep pair), and confirm the new test is slow-marked so `-m "not slow"` fast legs (ci.yml:111/185/418) deselect it.
</verification>

<success_criteria>
- The Predictor keeps mps for mamba models, keeps cuda, and demotes only other device types — pinned by three mock tests, not just the implementation
- fp16 force-off for mamba is provably unchanged on every device
- A real plant-dnamamba-BPE forward through DNAInference on MPS passes on this machine (the load itself was already measured green via the no-guard route; this test pins the Predictor route)
- CI fast legs cannot change color: no new collected tests on non-darwin except the registered typed skip; audit_skips allowlist stays well-formed
- The docs state the measured truth with dates, versions, and the ceiling caveat
- All commits land on branch v1.4 with no Co-Authored-By trailers; every dnallm/ change shipped with its pytest in the same change (Task 1)
</success_criteria>

<output>
Create .planning/quick/261024-dtw-relax-mamba-device-guard-in-predictor-to/261024-dtw-SUMMARY.md when done
</output>
