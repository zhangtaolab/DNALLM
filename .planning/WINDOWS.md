---
schema_version: 1
open_count: 7
waived_count: 3
fixed_count: 11
total_count: 21
last_updated: 2026-10-10T15:58:00.000Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | .planning/phases/01-harness-integrity-measured-baseline/01-01-PLAN.md |  | Task 2 verify grep 'tasks/metrics' false-positives on measured dispatcher dnallm/tasks/metrics.py; boundary re-proved with precise patterns (vendored dir absent, neighbors present) | fixed |  | 2026-09-29T17:37:20.663Z | 2026-10-01T14:32:27.252Z |
| 2 | 3 | unmet-truth | dnallm/inference/inference.py | 1643 | generate-from-DataLoader never appends to prompt_seqs (seqs.extend on itself); causallm generate over a DataLoader returns empty list | waived | Latent bug pinned by tests, out of v1 scope (milestone closed 2026-10-01): generate-from-DataLoader returns empty list (inference.py:1643). Deferred to next-milestone backlog — recorded in v1-MILESTONE-AUDIT.md tech-debt ledger | 2026-09-30T09:46:44.754Z | 2026-10-01T14:32:27.857Z |
| 3 | 3 | unmet-truth | dnallm/inference/mutagenesis.py | 429 | evaluate strategy max calls raw_score.index() on an ndarray (AttributeError) — latent bug documented as accepted residual | waived | Latent bug pinned by tests, out of v1 scope (milestone closed 2026-10-01): mutagenesis 'max' strategy calls ndarray.index() (mutagenesis.py:429) AttributeError. Deferred to next-milestone backlog — v1-MILESTONE-AUDIT.md tech-debt ledger | 2026-09-30T09:46:44.856Z | 2026-10-01T14:32:27.943Z |
| 4 | 03 | stub | dnallm/models/model.py | 264 | cosine_similarity loss_function constructs CosineEmbeddingLoss but calls it with (logits, labels) — missing target arg raises TypeError for every user selecting it (covered by test_forward_cosine_similarity_loss_crashes; fix deferred, semantics ambiguous) | waived | Latent bug pinned by test_forward_cosine_similarity_loss_crashes, out of v1 scope (milestone closed 2026-10-01): cosine_similarity loss TypeError, semantics ambiguous (model.py:264). Deferred to next-milestone backlog — v1-MILESTONE-AUDIT.md tech-debt ledger | 2026-09-30T10:39:17.694Z | 2026-10-01T14:32:28.031Z |
| 5 | 03 | deviation | .planning/phases/03-test-authoring-to-90-coverage/03-03-PLAN.md |  | Wave-3 verify gate 'assert not logs.exists()' is unsatisfiable: dnallm's import-time file sink (utils/logger.py:57-60) creates logs/dnallm.log at the pytest launch cwd on every suite run — waves 4-5 plans must gate on 'no logs/mcp_server.log at repo root' instead (sink also recorded in deferred-items.md) | fixed |  | 2026-09-30T11:33:06.961Z | 2026-10-01T14:32:27.339Z |
| 6 | 3 | deviation | tests/datahandling/test_dna_dataset.py |  | pytest 9.1.1 --collect-only emits no :: separators - the plan's grep -c :: tripwires were enforced as the equivalent 'N tests collected' counts (66/151 >= 35/55; trainer 38 >= 10), as in waves 1-3 | fixed |  | 2026-09-30T12:22:58.724Z | 2026-10-01T14:32:27.426Z |
| 7 | 04 | deviation | pyproject.toml |  | Plan 04-01 synthetic-drop verify as written (--ignore=tests/models/test_model.py) cannot go red: 91.65% under -m 'not slow'; corrected proof and 04-03 GATE-04 probe must ignore/delete the whole tests/models dir | fixed |  | 2026-09-30T16:32:51.841Z | 2026-10-01T14:32:27.511Z |
| 8 | 04 | deviation | .github/workflows/ci.yml |  | Plan 04-02 verify commands needed --workflow CI disambiguation (Docs Validation run stole the latest-push-run slot) and mid-run job logs are 404 on GitHub's API until completion - step-state is the runtime health proof | fixed |  | 2026-09-30T17:03:40.700Z | 2026-10-01T14:32:27.597Z |
| 9 | 04 | deviation | .planning/phases/04-ci-gate-enforcement/04-03-PLAN.md |  | Plan arithmetic defect: single-file tests/models/test_model.py deletion cannot clear the 90 floor under -m 'not slow' (91.64% green); probe target re-planned to directory deletion (78.92% red local, 78.91% CI) per the plan own rehearsal gate — resolved in 04-03 | fixed |  | 2026-09-30T18:17:11.661Z | 2026-10-01T14:32:27.683Z |
| 10 | 04 | deviation | .planning/phases/04-ci-gate-enforcement/04-03-PLAN.md |  | Verify mechanics: gh run view --job --log-failed gates on whole-run completion; evidence harvested via job-level logs API (gh api actions/jobs/<id>/logs) which serves completed jobs mid-run — resolved in 04-03 | fixed |  | 2026-09-30T18:17:11.751Z | 2026-10-01T14:32:27.771Z |
| 11 | 05 | deviation | example/notebooks/benchmark/benchmark.ipynb |  | Census FAIL row (05-04, D-07 ladder terminal): third registry model zhangtaolab/nucleotide-transformer-v2-100m-promoter not loadable on transformers 5.17 - remote code needs removed 4.x PretrainedConfig defaults (is_decoder/add_cross_attention); native-ESM route refuted (FFN shape mismatch); owner disposition pending per D-09 | open |  | 2026-10-02T05:12:38.864Z |  |
| 12 | 05 | stub | tests/examples/test_script_execution.py |  | environment-unavailable typed skip: generate_bpe_dataset.py pkl-production leg blocked by GAP-1-class remote-code gap (plant-nucleotide-transformer-BPE needs removed 4.x PretrainedConfig defaults on transformers 5.17); self-healing, owner disposition pending | open |  | 2026-10-02T05:48:40.008Z |  |
| 13 | 05 | unrun-verify | example/mcp_example |  | Census deferred-owner rows (05-06, D-08/T-05-16): both ollama mcp client notebooks never executed - ollama probe GREEN (qwen3.8:latest) but dnallm MCP server endpoint down and uv pip install cells must never touch the project venv; needs the Phase-8 ollama/VRAM coexistence plan (owner decision); durable gated tests skip network-unavailable with live probe evidence and fail loudly if both endpoints come up | open |  | 2026-10-02T08:34:21.484Z |  |
| 14 | 05 | deviation | dnallm/inference/benchmark.py | 296 | Census FAIL finding (05-06): Benchmark.run hardcodes self.datasets[di]['labels'] while example benchmark_config.yaml declares label_column 'label' - KeyError before any model loads; blocks the benchmark notebook (and its NT third-model disposition) until repaired; Phase 8 repair queue | fixed | run() label resolution honors configured label_column with 'labels' fallback and descriptive ValueError; red-first regression tests in TestRunLabelColumnResolution; note: census's ['sequence']-only shape stemmed from generate_dataset's silent path-to-sequence fallback, a separate latent bug deliberately left out of scope | 2026-10-02T08:34:21.569Z | 2026-10-10T14:39:32.164Z |
| 15 | 05 | skipped-test | tests/examples/test_notebook_execution.py |  | 05-06 gated typed skips (sanctioned, self-healing): optional-dep probe-then-execute for evo/megaDNA prerequisites, finetune_custom_head megaDNA demo cell and PlantCAD lora pair (mamba_ssm); environment-unavailable script-lane skip (05-05 pattern) unchanged; all matched by audit_skips against registered prefixes | open |  | 2026-10-02T08:34:21.655Z |  |
| 16 | 09 | unmet-truth | tests/benchmark/test_benchmark.py |  | Pre-existing fast-lane failure (found by 09-02 verify, reproduced at plan-start 3557e0b): TestBenchmark::test_plot_for_regression pandas TypeError float() argument ... not dict via _astype_nansafe in the plot path; not caused by any Phase 09 change (09-02 delta +8P/+0F/+0S); likely quick-task 13/14 Mapping fallout; logged in 09 deferred-items.md | fixed |  | 2026-10-05T15:45:36.353Z | 2026-10-06T01:31:42.584Z |
| 17 | quick-261007-vxx | unmet-truth | docs/user_guide/fine_tuning/getting_started.md |  | Push of ruff-format fix commit 97a7c30 to origin/dev blocked by GitHub receive-side Internal Server Error (4 attempts, Request IDs 8832:3513C8/C942:3774A8/B48E:2E4A69/991C:246D9C, 2026-10-07 15:07-15:12Z); local ruff format --check green at dev 97a7c30; re-run 'git push origin dev' when GitHub receive recovers to unblock ci.yml format gates + PR #40 | fixed | GitHub receive recovered; orchestrator re-push at 2026-10-07T15:17:44Z landed cfc8346..97a7c30 on origin/dev; CI + Docs Validation re-triggered | 2026-10-07T15:20:00.000Z | 2026-10-07T15:18:00.000Z |
| 18 | quick-261008-env | deviation | pyproject.toml |  | Windows fla installability gap (env adaptation 2026-10-08): pip install -e '.[fla]' on Windows succeeds but leaves fla broken - torch Windows wheels declare no triton dependency (Linux pulls it) and fla-core requires triton only under its [cuda]/[cpu] extras which the pyproject deliberately avoids, so fla imports top-level but fla.ops.* raises ModuleNotFoundError and test_chunk_kda_importable_when_fla_installed FAILS instead of skipping. Remedy installed in dnallm-cuda: triton-windows==3.7.1.post27 (community wheels; upstream triton ships no win_amd64) - KDA chunk kernel proven compiling+executing on RTX 5080 sm_120. Open owner decision: whether to declare triton-windows behind a Windows platform marker in the fla extra (third-party fork enters declared deps) or document the manual step | open |  | 2026-10-08T04:30:00.000Z |  |
| 19 | quick-261008-wfx | deviation | tests/examples/test_plant_helixseek_showcase.py |  | CRE showcase execution test requires the bedtools CLI and fails loud by design (test line ~480 asserts shutil.which('bedtools')); bedtools has no Windows build, so local Windows full-example runs must --deselect test_cre_notebook_executes_within_selection_bands (workaround in daily use since 2026-10-08). Open question: WSL-scope bedtools provisioning or permanent documented deselect; Linux nightly lanes unaffected. EVALUATION 2026-10-10 (owner-directed de-bedtools assessment): pyranges1 REJECTED as replacement — ruranges 0.2.7 ships no win_amd64 wheels (Windows would need Rust msvc source build; the pybedtools platform marker would simply transfer), requires-python >=3.12, and it buys zero Windows gain since CRE jaccard is subprocess-direct CLI anyway (join_overlaps left mode does map to the NER loj intersect, noted for completeness). CHOSEN DIRECTION (refined by owner 2026-10-10, platform-split): Linux/macOS KEEP bedtools/pybedtools — the frozen truth chain stays bedtools-computed, no re-freeze; Windows gets a parity-verified numpy evaluator (precedent dnallm/utils/genomic_coords.py) used only where bedtools cannot install — the CRE jaccard first, the NER example's pybedtools loj intersect second (larger surface: join semantics + ascending-order traversal). OWNER DIRECTIVE 2026-10-10: the NER example CODE ITSELF must change — generate_bpe_dataset.py and data_generation_and_inference.ipynb gain runtime platform detection (Linux keeps pybedtools.BedTool.intersect(loj=True), Windows dispatches to the numpy equivalent); the downstream frozen Anno metric (exon_f1=0.7522) must hold in band on BOTH paths, and the loj replacement needs its OWN parity check (numpy join vs `bedtools intersect -loj` — jaccard parity does not transfer to join semantics). Enabling evidence: quick-task spike 261010-wx2 (numpy vs bedtools v2.31.1 parity on the frozen selection pair jaccard=0.324727 + randomized property cases) — without parity the split would be two truths, so the split ships only if parity holds. v1.3 must pin the frozen-value comparison tolerance (numpy evaluator yields trailing float deltas, e.g. 0.3247270000001) and Windows CI (test-windows) becomes the numpy-path executor; revisit pyranges1 only if ruranges ships win_amd64 wheels. CHANGE SURFACE INVENTORY 2026-10-10 (exhaustive example/ sweep): (A) pybedtools runtime, exactly two files sharing one code path — generate_bpe_dataset.py:7,249 and data_generation_and_inference.ipynb cell 2/cell 8 (same loj intersect) plus cell 0's '!uv pip install pyfastx pybedtools' (install cell must skip pybedtools on Windows); (B) bedtools CLI subprocess — plant_helixseek_cre.ipynb cell 11 jaccard call (active; md cells 0/10/22 frozen-convention docs stay bedtools-worded on Linux) and .scratch/select_loci.py:662 (retired gitignored Phase-6 one-shot, OUT of scope); (C) docs-only — selection.md:35 historical methodology, unchanged. Test gates to make platform-aware: tests/examples/test_script_execution.py:128 (_gate_script_pybedtools), tests/examples/test_notebook_execution.py:1397 (_gate_pybedtools), tests/examples/test_plant_helixseek_showcase.py:523 (which('bedtools') fail-loud assert) — on Windows skip/fail becomes numpy-path pass; pyproject 'platform_system != Windows' marker unchanged (Linux-side dependency declaration untouched). | open |  | 2026-10-08T08:00:06.495Z |  |
| 20 | 10 | deviation | dnallm/inference/vep.py |  | Plan 10-04 coverage-proof substitution: pytest --cov=<module> fails repo-wide in the current shared dev venv — coverage 7.16.2 source_pkgs resolution imports dnallm under the active tracer and the utils-shims→torch→numpy 2.5.3 chain trips numpy's 'cannot load module more than once per process' C-extension guard during conftest import (reproduces on untouched test_mutagenesis.py and a minimal Coverage(source=['dnallm'])+import numpy probe; nothing installed by the plan). vep.py measured 100% (62/62) via 'coverage run --include=dnallm/inference/vep.py' — same instrument, no source import at start. CI legs (numpy 1.26.4/2.2.0) unaffected; open question: local venv numpy pin vs coverage interplay fix | open |  | 2026-10-09T11:07:16.415Z |  |
| 21 | 10 | stub | docs/user_guide/fine_tuning/peft_adapters.md | 163 | IA³ chapter section is an honest forward pointer by plan design (REV-03 split delivery, 10-03 D-08/plan prohibitions): finetune.use_ia3 and the ia3 YAML section exist as config surface, but the trainer branch, per-model target presets, and working examples complete in Phase 11-12 after PEFT-01 — the section names no nonexistent API and must be completed (then marked fixed) when the IA³ trainer branch ships | fixed | Stale-open verified at 2026-10-10 backlog review: IA³ trainer branch + per-family presets shipped in Phase 11 (transformer + Mamba proven, seed-pinned roundtrip) and peft_adapters.md:173+ now carries the full IA³ chapter (YAML example, preset-table selection, save/reload path) — completion condition met | 2026-10-09T11:15:30.000Z | 2026-10-10T14:14:31.103Z |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "01",
    "file": ".planning/phases/01-harness-integrity-measured-baseline/01-01-PLAN.md",
    "line": null,
    "description": "Task 2 verify grep 'tasks/metrics' false-positives on measured dispatcher dnallm/tasks/metrics.py; boundary re-proved with precise patterns (vendored dir absent, neighbors present)",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-29T17:37:20.663Z",
    "resolved_at": "2026-10-01T14:32:27.252Z",
    "milestone": null
  },
  {
    "id": 2,
    "kind": "unmet-truth",
    "phase": "3",
    "file": "dnallm/inference/inference.py",
    "line": 1643,
    "description": "generate-from-DataLoader never appends to prompt_seqs (seqs.extend on itself); causallm generate over a DataLoader returns empty list",
    "status": "waived",
    "reason": "Latent bug pinned by tests, out of v1 scope (milestone closed 2026-10-01): generate-from-DataLoader returns empty list (inference.py:1643). Deferred to next-milestone backlog — recorded in v1-MILESTONE-AUDIT.md tech-debt ledger",
    "recorded_at": "2026-09-30T09:46:44.754Z",
    "resolved_at": "2026-10-01T14:32:27.857Z",
    "milestone": null
  },
  {
    "id": 3,
    "kind": "unmet-truth",
    "phase": "3",
    "file": "dnallm/inference/mutagenesis.py",
    "line": 429,
    "description": "evaluate strategy max calls raw_score.index() on an ndarray (AttributeError) — latent bug documented as accepted residual",
    "status": "waived",
    "reason": "Latent bug pinned by tests, out of v1 scope (milestone closed 2026-10-01): mutagenesis 'max' strategy calls ndarray.index() (mutagenesis.py:429) AttributeError. Deferred to next-milestone backlog — v1-MILESTONE-AUDIT.md tech-debt ledger",
    "recorded_at": "2026-09-30T09:46:44.856Z",
    "resolved_at": "2026-10-01T14:32:27.943Z",
    "milestone": null
  },
  {
    "id": 4,
    "kind": "stub",
    "phase": "03",
    "file": "dnallm/models/model.py",
    "line": 264,
    "description": "cosine_similarity loss_function constructs CosineEmbeddingLoss but calls it with (logits, labels) — missing target arg raises TypeError for every user selecting it (covered by test_forward_cosine_similarity_loss_crashes; fix deferred, semantics ambiguous)",
    "status": "waived",
    "reason": "Latent bug pinned by test_forward_cosine_similarity_loss_crashes, out of v1 scope (milestone closed 2026-10-01): cosine_similarity loss TypeError, semantics ambiguous (model.py:264). Deferred to next-milestone backlog — v1-MILESTONE-AUDIT.md tech-debt ledger",
    "recorded_at": "2026-09-30T10:39:17.694Z",
    "resolved_at": "2026-10-01T14:32:28.031Z",
    "milestone": null
  },
  {
    "id": 5,
    "kind": "deviation",
    "phase": "03",
    "file": ".planning/phases/03-test-authoring-to-90-coverage/03-03-PLAN.md",
    "line": null,
    "description": "Wave-3 verify gate 'assert not logs.exists()' is unsatisfiable: dnallm's import-time file sink (utils/logger.py:57-60) creates logs/dnallm.log at the pytest launch cwd on every suite run — waves 4-5 plans must gate on 'no logs/mcp_server.log at repo root' instead (sink also recorded in deferred-items.md)",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-30T11:33:06.961Z",
    "resolved_at": "2026-10-01T14:32:27.339Z",
    "milestone": null
  },
  {
    "id": 6,
    "kind": "deviation",
    "phase": "3",
    "file": "tests/datahandling/test_dna_dataset.py",
    "line": null,
    "description": "pytest 9.1.1 --collect-only emits no :: separators - the plan's grep -c :: tripwires were enforced as the equivalent 'N tests collected' counts (66/151 >= 35/55; trainer 38 >= 10), as in waves 1-3",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-30T12:22:58.724Z",
    "resolved_at": "2026-10-01T14:32:27.426Z",
    "milestone": null
  },
  {
    "id": 7,
    "kind": "deviation",
    "phase": "04",
    "file": "pyproject.toml",
    "line": null,
    "description": "Plan 04-01 synthetic-drop verify as written (--ignore=tests/models/test_model.py) cannot go red: 91.65% under -m 'not slow'; corrected proof and 04-03 GATE-04 probe must ignore/delete the whole tests/models dir",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-30T16:32:51.841Z",
    "resolved_at": "2026-10-01T14:32:27.511Z",
    "milestone": null
  },
  {
    "id": 8,
    "kind": "deviation",
    "phase": "04",
    "file": ".github/workflows/ci.yml",
    "line": null,
    "description": "Plan 04-02 verify commands needed --workflow CI disambiguation (Docs Validation run stole the latest-push-run slot) and mid-run job logs are 404 on GitHub's API until completion - step-state is the runtime health proof",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-30T17:03:40.700Z",
    "resolved_at": "2026-10-01T14:32:27.597Z",
    "milestone": null
  },
  {
    "id": 9,
    "kind": "deviation",
    "phase": "04",
    "file": ".planning/phases/04-ci-gate-enforcement/04-03-PLAN.md",
    "line": null,
    "description": "Plan arithmetic defect: single-file tests/models/test_model.py deletion cannot clear the 90 floor under -m 'not slow' (91.64% green); probe target re-planned to directory deletion (78.92% red local, 78.91% CI) per the plan own rehearsal gate — resolved in 04-03",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-30T18:17:11.661Z",
    "resolved_at": "2026-10-01T14:32:27.683Z",
    "milestone": null
  },
  {
    "id": 10,
    "kind": "deviation",
    "phase": "04",
    "file": ".planning/phases/04-ci-gate-enforcement/04-03-PLAN.md",
    "line": null,
    "description": "Verify mechanics: gh run view --job --log-failed gates on whole-run completion; evidence harvested via job-level logs API (gh api actions/jobs/<id>/logs) which serves completed jobs mid-run — resolved in 04-03",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-30T18:17:11.751Z",
    "resolved_at": "2026-10-01T14:32:27.771Z",
    "milestone": null
  },
  {
    "id": 11,
    "kind": "deviation",
    "phase": "05",
    "file": "example/notebooks/benchmark/benchmark.ipynb",
    "line": null,
    "description": "Census FAIL row (05-04, D-07 ladder terminal): third registry model zhangtaolab/nucleotide-transformer-v2-100m-promoter not loadable on transformers 5.17 - remote code needs removed 4.x PretrainedConfig defaults (is_decoder/add_cross_attention); native-ESM route refuted (FFN shape mismatch); owner disposition pending per D-09",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-02T05:12:38.864Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 12,
    "kind": "stub",
    "phase": "05",
    "file": "tests/examples/test_script_execution.py",
    "line": null,
    "description": "environment-unavailable typed skip: generate_bpe_dataset.py pkl-production leg blocked by GAP-1-class remote-code gap (plant-nucleotide-transformer-BPE needs removed 4.x PretrainedConfig defaults on transformers 5.17); self-healing, owner disposition pending",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-02T05:48:40.008Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 13,
    "kind": "unrun-verify",
    "phase": "05",
    "file": "example/mcp_example",
    "line": null,
    "description": "Census deferred-owner rows (05-06, D-08/T-05-16): both ollama mcp client notebooks never executed - ollama probe GREEN (qwen3.8:latest) but dnallm MCP server endpoint down and uv pip install cells must never touch the project venv; needs the Phase-8 ollama/VRAM coexistence plan (owner decision); durable gated tests skip network-unavailable with live probe evidence and fail loudly if both endpoints come up",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-02T08:34:21.484Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 14,
    "kind": "deviation",
    "phase": "05",
    "file": "dnallm/inference/benchmark.py",
    "line": 296,
    "description": "Census FAIL finding (05-06): Benchmark.run hardcodes self.datasets[di]['labels'] while example benchmark_config.yaml declares label_column 'label' - KeyError before any model loads; blocks the benchmark notebook (and its NT third-model disposition) until repaired; Phase 8 repair queue",
    "status": "fixed",
    "reason": "run() label resolution honors configured label_column with 'labels' fallback and descriptive ValueError; red-first regression tests in TestRunLabelColumnResolution; note: census's ['sequence']-only shape stemmed from generate_dataset's silent path-to-sequence fallback, a separate latent bug deliberately left out of scope",
    "recorded_at": "2026-10-02T08:34:21.569Z",
    "resolved_at": "2026-10-10T14:39:32.164Z",
    "milestone": "v1.1"
  },
  {
    "id": 15,
    "kind": "skipped-test",
    "phase": "05",
    "file": "tests/examples/test_notebook_execution.py",
    "line": null,
    "description": "05-06 gated typed skips (sanctioned, self-healing): optional-dep probe-then-execute for evo/megaDNA prerequisites, finetune_custom_head megaDNA demo cell and PlantCAD lora pair (mamba_ssm); environment-unavailable script-lane skip (05-05 pattern) unchanged; all matched by audit_skips against registered prefixes",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-02T08:34:21.655Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 16,
    "kind": "unmet-truth",
    "phase": "09",
    "file": "tests/benchmark/test_benchmark.py",
    "line": null,
    "description": "Pre-existing fast-lane failure (found by 09-02 verify, reproduced at plan-start 3557e0b): TestBenchmark::test_plot_for_regression pandas TypeError float() argument ... not dict via _astype_nansafe in the plot path; not caused by any Phase 09 change (09-02 delta +8P/+0F/+0S); likely quick-task 13/14 Mapping fallout; logged in 09 deferred-items.md",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-05T15:45:36.353Z",
    "resolved_at": "2026-10-06T01:31:42.584Z",
    "milestone": "v1.1"
  },
  {
    "id": 17,
    "kind": "unmet-truth",
    "phase": "quick-261007-vxx",
    "file": "docs/user_guide/fine_tuning/getting_started.md",
    "line": null,
    "description": "Push of ruff-format fix commit 97a7c30 to origin/dev blocked by GitHub receive-side Internal Server Error (4 attempts, Request IDs 8832:3513C8/C942:3774A8/B48E:2E4A69/991C:246D9C, 2026-10-07 15:07-15:12Z); local ruff format --check green at dev 97a7c30; re-run 'git push origin dev' when GitHub receive recovers to unblock ci.yml format gates + PR #40",
    "status": "fixed",
    "reason": "GitHub receive recovered; orchestrator re-push at 2026-10-07T15:17:44Z landed cfc8346..97a7c30 on origin/dev; CI + Docs Validation re-triggered",
    "recorded_at": "2026-10-07T15:20:00.000Z",
    "resolved_at": "2026-10-07T15:18:00.000Z",
    "milestone": "v1.1"
  },
  {
    "id": 18,
    "kind": "deviation",
    "phase": "quick-261008-env",
    "file": "pyproject.toml",
    "line": null,
    "description": "Windows fla installability gap (env adaptation 2026-10-08): pip install -e '.[fla]' on Windows succeeds but leaves fla broken - torch Windows wheels declare no triton dependency (Linux pulls it) and fla-core requires triton only under its [cuda]/[cpu] extras which the pyproject deliberately avoids, so fla imports top-level but fla.ops.* raises ModuleNotFoundError and test_chunk_kda_importable_when_fla_installed FAILS instead of skipping. Remedy installed in dnallm-cuda: triton-windows==3.7.1.post27 (community wheels; upstream triton ships no win_amd64) - KDA chunk kernel proven compiling+executing on RTX 5080 sm_120. Open owner decision: whether to declare triton-windows behind a Windows platform marker in the fla extra (third-party fork enters declared deps) or document the manual step",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-08T04:30:00.000Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 19,
    "kind": "deviation",
    "phase": "quick-261008-wfx",
    "file": "tests/examples/test_plant_helixseek_showcase.py",
    "line": null,
    "description": "CRE showcase execution test requires the bedtools CLI and fails loud by design (test line ~480 asserts shutil.which('bedtools')); bedtools has no Windows build, so local Windows full-example runs must --deselect test_cre_notebook_executes_within_selection_bands (workaround in daily use since 2026-10-08). Open question: WSL-scope bedtools provisioning or permanent documented deselect; Linux nightly lanes unaffected. EVALUATION 2026-10-10 (owner-directed de-bedtools assessment): pyranges1 REJECTED as replacement — ruranges 0.2.7 ships no win_amd64 wheels (Windows would need Rust msvc source build; the pybedtools platform marker would simply transfer), requires-python >=3.12, and it buys zero Windows gain since CRE jaccard is subprocess-direct CLI anyway (join_overlaps left mode does map to the NER loj intersect, noted for completeness). CHOSEN DIRECTION (refined by owner 2026-10-10, platform-split): Linux/macOS KEEP bedtools/pybedtools — the frozen truth chain stays bedtools-computed, no re-freeze; Windows gets a parity-verified numpy evaluator (precedent dnallm/utils/genomic_coords.py) used only where bedtools cannot install — the CRE jaccard first, the NER example's pybedtools loj intersect second (larger surface: join semantics + ascending-order traversal). OWNER DIRECTIVE 2026-10-10: the NER example CODE ITSELF must change — generate_bpe_dataset.py and data_generation_and_inference.ipynb gain runtime platform detection (Linux keeps pybedtools.BedTool.intersect(loj=True), Windows dispatches to the numpy equivalent); the downstream frozen Anno metric (exon_f1=0.7522) must hold in band on BOTH paths, and the loj replacement needs its OWN parity check (numpy join vs `bedtools intersect -loj` — jaccard parity does not transfer to join semantics). Enabling evidence: quick-task spike 261010-wx2 (numpy vs bedtools v2.31.1 parity on the frozen selection pair jaccard=0.324727 + randomized property cases) — without parity the split would be two truths, so the split ships only if parity holds. v1.3 must pin the frozen-value comparison tolerance (numpy evaluator yields trailing float deltas, e.g. 0.3247270000001) and Windows CI (test-windows) becomes the numpy-path executor; revisit pyranges1 only if ruranges ships win_amd64 wheels. CHANGE SURFACE INVENTORY 2026-10-10 (exhaustive example/ sweep): (A) pybedtools runtime, exactly two files sharing one code path — generate_bpe_dataset.py:7,249 and data_generation_and_inference.ipynb cell 2/cell 8 (same loj intersect) plus cell 0's '!uv pip install pyfastx pybedtools' (install cell must skip pybedtools on Windows); (B) bedtools CLI subprocess — plant_helixseek_cre.ipynb cell 11 jaccard call (active; md cells 0/10/22 frozen-convention docs stay bedtools-worded on Linux) and .scratch/select_loci.py:662 (retired gitignored Phase-6 one-shot, OUT of scope); (C) docs-only — selection.md:35 historical methodology, unchanged. Test gates to make platform-aware: tests/examples/test_script_execution.py:128 (_gate_script_pybedtools), tests/examples/test_notebook_execution.py:1397 (_gate_pybedtools), tests/examples/test_plant_helixseek_showcase.py:523 (which('bedtools') fail-loud assert) — on Windows skip/fail becomes numpy-path pass; pyproject 'platform_system != Windows' marker unchanged (Linux-side dependency declaration untouched).",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-08T08:00:06.495Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 20,
    "kind": "deviation",
    "phase": "10",
    "file": "dnallm/inference/vep.py",
    "line": null,
    "description": "Plan 10-04 coverage-proof substitution: pytest --cov=<module> fails repo-wide in the current shared dev venv — coverage 7.16.2 source_pkgs resolution imports dnallm under the active tracer and the utils-shims→torch→numpy 2.5.3 chain trips numpy's 'cannot load module more than once per process' C-extension guard during conftest import (reproduces on untouched test_mutagenesis.py and a minimal Coverage(source=['dnallm'])+import numpy probe; nothing installed by the plan). vep.py measured 100% (62/62) via 'coverage run --include=dnallm/inference/vep.py' — same instrument, no source import at start. CI legs (numpy 1.26.4/2.2.0) unaffected; open question: local venv numpy pin vs coverage interplay fix",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-09T11:07:16.415Z",
    "resolved_at": null,
    "milestone": "v1.2"
  },
  {
    "id": 21,
    "kind": "stub",
    "phase": "10",
    "file": "docs/user_guide/fine_tuning/peft_adapters.md",
    "line": 163,
    "description": "IA³ chapter section is an honest forward pointer by plan design (REV-03 split delivery, 10-03 D-08/plan prohibitions): finetune.use_ia3 and the ia3 YAML section exist as config surface, but the trainer branch, per-model target presets, and working examples complete in Phase 11-12 after PEFT-01 — the section names no nonexistent API and must be completed (then marked fixed) when the IA³ trainer branch ships",
    "status": "fixed",
    "reason": "Stale-open verified at 2026-10-10 backlog review: IA³ trainer branch + per-family presets shipped in Phase 11 (transformer + Mamba proven, seed-pinned roundtrip) and peft_adapters.md:173+ now carries the full IA³ chapter (YAML example, preset-table selection, save/reload path) — completion condition met",
    "recorded_at": "2026-10-09T11:15:30.000Z",
    "resolved_at": "2026-10-10T14:14:31.103Z",
    "milestone": "v1.2"
  }
]
````
