---
schema_version: 1
open_count: 5
waived_count: 3
fixed_count: 8
total_count: 16
last_updated: 2026-10-06T01:31:42.584Z
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
| 14 | 05 | deviation | dnallm/inference/benchmark.py | 296 | Census FAIL finding (05-06): Benchmark.run hardcodes self.datasets[di]['labels'] while example benchmark_config.yaml declares label_column 'label' - KeyError before any model loads; blocks the benchmark notebook (and its NT third-model disposition) until repaired; Phase 8 repair queue | open |  | 2026-10-02T08:34:21.569Z |  |
| 15 | 05 | skipped-test | tests/examples/test_notebook_execution.py |  | 05-06 gated typed skips (sanctioned, self-healing): optional-dep probe-then-execute for evo/megaDNA prerequisites, finetune_custom_head megaDNA demo cell and PlantCAD lora pair (mamba_ssm); environment-unavailable script-lane skip (05-05 pattern) unchanged; all matched by audit_skips against registered prefixes | open |  | 2026-10-02T08:34:21.655Z |  |
| 16 | 09 | unmet-truth | tests/benchmark/test_benchmark.py |  | Pre-existing fast-lane failure (found by 09-02 verify, reproduced at plan-start 3557e0b): TestBenchmark::test_plot_for_regression pandas TypeError float() argument ... not dict via _astype_nansafe in the plot path; not caused by any Phase 09 change (09-02 delta +8P/+0F/+0S); likely quick-task 13/14 Mapping fallout; logged in 09 deferred-items.md | fixed |  | 2026-10-05T15:45:36.353Z | 2026-10-06T01:31:42.584Z |

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
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-02T08:34:21.569Z",
    "resolved_at": null,
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
    "status": "resolved",
    "reason": "GitHub receive recovered; orchestrator re-push at 2026-10-07T15:17:44Z landed cfc8346..97a7c30 on origin/dev; CI + Docs Validation re-triggered",
    "recorded_at": "2026-10-07T15:20:00.000Z",
    "resolved_at": "2026-10-07T15:18:00.000Z",
    "milestone": "v1.1"
  }
]
````
