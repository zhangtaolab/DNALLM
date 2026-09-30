---
schema_version: 1
open_count: 8
waived_count: 0
fixed_count: 0
total_count: 8
last_updated: 2026-09-30T17:03:40.700Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | .planning/phases/01-harness-integrity-measured-baseline/01-01-PLAN.md |  | Task 2 verify grep 'tasks/metrics' false-positives on measured dispatcher dnallm/tasks/metrics.py; boundary re-proved with precise patterns (vendored dir absent, neighbors present) | open |  | 2026-09-29T17:37:20.663Z |  |
| 2 | 3 | unmet-truth | dnallm/inference/inference.py | 1643 | generate-from-DataLoader never appends to prompt_seqs (seqs.extend on itself); causallm generate over a DataLoader returns empty list | open |  | 2026-09-30T09:46:44.754Z |  |
| 3 | 3 | unmet-truth | dnallm/inference/mutagenesis.py | 429 | evaluate strategy max calls raw_score.index() on an ndarray (AttributeError) — latent bug documented as accepted residual | open |  | 2026-09-30T09:46:44.856Z |  |
| 4 | 03 | stub | dnallm/models/model.py | 264 | cosine_similarity loss_function constructs CosineEmbeddingLoss but calls it with (logits, labels) — missing target arg raises TypeError for every user selecting it (covered by test_forward_cosine_similarity_loss_crashes; fix deferred, semantics ambiguous) | open |  | 2026-09-30T10:39:17.694Z |  |
| 5 | 03 | deviation | .planning/phases/03-test-authoring-to-90-coverage/03-03-PLAN.md |  | Wave-3 verify gate 'assert not logs.exists()' is unsatisfiable: dnallm's import-time file sink (utils/logger.py:57-60) creates logs/dnallm.log at the pytest launch cwd on every suite run — waves 4-5 plans must gate on 'no logs/mcp_server.log at repo root' instead (sink also recorded in deferred-items.md) | open |  | 2026-09-30T11:33:06.961Z |  |
| 6 | 3 | deviation | tests/datahandling/test_dna_dataset.py |  | pytest 9.1.1 --collect-only emits no :: separators - the plan's grep -c :: tripwires were enforced as the equivalent 'N tests collected' counts (66/151 >= 35/55; trainer 38 >= 10), as in waves 1-3 | open |  | 2026-09-30T12:22:58.724Z |  |
| 7 | 04 | deviation | pyproject.toml |  | Plan 04-01 synthetic-drop verify as written (--ignore=tests/models/test_model.py) cannot go red: 91.65% under -m 'not slow'; corrected proof and 04-03 GATE-04 probe must ignore/delete the whole tests/models dir | open |  | 2026-09-30T16:32:51.841Z |  |
| 8 | 04 | deviation | .github/workflows/ci.yml |  | Plan 04-02 verify commands needed --workflow CI disambiguation (Docs Validation run stole the latest-push-run slot) and mid-run job logs are 404 on GitHub's API until completion - step-state is the runtime health proof | open |  | 2026-09-30T17:03:40.700Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "01",
    "file": ".planning/phases/01-harness-integrity-measured-baseline/01-01-PLAN.md",
    "line": null,
    "description": "Task 2 verify grep 'tasks/metrics' false-positives on measured dispatcher dnallm/tasks/metrics.py; boundary re-proved with precise patterns (vendored dir absent, neighbors present)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-29T17:37:20.663Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 2,
    "kind": "unmet-truth",
    "phase": "3",
    "file": "dnallm/inference/inference.py",
    "line": 1643,
    "description": "generate-from-DataLoader never appends to prompt_seqs (seqs.extend on itself); causallm generate over a DataLoader returns empty list",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T09:46:44.754Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 3,
    "kind": "unmet-truth",
    "phase": "3",
    "file": "dnallm/inference/mutagenesis.py",
    "line": 429,
    "description": "evaluate strategy max calls raw_score.index() on an ndarray (AttributeError) — latent bug documented as accepted residual",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T09:46:44.856Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 4,
    "kind": "stub",
    "phase": "03",
    "file": "dnallm/models/model.py",
    "line": 264,
    "description": "cosine_similarity loss_function constructs CosineEmbeddingLoss but calls it with (logits, labels) — missing target arg raises TypeError for every user selecting it (covered by test_forward_cosine_similarity_loss_crashes; fix deferred, semantics ambiguous)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T10:39:17.694Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 5,
    "kind": "deviation",
    "phase": "03",
    "file": ".planning/phases/03-test-authoring-to-90-coverage/03-03-PLAN.md",
    "line": null,
    "description": "Wave-3 verify gate 'assert not logs.exists()' is unsatisfiable: dnallm's import-time file sink (utils/logger.py:57-60) creates logs/dnallm.log at the pytest launch cwd on every suite run — waves 4-5 plans must gate on 'no logs/mcp_server.log at repo root' instead (sink also recorded in deferred-items.md)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T11:33:06.961Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 6,
    "kind": "deviation",
    "phase": "3",
    "file": "tests/datahandling/test_dna_dataset.py",
    "line": null,
    "description": "pytest 9.1.1 --collect-only emits no :: separators - the plan's grep -c :: tripwires were enforced as the equivalent 'N tests collected' counts (66/151 >= 35/55; trainer 38 >= 10), as in waves 1-3",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T12:22:58.724Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 7,
    "kind": "deviation",
    "phase": "04",
    "file": "pyproject.toml",
    "line": null,
    "description": "Plan 04-01 synthetic-drop verify as written (--ignore=tests/models/test_model.py) cannot go red: 91.65% under -m 'not slow'; corrected proof and 04-03 GATE-04 probe must ignore/delete the whole tests/models dir",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T16:32:51.841Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 8,
    "kind": "deviation",
    "phase": "04",
    "file": ".github/workflows/ci.yml",
    "line": null,
    "description": "Plan 04-02 verify commands needed --workflow CI disambiguation (Docs Validation run stole the latest-push-run slot) and mid-run job logs are 404 on GitHub's API until completion - step-state is the runtime health proof",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-30T17:03:40.700Z",
    "resolved_at": null,
    "milestone": null
  }
]
````
