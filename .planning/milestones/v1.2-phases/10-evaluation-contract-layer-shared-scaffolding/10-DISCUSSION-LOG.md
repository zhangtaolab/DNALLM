# Phase 10: Evaluation Contract Layer & Shared Scaffolding - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-09
**Phase:** 10-Evaluation Contract Layer & Shared Scaffolding
**Areas discussed:** evaluate() API design, Guard loudness on flip, Scaffolding stub depth, Docs sweep scope

> Discussion was conducted in Chinese at the user's request (bioinformatics-scientist register); options below are English summaries of what was presented.

---

## evaluate() API design

### Q1: Where does evaluate(split=...) live?

| Option | Description | Selected |
|--------|-------------|----------|
| DNATrainer.evaluate 覆写 | Override HF Trainer.evaluate, signature-compatible (legacy kwargs passthrough, split= via predict path); needs signature-compat tests for external callers (wandb/sweeps) | ✓ |
| 新方法名 evaluate_split | New method name on DNATrainer, zero shadow risk, deviates from requirement's literal `evaluate(split=...)` | |
| 挂到 DNAInference | On the inference engine (has predict path); loses trainer state (best checkpoint, training config) | |

**User's choice:** DNATrainer.evaluate override
**Notes:** Shadow risk accepted; signature-compatibility tests are the countermeasure.

### Q2: Output shape?

| Option | Description | Selected |
|--------|-------------|----------|
| dict + result JSON | Metrics dict (canonical keys) + result JSON (split name, timestamp, metrics); REV-09 aggregate_seeds consumes the JSONs directly | ✓ |
| 仅返回 dict | Minimal interface; every experiment script re-implements serialization, formats drift | |
| dict + JSON + CSV | Paper-table friendly CSV; second on-disk format to maintain, no aggregation gain | |

**User's choice:** dict + result JSON

### Q3: Which weights does evaluate() use?

| Option | Description | Selected |
|--------|-------------|----------|
| 当前权重 | Weights the trainer holds post-train() (best checkpoint if load_best_model_at_end, else final epoch); one-line model-selection rule | ✓ |
| 当前权重+可选checkpoint | Optional checkpoint-path param; no v1.2 experiment needs it | |
| 强制末epoch权重 | Always final-epoch; contradicts HF ecosystem behavior, forces needless reload under early stopping | |

**User's choice:** Current weights

---

## Guard loudness on flip

### Q1: Does the behavior flip itself warn?

| Option | Description | Selected |
|--------|-------------|----------|
| 翻转时 WARN | WARN at flip moment: test excluded, differs from old behavior, how to opt in (allow_test_as_eval=True); same structure as reviewer R1-2c's ask | ✓ |
| 静默应用新默认 | Silent new default, CHANGELOG-only migration note; eval metrics vanish without explanation for old configs | |
| WARN+训练结束提醒 | Flip WARN + end-of-training reminder if never evaluated; second line is noise on mask/generation tasks | |

**User's choice:** WARN at flip time

### Q2: Where does the allow_test_as_eval switch live?

| Option | Description | Selected |
|--------|-------------|----------|
| TrainingConfig 字段 | Pydantic field in configs.py, YAML-configurable, default False; matches house pattern; dnallmmark declares intent in YAML | ✓ |
| 仅构造参数 | DNATrainer kwarg only; dnallmmark must pass it in code, not declaratively | |
| 双入口 | Config field + kwarg with precedence; consistency burden without v1.2 demand | |

**User's choice:** TrainingConfig field

---

## Scaffolding stub depth

### Q1: How complete are the stubs?

| Option | Description | Selected |
|--------|-------------|----------|
| 字段齐全+注册 | Final fields + validations + load_config() section registration + pyproject package-data in the one-pass change; Phase 11 B4/B5 never touch configs.py | ✓ |
| 薄壳+仅注册 | Empty BaseModel shells + section keys; B1/B4/B5 all return to configs.py for fields — collision risk returns | |
| 按需最小 | Register only provably-YAML sections; the "which sections are YAML" judgment itself must be made now and may be wrong | |

**User's choice:** Field-complete + registration

### Q2: When does TrainingConfig.use_ia3 land?

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 10 字段先行 | Field (default False + docstring) in Phase 10; validators + trainer branch with B1; interim window is repo-internal only | ✓ |
| 全部留 B1 | Field + validators + branch all in Phase 11; TrainingConfig surface finalizes one wave later | |
| 字段+校验都先行 | Field + cross-field validators in Phase 10; splits PEFT-01's acceptance surface across two phases | |

**User's choice:** Phase 10 field-first

---

## Docs sweep scope

### Q1: Terminology sweep surfaces?

Census presented: "DNA language model" (old) in 30 non-mirror docs pages, 11 mirror files, 1 example file, 3 README hits, 16 dnallm .py files; target phrasing nearly absent. API reference pages render docstrings via mkdocstrings.

| Option | Description | Selected |
|--------|-------------|----------|
| 全面统一 | docs + README + docstrings (avoiding same-wave A1/A2/A4-owned files) + the 1 example/mirror pair swept together; byte-identity preserved | ✓ |
| 不碰 example | Everything except example/ and mirrors; leaves one exception corner against the "terminology unified" paper claim | |
| 仅 docs+README | Smallest; API reference (docstring-rendered) stays inconsistent on the most visible pages | |

**User's choice:** Full sweep

### Q2: CHANGELOG commit-traceability mechanism?

Constraint presented: an entry committed with its fix cannot contain its own SHA.

| Option | Description | Selected |
|--------|-------------|----------|
| REV-ID即写+SHA收尾回填 | REV-ID + reviewer-comment ID inline at write time (greppable); SHA backfilled as link by Phase 12 C3 closeout (already roadmap-assigned) | ✓ |
| 仅 REV-ID 不回填 | No backfill; rebuttal letter greps git log manually | |
| 独立证据表 | Separate REV-ID→SHA table (docs/rebuttal-evidence.md); most direct, one more doc surface to maintain | |

**User's choice:** REV-ID inline + SHA backfill at closeout

---

## Claude's Discretion

- `split=` key semantics (any present split key vs only test/dev)
- vep.py kernel reuse strategy (extract shared kernels from mutagenesis.py vs self-contained adaptation) — planner decides under the no-refactor constraint
- WARN channel (logger vs trainer's existing print style)
- Result-JSON location/naming; exact Ia3Config field set (mirror peft IA3Config)
- Docstring-sweep sequencing within A3 vs owner-side sweeps

## Deferred Ideas

None — discussion stayed within phase scope.
