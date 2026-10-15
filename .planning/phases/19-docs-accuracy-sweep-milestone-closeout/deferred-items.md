# Phase 19 Deferred Items (out-of-scope discoveries from review batches)

Code-level findings surfaced by the docs accuracy sweep that are OUTSIDE the
sweep's docs-only scope (the batches edit docs/ + phase artifacts only).
Routed here per the sweep's scope fence; owners decide disposition.

| # | Found by | Finding | Evidence | Suggested fix |
|---|----------|---------|----------|---------------|
| 1 | 19-04 Task 1 (cli/usage.md review) | The flag-only invocation paths of the training/inference CLIs crash: `dnallm-train --model X --data Y --output Z` and `dnallm-inference --model X --input Y` (both the `dnallm` group subcommands in dnallm/cli/cli.py:45/:94 and the standalone dnallm/cli/train.py:36 / inference.py:36) build a minimal config dict WITHOUT the `task` section, and DNATrainer.__init__ / DNAInference.__init__ do `config["task"]` → KeyError: 'task' | Reproduced empirically: `DNATrainer(model=None, config={'model_name_or_path':'x','data_path':'y','output_dir':'o','finetune':{...}})` → `KeyError: 'task'`; same for DNAInference. The --config paths work | Give the minimal config a task section (e.g. TaskConfig(task_type="binary")) or require --config; docs currently carry a note that the flag-only path is broken (docs/user_guide/cli/usage.md) — remove the note when fixed |
| 2 | 19-04 Task 1 (models.md/common_workflows.md review) | Policy application for 19-03's flagged source="huggingface" pattern: 19-04 aligned RUNNABLE examples to lock-pinned modelscope routes (models.md, common_workflows.md, docs/index.md); fine_tuning/task_guides.md (19-03's batch) still cites source="huggingface" for plant-dnabert-BPE prose-side | models.lock:21 pins `ms zhangtaolab/plant-dnabert-BPE`; the finetune_binary notebook's executed route is source="modelscope" (hf string is a commented alternative) | 19-05/19-06/19-07 or the verifier may apply the same alignment to task_guides.md if the prose-vs-runnable distinction is not wanted |
