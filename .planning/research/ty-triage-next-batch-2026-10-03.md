# ty next-batch triage — post-baseline residual (2026-10-03)

Fresh `uvx ty check dnallm/` (ty 0.0.84, project venv py3.13.15, transformers 5.17.0)
after quick task 261003-0p0 (commits 7b0d360 + 9f0a2db). ZERO fixes applied by this
document — it is the input list for the next typing batch.

## Before / after

| Measurement | Diagnostics |
|---|---|
| 2026-10-02 report (`ty-check-dnallm-2026-10-02.txt`) | 568 |
| Live planning-time baseline, 2026-10-03 (pre-task, re-measured) | 570 |
| Flagless, after vendored excludes alone (planning-time proof) | 261 |
| After Task 1 (excludes + import suppressions + canonical renames) | 206 |
| FINAL — after Task 2 (TypedDict + 41 stale ignores removed + shim honesty) | **165** |

Residual rule histogram: invalid-argument-type x45, unresolved-attribute x30,
not-subscriptable x24, invalid-assignment x19, missing-argument x16,
invalid-return-type x9, no-matching-overload x5, call-non-callable x5,
unsupported-operator x3, not-iterable x3, deprecated x2, unsupported-base x1,
unknown-argument x1, invalid-parameter-default x1, invalid-method-override x1.

## Family ledger (what the baseline task did)

- **A — config excludes**: `[tool.ty.src] exclude` in pyproject.toml for exactly the three
  ruff-mirrored vendored paths (`dnallm/tasks/metrics/**`,
  `dnallm/models/special/mamba_npu.py`, `dnallm/finetune/megatron.py`). 570 -> 261 flagless.
  No `[[tool.ty.overrides]]` severity changes exist.
- **B — guarded-optional suppressions** (19 import lines, all verified absent from the venv
  by design): flash_attn (support.py), olmo (omnidna), lucagplm (lucaone handler +
  model.py raise-on-use site), gpn (gpn), evo2 + vortex (evo.py), evo + stripedhyena x3
  (evo.py), borzoi_pytorch x2 (borzoi), pyfaidx + polars (enformer data, raise-on-use),
  pacmap (plot), evo x2 (inference.py generate/score_sequences, raise-on-use).
- **C — canonical renames + lazy-export suppressions**: `PretrainedConfig` ->
  `PreTrainedConfig` at 6 sites (alias-identity verified live in transformers 5.17);
  `# ty: ignore[unresolved-import]` on transformers lazy exports at 25 import lines, each
  labeled `transformers lazy export, resolves live` (spot-verified via the project venv).
- **D — honest annotations**: `DNALLMConfig` TypedDict return for `load_config` (5-test
  contract); 41 blanket `# type: ignore` comments removed (ty proved them unused; the
  configured mypy lane re-demands none — zero keepers); transformers_compat shim polish
  (5 of 7 non-import diagnostics fixed by annotation/rename/early-return, 2 kept visible).
- **E — this list**: 165 residual diagnostics, enumerated below, severity-ordered by rule.

## error[invalid-argument-type] — 45 diagnostic(s)

Narrow or annotate the argument at the call site, or widen the callee signature. Read each individually: this class mixes under-annotation with genuine mismatches.

- `dnallm/cli/cli.py:53:42` — Argument to `DNATrainer.__init__` is incorrect.
- `dnallm/cli/cli.py:102:69` — Argument to `DNAInference.__init__` is incorrect.
- `dnallm/cli/cli.py:154:31` — Argument to `Benchmark.__init__` is incorrect.
- `dnallm/cli/inference.py:43:69` — Argument to `DNAInference.__init__` is incorrect.
- `dnallm/cli/train.py:43:42` — Argument to `DNATrainer.__init__` is incorrect.
- `dnallm/datahandling/data.py:369:20` — Argument to `DNADataset.__init__` is incorrect.
- `dnallm/datahandling/data.py:406:20` — Argument to `DNADataset.__init__` is incorrect.
- `dnallm/inference/benchmark.py:224:30` — Argument to bound method `list.append` is incorrect.
- `dnallm/inference/benchmark.py:299:22` — Method `__getitem__` of type `bound method str.__getitem__(key: SupportsIndex | slice[SupportsIndex | None, SupportsIndex | None, SupportsIndex | None], /) -> str` cannot be called with key of type `Literal["labels"]` on object of type `str`.
- `dnallm/inference/benchmark.py:314:77` — Argument to function `load_model_and_tokenizer` is incorrect.
- `dnallm/inference/benchmark.py:324:25` — Argument to function `load_model_and_tokenizer` is incorrect.
- `dnallm/inference/benchmark.py:342:21` — Argument to `DNADataset.__init__` is incorrect.
- `dnallm/inference/benchmark.py:445:16` — Argument to function `len` is incorrect.
- `dnallm/inference/benchmark.py:446:77` — Argument to bound method `DNAInference.calculate_metrics` is incorrect.
- `dnallm/inference/benchmark.py:452:37` — Argument to function `len` is incorrect.
- `dnallm/inference/benchmark.py:493:29` — Argument to function `load_model_and_tokenizer` is incorrect.
- `dnallm/inference/benchmark.py:493:54` — Argument to function `load_model_and_tokenizer` is incorrect.
- `dnallm/inference/benchmark.py:542:25` — Argument to bound method `Benchmark.evaluate_single_model` is incorrect.
- `dnallm/inference/inference.py:515:13` — Argument to `DataLoader.__init__` is incorrect.
- `dnallm/inference/inference.py:1163:17` — Argument to `DataLoader.__init__` is incorrect.
- `dnallm/inference/inference.py:1342:17` — Argument to function `plot_attention_map` is incorrect.
- `dnallm/inference/inference.py:1414:17` — Argument to function `plot_embeddings` is incorrect.
- `dnallm/inference/mutagenesis.py:178:13` — Argument to `DataLoader.__init__` is incorrect.
- `dnallm/mcp/client.py:139:47` — Argument is incorrect.
- `dnallm/mcp/client.py:158:35` — Argument is incorrect.
- `dnallm/mcp/server.py:1751:23` — Argument to `Mount.__init__` is incorrect.
- `dnallm/models/model.py:878:57` (x2) — Argument to function `_load_model_by_task_type` is incorrect.
- `dnallm/models/special/crossdna.py:409:17` — Argument is incorrect.
- `dnallm/tasks/metrics.py:219:26` — Argument expression after ** must be a mapping type.
- `dnallm/tasks/metrics.py:219:33` — Argument expression after ** must be a mapping type.
- `dnallm/tasks/metrics.py:219:50` — Argument expression after ** must be a mapping type.
- `dnallm/tasks/metrics.py:527:40` — Argument to constructor `zip.__new__` is incorrect.
- `dnallm/tasks/metrics.py:536:40` — Argument to constructor `zip.__new__` is incorrect.
- `dnallm/tasks/metrics.py:640:23` — Argument expression after ** must be a mapping type.
- `dnallm/tasks/metrics.py:641:23` — Argument expression after ** must be a mapping type.
- `dnallm/tasks/metrics.py:642:23` — Argument expression after ** must be a mapping type.
- `dnallm/tasks/metrics.py:643:23` — Argument expression after ** must be a mapping type.
- `dnallm/utils/transformers_compat.py:693:17` — Argument to function `zeros` is incorrect.
- `dnallm/utils/transformers_compat.py:700:17` — Argument to function `zeros` is incorrect.

## error[unresolved-attribute] — 30 diagnostic(s)

Dominated by union-narrowing debt: values from `config.get(...)`/Optional returns used without narrowing. Fixes are isinstance guards, `in` checks on TypedDict keys, or precise local annotations.

- `dnallm/datahandling/data.py:402:18` — Attribute `rename_column` is not defined on `dict[Unknown, Unknown]`, `MsDataset` in union `dict[Unknown, Unknown] | MsDataset | NativeIterableDataset`.
- `dnallm/datahandling/data.py:404:18` — Attribute `rename_column` is not defined on `dict[Unknown, Unknown]`, `MsDataset` in union `dict[Unknown, Unknown] | MsDataset | IterableDataset`.
- `dnallm/inference/benchmark.py:146:39` — Attribute `config_path` is not defined on `InferenceConfig`, `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:147:18` — Attribute `models` is not defined on `InferenceConfig`, `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:152:26` — Attribute `evaluation` is not defined on `InferenceConfig`, `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:155:47` — Attribute `output` is not defined on `InferenceConfig`, `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:174:19` — Attribute `metrics` is not defined on `InferenceConfig`, `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:175:23` — Attribute `output` is not defined on `InferenceConfig`, `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:281:30` — Attribute `name` is not defined on `TaskConfig`, `str` in union `Unknown | TaskConfig | str`.
- `dnallm/inference/benchmark.py:338:66` — Attribute `max_length` is not defined on `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:340:34` — Attribute `max_length` is not defined on `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:349:32` — Attribute `batch_size` is not defined on `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:350:33` — Attribute `num_workers` is not defined on `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:377:28` — Attribute `output_dir` is not defined on `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:378:45` — Attribute `output_dir` is not defined on `TaskConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:486:55` — Attribute `items` is not defined on `list[Unknown]`, `list[TaskConfig]`, `list[str]` in union `Unknown | dict[Unknown, Unknown] | list[Unknown] | list[TaskConfig] | list[str]`.
- `dnallm/inference/benchmark.py:512:43` — Attribute `split` is not defined on `None` in union `None | StratifiedKFold | KFold`.
- `dnallm/inference/benchmark.py:521:39` — Attribute `split` is not defined on `None` in union `None | StratifiedKFold | KFold`.
- `dnallm/inference/benchmark.py:588:21` — Attribute `task_type` is not defined on `InferenceConfig` in union `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/inference.py:485:24` — Attribute `features` is not defined on `DatasetDict` in union `Dataset | DatasetDict | Unknown`.
- `dnallm/inference/inference.py:489:24` — Attribute `features` is not defined on `DatasetDict` in union `Dataset | DatasetDict | Unknown`.
- `dnallm/inference/inference.py:498:45` — Attribute `features` is not defined on `DatasetDict` in union `Dataset | DatasetDict | Unknown`.
- `dnallm/inference/inference.py:501:45` — Attribute `features` is not defined on `DatasetDict` in union `Dataset | DatasetDict | Unknown`.
- `dnallm/inference/inference.py:2085:21` — Attribute `append` is not defined on `None` in union `None | Unknown`.
- `dnallm/mcp/server.py:778:24` — Module `asyncio` has no member `timeout`.
- `dnallm/mcp/server.py:900:24` — Module `asyncio` has no member `timeout`.
- `dnallm/mcp/server.py:1049:24` — Module `asyncio` has no member `timeout`.
- `dnallm/mcp/tests/test_config_manager.py:452:9` — Attribute `multi_model` is not defined on `None` in union `MCPServerConfig | None`.
- `dnallm/models/model.py:101:26` — Attribute `parameters` is not defined on `None` in union `Unknown | None`.
- `dnallm/models/model.py:201:34` — Attribute `long` is not defined on `bool` in union `Unknown | bool`.

## error[not-subscriptable] — 24 diagnostic(s)

Values inferred `None` (Optional returns / late-bound attributes) then subscripted. Add None-guards or honest local annotations before the subscript.

- `dnallm/inference/interpret.py:124:28` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/interpret.py:125:28` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:272:16` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:273:31` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:275:26` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:325:16` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:326:31` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:328:26` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:413:83` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:415:85` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:441:24` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:443:23` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/inference/mutagenesis.py:449:33` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/models/special/evo.py:316:38` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/models/special/evo.py:317:48` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:181:17` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:192:17` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:550:25` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:551:26` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:552:23` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:553:19` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:612:44` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:644:34` — Cannot subscript object of type `None` with no `__getitem__` method.
- `dnallm/tasks/metrics.py:645:34` — Cannot subscript object of type `None` with no `__getitem__` method.

## error[invalid-assignment] — 19 diagnostic(s)

Variable or attribute assigned a value outside its declared/inferred type. Rename locals across branch types or widen the annotation honestly.

- `dnallm/datahandling/data.py:134:9` — Object of type `None` is not assignable to attribute `stats` on type `DatasetDict | Dataset`.
- `dnallm/datahandling/data.py:779:9` — Object of type `Literal[True]` is not assignable to attribute `_is_encoded` on type `Dataset | DatasetDict | Unknown`.
- `dnallm/datahandling/data.py:1337:13` — Object of type `dict[Unknown, Unknown]` is not assignable to attribute `stats_for_plot` of type `DataFrame | None`.
- `dnallm/datahandling/data.py:1343:17` — Cannot assign to a subscript on an object of type `None`.
- `dnallm/finetune/trainer.py:257:13` — Object of type `Literal["regression"]` is not assignable to attribute `problem_type` on type `Any | Tensor | Module`.
- `dnallm/inference/benchmark.py:155:9` — Object of type `Unknown` is not assignable to attribute `output_dir` on type `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:161:17` — Object of type `Unknown` is not assignable to attribute `num_labels` on type `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:162:17` — Object of type `Unknown` is not assignable to attribute `label_names` on type `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:163:17` — Object of type `Unknown` is not assignable to attribute `threshold` on type `Unknown | InferenceConfig | TaskConfig`.
- `dnallm/inference/benchmark.py:282:27` — Object of type `Unknown | dict[Unknown, Unknown] | list[Unknown] | list[TaskConfig] | list[str]` is not assignable to `list[str] | dict[Unknown, Unknown] | None`.
- `dnallm/inference/benchmark.py:311:26` — Object of type `Unknown | TaskConfig | str` is not assignable to `str`.
- `dnallm/inference/inference.py:474:13` — Object of type `list[Unknown] | Dataset | Unknown` is not assignable to attribute `sequences` of type `list[str]`.
- `dnallm/inference/inference.py:666:22` — Object of type `MappingProxyType[str, Parameter]` is not assignable to `dict[Unknown, Unknown] | None`.
- `dnallm/inference/inference.py:2066:17` — Invalid subscript assignment with key of type `int` and value of type `NDArray[Unknown]` on object of type `list[list[Unknown]]`.
- `dnallm/inference/plot.py:721:25` — Object of type `Unknown | LayerChart | FacetChart | Chart` is not assignable to `Chart`.
- `dnallm/inference/plot.py:723:9` — Object of type `Unknown | HConcatChart | ConcatChart` is not assignable to `Chart`.
- `dnallm/models/model.py:1061:22` — Object of type `str | list[str] | None | list[None | Unknown] | list[None | str]` is not assignable to `str`.
- `dnallm/models/special/enformer_model/modeling_space.py:108:9` — Cannot assign to read-only property `heads` on object of type `Self@__init__`.
- `dnallm/utils/logger.py:113:40` — Object of type `dict[str, str | int]` is not assignable to `dict[str, str]`.

## error[missing-argument] — 16 diagnostic(s)

Dynamic-attribute calls ty cannot bind (`EvaluationModule.compute` reached through attribute lookup) — bind through a typed reference or annotate the module object.

- `dnallm/inference/benchmark.py:521:39` — No argument provided for required parameter `y` of bound method `StratifiedKFold.split`.
- `dnallm/tasks/metrics.py:181:17` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:192:17` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:215:19` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:216:19` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:217:18` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:218:25` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:542:18` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:610:18` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:611:24` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:623:29` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:626:26` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:627:22` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:628:23` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:629:31` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.
- `dnallm/tasks/metrics.py:634:31` — No argument provided for required parameter `self` of function `EvaluationModule.compute`.

## error[invalid-return-type] — 9 diagnostic(s)

Declared return types that do not cover what the function actually returns (Dataset vs DatasetDict unions, implicit None paths). Align the annotation with reality.

- `dnallm/datahandling/data.py:248:16` — Return type does not match returned value.
- `dnallm/datahandling/data.py:290:28` — Return type does not match returned value.
- `dnallm/inference/benchmark.py:567:10` — Function can implicitly return `None`, which is not assignable to return type `tuple[Chart | dict[str, Chart], Chart | dict[str, Chart]]`.
- `dnallm/inference/benchmark.py:630:20` — Return type does not match returned value.
- `dnallm/inference/mutagenesis.py:308:16` — Return type does not match returned value.
- `dnallm/inference/mutagenesis.py:346:16` — Return type does not match returned value.
- `dnallm/inference/plot.py:732:16` — Return type does not match returned value.
- `dnallm/mcp/model_manager.py:188:16` — Return type does not match returned value.
- `dnallm/tasks/metrics.py:231:16` — Return type does not match returned value.

## error[call-non-callable] — 5 diagnostic(s)

trainer.py's model-typed-as-Tensor cluster: an attribute assigned both nn.Module-ish and Tensor values. Split the attribute or annotate the union, then narrow at the call.

- `dnallm/finetune/trainer.py:216:13` — Object of type `Tensor` is not callable.
- `dnallm/finetune/trainer.py:405:21` — Object of type `Tensor` is not callable.
- `dnallm/finetune/trainer.py:413:17` — Object of type `Tensor` is not callable.
- `dnallm/finetune/trainer.py:471:21` — Object of type `Tensor` is not callable.
- `dnallm/finetune/trainer.py:479:17` — Object of type `Tensor` is not callable.

## error[no-matching-overload] — 5 diagnostic(s)

torch overloads rejected due to argument-type unions; fix the upstream union first (most vanish with the not-subscriptable/None fixes).

- `dnallm/inference/inference.py:2089:21` — No overload of function `stack` matches arguments.
- `dnallm/inference/inference.py:2091:26` — No overload of function `concatenate` matches arguments.
- `dnallm/mcp/model_manager.py:166:28` — No overload of function `sum` matches arguments.
- `dnallm/models/special/enformer_model/modeling_enformer.py:124:12` — No overload of function `linspace` matches arguments.
- `dnallm/models/special/enformer_model/modules.py:115:12` — No overload of function `linspace` matches arguments.

## error[unsupported-operator] — 3 diagnostic(s)

`in`/`**` applied to object/unknown receivers; annotate receiver types (enformer linspace result indexing follows the same root cause).

- `dnallm/inference/inference.py:670:12` — Unsupported `in` operation.
- `dnallm/models/special/enformer_model/modeling_enformer.py:128:19` — Unsupported `**` operation.
- `dnallm/models/special/enformer_model/modules.py:119:19` — Unsupported `**` operation.

## error[not-iterable] — 3 diagnostic(s)

Iteration over None/object-typed values; same None-guard pattern.

- `dnallm/inference/benchmark.py:159:22` — Object of type `object` is not iterable.
- `dnallm/mcp/tests/_network_skip.py:37:20` — Object of type `object` is not iterable.
- `dnallm/models/special/enformer_model/data.py:126:36` — Object of type `None` is not iterable.

## error[unknown-argument] — 1 diagnostic(s)

REAL-BUG CANDIDATE: `torch.dot(..., trans_b=...)` would raise TypeError at runtime if executed — verify the call path and correct the API usage.

- `dnallm/models/special/dnabert2.py:30:26` — Argument `trans_b` does not match any known parameter of function `dot`.

## error[invalid-parameter-default] — 1 diagnostic(s)

Default None for a non-Optional ndarray parameter; make the parameter `| None` (annotation-only).

- `dnallm/inference/mutagenesis.py:418:43` — Default value of type `None` is not assignable to annotated parameter type `ndarray[_AnyShape, dtype[Any]]`.

## error[invalid-method-override] — 1 diagnostic(s)

__getitem__ override signature diverges from the base; align key/return types (annotation-level).

- `dnallm/models/special/enformer_model/data.py:226:9` — Invalid override of method `__getitem__`.

## warning[deprecated] — 2 diagnostic(s)

torch.amp custom_fwd deprecation warnings in the vendored enformer modules; track upstream torch migration, low priority.

- `dnallm/models/special/enformer_model/modules.py:10:28` — The function `custom_fwd` is deprecated.
- `dnallm/models/special/enformer_model/modules.py:393:6` — The function `custom_fwd` is deprecated.

## warning[unsupported-base] — 1 diagnostic(s)

Class base ty cannot model (CrossDNA remote-code base) — annotation-visibility issue, not a runtime problem.

- `dnallm/models/special/crossdna.py:106:45` — Unsupported class base.

## .venv / tooling noise

The final run contains ZERO diagnostics located in `.venv` dependency files. The four
torch/triton dependency-file hits seen at planning time disappeared once the vendored
paths were excluded (ty no longer traverses them into torch internals). The 11 remaining
`.venv` arrow lines in the raw output are `info:` notes attached to first-party
diagnostics ("Matching overload defined here" pointers), not diagnostics themselves —
nothing to configure, nothing to fix in dependencies.

## Parked from Tasks 1–2 (judgment calls, not fixed here)

1. **MambaCache `torch.zeros(None, ...)` latent bug** (transformers_compat.py:693/:700):
   honest typing revealed `self.max_batch_size: int | None` reaching `torch.zeros` —
   upstream-inherited behavior, left visible per the plan's D3 rule (no behavior fixes in
   the baseline task).
2. **TypedDict -> bare `dict` consumer friction** (5 CLI sites, listed above): ty rejects
   what mypy accepts; consumers were intentionally left untouched (plan decision — zero
   consumer edits). Next batch: `Mapping[str, Any]` parameters or adopt `DNALLMConfig`.
3. **Raise-on-use optional-dep sites lack friendly ImportError messages**
   (model.py:72 lucagplm head branch; enformer data.py pyfaidx/polars; inference.py
   evo generate/score_sequences): function-local imports in absence-gated branches —
   suppressed as optional-dep by design, but a raw ImportError surfaces instead of the
   guarded handlers' helpful message. Cosmetic next-batch hardening.
4. **transformers_compat.py:552 `transformers.configuration_utils.PretrainedConfig`**:
   attribute access (not a `from transformers import`) that resolves both statically and
   at runtime — left as-is; the six rename sites were the plan's exact scope.
5. **mypy advisory lane is blocked from checking first-party code**: the configured run
   (`python_version = "3.10"`) dies on numpy's 3.12+ `type` statements in
   `numpy/__init__.pyi` before reaching any first-party file, so all 38 reported errors
   sit in vendored `dnallm/tasks/metrics/`. A supplementary `--python-version 3.13` run
   checks 191 first-party files (259 errors) and confirms ~19 formerly-ignore-masked
   mismatches in plot.py / interpret.py / mutagenesis.py / benchmark.py / trainer.py /
   data.py / modeling_space.py now surface (all also in this ty list). If the mypy lane
   is ever unblocked, expect those to appear in the advisory report.

