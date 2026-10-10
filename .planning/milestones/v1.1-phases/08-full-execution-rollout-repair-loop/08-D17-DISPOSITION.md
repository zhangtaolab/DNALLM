# D-17 Disposition Ledger — NT-REMOTE-STRUCTURAL Family (08-02, Task 1)

**Decision D-17** (08-CONTEXT): the NT-REMOTE-STRUCTURAL family disposition must be recorded
per item ("逐项记账"): shim depth evaluated first, repairable variants healed via deeper
shims, evidence-backed typed skip only when unrepairable. This ledger is that per-item
record; it closes the 05-04 hand-off and the Pitfall-1 evidence-poison (the hand-patched
dev-box ModelScope snapshot) that made every earlier local NT-family green unproven.

**Verdict summary: every item is shim-covered.** The 9-patch absence-gated layer in
`dnallm/utils/transformers_compat.py` (`apply_patches`, lines 1024-1034) closes every NT
rung on a PRISTINE snapshot — no cache patching, no typed skip needed for any of the four
items. The 05-04 "not vendored-pure-helper territory" termination is superseded exactly as
the sl7 owner instruction recorded.

## Per-item disposition (3 census items + script lane)

| item | census id / lane | model | exact failure signature the shim closes | closing shim(s) | verdict | evidence (test output) |
| --- | --- | --- | --- | --- | --- | --- |
| benchmark notebook | `notebooks/benchmark/benchmark.ipynb` (ACTIVE, 3rd model leg) | zhangtaolab/nucleotide-transformer-v2-100m-promoter (modelscope) | (a) `AttributeError: 'EsmConfig' object has no attribute 'is_decoder'` (remote modeling_esm.py:335/584-585 reads removed 4.x `PretrainedConfig` defaults); (b) remote `self.init_weights()` 4.x bookkeeping (modeling_esm.py:1124/1245/1345 — the second target the 05-06 hand patch rewrote to `post_init`); (c) forward-stage `get_extended_attention_mask` | `_patch_pretrained_config_legacy_defaults` (transformers_compat.py:541, closed map 507-510), `_patch_legacy_init_weights_bookkeeping` (:985), `_patch_get_extended_attention_mask` (:455) | **shim-covered** | `test_notebook_executes_end_to_end[notebooks/benchmark/benchmark.ipynb] PASSED` — pristine-cache run 2026-10-04 (run 1 of the Task-1 selection, ~16:48-16:52 local) and again in the final one-pass verify |
| NER finetune notebook | `notebooks/finetune_NER_task/finetune_NER_task.ipynb` (ACTIVE) | zhangtaolab/plant-nucleotide-transformer-BPE (modelscope) | `ValueError: Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'` (via dnallm/models/model.py:888 — the 05-06 census terminal; same remote-code shape as the promoter mirror) | `_patch_pretrained_config_legacy_defaults` (:541) | **shim-covered** | isolated re-run `1 passed in 2000.97s (0:33:20)` — full 3-epoch NER training + eval by real execution on shims alone (/tmp/08_02_ner_rerun.log, 2026-10-04 17:16-17:57) |
| embedding_attention notebook | `notebooks/embedding_attention.ipynb` (ACTIVE) | InstaDeepAI/nucleotide-transformer-v2-50m-multi-species (raw HF AutoModelForMaskedLM, trust_remote_code — NOT the dnallm loader) | `ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils'` (InstaDeepAI remote modeling_esm.py:40 — the 05-06 census pytorch_utils variant) | `_patch_remote_code_pruning_helpers` (:344) attaching to BOTH `modeling_utils` AND `pytorch_utils` (:373-376) | **shim-covered** | re-run `1 passed in 29.29s` — real forward + embedding/attention extraction + plots (/tmp/08_02_embedding_rerun.log, 2026-10-04 18:10) |
| script lane (EXEC-04) | `notebooks/finetune_NER_task/generate_bpe_dataset.py` (script lane) | zhangtaolab/plant-nucleotide-transformer-BPE (tokenizer+model via dnallm loader) | same `'EsmConfig' object has no attribute 'is_decoder'` load rung (marker at tests/examples/test_script_execution.py:43); typed skip was the 05-05 self-healing probe | `_patch_pretrained_config_legacy_defaults` (:541) | **shim-covered — healed to real green** (Task 2) | `test_generate_bpe_dataset_produces_artifact` PASSED by real execution twice (`4 passed in 37.91s` / `37.92s`, /tmp/08_02_script.log): fresh in-sandbox `rice_gene_ner_BPE.pkl` 11,299,268 B (committed baseline 12,067,286 B), artifact-existence/size/fresh-mtime assertions all reached; no SKIPPED line names generate_bpe_dataset (verify gate exit 0). The typed-skip marker conversion is unchanged dead-fallback code — it never fired |

## Pristine-restore proof (Pitfall 1 closed)

The 05-06 orchestrator probe had hand-patched the dev-box ModelScope snapshot
(`config.json` + `modeling_esm.py`, backups left as `*.dnallm-bak`), so every earlier local
benchmark-NT green was partly the hand patch, not the shims. Restore chain, 2026-10-04:

1. **Hand patch characterized before restore** (what the shims must now carry):
   - `config.json` patched copy added `"is_decoder": false` + `"add_cross_attention": false`
     (sha256 `cd81c54d…9896`, 1271 B) vs pristine backup (sha256 `6e57c3ef…c482`, 1217 B)
   - `modeling_esm.py` patched copy rewrote 3 sites `self.init_weights()` → `self.post_init()`
     (sha256 `20176a69…651`, 58175 B) vs pristine (sha256 `f446381c…874`, 58184 B)
2. **Restore:** `mv` each `.dnallm-bak` back over its target; gate `ls … | grep -c dnallm-bak`
   → **0** (also re-checked after the re-download below).
3. **Reproducibility proof:** deleted the snapshot dir entirely and re-fetched through
   dnallm's own loader path — `download_model("zhangtaolab/nucleotide-transformer-v2-100m-promoter",
   downloader=modelscope snapshot_download)` — "Download model … successfully", 9 files,
   365M safetensors. Re-fetched hashes byte-identical to the pristine backups:
   `config.json` `6e57c3ef76c0955f3712e3c81932aa22caa38bd973af2d47504bd0655e61c482`,
   `modeling_esm.py` `f446381cf54dd46cacc60df9fafac00475a65c39ce688cbb3b7cfad344aa6874`.
   Pitfall-1 warning sign ("green not reproducible after rm -rf of the cached snapshot")
   does not hold: all executions below ran against the re-fetched pristine snapshot.

## Repairs en route (REPAIR-01 triage)

| failure surfaced during re-execution | triage | fix | commit |
| --- | --- | --- | --- |
| `embedding_attention.ipynb` cell 9 `ImportError: cannot import name 'backend2gui' from 'IPython.core.pylabtools'` | **library / declared-dependency conflict** — `pygenometracks>=3.9` (notebook extra, landed 261004-dyw 11:35) pins `matplotlib<3.9`, whose `install_repl_displayhook` imports the IPython name removed in IPython 9; NOT an NT rung (model load + forward had already passed) | `ipython>=8.31,<9` declared in the notebook extra (pyproject.toml) + static/dynamic guards in `tests/test_extras_guard.py` (RED proven on the broken pair pre-fix); dev venv → IPython 8.39.0 | `439370f` (08-02) |
| run 1 of the selection: `finetune_NER_task` FAILED ~2 min after training start, no partial artifacts (non-CellExecutionError shape) | **transient environment** — did not reproduce on an isolated re-run (33:20 full-training PASS); box was concurrently running the owner's 7 live JupyterLab kernels; kernel-death-class failures leave no artifact by harness contract | none (non-reproducing); recorded here for honesty | — |
| script lane blocked: `rice.uga.edu` hard-down 19:00-19:50 (48-min probe window, 24/24 fails), then half-up at ~8KB/s — the documented input URLs unfetchable for a fresh per-run download | **infrastructure** (input-host outage; not a script/library defect) | `_seed_rice_input` in tests/examples/test_script_execution.py: census-cache-first seeding over the gitignored `.scratch/census-out/inputs/` mirror (the 05-06 campaign's own cache; byte-sizes identical to the 18:11 server download), cold cache keeps the URL download + honest `network-unavailable:` skip, 4xx still re-raises (WR-04); 3 network-free unit tests pin the contract | Task 2 commit (`test(08-02)`) |

## Gate

Task 1 verify (exit-code enforced): zero `dnallm-bak` remnants in the live cache dir AND
`pytest tests/examples/test_notebook_execution.py -k "benchmark or ner or embedding" -q`
exit 0 with no skip among the three NT-family tests. Final one-pass result:
see `## Final one-pass verify` below (filled after the clean-pass run completed).

## Final one-pass verify

2026-10-04 18:11-18:49 local, project venv (shims active, IPython 8.39.0), pristine
re-fetched snapshot:

```
$ ls ~/.cache/modelscope/hub/models/zhangtaolab/nucleotide-transformer-v2-100m-promoter/ | grep -c dnallm-bak
0
$ .venv/bin/python -m pytest "tests/examples/test_notebook_execution.py" -k "benchmark or ner or embedding" -q -rs
...
SKIPPED [1] ... optional-dep: execute notebooks/generation_evo_models/inference.ipynb (find_spec('stripedhyena') is None; find_spec('evo2') is None)
SKIPPED [1] ... optional-dep: execute notebooks/generation_megaDNA/inference.ipynb (find_spec('megaDNA') is None; find_spec('MEGABYTE_pytorch') is None)
SKIPPED [1] ... optional-dep: execute notebooks/finetune_generation/finetune_generation.ipynb (find_spec('megaDNA') is None; find_spec('MEGABYTE_pytorch') is None)
=========== 5 passed, 3 skipped, 27 deselected in 2164.31s (0:36:04) ============
PYTEST_EXIT=0
```

The 5 passes are generation (substring `ner` in "generation"), data_generation_and_inference,
**benchmark**, **embedding_attention**, **finetune_NER_task** — all three NT-family tests green by
real execution; the 3 skips are all sanctioned `optional-dep:` gated families (evo/megaDNA
prerequisites deliberately absent from the project venv — the D-03 current-state baseline),
none in the NT family. Log: /tmp/08_02_nt_verify.log.
