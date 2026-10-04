# 08-CENSUS-ROLLUP — D-03 Dev-Box Reconciliation Baseline (08-02, Task 3)

**Purpose.** D-03 requires a per-item full reconciliation baseline on the dev box after the
08-02 repairs: every census item (21 notebooks + 3 marimo apps + 1 script + the YAML leg)
with an honest state, plus the full fast lane. Every later repair plan (08-04..08-08) and
the final census (08-09) reconcile against THIS table.

**Execution environment** (2026-10-04 20:38-23:42 local): project venv (Python 3.13.15,
transformers 5.17.0, torch 2.11.0+cu130, IPython 8.39.0 post-08-02-fix), pristine
re-fetched NT ModelScope snapshot (see 08-D17-DISPOSITION.md), bedtools v2.31.1,
rice inputs seeded from the 05-06 census cache after the rice.uga.edu outage (see repairs).

**Lane composition.** The full lane ran as two passes plus two targeted re-runs because
(a) the first pass hit a 2h wall-clock cap mid `embedding_attention` (killed by the
executor's background limit, not by any test), and (b) two REPAIR-01 fixes landed mid-lane
(rice cache pre-seed; combined-notebook sibling seeding). Affected items were re-run to
green post-fix; every row below records its final post-repair state with the en-route
failure noted. Logs: /tmp/08_03_examples_full.log (pass A), /tmp/08_03_examples_rest.log
(pass B), /tmp/08_03_combined.log (combined re-run), /tmp/08_02_script.log (script),
/tmp/08_02_nt_verify.log (NT family), /tmp/08_03_fastlane.log (fast lane).

## Per-item census state

| item | lane | state | evidence / notes |
| --- | --- | --- | --- |
| notebooks/inference/inference.ipynb | ACTIVE | PASS | pass A dot 1 |
| notebooks/generation/inference.ipynb | ACTIVE | PASS | pass A; also in NT-family verify 18:11 |
| notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb | ACTIVE | PASS | pass A |
| notebooks/interpretation/interpretation.ipynb | ACTIVE | PASS | pass A |
| notebooks/data_prepare/predict/predict_data.ipynb | ACTIVE | PASS | pass A (0 code cells) |
| notebooks/finetune_binary/finetune_binary.ipynb | ACTIVE | PASS | pass A |
| notebooks/finetune_multi_labels/finetune_multi_labels.ipynb | ACTIVE | PASS | pass A (~24 min training) |
| notebooks/finetune_NER_task/data_generation_and_inference.ipynb | ACTIVE | PASS (post-repair) | pass A FAIL (rice.uga.edu served ~8KB/s; wget cell hit the 3600s cell timeout) → repaired (census-cache pre-seed; `wget -c` skips complete files) → pass B green |
| notebooks/benchmark/benchmark.ipynb | ACTIVE | PASS | pass A; D-17 pristine-cache proof (18:11 verify) |
| notebooks/data_prepare/finetune/finetune_data.ipynb | ACTIVE | PASS | pass A |
| notebooks/embedding_attention.ipynb | ACTIVE | PASS | killed mid-run by the executor's 2h cap in pass A (not a test failure; passed standalone 18:10 in 29.29s) → pass B green |
| notebooks/finetune_NER_task/finetune_NER_task.ipynb | ACTIVE | PASS | pass B (full 3-epoch training; standalone 33:20) |
| notebooks/inference_for_tRNA/inference.ipynb | ACTIVE | PASS | pass B |
| mcp_example/mcp_client_ollama_langchain_agents.ipynb | GATED (`_gate_ollama_stack`) | typed-skip-with-evidence | pass B `ssssssss` batch — endpoint probe state at run time carried in the skip message (261003-csd contract) |
| mcp_example/mcp_client_ollama_pydantic_ai.ipynb | GATED (`_gate_ollama_stack`) | typed-skip-with-evidence | same batch |
| notebooks/generation_evo_models/inference.ipynb | GATED (`_gate_evo`) | typed-skip-with-evidence | optional-dep (stripedhyena/evo2 absent by design) — family plan 08-04/08-05 un-gates |
| notebooks/generation_megaDNA/inference.ipynb | GATED (`_gate_megadna`) | typed-skip-with-evidence | optional-dep — family plan un-gates |
| notebooks/finetune_custom_head/finetune.ipynb | GATED (`_gate_megadna`) | typed-skip-with-evidence | optional-dep — family plan un-gates |
| notebooks/finetune_generation/finetune_generation.ipynb | GATED (`_gate_megadna`) | typed-skip-with-evidence | optional-dep — family plan un-gates |
| notebooks/lora_finetune_inference/lora_finetune.ipynb | GATED (`_gate_mamba`) | typed-skip-with-evidence | optional-dep (mamba_ssm absent) — family plan 08-06/08-07 un-gates |
| notebooks/lora_finetune_inference/lora_inference.ipynb | GATED (`_gate_mamba`) | typed-skip-with-evidence | optional-dep — family plan un-gates |
| marimo/inference/inference_demo.py | marimo | PASS | pass A (3 dots) |
| marimo/finetune/finetune_demo.py | marimo | PASS | pass A |
| marimo/benchmark/benchmark_demo.py | marimo | PASS | pass A |
| notebooks/finetune_NER_task/generate_bpe_dataset.py | script | PASS (healed, EXEC-04) | `4 passed in 37.91s` — artifact 11,299,268 B fresh in-sandbox; marker never fired (08-D17-DISPOSITION.md) |
| plant_helixseek CRE/Anno showcase notebooks | showcase lane | PASS | pass B showcase dots (structure + nightly exec + bands) |
| plant_helixseek_combined.ipynb (display figure) | showcase lane | PASS (post-repair) | pass B FAIL (`FileNotFoundError ../plant_helixseek_cre/data/chr1_5100001_5300000.fas` — fresh-sandbox sibling gap; the 08-01 runner failure reproduced) → repaired (COMBINED_EXTRA_INPUTS sibling seeding + contract tests) → targeted re-run green (`1 passed in 197.54s`, /tmp/08_03_combined.log) |
| YAML leg (21 configs) | fast YAML | PASS | `python3 scripts/validate_yaml.py` → "All YAML files passed validation." (21/21); tests/configuration/test_yaml_load.py 21 passed |

## Full fast lane (EXEC-05)

`.venv/bin/python -m pytest tests/ -m "not slow" -q` → **1761 passed, 1 skipped, 53
deselected, exit 0, 93.75s** (/tmp/08_03_fastlane.log). Baseline comparison: 08-01 recorded
1752 passed / 1 pre-existing skip; the +9 are exactly this plan's new fast tests (3
TestNotebookExtraMembers/TestNotebookKernelPlotCompat in tests/test_extras_guard.py, 3
TestRiceInputSeeding, 3 TestRiceCacheExtras). **Zero new skips** — the single skip is the
pre-existing baseline one.

## Repairs landed by this baseline (REPAIR-01)

| failure | triage | fix | commit |
| --- | --- | --- | --- |
| embedding_attention cell 9 ImportError backend2gui (IPython 9.17.1 × matplotlib 3.8.4) | library/declared-dep conflict (pgt pins mpl<3.9) | ipython>=8.31,<9 in notebook extra + static/dynamic guards | 439370f |
| rice.uga.edu hard-down 48 min then ~8KB/s half-up | infrastructure (input host) | script lane: census-cache-first `_seed_rice_input` + 3 unit tests; notebook lane: `_rice_cache_extras()` pre-seed for data_generation (`wget -c` skips complete files) + 3 unit tests | e90ff21 + Task 3 commit |
| combined showcase notebook FileNotFoundError on fresh sandbox | harness/content (missing sibling seeding — the 08-01 runner failure class) | `COMBINED_EXTRA_INPUTS` seeds the CRE zoom FASTA sibling; `TestCombinedSiblingSeeding` contract tests pin every `../` ref covered + every source committed | Task 3 commit |
| NER notebook run-1 kernel death (no artifacts, non-reproducing) | transient environment (owner's live JupyterLab kernels contending) | none — isolated re-run + pass B both green | — |

## Family view — what remains gated (plans 08-04..08-08 own the un-gating)

| family | items | gate | owning plan |
| --- | --- | --- | --- |
| evo (evo-1 / evo2) | generation_evo_models | optional-dep (stripedhyena + evo2) | evo family plan |
| megaDNA | generation_megaDNA, finetune_custom_head, finetune_generation | optional-dep (megaDNA + MEGABYTE_pytorch) | megaDNA family plan |
| PlantCAD / mamba | lora_finetune, lora_inference | optional-dep (mamba_ssm native build) | mamba/lora family plan |
| mcp + ollama | langchain + pydantic_ai client notebooks | endpoint-up probes (execute when both up) | mcp batch plan / runner infra |

Every gated skip in this baseline is an honest `optional-dep:`/`network-unavailable:` typed
skip carrying live probe evidence — the CURRENT-STATE baseline per D-03, not a failure.
