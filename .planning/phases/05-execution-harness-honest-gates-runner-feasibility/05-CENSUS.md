# Phase 5 Census Inventory — example/ Full-Tree Execution (D-08)

**Purpose.** Owner ruling 2026-10-02 (D-08, post-closure gap closure): the ENTIRE `example/`
tree executes — every notebook, marimo app, script and program — with nothing silently
omitted. This committed inventory is the acceptance artifact: every executable item (Table A)
and every input/config file (Table B) carries a lane and a verdict; the 05-06 campaign fills
each verdict with either a real-execution result or an evidence-backed typed skip. Table C
proves the tables cover the whole live tree.

**Execution environment** (dev box, GB10 class per D-09): NVIDIA GB10 (compute capability
12.1, driver 580.178.04, CUDA 13.0) — same hardware class as the `[self-hosted, dnallm-nightly]`
runner; project venv Python 3.13.15, torch 2.11.0+cu130, transformers 5.17.0, marimo 0.25.0,
nbclient 0.11.0, bedtools v2.31.1. Official GB10-runner confirmation stays post-merge
sequenced per D-04.

**Verdict vocabulary.** `PASS` (real execution, exit clean, artifacts produced) / `FAIL`
(execution error; feeds the repair backlog) / typed skip with a registered prefix from
`tests/expected_skips.yaml` — never a bare skip, never a silent omission.

**Ladder rule (D-05/D-06).** For every model-gated item: execute the notebook variant first;
on failure, fall back to the family's smallest viable variant; only when both fail record an
evidence-backed typed skip. Never skip-first.

**Typed-skip prefix assignment.** Follow the 05-FEASIBILITY.md prefix-assignment table:
evo2 / megaDNA families → `optional-dep:` (install-gated); pyBigWig →
`environment-unavailable:` (box toolchain); evo-1 and marimo → no skip expected. Ollama-gated
`mcp_example` items use the `network-unavailable:` prefix (MCP-01 convention; registered).
GAP-1-class NT remote-code items (benchmark third model; `zhangtaolab/plant-nucleotide-
transformer-BPE`) → `environment-unavailable:` carrying the exact traceback; owner
disposition pending per the 05-04 hand-off.

**Standing constraints.** Never recreate `tests/examples/conftest.py` (module-local fixtures;
owner-acknowledged override in 05-VERIFICATION.md frontmatter). One-off census/probe scripts
live in gitignored `.scratch/` — never committed (owner rule). dev+main are read-only for this
milestone; all work lands on `phs`; pushes are manual-only (D-09).

## Table A — Executable items (the census proper)

Item paths are POSIX, relative to `example/` — the same key form the 05-06 census manifest
uses. Budgets cite the specs dicts in `tests/examples/_execution.py`: notebooks as
`NOTEBOOK_EXEC_SPECS` cell/test seconds, the seeded marimo app as `MARIMO_EXEC_SPECS`
timeout/test seconds, the script as the `run_example_script` default timeout under its class
mark. Every verdict starts `pending`; 05-06 fills it.

| item path | class | model references | budget (cell/test) | durable lane | verdict | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| notebooks/inference/inference.ipynb | notebook / inference (pilot) | zhangtaolab/plant-dnagpt-BPE-promoter (modelscope) | 600/1800 | notebook execution test | pending | — |
| notebooks/inference_for_tRNA/inference.ipynb | notebook / inference | zhangtaolab/tRNADetector, zhangtaolab/tRNAPointer (configs) | 900/2700 | notebook execution test | pending | — |
| notebooks/generation/inference.ipynb | notebook / generation | zhangtaolab/plant-dnagpt-BPE | 900/2700 | notebook execution test | pending | — |
| notebooks/generation_evo_models/inference.ipynb | notebook / generation | togethercomputer/evo-1-131k-base, arcinstitute/evo2_1b_base | 1800/3600 | notebook execution test | pending | FEAS-01 verdicts: evo-1 FEASIBLE(small-variant 8k), evo2 FEASIBLE(noFP8 config) — enabling conditions in 05-FEASIBILITY.md |
| notebooks/generation_megaDNA/inference.ipynb | notebook / generation | lingxusb/megaDNA_updated | 900/2700 | notebook execution test | pending | FEAS-01: FEASIBLE(notebook-variant) given pinned clone + MEGABYTE_pytorch; generate leg has a known dnallm tokenizer error (Phase 8 repair candidate) |
| notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb | notebook / mutagenesis | zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast | 900/2700 | notebook execution test | pending | — |
| notebooks/interpretation/interpretation.ipynb | notebook / interpretation | zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf | 900/2700 | notebook execution test | pending | — |
| notebooks/embedding_attention.ipynb | notebook / embeddings+attention | InstaDeepAI/nucleotide-transformer-v2-50m-multi-species (raw HF AutoModelForMaskedLM, trust_remote_code=True — not the dnallm loader) | 900/2700 | notebook execution test | pending | — |
| notebooks/data_prepare/finetune/finetune_data.ipynb | notebook / data preparation | zhangtaolab/plant-dnabert-BPE; dataset zhangtaolab/plant-multi-species-core-promoters | 1800/3600 | notebook execution test | pending | — |
| notebooks/data_prepare/predict/predict_data.ipynb | notebook / data preparation (markdown-only: zero code cells) | — | 900/2700 | notebook execution test | pending | — |
| notebooks/finetune_binary/finetune_binary.ipynb | notebook / finetune (binary) | zhangtaolab/plant-dnabert-BPE; dataset plant-multi-species-core-promoters | 3600/7200 | notebook execution test | pending | — |
| notebooks/finetune_custom_head/finetune.ipynb | notebook / finetune (custom head) | zhangtaolab/plant-dnagpt-BPE (megaDNA referenced as alternative) | 3600/7200 | notebook execution test | pending | — |
| notebooks/finetune_generation/finetune_generation.ipynb | notebook / finetune (generation) | lingxusb/megaDNA_updated, zhangtaolab/plant-dnagpt-singlebase; input ath_cds.csv + pyfastx .fxi | 3600/7200 | notebook execution test | pending | — |
| notebooks/finetune_multi_labels/finetune_multi_labels.ipynb | notebook / finetune (multi-label) | zhangtaolab/plant-dnagpt-BPE; input maize_test.tsv | 3600/7200 | notebook execution test | pending | — |
| notebooks/finetune_NER_task/data_generation_and_inference.ipynb | notebook / dataset generation + inference | zhangtaolab/plant-dnagpt-6mer; rice inputs downloaded from rice.uga.edu (documented URLs, cell 9) | 3600/7200 | notebook execution test | pending | — |
| notebooks/finetune_NER_task/finetune_NER_task.ipynb | notebook / finetune (token NER) | zhangtaolab/plant-nucleotide-transformer-BPE; consumes rice_gene_ner_BPE.pkl | 3600/7200 | notebook execution test | pending | GAP-1-class: same checkpoint proven unloadable on transformers 5.17 by the script lane (05-05); ladder disposition 05-06 |
| notebooks/lora_finetune_inference/lora_finetune.ipynb | notebook / LoRA finetune | dataset plant-multi-species-core-promoters (model via config/dropdown) | 3600/7200 | notebook execution test | pending | — |
| notebooks/lora_finetune_inference/lora_inference.ipynb | notebook / LoRA inference | kuleshov-group/PlantCAD2-Small-l24-d0768 (huggingface source) | 1800/3600 | notebook execution test | pending | — |
| notebooks/benchmark/benchmark.ipynb | notebook / benchmark (3 models) | Plant DNABERT, Plant DNAGPT, Nucleotide Transformer (= zhangtaolab/nucleotide-transformer-v2-100m-promoter) via benchmark_config.yaml | 3600/7200 | notebook execution test | pending | 05-04 flag: third model = census FAIL row (remote modeling_esm.py needs removed 4.x PretrainedConfig defaults after the import shim); owner disposition pending (3 options in 05-04-SUMMARY) |
| mcp_example/mcp_client_ollama_langchain_agents.ipynb | notebook / MCP client (ollama) | ollama qwen3.8:latest via local MCP server (langchain agents) | 600/1800 | notebook execution test | pending | ollama-gated; network-unavailable prefix convention if gated |
| mcp_example/mcp_client_ollama_pydantic_ai.ipynb | notebook / MCP client (ollama) | ollama qwen3.8:latest via local MCP server (pydantic-ai) | 600/1800 | notebook execution test | pending | ollama-gated; network-unavailable prefix convention if gated |
| marimo/inference/inference_demo.py | marimo app / inference (export-html) | xlsx-driven dropdowns; defaults resolve to zhangtaolab/plant-dnabert-BPE-open_chromatin (modelscope, observed in the 05-05 live export) | 1200/1500 | marimo execution test | pending | lane proven green 05-05 (67KB HTML artifact, model cell executed on cuda) |
| marimo/finetune/finetune_demo.py | marimo app / finetune (export-html) | zhangtaolab/plant-dnagpt-BPE; dataset plant-multi-species-core-promoters | to be specced by 05-06 (MARIMO_EXEC_SPECS) | marimo execution test | pending | — |
| marimo/benchmark/benchmark_demo.py | marimo app / benchmark (export-html) | zhangtaolab/plant-dnabert-BPE-promoter, plant-dnagpt-BPE, plant-dnagpt-BPE-promoter (config.yaml + test.csv) | to be specced by 05-06 (MARIMO_EXEC_SPECS) | marimo execution test | pending | — |
| notebooks/finetune_NER_task/generate_bpe_dataset.py | script / dataset generation | zhangtaolab/plant-nucleotide-transformer-BPE (tokenizer+model via dnallm loader, modelscope); rice genome + GFF3 inputs (downloaded to sandbox) | 3000/3600 | script execution test | pending | 05-05: GAP-1-class environment-unavailable typed skip recorded (exact traceback in the skip); annotation-bed write block restored from the notebook (Rule 3); self-heals when the env gap closes |

## Table B — Input and config inventory (non-executable content)

Every OTHER file under `example/` — the nothing-silently-omitted accounting for content that
is not itself executed. "Fast YAML leg" = `scripts/validate_yaml.py` +
`tests/configuration/test_yaml_load.py`; "structural lane" = `tests/examples/test_examples.py`;
"docs mirror" = `scripts/check_docs_sync.py`.

| file | role | existing coverage lane |
| --- | --- | --- |
| marimo/benchmark/config.yaml | config input (benchmark_demo app) | fast YAML leg; structural lane; docs mirror |
| marimo/benchmark/.gitignore | app-local ignore hygiene (runtime artifacts) | repo hygiene (accounted; no execution lane by design) |
| marimo/benchmark/test.csv | data input (benchmark_demo prediction input) | structural lane |
| marimo/finetune/finetune_config.yaml | config input (finetune_demo app) | fast YAML leg; structural lane; docs mirror |
| marimo/inference/inference_config.yaml | config input (inference_demo app) | fast YAML leg; structural lane; docs mirror |
| marimo/inference/plant_DNA_LLMs_finetune_list.xlsx | model-registry data input (inference_demo dropdowns) | structural lane |
| notebooks/benchmark/benchmark_config.yaml | config input (benchmark notebook 3-model list) | fast YAML leg; structural lane; docs mirror |
| notebooks/benchmark/logs/dnallm.log | runtime log artifact (untracked on disk, gitignored) | none — v1 deferred item (import-time log sink); not census-executable |
| notebooks/data_prepare/finetune/finetune_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/data_prepare/finetune/logs/dnallm.log | runtime log artifact (untracked on disk, gitignored) | none — v1 deferred item (import-time log sink); not census-executable |
| notebooks/data_prepare/predict/inference_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/finetune_binary/finetune_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/finetune_custom_head/finetune_config.yaml | config input (carries head_config) | fast YAML leg; structural lane; docs mirror |
| notebooks/finetune_generation/Arabidopsis_thaliana.TAIR10.cds.all.fa.gz.fxi | pyfastx index artifact (genome companion) | structural lane; regenerated on execution |
| notebooks/finetune_generation/ath_cds.csv | data input (CDS sequences) | structural lane |
| notebooks/finetune_generation/finetune_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/finetune_multi_labels/maize_test.tsv | data input (multi-label test set) | structural lane |
| notebooks/finetune_multi_labels/multi_labels_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/finetune_NER_task/.gitignore | dir-local ignore hygiene (runtime artifacts) | repo hygiene (accounted) |
| notebooks/finetune_NER_task/ner_task_config.yaml | config input (NER task) | fast YAML leg; structural lane; docs mirror |
| notebooks/finetune_NER_task/rice_gene_ner_BPE.pkl | generated dataset artifact (input to finetune_NER_task notebook) | structural lane; regenerated in-sandbox by the script lane |
| notebooks/generation_evo_models/inference_evo_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/generation/generation_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/generation_megaDNA/inference_megaDNA_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/inference_for_tRNA/inference_model_config_tRNADetector.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/inference_for_tRNA/inference_model_config_tRNAPointer.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/inference/inference_config.yaml | config input (pilot) | fast YAML leg; structural lane; docs mirror |
| notebooks/inference/test.csv | data input (pilot sequences) | structural lane |
| notebooks/in_silico_mutagenesis/inference_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/interpretation/inference_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/lora_finetune_inference/finetune_config.yaml | config input (LoRA) | fast YAML leg; structural lane; docs mirror |
| notebooks/lora_finetune_inference/inference_config.yaml | config input | fast YAML leg; structural lane; docs mirror |
| notebooks/lora_finetune_inference/test.csv | data input | structural lane |
| notebooks/overview.md | docs (example-tree overview; DOCS_ONLY mirror relaxation) | docs mirror (both-sides .md must match byte-for-byte) |

## Table C — Completeness arithmetic

Counts at planning time (2026-10-02, plan 05-05):

| count | value | proven by |
| --- | --- | --- |
| Table A rows | 25 (21 notebooks + 3 marimo apps + 1 script) | commands below |
| Table B rows | 34 (21 YAML configs, 8 data inputs/artifacts, 2 .gitignore hygiene, 2 untracked logs, 1 overview.md) | commands below |
| git-tracked files under example/ | 57 | `git ls-files example \| wc -l` |
| untracked on-disk files | 2 at planning time (gitignored logs/dnallm.log runtime artifacts) — transient: a local notebook run can add one under any example dir; the live gates below are logs-tolerant | `git status --porcelain example` |
| on-disk files total | 59 at planning time = 25 + 34 (57 tracked + 2 logs) | `find example -type f \| wc -l` |

The nothing-silently-omitted gate: Table A + Table B enumerate every file the tree holds —
the 57 tracked files plus the known-transient gitignored log artifacts. Re-prove from the
repo root (section headers are assembled by concatenation so this verification block
cannot re-open the very awk ranges it documents):

```bash
CENSUS=.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md
SEC_A="## Table"" A"; SEC_B="## Table"" B"; SEC_C="## Table"" C"
# notebook rows == live tree (21)
test "$(find example -name '*.ipynb' | wc -l)" -eq "$(awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep -c 'ipynb')"
# marimo rows == live tree (3)
test "$(find example/marimo -name '*_demo.py' | wc -l)" -eq "$(awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep -c '_demo\.py')"
# exactly one script row
test "$(awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep -c 'generate_bpe_dataset\.py')" -eq 1
# every verdict starts pending (25 rows) — 05-06 flips these as verdicts land
test "$(awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep -c '| pending |')" -eq 25
# Table A data rows == 25; Table B data rows == 34
test $(( $(awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep -c '^| ') - 2 )) -eq 25
test $(( $(awk -v b="$SEC_B" -v c="$SEC_C" '$0~b,$0~c' $CENSUS | grep -c '^| ') - 2 )) -eq 34
# tracked baseline is exact
test "$(git ls-files example | wc -l)" -eq 57
# no untracked content outside the transient gitignored logs (nothing silently added)
test "$(find example -type f -not -path '*/logs/*' | wc -l)" -eq "$(git ls-files example | wc -l)"
# every Table A item exists on disk
awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep '^| ' | grep -v '^| item\|^| ---' | cut -d'|' -f2 | tr -d ' ' | while read -r p; do test -e "example/$p" || echo "MISSING: $p"; done
```

---

*Inventory committed by plan 05-05 (2026-10-02); verdicts to be filled by the 05-06 census campaign per D-08.*
