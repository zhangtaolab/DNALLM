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
| notebooks/inference/inference.ipynb | notebook / inference (pilot) | zhangtaolab/plant-dnagpt-BPE-promoter (modelscope) | 600/1800 | notebook execution test | PASS | 10.2s warm (7/7 code cells), 05-06 census re-run — .scratch/census-out/logs/notebooks__inference__inference.log |
| notebooks/inference_for_tRNA/inference.ipynb | notebook / inference | zhangtaolab/tRNADetector, zhangtaolab/tRNAPointer (configs) | 900/2700 | notebook execution test | FAIL | [OTHER: TF5-REMOTE-API] `ValueError: Failed to load model: cannot import name 'MambaCache' from 'transformers.cache_utils'` — tRNADetector remote modeling code imports a transformers-4.x API removed in 5.x; raised via dnallm/models/model.py:888; model downloaded cold (362MB) then failed at load — .scratch/census-out/logs/notebooks__inference_for_tRNA__inference.log |
| notebooks/generation/inference.ipynb | notebook / generation | zhangtaolab/plant-dnagpt-BPE | 900/2700 | notebook execution test | PASS | 35.8s — .scratch/census-out/logs/notebooks__generation__inference.log |
| notebooks/generation_evo_models/inference.ipynb | notebook / generation | togethercomputer/evo-1-131k-base, arcinstitute/evo2_1b_base | 1800/3600 | notebook execution test | FAIL | [OTHER: install-gated evo handlers] notebook variant (project venv, attempted first): `ImportError: EVO2 package is required for arcinstitute/evo2_1b_base but not installed` (dnallm/models/special/evo.py handler; evo-model/stripedhyena/evo2 prerequisites deliberately absent from the project venv). Ladder (D-05/D-06) walked variant-first then both fallback legs re-run PASS in the throwaway venv: spike/evo1-8k-fallback (evo-1-8k-base load 21.3s, forward 1.1s logits (1,256,512), generate OK 55.5s, 13.88GB VRAM) and spike/evo2-noFP8-fallback (evo2_1b_base via evo2-1b-8k-noFP8.yml, forward 0.5s, generate OK 13.3s, 2.32GB VRAM). Durable disposition: optional-dep typed skip (Task 3); notebook model-reference update stays Phase 8 per D-06 — .scratch/census-out/logs/notebooks__generation_evo_models__inference.log + spike__evo1-8k-fallback.log + spike__evo2-noFP8-fallback.log |
| notebooks/generation_megaDNA/inference.ipynb | notebook / generation | lingxusb/megaDNA_updated | 900/2700 | notebook execution test | FAIL | [OTHER: install-gated + dnallm generate bug] notebook variant: `ImportError: megaDNA package is required for lingxusb/megaDNA_updated but not installed` (dnallm/models/special/megadna.py:146; prerequisites deliberately absent from project venv). Ladder (D-05/D-06): variant attempted first (5.4s) — pinned-clone fallback leg recorded under spike/megadna-pinned-fallback (see gated rows below) — .scratch/census-out/logs/notebooks__generation_megaDNA__inference.log |
| notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb | notebook / mutagenesis | zhangtaolab/plant-dnagpt-BPE-promoter_strength_protoplast | 900/2700 | notebook execution test | PASS | 35.0s — .scratch/census-out/logs/notebooks__in_silico_mutagenesis__in_silico_mutagenesis.log |
| notebooks/interpretation/interpretation.ipynb | notebook / interpretation | zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf | 900/2700 | notebook execution test | PASS | 107.6s (deeplift + mutagenesis + motif cells) — .scratch/census-out/logs/notebooks__interpretation__interpretation.log |
| notebooks/embedding_attention.ipynb | notebook / embeddings+attention | InstaDeepAI/nucleotide-transformer-v2-50m-multi-species (raw HF AutoModelForMaskedLM, trust_remote_code=True — not the dnallm loader) | 900/2700 | notebook execution test | FAIL | [NT-REMOTE-STRUCTURAL: pytorch_utils variant] `ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils'` — InstaDeepAI remote modeling_esm.py:40 (raw AutoModelForMaskedLM trust_remote_code path; the 05-04 shim attaches to modeling_utils and is not in this notebook's import path anyway) — .scratch/census-out/logs/notebooks__embedding_attention.log |
| notebooks/data_prepare/finetune/finetune_data.ipynb | notebook / data preparation | zhangtaolab/plant-dnabert-BPE; dataset zhangtaolab/plant-multi-species-core-promoters | 1800/3600 | notebook execution test | FAIL | [BPE-TOKENIZER] `TypeError: 'dict' object is not an instance of 'Sequence' while processing 'vocab'` — raw `AutoTokenizer.from_pretrained("zhangtaolab/plant-dnabert-BPE")` demo cell dies in the DebertaV2Tokenizer slow path (transformers tokenization_deberta_v2.py:112; repo tokenizer_config declares tokenizer_class DebertaV2Tokenizer; transformers 5.x auto route) — .scratch/census-out/logs/notebooks__data_prepare__finetune__finetune_data.log |
| notebooks/data_prepare/predict/predict_data.ipynb | notebook / data preparation (markdown-only: zero code cells) | — | 900/2700 | notebook execution test | PASS | 0.5s — 0 code cells (documentation-only); kernel started and shut down clean — .scratch/census-out/logs/notebooks__data_prepare__predict__predict_data.log |
| notebooks/finetune_binary/finetune_binary.ipynb | notebook / finetune (binary) | zhangtaolab/plant-dnabert-BPE; dataset plant-multi-species-core-promoters | 3600/7200 | notebook execution test | PASS | 484.7s — full 3-epoch training through the dnallm loader (plant-dnabert-BPE tokenizer fine via the loader route; only the raw-AutoTokenizer path is BPE-broken) — .scratch/census-out/logs/notebooks__finetune_binary__finetune_binary.log |
| notebooks/finetune_custom_head/finetune.ipynb | notebook / finetune (custom head) | zhangtaolab/plant-dnagpt-BPE (megaDNA referenced as alternative) | 3600/7200 | notebook execution test | FAIL | [OTHER: megaDNA install-gated demo cell] training itself completed (417/417 steps, checkpoints written), then the megaDNA alternative-model demo cell died: `ImportError: megaDNA package is required for lingxusb/megaDNA_updated but not installed` (dnallm/models/special/megadna.py:146). Ladder: family fallback leg PASS under spike/megadna-pinned-fallback (see Gated ladder evidence) — .scratch/census-out/logs/notebooks__finetune_custom_head__finetune.log |
| notebooks/finetune_generation/finetune_generation.ipynb | notebook / finetune (generation) | lingxusb/megaDNA_updated, zhangtaolab/plant-dnagpt-singlebase; input ath_cds.csv + pyfastx .fxi | 3600/7200 | notebook execution test | FAIL | [OTHER: missing external genome input] `FileExistsError: the input fasta file Arabidopsis_thaliana.TAIR10.cds.all.fa.gz does not exists` at cell 2 (pyfastx Fasta) — the .fa.gz is a root-gitignored (`*.gz`) external input absent from the repo and this box (only its derived artifacts .fxi + ath_cds.csv are tracked); the notebook ships no download cell for it — .scratch/census-out/logs/notebooks__finetune_generation__finetune_generation.log |
| notebooks/finetune_multi_labels/finetune_multi_labels.ipynb | notebook / finetune (multi-label) | zhangtaolab/plant-dnagpt-BPE; input maize_test.tsv | 3600/7200 | notebook execution test | PASS | 1421.0s — full multi-label training (2625 steps) — .scratch/census-out/logs/notebooks__finetune_multi_labels__finetune_multi_labels.log |
| notebooks/finetune_NER_task/data_generation_and_inference.ipynb | notebook / dataset generation + inference | zhangtaolab/plant-dnagpt-6mer; rice inputs downloaded from rice.uga.edu (documented URLs, cell 9) | 3600/7200 | notebook execution test | PASS | 779.8s cold (20/20 code cells: rice downloads, bed build, GFF3 parse, 6mer dataset build + inference) — .scratch/census-out/logs/notebooks__finetune_NER_task__data_generation_and_inference.log |
| notebooks/finetune_NER_task/finetune_NER_task.ipynb | notebook / finetune (token NER) | zhangtaolab/plant-nucleotide-transformer-BPE; consumes rice_gene_ner_BPE.pkl | 3600/7200 | notebook execution test | FAIL | [NT-REMOTE-STRUCTURAL] `ValueError: Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'` (remote modeling_esm.py needs removed 4.x PretrainedConfig defaults; via dnallm/models/model.py:888) — exactly the 05-05 script-lane prediction for this checkpoint; one owner disposition now covers the NT family items (05-04 options) — .scratch/census-out/logs/notebooks__finetune_NER_task__finetune_NER_task.log |
| notebooks/lora_finetune_inference/lora_finetune.ipynb | notebook / LoRA finetune | dataset plant-multi-species-core-promoters (model via config/dropdown) | 3600/7200 | notebook execution test | FAIL | [OTHER: optional-dep mamba_ssm] `ValueError: Failed to load model: This modeling file requires the following packages that were not found in your environment: mamba_ssm` — PlantCAD2 remote code (same gate as lora_inference; dnallm `[mamba]` extra / native CUDA build absent from project venv) — .scratch/census-out/logs/notebooks__lora_finetune_inference__lora_finetune.log |
| notebooks/lora_finetune_inference/lora_inference.ipynb | notebook / LoRA inference | kuleshov-group/PlantCAD2-Small-l24-d0768 (huggingface source) | 1800/3600 | notebook execution test | FAIL | [OTHER: optional-dep mamba_ssm] `ValueError: Failed to load model: This modeling file requires the following packages that were not found in your environment: mamba_ssm` — PlantCAD2 remote code declares mamba_ssm required (dnallm `[mamba]` extra / native CUDA build absent from project venv); raised via dnallm/models/model.py:888 — .scratch/census-out/logs/notebooks__lora_finetune_inference__lora_inference.log |
| notebooks/benchmark/benchmark.ipynb | notebook / benchmark (3 models) | Plant DNABERT, Plant DNAGPT, Nucleotide Transformer (= zhangtaolab/nucleotide-transformer-v2-100m-promoter) via benchmark_config.yaml | 3600/7200 | notebook execution test | FAIL | [OTHER: dnallm benchmark column bug blocks the run pre-model] `KeyError: "Column labels not in the dataset. Current columns: ['sequence']"` at benchmark.run() — dnallm/inference/benchmark.py:296 hardcodes `self.datasets[di]["labels"]` while benchmark_config.yaml declares `label_column: label` (the csv's actual column); fails before ANY model loads, so the 05-04 NT third-model flag sits downstream of this blocker (NT gap proven separately by the NER notebook, the script and the 05-04 smoke). NT cache condition on this box: snapshot PATCHED by the orchestrator probe (config.json + is_decoder/add_cross_attention; modeling_esm.py init_weights→post_init; backups at *.dnallm-bak) — an NT load here would surface at the forward-stage get_extended_attention_mask rung instead — .scratch/census-out/logs/notebooks__benchmark__benchmark.log |
| mcp_example/mcp_client_ollama_langchain_agents.ipynb | notebook / MCP client (ollama) | ollama qwen3.8:latest via local MCP server (langchain agents) | 600/1800 | notebook execution test | deferred-owner (probe green; Phase-8 ollama plan required) | [OLLAMA-ENV] never executed (T-05-16): ollama probe GREEN at http://localhost:11434/api/tags (qwen3.8:latest present); dnallm MCP server http://localhost:8000/mcp NOT running; the notebook's `!uv pip install` cells must never run against the project venv — execution needs the Phase-8 ollama/VRAM coexistence plan (owner decision, D-08) — probe record in .scratch/census-out/manifest.json |
| mcp_example/mcp_client_ollama_pydantic_ai.ipynb | notebook / MCP client (ollama) | ollama qwen3.8:latest via local MCP server (pydantic-ai) | 600/1800 | notebook execution test | deferred-owner (probe green; Phase-8 ollama plan required) | [OLLAMA-ENV] never executed (T-05-16): same probe evidence as the langchain sibling (ollama GREEN qwen3.8:latest; MCP server endpoint down; pydantic-ai notebook has no install cells but needs the same server+ollama stack) — probe record in .scratch/census-out/manifest.json |
| marimo/inference/inference_demo.py | marimo app / inference (export-html) | xlsx-driven dropdowns; defaults resolve to zhangtaolab/plant-dnabert-BPE-open_chromatin (modelscope, observed in the 05-05 live export) | 1200/1500 | marimo execution test | PASS | 8.2s warm, 67KB-class HTML artifact re-proven (defaults observed: zhangtaolab/plant-dnabert-BPE-open_chromatin) — .scratch/census-out/logs/marimo__inference__inference_demo.log |
| marimo/finetune/finetune_demo.py | marimo app / finetune (export-html) | zhangtaolab/plant-dnagpt-BPE; dataset plant-multi-species-core-promoters | 3600/7200 (specced by 05-06; MARIMO_EXEC_SPECS entry added) | marimo execution test | PASS | 4.9s — export-html executes the reactive graph: config load + full UI construction ran clean; the training action is button-gated by app design (prepare() runs on click), so no model/dataset load in the headless export — .scratch/census-out/logs/marimo__finetune__finetune_demo.log |
| marimo/benchmark/benchmark_demo.py | marimo app / benchmark (export-html) | zhangtaolab/plant-dnabert-BPE-promoter, plant-dnagpt-BPE, plant-dnagpt-BPE-promoter (config.yaml + test.csv) | 3600/7200 (specced by 05-06; MARIMO_EXEC_SPECS entry added) | marimo execution test | PASS | 5.9s — export-html: config load + test.csv dataset load executed at module level; benchmark.run() is button-gated by app design — .scratch/census-out/logs/marimo__benchmark__benchmark_demo.log |
| notebooks/finetune_NER_task/generate_bpe_dataset.py | script / dataset generation | zhangtaolab/plant-nucleotide-transformer-BPE (tokenizer+model via dnallm loader, modelscope); rice genome + GFF3 inputs (downloaded to sandbox) | 3000/3600 | script execution test | FAIL | [NT-REMOTE-STRUCTURAL] real census re-run (rice inputs cached under .scratch/census-out/inputs/): `ValueError: Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'` after 21.6s — same terminal rung the 05-05 durable test self-heals around (its typed skip stays the durable disposition); annotation-bed repair from 05-05 confirmed working (the script got past bed/GFF3 processing to the model load) — .scratch/census-out/logs/notebooks__finetune_NER_task__generate_bpe_dataset.log |

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
| PASS verdicts (05-06 final) | 11 | Table A grep |
| FAIL verdicts (05-06 final) | 12 | Table A grep |
| deferred-owner verdicts (05-06 final) | 2 (the ollama mcp pair) | Table A grep |

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
# every verdict is filled (0 pend) — flipped by the 05-06 census campaign; the
# pattern is assembled by concatenation so this block cannot self-match
PEND_PAT='| pend''ing |'
test "$(awk -v a="$SEC_A" -v b="$SEC_B" '$0~a,$0~b' $CENSUS | grep -cF "$PEND_PAT")" -eq 0
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

## Gated ladder evidence (D-05/D-06 — variant first, then smallest viable, never skip-first)

Recorded by the 05-06 campaign as manifest items `spike/*` (throwaway `/tmp/feas-venv`,
transformers 4.57.6 + dnallm editable; the project `.venv` provably untouched — all spike-only
imports re-proven absent, pyproject porcelain-empty):

| ladder leg | result | evidence |
| --- | --- | --- |
| evo-1 notebook variant `togethercomputer/evo-1-131k-base` | (not re-run: 05-FEASIBILITY 4-attempt FAIL stands — remote code needs `pos_idx_in_fp32`, absent from every transformers in dnallm's span) | 05-FEASIBILITY.md evo-1 row; spike-logs/spike_evo1.log |
| evo-1 smallest viable `togethercomputer/evo-1-8k-base` (spike/evo1-8k-fallback) | PASS — load 21.3s warm, forward 1.1s logits (1,256,512), notebook-path generate OK 55.5s real DNA output, peak 13.88GB | .scratch/census-out/logs/spike__evo1-8k-fallback.log |
| evo2 notebook variant `arcinstitute/evo2_1b_base` auto-config | (not re-run: 05-FEASIBILITY FAIL stands — GB10 FP8 trap, Transformer Engine absent by necessity) | 05-FEASIBILITY.md evo2 row; spike-logs/spike_evo2.log |
| evo2 noFP8 fallback `evo2-1b-8k-noFP8.yml` (spike/evo2-noFP8-fallback) | PASS — load 21.3s, forward 0.5s logits (1,256,512), generate OK 13.3s, peak 2.32GB | .scratch/census-out/logs/spike__evo2-noFP8-fallback.log |
| megaDNA pinned clone `cb2f5ab4` + MEGABYTE_pytorch 0.2.1 (spike/megadna-pinned-fallback) | PASS (load 4.0s, forward 0.3s) with the known dnallm-side generate failure reproduced: `GENERATE_FAILED` megadna tokenizer single-string encode error (Phase 8 repair candidate) | .scratch/census-out/logs/spike__megadna-pinned-fallback.log |

*Inventory committed by plan 05-05 (2026-10-02); verdicts filled by the 05-06 census campaign per D-08.*

## Hand-off to owner (D-09 — flagged, not re-scoped)

**Campaign summary (2026-10-02, GB10 dev box):** 25 census items executed through the 05-05
harness lanes — **11 PASS / 12 FAIL / 2 deferred-owner**; every FAIL carries the exact terminal
traceback + log path in its Table A row; the gated ladder walked variant-first everywhere and
the throwaway-venv fallback legs re-proved all three spike families (evo1-8k, evo2-noFP8,
megaDNA-pinned). The project environment is provably untouched (pyproject porcelain-empty;
stripedhyena/evo2/MEGABYTE_pytorch/pyBigWig/langchain_ollama all absent from `.venv`).

### 1. Phase 7-9 overlap flag (owner rescoping requested)

D-08's "entire example/ tree executes" bar landed HERE, pulling forward work the Phase 7-9
charters still carry verbatim: **Phase 7** showcase-notebook write-back (its notebook set is
now census-executed); **Phase 8** full-green rollout + `models.lock` + REPAIR-04 + MCP-01
ollama infra (its repair queue is now concretely this census's FAIL list); **Phase 9** nightly
census wiring (the durable execution layer now exists: 8 active notebooks + 7 gated + 3 marimo
apps + script lane; full `tests/examples` run measured at ~50 min on GB10 — well inside the
900-min nightly, but the runtime note belongs to Phase 9's budget planning; owner floated
pytest-notebook/xdist — decision: keep the nbclient harness, xdist does not help a
GPU-bottlenecked lane). Roadmap rescoping is an owner action; this plan did not rewrite any
Phase 7-9 scope.

### 2. FAIL repair queue (the transformers-5 adaptation worklist, class-tagged)

**NT-REMOTE-STRUCTURAL (one disposition covers three items — 05-04 options: structural
PretrainedConfig default patch / transformers pin for this consumer / checkpoint re-export):**
- `notebooks/finetune_NER_task/finetune_NER_task.ipynb` — `ValueError: Failed to load model:
  'EsmConfig' object has no attribute 'is_decoder'` (remote modeling_esm.py; model.py:888)
- `notebooks/finetune_NER_task/generate_bpe_dataset.py` — same terminal rung (05-05 bed repair
  confirmed working; durable test = self-healing typed skip)
- benchmark third model (NT) — blocked BEHIND the benchmark.py bug below; proven separately by
  the 05-04 smoke. Cache note: the local ModelScope NT snapshot is PATCHED (config.json
  is_decoder/add_cross_attention + modeling_esm.py init_weights→post_init; `*.dnallm-bak`
  backups) — on this box an NT load would fail at the forward-stage
  `get_extended_attention_mask` rung instead

**BPE-TOKENIZER (zhangtaolab BPE-family tokenizer artifacts on transformers 5.x):**
- `notebooks/data_prepare/finetune/finetune_data.ipynb` — raw
  `AutoTokenizer.from_pretrained("zhangtaolab/plant-dnabert-BPE")` demo cell dies in the
  DebertaV2Tokenizer slow path (`tokenization_deberta_v2.py:112`,
  `TypeError: 'dict' object is not an instance of 'Sequence' while processing 'vocab'`).
  Scope note: the defect is the REPO's tokenizer artifacts (tokenizer_class declaration +
  tokenizer.json missing [UNK] per the orchestrator's probe), NOT dnallm — the dnallm loader
  route tokenizes the same family fine (finetune_binary trains on plant-dnabert-BPE). All
  cached zhangtaolab BPE repos declare DebertaV2Tokenizer; repair belongs upstream or via a
  dnallm-side tokenizer fallback (Phase 8 decision)

**OTHER (each described precisely — this is the new-findings class):**
- `notebooks/inference_for_tRNA/inference.ipynb` — tRNADetector remote code imports
  `MambaCache` from `transformers.cache_utils` (removed in 5.x) — second remote-API-removal
  family, distinct from the NT class
- `notebooks/embedding_attention.ipynb` — InstaDeepAI NT-v2-50m remote modeling_esm.py:40
  imports `find_pruneable_heads_and_indices` from `transformers.pytorch_utils` (the 05-04 shim
  attaches to modeling_utils only, and this notebook uses raw AutoModelForMaskedLM so the shim
  is not in its import path anyway) — shim-scope question for Phase 8
- `notebooks/benchmark/benchmark.ipynb` — **dnallm-side bug, fails before any model loads**:
  `dnallm/inference/benchmark.py:296` hardcodes `self.datasets[di]["labels"]` while
  benchmark_config.yaml declares `label_column: label` (KeyError: Column labels)
- `notebooks/finetune_custom_head/finetune.ipynb` — training fully green (417/417 steps); only
  the megaDNA alternative-model demo cell is install-gated (megadna.py:146); ALSO the
  dnallm-side megadna generate tokenizer bug reproduced in the fallback leg (Phase 8 repair
  candidate)
- `notebooks/finetune_generation/finetune_generation.ipynb` — expects
  `Arabidopsis_thaliana.TAIR10.cds.all.fa.gz` (root-gitignored external input, no download
  cell; only derived artifacts tracked)
- `notebooks/lora_finetune_inference/lora_finetune.ipynb` + `lora_inference.ipynb` — PlantCAD2
  remote code requires `mamba_ssm` (`[mamba]` extra / native CUDA build absent); durable gated
  tests self-heal when the extra lands
- `notebooks/generation_megaDNA/inference.ipynb` + `notebooks/generation_evo_models/inference.ipynb`
  — install-gated handlers (prerequisites deliberately absent from the project venv); durable
  optional-dep gated tests + proven fallback legs

### 3. evo notebook model-reference update (D-06 — Phase 8, not done here)

D-06 says Phase 8 executes the small variant and updates the notebook's model reference. The
05-06 ladder re-proved `evo-1-8k-base` end-to-end (load 21.3s, forward, generate OK) and evo2
via `evo2-1b-8k-noFP8.yml` — the update question (switch the reference vs install-gate +
models.lock entries carrying the prerequisites) is the owner's Phase 8 call.

### 4. EXEC-03 / EXEC-04 status

- **EXEC-03 (marimo apps, dev-box leg): COMPLETE** — all three apps census-green
  (inference_demo 8.2s, benchmark_demo 5.9s, finetune_demo 4.9s; MARIMO_EXEC_SPECS grown to
  all three; export-html executes each app's module-level graph, model actions are
  button-gated by app design).
- **EXEC-04 (example script, dev-box leg): PARTIAL** — the script runs standalone (05-05 bed
  repair proven in-campaign) but terminates at the NT structural rung; its durable test is the
  self-healing typed skip. REQ stays open for Phase 8 pending the NT owner disposition.

### 5. deferred-owner rows

Both `mcp_example` notebooks: ollama probe GREEN on this box (qwen3.8:latest present) —
execution deferred to the Phase-8 ollama/VRAM coexistence plan per T-05-16 (the uv pip install
cells must never run against the project venv; the dnallm MCP server endpoint is also not
running). Manifest outcome strings carry the probe evidence; the durable gated tests skip
network-unavailable with both live probe results in the message and fail loudly (owner
decision) if both endpoints ever come up before Phase 8 lands.

### 6. Unpushed phs range (manual-push-only rule)

`git log origin/dev..phs --oneline | wc -l` = 71 commits at campaign start (tip `2c70aad`),
growing with the 05-06 commits (census verdicts `330134a`, `147ea9a`, + Task 3 wiring/docs).
No pushes performed (D-09).
