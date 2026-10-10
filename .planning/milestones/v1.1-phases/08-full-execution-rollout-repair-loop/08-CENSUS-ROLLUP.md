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
| mcp_example/mcp_client_ollama_langchain_agents.ipynb | GATED (`_gate_ollama_stack`) | PASS | 08-08 un-gate (both-up dev-box execution): D-13 retry-window probe live, langchain under the isolated `dnallm-mcp-langchain` kernelspec — pair green in the canonical run (`2 passed / 0 SKIPPED`, /tmp/08_08_mcp.log) |
| mcp_example/mcp_client_ollama_pydantic_ai.ipynb | GATED (`_gate_ollama_stack`) | PASS | 08-08: same canonical pair run (pydantic_ai first hit a transient DeadKernelError under VRAM contention — the 08-02 NER run-1 class; isolated re-run green in 361s, then green in the canonical pair run) |
| notebooks/generation_evo_models/inference.ipynb | GATED (`_gate_evo`) | PASS | 08-06/08-07 un-gate: isolated dnallm-evo lane (dev-box giants tier + HF_HUB_OFFLINE) — executed for real green (5 passed / 0 SKIPPED, 49s; re-confirmed 08-07 census). 08-07 reconciliation also fixed the static import check (evo venv-only stack added to OPTIONAL_IMPORT_MODULES after the D-21 stamp cell's literal `import flash_attn` failed it in the project venv) |
| notebooks/generation_megaDNA/inference.ipynb | GATED (`_gate_megadna`) | PASS | 08-05 un-gate: pinned prereqs (megadna @ cb2f5ab4 + MEGABYTE_pytorch==0.2.1) installed into the project venv (reversible) + D-21 stamp + executable pinned install cell; executed for real green in the family lane (12 passed / 0 SKIPPED, 48:05, /tmp/08_05_family.log) |
| notebooks/finetune_custom_head/finetune.ipynb | GATED (`_gate_megadna`) | PASS | 08-05 un-gate: demo cell repaired (pinned install cell before the megaDNA load — census ImportError at megadna.py:146 gone) + D-21 stamp; DNAGPT and megaDNA trainings both executed green in the family lane |
| notebooks/finetune_generation/finetune_generation.ipynb | GATED (`_gate_megadna_isolated`) | PASS | 08-04 isolated dnallm-megadna lane first green (837s, 0 skips); 08-05 family reconciliation re-run green in the same lane |
| notebooks/lora_finetune_inference/lora_finetune.ipynb | GATED (`_gate_mamba`) | PASS | 08-08 un-gate: `.[mamba]` extra built into the dev-box project venv (causal_conv1d 1.7.0 + mamba-ssm 2.3.2.post1, test-mamba flags) — `_gate_mamba` probes green, training executed for real (3 passed / 0 SKIPPED incl. contract, /tmp/08_08_lora.log) |
| notebooks/lora_finetune_inference/lora_inference.ipynb | GATED (`_gate_mamba`) | PASS | 08-08: same lane; REPAIR-01 — huggingface.co unreachable on the dev box left the uncached `plantcad/...` LoRA adapter download dead while the cached base model passed; both lora specs now pin `HF_ENDPOINT=hf-mirror.com` (08-06 spec-env precedent) + `TestLoraMirrorEndpoint` contract |
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
| evo (evo-1 / evo2) | generation_evo_models | CLOSED (08-07): PASS by real execution on the isolated dnallm-evo lane — giants strategy verified at load time (A4) | evo family plan (08-06 + 08-07, done) |
| megaDNA | generation_megaDNA, finetune_custom_head, finetune_generation | CLOSED (08-05): all three PASS by real execution — siblings on the project venv (reversible pinned prereqs), finetune_generation on the isolated lane | megaDNA family plan (08-05, done) |
| PlantCAD / mamba | lora_finetune, lora_inference | CLOSED (08-08): both PASS by real execution — `.[mamba]` kernels in the project venv + mirror endpoint for the LoRA adapter | mamba/lora family plan (08-08, done) |
| mcp + ollama | langchain + pydantic_ai client notebooks | CLOSED (08-08): both PASS in the both-up state on the dev box (D-13 retry probe live; 6 MCP live-server probes green across both transports); runner enable = owner user_setup step (MCP-01) | mcp batch plan / runner infra (08-08 dev-box half done; ci.yml wiring 08-09) |

Every gated skip in this baseline is an honest `optional-dep:`/`network-unavailable:` typed
skip carrying live probe evidence — the CURRENT-STATE baseline per D-03, not a failure.

## evo family reconciliation + A4 verification (08-07, D-03)

Full examples census re-run post-evo-repairs (2026-10-04 20:00–22:31 UTC, three sequential
passes to stay under runner time caps): **188 passed / 5 honest typed skips / 0 failed** —
14 P (NER family + multi_labels, 51:28) + 25 P (megaDNA family, 48:21) + 149 P (rest,
50:34; one pre-fix failure `test_notebook_imports[evo]` re-run green post-fix, 106 P/1 S
on the fast module). The 5 skips: 1 benign no-imports (predict_data), 2 mcp
network-unavailable (MCP endpoint down at run time, ollama GREEN — probe evidence in
message), 2 lora optional-dep (mamba_ssm; 08-08's family). Zero evo skips —
`generation_evo_models` executed green in the census itself. Fast lane (EXEC-05):
**1803 passed / 1 pre-existing skip, exit 0, 91.46s** (+3 vs the 08-05 baseline = 08-06's
TestEvoIsolatedLane; zero new skips).

**A4 VERIFIED (CI-05 empirical proof).** The 08-07 execution loaded the evo-1 giants
snapshot offline (`HF_HUB_OFFLINE=1` in the spec env — a re-fetch was impossible by
construction; zero fetch lines in the execution log). The evo-1 snapshot dir still holds
**zero `.pt` files** (safetensors + index + configs only, 12.3GB window, refs/main
present) and the giants blob store is unchanged at 14,889 MiB = evo-1 12.3GB + evo2
2.7GB — no 16.81GB `pytorch_model.pt` re-fetch occurred. Note: `evo2_1b_base.pt` inside
the evo2 snapshot is that model's OWN native checkpoint format (present since 08-06's
recorded full fetch), not a CI-05 violation — the `.pt`-free guarantee is scoped to the
evo-1 giants snapshot it protects. Giants dir remains outside every cached path.

**08-07 repair (REPAIR-01):** the 08-06 D-21 stamp cell added a literal
`import flash_attn` to the evo notebook, which the static `test_notebook_imports` check
(rightly) failed in the project venv — the evo stack is FEASIBILITY-locked to the
throwaway venv. Fixed by extending `OPTIONAL_IMPORT_MODULES` (pybedtools precedent) with
`flash_attn/stripedhyena/evo2` + a comment naming the 08-06 lock; re-run green.

## mcp + lora family closure — D-03 family-close census (08-08)

Full examples lane re-run post-08-08-repairs (2026-10-05, three disjoint chunked
passes, both endpoints up for the mcp pair): **196 passed / 1 benign typed skip /
0 failed** — chunk A (NER|multi_labels, which case-insensitively also captures the
`generation_*` families incl. evo + megaDNA trio) 41 P in 1:10:32; chunk B
(megaDNA-only leftovers) 6 P; chunk C (rest, incl. the REAL lora pair ~11 min and
the REAL mcp pair ~25 min) 149 P + 1 S in 1:23:45. The single skip is the benign
no-imports entry (predict_data) — **zero lora skips, zero mcp skips**: all 21
notebooks now execute on the dev box. Fast lane: **1807 passed / 1 pre-existing
skip, exit 0, 98.30s** (+4 vs the 08-07 baseline = 08-08's TestLoraMirrorEndpoint
and the three D-13 retry-contract tests; zero new skips).

Stage cleanup recorded per the D-07 discipline (owner directive 2026-10-05):
post-census the orphaned streamable-http server (24.5GB RSS) was killed, sandbox
kernels torn down, memory asserted at 116 Gi available before any next stage.

08-09 inheritance: the mcp pair's both-up execution requires the MCP server on
:8000 (yaml port; CLI flags are dead) and ollama on 127.0.0.1:11434 — stage the
ci.yml lanes per the D-07 comment block in tests/examples/test_notebook_execution.py,
and keep the cleanup+assert discipline at every stage boundary (>=35Gi available
before entering a heavy stage).

## First-Dispatch Consumption (08-09, REPAIR-01 / owner dispositions)

The example-nightly job has dispatched twice on the `phs` ref (workflow_dispatch;
the 05:30 UTC schedule only fires post-merge from the default branch — D-04).

| dispatch | run id | when | duration | outcome |
| --- | --- | --- | --- | --- |
| 1 | 37184990854 | 2026-10-04 07:10:34Z | 64 s | stage-0 infra failure: pyBigWig sdist `ld: cannot find -lpython3.12` (sysconfig baked nonexistent /opt/hostedtoolcache LIBDIR) |
| 2 | 37185365961 | 2026-10-04 07:18:08Z | 66 min | FIRST COMPLETED dispatch — consumed below (junit artifact `example-nightly-junit` + full logs fetched via gh) |

**Dispatch-2 stage outcomes** (stage-results.txt from the log): `stage1-examples=1`,
`stage1-yaml=0`, both stage-4 skip audits 0 (every skip allowlisted), summary
exited 1 per D-08 (honest, not forever-green). Stage 1 measured
**6 failed / 147 passed / 7 skipped / 8 deselected in 3908.46s (1:05:08)** —
with NO restored model cache (see the quota section: "Cache not found" for both
the uv and models keys), i.e. every model fetch was cold and the lane still
finished in 65 min (ModelScope + hf-mirror are fast from this box).

**Per-failure classification** (every runner-side failure repaired or dispositioned;
nothing silently dropped):

| failing item | evidence (junit/log) | class | disposition |
| --- | --- | --- | --- |
| interpretation.ipynb | `ImportError: cannot import name 'backend2gui' from 'IPython.core.pylabtools'` (logomaker chain) | library/declared-dep conflict (IPython 9 × mpl 3.8.4) | REPAIRED — 08-02 commit 439370f (`ipython>=8.31,<9` in notebook extra), landed later the same day, post-dispatch |
| finetune_NER_task/data_generation_and_inference.ipynb | same backend2gui ImportError | same class | REPAIRED — same 08-02 commit |
| embedding_attention.ipynb | same backend2gui ImportError | same class | REPAIRED — same 08-02 commit |
| test_combined_notebook_executes_with_display_figure | `FileNotFoundError: ../plant_helixseek_cre/data/chr1_5100001_5300000.fas` (fresh-sandbox sibling gap — the 08-01 runner failure class, live again) | harness/content | REPAIRED — 08-02 Task 3 commit c5f0916 (`COMBINED_EXTRA_INPUTS` sibling seeding + contract tests) |
| test_cre_notebook_executes_within_selection_bands | test's own honest assert: `bedtools is not on PATH` (runner inventory probe confirmed MISSING; A10) | runner system dep | REPAIRED — 08-09 Task 2 commit ea1bfd6 (bedtools rootless micromamba/bioconda step onto GITHUB_PATH; no sudo/apt) |
| test_generate_bpe_dataset_produces_artifact | `NotImplementedError: "intersectBed" does not appear to be installed` (pybedtools in the script lane) | runner system dep (same bedtools class) | REPAIRED — same 08-09 Task 2 step |

**Per-skip classification** (7 skips, all audit-allowlisted): 1 benign
no-imports (predict_data — the permanent baseline skip) + 6 gated
`optional-dep:` skips (evo ×1, megaDNA ×3, lora ×2) — honest at the time: the
08-01 skeleton installed no gated-family prerequisites. 08-09 Task 2's prereq
steps (evo venv + flash-attn wheelhouse, mamba wheelhouse, megaDNA pinned
clone/venvs) turn exactly these six into real executions on the next dispatch.
The mcp pair was deselected from stage 1 (8 items) — stages 2/3 were
placeholders in that skeleton; they are wired now (Task 2).

**Runner inventory (probe step answers, dispatch 2):** bedtools MISSING (A10
confirmed — rootless step now covers it); GPU `NVIDIA GB10, [N/A]` (GB10 does
not report memory via that nvidia-smi query — the >=35Gi `free -h` discipline
is the operative guard on this shared box); `hf-mirror.com HTTP 200`;
ollama on 127.0.0.1:11434 is UP (the runner shares the dev box's systemd
service, owner-enabled — MCP-01's manual step is effectively satisfied for
stage 3, re-verified 2026-10-05).

**Open dispositions (owner actions / deferrals):**
1. Cache-quota decision — see the next section (THE open owner item).
2. First dispatch of the COMPLETED job (post-08-09): fired at hand-off on the
   pushed `phs` ref (see the final census section); its outcome lands
   post-merge per the Phase-5 D-04 runner-confirmation boundary. The one-time
   first-run costs (flash-attn ~50 min build, mamba kernel build, 12.9 GB
   giants prefetch) are wheel/eager-cache amortized afterward.

## Cache-quota measurement + decision record (08-09, CI-05 / research Pitfall 3 / A6-A8)

Measured 2026-10-05. **The decision is an owner call — evidence below, do not
guess** (research A8: pay-as-you-go is a billing choice).

**Runner-side GitHub cache store (definitive, `actions/caches` API):** exactly
2 entries, both `Linux-uv-cuda-*` from 2026-10-01 (3.15 GB + 7.22 GB =
**10.37 GB total — the store already sits at the 10 GB eviction threshold**).
**Zero models-cache entries have ever been saved**: both 2026-10-04 dispatches
restored nothing ("Cache not found") and their post-job saves produced no
entry — the source dirs (below) exceed the 10 GB single-entry cap, so the
save is rejected every night.

**Box-side cache dirs** (the runner shares $HOME with the dev box — same
user, same paths actions/cache would save):
- `~/.cache/huggingface/hub` = 84 GB: 28 GB evo-1-8k full-repo leftover
  (pre-giants era, includes the 16.81 GB `pytorch_model.pt`) **inside the
  cached path**, 19 GB `Qwen--Qwen3.8-27B` (manual/dev download), 38 GB
  legacy top-level blobs, plus small lock-model stubs.
- `~/.cache/modelscope/hub` = 13 GB: models 9.4 GB + datasets 2.8 GB.
- `~/models-giants` = 15 GB — the giants tier, **outside every cached path by
  construction** (D-14 verified; evo-1 safetensors-only 12.3 GB + evo2 2.7 GB).

**Clean lock-only arithmetic** (what a cache holding exactly the 24-row
models.lock + its dataset row would weigh, from per-model du):
ms lock models 9.62 GiB + ms lock dataset
(plant-multi-species-core-promoters) 1.5 GiB + hf lock models ~4.1 GiB
(evo2 2.7 + megaDNA 0.58 + InstaDeepAI ~0.5 + PlantCAD2 ~0.35) ≈
**15.2 GiB > 10 GB quota**. The plan's exclusion option (drop evo2, the
largest non-giant) lands at ~12.5 GiB — still over. Even ms-models-only
(9.62 GiB) fits only by evicting the uv caches and covering nothing else.

**Options for the owner** (numbers above):
a. Opt into cache pay-as-you-go (removes the 10 GB cap; GitHub billing
   decision).
b. Drop the models-cache layer entirely (de-facto operating mode since
   2026-10-04 and PROVEN GREEN: a fully cold stage 1 completed in 65 min with
   all downloads; nightly cost ≈ 15 GB of re-downloads, no hard failure).
c. Partial ms-only cache — poor value (evicts the uv caches; leaves hf models
   cold).
Whichever option: the box-side $HOME cleanup is prerequisite to (a) ever
working — while the 28 GB evo-1 leftover + 19 GB Qwen + 38 GB legacy blobs sit
inside `~/.cache/huggingface/hub`, every save from this runner is rejected
regardless of the lock content. The executor's targeted removal of the 28 GB
evo-1 leftover was denied by the execution permission policy (irreversible
deletion outside the pinned project tree) — recorded here as an OWNER ACTION:
`rm -rf ~/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base`
(28 GB, re-downloadable never needed: the giants tier holds the sanctioned
safetensors-only copy). The 19 GB Qwen and 38 GB legacy-blob trees are
out-of-scope dev artifacts flagged for the same owner cleanup pass.

**Giants stay outside the cache regardless of the option chosen** (Pitfall 3 /
D-14 — structural, not part of this decision).

## Final D-03 reconciliation census (08-09, phase-final)

**Full examples census** (single invocation, `.pytest tests/examples -q -rs`,
2026-10-05 04:24–07:24 UTC, both endpoints up for the mcp pair: MCP server on
:8000 streamable-http + ollama on 127.0.0.1:11434):
**196 passed / 1 benign typed skip / 0 failed in 10774.89s (2:59:34)**,
exit 0 (/tmp/08_09_examples.log). The single skip is the permanent
no-imports entry (predict_data) — **zero lora/mcp/megaDNA/evo skips: every
one of the 21 notebooks executes for real**, matching the 08-08 family-close
baseline counts exactly (196/1/0). The per-item table above (08-03 baseline +
family-close updates) is re-confirmed wholesale by this run: every
notebook/marimo/script/showcase/YAML row PASS. **Fast lane (EXEC-05): 1807
passed / 1 pre-existing skip / 53 deselected, exit 0, 97.55s**
(/tmp/08_09_fastlane.log) — identical counts to the 08-08 baseline, zero new
skips.

Stage discipline per the owner directive (>=35Gi before heavy stages,
before/after recorded): pre-census 115Gi avail → server start 112Gi → census
trough 39Gi (gated-family training peak, still above floor) → post-census
115Gi after server stop + process cleanup; port 8000 verified free after
teardown (/tmp/08_09_census_mem.log). The owner's live JupyterLab kernels
(idle, auto-respawned) were left intact — memory never approached the floor.

**Runtime-budget hand-off for Phase 9 (CI-06/CI-07):**
- First dispatch (skeleton, cold caches): stage 0 ~15 s warm-uv / stage 1
  65:08 with all model fetches cold.
- This final census (single serial invocation, warm box): 2:59:34 for all
  196 items.
- First dispatch of the COMPLETED job adds one-time costs: flash-attn sm_120
  wheel build (~50 min, cached on version+arch+torch afterward), mamba
  kernel wheel build (~40–90 min, cached), 12.9 GB giants prefetch (eager on
  disk, never re-fetched), evo/megadna throwaway-venv creation (~minutes from
  the warm local uv cache on this shared-$HOME box).
- Steady-state nightly estimate: stage 1 ~3 h + stages 2/3 ~45 min + stage 0
  ~10–20 min — well inside the 2700-min timeout backstop; the per-test marks
  remain the primary hang protection.
- Quota note for CI-06/07 planning: the models-cache layer is currently
  inert (see the quota section) — nightly runs re-download ~15 GB until the
  owner picks an option; runtime numbers above already include that cost.

**Hand-off dispatch:** the completed job's first real dispatch was fired on
the pushed `phs` ref at hand-off (workflow_dispatch; run id recorded in
08-09-SUMMARY.md). Its outcome lands per the Phase-5 D-04 post-merge
runner-confirmation boundary.
