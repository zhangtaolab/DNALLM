---
schema_version: 1
open_count: 12
waived_count: 3
fixed_count: 19
total_count: 34
last_updated: 2026-10-29T05:15:45.126Z
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
| 11 | 05 | deviation | example/notebooks/benchmark/benchmark.ipynb |  | Census FAIL row (05-04, D-07 ladder terminal): third registry model zhangtaolab/nucleotide-transformer-v2-100m-promoter not loadable on transformers 5.17 - remote code needs removed 4.x PretrainedConfig defaults (is_decoder/add_cross_attention); native-ESM route refuted (FFN shape mismatch); owner disposition pending per D-09 | fixed |  | 2026-10-02T05:12:38.864Z | 2026-10-13T02:29:17.247Z |
| 12 | 05 | stub | tests/examples/test_script_execution.py |  | environment-unavailable typed skip: generate_bpe_dataset.py pkl-production leg blocked by GAP-1-class remote-code gap (plant-nucleotide-transformer-BPE needs removed 4.x PretrainedConfig defaults on transformers 5.17); self-healing, owner disposition pending | fixed |  | 2026-10-02T05:48:40.008Z | 2026-10-13T02:29:17.411Z |
| 13 | 05 | unrun-verify | example/mcp_example |  | Census deferred-owner rows (05-06, D-08/T-05-16): both ollama mcp client notebooks never executed - ollama probe GREEN (qwen3.8:latest) but dnallm MCP server endpoint down and uv pip install cells must never touch the project venv; needs the Phase-8 ollama/VRAM coexistence plan (owner decision); durable gated tests skip network-unavailable with live probe evidence and fail loudly if both endpoints come up | open |  | 2026-10-02T08:34:21.484Z |  |
| 14 | 05 | deviation | dnallm/inference/benchmark.py | 296 | Census FAIL finding (05-06): Benchmark.run hardcodes self.datasets[di]['labels'] while example benchmark_config.yaml declares label_column 'label' - KeyError before any model loads; blocks the benchmark notebook (and its NT third-model disposition) until repaired; Phase 8 repair queue | fixed | run() label resolution honors configured label_column with 'labels' fallback and descriptive ValueError; red-first regression tests in TestRunLabelColumnResolution; note: census's ['sequence']-only shape stemmed from generate_dataset's silent path-to-sequence fallback, a separate latent bug deliberately left out of scope | 2026-10-02T08:34:21.569Z | 2026-10-10T14:39:32.164Z |
| 15 | 05 | skipped-test | tests/examples/test_notebook_execution.py |  | 05-06 gated typed skips (sanctioned, self-healing): optional-dep probe-then-execute for evo/megaDNA prerequisites, finetune_custom_head megaDNA demo cell and PlantCAD lora pair (mamba_ssm); environment-unavailable script-lane skip (05-05 pattern) unchanged; all matched by audit_skips against registered prefixes | fixed |  | 2026-10-02T08:34:21.655Z | 2026-10-13T02:29:17.575Z |
| 16 | 09 | unmet-truth | tests/benchmark/test_benchmark.py |  | Pre-existing fast-lane failure (found by 09-02 verify, reproduced at plan-start 3557e0b): TestBenchmark::test_plot_for_regression pandas TypeError float() argument ... not dict via _astype_nansafe in the plot path; not caused by any Phase 09 change (09-02 delta +8P/+0F/+0S); likely quick-task 13/14 Mapping fallout; logged in 09 deferred-items.md | fixed |  | 2026-10-05T15:45:36.353Z | 2026-10-06T01:31:42.584Z |
| 17 | quick-261007-vxx | unmet-truth | docs/user_guide/fine_tuning/getting_started.md |  | Push of ruff-format fix commit 97a7c30 to origin/dev blocked by GitHub receive-side Internal Server Error (4 attempts, Request IDs 8832:3513C8/C942:3774A8/B48E:2E4A69/991C:246D9C, 2026-10-07 15:07-15:12Z); local ruff format --check green at dev 97a7c30; re-run 'git push origin dev' when GitHub receive recovers to unblock ci.yml format gates + PR #40 | fixed | GitHub receive recovered; orchestrator re-push at 2026-10-07T15:17:44Z landed cfc8346..97a7c30 on origin/dev; CI + Docs Validation re-triggered | 2026-10-07T15:20:00.000Z | 2026-10-07T15:18:00.000Z |
| 18 | quick-261008-env | deviation | pyproject.toml |  | Windows fla installability gap (env adaptation 2026-10-08): pip install -e '.[fla]' on Windows succeeds but leaves fla broken - torch Windows wheels declare no triton dependency (Linux pulls it) and fla-core requires triton only under its [cuda]/[cpu] extras which the pyproject deliberately avoids, so fla imports top-level but fla.ops.* raises ModuleNotFoundError and test_chunk_kda_importable_when_fla_installed FAILS instead of skipping. Remedy installed in dnallm-cuda: triton-windows==3.7.1.post27 (community wheels; upstream triton ships no win_amd64) - KDA chunk kernel proven compiling+executing on RTX 5080 sm_120. Open owner decision: whether to declare triton-windows behind a Windows platform marker in the fla extra (third-party fork enters declared deps) or document the manual step | open |  | 2026-10-08T04:30:00.000Z |  |
| 19 | quick-261008-wfx | deviation | tests/examples/test_plant_helixseek_showcase.py |  | CRE showcase execution test requires the bedtools CLI and fails loud by design (test line ~480 asserts shutil.which('bedtools')); bedtools has no Windows build, so local Windows full-example runs must --deselect test_cre_notebook_executes_within_selection_bands (workaround in daily use since 2026-10-08). Open question: WSL-scope bedtools provisioning or permanent documented deselect; Linux nightly lanes unaffected. EVALUATION 2026-10-10 (owner-directed de-bedtools assessment): pyranges1 REJECTED as replacement — ruranges 0.2.7 ships no win_amd64 wheels (Windows would need Rust msvc source build; the pybedtools platform marker would simply transfer), requires-python >=3.12, and it buys zero Windows gain since CRE jaccard is subprocess-direct CLI anyway (join_overlaps left mode does map to the NER loj intersect, noted for completeness). CHOSEN DIRECTION (refined by owner 2026-10-10, platform-split): Linux/macOS KEEP bedtools/pybedtools — the frozen truth chain stays bedtools-computed, no re-freeze; Windows gets a parity-verified numpy evaluator (precedent dnallm/utils/genomic_coords.py) used only where bedtools cannot install — the CRE jaccard first, the NER example's pybedtools loj intersect second (larger surface: join semantics + ascending-order traversal). OWNER DIRECTIVE 2026-10-10: the NER example CODE ITSELF must change — generate_bpe_dataset.py and data_generation_and_inference.ipynb gain runtime platform detection (Linux keeps pybedtools.BedTool.intersect(loj=True), Windows dispatches to the numpy equivalent); the downstream frozen Anno metric (exon_f1=0.7522) must hold in band on BOTH paths, and the loj replacement needs its OWN parity check (numpy join vs `bedtools intersect -loj` — jaccard parity does not transfer to join semantics). Enabling evidence: quick-task spike 261010-wx2 (numpy vs bedtools v2.31.1 parity on the frozen selection pair jaccard=0.324727 + randomized property cases) — without parity the split would be two truths, so the split ships only if parity holds. v1.3 must pin the frozen-value comparison tolerance (numpy evaluator yields trailing float deltas, e.g. 0.3247270000001) and Windows CI (test-windows) becomes the numpy-path executor; revisit pyranges1 only if ruranges ships win_amd64 wheels. CHANGE SURFACE INVENTORY 2026-10-10 (exhaustive example/ sweep): (A) pybedtools runtime, exactly two files sharing one code path — generate_bpe_dataset.py:7,249 and data_generation_and_inference.ipynb cell 2/cell 8 (same loj intersect) plus cell 0's '!uv pip install pyfastx pybedtools' (install cell must skip pybedtools on Windows); (B) bedtools CLI subprocess — plant_helixseek_cre.ipynb cell 11 jaccard call (active; md cells 0/10/22 frozen-convention docs stay bedtools-worded on Linux) and .scratch/select_loci.py:662 (retired gitignored Phase-6 one-shot, OUT of scope); (C) docs-only — selection.md:35 historical methodology, unchanged. Test gates to make platform-aware: tests/examples/test_script_execution.py:128 (_gate_script_pybedtools), tests/examples/test_notebook_execution.py:1397 (_gate_pybedtools), tests/examples/test_plant_helixseek_showcase.py:523 (which('bedtools') fail-loud assert) — on Windows skip/fail becomes numpy-path pass; pyproject 'platform_system != Windows' marker unchanged (Linux-side dependency declaration untouched). | open |  | 2026-10-08T08:00:06.495Z |  |
| 20 | 10 | deviation | dnallm/inference/vep.py |  | Plan 10-04 coverage-proof substitution: pytest --cov=<module> fails repo-wide in the current shared dev venv — coverage 7.16.2 source_pkgs resolution imports dnallm under the active tracer and the utils-shims→torch→numpy 2.5.3 chain trips numpy's 'cannot load module more than once per process' C-extension guard during conftest import (reproduces on untouched test_mutagenesis.py and a minimal Coverage(source=['dnallm'])+import numpy probe; nothing installed by the plan). vep.py measured 100% (62/62) via 'coverage run --include=dnallm/inference/vep.py' — same instrument, no source import at start. CI legs (numpy 1.26.4/2.2.0) unaffected; open question: local venv numpy pin vs coverage interplay fix | fixed | FIXED 2026-10-14 (17-02 Task 1, PUBS-05 AS-1 fallback path): root cause probed live as TWO facets — numpy 2.4-series one-init guard (reload UserWarning became hard ImportError upstream; matrix 2.2.0 OK / 2.3.5 OK / 2.4.6 CRASH / 2.5.3 CRASH, so any future local pin ceiling is <2.4, NOT <2.5) drives the pandas/numpy masking cascade, but the torch facet (torch/overrides.py:1779 'RuntimeError: function _has_torch_function already has a docstring' — overrides double-execution under pytest-cov's dotted-source_pkgs early import) proved INDEPENDENT of numpy version: with the venv pinned numpy==2.2.0 the repro 'pytest tests/inference/test_mutagenesis.py --cov=dnallm.utils.sequence --co -q' cleared the pandas import yet still failed on the torch facet → per the plan's fallback contract numpy was restored 2.5.3 (restore cmd: uv pip install 'numpy==2.5.3') and the documented coverage-CLI forms are THE fix, both proven live post-restore on the shared dev venv: (1) bare --cov reading source_pkgs ['dnallm'] from pyproject — collect-only exits 0, 45 collected; (2) coverage run -m pytest + coverage report — 'coverage run --include=dnallm/inference/vep.py -m pytest tests/inference/test_vep.py -m not slow -q' → 56 passed 5 deselected, report row vep.py 260/260 100% (phase-10 full-run basis: 62/62); fast lane '-m not slow --no-cov' green post-restore (2550 passed / 1 skipped); AS-1 refined: the torch facet is NOT a symptom of the numpy guard | 2026-10-09T11:07:16.415Z | 2026-10-14T11:37:34.000Z |
| 21 | 10 | stub | docs/user_guide/fine_tuning/peft_adapters.md | 163 | IA³ chapter section is an honest forward pointer by plan design (REV-03 split delivery, 10-03 D-08/plan prohibitions): finetune.use_ia3 and the ia3 YAML section exist as config surface, but the trainer branch, per-model target presets, and working examples complete in Phase 11-12 after PEFT-01 — the section names no nonexistent API and must be completed (then marked fixed) when the IA³ trainer branch ships | fixed | Stale-open verified at 2026-10-10 backlog review: IA³ trainer branch + per-family presets shipped in Phase 11 (transformer + Mamba proven, seed-pinned roundtrip) and peft_adapters.md:173+ now carries the full IA³ chapter (YAML example, preset-table selection, save/reload path) — completion condition met | 2026-10-09T11:15:30.000Z | 2026-10-10T14:14:31.103Z |
| 22 | 13 | unrun-verify | .github/workflows/ci.yml |  | Phase 13 D-04 human-check deferred to the runner: first post-merge example-nightly (05:30 UTC schedule or owner dispatch) must come green — stage-1 pair typed-skips network-unavailable with probe evidence, census 216/217 (1 deselected), 4-member stage-4 audit, zero ledger lines; executor did NOT auto-dispatch (a ci.yml workflow_dispatch would co-enqueue the dispatch-only feas-spike job on the self-hosted box); standing local evidence: pair 2 typed skips + audit exit 0 (117s) and coverage-nightly exercising the same typed skip green | fixed |  | 2026-10-11T09:02:07.882Z | 2026-10-13T00:38:07.097Z |
| 23 | 14 | unrun-verify | .github/workflows/ci.yml |  | Phase 14 D-06 CI-side re-verification deferred to the runner: re-dispatched run 38354875692 (headSha 2c737516, ref v1.3) coverage-nightly must come green — first run 38337783698 was red ONLY on the documented rice.uga.edu fasta-truncation transient (osA1_r7.asm.fa.gz 22.1MB of ~115MB, fasta Chr1-Chr2 only; models-lane smoke PASSED 3.400s on CI's transformers — drift refuted); phase close blocked until green per plan 14-02 Task 3 PENDING + 14-EVIDENCE.md section H.4 | fixed |  | 2026-10-13T06:09:34.306Z | 2026-10-13T13:18:38.402Z |
| 24 | 16 | unrun-verify | .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-EVIDENCE.md |  | 16-04 SC4 nightly evidence PENDING — dispatched run 38441258701 queued behind foreign CI work; both nightly jobs must conclude success and be recorded in 16-EVIDENCE.md section A before phase close | fixed |  | 2026-10-13T18:21:31.795Z | 2026-10-14T09:37:03.668Z |
| 25 | 17 | unrun-verify | .planning/phases/17-publish-chain-robustness/17-03-SUMMARY.md | 106 | PyPI trusted-publisher registration for dnallm deferred by owner 2026-10-14 — five exact strings on record in 17-03-PLAN Task 2 / this SUMMARY; resume: register at pypi.org/manage/project/dnallm/publishing then re-run /gsd-execute-phase 17 (17-03 completes in minutes: registration record + README trigger docs + token-deletion carried item) | open |  | 2026-10-14T11:56:57.844Z |  |
| 26 | 18 | unrun-verify | docs/getting_started/installation.md |  | 18-01 pip-side alias full-closure resolve deferred: pip HTTP stack stalls against this box's network (5 attempts: direct PyPI x2, tuna mirror bulk+resolver, pinned dual-index metadata crawl) while uv/gh/curl unaffected; mechanism already live-probed (pip 26.2.1, 18-RESEARCH) + wheel METADATA alias line + guard test + uv parity green; re-run: uv venv --seed /tmp/xp --python 3.12 && /tmp/xp/bin/python -m pip install "$(ls dist/*.whl)[base]" on a healthy-network host | open |  | 2026-10-14T14:13:29.034Z |  |
| 27 | 18 | unrun-verify | pyproject.toml |  | 18-03 Windows fla KDA re-verification deferred by owner (稍后再做, coordinator-relayed 2026-10-14 ~14:5xZ) — WPAR-07 acceptance half + T-18-SC SUS retirement + WINDOWS #18 closure all gate on real-hardware execution (RTX 5080); full checklist in 18-03-SUMMARY.md Task 1 Record; resume on the Windows box from origin/v1.3 (>=399f9f1, the surgery commit): (1) pip install -e ".[fla]" in a fresh venv or the dnallm-cuda env — NO manual triton-windows step, the marker must pull it; (2) pip show triton-windows -> Version: 3.7.1.post27; (3) python -c "import torch; print(torch.__version__, torch.version.cuda)" and record both; (4) python -c "from fla.ops.kda.chunk import chunk_kda; print(callable(chunk_kda))" -> True (equivalent: pytest tests/models/test_plant_helixseek_fla_kernels.py — test_chunk_kda_importable_when_fla_installed gates this import); (5) compile + execute a KDA chunk kernel under the resolved torch (WINDOWS #18 remedy precedent, RTX 5080 sm_120) and record the output; (6) capture env/venv, triton-windows version, torch version + cuda build, import result, kernel evidence, date — then re-run /gsd-execute-phase 18 (18-03 Task 2 closes WINDOWS #18, retires the SUS flag on the evidence, marks WPAR-07) | open |  | 2026-10-14T15:49:25.972Z |  |
| 28 | post-v1.3 | deviation | example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb |  | First Windows GPU full-suite census (2026-10-19, dnallm-cuda env, RTX 5080, 2498P/120F/32S) surfaced the last unguarded bedtools-ecosystem surface: CRE notebook cell 16's presentation-only zoom figure invokes the pgt CLI (pygenometracks), which pyproject gates platform_system != 'Windows' (line 56, matplotlib-cap rationale) — compute path (cell 11 jaccard numpy dispatch) was already platform-split by 16-01. Disposition: sys.platform == 'win32' seam at the render call only (POSIX branch byte-preserved, data-file writes unguarded, committed outputs kept as the Linux-execution record), docs mirror synced byte-identical. census residue taxonomy: ~92 failures were VRAM-exhaustion cascade (monolithic full-suite run on 16GB vs CI's D-07 staged-serial discipline — not a code defect; affected files re-ran green on the freed GPU), 25 were scikit-allel missing from the owner env (installed per pyproject constraint scikit-allel>=1.3.13,<2; test_vep 56P/5S green), this 1 was the real surface gap. Follow-up: 6 residual failures re-run exclusive-GPU fresh-process — 5 were CUDA-context poisoning from the first real OOM (4 re-verified green standalone: ia3 roundtrip, with_config_file, both TestUnload; mcp_functionality excluded from Windows scope per owner directive 2026-10-19 — MCP not tested on this box, same convention family as the ollama disposition); the one genuine >16GB task is test_ia3_training_mamba (mambapy pscan pure-torch scan, modeling_mamba.py:306 slow_forward → pscan.py:169) — dispositioned as a Windows-GPU-only typed skip at test head (sys.platform=='win32' and cuda available → pytest.skip environment-unavailable, 0.37s verified 2026-10-19); Linux nightly (GB10) and Windows CPU runs still execute it unchanged. Also noted: check_docs_sync gives a false DIFFER on Windows autocrlf=true checkouts (git blobs identical; normalize EOL before trusting a local red). | fixed |  | 2026-10-19T09:40:00.000Z |  |
| 29 | 27.1 | deviation | docs/user_guide/preset_loading.md |  | Pre-existing repo-wide ruff format failure (untouched by 27.1-01); plan-level format gate not green, touched files clean; logged in phase deferred-items.md | open |  | 2026-10-28T12:09:18.962Z |  |
| 30 | 27.2 | deviation | scripts/census_compat.py |  | 27.2-03 Rule 1: census_entry now threads the resolved config dir into _remote_record so CrossDNA's auto_map jobs resolve against the checkpoint subdirectory (root-only resolution left the record census-blocked). | open |  | 2026-10-29T04:56:47.578Z |  |
| 31 | 27.2 | deviation | tests/models/test_census_compat.py |  | 27.2-03 Rule 1: disposition pins (_is_failure_record, _needs_disposition) treat a green load+forward as superseding historical previous_status failure stamps, matching the merge_records green-drop rule. | open |  | 2026-10-29T04:56:47.650Z |  |
| 32 | 27.2 | deviation | tests/models/test_compat_docs_consistency.py |  | 27.2-03 Rule 1: blocked-map docs pin no longer requires a non-empty census-blocked roster (the campaign measured it to zero); per-row fail-closed mapping stays armed. | open |  | 2026-10-29T04:56:47.741Z |  |
| 33 | 27.2 | deviation | docs/user_guide/optional_dependencies.md |  | 27.2-04 Rule 3: the plan's verify command yaml.safe_load(nav) cannot parse mkdocs.yml's pre-existing !!python/name: emoji tags - the parse leg runs with yaml.UnsafeLoader (the loader mkdocs config requires); mkdocs build --strict is the independent structural check. | open |  | 2026-10-29T05:15:45.029Z |  |
| 34 | 27.2 | deviation | docs/user_guide/optional_dependencies.md |  | 27.2-04 Rule 1: GENERanno guide section's meta-sentence ('you never claim the model works as-is') replaced with direct plain-language wording per the 2026-10-21 biologist-readable directive. | open |  | 2026-10-29T05:15:45.126Z |  |

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
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-02T05:12:38.864Z",
    "resolved_at": "2026-10-13T02:29:17.247Z",
    "milestone": "v1.1"
  },
  {
    "id": 12,
    "kind": "stub",
    "phase": "05",
    "file": "tests/examples/test_script_execution.py",
    "line": null,
    "description": "environment-unavailable typed skip: generate_bpe_dataset.py pkl-production leg blocked by GAP-1-class remote-code gap (plant-nucleotide-transformer-BPE needs removed 4.x PretrainedConfig defaults on transformers 5.17); self-healing, owner disposition pending",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-02T05:48:40.008Z",
    "resolved_at": "2026-10-13T02:29:17.411Z",
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
    "status": "fixed",
    "reason": "run() label resolution honors configured label_column with 'labels' fallback and descriptive ValueError; red-first regression tests in TestRunLabelColumnResolution; note: census's ['sequence']-only shape stemmed from generate_dataset's silent path-to-sequence fallback, a separate latent bug deliberately left out of scope",
    "recorded_at": "2026-10-02T08:34:21.569Z",
    "resolved_at": "2026-10-10T14:39:32.164Z",
    "milestone": "v1.1"
  },
  {
    "id": 15,
    "kind": "skipped-test",
    "phase": "05",
    "file": "tests/examples/test_notebook_execution.py",
    "line": null,
    "description": "05-06 gated typed skips (sanctioned, self-healing): optional-dep probe-then-execute for evo/megaDNA prerequisites, finetune_custom_head megaDNA demo cell and PlantCAD lora pair (mamba_ssm); environment-unavailable script-lane skip (05-05 pattern) unchanged; all matched by audit_skips against registered prefixes",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-02T08:34:21.655Z",
    "resolved_at": "2026-10-13T02:29:17.575Z",
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
    "status": "fixed",
    "reason": "GitHub receive recovered; orchestrator re-push at 2026-10-07T15:17:44Z landed cfc8346..97a7c30 on origin/dev; CI + Docs Validation re-triggered",
    "recorded_at": "2026-10-07T15:20:00.000Z",
    "resolved_at": "2026-10-07T15:18:00.000Z",
    "milestone": "v1.1"
  },
  {
    "id": 18,
    "kind": "deviation",
    "phase": "quick-261008-env",
    "file": "pyproject.toml",
    "line": null,
    "description": "Windows fla installability gap (env adaptation 2026-10-08): pip install -e '.[fla]' on Windows succeeds but leaves fla broken - torch Windows wheels declare no triton dependency (Linux pulls it) and fla-core requires triton only under its [cuda]/[cpu] extras which the pyproject deliberately avoids, so fla imports top-level but fla.ops.* raises ModuleNotFoundError and test_chunk_kda_importable_when_fla_installed FAILS instead of skipping. Remedy installed in dnallm-cuda: triton-windows==3.7.1.post27 (community wheels; upstream triton ships no win_amd64) - KDA chunk kernel proven compiling+executing on RTX 5080 sm_120. Open owner decision: whether to declare triton-windows behind a Windows platform marker in the fla extra (third-party fork enters declared deps) or document the manual step",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-08T04:30:00.000Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 19,
    "kind": "deviation",
    "phase": "quick-261008-wfx",
    "file": "tests/examples/test_plant_helixseek_showcase.py",
    "line": null,
    "description": "CRE showcase execution test requires the bedtools CLI and fails loud by design (test line ~480 asserts shutil.which('bedtools')); bedtools has no Windows build, so local Windows full-example runs must --deselect test_cre_notebook_executes_within_selection_bands (workaround in daily use since 2026-10-08). Open question: WSL-scope bedtools provisioning or permanent documented deselect; Linux nightly lanes unaffected. EVALUATION 2026-10-10 (owner-directed de-bedtools assessment): pyranges1 REJECTED as replacement — ruranges 0.2.7 ships no win_amd64 wheels (Windows would need Rust msvc source build; the pybedtools platform marker would simply transfer), requires-python >=3.12, and it buys zero Windows gain since CRE jaccard is subprocess-direct CLI anyway (join_overlaps left mode does map to the NER loj intersect, noted for completeness). CHOSEN DIRECTION (refined by owner 2026-10-10, platform-split): Linux/macOS KEEP bedtools/pybedtools — the frozen truth chain stays bedtools-computed, no re-freeze; Windows gets a parity-verified numpy evaluator (precedent dnallm/utils/genomic_coords.py) used only where bedtools cannot install — the CRE jaccard first, the NER example's pybedtools loj intersect second (larger surface: join semantics + ascending-order traversal). OWNER DIRECTIVE 2026-10-10: the NER example CODE ITSELF must change — generate_bpe_dataset.py and data_generation_and_inference.ipynb gain runtime platform detection (Linux keeps pybedtools.BedTool.intersect(loj=True), Windows dispatches to the numpy equivalent); the downstream frozen Anno metric (exon_f1=0.7522) must hold in band on BOTH paths, and the loj replacement needs its OWN parity check (numpy join vs `bedtools intersect -loj` — jaccard parity does not transfer to join semantics). Enabling evidence: quick-task spike 261010-wx2 (numpy vs bedtools v2.31.1 parity on the frozen selection pair jaccard=0.324727 + randomized property cases) — without parity the split would be two truths, so the split ships only if parity holds. v1.3 must pin the frozen-value comparison tolerance (numpy evaluator yields trailing float deltas, e.g. 0.3247270000001) and Windows CI (test-windows) becomes the numpy-path executor; revisit pyranges1 only if ruranges ships win_amd64 wheels. CHANGE SURFACE INVENTORY 2026-10-10 (exhaustive example/ sweep): (A) pybedtools runtime, exactly two files sharing one code path — generate_bpe_dataset.py:7,249 and data_generation_and_inference.ipynb cell 2/cell 8 (same loj intersect) plus cell 0's '!uv pip install pyfastx pybedtools' (install cell must skip pybedtools on Windows); (B) bedtools CLI subprocess — plant_helixseek_cre.ipynb cell 11 jaccard call (active; md cells 0/10/22 frozen-convention docs stay bedtools-worded on Linux) and .scratch/select_loci.py:662 (retired gitignored Phase-6 one-shot, OUT of scope); (C) docs-only — selection.md:35 historical methodology, unchanged. Test gates to make platform-aware: tests/examples/test_script_execution.py:128 (_gate_script_pybedtools), tests/examples/test_notebook_execution.py:1397 (_gate_pybedtools), tests/examples/test_plant_helixseek_showcase.py:523 (which('bedtools') fail-loud assert) — on Windows skip/fail becomes numpy-path pass; pyproject 'platform_system != Windows' marker unchanged (Linux-side dependency declaration untouched).",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-08T08:00:06.495Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 20,
    "kind": "deviation",
    "phase": "10",
    "file": "dnallm/inference/vep.py",
    "line": null,
    "description": "Plan 10-04 coverage-proof substitution: pytest --cov=<module> fails repo-wide in the current shared dev venv — coverage 7.16.2 source_pkgs resolution imports dnallm under the active tracer and the utils-shims→torch→numpy 2.5.3 chain trips numpy's 'cannot load module more than once per process' C-extension guard during conftest import (reproduces on untouched test_mutagenesis.py and a minimal Coverage(source=['dnallm'])+import numpy probe; nothing installed by the plan). vep.py measured 100% (62/62) via 'coverage run --include=dnallm/inference/vep.py' — same instrument, no source import at start. CI legs (numpy 1.26.4/2.2.0) unaffected; open question: local venv numpy pin vs coverage interplay fix",
    "status": "fixed",
    "reason": "FIXED 2026-10-14 (17-02 Task 1, PUBS-05 AS-1 fallback path): root cause probed live as TWO facets — numpy 2.4-series one-init guard (reload UserWarning became hard ImportError upstream; matrix 2.2.0 OK / 2.3.5 OK / 2.4.6 CRASH / 2.5.3 CRASH, so any future local pin ceiling is <2.4, NOT <2.5) drives the pandas/numpy masking cascade, but the torch facet (torch/overrides.py:1779 'RuntimeError: function _has_torch_function already has a docstring' — overrides double-execution under pytest-cov's dotted-source_pkgs early import) proved INDEPENDENT of numpy version: with the venv pinned numpy==2.2.0 the repro 'pytest tests/inference/test_mutagenesis.py --cov=dnallm.utils.sequence --co -q' cleared the pandas import yet still failed on the torch facet → per the plan's fallback contract numpy was restored 2.5.3 (restore cmd: uv pip install 'numpy==2.5.3') and the documented coverage-CLI forms are THE fix, both proven live post-restore on the shared dev venv: (1) bare --cov reading source_pkgs ['dnallm'] from pyproject — collect-only exits 0, 45 collected; (2) coverage run -m pytest + coverage report — 'coverage run --include=dnallm/inference/vep.py -m pytest tests/inference/test_vep.py -m not slow -q' → 56 passed 5 deselected, report row vep.py 260/260 100% (phase-10 full-run basis: 62/62); fast lane '-m not slow --no-cov' green post-restore (2550 passed / 1 skipped); AS-1 refined: the torch facet is NOT a symptom of the numpy guard",
    "recorded_at": "2026-10-09T11:07:16.415Z",
    "resolved_at": "2026-10-14T11:37:34.000Z",
    "milestone": "v1.2"
  },
  {
    "id": 21,
    "kind": "stub",
    "phase": "10",
    "file": "docs/user_guide/fine_tuning/peft_adapters.md",
    "line": 163,
    "description": "IA³ chapter section is an honest forward pointer by plan design (REV-03 split delivery, 10-03 D-08/plan prohibitions): finetune.use_ia3 and the ia3 YAML section exist as config surface, but the trainer branch, per-model target presets, and working examples complete in Phase 11-12 after PEFT-01 — the section names no nonexistent API and must be completed (then marked fixed) when the IA³ trainer branch ships",
    "status": "fixed",
    "reason": "Stale-open verified at 2026-10-10 backlog review: IA³ trainer branch + per-family presets shipped in Phase 11 (transformer + Mamba proven, seed-pinned roundtrip) and peft_adapters.md:173+ now carries the full IA³ chapter (YAML example, preset-table selection, save/reload path) — completion condition met",
    "recorded_at": "2026-10-09T11:15:30.000Z",
    "resolved_at": "2026-10-10T14:14:31.103Z",
    "milestone": "v1.2"
  },
  {
    "id": 22,
    "kind": "unrun-verify",
    "phase": "13",
    "file": ".github/workflows/ci.yml",
    "line": null,
    "description": "Phase 13 D-04 human-check deferred to the runner: first post-merge example-nightly (05:30 UTC schedule or owner dispatch) must come green — stage-1 pair typed-skips network-unavailable with probe evidence, census 216/217 (1 deselected), 4-member stage-4 audit, zero ledger lines; executor did NOT auto-dispatch (a ci.yml workflow_dispatch would co-enqueue the dispatch-only feas-spike job on the self-hosted box); standing local evidence: pair 2 typed skips + audit exit 0 (117s) and coverage-nightly exercising the same typed skip green",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-11T09:02:07.882Z",
    "resolved_at": "2026-10-13T00:38:07.097Z",
    "milestone": "v1.3"
  },
  {
    "id": 23,
    "kind": "unrun-verify",
    "phase": "14",
    "file": ".github/workflows/ci.yml",
    "line": null,
    "description": "Phase 14 D-06 CI-side re-verification deferred to the runner: re-dispatched run 38354875692 (headSha 2c737516, ref v1.3) coverage-nightly must come green — first run 38337783698 was red ONLY on the documented rice.uga.edu fasta-truncation transient (osA1_r7.asm.fa.gz 22.1MB of ~115MB, fasta Chr1-Chr2 only; models-lane smoke PASSED 3.400s on CI's transformers — drift refuted); phase close blocked until green per plan 14-02 Task 3 PENDING + 14-EVIDENCE.md section H.4",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-13T06:09:34.306Z",
    "resolved_at": "2026-10-13T13:18:38.402Z",
    "milestone": "v1.3"
  },
  {
    "id": 24,
    "kind": "unrun-verify",
    "phase": "16",
    "file": ".planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-EVIDENCE.md",
    "line": null,
    "description": "16-04 SC4 nightly evidence PENDING — dispatched run 38441258701 queued behind foreign CI work; both nightly jobs must conclude success and be recorded in 16-EVIDENCE.md section A before phase close",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-13T18:21:31.795Z",
    "resolved_at": "2026-10-14T09:37:03.668Z",
    "milestone": "v1.3"
  },
  {
    "id": 25,
    "kind": "unrun-verify",
    "phase": "17",
    "file": ".planning/phases/17-publish-chain-robustness/17-03-SUMMARY.md",
    "line": 106,
    "description": "PyPI trusted-publisher registration for dnallm deferred by owner 2026-10-14 — five exact strings on record in 17-03-PLAN Task 2 / this SUMMARY; resume: register at pypi.org/manage/project/dnallm/publishing then re-run /gsd-execute-phase 17 (17-03 completes in minutes: registration record + README trigger docs + token-deletion carried item)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-14T11:56:57.844Z",
    "resolved_at": null,
    "milestone": "v1.3"
  },
  {
    "id": 26,
    "kind": "unrun-verify",
    "phase": "18",
    "file": "docs/getting_started/installation.md",
    "line": null,
    "description": "18-01 pip-side alias full-closure resolve deferred: pip HTTP stack stalls against this box's network (5 attempts: direct PyPI x2, tuna mirror bulk+resolver, pinned dual-index metadata crawl) while uv/gh/curl unaffected; mechanism already live-probed (pip 26.2.1, 18-RESEARCH) + wheel METADATA alias line + guard test + uv parity green; re-run: uv venv --seed /tmp/xp --python 3.12 && /tmp/xp/bin/python -m pip install \"$(ls dist/*.whl)[base]\" on a healthy-network host",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-14T14:13:29.034Z",
    "resolved_at": null,
    "milestone": "v1.3"
  },
  {
    "id": 27,
    "kind": "unrun-verify",
    "phase": "18",
    "file": "pyproject.toml",
    "line": null,
    "description": "18-03 Windows fla KDA re-verification deferred by owner (稍后再做, coordinator-relayed 2026-10-14 ~14:5xZ) — WPAR-07 acceptance half + T-18-SC SUS retirement + WINDOWS #18 closure all gate on real-hardware execution (RTX 5080); full checklist in 18-03-SUMMARY.md Task 1 Record; resume on the Windows box from origin/v1.3 (>=399f9f1, the surgery commit): (1) pip install -e \".[fla]\" in a fresh venv or the dnallm-cuda env — NO manual triton-windows step, the marker must pull it; (2) pip show triton-windows -> Version: 3.7.1.post27; (3) python -c \"import torch; print(torch.__version__, torch.version.cuda)\" and record both; (4) python -c \"from fla.ops.kda.chunk import chunk_kda; print(callable(chunk_kda))\" -> True (equivalent: pytest tests/models/test_plant_helixseek_fla_kernels.py — test_chunk_kda_importable_when_fla_installed gates this import); (5) compile + execute a KDA chunk kernel under the resolved torch (WINDOWS #18 remedy precedent, RTX 5080 sm_120) and record the output; (6) capture env/venv, triton-windows version, torch version + cuda build, import result, kernel evidence, date — then re-run /gsd-execute-phase 18 (18-03 Task 2 closes WINDOWS #18, retires the SUS flag on the evidence, marks WPAR-07)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-14T15:49:25.972Z",
    "resolved_at": null,
    "milestone": "v1.3"
  },
  {
    "id": 28,
    "kind": "deviation",
    "phase": "post-v1.3",
    "file": "example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb",
    "line": null,
    "description": "First Windows GPU full-suite census (2026-10-19, dnallm-cuda env, RTX 5080, 2498P/120F/32S) surfaced the last unguarded bedtools-ecosystem surface: CRE notebook cell 16's presentation-only zoom figure invokes the pgt CLI (pygenometracks), which pyproject gates platform_system != 'Windows' (line 56, matplotlib-cap rationale) — compute path (cell 11 jaccard numpy dispatch) was already platform-split by 16-01. Disposition: sys.platform == 'win32' seam at the render call only (POSIX branch byte-preserved, data-file writes unguarded, committed outputs kept as the Linux-execution record), docs mirror synced byte-identical. census residue taxonomy: ~92 failures were VRAM-exhaustion cascade (monolithic full-suite run on 16GB vs CI's D-07 staged-serial discipline — not a code defect; affected files re-ran green on the freed GPU), 25 were scikit-allel missing from the owner env (installed per pyproject constraint scikit-allel>=1.3.13,<2; test_vep 56P/5S green), this 1 was the real surface gap. Follow-up: 6 residual failures re-run exclusive-GPU fresh-process — 5 were CUDA-context poisoning from the first real OOM (4 re-verified green standalone: ia3 roundtrip, with_config_file, both TestUnload; mcp_functionality excluded from Windows scope per owner directive 2026-10-19 — MCP not tested on this box, same convention family as the ollama disposition); the one genuine >16GB task is test_ia3_training_mamba (mambapy pscan pure-torch scan, modeling_mamba.py:306 slow_forward → pscan.py:169) — dispositioned as a Windows-GPU-only typed skip at test head (sys.platform=='win32' and cuda available → pytest.skip environment-unavailable, 0.37s verified 2026-10-19); Linux nightly (GB10) and Windows CPU runs still execute it unchanged. Also noted: check_docs_sync gives a false DIFFER on Windows autocrlf=true checkouts (git blobs identical; normalize EOL before trusting a local red).",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-19T09:40:00.000Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 29,
    "kind": "deviation",
    "phase": "27.1",
    "file": "docs/user_guide/preset_loading.md",
    "line": null,
    "description": "Pre-existing repo-wide ruff format failure (untouched by 27.1-01); plan-level format gate not green, touched files clean; logged in phase deferred-items.md",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-28T12:09:18.962Z",
    "resolved_at": null,
    "milestone": "v1.4"
  },
  {
    "id": 30,
    "kind": "deviation",
    "phase": "27.2",
    "file": "scripts/census_compat.py",
    "line": null,
    "description": "27.2-03 Rule 1: census_entry now threads the resolved config dir into _remote_record so CrossDNA's auto_map jobs resolve against the checkpoint subdirectory (root-only resolution left the record census-blocked).",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T04:56:47.578Z",
    "resolved_at": null,
    "milestone": "v1.4"
  },
  {
    "id": 31,
    "kind": "deviation",
    "phase": "27.2",
    "file": "tests/models/test_census_compat.py",
    "line": null,
    "description": "27.2-03 Rule 1: disposition pins (_is_failure_record, _needs_disposition) treat a green load+forward as superseding historical previous_status failure stamps, matching the merge_records green-drop rule.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T04:56:47.650Z",
    "resolved_at": null,
    "milestone": "v1.4"
  },
  {
    "id": 32,
    "kind": "deviation",
    "phase": "27.2",
    "file": "tests/models/test_compat_docs_consistency.py",
    "line": null,
    "description": "27.2-03 Rule 1: blocked-map docs pin no longer requires a non-empty census-blocked roster (the campaign measured it to zero); per-row fail-closed mapping stays armed.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T04:56:47.741Z",
    "resolved_at": null,
    "milestone": "v1.4"
  },
  {
    "id": 33,
    "kind": "deviation",
    "phase": "27.2",
    "file": "docs/user_guide/optional_dependencies.md",
    "line": null,
    "description": "27.2-04 Rule 3: the plan's verify command yaml.safe_load(nav) cannot parse mkdocs.yml's pre-existing !!python/name: emoji tags - the parse leg runs with yaml.UnsafeLoader (the loader mkdocs config requires); mkdocs build --strict is the independent structural check.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T05:15:45.029Z",
    "resolved_at": null,
    "milestone": "v1.4"
  },
  {
    "id": 34,
    "kind": "deviation",
    "phase": "27.2",
    "file": "docs/user_guide/optional_dependencies.md",
    "line": null,
    "description": "27.2-04 Rule 1: GENERanno guide section's meta-sentence ('you never claim the model works as-is') replaced with direct plain-language wording per the 2026-10-21 biologist-readable directive.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T05:15:45.126Z",
    "resolved_at": null,
    "milestone": "v1.4"
  }
]
````
