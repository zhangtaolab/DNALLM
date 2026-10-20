# DNALLM TUI Opportunity Survey — v1.4 planning input

Quick task 261020-gk4 (analysis only — zero changes to `dnallm/`, `ui/`, `tests/`, `configs/`, `scripts/`).
Sibling platform: `/home/forrest/Github/dnallmmark` (read-only reference; cited with the `dnallmmark:` prefix, never against this checkout).

## Part 0 — Method

- Scanned in this order: (1) the dnallmmark TUI's 9 screens and 16 service modules (module + class docstrings, read directly — current function extracted, history notes ignored); (2) DNALLM's user-facing surfaces — click group commands in `dnallm/cli/cli.py`, the standalone `dnallm/cli/vep.py`, all 7 console scripts in `pyproject.toml`, the 3 `ui/` Gradio apps, the 17-tool MCP roster in `dnallm/mcp/server.py`, the YAML config path (`configs/` + `dnallm/configuration/configs.py`), the model registry (`dnallm/models/model_info.yaml`, `dnallm/models/modeling_auto.py`), and the progress/output UX of training and inference (`dnallm/finetune/trainer.py`, `dnallm/inference/inference.py`, `dnallm/inference/plot.py`, `dnallm/utils/training_plots.py`).
- Every claim below was verified by reading the cited file in this checkout (or, for `dnallmmark:` citations, in the sibling checkout) — not inferred from names.
- Ranking criteria: UX value = user-facing pain × frequency of the workflow. Feasibility (API hooks already present vs. new surface needed) is noted but secondary.
- Candidates that replicate a dnallmmark screen's delivered UX were killed to Part 4 regardless of how attractive they look in isolation.

## Part 1 — dnallmmark exclusion baseline (what its TUI already owns)

Screens (`/home/forrest/Github/dnallmmark/tui/screens/`):

- `dnallmmark:tui/screens/selection.py` — slot-2 MODELS table: mark column, committed compat verdicts, honest per-model download-state column, k9s-style type-to-filter, spacebar marks, preset/template modals, `d` model download through the shared queue's model lane.
- `dnallmmark:tui/screens/run_config.py` — slot-4 run composition: own model/dataset pickers, params pane, dirs section, live same-source dry-run preview line, terraform-style launch-confirmation pipeline, choice modals.
- `dnallmmark:tui/screens/wizard.py` — custom-onboarding wizard modal family: add-model / add-dataset four-step forms, collect-all validation report view, real-subprocess gauntlet panel, session-undo, import/export views.
- `dnallmmark:tui/screens/monitor.py` — slot-5 monitoring, two tabs: Running (one card per live sweep run with manifest scale and progress) and the history view, all rendered from the pure artifact-tree scanner.
- `dnallmmark:tui/screens/matrix.py` — read-only results-density matrix: one row per model, one column per dataset, presence glyphs, full task-key drill-down modal.
- `dnallmmark:tui/screens/data_manager.py` — slot-3 DATASETS screen: full official dataset list with download-state/status columns, persistent five-state download queue with live progress cells, mismatch banners, Zenodo whole-package line.
- `dnallmmark:tui/screens/env_check.py` — environment-check panel: eight-row kernel-results table filled as results arrive, last-check line, advisory gate verdict with missing-data routing, verbatim drill-down modal.
- `dnallmmark:tui/screens/about.py` — read-only info screen: five fixed rows (software / data / milestone / dependencies / repository), wrap-never-elide, tolerant of absent probes.
- `dnallmmark:tui/screens/nav.py` — navigation chrome: digit-prefixed top menu bar (six slots, active-slot marking) and bottom submenu bar (screen keycaps + help/quit).

Services (`/home/forrest/Github/dnallmmark/tui/services/`):

- `dnallmmark:tui/services/about.py` — derives the about-screen rows: version from pyproject + git sha, official-only registry counts, dependency probes via importlib.metadata.
- `dnallmmark:tui/services/custom_entries.py` — the only write seam over user-owned custom model/dataset registries (`~/.config/dnallmmark/*.json`): tolerant read, gauntlet-land, session-undo, enabled-toggle.
- `dnallmmark:tui/services/custom_validation.py` — collect-all candidate validator: every field problem as ordered pairs, never raises on untrusted input shapes.
- `dnallmmark:tui/services/download_queue.py` — download orchestration core: persistent five-state queue, ModelScope channel resolution, presence probe, Popen engine with progress parsing, post-landing verification, process-group cancel.
- `dnallmmark:tui/services/env_check.py` — env-check orchestration + same-day cache: runs the env_smoke kernels, persists structured results under the XDG state root.
- `dnallmmark:tui/services/env_gate.py` — pure gate judgment: classifies kernel failures into CRITICAL (block launch) vs ROUTING (missing data → data management).
- `dnallmmark:tui/services/gauntlet.py` — error-detection gauntlet: dry-run precheck + audit legs as real subprocesses; a custom entry lands only after both pass.
- `dnallmmark:tui/services/quick_check.py` — lightweight pre-launch revalidation of the launch-required kernel subset (everything except the heavyweight matmul smoke).
- `dnallmmark:tui/services/registries.py` — registry view service: canonical model/dataset entry lists, presence mapping, counts; the one-directional tui→pipeline seam.
- `dnallmmark:tui/services/run_config.py` — composition brain: run-spec model + collect-all validation, XDG persistence, dry-run preview, typed-phrase launch verdict, run_sweep argv composition.
- `dnallmmark:tui/services/selection.py` — mark-set / preset / filter grammar: presets derived from the sweep priorities file, custom-preset persistence, the pure filter grammar the selection Input drives.
- `dnallmmark:tui/services/settings.py` — XDG roots + atomic writes: the single persistence-location implementation (state vs. config split).
- `dnallmmark:tui/services/sweep_launcher.py` — launch lifecycle: injectable Popen seam with detached tee launch, live-sweep registry with kill/detach, env preflight, structurally-enforced phrase gate.
- `dnallmmark:tui/services/sweep_monitor.py` — pure artifact-tree scanner: three-state run discovery (finished/running/artifacts-only), tolerant record readers with last-known-good retention, per-(model, task, seed) cell-state derivation.
- `dnallmmark:tui/services/templates.py` — run-template service: strict schema validation with collect-all problem listing, valid-by-construction export, priorities-tier interop.
- `dnallmmark:tui/services/zenodo_bundle.py` — Zenodo whole-package lane: record publication probe (pure state, never persisted) and bundle download/extract driver.

Synthesis — territory dnallmmark's TUI owns end to end: the benchmark sweep lifecycle. Model/dataset selection for sweeps (with compat verdicts and download management through a persistent queue), run composition with dry-run preview and gated launch, live + historical sweep monitoring from artifact trees, results presence matrix, environment checking/gating, custom-entry onboarding through a validation gauntlet, and shell/settings/about chrome. All of it is scoped to its own benchmark registries and its `run_sweep` pipeline — it never loads a DNALLM model for inference, never touches DNALLM training, MCP, interpretation, or variant effect prediction.

## Part 2 — DNALLM user-facing surface inventory (what exists today)

Click group (`dnallm/cli/cli.py`, console script `dnallm`):

- `dnallm train` — refuses flag-only invocation with a UsageError pointing at the generator; with `--config` it loads YAML, builds `DNATrainer`, and runs `train()` to completion. One-shot process: config → full training run → exit.
- `dnallm inference` — same shape: UsageError without `--config`, then one `DNAInference` instance, one `infer()`, exit. Results go to a logger line or a printed path; no inspection loop.
- `dnallm benchmark` — accepts `--config` or bare `--model`+`--data`; constructs `Benchmark` and runs it; prints "Benchmark completed".
- `dnallm mutagenesis` — stub: prints "Mutagenesis analysis completed" without constructing anything (in-source comments admit the wiring is missing). The real implementation lives behind the separate `dnallm-mutagenesis` console script (`dnallm/cli/mutagenesis.py`), so the two entry points disagree.
- `dnallm model_config_generator` — delegates through `dnallm/cli/model_config_generator.py` (a shim) to the root `cli/model_config_generator.py`: numbered click-prompt menus (choose 1-4, paged model numbers, search) that author a NEW config; no editing of existing files, no validation preview before save.
- `dnallm mcp_server` — delegates to the argparse `main` in `dnallm/mcp/server.py`; stdio/sse/streamable-http transports; once launched, the only operator surface is whatever the MCP client shows.

Standalone and console scripts (`pyproject.toml` `[project.scripts]` — all 7: `dnallm`, `dnallm-train`, `dnallm-inference`, `dnallm-model-config-generator`, `dnallm-mcp-server`, `dnallm-mutagenesis`, `dnallm-vep`):

- `dnallm-vep` (`dnallm/cli/vep.py`) — NOT wired into the click group. Options: config / vcf / reference / model_name / source / paradigm / output. Per invocation it loads the model fresh, scores the entire VCF via `evaluate_vcf` with no progress output, then dumps one JSON to stdout or `--output`.
- `dnallm-train` / `dnallm-inference` (`dnallm/cli/train.py`, `dnallm/cli/inference.py`) — legacy one-command wrappers around the same config-driven flows.

Gradio apps (`ui/`):

- `ui/generation_task_app.py` — browser app for interactive generation: load model into a module-global `GLOBAL_STATE` engine, prompt, generate. Keeps the model warm across turns, at the cost of a browser tab and a web round-trip.
- `ui/model_config_generator_app.py` — web form version of the config generator; `ui/run_config_app.py` is its launcher shim.

MCP server (`dnallm/mcp/server.py` — 17 registered tools): `dna_sequence_predict`, `dna_batch_predict`, `dna_multi_model_predict`, `list_loaded_models`, `get_model_info`, `list_models_by_task_type`, `get_all_available_models`, `health_check`, `dna_stream_predict`, `dna_stream_batch_predict`, `dna_stream_multi_model_predict`, `dna_mutagenesis`, `dna_interpret`, `ism_scan`, `hotspots`, `zero_shot_score`. Notably there are NO load/unload tools: `ModelManager` implements `load_model` / `unload_model` / `unload_all_models` / `get_model_status` (`dnallm/mcp/model_manager.py`) but none of that is registered as a tool, so remote operators cannot manage loaded models after startup.

Config workflow: hand-edited YAML (`configs/finetune_config.yaml`, `configs/inference_config.yaml`, `configs/lora_config.yaml`, `configs/ner_config.yaml` as examples) validated by Pydantic models at `load_config` (`dnallm/configuration/configs.py`). An invalid field raises a `ValidationError` straight through the CLI — the user sees a raw traceback naming the Pydantic internals, not a field-level error panel.

Model registry: `dnallm/models/model_info.yaml` — 221 entries (34 pretrained: 23 `mask` + 11 `generation`; 187 finetuned: 115 binary, 19 multiclass, 42 regression, 3 multilabel, 6 classification, 2 token), each with name / repo id / task_type / label metadata. `dnallm/models/modeling_auto.py` maps 37 named families in `PRETRAIN_MODEL_MAPS`, with per-family load quirks handled under `dnallm/models/special/` (12 handler modules, e.g. evo, enformer, gpn, megadna). There is no browsing surface: users must grep the YAML or read docs to discover what exists.

Training UX (`dnallm/finetune/trainer.py`): an HF `Trainer` run — the default tqdm progress line plus callback logging; early stopping via `EarlyStoppingCallback`; optional Optuna hyperparameter search; `report_to` tensorboard/wandb if configured. Visual review is strictly post-hoc: `plot_history` writes `training_loss.png` and `lr_schedule.png` (`dnallm/utils/training_plots.py`) after the run.

Inference / analysis UX (`dnallm/inference/inference.py`, `dnallm/inference/plot.py`): batch inference shows a tqdm bar (`batch_infer`); `infer` / `infer_file` / `generate` / `scoring` are run-once methods returning results to the caller. Everything visual — metrics bars, ROC-style curves, attention maps, embedding projections, attribution charts, mutagenesis heatmaps (`plot_bars`, `plot_attention_map`, `plot_embeddings`, `plot_attributions_token` / `_line` / `_multi`, `plot_muts`) — is serialized by altair to an HTML/PNG file the user opens in a browser. Interpretability (`dnallm/inference/interpret.py`) and ISM/hotspots (`dnallm/inference/mutagenesis.py`) produce arrays that only reach the user through those saved files.

## Part 3 — Ranked TUI opportunities

Ranked by UX value (pain × frequency). R1 = highest.

### R1. Interactive model exploration console — amortize the model load

- Current UX: every question about a sequence pays the full model download + weight load again — `dnallm/cli/cli.py` `inference` builds one `DNAInference`, runs one `infer()`, and exits; the only warm-session alternative is the browser-based Gradio app (`ui/generation_task_app.py`), which keeps the engine in a module-global but only exposes generation, not predict/embed/score.
- TUI value: a persistent terminal session where the model loads once and the user probes freely — predict a pasted sequence, batch a file, pull embeddings or attention, generate completions, switch LoRA adapter — with instant follow-up questions instead of a 30 s-to-minutes reload per prompt. This is the single highest-frequency pain in the inventory: it hits every interactive user on every invocation.
- Non-overlap: `dnallmmark:tui/screens/selection.py` picks models from dnallmmark's own benchmark registries to mark for sweeps (no model is ever loaded for inference there), and `dnallmmark:tui/services/download_queue.py` manages download state only — neither offers a loaded-model REPL for sequence probing, prediction, or embedding.
- API hooks: `DNAInference` already retains the loaded model between calls (`dnallm/inference/inference.py`: `infer`, `batch_infer`, `infer_seqs`, `infer_file`, `generate`, `scoring`, `plot_attentions`, `plot_hidden_states`); `load_model_and_tokenizer` (`dnallm/models/model.py`) for session-start model choice; `Mutagenesis` (`dnallm/inference/mutagenesis.py`) can ride the same warm engine.

### R2. Live fine-tune dashboard — watch the run, not a tqdm line

- Current UX: a training run shows the HF Trainer's single tqdm line and nothing else; loss/lr curves exist only after the fact as `training_loss.png` / `lr_schedule.png` written by `plot_history` (`dnallm/finetune/trainer.py` delegating to `dnallm/utils/training_plots.py`).
- TUI value: a live dashboard during training — streamed loss/lr/metric curves per epoch from a `TrainerCallback`, early-stopping patience state, LoRA trainable-parameter summary, ETA from step rate, and a clean interrupt that still triggers the post-run plot export. Turns "wait hours, then open two PNGs" into an observable, abortable run.
- Non-overlap: `dnallmmark:tui/screens/monitor.py` + `dnallmmark:tui/services/sweep_monitor.py` monitor benchmark sweep processes by scanning their artifact trees after launch — an out-of-process file reader; an in-process HF training loop with live callback events is a different lifecycle entirely, and dnallmmark never trains anything.
- API hooks: `TrainerCallback` (`on_log`, `on_epoch_end`, `on_step_end`) composed in the existing callbacks list of `dnallm/finetune/trainer.py` (where `EarlyStoppingCallback` already assembles at lines 711-730); `trainer.state.log_history`; `plot_history` unchanged for export; `TrainingConfig.report_to` (`dnallm/configuration/configs.py`) as the precedent for a `tui` reporting target.

### R3. MCP server control room — launch, observe, load/unload

- Current UX: the operator starts the server blind via `dnallm mcp_server` (stdio/sse/streamable-http in `dnallm/cli/cli.py`) and afterwards has only the `health_check` / `list_loaded_models` / `get_model_info` tools (`dnallm/mcp/server.py`) — and since load/unload are deliberately not registered as tools, a loaded-model set cannot be changed without restarting the process.
- TUI value: one terminal that launches the server with a transport/host/port picker, then shows a live table of configured vs. loaded models (status, memory), polls `health_check`, streams model-load events, and issues load/unload (in-process through `ModelManager`, since the tool roster intentionally omits it). This is the only surface where a multi-model MCP deployment is operable at all.
- Non-overlap: nothing in dnallmmark touches MCP, serving, or model lifecycle — its closest screens (`dnallmmark:tui/screens/run_config.py`, `dnallmmark:tui/screens/monitor.py`) compose and watch its own sweep processes, not a network server hosting inference engines.
- API hooks: `ModelManager.load_model` / `unload_model` / `unload_all_models` / `get_model_status` / `get_loaded_models` (`dnallm/mcp/model_manager.py`); `DNALLMMCPClient.call` / `acall` for the remote variant (`dnallm/mcp/client.py`); `health_check` + `list_loaded_models` tools (`dnallm/mcp/server.py`); `MCPConfigManager` (`dnallm/mcp/config_manager.py`) for the configured-model list.

### R4. Terminal YAML config validator — errors as a panel, not a traceback

- Current UX: an invalid config raises a Pydantic `ValidationError` straight out of `load_config` (`dnallm/configuration/configs.py`) — the user gets a raw traceback naming validator internals; the guided alternative (`cli/model_config_generator.py` via the `dnallm/cli/model_config_generator.py` shim) is numbered click-prompt menus that only author NEW configs, with no editing of existing ones and no validation preview.
- TUI value: a split-pane editor — YAML on the left, a live collect-all error panel on the right (every bad field at once, mapped to file/line), green-light save, and section awareness for task / inference / finetune / lora / ia3 / vep / sweep. Incremental over the existing generator: it edits existing configs (the common case: tweak last run's YAML) and it previews the exact errors the run would hit, before the run.
- Non-overlap: `dnallmmark:tui/screens/run_config.py` + `dnallmmark:tui/services/run_config.py` compose dnallmmark's own benchmark `RunSpec` (models/datasets/params flattened to `run_sweep` argv) — they never read or write DNALLM's training/inference YAML schema.
- API hooks: the per-section Pydantic constructors invoked inside `load_config` (`TaskConfig`, `InferenceConfig`, `TrainingConfig`, `LoraConfig`, `Ia3Config`, `VepConfig` in `dnallm/configuration/configs.py`); `ValidationError` iteration for the collect-all panel; `model_info.yaml` metadata for model-name suggestions.

### R5. VEP batch runner — progress and hit triage for variant scoring

- Current UX: `dnallm-vep` (`dnallm/cli/vep.py`) loads the model, scores the entire VCF through `evaluate_vcf` with zero progress output (`dnallm/inference/vep.py` has no progress reporting), then dumps one JSON blob to stdout or `--output` — for a large VCF the user stares at a silent terminal, then hand-greps the JSON for strong hits.
- TUI value: per-variant progress with ETA, a running top-scores table (sortable, filterable by CLNVC consequence class), summary statistics (scored / skipped-indel / tokenization-shift fractions the result already carries), and a final export — all without losing the JSON artifact.
- Non-overlap: dnallmmark has no variant-effect surface of any kind — its dataset/monitor screens (`dnallmmark:tui/screens/data_manager.py`, `dnallmmark:tui/screens/monitor.py`) track benchmark data downloads and sweep runs, not genomic variant scoring.
- API hooks: `evaluate_vcf` + the `VepResult.to_dict()` payload (`dnallm/inference/vep.py`); `load_model_and_tokenizer` (`dnallm/models/model.py`); `VepConfig` (`dnallm/configuration/configs.py`) for defaults; the R1 warm-engine pattern applies here too (score additional VCFs without reloading).

### R6. In-terminal interpretation triage — read results without leaving the terminal

- Current UX: attribution, ISM, and hotspot results reach the user only as serialized altair files — `plot_attributions_token` / `_line` / `_multi` and `plot_muts` in `dnallm/inference/plot.py` save HTML/PNG the user must open in a browser; the underlying arrays (`dnallm/inference/interpret.py`, `dnallm/inference/mutagenesis.py`) have no text rendering at all.
- TUI value: triage-grade in-terminal views — top-k attributed positions as labeled bar rows, per-hotspot ISM delta sparklines, a sequence ruler with highlighted residues — enough to decide in seconds whether a run is worth the full browser figure (which remains the export path). Honest feasibility note: a terminal cannot reproduce chart fidelity, so this is deliberately a triage layer, not a replacement.
- Non-overlap: `dnallmmark:tui/screens/matrix.py` renders benchmark result presence glyphs (model × dataset cells) — aggregate sweep coverage, not per-sequence attribution or mutagenesis data.
- API hooks: `DNAInterpret` attribution methods and `_ids_to_tokens` (`dnallm/inference/interpret.py`); `Mutagenesis.evaluate` / `process_ism_data` / `find_hotspots` (`dnallm/inference/mutagenesis.py`); the pre-serialization arrays consumed by `dnallm/inference/plot.py`.

## Part 4 — Excluded: overlaps dnallmmark's TUI

Candidates evaluated and killed because dnallmmark already delivers this UX (per the owner's "don't repeat what dnallmmark implemented" rule):

- Model registry browser with download management — dnallmmark's selection screen already delivers exactly this table UX: filterable model list, compat verdict column, honest per-model download-state column, `d`-key download through a persistent queue (`dnallmmark:tui/screens/selection.py` + `dnallmmark:tui/services/download_queue.py`, whose model lane is explicitly "the dnallm authority"). Even though the underlying catalog differs (dnallmmark's benchmark registries vs. `dnallm/models/model_info.yaml`'s 221 entries), building a DNALLM-side browser would reproduce the same screen; model-discovery needs are absorbed by R1 (session-start model choice) and R4 (model-name suggestions in the config editor).
- Benchmark / sweep run monitoring — covered by the two-tab monitor plus its pure scanner (`dnallmmark:tui/screens/monitor.py`, `dnallmmark:tui/services/sweep_monitor.py`, `dnallmmark:tui/services/sweep_launcher.py`).
- Dataset download management — covered by the datasets screen and its queue/Zenodo lanes (`dnallmmark:tui/screens/data_manager.py`, `dnallmmark:tui/services/download_queue.py`, `dnallmmark:tui/services/zenodo_bundle.py`).
- Environment check / launch gating — covered by the env-check panel and its kernel services (`dnallmmark:tui/screens/env_check.py`, `dnallmmark:tui/services/env_check.py`, `dnallmmark:tui/services/env_gate.py`, `dnallmmark:tui/services/quick_check.py`).
- Results matrix viewing — covered by the read-only density matrix (`dnallmmark:tui/screens/matrix.py`).
- Custom-entry onboarding wizard (add-your-own model/dataset with validation) — covered by the wizard family and its services (`dnallmmark:tui/screens/wizard.py`, `dnallmmark:tui/services/custom_entries.py`, `dnallmmark:tui/services/custom_validation.py`, `dnallmmark:tui/services/gauntlet.py`).
- About / dependency info panel — covered by `dnallmmark:tui/screens/about.py` + `dnallmmark:tui/services/about.py`.

Non-UX defect noticed during inventory (not a TUI item, recorded for v1.4 triage): the `dnallm mutagenesis` click subcommand is a stub that prints success without running anything (`dnallm/cli/cli.py`), while the real flow lives only under the `dnallm-mutagenesis` console script (`dnallm/cli/mutagenesis.py`) — the two entry points disagree.

