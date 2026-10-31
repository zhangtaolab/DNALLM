# Feature Research

**Domain:** Pre-flight UX (data validation → model-suitability → config generation/validation → download) + registry-as-truth-source for DNALLM v1.4
**Researched:** 2026-10-20
**Confidence:** HIGH for internal repo facts (registry schema, `DNADataset` surface, `download_model` signature, config validation, device chain — read directly in this checkout); MEDIUM for ecosystem UX patterns (cross-verified across official docs + independent secondary sources); the one ModelScope metadata-API finding rests on thinner sourcing and is flagged MEDIUM-LOW inline

"Features" here are the v1.4 Track A target capabilities from PROJECT.md §Current Milestone. "Users" are plant/animal/microbe genomics researchers who bring their own data, fine-tune or run DNA LLMs over SSH on GPU servers, and today hit raw Pydantic tracebacks, silent tokenizer/charset mismatches, and multi-GB downloads that precede a compatibility failure. This file answers: how do comparable mature tools (bioinformatics workflow engines, docker/uv-style init wizards, ML experiment trackers, Textual apps, the sibling dnallmmark TUI) shape each stage of the pre-flight flow, and what is table stakes vs differentiator vs anti-feature for a v1 — so REQUIREMENTS.md can scope REQ-IDs that are testable and atomic.

Builds directly on `.planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md` Part 3 R4 and Part 5 §5.3 (the four-capability pipeline table: 数据整理/数据规范/适配检测/配置生成校验 → one wizard). Owner scope pins: Track A only, ModelScope-first, lab models (`zhangtaolab/*`) first, API-first core (`validate_data` / `check_compatibility` / `generate_config` as pure functions).

---

## How comparable tools shape this flow (evidence base)

Confidence: MEDIUM (each pattern cross-verified across ≥2 sources; links in §Sources).

| Tool | Pre-flight pattern | What DNALLM should copy |
|------|--------------------|-------------------------|
| `docker init` | Detects platform via signature files (pyproject.toml, package.json…), falls back to manual pick; asks 3–6 questions **with defaults in parens**; generates artifacts; warns unrecoverably before overwriting existing files | Detect-first-then-ask (sniff the data file format/columns before prompting); every wizard question ships a pre-filled default; never overwrite an existing config without an explicit warning |
| nf-core / nf-schema (Nextflow) | `validateParameters()` runs **before any process executes**; logs **all** failed parameters together, then throws once (collect-all-then-stop); explicit two-stage order: params validation → input-data validation → execute; knobs `lenientMode`, `failUnrecognisedParams` | Validate everything up front, report every problem in one pass, never the first-error-only traceback; keep params-level and data-level validation as distinct stages with distinct messaging |
| pandera (data validation) | Default fail-fast `SchemaError`; opt-in `lazy=True` collects all failures into a plural `SchemaErrors` with a `failure_cases` summary table (reason code → count) | The lazy/collect-all mode is the model for `validate_data`'s return shape: structured issue list, not an exception; summary counts by reason category |
| Snakemake `-n` dry-run | Builds the DAG, reports missing input files and execution reasons **before** spending compute; cannot catch everything — a green dry run is explicitly not a green run | Pre-flight's honesty contract: check everything checkable cheaply (existence, schema, sizes, metadata), and be explicit about what only the real run can prove |
| `wandb init` | Interactive entity/project picker; explicit flags to skip the wizard entirely; first-run credentials prompt kept separate from config authoring | Wizard is optional and complements, never replaces, the flag/YAML path; keep auth/credentials out of the config wizard |
| Ultralytics settings | Non-interactive persistent JSON settings (`yolo settings key=value`, `reset`); `yolo checks` prints a system/hardware snapshot as a separate verb | Pre-flight's environment panel is a read-only snapshot verb, not a settings manager; wizard state (last data file, last model) persisted separately from user configs |
| Textual apps (terok wizard PR #797, workers guide) | Multi-step wizard = one `ModalScreen` per step with typed dismiss results + sentinel objects for Back-with-prefill; validation shown as inline per-field errors on submit; long work in `@work(thread=True, exclusive=True)` workers; workers are tied to DOM nodes — popping a screen cancels its downloads; `ProgressBar.update(progress=, total=)` + `call_from_thread` for thread→UI | The concrete TUI mechanics for all four panels; cancellation for free by scoping download workers to the screen |
| Hugging Face Hub metadata | `model_info().safetensors.total` → param count → VRAM estimate; `model_info(files_metadata=True)` + sum of `sibling.size` → actual disk bytes; caveat: param×4 overstates disk ~6× for quantized checkpoints | Two distinct size numbers: disk-before-download (blob sizes) and VRAM-to-load (param count) — both belong in the suitability verdict |
| ModelScope Hub | REST `GET /api/v1/models/{repo}/repo/files` returns per-file `Path`/`Size`; `GET /api/v1/models/{id}` returns `StorageSize` (often 0 in list responses — fetch detail per model). *Confidence MEDIUM-LOW: SDK-level per-file-size access via `HubApi` not confirmed by docs.* | Availability/size probing is possible ModelScope-first, but the size-probe seam needs a phase-level spike; treat "list returns StorageSize=0" as a known trap |
| dnallmmark TUI (sibling, read-verified) | Persistent five-state download queue with progress parsing; env-check as an eight-row kernel table with CRITICAL-vs-ROUTING advisory gate; wizard framework with collect-all validation reports; selection screen with compat-verdict + download-state columns | These are the merge-or-build baseline (survey §5.4): Track A should build API-first and leave the full persistent queue/substrate to the Track B merge — do not re-implement what will be merged |

---

## Feature Landscape

### Table Stakes (Users Expect These)

Missing any of these and the pre-flight flow feels broken or dangerous. All are prerequisites for the milestone goal ("one terminal flow from own data to validated, model-compatible config").

| # | Feature | Why Expected | Complexity | Notes & Existing Dependencies |
|---|---------|--------------|------------|-------------------------------|
| TS-1 | **Read-only data profile panel**: row count, sequence-length distribution (min / median / p95 / max, plus truncation-relevant histogram), charset census (count per character: ACGT, IUPAC ambiguity codes, N, digits, whitespace, junk), GC-content range, label cardinality / label-values census | Every data tool shows what it loaded before acting on it (docker init detects before asking; snakemake -n shows the DAG). Today `DNADataset` either mutates silently or shows 10 rows via `head()` | MEDIUM | **Gap:** `DNADataset.validate_sequences` (data.py:864) is a *mutating filter* (drops whole rows, prints one warning line) — there is no read-only scan. Build `validate_data()` as a pure profiling pass reusing `load_local_data` loaders (fasta/csv/tsv/json/parquet/txt/pkl — all exist), `check_sequence` + `calc_gc_content` (`utils/sequence.py`), `_extract_labels`, `get_split_lengths` (exist). Must run in a worker (large FASTA scans are slow) |
| TS-2 | **Collect-all validation report, never a raw traceback**: every problem at once — file/format errors, empty/short/long sequences, charset violations, label problems (single-class labels, mismatched `num_labels`, bad multilabel separators) — each with a severity, a reason code, and row-count impact | nf-schema logs all failed params then throws once; pandera `lazy=True` is the established data-validation UX; TUI survey R4 documents today's failure mode (raw Pydantic traceback naming validator internals). Users over SSH cannot iterate one-error-per-run | MEDIUM | Depends on: TS-1's scan + iterating `pydantic.ValidationError.errors()` from `load_config` (exception already carries all field errors — DNALLM just dumps it as a traceback today). Issue list shape should mirror pandera's `failure_cases` (reason → count → sample rows). **Do not** build on top of `validate_sequences`'s drop-rows behavior — that destroys the evidence the panel needs to show |
| TS-3 | **Explicit three-state verdicts with reasons**: every check reports OK / WARN / FAIL plus one human sentence (and a machine-readable reason code), never a bare boolean | dnallmmark's env_gate classifies CRITICAL (block) vs ROUTING (advise); nf-core separates errors from warnings (`lenientMode`). A 236-model registry guarantees partial-match situations that must render as "usable with caveats," not false FAILs | LOW | Pure judgment layer over TS-1/TS-2/TS-4 outputs. Enforce: WARN never blocks progression (see AF-7) |
| TS-4 | **Model-suitability verdict: task × length × tokenizer/charset** (the milestone's `check_compatibility` core): (a) task_type match between user goal and registry entry (binary/multiclass/regression/multilabel/token/mask/generation); (b) user length distribution vs model max context (report *fraction of sequences truncated* at candidate `max_length`, not just min/max compare); (c) tokenizer family (BPE / k-mer / singlebase) vs data charset — incl. the documented 13 strict-charset families hazard (data.py docstring) and the case-sensitivity rules of `check_sequence` | The survey §5.2 ranks "silently wrong results from tokenizer/charset mismatch" and "12.3 GB download before a task/length incompatibility surfaces" as today's worst failure modes; no comparable DNA-LLM tool checks any of this — for DNALLM users this is the reason pre-flight exists | MEDIUM-HIGH | **Critical dependency — registry metadata gap:** `model_info.yaml` carries `task_type`/`num_labels`/`label_names`/`threshold`/`base_model` for finetuned entries (structured, 187/187 after 261020-kwt) but tokenizer type appears **only in free-text `describe` strings** ("…with 6mer tokenizer") and there is **no structured max_length/size field at all**. The verdict layer needs either (a) registry enrichment (adds `tokenizer_type`, `max_length` per entry — this is the registry-as-truth-source workstream's data-layer work) or (b) on-demand probing of the model's `config.json` (network-dependent, needs cache). Recommend (a) for the 236 known entries + (b) fallback for unknown repos. Length check consumes TS-1's distribution |
| TS-5 | **Availability dimension: source / size / cache / revision, ModelScope-first**: per candidate model — already-cached-locally? (scan the HF/ModelScope cache dirs), repo reachable on ModelScope (HF fallback), download size **before** committing (blob-size sum, not param estimate), pinned/default revision | Milestone scope pins availability as an explicit `check_compatibility` dimension; wasted-download avoidance is the survey's top-ranked pre-flight pain; HF/ModelScope metadata APIs make size-before-download standard practice | MEDIUM | Depends on: registry repo ids (exist), local cache probing (new but trivial), ModelScope/HF metadata APIs. **Spike needed** on ModelScope per-file sizes via `HubApi` (REST documented; SDK path unconfirmed — MEDIUM-LOW). `download_model` (model.py:410) already takes `revision` + classifies failure reasons (network / not-found) — reuse its reason vocabulary in the UI. HF caveat from research: param_count×4 overstates disk ~6× for quantized checkpoints — use blob sizes for disk, params for VRAM |
| TS-6 | **Device/environment panel**: GPU presence + VRAM, device selection order shown (CUDA→MPS→XPU→CPU already in `_get_device`, model.py:788), torch/transformers/numpy versions, disk free vs pending download size, optional-dependency presence (bitsandbytes, captum…) | First panel of the wizard (survey §5.4: env-check doubles as pre-flight's first panel); `yolo checks` establishes the snapshot-verb precedent; VRAM-vs-model-size is the cheapest FAIL to detect | LOW | All detection primitives exist (`_get_device`, `torch.cuda`, `importlib.metadata`). VRAM estimate needs TS-5's param count or registry size field. Render as a static table filled as probes complete (dnallmmark's eight-row pattern), not a blocking gauntlet |
| TS-7 | **Config generation emitting *validated* YAML**: form fields pre-filled from registry metadata (finetuned entries contribute `num_labels`/`label_names`/`threshold` verbatim — that's registry-as-truth-source payoff), live collect-all validation against the same Pydantic models `load_config` uses, green-light gate before save, plus round-trip **load existing YAML → edit → revalidate** (the common case is tweaking last run's config, per survey R4) | The current numbered-menu generator authors-new-only with no validation preview; docker init generates artifacts with overwrite warnings; nf-core launch validates the params file it helps you write. Emitting a config that then tracebacks on `dnallm train` would reproduce the exact UX this milestone exists to kill | MEDIUM | Depends on: TS-2's error unwrapping + TS-4's metadata for defaults + the Pydantic section classes (`TaskConfig`, `TrainingConfig`, `InferenceConfig`, `LoraConfig`… all exist in `configuration/configs.py`). Save path must refuse when validation fails. Overwrite-existing-config requires an explicit warning (docker init precedent). Note `InferenceConfig.max_length=512` default interacts with TS-4b — the form should propose a data-derived value |
| TS-8 | **Keyboard-complete navigation: Back/Next with prefill, resume state** | Users over SSH iterate (fix data → recheck → re-pick model); terok's sentinel-object Back-with-prefill pattern is the Textual idiom; every wizard studied supports revisiting earlier steps | LOW-MEDIUM | TUI mechanics only (ModalScreen stack + typed dismiss results). Wizard session state (last file, last verdicts) persisted via XDG state root (dnallmmark `settings.py` pattern) — but v1 can keep it in-memory if scope-pressured |
| TS-9 | **Sequential download with progress + cancel** for the selected model: one model at a time, byte/file progress, cancellable, failure reason surfaced in the panel (reusing `download_model`'s reason classification) | A wizard that ends in "now go run `snapshot_download` yourself" fails the one-flow goal; queue *feedback* is in the milestone target list | MEDIUM | `download_model(model_name, downloader, revision, max_try=10, allow_patterns)` exists but is **blocking with no progress callback** — needs a thin progress-bridge wrapper (tqdm hook or xet/hf_transfer callback → Textual worker). Cancellation: scope the download worker to its screen so popping cancels it (Textual worker-DOM binding). ModelScope-first downloader selection already exists in the loading chain |
| TS-10 | **API-first core consumable without the TUI**: `validate_data` / `check_compatibility` / `generate_config` as pure, pytest-coverable functions in `dnallm/` (no Textual imports at the core layer); TUI is one shell | Milestone goal verbatim ("one-core-many-shells"); the >90% coverage gate makes a TUI-entangled core untestable by house rules (every `dnallm/` change ships with pytest coverage — owner rule); future MCP tools + notebooks consume the same verdicts | LOW (design discipline, enforced continuously) | The dnallmmark architecture (screens presentation-only over services) is the proven local template. Keep `dnallm/tui/` behind the `tui` extra; core functions importable from base install |

### Differentiators (Competitive Advantage)

None of these is required for the flow to function; each converts registry-as-truth-source from a data-layer chore into user-visible value. Aligned with PROJECT.md Core Value for v1.4.

| # | Feature | Value Proposition | Complexity | Notes & Existing Dependencies |
|---|---------|-------------------|------------|-------------------------------|
| D-1 | **Truncation-fraction model ranking**: rank candidate models by what fraction of *the user's actual sequences* survive un-truncated (p95/p99 vs max context), shown as a per-model column in the suitability shortlist | Turns TS-4b from a per-model binary into a dataset-aware ranking — no comparable DNA-LLM or general ML tool ranks by *your* length distribution; this is only possible because the registry carries max-context metadata (registry-as-truth-source payoff) | LOW once TS-4 metadata exists | Pure arithmetic over TS-1's distribution + TS-4's registry field. Highest value-per-LOC item in this file |
| D-2 | **Tokenizer dry-run on a sample**: empirically tokenize ~100 sampled sequences with the candidate's tokenizer; report actual truncation rate, unknown-token rate, and charset rejections (per strict-charset family) | Metadata checks (TS-4c) can lie or be stale; an empirical sample run catches k-mer/BPE/charset mismatches the registry can't express — the "silently wrong results" failure mode from survey §5.2 | MEDIUM-HIGH | Depends on tokenizer loading (`AutoTokenizer` → `PreTrainedTokenizerFast` → `DNAOneHotTokenizer` fallback chain exists in model.py:543-568). Cost: needs the tokenizer files (small download) — gate behind the availability check. Sample must be seeded/deterministic for reproducibility |
| D-3 | **base↔finetuned lineage routing**: when the user's task/labels don't match any finetuned entry, suggest the matching *base* model for fine-tuning; when they do match, surface the finetuned head with its frozen `label_names`/`threshold` already wired into the config | The registry's unique asset (187 finetuned entries, `base_model` backfilled 261020-kwt); guides the fine-tune-vs-use-pretrained decision no other tool makes explicit; directly serves lab-models-first | MEDIUM | Depends on: registry `base_model` axis (exists, complete) + TS-4a task matching. Label-set comparison against `label_names` (exists per finetuned entry). Render as a two-branch suggestion in the suitability panel |
| D-4 | **Lab-models-first presentation**: every model list ranks `zhangtaolab/*` entries first, with availability/size columns making "smallest lab model that passes" the path of least resistance | Owner constraint made systematic; the existing generator already sorts zhangtaolab-first (verified: `_sort_models_by_priority` in `cli/model_config_generator.py`) — extend the same policy to TUI lists with the new availability columns | LOW | Registry name prefixes + existing sort precedent. Combine with D-1 so the default sort is "lab model, passes, least truncation" |
| D-5 | **Charset-compatibility matrix row per strict family**: in the data panel, show pass-rate of the user's sequences against each of the ~13 strict-charset families (and the common-subset warning) so cross-model choices are honest from the start | The `validate_sequences` docstring documents the cross-model comparability hazard ("each model silently evaluates a DIFFERENT subset"); pre-flight is the only place this can be shown *before* results diverge | LOW-MEDIUM | Depends on TS-1 charset census + a per-family charset table (derivable from the 236 registry entries' tokenizer types — same metadata work as TS-4c) |
| D-6 | **Same core exposed as MCP tools / notebook functions** (post-TUI shells) | One-core-many-shells: `check_compatibility` as an MCP tool lets LLM agents pre-flight before `dna_predict`; notebooks get a validation cell | LOW if TS-10 holds | Pure additional consumers of the TS core; scope-guard: v1.4 Track A delivers the *seams* (stable signatures + docs), full MCP registration can trail |
| D-7 | **Offline/degraded mode**: every network-dependent check (TS-5 availability, D-2 dry-run) individually skippable with an explicit "skipped — network" verdict state rather than an error | SSH users behind GFW/mirrors (HF_ENDPOINT=hf-mirror.com is already a first-class toggle in model.py:380-391) hit hub connectivity constantly; wizard must never dead-end on a probe | LOW | Design rule: availability checks return UNKNOWN state, never raise, when network fails; cache last-known results (dnallmmark env-check same-day cache pattern) |

### Anti-Features (Commonly Requested, Often Problematic)

Documented to prevent scope creep. Track A ends at: validated config + selected model downloaded.

| # | Anti-Feature | Why Requested | Why Problematic | Alternative |
|---|--------------|---------------|-----------------|-------------|
| AF-1 | **Mutating/“auto-fixing” user data from the wizard** (strip bad chars, drop rows, rewrite files in place) | "Just clean it for me" is the obvious ask once the panel shows the dirt | Destroys user data trust; recreates `validate_sequences`' silent-row-drop hazard at file level; makes validation results unreproducible (the data changed underneath the verdict) | Report issues + emit the *exact* suggested code/config (e.g., the `validate_sequences(valid_chars=…, minl=…)` call with drop counts) as a copyable artifact; user applies it deliberately. Pre-flight analysis is strictly read-only |
| AF-2 | **Launching training/inference from the wizard** | "One flow" tempts "…and then run it" | That is Track C (R2 dashboard) territory; couples pre-flight validation state to run-lifecycle concerns; wizard would need to survive a hours-long training process | The wizard's terminal screen prints the exact `dnallm train --config <emitted.yaml>` command (and offers to copy it). Clean handoff, zero lifecycle coupling |
| AF-3 | **Benchmark features** (multi-model × multi-dataset matrices, sweep composition) | Registry + shortlist looks like dnallmmark's selection screen | Explicitly out of owner scope (Track B/downstream); the matrix is a benchmark concept (survey §5.4 keeps it downstream) | Pre-flight shortlist is task-scoped and data-driven; benchmark browsing waits for the Track B merge decision |
| AF-4 | **Full registry browser (all 236 entries, all metadata)** | "While you're in there, let me browse" | Duplicates the Track B merge target (dnallmmark selection screen) — building it fresh in Track A is the duplicate development the owner prohibited | Suitability shortlist only: entries filtered+ranked by the user's data/task (D-1/D-4); a bare "browse all" affordance can defer entirely |
| AF-5 | **Building a YAML text editor** | "Let me hand-edit the YAML inside the TUI" | A from-scratch editor is a tar pit (syntax, undo, scrolling) and re-fights what structured forms + validation preview already solve; raw-editing hides validation state | Structured form fields + a **read-only rendered YAML preview pane** that live-updates (survey R4's split-pane, minus the editing half); real hand-edits happen in the user's editor, and TS-7's round-trip revalidates on load |
| AF-6 | **Silent background network probes** (availability checks firing on every screen change) | Fresh data feels more accurate | Surprise hub traffic on metered/flaky SSH links; slow wizard stalls attributed to "it's hanging"; mirror-endpoint users get inconsistent results | Availability is an explicit user-triggered action with visible per-model results + D-7's cached/skipped states |
| AF-7 | **Hard blocking on WARN verdicts** (wizard refuses to continue on CPU, or on a 5% truncation rate) | "Safety" | nf-core's CRITICAL-vs-ROUTING split exists precisely because over-gating trains users to bypass the tool; researchers legitimately run CPU-small-models and truncated-tail datasets | Three-state verdicts (TS-3): only FAIL blocks progression; WARN shows impact + continues; the user's informed choice is recorded in the emitted config (e.g., `max_length` set with a comment) |
| AF-8 | **A persistent multi-download queue with retries/resume in Track A** | dnallmmark's five-state queue is impressive and mergeable | Track A needs one model downloaded with feedback; a persistent queue is substrate (Track B merge, survey §5.4 verdict MERGE) — building it fresh now is duplicate development | TS-9's sequential download now; adopt the merged queue later without API churn (keep `download_model` as the core seam) |
| AF-9 | **Re-implementing validation logic inside the TUI layer** | Screens "just need a quick check" | Breaks TS-10 (one-core-many-shells), makes the coverage gate unmeetable for the logic, and forks truth between CLI/TUI/MCP futures | Every verdict displayed in the TUI is a return value of the API core; screens contain zero domain logic (dnallmmark's presentation-only screens rule) |

---

## Feature Dependencies

```
[TS-10 API-first core]  (design constraint over everything — core functions before/with any screen)

[TS-1 data profile] ──requires──> existing DNADataset loaders + check_sequence/calc_gc_content (exist)
      │
      ├──enables──> [TS-2 collect-all validation] ──requires──> pydantic ValidationError iteration (exists)
      │                    │
      │                    └──enables──> [TS-7 config gen+validate] ──requires──> Pydantic config classes (exist)
      │                                      │                        registry metadata prefill
      │                                      │
      ├──enables──> [TS-4 suitability: task×length×tokenizer]
      │                    │    └──requires──> registry enrichment: structured tokenizer_type + max_length
      │                    │                    (MISSING today — registry-as-truth-source data-layer work,
      │                    │                     with on-demand config.json probe as fallback)
      │                    │
      │                    └──enables──> [D-1 truncation ranking]  [D-3 lineage routing]  [D-5 charset matrix]
      │                                     (D-3 also requires base_model axis — exists, complete)
[TS-6 env panel] ──requires──> _get_device + importlib.metadata (exist)
      │              └──enhances──> [TS-5 availability] (disk-free vs size; VRAM vs params)
      │
[TS-5 availability: source/size/cache/revision]
      │    └──requires──> registry repo ids (exist) + ModelScope/HF metadata APIs (SPIKE: ModelScope per-file sizes)
      │    └──enhances──> [D-2 tokenizer dry-run] (small tokenizer download gated on reachability)
      │    └──enables──> [TS-9 download w/ progress]
      │                      └──requires──> download_model (exists) + progress-callback bridge (NEW, thin)
      │
[TS-3 verdict states] ──requires──> outputs of TS-1/2/4/5 (pure judgment layer)
[TS-8 wizard nav]     ──independent (TUI mechanics; composes with all panels)
[D-7 offline mode]    ──cross-cuts──> TS-5, D-2 (UNKNOWN-not-raise rule)

[AF-2 config handoff] means: no dependency edge from any pre-flight feature into training/inference run paths
```

### Dependency Notes

- **TS-4 requires registry enrichment — the critical-path gap.** The registry's structured fields stop at `task_type`/`num_labels`/`label_names`/`threshold`/`base_model`; tokenizer type lives only inside free-text `describe` strings and no entry carries max-context or size. Suitability verdicts on the 236 known entries therefore depend on the registry-as-truth-source workstream landing *structured* `tokenizer_type` + `max_length` (size desirable) first, or falling back to per-model `config.json` probes (network-bound, cache-mandatory). This ordering constraint should shape the roadmap: registry data layer before the verdict API's non-empirical paths.
- **TS-1/TS-2 build only on existing loaders — no new data-layer work.** The new surface is a read-only scanner + structured issue list; the mutation-style `validate_sequences` is deliberately *not* the foundation (it drops the rows the panel must count).
- **TS-9's only new code is a progress bridge.** `download_model` already handles retry/reason-classification/revision/allow_patterns; the bridge converts downloader progress (tqdm hook) into worker messages. Cancellation rides Textual's worker-DOM binding.
- **D-3 is unlocked already.** The `base_model` axis is complete (187/187 after quick task 261020-kwt) — lineage routing needs no registry work beyond TS-4a's task matching.
- **D-2 conflicts with nothing but costs network.** It is the only differentiator that downloads anything during validation; gate it behind TS-5's reachability result and D-7's skip affordance.
- **TS-10 (AF-9's positive form) conflicts with any TUI-first implementation order:** if screens land before the core, the coverage gate and the future MCP/notebook shells both regress. Core functions must land in the same phase as or before their first screen.

---

## MVP Definition

### Launch With (v1 — Track A)

Minimum to validate the milestone goal: *one terminal flow from own data to a validated, model-compatible config*.

- [ ] TS-10 API core: `validate_data()` / `check_compatibility()` / `generate_config()` pure functions, pytest-covered, no Textual imports — the milestone's testability and one-core-many-shells foundation
- [ ] TS-1 read-only data profile (stats + charset census + length distribution) — the flow's first user-visible payoff and the input to every later verdict
- [ ] TS-2 collect-all validation report with severities — kills the raw-traceback failure mode, the survey's #1 pre-flight pain
- [ ] TS-4 suitability verdicts (task × length × tokenizer) over registry metadata — requires the registry enrichment (tokenizer_type/max_length) to land first; the flow's namesake capability
- [ ] TS-5 availability dimension (cache/ModelScope-first reachability/size-before-download) — prevents the wasted-12.3-GB-download class; includes the ModelScope size-probe spike outcome
- [ ] TS-7 config generation with live validation + edit-existing round-trip — the flow's output artifact; absorbs survey R4 verbatim
- [ ] TS-3 three-state verdicts with WARN-never-blocks — the honesty contract the whole flow hangs on
- [ ] TS-9 sequential download with progress + cancel (single-model; AF-8 defers the queue) — completes "select-to-download via ModelScope-first path"
- [ ] TS-6 env panel (device/VRAM/versions/disk) — cheap, and the wizard's opening screen
- [ ] Registry runtime consumption (`load_preset_model` resolving finetuned entries with heads/labels/threshold; base↔finetuned axis) — milestone goal verbatim; also supplies TS-7's prefill
- [ ] D-4 lab-models-first ranking — owner constraint, near-zero cost on top of TS-4

### Add After Validation (v1.x)

- [ ] D-1 truncation-fraction model ranking — once TS-4 metadata is proven stable; highest value-per-LOC
- [ ] D-2 tokenizer dry-run on a sample — once TS-5 reachability gating is field-proven; converts metadata trust into empirical trust
- [ ] D-3 base↔finetuned lineage routing — once task matching (TS-4a) has real-user feedback
- [ ] D-5 strict-family charset matrix — once TS-1 census + TS-4c tokenizer table coexist
- [ ] TS-8 persisted wizard resume state (in-memory Back/prefill ships in v1; cross-session resume trails)
- [ ] D-6 MCP tools over the same core — after signatures stabilize through TUI usage
- [ ] Track B merges (persistent queue, env-check gauntlet, wizard framework consolidation) — separate track, consumes this core

### Future Consideration (v2+)

- [ ] Dataset *download* management for preset datasets (ModelScope-hosted `PRESET_DATASETS`) — Track B split-merge territory; v1 pre-flight covers the user's *own* files only
- [ ] Benchmark-oriented surfaces (Track C / dnallmmark)
- [ ] Auto-applied data cleanup recipes (AF-1's controlled descendant: emitted-and-reviewed transforms, never silent)

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Sequencing note |
|---------|------------|---------------------|----------|-----------------|
| TS-10 API core | HIGH (enables all) | LOW | P1 | First — everything consumes it |
| TS-1 data profile | HIGH | MEDIUM | P1 | After/with core; feeds all verdicts |
| TS-2 collect-all validation | HIGH | MEDIUM | P1 | With TS-1 (same scan) |
| Registry enrichment (tokenizer_type/max_length) | Indirect HIGH | MEDIUM | P1 | Before TS-4's metadata paths — critical-path gap |
| TS-4 suitability verdicts | HIGH (the flow's namesake) | MEDIUM-HIGH | P1 | After registry enrichment + TS-1 |
| TS-5 availability | HIGH | MEDIUM | P1 | Spike ModelScope size probe early |
| TS-7 config gen + round-trip | HIGH (the output artifact) | MEDIUM | P1 | After TS-2 + registry metadata for prefill |
| TS-3 verdict states | HIGH | LOW | P1 | Judgment layer; land with first verdict-producing feature |
| TS-9 download w/ progress | HIGH (flow completion) | MEDIUM | P1 | Bridge over `download_model`; last in chain |
| TS-6 env panel | MEDIUM | LOW | P1 | Cheap; wizard's first screen |
| TS-8 Back/prefill nav | MEDIUM | LOW-MEDIUM | P1 (in-memory) / P2 (persisted) | TUI mechanics, parallel-safe |
| D-4 lab-models-first | MEDIUM (owner constraint) | LOW | P1 | Sort policy atop TS-4 |
| D-1 truncation ranking | HIGH | LOW | P2 | First post-v1 add |
| D-2 tokenizer dry-run | HIGH | MEDIUM-HIGH | P2 | Empirical trust layer |
| D-3 lineage routing | HIGH | MEDIUM | P2 | Registry's unique payoff |
| D-5 charset matrix | MEDIUM | LOW-MEDIUM | P2 | Honesty for cross-model users |
| D-6 MCP shells | MEDIUM | LOW | P2 | Seam-stability dependent |
| D-7 offline/degraded mode | MEDIUM | LOW | P2 (design rule in v1, full caching P2) | UNKNOWN-not-raise from day one |

---

## Competitor Feature Analysis

| Feature | docker init | nf-core / nf-schema | pandera | wandb/Ultralytics | dnallmmark (sibling) | DNALLM v1.4 approach |
|---------|-------------|---------------------|---------|-------------------|----------------------|----------------------|
| Detect-before-asking | Signature-file platform detection | — | Schema-first inference | — | Registry presence probes | Sniff data file format/columns → prefill wizard defaults |
| Collect-all errors | Overwrite warning | All failed params logged, then one throw | `lazy=True` → `SchemaErrors` w/ `failure_cases` | — | Wizard collect-all validation report | TS-2: structured issue list w/ severity + reason codes; same unwrapping serves config panel |
| Data-level validation stage | — | Explicit 2nd stage after params | Core competency | — | Gauntlet (subprocess) | TS-1/TS-2: read-only scan; params (config) and data validated as distinct stages |
| Size-before-download | — | — | — | — | Presence probe + queue state | TS-5: blob-size sum (disk) + params (VRAM), ModelScope-first, cached-local short-circuit |
| Suitability vs *user's data* | — | — | — | — | Committed compat verdicts (static) | TS-4 + D-1: length-distribution-aware ranking — beyond all five comparators |
| Wizard mechanics | Q&A with defaults | — | — | Entity/project picker | 4-step ModalScreen forms | TS-8: ModalScreen stack, typed dismiss, Back-with-prefill (terok pattern) |
| Long-task UX | — | — | — | — | Popen + progress parsing, 5-state queue | TS-9: `@work(thread=True)` + ProgressBar + screen-scoped cancel; persistent queue deferred (AF-8) |
| Edit-existing round-trip | Reuses existing compose filename | — | — | `wandb init` updates in-place | Session-undo | TS-7: load → edit → revalidate → gated save |
| Degraded/offline behavior | Manual platform fallback | `lenientMode` | — | offline mode flag | Tolerant absent probes | D-7: UNKNOWN verdict state, never raise, cache last-known |

---

## Sources

- DNALLM checkout (HIGH — read directly, 2026-10-20): `dnallm/datahandling/data.py` (loaders, `validate_sequences` mutation semantics + charset-hazard docstring, `head`/`show`/`get_split_lengths`), `dnallm/models/model_info.yaml` (schema: structured finetuned fields incl. `base_model`; tokenizer info only in `describe` text; no max_length/size fields), `dnallm/models/model.py` (`download_model` signature/retry/reason-classification; `_get_device`; tokenizer fallback chain), `dnallm/configuration/configs.py` (Pydantic section classes; `InferenceConfig.max_length=512`), `dnallm/utils/sequence.py` (`check_sequence`, `calc_gc_content`), `cli/model_config_generator.py` (numbered menus, author-new-only, zhangtaolab-priority sort)
- dnallmmark sibling checkout via TUI survey (HIGH — survey read-verified): screens/services inventory, merge-or-build verdicts (survey Part 5)
- [docker init reference](https://docs.docker.com/reference/cli/docker/init/) + [Docker Java containerize guide](https://docs.docker.com.tw/guides/java/containerize/) (MEDIUM) — wizard flow, defaults, overwrite warnings
- [nf-schema validation docs](https://nextflow-io.github.io/nf-schema/2.5.1/parameters/validation/) + [Nextflow input-validation training](https://training.nextflow.io/latest/nfcore_build/04_input_validation/) (MEDIUM) — collect-all-then-stop, two-stage order
- [pandera lazy validation](https://github.com/unionai-oss/pandera/blob/main/docs/source/lazy_validation.md) + [pandera error reports](https://pandera.readthedocs.io/en/stable/error_report.html) (MEDIUM) — collect-all issue-list shape
- [terok TUI wizard PR #797](https://github.com/terok-ai/terok/pull/797) + [Textual workers guide](https://python-cli-toolcraft.com/advanced-input-parsing-user-experience/building-terminal-uis-with-textual/running-background-work-in-textual-with-workers/) + [Textual modal-screen discussion](https://github.com/Textualize/textual/discussions/5388) (MEDIUM) — ModalScreen wizard, typed dismiss, workers/progress/cancel
- [huggingface_hub metadata patterns](https://github.com/huggingface/metadata-sniffer/blob/main/02_process_code.py) + [MTEB model_size util](https://huggingface.co/spaces/mteb/leaderboard/raw/c05fcecdd9c2540133679b32f66b14e3cef44a7/utils/model_size.py) + [dataset-size SO 77139182](https://stackoverflow.com/questions/77139182/how-can-i-see-the-size-of-a-huggingface-dataset-before-downloading-it) (MEDIUM) — params-vs-blob-size duality, quantization caveat
- ModelScope Hub API (MEDIUM-LOW — REST endpoints documented via [swama DeepWiki](https://deepwiki.com/Trans-N-ai/swama/6.2-model-repository-integration) and [omlx PR #1351](https://github.com/jundot/omlx/pull/1351); SDK per-file-size path unconfirmed — flagged as a phase spike)
- [Snakemake dry-run behavior](https://informatics.fas.harvard.edu/workshops/snakemake/run/) + [ARMOR wiki](https://github.com/csoneson/ARMOR/wiki/Running-the-analysis) (MEDIUM) — preflight honesty contract
- [wandb init CLI](https://docs.wandb.ai/ja/models/ref/cli/wandb-init) + [Ultralytics settings](https://docs.ultralytics.com/zh/usage/settings) (MEDIUM) — optional-wizard + snapshot-verb precedents

---
*Feature research for: DNALLM v1.4 Pre-flight UX & Registry-as-Truth-Source (Track A)*
*Researched: 2026-10-20*
