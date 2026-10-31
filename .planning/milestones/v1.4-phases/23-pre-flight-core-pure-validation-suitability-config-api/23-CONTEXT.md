# Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API - Context

**Gathered:** 2026-10-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver `dnallm/preflight/` — a pure, synchronous, offline-safe, pytest-covered API: `validate_data()` (read-only collect-all data report), `check_compatibility()` (OK/WARN/FAIL verdicts + intent-routed lab-first model listings), a ModelScope-first availability probe (cached / remote-confirmed / unknown, injectable, no weights), and `generate_config()` (validated emission + load-existing → edit → revalidate round-trip). Plus the one sanctioned refactor: a shared `validate_sections` path factored out of `load_config` (configs.py:698-747). Requirements PFC-01..PFC-07. No TUI (Phase 25), no `load_preset_model` changes (Phase 24), no MCP tools (Phase 26 / future D-6).

</domain>

<decisions>
## Implementation Decisions

### 数据输入面 (validate_data)
- **D-01:** Input = file path (`str | Path`, including pre-split dict-of-splits `{"train": ..., "test": ...}` aligned with `DNADataset.load_local_data`) **plus** plain `list[str]` sequences. Hub dataset names do NOT enter `validate_data` — the network boundary stays with the probe layer. — **Reversibility:** costly — the input contract is the surface every future shell (TUI, MCP tools, agent) calls; widening later is free but narrowing breaks shells.
- **D-02:** Sequence-column detection = deterministic heuristic auto-detection (per-column charset/length-distribution scoring) + explicit parameter override. The heuristic layer is the exact layer a future LLM-agent shell reuses — no LLM in this loop. — **Reversibility:** reversible.
- **D-03:** Underlying reader = a lightweight standalone reader inside preflight (pandas/json/pyarrow + hand-written fasta parse), importing NO `datasets`/`torch`/`transformers`. Format-parity with `DNADataset` is enforced by a structure test that double-reads the same file through both paths and pins row/column agreement. — **Reversibility:** reversible — reader is swappable behind the input contract.
- **D-04:** Format surface = csv/tsv/json/parquet/arrow/fasta/txt (full alignment with DNADataset minus pkl). `pkl`/`pickle` inputs return an explicit unsupported-format reason code — a validation API must not `pickle.load` untrusted user files. xlsx is deferred (needs openpyxl = new dep). — **Reversibility:** reversible.

### 判定校准 (verdict semantics)
- **D-05:** Master principle: **FAIL is reserved for hard incompatibility** — the model literally cannot work on this data (charset-rejected characters, task-type mismatch on a finetuned entry). Anything mitigable by config (truncation length) or data cleanup is at most WARN + counts; the user is never dead-ended. — **Reversibility:** costly — the verdict vocabulary is a cross-shell contract consumed by Phase 25 TUI panels, Phase 26 finding conventions, and future D-6 agent tools; re-grading later ripples through their tests.
- **D-06:** Length/truncation: length dimension has only OK/WARN — truncation NEVER fails. Report carries p50/p95/p99/max vs the model's effective max_length with a tokenizer-type-adjusted character budget (k-mer ÷ k, char 1:1, BPE conservative ratio), plus truncated-row count and fraction. Exact thresholds are planner discretion. — **Reversibility:** reversible.
- **D-07:** Charset: data chars ⊆ model charset → OK; N/ambiguity codes accepted by the model → WARN with per-character counts; charset-外 characters → that model FAILs (other models remain usable — per-model verdict, never a global error). The report counts; it never drops rows (contrast: `validate_sequences` whole-row destructive filter). — **Reversibility:** reversible.
- **D-08:** Task matching: finetuned entries hard-match (`task_type` ≠ user task → FAIL with reason code); base models are OK for any supervised task (heads bolt on via `DNALLMforSequenceClassification`); generation/MLM backbones for supervised tasks → WARN (head support uncertain). — **Reversibility:** reversible.
- **D-09:** Census conventions (stated as defaults, owner-confirmed): stats normalized via `upper()` (lowercase soft-mask counts as the same base; lowercase fraction reported separately as info); empty/too-short sequences (<20, configurable) get error-level reason codes; all-N sequences get warn. Report severity annotations only — never drop, never block. — **Reversibility:** reversible.

### 可用性探测
- **D-10:** Granularity = `probe_availability(model)` single-model primary API; `check_compatibility` listings are network-free by default with an OPTIONAL shortlist batch probe parameter (filtered top N≈20). Never probe all 236 entries. Injectable seams keep the unit suite network-free (PFC-03). — **Reversibility:** reversible.
- **D-11:** Hub fallback = probe the registry entry's own `source` first (lab models carry `source=modelscope`, so lab entries are ModelScope-first by construction); on miss/failure probe the other hub; when both hubs have it, report ModelScope as primary source. — **Reversibility:** reversible.
- **D-12:** `cached` detection = three-way full coverage: HF `scan_cache_dir()` + ModelScope dual-layout resolver (plan-time research spike already flagged) + local-path entries judged by direct existence. In-process memo cache (same model not re-probed within a process; registry data is immutable so no cross-process persistence); per-probe default timeout ≈5 s, configurable; rate-limit/timeout/any exception → UNKNOWN, never "unavailable". — **Reversibility:** reversible.

### generate_config
- **D-13:** Emits a complete runnable config — task + model + finetune + inference + lora sections with model-derived defaults: selected model → truncation/max_length from `effective_max_length`; finetuned model → `num_labels`/`label_names`/`threshold` prefilled from the registry entry; model names resolve through the registry, unknown model = field-level error (collect-all). — **Reversibility:** reversible.
- **D-14:** Round-trip input = YAML path (`str | Path`) OR already-parsed dict; edits apply as a patch dict then revalidate through the shared `validate_sections` path. — **Reversibility:** reversible.
- **D-15:** Output = return the full config object (with validation report); write to YAML only when an optional `output_path` is given (dump logic lives in the same module, same test suite). — **Reversibility:** reversible.

### Claude's Discretion
- Report/verdict object class names and field tables; reason-code enum naming (machine-readable, agent-friendly per the deferred agent-shell vision).
- Exact truncation WARN thresholds (quantile/fraction), the BPE conservative budget ratio, shortlist size N, timeout default, probe concurrency cap.
- Standalone reader internal structure; concrete lock points of the double-read parity structure test.
- Generation/regression/multilabel section differences and multilabel separator defaults (follow existing TaskConfig conventions).
- Whether preflight joins the `dnallm/__init__.py` root facade — leaning NO (import-light startup; import from `dnallm.preflight`), planner decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone scope & requirements
- `.planning/REQUIREMENTS.md` — PFC-01..PFC-07 define this phase's checkable requirements; Out of Scope pins what must NOT creep in
- `.planning/ROADMAP.md` §Phase 23 — goal, 5 success criteria, cross-cutting invariants (main frozen, fail_under=90, per-module ≥96%, no omit entries, `--research-phase` note with the two spikes)

### Research evidence base (2026-10-20, HIGH confidence)
- `.planning/research/SUMMARY.md` — §Phase 2 (C: Pre-flight Core) and Pitfall 5 are directly load-bearing; recommended stack and probe API verification
- `.planning/research/ARCHITECTURE.md` — component 2 design (`dnallm/preflight/`, report objects, injectable probe seam); `validate_sections` factor-out
- `.planning/research/PITFALLS.md` — Pitfall 5 (probe traps: rate limits, injectable seams, HF env vars read at import, two ModelScope cache layouts, `HF_HUB_OFFLINE`), Pitfall 1's sync-core boundary
- `.planning/research/STACK.md` — probe adapters (huggingface-hub 1.33.0 / modelscope 1.34.0), zero-new-deps conclusion for this phase

### Phase 22 contract (upstream dependency)
- `dnallm/models/registry.py` — the loader this phase consumes: `resolve`/`try_resolve`/`get_by_task_type`/`get_by_base_model`/`all_entries`/`is_lab_model`/`sort_key_lab_first`/`effective_max_length`/`effective_tokenizer_type`
- `.planning/phases/22-registry-foundation-runtime-loader-enrichment/22-CONTEXT.md` — loader validation strength, read-only-forever discipline, base↔finetuned inheritance at query time

### Refactor & anti-pattern anchors (in-repo)
- `dnallm/configuration/configs.py:698-747` — `load_config` body containing the validation logic to factor out as shared `validate_sections` (the one sanctioned refactor, no behavior change)
- `dnallm/datahandling/data.py:82` — `load_local_data` format/column contract (seq_col/label_col defaults, supported suffixes) the reader must stay parity-locked with
- `dnallm/datahandling/data.py:864` — `validate_sequences`: the destructive whole-row filter the report must NOT build on; its Warning note documents the 13 strict-charset families and cross-model comparability hazard
- `dnallm/utils/sequence.py:89` — `check_sequence` / `calc_gc_content` / `seq2kmer`: reusable helpers and the `set(seq.upper()) - set(valid_chars)` charset idiom
- `dnallm/models/model.py:317-377,494,1545` — download retry pattern (probe-seam precedent), the runtime `HF_ENDPOINT` toggle (spike target), the hardcoded ModelScope cache path (layout-resolver target)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm/models/registry.py` (Phase 22): everything the suitability check needs is already typed and tested — `effective_max_length`/`effective_tokenizer_type` resolve base↔finetuned inheritance; `sort_key_lab_first` implements PFC-07 ordering; `get_by_task_type`/`get_by_base_model` power intent routing (PFC-05)
- `utils/sequence.py` `check_sequence`/`calc_gc_content`/`seq2kmer` — census helpers and charset idiom
- `dnallm/tasks/metric_registry.py` (v1.2) — precedent for an import-light pure module proven by structure tests; same discipline for preflight

### Established Patterns
- Pure-function core + injectable seam (`download_model(..., downloader=...)` precedent) — probes take injected adapters, unit suite network-free
- Same-change pytest rule; per-module ≥96%, no omit entries; matchable `ValueError` conventions
- Pydantic `BaseModel` report objects (registry entries precedent: typed, frozen where sensible) — and reports must be agent-friendly: structured, self-describing, machine-readable reason codes

### Integration Points
- `dnallm/preflight/` (NEW) — consumed by Phase 25 TUI (panels render reports; no logic in screens), Phase 26 (finding-type conventions), future D-6 MCP tools / agent shells
- `validate_sections` (NEW, factored from configs.py:698-747) — shared by `load_config` and `generate_config`; zero behavior change on the load path
- `dnallm preflight` click subcommand and root-facade exports are Phase 25 / planner territory, not this phase's blockers

</code_context>

<specifics>
## Specific Ideas

- Owner's driving vision (2026-10-20): the ultimate user is a 麻瓜 — someone with zero bioinformatics/ML background. Phase 23's deterministic heuristics (column detection, charset census, task matching) are deliberately designed as the layer a future LLM agent will reuse to turn a muggle's raw spreadsheet into a DNALLM-recognized format and task.
- Owner's framing carried from project memory: ModelScope preferred source; `zhangtaolab/*` models prioritized in every listing (PFC-07).

</specifics>

<deferred>
## Deferred Ideas

- **Agent 壳·麻瓜数据引导** (future Track, NOT v1.4): a conversational agent (Pi / Claude / pydantic-ai; reference https://pi.dev/docs/latest — SDK/RPC/extensions entry points) consumes the pure-function core via D-6 (MCP tools over the pre-flight core), auto-recognizing sequence/label columns, inferring task type, and walking non-expert users from raw data to a DNALLM-recognized format and task. Verdict logic stays in the core forever — never in the LLM. Phase 23's answer: agent-friendly report objects + deterministic detection heuristics, so the future shell needs zero rework.
- **xlsx/Excel input support** for `validate_data` — needs openpyxl (new dependency, outside v1.4's two-dep budget); muggles save-as-csv meanwhile.

</deferred>

---

*Phase: 23-Pre-flight Core — Pure Validation / Suitability / Config API*
*Context gathered: 2026-10-20*
