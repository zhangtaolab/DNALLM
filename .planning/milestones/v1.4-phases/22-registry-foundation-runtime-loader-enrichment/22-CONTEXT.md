# Phase 22: Registry Foundation — Runtime Loader & Enrichment - Context

**Gathered:** 2026-10-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Make `dnallm/models/model_info.yaml` (236 entries: 49 pretrained + 187 finetuned) runtime-consumable as the single source of truth: one cached, read-only, typed loader (`dnallm/models/registry.py`) as the sole parse owner, plus structured `tokenizer_type` / `max_length` enrichment landed via scripted text insertion, plus extended registry-integrity tests. Deliverables are REG-01 / REG-02 / REG-03. No TUI, no preflight functions, no `load_preset_model` changes (Phases 23–26).

</domain>

<decisions>
## Implementation Decisions

### Enrichment value source (owner, 2026-10-20)
- **D-01:** Probe each model's remote `config.json` (HF / ModelScope metadata APIs, no weight downloads) and COMMIT the measured values into the registry — one-time probe cost buys permanent offline availability (SSH/lab-environment priority). No runtime fallback probing in this phase. — **Reversibility:** reversible — committed values are data, re-probable at any time.

### max_length coverage (owner, 2026-10-20)
- **D-02:** `max_length` is populated for ALL 49 pretrained bases; finetuned entries inherit the base value via `base_model` (all 187 resolve, proven by 261020-kwt) unless an entry explicitly overrides it. Inheritance is resolved by the loader at query time, not duplicated into YAML. — **Reversibility:** reversible — inheritance rule lives in loader code.

### Third-party enrichment scope (owner, 2026-10-20)
- **D-03:** Enrichment order is lab-first: all 174 `zhangtaolab/*` entries are mandatory; the 62 third-party entries are best-effort — probe failures (rate limits, missing repos, API instability) are recorded as missing values and DO NOT block the phase. Third-party gaps surface as absent metadata, never as errors. — **Reversibility:** reversible.

### Loader validation strength (owner, 2026-10-20)
- **D-01b:** Registry entries are validated with Pydantic models at load time — a malformed entry fails loudly at first registry access (early exposure), consistent with "the registry is the truth source". Optional fields (`tokenizer_type`, `max_length`) are `X | None`; core fields (name/model/task) are required. — **Reversibility:** costly — schema becomes a published contract consumed by Phases 23/24/26; loosening later ripples through their tests.

### Claude's Discretion
- Exact Pydantic model shape / field granularity (planner decides; keep it close to the existing YAML schema).
- `reload()` semantics and cache-invalidation details (standard `lru_cache` clear + re-parse; no thread-safety contract needed for this phase).
- Whether `is_lab_model` / lab-first sort key is a module-level function or an entry property (either satisfies PFC-07's contract).
- Probe script location (one-off scripts confined to gitignored `.scratch/` per repo convention, or `scripts/` — planner's call; the COMMITTED OUTPUT is what matters).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone scope & requirements
- `.planning/REQUIREMENTS.md` — REG-01/02/03 define this phase's checkable requirements; Out of Scope pins what must NOT creep in
- `.planning/ROADMAP.md` §Phase 22 — goal, success criteria, cross-cutting invariants (main frozen, fail_under=90, per-module ≥96%, no omit entries)

### Research evidence base (2026-10-20, HIGH confidence)
- `.planning/research/SUMMARY.md` — synthesized findings; §Registry Runtime Loader phase and Pitfalls 3/5 are directly load-bearing here
- `.planning/research/ARCHITECTURE.md` — loader design (component 1), dead-path finding, probe API verification (HubApi.get_model_files / HfApi.model_info(files_metadata=True))
- `.planning/research/STACK.md` — CSafeLoader timing (5.8ms vs 69ms), probe adapters, zero-new-deps conclusion
- `.planning/research/PITFALLS.md` — Pitfall 3 (YAML provenance destruction: loader born read-only, no round-trip API) and Pitfall 5 (probe traps: rate limits, injectable seams, shortlist-only probing)

### Registry precedents & integrity suite (in-repo)
- `tests/models/test_model_info_registry.py` — the 261020-kwt integrity suite this phase EXTENDS (REG-03)
- `.planning/quick/261020-kwt-registry-data-layer-fix-add-15-missing-p/261020-kwt-SUMMARY.md` — provenance comment block location, scripted text-insertion discipline, base_model 187/187 derivation record
- `dnallm/models/model_info.yaml` — the artifact itself (read-only target; edits by scripted text insertion only)
- `dnallm/models/modeling_auto.py` — MODEL_INFO family layer (stays untouched this phase; loader coexists with it)

### TUI survey (upstream rationale)
- `.planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md` Part 5 — §5.3 API-first rule and §5.5 Track A scope that this phase grounds

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/models/test_model_info_registry.py` (261020-kwt): 12-test integrity suite — variants present, task-type correctness, no-"Casual" guard, base_model resolution. REG-03 extends this file's patterns (provenance-presence, no-write-API, enrichment invariants).
- `pyyaml` + `getattr(yaml, "CSafeLoader", yaml.SafeLoader)` fallback pattern (measured 12x faster on this exact file).
- `importlib.resources` package-data access — `model_info.yaml` is already declared in `[tool.setuptools.package-data]`.

### Established Patterns
- Registry edits: scripted text insertion keyed on unique `model:` lines (proven at scale in 261020-kwt: 164 insertions, provenance byte-preserved). Never `yaml.safe_load` → `dump`.
- Same-change pytest rule: every `dnallm/` modification ships with its tests in the same change (owner rule, all milestones).
- Matchable `ValueError` with descriptive messages for invalid registry lookups (repo-wide convention).

### Integration Points
- `dnallm/models/registry.py` (NEW) — consumed later by Phase 23 (`dnallm/preflight/`), Phase 24 (`load_preset_model` rewire at `dnallm/models/model.py:~1588`), Phase 26 (MCP consistency). This phase lands it with ZERO consumers rewired — the loader exists and is tested, nothing imports it in production paths yet (planner may add facade exports if cheap).
- `dnallm/models/model_info.yaml` — enrichment adds `tokenizer_type` / `max_length` fields (and optional explicit `max_length` overrides on finetuned entries that differ from their base).

</code_context>

<specifics>
## Specific Ideas

- Owner's framing for the enrichment order: 实验室模型优先 (lab 174 mandatory) — matches the ModelScope-first / lab-first directives recorded in project memory 2026-10-20.
- SSH/offline usability is the recurring rationale: probe-once-and-commit was chosen explicitly so the registry answers suitability questions with zero network at runtime.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 22-Registry Foundation — Runtime Loader & Enrichment*
*Context gathered: 2026-10-20*
