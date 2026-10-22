# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra - Context

**Gathered:** 2026-10-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver `dnallm/tui/` + `dnallm preflight` — a Textual wizard (env → data → suitability → config → optional select-to-download) that is presentation-only over the Phase-23 core (`dnallm/preflight/`). The `tui` extra (`textual>=8.2,<9`, `platformdirs>=3.6,<5`) with `test = ["dnallm[tui]", ...]` riding every pytest CI leg, the structure tests (zero textual imports outside `dnallm/tui/`; zero torch/transformers direct imports inside it), the thread-worker seam, and the deprecated `base`→`full` extras-alias removal (PKG-01) land atomically BEFORE any panel. Includes the CR-01 root-facade lazy refactor this phase owns. Requirements TUI-01..TUI-08 + PKG-01. No training launch, no persistent multi-file download queue, no raw YAML editing, no snapshot testing (Out-of-Scope table in REQUIREMENTS.md governs).

</domain>

<decisions>
## Implementation Decisions

### 启动与环境面板 (Startup & env panel — CR-01 owned here)
- **D-01:** Root facade becomes a PEP 562 lazy facade (`__getattr__` in `dnallm/__init__.py`): `import dnallm.preflight` stops pulling torch/transformers at runtime; the wizard starts textual immediately — version rows render instantly via `importlib.metadata`, device/VRAM/disk rows show a "checking…" placeholder and fill from a background thread worker once the heavy imports complete. This CLOSES the Phase-23 CR-01 deferral (runtime `sys.modules` purity) rather than accepting it. Two research traps are explicitly in scope for `--research-phase`: (a) the load-bearing `dnallm.utils`-before-`dnallm.models` ordering (transformers ≥5.19 accelerator query on CUDA-less CI legs) must move into `dnallm/models/__init__.py` itself so the guarantee travels with the consumer — direct `import dnallm.models` today bypasses any root-init ordering; (b) `dnallm/utils/__init__.py` eagerly imports the compat shims — if they pull torch transitively, utils' init must lazy-load them too. — **Reversibility:** costly — undo re-couples every shell's import cost and must restore the positional ordering guarantee; the 400+-test suite is the regression net.
- **D-02:** Environment-info collector lives in a NEW module `dnallm/utils/env_info.py` (not in preflight, not in tui): core-owned, torch imported inside the function body only, invoked from a worker thread; screens render its returned report. The Phase-23 zero-torch AST invariant on `dnallm/preflight/` stays untouched; TUI-01's zero-torch rule for `dnallm/tui/` stays green.
- **D-03:** Env panel shows exactly the TUI-03 enumeration, nothing more: dnallm version / Python / torch (CUDA availability, device name, VRAM total & free) / transformers / numpy / datasets / free disk of model-cache roots. Every row phrased so a biologist understands it ("VRAM 24 GB / 21 GB free"). No driver/CPU/RAM 体检-style extras.

### 可用性探测 (Availability probing — WR-01 surfaced)
- **D-04:** On entering the suitability panel: local cached states are computed for ALL visible rows (pure path checks + cache-dir scans — offline and free), AND the top-of-list shortlist (N≈20, lab-first order — Phase-23 D-10) is remote-probed automatically with "probing…" placeholders; rows outside the shortlist stay UNKNOWN with a per-row on-demand probe key. Never probe all entries.
- **D-05:** WR-01 resolution — third-party family names whose hosting is unprobed (hardcoded `source="modelscope"`, Phase-24 review WR-01) are listed normally (lab-first ordering naturally puts them lower); the availability column honestly shows UNKNOWN / "hosting not verified"; on-demand probing reveals the truth. No hiding, no screen-layer filtering, no false source claims.
- **D-06:** Rows in UNKNOWN state remain downloadable — UNKNOWN ≠ unavailable (PFC-06); the download path has its own retry. The user is never dead-ended by a degraded probe.

### 流程形态 (Wizard flow shape)
- **D-07:** Dual entry: the wizard's first interaction offers "start from your data (new)" vs "edit an existing config file" (straight to the config panel), AND `dnallm preflight --config my.yaml` jumps directly to the config panel, skipping the first-screen choice.
- **D-08:** The data panel is OPTIONAL: the suitability panel is usable with no data loaded (task filter + registry facts: max_length/tokenizer/task matching), with data-driven dimensions showing a plain "no data provided yet" state; verdicts complete once data is loaded. Core support verified live: `check_compatibility(data=None, ...)` degrades length/charset to NO_DATA while the task dimension still computes.
- **D-09:** Select-to-download is the optional FINAL step: after the config panel emits the validated YAML, a closing screen offers downloading the selected model (visible progress + working cancel), then a "all done" summary. The user may quit at any point without downloading. Matches the roadmap flow order (env → data → suitability → config → select-to-download); sequential single-model download only.
- **D-10:** Wizard copy is English, plain language (owner directive 2026-10-21: every user-facing message understandable to biologists/bioinformaticians; technical identifiers verbatim).

### 配置面板 (Config panel — absorbs survey R4)
- **D-11:** Structured form fields for the five core sections only (task / model / finetune / inference / lora — the Phase-23 D-13 `generate_config` emission set). Sections present in a loaded YAML but outside that set (ia3/vep/sweep) are preserved verbatim and displayed read-only — never modified, never lost (23-04 zero-data-loss round-trip holds).
- **D-12:** Architectural facts are read-only, policy fields are editable: a finetuned entry's `num_labels`/`label_names`/`threshold` display read-only with a note ("from the model checkpoint; loading uses this value") — consistent with Phase-24 D-04 (registry always wins at load; an editable field would be silently overridden anyway). `max_length` is prefilled from `effective_max_length` but EDITABLE (truncation is a user policy choice — Phase-23 D-06: truncation never FAILs). Training hyperparameters (epochs/lr/batch_size/output_dir/…) are fully editable.
- **D-13:** Live validation as a resident split pane: form on the left, collect-all error list on the right — every field change revalidates through the shared `validate_sections` path, listing every problem as `section.field → plain-language message` until clean — plus a read-only YAML preview. The export/save action is DISABLED until validation is clean (SC3: the user leaves with a validated config; no invalid file is ever written).

### Claude's Discretion
- Shortlist size N (≈20 default), probe concurrency cap, placeholder/copy wording details.
- Screen-class structure (Textual screens vs panes), widgets, CSS/styling, navigation chrome; back-navigation mechanics (default expectation: back always allowed, accumulated state kept, changed upstream inputs recompute downstream views on re-entry).
- Worker-seam mechanism (message types vs `call_from_thread`) and download cancel semantics (cooperative vs abandon+resume) — both explicitly roadmap-flagged for `--research-phase`, not pre-decided here.
- `tui`-extra membership vs `full`-alias-removal install-line ordering within the atomic first act — roadmap research flag.
- Pilot test batch structure and wall-time budget (Pilot-in-this-gate documented but unproven in this repo).
- Env-row ordering/grouping; YAML-preview placement; default export filename; summary-screen contents.
- Lazy-facade mechanism details beyond D-01's contract (e.g. whether `dnallm/__init__.py` keeps eager `__version__`/`load_config`, how `from dnallm import *` parity is tested).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope & requirements
- `.planning/REQUIREMENTS.md` — TUI-01..TUI-08 + PKG-01 checkable requirements; Out-of-Scope table (training launch, persistent download queue, raw YAML editor, snapshot testing, auto-cleanup, Track B/C)
- `.planning/ROADMAP.md` §Phase 25 — goal, 5 success criteria, first-act atomicity, `--research-phase` note (worker-seam shape, cancel semantics, extra-vs-alias ordering, Pilot wall-time budget)

### Survey evidence base (2026-10-20)
- `.planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md` — Part 5: pre-flight cluster rationale (§5.2), one-pipeline-not-four-TUIs design (§5.3), merge-or-build verdicts (§5.4 — Track A is a FRESH build; dnallmmark has no user-data pre-flight), Track A/B/C split (§5.5); dnallmmark screen/service inventory (Part 1) as read-only pattern reference

### Upstream phase contracts
- `.planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-CONTEXT.md` — the core the TUI renders: D-05 FAIL-only-for-hard-incompatibility, D-06 truncation WARN, D-10 shortlist probe, D-13 config emission set, D-14/15 round-trip + output_path-only writes
- `.planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-CONTEXT.md` — D-04 registry-wins patch semantics (backs D-12's read-only facts); OQ-1 STRICT known limitation (6 third-party classification entries unloadable by name)
- `.planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-REVIEW-DISPOSITION.md` — WR-01 deferral record (unprobed third-party hosting; resolved by D-05 here)

### Core surface (what the panels render)
- `dnallm/preflight/__init__.py` — facade: `validate_data` / `check_compatibility` / `probe_availability` / `generate_config` / `edit_config` / `effective_endpoints`; docstring documents the runtime-purity deferral D-01 closes
- `dnallm/preflight/report.py` — typed report objects: `DataReport`/`SplitReport`/`LengthStats`/`CharsetCensusEntry`/`CompatibilityReport`/`ListingEntry`/`ModelVerdict`/`DimensionVerdict`/`AvailabilityReport`/`Finding`/`SectionError`/`GeneratedConfig` + severity/reason enums
- `dnallm/preflight/compatibility.py` — `check_compatibility(data=None, *, task_type, models, intent, prober, probe_shortlist)`; `data=None` no-data mode verified live 2026-10-21
- `dnallm/preflight/config_gen.py` — `generate_config` / `edit_config` round-trip + `_dump_yaml` write-only-with-output_path contract

### Refactor & wiring targets
- `dnallm/__init__.py` — the root facade D-01 makes lazy; in-source comment documents the utils-before-models ordering constraint (trap a)
- `dnallm/utils/__init__.py` — eager compat-shim imports (trap b); future home of `env_info.py` (D-02)
- `dnallm/cli/cli.py` — click group where the `preflight` subcommand lands; lazy-import precedent in command bodies
- `dnallm/models/model.py` — `download_model` (retry pattern; the function select-to-download wraps, called from a worker)
- `pyproject.toml` — `[project.optional-dependencies]` (tui extra lands here; deprecated `base` alias → removal per PKG-01; `test` extra gains `dnallm[tui]`); `[project.scripts]`
- `.github/workflows/ci.yml` — the `uv pip install -e ".[full]"` legs the install-line changes touch
- `tests/preflight/test_structure.py` — the AST import-discipline structure-test pattern to mirror for `dnallm/tui/`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm/preflight/` facade + typed reports — every panel's render target; no new domain logic needed anywhere in screens
- `check_compatibility(prober=..., probe_shortlist=...)` injectable seams — Pilot tests run network-free by injecting fake probers (same discipline as the Phase-23 unit suite)
- `importlib.metadata` version probing (dnallmmark `about` service pattern) — instant version rows without importing torch
- `download_model` retry/backoff — the download worker just wraps it; cancel is a UX-layer concern on top
- dnallmmark TUI (sibling repo, read-only reference): env_check 8-row panel, wizard form machinery, download queue progress cells — pattern references only; Track A is a fresh build (survey §5.5)

### Established Patterns
- AST structure tests (`tests/preflight/test_structure.py`) — the TUI-01 enforcement mechanism already exists as a proven pattern
- Same-change pytest rule; per-module ≥96%, no omit entries; `fail_under=90` global gate unaffected
- Pure core + thin shell: panel→core-function table test (TUI-08) mirrors the Phase-23 facade-integration discipline
- Function-local heavy imports (CLI precedent) — the TUI worker seam follows the same lazy-import idiom

### Integration Points
- `dnallm preflight` subcommand in `dnallm/cli/cli.py` (lazy import + friendly `pip install dnallm[tui]` message — TUI-02)
- `dnallm/__init__.py` lazy facade (D-01) — touches every import path; land with an import-parity structure test (every `__all__` name resolves identically) + a runtime-purity test (`import dnallm.preflight` → no torch in `sys.modules`)
- `pyproject.toml` extras + CI install lines + `base`-alias removal — one atomic first-act change
- Worker seam — the single bridge between the asyncio/textual loop and every blocking core call (env probe, validate_data, check_compatibility probes, download)

</code_context>

<specifics>
## Specific Ideas

- Measured 2026-10-21 (dev box, warm disk): `import dnallm.preflight` = 3.7 s, pulling torch + transformers via the package-init chain — the number that motivated D-01; cold-cache first runs on a user laptop are the real target (est. 10–30 s of dead terminal).
- Owner's driving vision (carried from Phase 23): the ultimate user is a 麻瓜 — zero bioinformatics/ML background; the wizard is once-per-project onboarding, so minute-zero UX outranks everything.
- ModelScope preferred source; `zhangtaolab/*` ranks first in every listing (PFC-07 carried into TUI-05); lab-first ordering is also what makes D-04's shortlist and D-05's honest-lower-placement work.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Pre-existing deferrals (Track B substrate merge, Track C post-flight surfaces, issues #62/#63/#64) remain governed by REQUIREMENTS.md and were not re-opened.

</deferred>

---

*Phase: 25-Terminal UI — Pre-flight Wizard & tui Extra*
*Context gathered: 2026-10-21*
