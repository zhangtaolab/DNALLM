# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra - Research

**Researched:** 2026-10-21
**Domain:** Textual TUI over a pure-Python pre-flight core; packaging/extras governance; import-purity refactor (CR-01)
**Confidence:** HIGH on codebase facts (every in-repo value below was read from source this session); HIGH on Textual thread-worker/Pilot mechanics (official docs); MEDIUM on cancel UX and Pilot wall-time (no in-repo precedent — both are discretion areas)

## Summary

Phase 25 is a presentation shell over a core that already exists. `dnallm.preflight` exposes six facade functions and a frozen Pydantic report inventory covering every fact the wizard must render — data census, compatibility verdicts, availability states, generated configs with collect-all errors. The TUI adds no domain logic; its only genuinely new infrastructure is (1) the packaging/`tui`-extra first act, (2) the structure tests that enforce the purity boundary, (3) one thread-worker seam that bridges Textual's asyncio loop to blocking core calls, and (4) the D-01 lazy-facade refactor that makes `import dnallm.preflight` stop pulling torch/transformers.

Two research findings change the shape of the D-01 refactor from what the phase description implies. First, making only the root facade lazy is not enough: `dnallm/models/__init__.py:1` imports `.model` (which imports torch at `dnallm/models/model.py:12`) on *any* `dnallm.models.*` import, and both preflight modules that touch the registry (`availability.py:53`, `compatibility.py:34`) import `..models.registry` — which executes that package init. Second, `dnallm/utils/__init__.py` has **three** torch paths, not one: the two compat shims (`:7-8`, and `transformers_compat.py:19` does `import torch`) *and* `support.py:2` (`from torch.cuda import get_device_capability`, eagerly re-exported at `utils/__init__.py:42`). The fix set is therefore: PEP 562 lazy facades in `dnallm/__init__.py`, `dnallm/models/__init__.py`, and `dnallm/utils/__init__.py` (lazy-loading the shims *and* `support`), with the load-bearing utils-before-models ordering guarantee moved into `dnallm/models/__init__.py`'s lazy resolution — exactly what CONTEXT D-01 traps (a) and (b) prescribe, now confirmed against the source.

On the async side, Textual's own documentation is explicit that thread workers cannot be interrupted — "you can't cancel threads in the same way as coroutines, but you *can* manually check if the worker was cancelled", and `Worker.is_cancelled` carries the note "cancelled work may still be running". Combined with `download_model`'s blanket `except Exception` retry loop (`dnallm/models/model.py:447-486`), this makes *cooperative* cancellation the only honest design, and requires a `BaseException`-derived cancellation sentinel to escape the retry loop at all. The UI must say "stopping after the current attempt", never pretend the download was interrupted.

**Primary recommendation:** Build `dnallm/tui/` as a presentation-only `App` — one screen class per wizard step, one `workers.py` seam whose `@work(thread=True)` functions wrap the six preflight facade calls and report back via typed, thread-safe `post_message` payloads; land the `tui` extra + `test`/`all` wiring + `base`-alias removal + structure tests + worker seam in one atomic commit before any panel; close CR-01 with lazy facades in root/models/utils and prove it with a subprocess-based `sys.modules` purity test plus an import-parity test.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**启动与环境面板 (Startup & env panel — CR-01 owned here)**

- **D-01:** Root facade becomes a PEP 562 lazy facade (`__getattr__` in `dnallm/__init__.py`): `import dnallm.preflight` stops pulling torch/transformers at runtime; the wizard starts textual immediately — version rows render instantly via `importlib.metadata`, device/VRAM/disk rows show a "checking…" placeholder and fill from a background thread worker once the heavy imports complete. This CLOSES the Phase-23 CR-01 deferral (runtime `sys.modules` purity) rather than accepting it. Two research traps are explicitly in scope for `--research-phase`: (a) the load-bearing `dnallm.utils`-before-`dnallm.models` ordering (transformers ≥5.19 accelerator query on CUDA-less CI legs) must move into `dnallm/models/__init__.py` itself so the guarantee travels with the consumer — direct `import dnallm.models` today bypasses any root-init ordering; (b) `dnallm/utils/__init__.py` eagerly imports the compat shims — if they pull torch transitively, utils' init must lazy-load them too. — **Reversibility:** costly — undo re-couples every shell's import cost and must restore the positional ordering guarantee; the 400+-test suite is the regression net.
- **D-02:** Environment-info collector lives in a NEW module `dnallm/utils/env_info.py` (not in preflight, not in tui): core-owned, torch imported inside the function body only, invoked from a worker thread; screens render its returned report. The Phase-23 zero-torch AST invariant on `dnallm/preflight/` stays untouched; TUI-01's zero-torch rule for `dnallm/tui/` stays green.
- **D-03:** Env panel shows exactly the TUI-03 enumeration, nothing more: dnallm version / Python / torch (CUDA availability, device name, VRAM total & free) / transformers / numpy / datasets / free disk of model-cache roots. Every row phrased so a biologist understands it ("VRAM 24 GB / 21 GB free"). No driver/CPU/RAM 体检-style extras.

**可用性探测 (Availability probing — WR-01 surfaced)**

- **D-04:** On entering the suitability panel: local cached states are computed for ALL visible rows (pure path checks + cache-dir scans — offline and free), AND the top-of-list shortlist (N≈20, lab-first order — Phase-23 D-10) is remote-probed automatically with "probing…" placeholders; rows outside the shortlist stay UNKNOWN with a per-row on-demand probe key. Never probe all entries.
- **D-05:** WR-01 resolution — third-party family names whose hosting is unprobed (hardcoded `source="modelscope"`, Phase-24 review WR-01) are listed normally (lab-first ordering naturally puts them lower); the availability column honestly shows UNKNOWN / "hosting not verified"; on-demand probing reveals the truth. No hiding, no screen-layer filtering, no false source claims.
- **D-06:** Rows in UNKNOWN state remain downloadable — UNKNOWN ≠ unavailable (PFC-06); the download path has its own retry. The user is never dead-ended by a degraded probe.

**流程形态 (Wizard flow shape)**

- **D-07:** Dual entry: the wizard's first interaction offers "start from your data (new)" vs "edit an existing config file" (straight to the config panel), AND `dnallm preflight --config my.yaml` jumps directly to the config panel, skipping the first-screen choice.
- **D-08:** The data panel is OPTIONAL: the suitability panel is usable with no data loaded (task filter + registry facts: max_length/tokenizer/task matching), with data-driven dimensions showing a plain "no data provided yet" state; verdicts complete once data is loaded. Core support verified live: `check_compatibility(data=None, ...)` degrades length/charset to NO_DATA while the task dimension still computes.
- **D-09:** Select-to-download is the optional FINAL step: after the config panel emits the validated YAML, a closing screen offers downloading the selected model (visible progress + working cancel), then a "all done" summary. The user may quit at any point without downloading. Matches the roadmap flow order (env → data → suitability → config → select-to-download); sequential single-model download only.
- **D-10:** Wizard copy is English, plain language (owner directive 2026-10-21: every user-facing message understandable to biologists/bioinformaticians; technical identifiers verbatim).

**配置面板 (Config panel — absorbs survey R4)**

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

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. Pre-existing deferrals (Track B substrate merge, Track C post-flight surfaces, issues #62/#63/#64) remain governed by REQUIREMENTS.md and were not re-opened.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TUI-01 | `tui` extra (`textual>=8.2,<9`, `platformdirs>=3.6,<5`); `test` extra gains `dnallm[tui]`; structure tests enforce zero textual outside `dnallm/tui/` and zero torch/transformers inside it — atomic | Verified extras layout + CI install lines (all legs use `.[full]` → `test` → `dnallm[tui]`); AST structure-test pattern proven in `tests/preflight/test_structure.py`; textual 8.2.8 exists and fits the range |
| TUI-02 | `dnallm preflight` click subcommand via lazy import with friendly `pip install dnallm[tui]` message | CLI lazy-import precedent read in `dnallm/cli/cli.py` (command bodies import heavy modules locally); `ImportError` → `click.echo` + `sys.exit(1)` matches the MCP command's handling |
| TUI-03 | Env panel renders device/VRAM/library versions/disk from the core; no logic in screens | D-02's `dnallm/utils/env_info.py`; `importlib.metadata` for instant version rows; torch calls confined to function body |
| TUI-04 | Data panel renders `validate_data()` report (stats, per-rule failures, counts, reason codes) | `DataReport`/`SplitReport`/`Finding` inventory read in `dnallm/preflight/report.py`; reason codes are `StrEnum` values — shell routes on codes, never re-derives verdicts |
| TUI-05 | Suitability panel renders `check_compatibility()`/availability verdicts, lineage routing, lab-first ordering; selection feeds next step | `check_compatibility(data=None, *, task_type, models, intent, prober, probe_shortlist)` read at `compatibility.py:132`; `ListingEntry.is_lab` + lab-first ordering come from the core; injectable `prober` makes Pilot tests network-free |
| TUI-06 | Config panel generates/edits YAML with live collect-all validation; emits validated config | `generate_config(model, *, task_type, output_path)` and `edit_config(existing, patch, *, output_path)` read in `config_gen.py:206,382`; write-only-with-`output_path` contract preserved |
| TUI-07 | Select-to-download runs `download_model` sequentially with progress and cancel; all blocking work in thread workers | `download_model(model_name, downloader, revision=None, max_try=10, allow_patterns=None)` read at `model.py:411`; Textual thread-worker + cancel semantics cited below; cooperative-cancel design specified |
| TUI-08 | Pilot-functional tests only; panel→core-function table test; per-module ≥96%, no omit entries | `App.run_test()`/`Pilot` documented patterns; repo pytest already `--asyncio-mode=auto`; coverage omit list verified to carry no tui entry |
| PKG-01 | One-cycle `full` extras alias removed, sequenced atomically with TUI-01's install lines | `base = ["dnallm[full]"]` alias + deprecation comment read in `pyproject.toml`; docs deprecation notes at `README.md:178` and `docs/getting_started/installation.md:257` |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

Actionable directives the planner must honor:

- **Same-change pytest rule:** any `dnallm/` code change ships with pytest coverage in the same change.
- **Coverage:** `fail_under=90` global; per-module ≥96% for new code; **no omit entries** — TUI modules stay in the denominator (mirrors the preflight structure test that forbids omit entries).
- **Style:** ruff format/check, line-length **100**, Google-style docstrings, `str | None` / `list[int]` PEP 604 syntax, relative imports inside `dnallm/`, absolute imports in tests.
- **Lazy imports:** heavy dependencies import inside function/command bodies (CLI precedent) — the TUI's textual import sits inside the `preflight` command body, and torch imports sit inside `env_info.py` function bodies.
- **No `print()` in library code** (ruff `T20`); use the project logger (`from ..utils import get_logger`).
- **Registry sole-mentioner guard:** no `dnallm/` module may name the registry YAML file by filename outside `dnallm/models/registry.py` — `tests/models/test_registry.py`'s grep guard fails CI on documentation mentions too. TUI docstrings must not name it.
- **Structure-test discipline:** every new structure guard is probe-verified red under a deliberate local violation before being committed green (Phase 22/23 precedent).
- **Commits:** no Co-Authored-By trailers. Branch `v1.4` only — `main` is frozen.
- **User-facing copy:** plain language understandable to biologists/bioinformaticians (D-10); technical identifiers verbatim.
- **Model listing:** ModelScope-first, `zhangtaolab/*` first — both already implemented in the core; the TUI must not re-sort.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Environment fact collection | Core (`dnallm/utils/env_info.py`, D-02) | — | Torch-dependent facts belong to a core-owned collector; the TUI only renders its returned report |
| Data validation / census | Core (`dnallm/preflight/validation.py`) | — | Pure, read-only, AST-guarded against heavy imports |
| Compatibility & availability verdicts | Core (`dnallm/preflight/compatibility.py`, `availability.py`) | — | Verdict logic, reason codes, lab-first ordering all live here; shells route on codes |
| Config generation & editing | Core (`dnallm/preflight/config_gen.py`) | — | `validate_sections` is the single validation path; the TUI must never re-implement it |
| Blocking-call bridging | TUI infrastructure (`dnallm/tui/workers.py`) | — | The single bridge between the asyncio loop and every blocking core call |
| Wizard presentation / navigation | TUI (`dnallm/tui/app.py`, `screens/`) | — | Screens, widgets, CSS, flow state — no domain logic |
| Model download execution | Core (`dnallm/models/model.py:download_model`) | TUI worker (wraps, reports progress) | Retry/backoff already implemented; the TUI adds a cancellable wrapper + progress reporting |
| Package surface / extras | Packaging (`pyproject.toml`, CI install lines) | — | Atomic first act, sequenced with PKG-01 |
| Import purity of the core | Package facades (`dnallm/__init__.py`, `models/__init__.py`, `utils/__init__.py`) | — | D-01: three lazy facades; the ordering guarantee moves to its consumer |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `textual` | `>=8.2,<9` (latest 8.2.8) | TUI application framework: App/Screen stack, widgets, CSS, workers | Locked by CONTEXT/REQUIREMENTS; official repo `github.com/Textualize/textual`; requires-python `>=3.9,<4.0`, classifiers through 3.14 → covers the 3.11/3.12/3.13 CI matrix [VERIFIED: pypi.org/pypi/textual/json] |
| `platformdirs` | `>=3.6,<5` (latest 4.13.5; installed 4.12.1) | Cross-platform config/cache directory resolution for the wizard's defaults | Locked by CONTEXT; textual itself declares `platformdirs<5,>=3.6.0` — the extra's range matches its own transitive constraint exactly [VERIFIED: pypi.org/pypi/textual/json requires_dist] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest-asyncio` | (already in `test` extra) | Async Pilot tests | Repo already runs `--asyncio-mode=auto` [VERIFIED: pyproject.toml `[tool.pytest.ini_options]` addopts] |
| `rich` | `>=14.2.0` via textual (repo declares `rich>=13.7.0`; dev venv has 15.0.0) | Textual's rendering dependency | No action needed in the dev venv; fresh CI resolves ≥14.2 through textual |
| `importlib.metadata` | stdlib | Instant version rows without importing torch | Env panel version rows (D-01) |
| `pydantic` | (already) | Report/config types consumed by the TUI | Never construct reports in the TUI — receive them |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Textual | `prompt_toolkit` / `questionary` / `rich`-only | No screen-stack app model or worker lifecycle; CONTEXT locks Textual — do not re-open |
| Textual workers | Hand-rolled `threading.Thread` + `asyncio.Queue` | Reimplements worker state, cancel plumbing, error routing, shutdown; Textual's Worker API is the blessed path |
| `pytest-textual-snapshot` | Pilot-functional assertions | Explicitly out of scope (cross-OS glyph diffs on the Windows leg); snapshots forbidden in v1.4 |

**Installation:**
```bash
uv pip install -e ".[tui]"     # textual + platformdirs
uv pip install -e ".[full]"    # CI legs: full → dev,test,notebook,mcp → test → dnallm[tui]
```

**Version verification:** `pip index versions textual` → latest 8.2.8 (8.2.0–8.2.8 available; range `<9` fits). `pip index versions platformdirs` → latest 4.13.5 (3.6.0–4.13.5 available; range `<5` fits). `uv pip install --dry-run "textual>=8.2,<9"` in the dev venv resolves to exactly `textual==8.2.8` + `linkify-it-py==2.2.0` — no rich downgrade/upgrade in this venv (rich 15.0.0 already satisfies textual's `>=14.2.0`).

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `textual` | PyPI | First release 2021 (0.1.0 in the version list); 8.2.8 current | unknown (seam cannot read PyPI download stats) | `github.com/Textualize/textual` | [SUS] — reason: `unknown-downloads` | Retained — locked by CONTEXT; official upstream repo confirmed; the SUS reason is a seam artifact |
| `platformdirs` | PyPI | Project predates 2.0 (2010s-era, tox-dev); 4.13.5 newest release published 2026-10-19 | unknown | `github.com/tox-dev/platformdirs` | [SUS] — reasons: `too-new`, `unknown-downloads` | Retained — `too-new` reflects the freshly published 4.13.5 release, not the project; also a textual direct dependency |

**Packages removed due to [SLOP] verdict:** none

**Packages flagged as suspicious [SUS]:** `textual`, `platformdirs` — both verdicts trace to missing download-stat signals in the legitimacy seam (`weeklyDownloads: null`), plus `platformdirs`'s latest-release age. Both are already locked by CONTEXT.md, both resolve to canonical upstream repositories, and neither carries a `postinstall` script. Planner should record this audit in the first-act plan; a `checkpoint:human-verify` before install is optional given the locked decision and the evidence above.

## Architecture Patterns

### System Architecture Diagram

```
┌──────────────┐   dnallm preflight [--config PATH]
│  CLI (click) │──────────────┐  lazy import inside the command body
└──────────────┘              ▼
                     ┌──────────────────┐        ImportError → "pip install dnallm[tui]"
                     │  dnallm/tui/app  │◄────────(friendly message, exit 1)
                     │  DNALLM Wizard   │
                     └────────┬─────────┘
        ┌──────────┬──────────┼──────────┬───────────┐
        ▼          ▼          ▼          ▼           ▼
   EnvScreen  DataScreen  SuitScreen  ConfigScreen  DownloadScreen
   (renders)  (renders)   (renders)   (form+errors)  (progress+cancel)
        │          │          │          │           │
        └──────────┴──────────┴──────────┴───────────┘
                      │  post typed Message (thread-safe)
                      ▼
              ┌───────────────────┐
              │  workers.py seam  │  @work(thread=True), exit_on_error=False
              │  (only bridge)    │  get_current_worker().is_cancelled
              └────────┬──────────┘
        ┌──────────────┼───────────────┬────────────────┐
        ▼              ▼               ▼                ▼
 env_info.collect  validate_data  check_compatibility  generate_config/edit_config
 (torch inside     (preflight)    (preflight; prober    (preflight; output_path
  the function)                    injectable)           write-only contract)
                                                              │
                                                              ▼
                                                       download_model(…, downloader)
                                                       (retry loop; cancellable wrapper)
```

Data flow: user input → screen → seam call → core function (in a thread) → frozen Pydantic report/mapping → typed message → screen render. No arrow ever goes screen → verdict logic.

### Recommended Project Structure

```
dnallm/
├── tui/
│   ├── __init__.py        # public entry: run_preflight(config_path=None) — no logic
│   ├── app.py             # DNALLMPreflightApp(App[None]) + screen routing
│   ├── messages.py        # typed Message payloads (worker → UI)
│   ├── workers.py         # the single thread-worker seam
│   ├── screens/           # env.py, data.py, suitability.py, config.py, download.py, summary.py
│   ├── widgets/           # reusable tables/forms (optional; only if genuinely reused)
│   └── styles.tcss        # Textual CSS
├── utils/
│   └── env_info.py        # D-02: core-owned env collector; torch imported inside the function
tests/
└── tui/
    ├── test_app.py        # Pilot-functional flow tests
    ├── test_workers.py    # seam unit tests with fake probers/downloaders
    ├── test_screens.py    # panel→core-function table test (no domain logic)
    ├── test_env_info.py   # collector unit tests (torch mocked)
    └── test_structure.py  # TUI-01 AST guards (mirror tests/preflight/test_structure.py)
```

### Pattern 1: Thread worker + typed message (the single seam)

**What:** Every blocking core call runs in a function decorated `@work(thread=True, exit_on_error=False)`; the function never touches widgets. It calls `post_message(...)` (thread-safe per docs) with a typed payload; the screen handles the message on the UI thread. Use `call_from_thread` only for a single direct setter-style update, and prefer messages when more than one update is needed.

**When to use:** env collection, `validate_data`, `check_compatibility` (per batch / shortlist probe), config validation, download progress.

**Example:**
```python
# Source: https://textual.textualize.io/guide/workers/
from textual import work
from textual.worker import get_current_worker

@work(thread=True, exit_on_error=False)
def collect_env(self) -> None:
    worker = get_current_worker()
    report = env_info.collect()          # torch imports inside the function
    if not worker.is_cancelled:          # cooperative check before UI effect
        self.app.post_message(EnvReportReady(report))
```

**Docs contract:** "you should **avoid calling methods on your UI directly** from a threaded worker, or setting reactive variables"; "Most Textual functions are not thread-safe" but `post_message` "is thread-safe" [CITED: textual.textualize.io/guide/workers/]. `exit_on_error=False` changes the default "exit the app and display the traceback" behavior into a worker `ERROR` state you can render [CITED: textual.textualize.io/guide/workers/].

### Pattern 2: Worker state handling

**What:** Results also arrive as `Worker.StateChanged` events (`on_worker_state_changed`), which Textual posts to the widget where the worker was created; the result lives on `worker.result`, the exception on `worker.error`. Prefer the state-changed handler over `worker.wait()` inside a message handler — the docs warn `wait()` "will also prevent the widget from updating until the worker returns" [CITED: textual.textualize.io/guide/workers/].

**When to use:** error rows (surface `worker.error` as a plain-language message), completion side-effects, progress display via `Worker.update(completed_steps, total_steps)`.

### Pattern 3: Panel→core-function table test (TUI-08)

**What:** A parameterized test that asserts, for each screen module, the exact set of core functions it is allowed to import/call — e.g. `DataScreen` → `{validate_data}`, `SuitabilityScreen` → `{check_compatibility, probe_availability}`, `ConfigScreen` → `{generate_config, edit_config}`, `DownloadScreen` → `{download_model}`. Anything else (verdict arithmetic, sorting, path scanning) fails the test. This operationalizes "presentation-only" as a checkable table, mirroring the Phase-23 facade-integration discipline.

### Pattern 4: PEP 562 lazy facades with the ordering guarantee at the consumer

**What:** Replace eager module-level imports with module `__getattr__`/`__dir__` that import on first attribute access. The root facade keeps `__all__` identical. The **models** facade owns the ordering guarantee: when resolving a heavy name it first imports `dnallm.utils._transformers_compat` (installing the shims) *before* importing `.model`, so the transformers ≥5.19 accelerator-query patch is always in place — "the guarantee travels with the consumer" (D-01 trap a).

```python
# dnallm/__init__.py (shape; CONTEXT leaves exact membership to discretion)
from importlib import import_module

__all__ = [...]  # unchanged

_LAZY = {
    "Benchmark": "dnallm.inference",
    "DNADataset": "dnallm.datahandling",
    ...
}

def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value          # cache for subsequent access
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
```

**Verified constraints this pattern must satisfy:**
- `tests/preflight/test_structure.py:102-109` asserts `"preflight" not in dnallm/__init__.py` — the lazy facade must never mention preflight [VERIFIED: tests/preflight/test_structure.py:102-109].
- Import parity: every `__all__` name resolves identically, `from dnallm import *` works, `dir(dnallm)` includes all members.
- Runtime purity must be tested in a **subprocess** (`python -I -c "import dnallm.preflight; ..."`), because the test process has already imported `dnallm` — an in-process `sys.modules` assertion is vacuous.

### Anti-Patterns to Avoid

- **UI mutation from a worker thread** — Textual functions are not thread-safe; only `post_message` (and `call_from_thread` for single updates) crosses the boundary [CITED: textual.textualize.io/guide/workers/].
- **Blocking the event loop** — any direct `check_compatibility`/`probe_availability`/`download_model` call from a message handler freezes the app (SC4).
- **Verdict logic in screens** — re-deriving OK/WARN/FAIL from report facts breaks the "reason codes are machine-readable by design" contract (`report.py:11-14`) and the panel→core table test.
- **Eager textual/torch imports in the TUI package** — breaks TUI-01's structure guards; textual imports only inside `dnallm/tui/`, torch never.
- **Writing YAML outside the preflight contract** — only `generate_config`/`edit_config` with an explicit `output_path` write files; the TUI must not `yaml.dump` itself.
- **Snapshot/golden tests** — out of scope (Windows glyph diffs).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|------------|-------------|-----|
| Async app loop, screen stack, focus, key bindings | Custom asyncio UI loop | `textual.App` / `Screen` | Focus/paint/event plumbing is the product itself |
| Worker lifecycle, cancel flags, shutdown | `threading.Thread` + queues | Textual `Worker`/`WorkerManager` (incl. `cancel_all()`, `wait_for_complete()`, `cancelled_event`) | Worker states, error routing, and exit semantics are already specified [CITED: textual.textualize.io/api/worker_manager/] |
| Headless UI testing | Custom terminal emulation | `App.run_test()` + `Pilot` (`press`, `click`, `pause`) | Headless by design; `pause()` settles pending messages [CITED: textual.textualize.io/guide/testing/] |
| Version strings | Manual `__version__` chains | `importlib.metadata.version("dnallm")` etc. | D-01: instant rows without importing torch |
| Cross-platform config/cache dirs | `os.path.expanduser` heuristics | `platformdirs.user_config_dir/user_cache_dir` | Locked dependency; handles Windows/macOS/Linux |
| Download retry/backoff | New retry logic in the TUI | `download_model` (`max_try`, reason classification) | Already implemented and tested (`model.py:411`) |
| YAML validation | Screen-side field checks | `validate_sections` via `generate_config`/`edit_config` | Single validation path; collect-all errors with `section.field` paths |
| Progress UI | Hand-rolled bars | `ProgressBar` + `Worker.update(completed_steps, total_steps)` | Standard widget + worker-introspection progress |

**Key insight:** every "don't hand-roll" item here is already a solved problem in either Textual or `dnallm.preflight`/`dnallm.models`; the TUI's only original code is wiring, rendering, and the cancellation wrapper.

## Roadmap Research Flags — Resolved

### Flag 1 — Worker-seam shape: message types vs `call_from_thread`

**Recommendation: typed messages as the primary (ideally only) seam.**

- `post_message` is documented as thread-safe; `call_from_thread` exists for direct UI calls from a thread [CITED: textual.textualize.io/guide/workers/]. The docs' own guidance is to send custom messages "for multiple UI updates".
- Messages give a typed payload contract (`EnvReportReady(DataReport)`, `CompatReady(CompatibilityReport)`, `DownloadProgress(done, total, label)`, `DownloadFinished(path | None)`, `WorkerFailed(plain_message)`), which is what the panel→core table test and the presentation-only rule want: screens handle *reports*, never re-derive them.
- One idiom is reviewable and testable; two idioms (messages + `call_from_thread`) invite inconsistent threading discipline.
- Recommendation: `call_from_thread` only where a single setter-style update exists and no payload type is warranted — none is currently required by the flow.

### Flag 2 — Download cancel semantics: cooperative vs abandon+resume

**Recommendation: cooperative cancellation with an honest UI state; "abandon" is the fallback behavior the process already has.**

Evidence:

1. Textual: "you can't cancel threads in the same way as coroutines, but you *can* manually check if the worker was cancelled"; `Worker.is_cancelled` — "Note that cancelled work may still be running"; `Worker.cancelled_event` — "A threading event set when the worker is cancelled." [CITED: textual.textualize.io/guide/workers/, textual.textualize.io/api/worker/]. **Threads cannot be interrupted.**
2. `download_model` retries on `except Exception` (break only on "not found" / 404) and raises `ValueError` only after `max_try` [VERIFIED: dnallm/models/model.py:447-486]. A plain `Exception` (or even `ValueError`) raised by a cancellable downloader wrapper would be swallowed and retried up to 10 times, sleeping 1 s per retry.
3. The `downloader` argument is injectable, so the TUI can wrap it: check `get_current_worker().cancelled_event.is_set()` before each attempt, and on cancel raise a **`BaseException`-derived sentinel** (`class DownloadCancelled(BaseException)`) that `except Exception` cannot catch. The worker catches it explicitly and posts `DownloadFinished(None)` / renders "cancelled".

Design contract to plan against:
- Cancel is **cooperative**: it takes effect before the next attempt. A download already inside `snapshot_download` runs to completion of that attempt; the UI must say so in plain language ("Stopping — this download attempt will finish first").
- Do **not** claim immediate interruption; do not kill threads.
- On app quit with a download in flight, `workers.cancel_all()` + explicit copy explaining the background attempt may still finish before the process exits.
- If the owner ever wants a hard cancel, the only honest mechanism is a subprocess download (killable) — record as a future option, not in this phase (no persistent queue; sequential single model).

### Flag 3 — `tui` extra membership vs `full`-alias removal ordering

**Recommendation: one atomic commit, in this order — and `all` gains `tui` too.**

1. `[project.optional-dependencies]`: add `tui = ["textual>=8.2,<9", "platformdirs>=3.6,<5"]`; add `"dnallm[tui]"` to `test`; add `tui` to `all` (`dnallm[full,dev,test,notebook,docs,ui,mcp,fla,tui]`) — TUI-01 requires the `test` wiring; `all` is the "everything" install and silently missing the new extra would be the same class of under-install hazard the `base` alias exists to prevent.
2. Delete `base = ["dnallm[full]"]` and its deprecation comment (PKG-01) **in the same commit** — the removal is one-cycle pre-committed, and the install lines are the shared surface.
3. Update the two deprecation notes that describe the alias (`README.md:178`, `docs/getting_started/installation.md:257`) so the docs no longer promise a removed extra.
4. Structure tests + worker seam land in the same commit (worker seam imports textual; structure tests need the package present). No panel exists yet.
5. CI needs **no install-line edits**: every pytest leg installs `-e ".[full]"` (`ci.yml:92,172,261,341,408,543,758` — some with extra flags) and `full` composes `dev,test,notebook,mcp`; once `test` carries `dnallm[tui]`, textual rides every leg including Windows [VERIFIED: .github/workflows/ci.yml install lines].

Ordering rationale: the extra must exist before `test` can reference it in the same file (trivial, but reviewers read the diff top-down); alias removal rides the same install-line diff so there is exactly one release-note moment. The `dev` extra composes `dnallm[test,notebook]`, so dev installs get textual transitively — no separate edit.

### Flag 4 — Pilot test-batch wall-time budget

**Recommendation: keep the TUI Pilot suite ≤ ~30 s locally, ≤ 60 s per test (hard ceiling 300 s from `--timeout=300`), and never touch the network or torch in Pilot tests.**

Evidence and tactics:
- Official docs give no wall-time guidance; they recommend `pilot.pause()` to settle pending messages and warn that asserting immediately after a message can race [CITED: textual.textualize.io/guide/testing/]. `app.workers.wait_for_complete()` exists on `WorkerManager` ("Wait for workers to complete.") and is the deterministic join for worker-completing tests [CITED: textual.textualize.io/api/worker_manager/].
- The repo runs `--asyncio-mode=auto` and `--timeout=300` [VERIFIED: pyproject.toml `[tool.pytest.ini_options]` addopts] — async Pilot tests need no decorator; a hung test dies at 300 s but the suite should never get close.
- Network-free by construction: inject fake `prober` callables into `check_compatibility` (the Phase-23 unit-suite discipline). Downloads in tests use a fake `downloader` function (the injectable seam of `download_model`) so progress/cancel are exercised instantly.
- `run_test()` defaults to size (80, 24); set an explicit size per test (e.g. `(100, 40)`) so layout assertions are stable and match the smallest real terminal users have.
- Budget shape: one smoke "flow completes" Pilot test; one per screen ("renders report facts / emits validated config / cancel path"); the panel→core table test as plain unit tests. If the suite creeps past ~30 s, split by screen and mark nothing slow — slow-marker is for model downloads, not UI.

## Runtime State Inventory

This phase is a code-only refactor (lazy facades) + new package. No renamed strings, no stored state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified: no database/datastore in `dnallm/` holds import-path or facade names | None |
| Live service config | None — verified: no external service config names dnallm import symbols | None |
| OS-registered state | None — verified: no launchd/systemd/Task Scheduler entries reference `dnallm.__init__` | None |
| Secrets/env vars | None — verified: `grep` for env-var reads in `dnallm/__init__.py` / `models/__init__.py` / `utils/__init__.py` is empty; env vars set programmatically are unrelated to facades | None |
| Build artifacts | Editable install (`-e .`) + `__pycache__` — pure-Python import changes; no compiled artifacts reference the facade | None (fresh import regenerates) |

## Common Pitfalls

### Pitfall 1: Treating thread-worker cancel as an interrupt
**What goes wrong:** UI shows "cancelled", but the download keeps running and later writes/returns; or the app hangs waiting for a thread that never stops.
**Why it happens:** Textual thread workers cannot be interrupted — `cancel()` cancels the *worker*, not the OS thread; cancelled work "may still be running" [CITED: textual.textualize.io/api/worker/].
**How to avoid:** cooperative checks before each attempt; honest copy; `cancel_all()` on quit; never block the UI thread on a thread's completion.
**Warning signs:** a test that asserts "cancel → no further downloader calls" without a fake downloader that counts calls; a "cancel" button that awaits anything.

### Pitfall 2: `download_model`'s `except Exception` swallows cancellation
**What goes wrong:** the wrapper raises `RuntimeError("cancelled")`; `download_model` treats it as a network issue, retries 10×, sleeps 1 s each, then raises `ValueError("Model … download failed.")` after ~10 s.
**Why it happens:** blanket `except Exception` at `model.py:468` [VERIFIED: dnallm/models/model.py:447-486].
**How to avoid:** a `BaseException`-derived sentinel; catch it inside the worker and post a cancelled message. Do not modify `download_model` (out of scope).
**Warning signs:** cancel tests taking ~10 s.

### Pitfall 3: Root-only lazy facade leaves torch in `sys.modules`
**What goes wrong:** purity test fails — or worse, passes in-process because the test module already imported dnallm.
**Why it happens:** `dnallm/models/__init__.py:1` imports `.model` (torch at `model.py:12`); `dnallm/utils/__init__.py:7-8` imports the shims, `transformers_compat.py:19` imports torch, and `:42` re-exports `support` whose `support.py:2` is `from torch.cuda import get_device_capability`; preflight reaches the registry via `from ..models.registry import …` (`availability.py:53`, `compatibility.py:34`) which executes the models package init [VERIFIED: all four files read this session].
**How to avoid:** all three lazy facades; enumerate the utils torch surface as **shims + support**; subprocess purity test.
**Warning signs:** `import dnallm.preflight` under 1 s but `torch` still in `sys.modules`.

### Pitfall 4: Ordering guarantee lost in the lazy models facade
**What goes wrong:** on the CUDA-less CI legs, the first `from transformers import …` from `.model` raises during the accelerator query because the transformers shim was never imported first.
**Why it happens:** the guarantee currently lives only as a comment + positional import order in `dnallm/__init__.py:24-30`; a lazy facade runs it on attribute access, and a direct `import dnallm.models` never ran root init at all [VERIFIED: dnallm/__init__.py:24-30].
**How to avoid:** `dnallm/models/__init__.py` `__getattr__` imports `dnallm.utils._transformers_compat` (and `_cuda_compat`) before resolving any `.model`-backed name; a test that directly imports `dnallm.models` and resolves `load_model_and_tokenizer` with torch present.
**Warning signs:** test-cuda leg failing only on model-symbol resolution.

### Pitfall 5: Markup/escape injection from data-derived strings
**What goes wrong:** a model name, label name, or data finding containing `[` renders as Textual markup and breaks the layout — or an OSC sequence reaches the terminal.
**Why it happens:** Textual renders Rich markup by default in many widgets.
**How to avoid:** `markup=False` on `Static`/`Label` for user/model-derived text, or `rich.markup.escape`; never format registry strings into a markup template.
**Warning signs:** labels like `[unlabeled]` vanishing from a table.

### Pitfall 6: Screens that re-sort or re-filter the listing
**What goes wrong:** lab-first ordering or intent routing drifts between core and shell; TUI-05 and PFC-07 tests pass separately but the UX disagrees.
**Why it happens:** convenience sorting in a `DataTable`.
**How to avoid:** render `ListingEntry` order as returned; any filter must be an argument to `check_compatibility` (`task_type`, `intent`), not a screen-side list comprehension.

### Pitfall 7: Pilot tests racing message settlement
**What goes wrong:** intermittent failures asserting on state right after `press`/`click`.
**Why it happens:** messages bubble asynchronously [CITED: textual.textualize.io/guide/testing/].
**How to avoid:** `await pilot.pause()`; for worker-completing flows `await app.workers.wait_for_complete()`; inject fakes so workers finish deterministically.

### Pitfall 8: Coverage denominator surprise
**What goes wrong:** coverage gate drops when new TUI modules land lightly tested, or someone "fixes" it with an omit entry.
**Why it happens:** new code is in the denominator immediately; per-module ≥96% required; no omit entries (structure test enforces for preflight and should enforce for tui) [VERIFIED: pyproject.toml `[tool.coverage.run]` omit list has no tui/preflight entry].
**How to avoid:** land worker/screen unit tests with each screen; the panel→core table test and Pilot smoke tests cover the app/wiring modules.

### Pitfall 9: Rich floor mismatch on fresh resolves
**What goes wrong:** textual 8.2.8 requires `rich>=14.2.0`; the repo declares `rich>=13.7.0` [VERIFIED: pyproject.toml:56] — fresh environments silently move to rich 14/15.
**Why it happens:** dependency resolution picks the higher constraint.
**How to avoid:** dev venv already has rich 15.0.0 and the dry-run shows no rich change; still run the existing suite on a leg with textual installed to confirm no rich-14 behavior change in CLI output before merging the first act.
**Warning signs:** truncated/oddly wrapped CLI tables after the first act.

### Pitfall 10: Windows console constraints
**What goes wrong:** relying on glyphs/widths that break under the Windows leg (3.12) or a user's 80×24 console.
**Why it happens:** font and encoding differences; snapshots were excluded for exactly this reason.
**How to avoid:** design for 80×24, ASCII-safe copy, no snapshot/golden tests; Pilot tests run headless on all legs.

## Code Examples

### PEP 562 lazy facade (root, shape)

```python
# Source: PEP 562 (module __getattr__ / __dir__); project pattern per CONTEXT D-01
from importlib import import_module

from .version import __version__

__all__ = [...]  # unchanged from today

_LAZY = {
    "Benchmark": "dnallm.inference",
    "DNADataset": "dnallm.datahandling",
    "DNAInference": "dnallm.inference",
    "DNAInterpret": "dnallm.inference",
    "DNATrainer": "dnallm.finetune",
    "Mutagenesis": "dnallm.inference",
    "cli": "dnallm.cli.cli",
    "get_logger": "dnallm.utils",
    "load_config": "dnallm.configuration",
    "load_model_and_tokenizer": "dnallm.models",
    "setup_logging": "dnallm.utils",
}

def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
```

Note: `load_model_and_tokenizer` resolving through `dnallm.models` re-enters the models lazy facade, which is where the shim ordering lives — the root facade stays free of ordering logic.

### Models lazy facade with the ordering guarantee (D-01 trap a)

```python
# dnallm/models/__init__.py — guarantee travels with the consumer
from importlib import import_module

__all__ = [...]  # unchanged

_HEAVY = {"load_model_and_tokenizer", "DNALLMforSequenceClassification"}

def __getattr__(name: str):
    if name in _HEAVY:
        # transformers_compat must install its device-query patch before the
        # first modeling-symbol resolution (transformers >= 5.19 accelerator
        # query on CUDA-built torch without a visible GPU).
        import_module("dnallm.utils._transformers_compat")
        import_module("dnallm.utils._cuda_compat")
    module = import_module(".model" if name in _HEAVY else ".registry")
    ...
```

### Thread worker + typed message (seam)

```python
# Source: https://textual.textualize.io/guide/workers/
from textual import work
from textual.worker import get_current_worker

@work(thread=True, exit_on_error=False)
def check_compat(self, data_report, task_type: str, prober) -> None:
    worker = get_current_worker()
    report = check_compatibility(  # blocking core call
        data_report, task_type=task_type, prober=prober, probe_shortlist=20
    )
    if worker.is_cancelled:
        return
    self.app.post_message(CompatReady(report))   # post_message is thread-safe
```

### Cancellable download wrapper (Flag 2)

```python
# dnallm/tui/workers.py — no domain logic, only plumbing
class DownloadCancelled(BaseException):
    """Escapes download_model's `except Exception` retry loop."""

@work(thread=True, exit_on_error=False)
def download(self, model: str) -> None:
    worker = get_current_worker()

    def cancellable_downloader(*args, **kwargs):
        if worker.cancelled_event.is_set():
            raise DownloadCancelled()
        return snapshot_download(*args, **kwargs)

    try:
        path = download_model(model, cancellable_downloader)
    except DownloadCancelled:
        self.app.post_message(DownloadFinished(result=None, cancelled=True))
        return
    self.app.post_message(DownloadFinished(result=path, cancelled=False))
```

### Pilot test (headless, deterministic)

```python
# Source: https://textual.textualize.io/guide/testing/
async def test_suitability_renders_verdicts():
    app = PreflightApp(prober=fake_prober, downloader=fake_downloader)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press("enter")          # advance env → data
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.selected_model is not None
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `call_from_thread` for every worker→UI hop | `post_message` typed messages (thread-safe), state-changed handlers | Textual 0.5x+ docs | One reviewable seam; payload types |
| Eager package `__init__` re-exports | PEP 562 `__getattr__` lazy facades | Python 3.7+ | Root/models/utils purity; instant shell startup |
| `pytest-textual-snapshot` everywhere | Pilot-functional assertions | Textual testing guide, v1.4 decision | No cross-OS glyph flake; Windows-safe |
| `os.path` user dirs | `platformdirs` | Long-standing | Windows/macOS/Linux correctness |

**Deprecated/outdated:**
- The `base` extras alias: one-cycle alias removed this phase (PKG-01).
- Snapshot/golden TUI testing in v1.4: explicitly out of scope.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Pilot suite ≤ ~30 s local / ≤ 60 s per test is achievable with fakes | Flag 4 | Slower CI legs; mitigate by splitting per-screen tests |
| A2 | Cooperative cancel (finish current attempt) is acceptable UX to the owner | Flag 2 | If hard-kill is required, the only option is a subprocess download — larger scope |
| A3 | `BaseException` sentinel escapes `download_model`'s retry loop in practice | Pitfall 2 | A 2-line Wave-0 probe (fake downloader raising the sentinel through `download_model`) confirms it before screen work |
| A4 | Env facts (VRAM total/free) come from `torch.cuda.mem_get_info` / device properties in `env_info.py` | TUI-03 | Different torch API names across versions; assert at implementation |
| A5 | Textual behaves identically headless on Windows CI | Pitfall 10 | Docs claim headless testing; Windows leg will prove it in CI |
| A6 | No repo code regresses under rich ≥14.2 | Pitfall 9 | Existing suite run with textual installed (first act) catches it |
| A7 | `platformdirs` is used by the TUI for default config/export paths | Standard Stack | If unused, the extra still matches textual's own constraint — harmless |

## Open Questions

1. **Hard cancel expectation.** What should "cancel" promise on a slow link — "stop after this attempt" (recommended, cooperative) or true interruption (requires subprocess download)? Recommendation: cooperative now; record subprocess as a future Track-C/B item.
2. **Does `all` gain `tui`?** Recommendation: yes (everything-install parity); ask owner only if they want `all` to stay byte-identical to today.
3. **Default export filename / directory.** Discretion — suggest `dnallm_config.yaml` in CWD (matching CLI examples) rather than platformdirs, unless the owner prefers `user_config_dir("dnallm")`.
4. **Shortlist N and probe concurrency cap.** Discretion — D-04 suggests N≈20; recommend probing sequentially or with a small cap (2–3) to respect hub rate limits.
5. **Back-navigation recompute granularity.** Discretion — recommend recompute-on-re-entry only for views whose upstream inputs changed (cheap: reports are already in memory).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `textual` | TUI runtime, Pilot tests | ✗ in dev venv | — (8.2.8 on PyPI) | First act installs `.[tui]`; CI gets it via `test` extra |
| `platformdirs` | TUI defaults | ✓ | 4.12.1 | textual's own dependency |
| `rich` | textual renderer | ✓ | 15.0.0 (needs ≥14.2.0) | — |
| `pytest-asyncio` | Pilot async tests | ✓ (test extra) | auto mode configured | — |
| `uv` | install lines | ✓ | — | pip lane documented |
| Official Textual docs | research | ✓ (fetched) | — | — |

**Missing dependencies with no fallback:** none — `textual` is the first act's install target.

**Missing dependencies with fallback:** none.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Local CLI wizard; no accounts |
| V3 Session Management | no | No server session |
| V4 Access Control | no | Local user's own files |
| V5 Input Validation | yes | Pydantic (`validate_sections` via preflight); `click.Path` for `--config`; path inputs resolved, never interpolated into shell commands |
| V6 Cryptography | no | No secrets handled |

### Known Threat Patterns for a local Textual + PyYAML stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Terminal escape/OSC injection via model names or data-derived labels | Tampering / Spoofing | `markup=False` / `rich.markup.escape` for all untrusted strings; never build markup templates from registry/data values |
| Unsafe YAML deserialization of a user-supplied config | Tampering | Preflight's safe-loader family only; the TUI never loads YAML itself |
| Path traversal via downloaded model paths | Tampering | `download_model` resolves through hub cache roots; the TUI displays the returned path, never constructs one |
| Network probing leaking data | Information Disclosure | Probes are metadata-only (existence/size/revision), ModelScope-first, bounded timeout, degrade to UNKNOWN (PFC-03/06) |
| Writing an invalid config to disk | Repudiation | Save disabled until validation clean (D-13); `generate_config`/`edit_config` write only with explicit `output_path` |

## Sources

### Primary (HIGH confidence)
- `dnallm/__init__.py:20-34` — eager root facade + utils-before-models comment (read this session)
- `dnallm/models/__init__.py:1-37` — eager `.model` import (torch path)
- `dnallm/models/model.py:12-14` — torch/transformers module-level imports
- `dnallm/models/model.py:411-416, 447-486` — `download_model` signature + retry loop
- `dnallm/models/registry.py:33` — registry imports `..utils` (triggers utils init)
- `dnallm/utils/__init__.py:7-8, 42` — eager shim + `support` imports
- `dnallm/utils/transformers_compat.py:14-19, 1399` — `from __future__ import annotations`, `import torch`, `apply_patches()` at import
- `dnallm/utils/support.py:2` — `from torch.cuda import get_device_capability`
- `dnallm/preflight/__init__.py:1-98` — facade docstring + `__all__`
- `dnallm/preflight/report.py:59-391` — report inventory + reason-code enums
- `dnallm/preflight/compatibility.py:132-141` — `check_compatibility` signature
- `dnallm/preflight/validation.py:82-87` — `validate_data` signature
- `dnallm/preflight/config_gen.py:206-224, 382-404` — `generate_config` / `edit_config`
- `dnallm/preflight/availability.py:53, 97-127` — registry import + `probe_availability`
- `dnallm/preflight/validation.py:40-41`, `report.py:31`, `availability.py:54`, `config_gen.py:59`, `compatibility.py:44` — preflight's `..utils` imports
- `tests/preflight/test_structure.py:1-158` — AST structure-test pattern (incl. `:102-109` root-facade guard)
- `dnallm/cli/cli.py:1-95, 280-321` — click group + lazy-import precedent + ImportError handling
- `pyproject.toml` — extras (`dev`, `test`, `full`, `base`, `all`), scripts, pytest addopts, coverage omit list, `rich>=13.7.0`
- `.github/workflows/ci.yml` — all install lines (`.[full]` variants) and the leg matrix
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `.planning/phases/25-…/25-CONTEXT.md` — scope, decisions, constraints
- PyPI JSON metadata for `textual` 8.2.8 and `platformdirs` — requires-python, classifiers, requires_dist
- `uv pip install --dry-run "textual>=8.2,<9"` — resolves `textual==8.2.8`, `linkify-it-py==2.2.0`, no rich change
- Live measurements: `import dnallm.preflight` = 3.89 s with torch/transformers/datasets in `sys.modules`; `import dnallm.models` standalone pulls torch

### Secondary (MEDIUM confidence)
- Textual Workers guide — thread workers, `call_from_thread`, `is_cancelled`, state transitions, `exit_on_error`
- Textual Worker API — `cancel()`, `is_cancelled` note, `cancelled_event`, `WorkerState`
- Textual WorkerManager API — `wait_for_complete(workers=None)`, `cancel_all`, `cancel_group`, `cancel_node`
- Textual Testing guide — `run_test()` headless async context manager, `Pilot.press/click/pause`, pytest-asyncio guidance

### Tertiary (LOW confidence)
- Pilot suite wall-time target (~30 s) — no in-repo precedent (Assumptions A1)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified on PyPI; ranges locked and compatible; dry-run resolution observed
- Architecture: HIGH — every layer boundary read from source; Textual semantics cited from official docs
- Pitfalls: HIGH for import/cancel pitfalls (source + docs evidence); MEDIUM for markup/Windows (docs-level guidance, no in-repo occurrence)

**Research date:** 2026-10-21
**Valid until:** 2026-11-20 (30 days; Textual 8.x and the preflight facade are stable surfaces)

**Environment notes for the planner:**
- The GSD knowledge graph (`.planning/graphs/graph.json`) is stale (380 h, 711 commits behind `604568b`) — graph-derived relationships were not used; all facts above come from direct source reads.
- Codebase-drift pre-check at plan time reports 215 changed structural elements since the last codebase map; advisory only — refresh with `/gsd-map-codebase` if intel docs are wanted before execution.
- `textual` is not installed in the dev venv; the first act installs it (`uv pip install -e ".[tui]"`).
