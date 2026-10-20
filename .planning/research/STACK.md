# Stack Research

**Domain:** Stack additions for a pre-flight terminal UI (data validation → model-suitability → config generation) plus runtime registry consumption on an existing DNA-LLM toolkit (`dnallm` 1.3.3, py≥3.11)
**Milestone:** v1.4 "Pre-flight UX & Registry-as-Truth-Source" (Track A first)
**Researched:** 2026-10-20
**Confidence:** HIGH — every version/API claim below was either live-verified in the DNALLM dev venv (textual 8.2.8, huggingface-hub 1.33.0, modelscope 1.34.0, rich 15.0.0, pyyaml w/ CSafeLoader), verified against PyPI's JSON API, the official Textual docs, or cross-checked against the production sibling repo (`/home/forrest/Github/dnallmmark`, textual 8.2.8 + platformdirs 4.13.0 in daily CI for months). Per-claim tiers in Sources.

## Headline Verdict

**Exactly two new runtime dependencies, both isolated in a `tui` extra: `textual>=8.2,<9` and `platformdirs>=3.6,<5`. Nothing from the textual-dev family (textual-dev, textual-serve, pytest-textual-snapshot) enters any committed extra. Zero new test frameworks: the Pilot harness and SVG screenshot export are core-Textual APIs, and `pytest-asyncio` with `asyncio_mode=auto` is already configured in DNALLM's pyproject. Zero new dependencies for registry loading (pyyaml is already core; CSafeLoader measured 12× faster than pure-Python `safe_load` on the 236-entry registry) and zero for availability probing — the installed `huggingface_hub` and `modelscope` versions expose every needed probe (existence, per-file sizes without weight download, revisions, cache state), all live-verified against `zhangtaolab/plant-dnabert-BPE`. The coverage gate survives unchanged: `test = ["dnallm[tui]", ...]` rides textual onto every pytest-running CI leg with zero workflow edits, so `dnallm/tui/` is honestly measurable under `source_pkgs = ["dnallm"]` and needs no coverage-omit entry.**

---

## Recommended Stack

### A. TUI runtime — the `tui` extra

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **textual** | `>=8.2,<9` (latest on PyPI: **8.2.8**, verified 2026-10-20 via PyPI JSON API) | Async terminal app framework for the pre-flight console (env panel → data stats/validation → suitability → config edit+validate → validated YAML emit; select-to-download) | The sibling dnallmmark runs a production 9-screen/16-service TUI on exactly this pin since 2026-10-11 with daily CI — the capability is proven against this exact codebase family. 8.2.8 *is* the current release, so `>=8.2,<9` tracks the live major with no exposure to 9.x breakage. Pin rationale (inherited from the sibling's D-01, still correct): Textual majors are frequent and real — upgrades are deliberate `uv lock --upgrade-package textual` + CHANGELOG-scan events, never resolver accidents. Requires-Python `<4.0,>=3.9` (classifiers 3.9–3.14) — comfortably spans DNALLM's py3.11–3.13 CI matrix |
| **platformdirs** | `>=3.6,<5` (installed 4.12–4.13 across venvs) | XDG state/config split for any TUI-side persistence (last-check cache, wizard state) via `user_state_dir()` / `user_config_dir()` | Textual 8.2.8 already declares `platformdirs>=3.6.0,<5` transitively — but the rule (sibling D-04, correct dependency hygiene) is: **code that imports a package directly must declare it directly**. If `dnallm/tui/` imports platformdirs for its XDG roots, DNALLM declares it; identical band to textual's own, so the resolver can never fight itself. No transitive-luck imports |

**Direct dependencies textual brings (all already satisfied or conflict-free):** `rich>=14.2.0` (DNALLM floor is `>=13.7.0`, both venvs run rich 15.0.0 — floor stays, resolver picks ≥14.2 when tui is installed), `markdown-it-py[linkify]`, `mdit-py-plugins`, `pygments>=2.19.2,<3`, `typing-extensions` — all pure-Python wheels, all already present transitively via rich/pydantic. Verified: DNALLM's own venv and the sibling venv both run textual 8.2.8 alongside rich 15.0.0 with zero conflicts.

**pyproject.toml shape (integration — this exact edit):**

```toml
[project.optional-dependencies]
tui = [
    "textual>=8.2,<9",
    "platformdirs>=3.6,<5",
]
test = [
    "dnallm[tui]",   # ← the only existing-extra change: textual rides every
                     #   pytest CI leg (.[full] ⊇ dev ⊇ test) with zero
                     #   workflow edits — the sibling's proven D-01 posture
    # ...existing test deps unchanged...
]
```

- Self-referential extras are the established in-repo pattern (`dev = ["dnallm[test,notebook]", ...]`, `full = ["dnallm[dev,test,notebook,mcp]", ...]`); pip (≥21.2) and uv both support them. The pin lives in exactly one place (`tui`).
- Do NOT add `tui` to `full` separately — it already arrives transitively via `test`, and `full` is the alias surface being pruned this cycle (the deprecated `base` alias removal is already a committed v1.4 obligation in the pyproject comment). `all` gains `tui` (`dnallm[full,dev,test,notebook,docs,ui,mcp,fla,tui]`).
- **Facade constraint (hard):** `dnallm/__init__.py` eagerly imports every subpackage (`from .cli import cli`, etc.). `dnallm/tui/` must NOT be added to the facade — otherwise bare `pip install dnallm` (no extras) breaks at import, and the giants venv (`.[cuda130]` only, `dnallm/finetune/megatron.py`-class installs) breaks too. Entry is a click subcommand (`dnallm preflight`) whose command body lazy-imports the TUI — the exact pattern `dnallm/cli/cli.py` already uses for `DNATrainer`/`DNAInference` — with a friendly `pip install dnallm[tui]` ImportError message on bare installs (the optional-dep guard pattern at `dnallm/models/model.py:421-448`).

### B. TUI testing inside pytest — no new frameworks

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **`App.run_test()` + `Pilot`** (core textual) | ships in textual 8.2.8 (`textual/pilot.py`) | Headless app testing: `async with app.run_test(size=(120, 40)) as pilot:` → `pilot.press(...)`, `pilot.click("#selector")`, `pilot.pause()` | Core Textual API, **not** textual-dev — official testing guide states run_test is part of Textual itself and any asyncio-capable framework works. Signature live-verified: `run_test(*, headless=True, size=(80, 24), tooltips=False, notifications=False, message_hook=None)`. Headless mode "prevents Textual from updating the terminal but otherwise behaves as normal" — no console is ever touched, which is what makes the Windows CI leg a non-event. The sibling runs 10+ Pilot test modules this way in CI since 2026-10 |
| **SVG golden corpus via `App.export_screenshot()`** (core textual) | ships in textual 8.2.8 | Drift detection on rendered screens: export SVG in-test, plain UTF-8 string equality against committed files under `tests/tui/__snapshots__/*.svg` | Core API (`export_screenshot(*, title=None, simplify=False) -> str`), byte-deterministic across independent app instances on textual 8.2.8 (sibling probe-verified 2026-10-11, CI-compared daily since). This is the **fallback lane that became the proven lane**: it delivers snapshot testing with zero additional packages. Discipline rules (from the sibling's committed contract — adopt verbatim): ONE terminal size (120×40) for every golden; frozen constructor-injected deterministic fixtures (zero wall-clock reads, zero filesystem scans, zero network); cursor blink disabled on every `Input` before export (blink phase is wall-clock state); static glyph only, no spinner frames; explicit `encoding="utf-8"` on every golden read/write. Regeneration ONLY via a dedicated script path (`python tests/tui/test_snapshots.py --regenerate`), never a pytest flag — CI compares and goes red on drift, it never regenerates |
| **pytest-asyncio** (already present) | `>=1.0` in test extra, `--asyncio-mode=auto` already in addopts | Runs the async Pilot tests | Zero configuration change needed — DNALLM's pytest config already has exactly what Textual's testing guide prescribes for marker-free async tests (the MCP tests already exercise this mode) |

**Collection trap (sibling 08-03 lesson — codified for our test layout):** pytest imports every collected module BEFORE `-m` marker filtering. Every module in `tests/tui/` imports textual at import time, so any pytest leg lacking textual fails **collection**, not gracefully. With `test ⊇ dnallm[tui]`, every pytest-running leg has it (see coverage section) — therefore do NOT add `pytest.importorskip("textual")` guards: they would create `optional-dep:`-typed skips that the `expected_skips.yaml`/`audit_skips.py` discipline would have to allowlist for a condition that never occurs on CI. The guard's only reachable effect would be hiding a broken extras wiring.

### C. Registry runtime loading — zero new dependencies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **pyyaml** (already core dep) | `>=6.0` | Load `dnallm/models/model_info.yaml` (236 entries: 49 pretrained + 187 finetuned) at runtime as the truth source | Measured on this box (2026-10-20): pure-Python `yaml.safe_load` ≈ **69 ms** per load; `yaml.load(text, Loader=yaml.CSafeLoader)` ≈ **5.8 ms** — 12× faster via the bundled libyaml. pyyaml wheels ship CSafeLoader on every platform in the CI matrix (linux/macOS/windows) — verified present in the dev venv. Pattern: `Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)` fallback so a libyaml-less environment (source build) degrades instead of crashing |
| **Cached loader** (load-once, stdlib) | `functools.lru_cache(maxsize=1)` or module-level lazy singleton in a new `dnallm/models/registry.py` | Single runtime consumption seam: typed accessors (by name, by task_type, by base_model lineage, finetuned-entry head/labels/threshold resolution) over the parsed registry | 69 ms per uncached call is imperceptible once but *very* perceptible inside a TUI filter loop re-querying per keystroke — the cache turns registry reads into dict lookups. The registry is **packaged immutable data**: no invalidation is ever needed within a process, so load-once is correct, not a compromise. This module becomes the one seam `load_preset_model`, the pre-flight core (`check_compatibility`), the MCP config consistency check, and the TUI all consume — matching the one-core-many-shells architecture rule from the v1.4 scope |

**Design notes:** return plain dicts (byte-compatible with the existing `MODEL_INFO` shape in `modeling_auto.py`) so the family layer and existing structure tests (`tests/models/test_model_info_registry.py`, which already parses the YAML directly and pins 49/187 counts) keep working. Optional hardening: validate the finetuned-entry schema with pydantic v2 (already core) at load time so schema drift in the now-runtime-load-bearing YAML fails loudly at first registry access instead of surfacing as wrong behavior mid-workflow. The hardcoded `MODEL_INFO` dict remains the *family* layer (`modeling_auto.py`); the YAML becomes the *entry* truth — the registry loader is the boundary between them.

### D. ModelScope/HF availability probing — zero new dependencies (all live-verified 2026-10-20)

The `check_compatibility` availability dimension (source / size / cache / revision, ModelScope-first) needs nothing beyond the pinned `modelscope[framework]>=1.23.2` and `huggingface-hub>=0.29.0`. Verified against the installed versions (modelscope 1.34.0, huggingface-hub 1.33.0) with live read-only probes of `zhangtaolab/plant-dnabert-BPE` — no weights were downloaded by any of these calls:

| Probe dimension | ModelScope (first, owner constraint) | HuggingFace (fallback) | Verified evidence |
|---|---|---|---|
| Repo exists | `HubApi.repo_exists(repo_id, re_raise=False) -> bool` (no-raise boolean path) | `huggingface_hub.repo_exists(repo_id) -> bool` (top-level, since ~0.20; floor 0.29 safe) | MS signature inspected; HF live: returned `True` |
| Files + sizes (no weights download) | `HubApi.get_model_files(model_id, recursive=True) -> List[dict]` — entries carry `Name`/`Path`/`Size` (bytes)/`Sha256`/`Revision`/`Type`/`IsLFS`/`CommittedDate` | `HfApi.model_info(repo_id, files_metadata=True)` → `.sha` + `.siblings[].rfilename/.size` | MS live: 9 files, `model.safetensors` = 368,818,524 B; HF live: same file = **identical byte count** (cross-source consistency holds for the lab models) — total repo ≈ 370 MB known before any download |
| Revisions | `HubApi.list_model_revisions(model_id)` / `get_model_branches_and_tags` | `list_repo_refs(repo_id) -> GitRefs` (branches + tags) | Signatures inspected on both installed versions |
| Metadata (labels/describe) | `HubApi.get_model(model_id) -> dict` (raises `NotExistError` if absent) | same `model_info` payload | Signature inspected |
| Cache state (offline) | `snapshot_download(model_id, local_files_only=True)` → local path if cached, `ValueError` if not; cache layout is `~/.cache/modelscope/hub/models/{org}/{name}` (plain org/name dirs — NOT HF-style `models--org--name`), so an `isdir` fast-path prefilter is cheap | `scan_cache_dir() -> CachedRepo` set with `repo_id`/`size_on_disk`/`repo_path`; fully offline | MS live: cached path returned for plant-dnabert-BPE, clean `ValueError` for a fabricated id; HF live: 8 cached repos enumerated with sizes |

**Probe-wrapper requirements (stack-relevant design constraints):**
- **Suppress the ModelScope local-probe warning** — `snapshot_download(local_files_only=True)` emits `"We can not confirm the cached file is for revision: master"`; the probe wrapper should filter or tolerate it rather than spamming the TUI log pane (observed live).
- **Mirror respect is free:** `huggingface_hub` honors `HF_ENDPOINT` automatically, so the existing `_setup_huggingface_mirror` / hf-mirror.com flow (nightly job sets it job-wide) needs no probe-side changes. ModelScope's endpoint defaults to modelscope.cn — the ModelScope-first order the owner mandated is the default behavior of the default client.
- **Timeouts:** HF `model_info` accepts `timeout=`; ModelScope `HubApi` methods do not expose per-call timeouts (session-level only) — wrap each probe in the pre-flight core's own bounded-wait guard rather than relying on library defaults. Keep probes as **pure synchronous functions with injectable fakes** (the dnallmmark `registries`/`download_queue` service pattern): tests never touch the network; the TUI invokes them through `asyncio.to_thread` / a Textual worker so panels stay responsive.
- **ModelScope API surface caveat:** `HubApi` is public and is what modelscope's own CLI uses, but ModelScope does not publish a deprecation contract the way huggingface_hub documents its stable API. Treat the probe layer as OUR adapter (one module, faked in tests) so a modelscope minor-release signature change is a one-file fix, never a test-suite outage. This is the one MEDIUM-confidence seam in an otherwise HIGH-confidence stack.

### E. Windows CI leg compatibility

| Concern | Finding | Action |
|---------|---------|--------|
| Textual on Windows | Officially supported ("Linux, macOS, Windows and probably any OS where Python also runs"; Windows Terminal "runs Textual apps beautifully"). All textual/platformdeps/pygments/etc. dependencies are pure-Python wheels — **zero compile risk** on `windows-latest`, unlike the v1.3 pygenometracks/pybigwig class of problem | Nothing — the install just works |
| VT sequences / ANSI | Applies only to *live* terminal use, where Textual's own console driver enables VT processing (Windows 10+; runners are Server 2022+). **CI tests never touch a console**: `run_test()` is headless by default and bypasses the terminal entirely | Nothing for CI; recommend Windows Terminal in user docs |
| UTF-8 | **The one real pitfall.** Python 3.11–3.13 on Windows still defaults `open()` to the locale codepage (cp1252 on the runners), and the repo is full of box-drawing/unicode content (existing test files use `# ────` separators; SVG goldens embed unicode) | Every golden-SVG read/write and every unicode fixture file access in `tests/tui/` passes `encoding="utf-8"` explicitly. Never rely on locale default encoding anywhere in the tui package. (UTF-8-default arrives only at py3.15; the matrix tops out at 3.13) |
| Console codepage for test output | Existing suite is already green on the windows leg; pytest output paths unchanged | Nothing |

## Installation

```bash
# User-facing TUI (the new extra)
uv pip install -e ".[tui]"          # or: pip install "dnallm[tui]"

# Everything already covered — every existing pytest lane gets textual for free
# via test ⊇ dnallm[tui]; no CI workflow file changes required:
#   coverage-gate  : uv pip install -e ".[full]"          (py3.11/3.12/3.13 × numpy 2.2.0)
#   test-windows   : uv pip install -e ".[full]"          (py3.12)
#   coverage-nightly/test-mamba: ".[full,fla]" (+".[mamba]") on dnallm-nightly
#   example-nightly: ".[full,fla,dev,mcp]"

# Launch (after implementation lands)
dnallm preflight                     # lazy-imports dnallm.tui inside the click body
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Textual 8.x | Rich `rich.prompt` / plain click prompts | A single-question CLI flow (no panels, no async app) — the existing `model_config_generator` numbered menus are this; the pre-flight wizard with live validation panels is not. Also already in-tree, but it is a console-*rendering* library, not an app framework |
| Textual 8.x | urwid / prompt_toolkit | Effectively never here: urwid's maintenance and Python-3.13+ posture are weaker and it has no async model matching the pre-flight's probe-in-worker needs; prompt_toolkit is a toolkit (you build the app model) not a framework — and neither has the sibling's 30k-LOC production proof against this exact domain |
| SVG golden corpus (core `export_screenshot`) | `pytest-textual-snapshot` plugin | Only if the plugin ships an un-pinned syrupy AND the project adopts pytest 9. Today the plugin's `syrupy==4.8.0` exact pin forces `pytest<9` into the resolve (multi-source verified; it hard-blocked the sibling on pytest 9.1.1). DNALLM's `pytest>=8.4` floor would resolve today but would newly inherit the pytest cap for zero unique capability — the SVG lane already delivers drift detection |
| Cached CSafeLoader | Uncached `yaml.safe_load` per call | Never for this registry: 69 ms per call inside a per-keystroke filter loop is visible jank; the data is immutable per-process so the cache is free correctness |
| Plain-dict accessors | Pydantic-validated entry models | Optional hardening once the schema stabilizes; costs single-digit ms at load. Skip for the first cut if it would delay the loader landing — the existing structure tests already pin the schema from the test side |
| `HubApi` / `HfApi` adapters | Hand-rolled REST (`requests` → modelscope.cn `/api/v1/...`) | Never: the hub libraries are already pinned deps, handle auth/endpoint/response-shape/retries, and are what the CLI of each hub itself uses — hand-rolled REST re-implements a dependency you already ship |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **textual-dev** in any committed extra | Devtools package (dev server, `textual console`); pulls an additional dependency tail; the sibling's owner policy (constraint 10) excludes it from committed groups and nothing in pre-flight needs it — testing and screenshots are core-Textual APIs | On-demand only: `uv run --with textual-dev ...` when debugging an app locally |
| **textual-serve** | Serves TUIs over the web — a deployment product, irrelevant to a local pre-flight console; same exclusion policy | Nothing; if a web pre-flight is ever wanted, the Gradio `ui/` extra already covers browser UX |
| **pytest-textual-snapshot** | `syrupy==4.8.0` exact pin ⇒ `pytest<9` resolver cap (would newly constrain DNALLM's resolve); the `--no-deps` community workaround silently violates the committed floor — forbidden class of fix; and it is textual-dev family | The SVG golden corpus (section B) — same drift detection, zero dependencies |
| **A coverage-omit entry for `dnallm/tui/*`** | Weakens the `fail_under=90` hard gate: `source_pkgs=["dnallm"]` measures the tui package automatically, and since textual rides every pytest leg the code is honestly coverable — omitting it would hide uncovered TUI code from the gate the whole v1 milestone exists to enforce | Land tui modules with same-change tests (owner rule); structure as pure services + thin screens so Pilot tests + SVG goldens + service unit tests cover it |
| **`pytest.importorskip("textual")` in tests/tui** | Creates typed skips that never fire on CI (every pytest leg installs test ⊇ tui) but would mask an extras-wiring breakage as a green skip; skip discipline requires allowlisting each one | The extras wiring itself (test ⊇ tui); fail loudly at collection if the wiring breaks |
| **New YAML libraries** (ruamel.yaml, strictyaml) | pyyaml is already core; the registry is our own data needing no round-trip comment preservation or schema-language indirection | pyyaml + cached loader; (optional) pydantic for schema validation |
| **Adding `dnallm.tui` to the package facade** | `dnallm/__init__.py` eagerly imports every listed subpackage — adding tui makes bare `pip install dnallm` (extras-free) fail at `import dnallm`, and breaks the giants venv (`.[cuda130]`-only install) | Click subcommand with lazy import inside the command body + guarded ImportError with install hint |

## Stack Patterns by Variant

**If the TUI needs persisted state (last env-check, wizard position):**
- Use `platformdirs.user_state_dir("dnallm", ...)` / `user_config_dir(...)` for the state-vs-config split (verified live: resolves to `~/.local/state/dnallm` on Linux) — never `~/.dnallm` hand-rolled paths, never config files for transient state.

**If a suitability verdict must know total download size:**
- Sum `Size` over `get_model_files(..., recursive=True)` entries (ModelScope-first), falling back to `model_info(files_metadata=True).siblings[].size` — both verified to agree byte-for-byte on the lab models; subtract already-cached bytes via `scan_cache_dir()` (HF) / `local_files_only` presence (ModelScope) for the honest "remaining to download" number.

**If TUI panels must stay responsive during probes:**
- Probes stay synchronous pure functions; the TUI calls them via `asyncio.to_thread` (or a Textual worker) and renders results as they land — the sibling's env-check "eight-row table filled as results arrive" pattern.

**If the registry grows past ~1 MB or gains hot reload needs (not in v1.4):**
- Then consider parsing at build/packaging time into a faster format; at 236 entries and 5.8 ms cached-once, YAML is nowhere near a bottleneck.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| textual 8.2.8 | Python 3.11 / 3.12 / 3.13 (CI matrix) | requires-python `>=3.9,<4.0`; classifiers list 3.9–3.14; pure wheel `py3-none-any` |
| textual 8.x `rich>=14.2.0` | dnallm floor `rich>=13.7.0` | No conflict: floor stays, resolver selects ≥14.2 (15.0.0 installed in both reference venvs). Do not add a rich cap |
| platformdirs `>=3.6,<5` | textual 8.x's own identical band | Keep the bands aligned character-for-character so the resolver never sees competing constraints |
| pytest-textual-snapshot (NOT used) | would force `pytest<9` via `syrupy==4.8.0` | Current DNALLM resolve (pytest≥8.4) would cap; sibling hard-blocked on pytest 9.1.1. Excluded by policy and by capability |
| HF probe APIs (`model_info(files_metadata=)`, `scan_cache_dir`, `repo_exists`) | huggingface-hub floor `>=0.29.0` (installed 1.33.0) | All predate the 0.29 floor by years; verified working on 1.33.0 live. `HF_ENDPOINT` mirror honored automatically |
| ModelScope probe APIs (`HubApi.get_model_files`, `repo_exists`, `list_model_revisions`; `snapshot_download(local_files_only=True)`) | modelscope floor `>=1.23.2` (installed 1.34.0) | Verified live on 1.34.0 only — the one seam to keep adapter-isolated (see MEDIUM note in section D) |
| pyyaml CSafeLoader | `pyyaml>=6.0` wheels on linux/win/macOS | libyaml bundled in the wheels; `getattr(yaml, "CSafeLoader", yaml.SafeLoader)` fallback for source builds |

## Coverage-Gate Interaction (the fail_under=90 invariant)

1. `[tool.coverage.run] source_pkgs = ["dnallm"]` automatically includes `dnallm/tui/**` — no config change, no new omit entry, denominator grows honestly.
2. `test = ["dnallm[tui]", ...]` propagates textual to every pytest-running leg: coverage-gate (py3.11/3.12/3.13), test-windows, coverage-nightly and test-mamba (`.[full,fla]` ⊇ full ⊇ dev ⊇ test), example-nightly (`.[full,fla,dev,mcp]`). Zero `.github/workflows/ci.yml` edits.
3. The giants venv (`.[cuda130]` only, `scripts/finetune_mamba2_megatron.py`-class notebook execution) never collects `tests/tui/` — it runs kernels, not the pytest tree — so it is unaffected by the facade constraint too.
4. New tui modules land with same-change tests (owner rule, all phases): structure as pure services (unit-testable, fake-injectable) + thin screens (Pilot tests + SVG goldens) so the package lands above the per-module bar the milestone history holds (~96%+), and the 90% floor survives the larger denominator. The v1 precedent (45.92% → 96.30% test-first) is the template.
5. The `base` alias removal (already a committed v1.4 obligation in the pyproject comment) is unrelated to the tui extra but lands in the same extras-surgery change — sequence it so the alias-removal PR is not entangled with the tui addition.

## Sources

- **PyPI JSON API `pypi.org/pypi/textual/json`** (fetched 2026-10-20) — latest = 8.2.8; requires-python; direct deps incl. `platformdirs>=3.6.0,<5`, `rich>=14.2.0` — HIGH (first-party registry)
- **Textual official docs — Testing guide** (`textual.textualize.io/guide/testing/`) — run_test/Pilot headless semantics, textual-dev not required, pytest-textual-snapshot is optional and only for `snap_compare`, asyncio_mode=auto — HIGH
- **Textual official docs — Getting started / FAQ** — Windows officially supported, Windows Terminal recommended; no VT/ANSI caveats for headless use — HIGH
- **Web search (multi-source): pytest-textual-snapshot syrupy pin** — Textualize poetry.lock (syrupy 4.8.0 `pytest>=7.0.0,<9.0.0`), Codeberg hypergumbo nightly-conflict fix, Alpine build logs — HIGH (three independent sources)
- **Live local probes (dev venv, 2026-10-20)** — ModelScope `HubApi.get_model_files` field shape + byte sizes; `snapshot_download(local_files_only=True)` cached/uncached behavior + cache layout; HF `model_info(files_metadata=True)` sizes/sha; `scan_cache_dir()`; top-level `repo_exists`; pyyaml safe_load vs CSafeLoader timing (69 ms vs 5.8 ms, 236-entry registry); `run_test`/`export_screenshot` signatures on textual 8.2.8 — HIGH (first-party execution; registry counts 49+187 confirmed)
- **Sibling production repo** `/home/forrest/Github/dnallmmark` — pyproject `tui` group (exact pins + constraint-10 comment excluding textual-dev/textual-serve), `tests/tui/` Pilot + SVG-golden corpus with committed fallback-lane rationale, Makefile D-01 "textual rides the existing CI test lane, zero workflow edits" — HIGH (months of green CI as evidence)
- **In-session scope input** — `.planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md` Part 5 (v1.4 scope, Track A first, `tui` extra mirroring dnallmmark's pin) — context, not evidence

---
*Stack research for: DNALLM v1.4 Pre-flight TUI & Registry-as-Truth-Source*
*Researched: 2026-10-20*
