# Pitfalls Research

**Domain:** Adding a pre-flight Textual TUI + pure-function validation/suitability core + runtime registry consumption + ModelScope-first availability probing to DNALLM — a mature, heavily-gated sync PyTorch library (96%+ coverage hard gate `fail_under=90` on every `--cov` leg including Windows, CI matrix py3.11–3.13 / numpy 1.26.4 & 2.2.0 / windows-latest leg, typed-skip allowlist discipline, asyncio MCP layer, `model_info.yaml` registry with load-bearing provenance comments).
**Researched:** 2026-10-20
**Confidence:** HIGH for repo-grounded facts (verified against source at branch `v1.4`); MEDIUM for external behavior backed by official docs (Textual testing/workers guides, huggingface_hub env-vars + rate-limits docs, ruamel round-trip semantics, pytest-textual-snapshot); LOW for undocumented externals (ModelScope API rate limits — no official doc found). Each pitfall notes its grounding.

**Grounding:** `pyproject.toml` (coverage `fail_under=90` + 7-entry omit list, pytest `--asyncio-mode=auto` / `--timeout=300` / `--strict-markers`, extras incl. `full` alias + `tui` not yet existing; installed env: pytest 9.1.1, pytest-asyncio 1.4.0, textual NOT installed), `.github/workflows/ci.yml` (all legs `uv pip install -e ".[full]"`-shaped; test-windows runs bare `--cov`; skip audit in 4 jobs), `dnallm/models/model_info.yaml` (49 pretrained + 187 finetuned entries; PlantHelixSeek provenance comment block at lines ~1876–1890), `dnallm/models/model.py` (runtime `HF_ENDPOINT` set/delete at ~494–498; hardcoded `~/.cache/modelscope/hub` at ~1545; `download_model` retry loop), `dnallm/mcp/configs/mcp_server_config.yaml` + `configs/*_inference_config.yaml` (duplicate name/path/source/task_type/label_names/threshold metadata), `tests/expected_skips.yaml` + `scripts/audit_skips.py`, `.planning/quick/261020-kwt-*` (registry data-layer fix: scripted text insertion, never a yaml.dump round-trip), `.planning/quick/261020-gk4-*` TUI-SURVEY Part 5 (owner directives; pre-flight cluster; merge-or-build verdicts), `.planning/PROJECT.md` v1.4 scope boundaries.

**Phase labels:** roadmap not yet built. Labels are topic-keyed to the owner's build order: **Phase R** (registry runtime loader — `model_info.yaml` consumed at runtime, `load_preset_model` resolution), **Phase C** (pre-flight core — `validate_data` / `check_compatibility` / `generate_config` pure functions), **Phase T** (TUI panels — env → data → suitability → config edit+validate, select-to-download), **Phase M** (MCP config ↔ registry consistency check + one-core-many-shells wiring). Ordering: R → C → T → M (R first because C's suitability verdicts and M's consistency check both consume the registry loader; C before T because the TUI must be a thin consumer, never the owner of verdict logic — TUI-SURVEY §5.3 architecture rule).

---

## Critical Pitfalls

### Pitfall 1 (T): Sync torch/dataset/network code runs on the Textual event loop — the whole UI freezes, tests hang, and pytest-timeout kills the leg

**What goes wrong:**
The pre-flight pipeline's heaviest steps are all blocking-sync: `DNADataset` stats over a user FASTA (pandas/torch), charset/length scans, `check_compatibility` cache detection (filesystem walks over multi-GB model caches), ModelScope/HF existence probes (network RTT), and select-to-download (`download_model`, minutes to hours for evo-class ~12.3 GB). Textual is a single-threaded asyncio message pump: any of these called directly inside `on_mount`, a key handler, or a reactive watcher blocks processing of ALL messages — no repaint, no key input, no cancel. In the test suite the same call under `run_test()` hangs the Pilot and `--timeout=300` fires, turning a UX bug into a red CI leg.

**Why it happens:**
The core library is synchronous PyTorch end to end; the developer writes what reads naturally (`result = validate_data(path)` in the handler). Textual's `@work` decorator and `call_from_thread` are opt-in machinery nobody on this repo has used yet (textual is not even installed in the dev venv). The failure is also load-dependent: a toy CSV in dev validates in 50 ms and hides the freeze until a real 2 GB FASTA meets a user.

**How to avoid:**
- Design rule (borrow the sibling's proven shape, per the milestone priors): every call from a widget into I/O-land goes through a thread worker — `@work(thread=True, exclusive=True)` — with results delivered via `post_message` (thread-safe) or `call_from_thread`, never direct reactive writes from the worker. dnallmmark's deliberate pattern (all service I/O in `@work(exclusive=True, thread=True)` workers, run-id-tagged messages so a stale run can't overwrite a fresh one) is the exact template; the stale-overwrite race is what `exclusive=True` cancels.
- Threads cannot be cancelled: the download path needs cooperative cancellation (check `worker.is_cancelled` / a cancel Event inside `download_model`'s existing retry-loop seam — it already re-evaluates state per attempt), or accept cancel-means-abandon with the queue resuming later (dnallmmark's five-state persistent queue precedent).
- Never `await worker.wait()` inside a message handler — that deadlocks the widget; drive UI off `Worker.StateChanged`.
- Worker exceptions exit the app by default — set `exit_on_error=False` on pre-flight workers and render the error in the panel (the project's collect-all error convention).
- Test pattern: TUI tests inject a deliberately slow fake service (e.g. `time.sleep`-stubbed) and assert the UI keeps processing input mid-flight (`await pilot.press(...)` while the worker runs), plus assert the run-id/stale-guard (start probe A, immediately probe B, assert A's late result never lands).
- Keep the sync core sync: do NOT make `validate_data`/`check_compatibility` async "for the TUI" — the MCP layer and notebooks consume them synchronously; the TUI bridges via workers (same bridge `ModelManager._load_model_sync` already uses for the MCP asyncio layer).

**Warning signs:**
- A widget method calling `dnallm.` I/O functions (`load_config`, dataset loaders, `download_model`, probe functions) without a worker decorator.
- `await` on a worker inside `on_*` handlers; `time.sleep` in any `async def`; reactives written from a thread context.
- TUI tests that only use tiny fixture files (freeze never reproduces locally).

**Phase to address:** Phase T — the worker seam is the first thing built, before any panel logic (it is to the TUI what the pure-function core is to Phase C).

---

### Pitfall 2 (T/C): Coverage-gate erosion — new `dnallm/tui/` code lands in the denominator on every `--cov` leg, while no CI leg installs the `tui` extra

**What goes wrong:**
Two coupled traps, both verified against the current CI shape:
1. **Import availability:** every CI leg installs `.[full]`-shaped extras (ci.yml lines 92/172/261/341/408/543/758). A new `tui` extra that is not a member of whatever those legs resolve will not exist there — TUI test files importing textual at module level then ERROR at collection (a census-breaking, suite-red failure), or get skip-guarded into never running.
2. **Denominator math:** `[tool.coverage.run] source_pkgs=["dnallm"]` includes `dnallm/tui/` on EVERY leg the moment it exists, including test-windows (bare `--cov` → `fail_under=90` fires there too). The suite sits at 96.7%+ with a hard-won 7-entry omit list; a few hundred lines of half-driven widget code drag the number, and the tempting "fixes" are exactly the erosion the gate exists to prevent: new omit entries for `dnallm/tui/*`, blanket `# pragma: no cover` on widget bodies, or ever-green `optional-dep:` skips on legs that could simply install the extra (the dishonesty class the project explicitly bans).
Bonus coupling: v1.4 is also the milestone where the `full` alias is removed (PROJECT.md Current State). The `tui`-extra membership decision made against `full` today silently changes when the alias dies — the pip warn-and-continue class from the v1.3 research (unknown extra warns and exits 0, under-installing).

**Why it happens:**
The gate measures the package, not the test plan; TUI code is the classic "awkward to drive" module class (real projects either reach ~90% via `run_test`+Pilot or exclude their view/modal files — the exclusion route is well-trodden and wrong for this repo).

**How to avoid:**
- Good news first (verified, official docs): `App.run_test()` runs the app IN-PROCESS via a headless driver — the message loop is a background task in the test process, so pytest-cov measures TUI code like any unit test. No snapshot infrastructure and no subprocess-coverage tricks are needed for line coverage.
- CI guard, atomic: the same change that adds the `tui` extra adds it to the install lines of at least `coverage-gate` and `test-windows` (textual is pure-Python; windows-latest is fine), so the TUI tests RUN and COUNT everywhere the denominator lives. Record the decision against the `full`-alias removal plan so the two extras changes don't strand each other.
- Architecture guard (already the owner's plan — hold to it): verdict/suitability/validation logic lives ONLY in the Phase-C pure functions; widgets are presentation. Concretely enforce with a structure test: `dnallm/tui/**` must not import torch/transformers/datasets directly (services own I/O; dnallmmark's "screens are presentation-only over services" docstring convention, made executable).
- Per-module ≥96% standard applies to `dnallm/tui/*` like any Phase-C module; no new omit entries; pragma budget stays at ~3 (v1 precedent).
- TUI tests live in `tests/tui/` (mirroring package layout) with real `run_test` driving — not `optional-dep:` skips.

**Warning signs:**
- A pyproject diff adding `tui = [...]` with no ci.yml install-line diff in the same commit.
- Coverage-gate or test-windows reporting `dnallm/tui/` modules at low coverage while the aggregate still passes (the buffer is being eaten).
- Any `optional-dep:` skip whose leg could have installed textual; any `[tool.coverage.run] omit` addition mentioning tui.
- `pip does not provide the extra 'tui'` (or the post-`full`-removal equivalent) in CI logs — treat as a broken leg even when green.

**Phase to address:** Phase T first act (extra + CI install wiring + structure test before the first panel lands); Phase M re-checks when the `full` alias removal touches the same install lines.

---

### Pitfall 3 (R): A registry round-trip dump destroys the provenance comments — the exact defect this repo already dodged once

**What goes wrong:**
`model_info.yaml` is not just data: it carries the PlantHelixSeek freeze provenance block (label-order source citations, checkpoint commit shas, the REG-02 reasoning — lines ~1876–1890) and a curated section structure. PyYAML discards comments AT PARSE TIME by design; any code path that does `yaml.safe_load` → mutate → `yaml.dump` emits a comment-free, re-quoted, re-flowed file in one commit. The registry-as-truth-source milestone multiplies the tempted paths: a runtime loader that "normalizes" what it reads, a TUI add-entry feature, an MCP consistency checker that "helpfully" rewrites the registry to match configs, or a maintenance script. ruamel round-trip mode is better but not safe either: deleting keys/list entries can destroy adjacent comments, `dict.update()` wipes comment metadata, and it is not byte-stable (trailing whitespace, quoting).

**Why it happens:**
Round-tripping feels like the clean, schema-guaranteed way to edit YAML; the comment loss is invisible to every functional test (the parsed data is identical), so nothing red fires — the provenance record just silently disappears from HEAD. The repo's own 261020-kwt fix proved the correct alternative: scripted text insertion keyed on unique `model: "<repo>"` lines (164 added / 0 modified-deleted lines; provenance block byte-unchanged).

**How to avoid:**
- Design rule: the registry is READ-ONLY at runtime. The Phase-R loader parses and never writes; there is no sanctioned runtime mutation path. Any registry maintenance (adding entries, editing metadata) is scripted text manipulation or manual edit, with the kwt-style keying discipline.
- Registry-integrity guard, extend the existing `tests/models/test_model_info_registry.py` (9 no-network tests from 261020-kwt) with: (a) a provenance-block presence test (the PlantHelixSeek comment block's sentinel lines must appear verbatim); (b) a no-round-trip regression test over the real file — assert that loading via the Phase-R loader produces a parsed model with NO write API surface (loader returns frozen/typed objects, no `save_registry`-style function exists in the module — assert via structure test or simply never build one).
- If a future TUI/MCP feature genuinely needs registry writes: it goes through a dedicated writer module with a byte-stability test (write-back of an unmodified parse must be a no-op diff) — until that exists, grep gate `yaml.dump|safe_dump` over `dnallm/models/` stays at zero sanctioned hits.
- User-facing YAML (`generate_config` output) is NOT registry YAML — no provenance to lose there; that path may dump freely into fresh files.

**Warning signs:**
- Any diff on `dnallm/models/model_info.yaml` where the added/removed line counts are near-equal (a rewrite, not an insertion).
- `git diff` on the registry touching the PlantHelixSeek comment lines or re-quoting keys repo-wide.
- A loader exposing dicts that callers mutate then persist; any function in `dnallm/models/` importing `yaml.safe_dump`/`dump`.

**Phase to address:** Phase R — the loader is born read-only, the integrity tests extend in the same change (owner rule: lib changes ship same-change pytest).

---

### Pitfall 4 (M): The MCP ↔ registry consistency check joins on the wrong key, compares only shallow fields, or auto-writes fixes — manufacturing silent prediction corruption

**What goes wrong:**
The dual-metadata surface is verified real: `mcp_server_config.yaml` maps service names → `model_name` (a registry display title) → a per-model inference YAML that re-states `task.task_type`, `num_labels`, `label_names`, `threshold`, `model.path`, `model.source` — all fields the registry now owns. Three design mistakes, each with a distinct blast radius:
1. **Joining on display titles.** `model_name: "Plant DNABERT BPE promoter"` is a mutable human title — 261020-kwt corrected 2 tRNA title typos and 3 task-class strings in exactly these name fields. A checker keyed on titles either false-alarms on every cosmetic rename or (worse) "resolves" fuzzy matches. Join on `model.path` (repo id) — stable, unique per registry entry, and already the lineage anchor (`base_model` backfill keyed on repo ids).
2. **Comparing only names/presence.** The drift that matters semantically is `label_names` order (frozen label order is a registry contract — the PlantHelixSeek freeze exists because configs carry no label semantics), `num_labels`, `task_type`, `threshold`, `source` (hf vs modelscope changes the entire download/cache path). A name-only checker passes while every prediction is wrong.
3. **Auto-sync fixers.** A checker that rewrites the MCP config (or registry) to "repair" drift writes user-owned config files from inside a server startup path, with the same round-trip hazards as Pitfall 3 and no audit trail. Consistency checking is diagnosis; repair is a human act.

**Why it happens:**
Consistency checks get specified as "make sure the configs agree" — which reads as "make them agree." And the title join is the path of least resistance because `model_name` is literally the field labeled like a foreign key.

**How to avoid:**
- Phase-M deliverable is a PURE function: `check_config_registry_consistency(mcp_config, registry) -> list[Finding]` — collect-all, never raises, never writes (mirror the Phase-C core contract). Findings carry field, both values, file+line where derivable.
- Key on repo id (`model.path` ↔ registry `model:`); treat title mismatch as at most an INFO-level finding.
- Compare the full semantic set (task_type, num_labels, label_names ORDER, threshold, source, and registry head-metadata like `base_model` resolvability for finetuned entries — the kwt integrity tests already prove every finetuned entry has `base_model`; the MCP check inherits that invariant).
- Test pattern: a fixture pair matrix — one clean pair plus one fixture per drift class (label order swapped, num_labels off by one, source hf-vs-ms, threshold drift, orphaned MCP entry, registry entry absent) — asserting the exact finding emitted per class. Plus a committed-configs test that runs the checker over the real `dnallm/mcp/configs/` (no-network) so drift in the repo itself fails CI.
- MCP wiring (per PROJECT.md active requirement): the checker surfaces as tool/startup warnings; the fix is a human edits the YAML.

**Warning signs:**
- A checker whose signature returns a bool or `dict` without file/line provenance; any `open(..., "w")` reachable from the consistency path.
- Fuzzy matching (`difflib`, `fuzz`) anywhere in the join.
- Findings that mention titles but never label order / threshold / source.

**Phase to address:** Phase M — but the fixture-matrix shape is decided at plan time; the pure-function seam means Phase C's contract conventions apply from the first commit.

---

### Pitfall 5 (C/T): Availability probing — network calls leaking into unit tests, HF/ModelScope rate limits, env vars read at import time, and cache layouts that lie

**What goes wrong:**
The availability dimension (`check_compatibility`'s source/size/cache/revision probing) is the most trap-dense feature in the milestone:
1. **Network in unit tests.** Any probe that touches `HfApi`/ModelScope by default drags network into the fast leg — the suite's network-free fast lane breaks (flaky, IP-banned, or CI-runner-dependent), or gets skip-guarded into dishonesty.
2. **Rate limits are real and shared.** Official HF limits: 5-minute windows, anonymous per-IP 500 requests/5min (authenticated 1,000+); `model_info` sits in the API bucket; transformers ≥5 can burst `model_info` calls into 429 (LlamaFactory PR #10155 — and this repo rides transformers 5.17). The nightly box and CI runners share egress IPs; a "probe all 236 registry entries" loop is 236 API calls in seconds — a 429 storm, and 429s mis-read as "model unavailable" turn rate-limiting into false negative verdicts. ModelScope limits: officially undocumented (LOW confidence — treat as unknown, design for HTTP-429-class resilience regardless).
3. **Env vars are read at IMPORT time by huggingface_hub** (official docs): `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT` are baked at import. Two live consequences: (a) `dnallm/models/model.py` already toggles `os.environ["HF_ENDPOINT"]` at RUNTIME (~line 494) — after huggingface_hub import that toggle may be inert, so the pre-flight env panel must report the EFFECTIVE endpoint (read the library's resolved constants, not `os.environ`), and any in-TUI endpoint toggle needs the subprocess boundary to actually take effect; (b) `HUGGINGFACE_HUB_CACHE` is deprecated in favor of `HF_HUB_CACHE`, with `HF_HOME`/`XDG_CACHE_HOME` precedence — cache detection reading only `~/.cache/huggingface` mis-reports every custom-layout machine as "not cached".
4. **Cache-layout variance.** HF: blobs/snapshots/refs layout resolvable via `try_to_load_from_cache` / `snapshot_download(local_files_only=True)` — use those, not path spelunking. ModelScope: TWO layouts in the wild (pre-1.23.0 `hub/<org>/<model>` vs post-1.23.0 `hub/models/--<org>/<model>`, auto-migration, `MODELSCOPE_CACHE` historically inconsistent about appending `hub/` — issue #845) — and `dnallm/models/model.py:1545` HARDCODES `~/.cache/modelscope/hub`. A cache detector supporting one layout reports false "not downloaded" and the user re-pulls 12 GB.
5. **`HF_HUB_OFFLINE=1` makes every HfApi call RAISE `OfflineModeIsEnabled`** — an offline user's "probe remote availability" must degrade to cache-only verdicts, not crash.

**Why it happens:**
"Is the model available?" reads like one boolean; it is actually a three-valued, cache-aware, rate-limited, environment-sensitive question. Each sub-trap is invisible in a dev session (cached models, warm IP, default env).

**How to avoid:**
- Probe interface is injectable and network-free by default in unit tests: the pure core takes a `remote_probe` callable (or pre-fetched facts); tests inject fakes. Real-network probing tests live behind the existing `slow` marker + typed `network-unavailable:` skip prefix with probe evidence in the message (existing D-08 convention).
- Availability is a three-state verdict per source: `cached` (local detection, offline-safe) / `remote-confirmed` / `unknown` (rate-limited, offline, or error — NEVER mapped to "unavailable"). A 429 or timeout yields `unknown` with the reason; the TUI renders it as such.
- Never probe the full registry: the suitability panel probes the user's candidate shortlist (ModelScope-first, `zhangtaolab/*` prioritized). If a bulk check ever exists, it is sequential with backoff honoring `RateLimit`-style headers, and its results cached (dnallmmark env-check same-day cache precedent).
- Cache detection: HF via `huggingface_hub` APIs (`try_to_load_from_cache`, honoring resolved constants rather than re-reading env); ModelScope via a layout-aware resolver covering BOTH layouts + `MODELSCOPE_CACHE` (and fix or wrap the hardcoded path in model.py — scope-permitting; otherwise the detector documents the hardcode).
- Env panel reads effective values through library constants (e.g. huggingface_hub's resolved endpoint constant), and its "toggle endpoint" action spawns the affected work in a subprocess (or documents restart-required) rather than promising an in-process switch.

**Warning signs:**
- `check_compatibility` unit tests that pass offline only because the dev machine is offline; probe calls not behind an injected seam.
- Probe code mapping every exception to `available=False`.
- Cache detection built from hardcoded paths (`~/.cache/...` literals) or single-layout assumptions.
- CI logs showing 429s from HF/ModelScope; nightly runs newly slow from re-downloads the cache detector missed.

**Phase to address:** Phase C (the pure core owns the three-state verdict + injectable probe seam + cache resolvers); Phase T consumes verdicts and adds the subprocess boundary for env toggles; real-network acceptance rides the nightly lane only.

---

### Pitfall 6 (T): Terminal-environment variance and the snapshot-infra temptation — conhost degradation, MSYS2/SSH edge cases, and SVG goldens that differ per OS

**What goes wrong:**
Two failure families:
1. **Terminal variance in the field:** Textual on Windows enables VT processing via the Win32 Console API; legacy conhost degrades rendering (symbols/glyphs — textual#4974; IME text loss — textual#5456; no truecolor/emoji/OSC 52), MSYS2/mintty mouse input is broken (msys2#22188), and `WT_SESSION`-style terminal detection is unreliable (rich#140). SSH sessions add another variance axis (TERM, terminfo, latency). The dev machine (Linux + a modern terminal) shows none of this; the primary dev machine per CI comments is Windows 11 — users will hit conhost first.
2. **Snapshot testing adopted as the default TUI test style:** `pytest-textual-snapshot` compares SVG screenshots; first run ALWAYS fails (no baseline — must `--snapshot-update` after eyeballing), and SVG output embeds OS-specific glyphs/font references — third-party practice is to pin one OS for golden generation, pin terminal size, freeze clocks, and settle animations, because cross-OS font/glyph differences produce false diffs. The docs' "works well in CI on all supported operating systems" is about running, not about goldens being byte-portable. Worst case for this repo: snapshots generated on Linux, test-windows leg red on every run, someone "fixes" it by regenerating goldens on Windows — now both legs red against each other's baselines.

**Why it happens:**
Snapshot tests look like the fastest way to "test" a UI; headless `run_test` feels like it tests less because it asserts behavior, not pixels. And nobody on the repo has shipped a TUI before, so conhost/MSYS2 realities are folklore until a bug report arrives.

**How to avoid:**
- Default test style = functional Pilot tests through `run_test()` (in-process, headless, no TTY needed — CI's no-TTY is a non-issue for these; they double as the coverage engine per Pitfall 2). Assert on widget state and messages, not rendered text grids where avoidable; where layout matters, assert on the widget tree / line contents (Textual export APIs), not SVG.
- Snapshot infra is NOT adopted in v1.4 (it would be a new dev-dependency and a new flakiness class). If ever adopted later: goldens generated on exactly ONE CI OS, terminal size pinned via `run_test(size=...)`, clocks/animations settled, and the snapshot suite deselected (not skipped-loose) on other OSes.
- Windows exposure is real but test-windows only proves the headless path — keep textual installed there (Pitfall 2) so import/mount logic is at least executed on Windows each push.
- TUI feature scope defends against variance: avoid mouse-dependent interactions for core flow (keyboard-first, dnallmmark's digit-prefixed nav precedent), avoid truecolor/emoji in verdict glyphs, degrade gracefully on narrow terminals (SSH default 80 cols).
- Document supported terminals honestly in the TUI docs (Windows Terminal recommended over conhost) — the project's docs-honesty convention.

**Warning signs:**
- `pytest-textual-snapshot` or SVG files appearing in the repo/tests; `--snapshot-update` in any CI script.
- Tests asserting exact full-screen renders; mouse-only paths to core functionality.
- Bug reports mentioning cmd.exe/PowerShell-conhost rendering or mintty input.

**Phase to address:** Phase T (test-style decision at plan time: Pilot-functional, no snapshot infra); the terminal-support documentation rides the milestone's docs pass.

---

### Pitfall 7 (C/T/M): Event-loop flakiness in the TUI test batch — pass-alone-fail-together teardown, and the anyio/pytest-asyncio seam under an MCP client

**What goes wrong:**
The suite already runs `--asyncio-mode=auto` (pytest-asyncio 1.4.0 / pytest 9.1.1) for the MCP layer. Adding a TUI test batch introduces two known integration hazards:
1. **Loop-teardown order sensitivity:** the pytest-asyncio ≥0.25 era broke Textual's own suite with `RuntimeError: There is no current event loop in thread 'MainThread'` — tests passing individually, failing when run together. Modern versions are fixed, but the failure CLASS (per-test app instances leaving timers/workers/drivers alive into the next test) is exactly what a batch of `run_test` apps with pending thread workers produces: a test that exits `run_test()` while its download worker still sleeps holds a reference to a dead app; the next test's assertions see stale messages or the worker's late `post_message` raises into the wrong loop.
2. **anyio vs pytest-asyncio dual-auto conflict:** the MCP SDK (FastMCP) is anyio-based; anyio's pytest plugin in auto mode officially conflicts with pytest-asyncio auto mode. Today the repo's MCP tests run under pytest-asyncio only — but a Phase-M/T test that starts an MCP client (or imports something registering the anyio plugin) inside the same suite can flip the whole async test population's behavior.

**Why it happens:**
Both are invisible at single-test scale and surface as "flaky CI" — which on this repo triggers the worst possible response: someone marks the test flaky/skipped rather than fixing teardown (the ever-green-skip dishonesty again).

**How to avoid:**
- Every TUI test's `run_test()` context manager body awaits worker quiescence before exit: `await pilot.pause()` after actions, and for worker-driving tests, await the specific terminal message/state (never exit on a wall-clock sleep); where a long-lived worker is deliberate, cancel it and await cancellation inside the test.
- Structure tests so each test builds its own App instance (no module-level app); no cross-test app state.
- Keep TUI tests plain pytest-asyncio (auto mode as-is); never add anyio marks; if a test genuinely needs an MCP client session, prefer the existing MCP test patterns (their own client/transport harness) over embedding a live client inside `run_test` — or drive it via the pure core seam instead.
- Guard the guard: the typed-skip system means a "flaky TUI test" skip needs an allowlist entry — which forces the conversation to stay honest. Budget for making tests deterministic instead (deterministic fakes, no real timers — inject clocks where the core exposes timeouts).

**Warning signs:**
- TUI tests that pass with `-p no:randomly`-style single runs but fail in batch; "no current event loop" in CI logs.
- `asyncio.sleep(...)` used as a synchronization primitive in tests (race by construction).
- Import of `anyio` pytest plugin or `@pytest.mark.anyio` anywhere under `tests/`.

**Phase to address:** Phase T (teardown discipline from the first TUI test file — retrofitting determinism into a flaky batch is far more expensive).

---

### Pitfall 8 (T, roadmap-level): Wizard scope creep — the pre-flight TUI absorbing post-flight surfaces and config-editor ambitions

**What goes wrong:**
The owner already scoped this once (PROJECT.md: Track B substrate merge and Track C R2/R3/R5/R6 are NOT v1.4), and the survey's §5.3 pins the wizard to four capabilities. The creep vectors are attractive and adjacent: the suitability panel drifting into R1's loaded-model exploration REPL; the config panel drifting from "edit + validate + emit" into authoring arbitrary new configs with model discovery; select-to-download drifting into a full download-manager UI (dnallmmark's data_manager territory); the data panel drifting into benchmark-dataset management (explicitly SPLIT-MERGE downstream). Every absorbed capability adds widget surface that is (a) hard to drive under `run_test` (Pitfall 2/6 pressure), (b) tempting to give its own logic instead of routing through the Phase-C core, and (c) re-doing something the sibling already owns — violating the owner's "no duplicate development" directive.

**Why it happens:**
A wizard that validates data naturally sits next to "and then just run it" — each increment looks like UX polish. The TUI-SURVEY Part 4 record shows this milestone's planning already fought this battle once (the original seven exclusions).

**How to avoid:**
- Anti-feature list pinned in the roadmap (from PROJECT.md + survey §5.4/§5.5): no loaded-model exploration (R1 core), no training dashboard (R2), no MCP control room (R3), no VEP runner (R5), no interpretation triage (R6), no benchmark-dataset download management, no custom-entry gauntlet wizard (Track B merge candidates, later).
- Hard architecture rule made testable: every panel's every action maps 1:1 onto a Phase-C core function call (validate_data / check_compatibility / generate_config / registry queries / download via the ModelScope-first path). A panel action with no core function beneath it is scope creep by definition — the Phase T plan lists the panel→function table explicitly.
- The config edit+validate panel ABSORBS R4 but only as stated: edit existing configs + live collect-all validation + validated emit — not a general YAML editor, not a new-config authoring wizard (the click generator remains the authoring tool).
- `zhangtaolab/*`-first ordering is UX prioritization inside the suitability panel, not a filtering feature that hides non-lab models.

**Warning signs:**
- TUI modules importing inference/training entry points (only registry queries, core functions, and download belong in scope).
- Panel→function table rows with no core function name; widgets containing verdict logic inline.
- Roadmap items phrased "and while we're in the TUI, also…"

**Phase to address:** Roadmap-creation time (the anti-feature list goes into the v1.4 roadmap's scope boundaries); enforced per-phase via the panel→function structure test in Phase T.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Calling sync core functions directly in TUI handlers (no workers) for "fast" ops | Less machinery; works on toy data | Event-loop freezes on real data; hangs under pytest-timeout | Never — the worker seam is load-bearing from day one |
| `optional-dep:` skips for TUI tests on legs that could install `.[tui]` | Green CI without touching install lines | TUI code unexecuted + uncovered; the ever-green-skip dishonesty class | Only on legs that genuinely cannot install it (none known — textual is pure-Python) |
| Omitting `dnallm/tui/*` from coverage "because UI" | Gate stays green while TUI ships untested | The 7-entry omit list discipline breaks; every future UI module cites the precedent | Never |
| Registry "normalizing" write-back on load (round-trip) | Clean typed API over the YAML | Provenance comments destroyed silently (the defect kwt dodged) | Never — loader is read-only; maintenance is scripted text edits |
| Boolean availability (`exists=True/False`) instead of three-state | Simpler verdict type | Rate-limited/offline states masquerade as "unavailable"; users re-download 12 GB | Never for remote probes; two-state acceptable for pure-local facts |
| Title-keyed MCP↔registry join | Works today with matching titles | Every cosmetic rename breaks resolution; fuzzy-match "fixes" corrupt joins | Never — repo id is the key |
| Auto-sync fixer in the consistency check | One-click remediation UX | Silent writes to user configs; round-trip hazards; no audit trail | Never in v1.4; a future explicit `--fix` CLI is the sanctioned shape if demanded |
| Snapshot goldens as primary TUI tests | Fast visual coverage feeling | Cross-OS font/glyph false diffs; `--snapshot-update` becomes routine (gate theater) | Not in v1.4; revisit only with single-OS golden discipline |
| `asyncio.sleep` synchronization in TUI tests | Tests pass most runs | Batch-order flakiness; skip-creep pressure | Never — await terminal messages/states |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| huggingface_hub env vars (`HF_HOME`/`HF_HUB_CACHE`/`HF_ENDPOINT`) | Reading/writing `os.environ` at runtime and assuming effect | Vars are read at IMPORT time (official docs): report effective values from library constants; endpoint toggles need a subprocess boundary or restart; honor deprecated `HUGGINGFACE_HUB_CACHE` |
| HF Hub API (model_info) | Bulk-probing the registry; treating 429/timeout as "unavailable" | Probe shortlists only; anonymous per-IP 500 req/5min; 429 → `unknown` verdict with reason; transformers≥5 can burst model_info (LlamaFactory #10155) |
| HF cache detection | Path-spelunking `~/.cache/huggingface` | `try_to_load_from_cache` / `snapshot_download(local_files_only=True)`; respect `HF_HUB_OFFLINE` (raises `OfflineModelIsEnabled`) and Windows symlink degradation (`HF_HUB_DISABLE_SYMLINKS`) |
| ModelScope hub | Assuming one cache layout; trusting the hardcoded `~/.cache/modelscope/hub` | Both layouts (pre/post v1.23.0 `hub/models/--org/model`), `MODELSCOPE_CACHE` quirks (#845); existence via `HubApi().get_model` try/except or `repo_exists`; rate limits unknown — resilient to 429-class responses |
| ModelScope-first download path | Reusing HF retry semantics verbatim | `download_model`'s retry loop is HF-shaped; MS failures classify differently (kwt-era evidence: ms pins via `git ls-remote`); keep the existing source-dispatch, add MS-aware reason classes |
| Windows consoles | Assuming VT everywhere; mouse-first interactions | Recommend Windows Terminal in docs; keyboard-first core flow; conhost degrades glyphs/IME (textual#4974/#5456); test-windows proves headless import/mount only |
| CI runners (no TTY) | Assuming TUI code can't be tested in CI | `run_test()` is headless and in-process — no TTY needed, coverage measured; install `.[tui]` on the legs that run it |
| pytest-asyncio + MCP SDK (anyio) | Mixing anyio plugin auto mode with pytest-asyncio auto mode | Stay plain pytest-asyncio auto (current setup); no anyio marks; MCP-client scenarios reuse existing MCP test harness patterns |
| Existing gates (census/skips) | New TUI test files without re-pinning | Any test-set change re-measures the census triple + `expected_skips.yaml` delta same-change (v1.3 Pitfall 6 discipline still applies); register any new marker in pyproject (`--strict-markers`) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Probing all 236 registry entries on panel open | Seconds-to-minutes stall; 429 storm; nightly box IP flagged | Probe the candidate shortlist on demand; cache verdicts (same-day cache precedent); ModelScope-first single-source probe per model | First real user session / first nightly with the TUI tests |
| Re-running `validate_data` stats on every keystroke/config edit | UI jank on large files; duplicate full-file scans | Stats computed once per loaded dataset, keyed by path+mtime; validation re-runs on the cached stats | Files >100 MB |
| Cache detection walking the whole HF/MS cache tree per check | Multi-second per-model latency, disk thrash on the nightly box's giants tier | Directory-existence + `refs` lookup per repo (O(1) per model), never tree walks; resolve root once per session | GB-scale caches (nightly box) |
| Per-request remote checks with no concurrency bound in the TUI | Worker thread explosion when the user arrow-keys through the model list | `exclusive=True` worker groups per panel (cancels superseded probes); debounce selection changes | Fast keyboard navigation |
| TUI test batch wall-time growth (each test boots an app) | Fast leg budget creep (coverage-gate 90-min cap) | Lean app fixtures (mount one screen per test, not the full wizard); shared no-network fakes; measure the batch in the first Phase-T PR (D-12 budget precedent) | ~50+ TUI tests |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Validators raising on hostile data shapes (user CSV/FASTA/YAML) | Pre-flight crashes on malformed input — the panel exists precisely to surface these; a raise is a DoS of the feature | Collect-all validators never raise on untrusted shapes (dnallmmark `custom_validation` precedent); every problem becomes an ordered finding |
| Unvalidated model names from user config reaching download/probe URLs | Path/repo-id injection into hub APIs or filesystem paths | Repo ids validated against registry-resolved values or strict `org/name` pattern before any probe/download; local paths only via explicit opt-in |
| Registry or MCP config writes driven by remotely-fetched metadata | A hostile/renamed hub repo rewriting local config through the consistency path | Consistency check never writes (Pitfall 4); registry read-only (Pitfall 3); repairs are human edits |
| Downloaded model content treated as trusted code (`trust_remote_code=True` heritage) | Remote-code execution class inherent to the loading chain — pre-flight only SELECTS, must not LOAD | Pre-flight never instantiates models (no `from_pretrained` in Phase C/T code paths); keep selection and loading separated until the user explicitly proceeds |
| Probe results cached to a shared world-readable location | Registry/machine fingerprint leak on multi-user boxes | Any probe cache under the user's own cache root with default perms; no tokens in verdict payloads |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|------------------|
| Raw Pydantic `ValidationError` shown in the TUI (today's CLI behavior imported unchanged) | Validator-internals tracebacks — the #1 complaint the pre-flight cluster exists to kill | Collect-all panel: every bad field at once, mapped to section/field, with the fix hint; the core returns findings, the TUI renders them |
| Boolean availability rendered as ✓/✗ | "Unavailable" for rate-limited/offline states; user abandons a perfectly cached model | Three-state glyphs with reason lines (`cached`, `remote-confirmed`, `unknown: rate-limited`) |
| Silent charset/tokenizer mismatch surfacing post-download (or never) | Wasted 12 GB download; silently wrong results — the worst failure mode in the product (survey §5.2) | Charset/alphabet/length checks are the FIRST validation findings, before any download is offered |
| Probe verdicts presented as timeless | User acts on a stale "not cached" from yesterday | Verdicts carry a timestamp + invalidation on source/env change; panel re-probes on focus if stale |
| Wizard that only authors NEW configs | The common case (tweak last run's YAML) unsupported | Edit-existing is the primary flow (R4 absorbed as stated); authoring stays with the click generator |
| Download with no cancel/resume story | Hours stuck on a 12 GB pull inside a TUI | Cooperative cancel; queue state survives exit (dnallmmark five-state queue precedent); resumable snapshot downloads |

## "Looks Done But Isn't" Checklist

- [ ] **Event-loop safety:** every I/O path from widgets goes through `@work(thread=True, exclusive=True)` — verify a test injects a slow fake and proves input processing mid-flight (Pitfall 1).
- [ ] **Coverage wiring:** `.[tui]` installed on coverage-gate AND test-windows; `dnallm/tui/` modules at ≥96% with NO new omit entries; no `optional-dep:` TUI skips on legs that installed the extra (Pitfall 2).
- [ ] **Structure rule:** `dnallm/tui/**` imports no torch/transformers/datasets; every panel action maps to a core function (Pitfall 8's table exists as a test).
- [ ] **Registry integrity:** provenance-block presence test passes; loader exposes no write path; `git diff` on model_info.yaml after any Phase-R change is insertion-shaped only (Pitfall 3).
- [ ] **Consistency check:** fixture matrix covers every drift class (label ORDER, num_labels, task_type, threshold, source, orphan, missing); keyed on repo id; committed `dnallm/mcp/configs/` checked in CI; checker provably writes nothing (Pitfall 4).
- [ ] **Probing:** unit tests pass with network blocked; 429 fixture → `unknown` (not unavailable); HF offline env → cache-only verdicts, no raise; ModelScope both cache layouts detected in tests with synthetic trees (Pitfall 5).
- [ ] **Env panel:** reports EFFECTIVE endpoint/cache dirs from library constants (import-time semantics respected); MS layout reported correctly on a machine using `MODELSCOPE_CACHE` (Pitfall 5).
- [ ] **Async hygiene:** TUI test batch green in random order / repeated runs; no `asyncio.sleep` synchronization; no anyio marks (Pitfall 7).
- [ ] **Gate honesty:** census triple re-pinned and `expected_skips.yaml` delta present in the same commit as any TUI test-set change; new markers registered (existing discipline).
- [ ] **Docs:** supported-terminals note (Windows Terminal over conhost); the `tui` extra documented with its install lane; no docs mention of snapshot testing as the test approach.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Event-loop blocking shipped (UI freezes on real data) | MEDIUM | Move the offending calls into workers (mechanical if the panel→function table held); add the slow-fake responsiveness test; no data-model change needed |
| Coverage gate breached by TUI modules | MEDIUM | Stop-gap: none honest — write the missing Pilot tests (run_test drives everything in-process); if a leg lacks the extra, fix the install line, never the omit list |
| Registry provenance comments destroyed by a round-trip commit | HIGH | Restore the file from git history (comments are unrecoverable from the data alone); add the provenance-presence test that should have existed; audit the writer for other files touched |
| Consistency checker joined on titles / auto-wrote configs | MEDIUM | Re-key on repo id; strip write paths; re-run fixture matrix; audit any config files the fixer touched (git diff on dnallm/mcp/configs/) |
| False "unavailable" verdicts from rate limits | LOW | Map 429/timeout to `unknown` with reason; add the 429 fixture; re-probe on demand |
| Cache detector misses ModelScope new layout | LOW | Add the layout resolver + synthetic-tree test; users recover by re-probing (the damage is a wasted download, not corruption) |
| TUI test batch flaky in CI | MEDIUM | Fix teardown (await terminal messages; cancel workers); if a test resists determinism, isolate its app lifecycle — do NOT reach for the skip allowlist |
| Snapshot goldens adopted then cross-OS red | HIGH (if kept) | Deselect snapshots on non-golden OSes or drop the infra entirely for Pilot-functional tests; regenerate goldens on exactly one OS |
| Scope creep discovered mid-phase | LOW-MEDIUM | Cut to the panel→function table (creep has no core function beneath it); move cut items to Track B/C backlog — the owner already assigned their home |

## Pitfall-to-Phase Mapping

Build order: **R → C → T → M** (R first: both C and M consume the runtime registry loader; C before T: the TUI is a thin consumer by architecture rule; M last: consistency checking needs both the loader and the core's finding-type conventions; MCP tool exposure of core functions rides M's wiring).

| Phase (topic) | Order | Pitfalls Prevented | Verification That Prevention Worked |
|---------------|-------|--------------------|-------------------------------------|
| Phase R — registry runtime loader (`load_preset_model`, finetuned-entry resolution with heads/labels/threshold, base↔finetuned axis) | 1st | 3 (+ keying groundwork for 4) | Loader read-only (no write API; structure test); provenance-block integrity tests extended; kwt's 9 registry tests still green unchanged; registry diff insertion-shaped after any data change |
| Phase C — pre-flight core (`validate_data` / `check_compatibility` / `generate_config` pure functions; availability dimension) | 2nd | 5, 4's contract shape (pure-finding seam), 1's sync-core boundary | Core is sync + network-free under injected probes; three-state availability with 429/offline fixtures; cache resolvers tested against synthetic HF+MS (both layouts) trees; collect-all validators never raise on hostile fixtures; ≥96% per module, no new deps beyond the existing footprint |
| Phase T — TUI panels + `tui` extra + CI wiring | 3rd | 1, 2, 6, 7, 8 | Worker-seam responsiveness test green; `.[tui]` on coverage-gate + test-windows install lines; `dnallm/tui/**` ≥96% with no omit additions; panel→function structure test; test batch green in random order; no snapshot infra; census triple + skips re-pinned same-change |
| Phase M — MCP consistency check + one-core-many-shells | 4th | 4 (+ re-check 2's extras wiring after `full`-alias removal touches install lines) | Fixture-matrix green per drift class; committed-configs check green in CI; checker writes nothing (fixture asserts FS untouched); MCP tools calling core functions reuse the same verdict types; `pip does not provide the extra` absent from all CI logs |

**Ordering rationale:** R before everything because the registry loader is the data contract both later phases consume — and Pitfall 3 (read-only loader) is only cheap on day one. C before T so the TUI never owns verdict logic (Pitfalls 1/8's architecture half). T's first act is the CI/extra wiring (Pitfall 2), not panels. M last because its fixture matrix wants the loader's keying and the core's finding types to already exist.

**Research flags for phases (for the roadmap author):**
- Phase T: needs plan-time decisions — worker-seam shape (message types vs call_from_thread), download cancel semantics (cooperative vs abandon+resume), and the `tui`-extra membership vs the `full`-alias removal ordering (same-commit atomicity).
- Phase C: needs plan-time spikes — effective-endpoint reporting through huggingface_hub resolved constants (verify the runtime `HF_ENDPOINT` toggle's actual effect in-process at the installed versions), and the ModelScope cache-layout resolver against the installed modelscope version.
- Phase M: needs an owner call — whether the consistency check also covers `configs/*.yaml` user-facing examples or only `dnallm/mcp/configs/`.
- Unknown/LOW-confidence external: ModelScope API rate limits (no official doc found — design for resilience, don't encode assumptions); cross-OS SVG snapshot portability (moot if snapshot infra stays out).

## Sources

- Repository (HIGH confidence, primary; verified at branch `v1.4`): `pyproject.toml` (coverage/omit/fail_under, pytest ini, extras); `.github/workflows/ci.yml` (install lines 92/172/261/341/408/543/758; test-windows bare `--cov` line 185; skip-audit in 4 jobs); `dnallm/models/model_info.yaml` (provenance block ~1876–1890; 49+187 entries); `dnallm/models/model.py` (HF_ENDPOINT runtime toggle ~494–498; hardcoded MS cache ~1545; download retry loop); `dnallm/mcp/configs/mcp_server_config.yaml` + `promoter_inference_config.yaml` (dual metadata fields); `dnallm/mcp/config_manager.py`; `tests/expected_skips.yaml`; `.planning/quick/261020-kwt-*` (round-trip dodge, scripted insertion, registry integrity tests); `.planning/quick/261020-gk4-*` TUI-SURVEY Part 5 (owner directives, pre-flight cluster, merge verdicts); `.planning/PROJECT.md` (v1.4 scope boundaries, Track A/B/C).
- Textual official docs (MEDIUM-HIGH; primary for framework behavior): testing guide — https://textual.textualize.io/guide/testing/ (run_test/Pilot headless, asyncio auto mode documented, snapshot first-run failure, CI claim); workers guide — https://textual.textualize.io/guide/workers/ (thread=True/exclusive=True, call_from_thread/post_message, wait() deadlock, exit_on_error).
- huggingface_hub official docs (MEDIUM-HIGH): environment variables — https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables (import-time reading of HF_HOME/HF_HUB_CACHE/HF_ENDPOINT; HUGGINGFACE_HUB_CACHE deprecated; HF_HUB_OFFLINE raise; symlink degradation); Hub rate limits — https://huggingface.co/docs/hub/rate-limits (5-min windows; anonymous 500/5min per IP; 429 + RateLimit headers).
- pytest-asyncio/anyio/Textual interplay (MEDIUM, cross-checked): anyio testing docs auto-mode conflict — https://anyio.readthedocs.io/en/latest/testing.html; pytest-asyncio ≥0.25 Textual suite breakage + Ubuntu textual 2.1.1 patch — https://lists.ubuntu.com/archives/plucky-changes/2025-March/012326.html and https://blog.gitcode.com/c7cd6ec8f8a51f698d5cd4e7c797875f.html.
- pytest-textual-snapshot + SVG determinism (MEDIUM): https://github.com/Textualize/pytest-textual-snapshot (SVG comparison, --snapshot-update, HTML report); third-party determinism guidance (pin OS for goldens, pin size, freeze clocks) — https://www.claudepluginhub.com/skills/testland-qa-cli-tools-plugins-qa-cli-tools/tui-snapshot-tester and https://raw.githubusercontent.com/sipyourdrink-ltd/bernstein/refs/heads/main/docs/contributing/render-freshness.md; cell-grid alternative — https://pandelisz.github.io/tuiwright/snapshots/cell-grid/.
- Coverage of Textual code via run_test (MEDIUM, multi-source): DeepWiki Textual testing pages (in-process HeadlessDriver); https://github.com/curdriceaurora/Local-File-Organizer/issues/573 (~44%→90% via run_test); VMx notes_showcase pyproject (modal/view coverage exclusion as the anti-pattern).
- YAML comment preservation (MEDIUM, multi-source incl. official): PyYAML drops comments by design — https://docs.spec-kitty.ai/configuration/yaml-libraries.html, https://stackoverflow.com/q/60891175; ruamel round-trip semantics + caveats (key deletion vs comments, dict.update, C parser) — https://yaml.dev/doc/ruamel.yaml/api/, https://stackoverflow.com/q/72732098, https://deepwiki.com/commx/ruamel-yaml/3.1-round-trip-processing.
- ModelScope (LOW-MEDIUM; SDK internals + issue tracker, no official rate-limit doc): cache-layout change v1.23.0 + MODELSCOPE_CACHE inconsistency — https://github.com/modelscope/modelscope/issues/845 and changelog; existence-check patterns (`HubApi().get_model` try/except, `check_model_is_id`) — https://gitcode.com/GitHub_Trending/mo/modelscope/blob/f9dd6794c40ae9062b5c3f9f9463b0675ce1ce19/modelscope/hub/check_model.py; `repo_exists` in newer modelscope-hub — https://pypi.org/project/modelscope-hub/0.1.8/.
- HF model_info 429 burst under transformers≥5 (MEDIUM): https://github.com/hiyouga/LlamaFactory/pull/10155.
- Windows terminal issues (MEDIUM): textual#4974 (conhost symbols), textual#5456 (conhost IME), msys2/MINGW-packages#22188 (mintty mouse), Textualize/rich#140 (WT_SESSION detection) — all via GitHub.
- LOW confidence, deliberately not relied upon: ModelScope API rate-limit numbers (undocumented); HF_ENDPOINT runtime-toggle inertness in the installed huggingface_hub version (import-time reading is documented; the specific in-process effect at this repo's pinned versions needs the Phase-C spike).

---
*Pitfalls research for: DNALLM v1.4 — Pre-flight UX & Registry-as-Truth-Source*
*Researched: 2026-10-20*
