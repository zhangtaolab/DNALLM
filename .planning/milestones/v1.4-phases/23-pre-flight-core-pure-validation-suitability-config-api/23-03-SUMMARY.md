---
phase: 23-pre-flight-core-pure-validation-suitability-config-api
plan: "03"
subsystem: testing
tags: [availability-probe, huggingface-hub, modelscope, cache-detection, bounded-wait, preflight, tdd]

# Dependency graph
requires:
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api
    provides: 23-01 report.py type inventory (AvailabilityReport/State/Hub/ReasonCode) + structure guards that pre-authorize availability.py as the hub-client module
  - phase: 22-registry-foundation-runtime-loader-enrichment
    provides: registry.try_resolve / is_lab_model (the D-11 source proxy)
provides:
  - probe_availability — the 23-02 prober seam's default implementation (cached / remote_confirmed / unknown, size+revision, never raises)
  - effective_endpoints — library-constant endpoint/cache-root reporting for Phase 25's env panel
  - modelscope_cache_candidates — dual-layout (current + legacy, dots->___) pure-isdir resolver
  - The exhaustive failure taxonomy (RATE_LIMITED / OFFLINE / PROBE_TIMEOUT / NOT_FOUND_ON_EITHER_HUB / PROBE_ERROR) pinned by network-free tests
affects: [23-05 facade + check_compatibility wiring, 25 TUI env panel + select-to-download, 26 MCP availability findings]

# Actuals (#2632) — same scale as the plan estimate (chars/4 over the realized diff)
actuals:
  tokens: 14005
  tasks: 3
  commits: 5

tech-stack:
  added: []  # zero new dependencies (huggingface_hub/modelscope already core)
  patterns:
    - "Compute-under-lock memo: probe, resolve, and report-building all run inside one threading.Lock — at-most-once semantics without per-key in-flight machinery"
    - "Core-owned time bound: hub calls run on a shared ThreadPoolExecutor with result(timeout); abandoned calls finish naturally"
    - "Classifier-in-flow adapters: adapters raise raw; hub-specific exception classes are classified one level up so fakes can raise real exception instances"
    - "Lazy-module patching: modelscope's lazy package attribute access raises — bind submodules via importlib.import_module for test patching"

key-files:
  created:
    - dnallm/preflight/availability.py
    - tests/preflight/test_availability.py
  modified: []

key-decisions:
  - "D-11's source-first hub order is implemented via the is_lab_model() proxy — RegistryEntry has no source field; ModelScope is probed first for EVERY entry and source_hub reports modelscope when both hubs carry the repo (flagged for Phase 26 review)"
  - "Local cache detection checks ModelScope before HF (mirroring the remote order) so a repo cached on both hubs reports modelscope"
  - "Timeouts are terminal: a bounded-wait timeout ends the probe as unknown/PROBE_TIMEOUT immediately, while a raising adapter falls through to the other hub (a failing hub never short-circuits the probe)"
  - "HF cache lookups are best-effort misses: try_to_load_from_cache raises HFValidationError on path-like ids — a cache-lookup failure is a miss, never a probe failure"
  - "The injected prober runs under the same bounded wait and never-raise wrapper as the default adapters (PFC-06 holds for the seam too)"
  - "except OSError in the du-size walk stays uncovered (2 lines, 99% total): covering it needs a racing file deletion, and the dangling-symlink route would add a Windows-hostile test plus a new unmatched skip"

requirements-completed: [PFC-03, PFC-06]

coverage:
  - id: D1
    description: "Offline half of probe_availability: three-way local cache detection (local-path isdir -> cached/local; ModelScope dual-layout isdir with exact du size; HF try_to_load_from_cache + scan_cache_dir with size_on_disk and snapshot revision), lock-guarded memo with alias keys, clear_probe_memo hook"
    requirement: PFC-03
    verification:
      - kind: unit
        ref: "tests/preflight/test_availability.py#local path detection, HF cache snapshot/sentinel/scan-failure, MS dual-layout current+legacy, memo second-probe/alias/clear/8-thread-concurrency"
        status: pass
    human_judgment: false
  - id: D2
    description: "effective_endpoints() reporting huggingface_hub.constants.ENDPOINT (never os.environ, proven against a lying HF_ENDPOINT) and the ModelScope cache root via get_modelscope_cache_dir()"
    requirement: PFC-03
    verification:
      - kind: unit
        ref: "tests/preflight/test_availability.py#test_effective_endpoints_reports_library_constants"
        status: pass
      - kind: other
        ref: "command: python -c effective_endpoints equality check (plan Task 1 verify) — 'endpoints ok'"
        status: pass
    human_judgment: false
  - id: D3
    description: "Remote half: ModelScope-first fallback chain (MS confirms -> HF never called; clean miss -> HF fallback; both miss -> NOT_FOUND_ON_EITHER_HUB), exact int Size sums and Revision/sha metadata from fake hub clients, bounded wait firing PROBE_TIMEOUT within wall time on both hub branches and on the injected prober"
    requirement: PFC-03
    verification:
      - kind: unit
        ref: "tests/preflight/test_availability.py#remote chain order, adapter bodies, timeout trio, seam bypass/verbatim"
        status: pass
      - kind: other
        ref: "command: python -c seam contract check (plan Task 2 verify) — 'seam ok'; wired through check_compatibility(prober=probe_availability) returns a real availability state"
        status: pass
    human_judgment: false
  - id: D4
    description: "Failure taxonomy + degraded mode (PFC-06): 429 -> RATE_LIMITED, OfflineModeIsEnabled -> OFFLINE, plain ConnectionError -> PROBE_ERROR with class name, RepositoryNotFound/Gated = clean miss (never unknown), MS error falls through to HF and names its class when HF also misses; network severed -> cached local id still cached, uncached id unknown with reason, registry name resolves through try_resolve then degrades; probe_availability never raises"
    requirement: PFC-06
    verification:
      - kind: unit
        ref: "tests/preflight/test_availability.py#failure taxonomy section + degraded-mode section (raising adapters, hermetically neutralized caches)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Exactly one slow-marked real-network acceptance test probing the dual-homed lab model (30 s bound, well-formed report, positive int size when remote_confirmed) with the typed network-unavailable: skip prefix matching the existing expected_skips.yaml entry"
    verification:
      - kind: integration
        ref: "tests/preflight/test_availability.py#test_slow_real_network_dual_homed_lab_model — passed live on this box (network up); deselected by -m 'not slow' locally per the basic-tests-only directive, exercised by CI's slow leg"
        status: pass
    human_judgment: false

# Metrics
duration: 18 min
completed: 2026-10-20
status: complete
plan_head_before: b055490bea03e47467a0bd88c3b351db5752720b
plan_head_after: c9bc111dfb1cfaac94598473bb9a8add4acc9b61
commits: 5
---

# Phase 23 Plan 03: Availability Probe Summary

**ModelScope-first probe_availability with side-effect-free three-way cache detection, an exhaustive failure-to-UNKNOWN taxonomy behind a core-owned bounded wait, a lock-guarded memo, and library-constant endpoint reporting — 36 network-free unit tests + one typed-skip slow-lane acceptance, 99% module coverage**

## Performance

- **Duration:** 18 min
- **Started:** 2026-10-20T16:44:01Z
- **Completed:** 2026-10-20T17:02:51Z
- **Tasks:** 3/3
- **Files created:** 2 (1,285 insertions)

## TDD Gate Compliance

Tasks 1 and 2 carried `tdd="true"`; both follow the test → feat commit contract:

| Task | RED | GREEN | REFACTOR |
|------|-----|-------|----------|
| 1 (offline half) | e472f9b (ModuleNotFoundError on absent module — house precedent adeb4fd; 13 tests selected, all erroring) | b0ea6dc | not needed (code born clean, ruff-formatted) |
| 2 (remote half) | 3144449 (15 failing: 13 on absent adapter symbols, 2 assertion failures through the Task-1 stub; 15 prior green) | 3081a99 | not needed |
| 3 (tests only) | — | c9bc111 | — |

RED evidence note: this is a `type: execute` plan (not `type: tdd`), so the formal `tdd-red-evidence` classifier gate does not apply; the commit-scope contract (test → feat, RED fails, GREEN passes) was followed and is reflected above.

## Accomplishments

- `probe_availability(model, *, prober=None, timeout=5.0)`: memo-guarded resolution (raw alias + resolved repo-id keys, compute under one `threading.Lock` so concurrent probes of one id resolve and probe at most once — pinned by an 8-thread test), injectable prober seam under the same bounded wait, and an exhaustive never-raise wrapper (PFC-06)
- Three-way local cache detection with zero writes anywhere: local-path `isdir` → cached/local; ModelScope dual-layout candidates (`{root}/models/{org}/{name}` + legacy `{root}/{org}/{name}`, dots → `___`, root honoring `MODELSCOPE_CACHE`) with exact recursive-du sizes; HF `try_to_load_from_cache` + `scan_cache_dir` (size_on_disk, revision from the snapshot path) — never snapshot_download, so probing never litters a `.mdl`
- Remote chain, ModelScope first for every entry (D-11 via the is_lab_model proxy): MS confirmation returns without calling HF; exact int sums of MS `Size` fields / HF sibling sizes with `Revision`/`sha`; every failure degrades to unknown with a pinned distinct reason — RATE_LIMITED (429), OFFLINE (OfflineModeIsEnabled), PROBE_TIMEOUT (core-owned bounded wait on a shared executor), NOT_FOUND_ON_EITHER_HUB (both hubs cleanly missed; RepositoryNotFound/Gated on one hub means "try the other", never unknown), PROBE_ERROR with the exception class name
- Degraded-mode proofs (SC-5): with every remote adapter raising, a locally cached id still probes cached, an uncached id degrades to unknown with reason, and a registry entry name resolves through `try_resolve` then degrades identically
- `effective_endpoints()` reads `huggingface_hub.constants.ENDPOINT` (proven immune to a lying `HF_ENDPOINT` env var — Spike 1) and the ModelScope cache root; exactly one slow-marked real-network acceptance test shipped with the typed `network-unavailable:` skip prefix (existing expected_skips entry, unchanged file)
- Quality: preflight lane 145 passed + 1 deselected; registry lane 41 passed (sole-mentioner guard green — the module's `model_info` hit is the HF API method, not the registry data filename); ruff clean and formatted; availability.py coverage 99% (gate ≥96) network-free

## Task Commits

1. **Task 1 RED (offline tests)** — `e472f9b`
2. **Task 1 GREEN (offline implementation)** — `b0ea6dc`
3. **Task 2 RED (remote + taxonomy tests)** — `3144449` (amended once to fix an F821 alias bug caught by ruff before GREEN)
4. **Task 2 GREEN (remote implementation)** — `3081a99`
5. **Task 3 (degraded proofs + slow lane + coverage)** — `c9bc111`

**Plan metadata:** recorded in the completion report

## Files Created/Modified

- `dnallm/preflight/availability.py` — the probe module: public `probe_availability` / `effective_endpoints` / `clear_probe_memo` / `modelscope_cache_candidates`; internal resolution, three-way cache resolvers, MS/HF adapters, classifier, bounded wait, memo
- `tests/preflight/test_availability.py` — 37 tests: offline half, remote chain, failure taxonomy, degraded mode, one slow real-network acceptance

## Decisions Made

See key-decisions in frontmatter. Headline for Phase 26: **D-11's source-first hub order is implemented via the `is_lab_model()` proxy** — `RegistryEntry` has no `source` field (schema frozen, `extra="forbid"`), all `zhangtaolab/*` repos are dual-homed, and the owner directive makes ModelScope primary, so the chain probes ModelScope first for every entry and reports `source_hub=modelscope` when both hubs carry a repo. If a real `source` field is ever wanted, that is a registry-schema edit outside this phase.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two MS-cache test fixtures created directories for a different repo id than they probed**
- **Found during:** Task 1 (GREEN)
- **Issue:** the tests built `{tmp}/models/org/name___with___dots` but probed `org/definitely-not-registered` (no dots) — the candidates could never match
- **Fix:** fixtures now probe `org/name.with.dots` / `org/name`, matching the created layouts
- **Files modified:** tests/preflight/test_availability.py
- **Verification:** both MS cache tests green
- **Committed in:** b0ea6dc

**2. [Rule 1 - Bug] Memo alias test was environment-dependent (real hub caches on this box)**
- **Found during:** Task 1 (GREEN)
- **Issue:** probing the real BPE repo hit this machine's ModelScope cache (239 repos) and returned cached before reaching the patched remote seam — would behave differently in CI
- **Fix:** the alias test isolates `_probe_cached_local` to return None (unit isolation); the degraded-mode tests later neutralize caches hermetically via `MODELSCOPE_CACHE` + a patched `try_to_load_from_cache`
- **Files modified:** tests/preflight/test_availability.py
- **Verification:** alias test green and deterministic
- **Committed in:** b0ea6dc

**3. [Rule 1 - Bug] `try_to_load_from_cache` raises `HFValidationError` on path-like ids**
- **Found during:** Task 1 (GREEN)
- **Issue:** a nonexistent local path falls through to the HF cache lookup, whose `@validate_hf_hub_args` decorator raises on the malformed repo id — the exhaustive wrapper turned this into unknown/PROBE_ERROR instead of a clean miss (violating the "nonexistent local path → unknown via the remote path, never an exception" truth's spirit)
- **Fix:** the cache lookup is wrapped best-effort: a lookup failure is a miss, never a probe failure
- **Files modified:** dnallm/preflight/availability.py
- **Verification:** nonexistent-path test falls through to the remote sentinel
- **Committed in:** b0ea6dc

**4. [Rule 1 - Bug] Test-local `SimpleNamespace as NS` alias leaked across function scope (ruff F821)**
- **Found during:** Task 2 (RED commit, caught by ruff before GREEN)
- **Issue:** the alias was imported inside one test but used inside another — undefined name at run time
- **Fix:** module-level `SimpleNamespace` used directly; RED commit amended so no broken intermediate state exists
- **Files modified:** tests/preflight/test_availability.py
- **Verification:** ruff clean; RED re-verified 15 failed / 15 passed after amend
- **Committed in:** 3144449

**5. [Rule 3 - Blocking] modelscope's lazy package makes `modelscope.hub.api` attribute traversal raise**
- **Found during:** Task 2 (GREEN)
- **Issue:** `monkeypatch.setattr(modelscope.hub.api, ...)` resolves `modelscope.hub` through the lazy loader, which raises `AttributeError: Cannot import available module of hub` — patching the MS adapter body was impossible via attribute chains (the source-side `from modelscope.hub.api import HubApi` is safe: sys.modules + IMPORT_FROM fallback)
- **Fix:** the test binds the submodule once via `importlib.import_module("modelscope.hub.api")` and patches that object
- **Files modified:** tests/preflight/test_availability.py
- **Verification:** MS adapter-body test green; all 30 then 37 tests green
- **Committed in:** 3081a99

---

**Total deviations:** 5 auto-fixed (4 × Rule 1, 1 × Rule 3)
**Impact on plan:** All fixes were test-mechanics or defensive-cache-lookup issues; no scope creep, no contract changes — the plan's truths all hold at HEAD.

## Issues Encountered

None beyond the deviations above. One transient artifact (`.coverage-p2303` from the plan's own verify command, not matched by the existing `.coverage.*` gitignore pattern) was deleted after verification rather than committed or gitignored.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The prober seam is live end-to-end: `check_compatibility(..., prober=probe_availability)` verified working (23-05 wires it into the facade; the `Callable[[str], AvailabilityReport]` signature already matches)
- `effective_endpoints()` is ready for Phase 25's env panel (Spike-1-true reporting; mirror toggling is restart-required UX, documented in the module docstring)
- Module coverage 99% with the only uncovered lines being a racing-delete du guard — no omit-list entries, no new skips, expected_skips.yaml unchanged
- The slow-lane acceptance test passed live on this box (network up); CI's slow leg exercises it where network differs

## Self-Check: PASSED

- dnallm/preflight/availability.py — FOUND
- tests/preflight/test_availability.py — FOUND
- Commits e472f9b, b0ea6dc, 3144449, 3081a99, c9bc111 — all ancestors of HEAD (verified below)

---
*Phase: 23-pre-flight-core-pure-validation-suitability-config-api*
*Completed: 2026-10-20*
