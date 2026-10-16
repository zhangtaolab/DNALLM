---
phase: 13-ollama-surface-disposition
plan: 01
subsystem: testing
tags: [ci, github-actions, pytest, typed-skips, census-gate, ollama, example-nightly]

requires:
  - phase: 09-ci-wiring-census-verification
    provides: the example-nightly staged-serial job, D-03 census hard gate, D-13 hygiene floors, stage-4 junit audit + fail-soft summary
  - phase: 08-full-execution-rollout-repair-loop
    provides: the probe-then-execute gate (_gate_ollama_stack, network-unavailable: typed skip) and the ollama systemd unit + runner README
provides:
  - Deleted example-nightly stage-3 ollama batch (D-04) with a dated deletion-marker naming the restore path
  - The mcp_example pair typed-skipping in the stage-1 main batch via the existing self-healing D-13 gate (executes where the stack is up)
  - Census triple re-pinned 216/217 tests collected (1 deselected) under the -m "not giants"-only selector (stages 0.5 and 1 identical)
  - Comment/comment-contract reconciliation: ci.yml header + D-12 actuals + stage-2.5 framing, test module D-07 comment, expected_skips.yaml entry comment, both READMEs
  - The verbatim 56-line sanctioned residue record (OLLM-03 oracle, reproduced below)
affects: [14-gap-1-windows-disposition, 19-docs-accuracy-sweep, runner-ops, example-nightly]

actuals:
  tokens: 5521        # chars/4 over the realized production diff (5 files, +74/-83); plan estimate was 38000
  tasks: 3
  commits: 2          # MEASURED: git rev-list --count b67e772..3fcfef7 (production; the docs commit follows)

plan_head_before: b67e7721b5f6c83e75ae6e1bbc98d7fa184f05be
plan_head_after: 3fcfef74b313438fcd86ccd782818707b2851719

tech-stack:
  added: []            # deletion + re-pin + documentation phase; zero new dependencies or tools
  patterns:
    - "Disposition via CI shape: withdraw the environment, let the existing probe-then-execute gate fire its already-allowlisted typed skip (D-04)"
    - "Deletion-marker comment discipline (6218948 pattern): decision code + date + rationale + restore path left at the deleted step's site"
    - "Census re-pin discipline (c99fa9d pattern): measure with the exact new flags, change only the two triple-bearing lines, record the measured triple in the commit message"

key-files:
  created:
    - .planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md
  modified:
    - .github/workflows/ci.yml
    - tests/examples/test_notebook_execution.py
    - tests/expected_skips.yaml
    - .github/workflows/README.md
    - scripts/runner/README.md

key-decisions:
  - "Skip realized by CI shape (stage-3 deletion + existing D-13 gate), never by unconditional skip marks — zero functional edits to any test file (D-01/D-04)"
  - "Census re-pinned to the measured 216/217 (1 deselected) with the deselect dropped from BOTH stages 0.5 and 1; grep pattern never widened"
  - "Stage-2.5 keeps the literal 'Stage 2.5' name prefix (the _load_sse_probe_block consumer) and is reframed as end-of-run hygiene before the stage-4 audits"
  - "expected_skips.yaml touched comment-only; the pre-existing 'network-unavailable:' prefix entry is the sole sanctioned match path (no matcher changes)"
  - "Runner-side green evidence deferred to the next 05:30 UTC example-nightly (or owner dispatch) — no auto-dispatch, since a ci.yml workflow_dispatch would co-enqueue the dispatch-only feas-spike job on the self-hosted box"

requirements-completed: [OLLM-01, OLLM-02, OLLM-03]

coverage:
  - id: D1
    description: "Stage-3 ollama batch deleted with D-04 marker; the mcp_example pair typed-skips via the existing gate with live probe evidence (skip-not-remove disposition realized by CI shape)"
    requirement: OLLM-01
    verification:
      - kind: integration
        ref: "pytest tests/examples/test_notebook_execution.py -k 'mcp_example or TestGateMatrix' -> 4 passed, 2 skipped (network-unavailable: with full 30-attempt probe log)"
        status: pass
      - kind: unit
        ref: "TestGateMatrix (4 tests: both-up plain return; server-down/ollama-down/both-down typed skips)"
        status: pass
      - kind: other
        ref: "scripts/audit_skips.py over the pair junit -> 'OK: every skip ... matches the allowlist', exit 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "Census triple re-pinned and reproducing under the new selector, twice identically; stage-0.5/stage-1 selector parity; stage-4 audit list has exactly 4 junit members"
    requirement: OLLM-02
    verification:
      - kind: other
        ref: "collect-only -m 'not giants' -> 216/217 tests collected (1 deselected), twice identically (normalized per the stage-0.5 gate's own sed)"
        status: pass
      - kind: other
        ref: "grep checks: NO-DESELECT, STAGE3-GONE, 'Stage 2.5' literal present, D-19 cron gate unchanged, 4-member junit list"
        status: pass
    human_judgment: false
  - id: D3
    description: "Post-sweep ollama residue equals exactly the sanctioned 56-line list (5/30/5/1/15 per file; zero hits outside the 5 swept files), recorded verbatim below"
    requirement: OLLM-03
    verification:
      - kind: other
        ref: "grep -rn ollama tests/ | wc -l = 56; grep -rln ollama tests/ = exactly the 5 swept files (RESIDUE-56 composite check)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Allowlist comment refreshed comment-only; both READMEs accurate for the deleted machinery with every committed-content pin literal intact; full fast lane green"
    requirement: OLLM-02
    verification:
      - kind: unit
        ref: "pytest tests/test_runner_infra_contracts.py tests/test_extras_guard.py -q -> 13 passed (README pins, unit pins, model-swap pins, extras bracket)"
        status: pass
      - kind: other
        ref: "matcher-diff gate: exact/prefix/reason_like set byte-identical to HEAD; yaml diff comment-lines-only"
        status: pass
      - kind: integration
        ref: "full fast lane: pytest -m 'not slow' --no-cov -q -> 2487 passed, 2 skipped, 0 failed in 105.86s"
        status: pass
    human_judgment: false
  - id: D5
    description: "First post-merge example-nightly run comes green (stage-1 pair typed-skips on the runner, census gate passes, 4-member audit, zero ledger lines)"
    requirement: OLLM-02
    verification: []
    human_judgment: true
    rationale: "Runner-side nightly execution cannot be observed from this executor (self-hosted box, schedule 05:30 UTC); the executor must not auto-dispatch a ci.yml workflow (it would co-enqueue the dispatch-only feas-spike job). Standing evidence: the local end-to-end proof under exact post-change conditions and coverage-nightly already exercising the same typed skip green."

duration: 17 min
completed: 2026-10-11
status: complete
---

# Phase 13 Plan 01: Ollama Surface Disposition Summary

**Deleted the example-nightly stage-3 ollama batch so the mcp_example pair typed-skips via the existing probe-then-execute gate (2 typed skips + audit exit 0 proven locally); census re-pinned 216/217 (1 deselected); residue exactly the sanctioned 56 with zero functional test-file edits.**

## Performance

- **Duration:** 17 min (task execution window)
- **Started:** 2026-10-11T08:45:47Z
- **Completed:** 2026-10-11T09:02:46Z
- **Tasks:** 3/3
- **Files modified:** 5 (ci.yml, test_notebook_execution.py comment block, expected_skips.yaml comment, workflows README, runner README)

## Accomplishments

- The D-04 disposition landed as ONE atomic CI-shape change: stage 3 deleted with a dated marker, the `-k "not mcp_example"` deselect dropped from stages 0.5 and 1, the census triple re-pinned to the measured `216/217 tests collected (1 deselected)`, the stage-4 audit list cut to 4 junit members, and stage-2.5 reframed as pre-audit hygiene (literal `Stage 2.5` prefix kept for `_load_sse_probe_block`)
- The typed-skip chain proven end-to-end locally under exact post-change conditions: gated pair → `network-unavailable:` messages carrying the full 30-attempt probe log → allowlist prefix match → `audit_skips.py` OK exit 0 (117s for the pair, refused-connection case); TestGateMatrix 4/4 green as the deterministic skip-path proof
- Zero functional edits to any test file: the only test-file change is the rewritten D-07 stage-contract comment block (3 ollama-mention lines), with the gate, matrix tests, GATED_NOTEBOOKS rows, and timeout rows byte-identical
- Neighbor contracts all green: full fast lane 2487 passed / 2 skipped (the pair) / 0 failed; runner-infra + extras guard pins 13 passed; both mcp_example notebooks and their docs mirrors untouched (D-03 freeze held)

## Task Commits

1. **Task 1: End-to-end disposition — delete nightly stage 3, let the existing gate skip the mcp pair typed, re-pin the census** — `ac4115f` (ci)
2. **Task 2: Accuracy ride-along — skip-allowlist comment refresh + both README fixes with pin literals intact** — `3fcfef7` (docs; amended once for the residue-safe comment wording — see Deviations)
3. **Task 3: Disposition proof — residue equality, census idempotency, fast-leg regression, nightly evidence record** — this SUMMARY (no code changes; the proof artifacts live here)

**Tracer feedback gate (Task 1, auto mode active):** verify re-run end-to-end against the committed state — 5/5 checks green (CENSUS-MATCH, NO-DESELECT, STAGE3-GONE, 8 pin tests passed, pair 2 typed skips + audit exit 0) — expanded to Tasks 2-3 without a checkpoint.

## Measured Census Evidence (OLLM-02)

Stage-0.5-shaped command, run twice after the change (full raw lines; the timing suffix is run-dependent, the triple is not):

```
run 1: =============== 216/217 tests collected (1 deselected) in 0.89s ================
run 2: =============== 216/217 tests collected (1 deselected) in 0.82s ================
normalized (the stage-0.5 gate's own sed): 216/217 tests collected (1 deselected)  — identical both runs
```

Selector parity: the stage-0.5 collect command and the stage-1 pytest invocation both carry exactly `-m "not giants"` and no `-k` argument (ci.yml lines 863-864 and 891-892 post-change).

## OLLM-03 Sanctioned Residue Record (verbatim, 56 lines)

`grep -rn ollama` over the 5 swept files in the fixed ledger order — tree-wide count is exactly 56, zero hits outside these files. Re-running the grep reproduces this byte-for-byte:

```
tests/examples/_execution.py:245:    str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_langchain_agents.ipynb"): {
tests/examples/_execution.py:246:        # Agent-loop budget: the qwen3.5:4b ollama turns plus real MCP tool
tests/examples/_execution.py:252:        # live ollama service was NOT reconfigured, so the in-repo unit's
tests/examples/_execution.py:276:    str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_pydantic_ai.ipynb"): {
tests/examples/_execution.py:1237:    gated layer for the ollama/mcp-server example notebooks: the skip
tests/examples/test_notebook_execution.py:684:    """Execute-state gate semantics for the ollama/mcp stack (D-08, 261003-csd).
tests/examples/test_notebook_execution.py:688:    (ollama/VRAM coexistence sanctioned).  Any endpoint down stays an
tests/examples/test_notebook_execution.py:693:    OLLAMA_UP = (True, "ollama-fake-evidence")
tests/examples/test_notebook_execution.py:694:    OLLAMA_DOWN = (False, "ollama-down-evidence")
tests/examples/test_notebook_execution.py:701:        ollama: tuple[bool, str],
tests/examples/test_notebook_execution.py:707:            return ollama if url == OLLAMA_URL else server
tests/examples/test_notebook_execution.py:713:        # The ollama leg probes through the D-13 retry window (08-08); a
tests/examples/test_notebook_execution.py:720:        assert _gate_ollama_stack("nb.ipynb") is None
tests/examples/test_notebook_execution.py:723:        ("ollama", "server"),
tests/examples/test_notebook_execution.py:725:        ids=["server-down", "ollama-down", "both-down"],
tests/examples/test_notebook_execution.py:730:        ollama: tuple[bool, str],
tests/examples/test_notebook_execution.py:734:        self._patch_probes(monkeypatch, ollama, server)
tests/examples/test_notebook_execution.py:736:            _gate_ollama_stack("nb.ipynb")
tests/examples/test_notebook_execution.py:739:        assert "ollama-fake-evidence" in message or "ollama-down-evidence" in message, message
tests/examples/test_notebook_execution.py:758:    LANGCHAIN_SPEC = str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_langchain_agents.ipynb")
tests/examples/test_notebook_execution.py:759:    PYDANTIC_SPEC = str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_pydantic_ai.ipynb")
tests/examples/test_notebook_execution.py:1266:    at the defaults) -- a runner-fresh ollama service may still be loading
tests/examples/test_notebook_execution.py:1293:def _gate_ollama_stack(nb_name: str) -> None:
tests/examples/test_notebook_execution.py:1296:    The notebooks talk to ollama at :11434 AND the dnallm MCP server at
tests/examples/test_notebook_execution.py:1301:    retired the never-auto-execute sentinel; ollama/VRAM coexistence is
tests/examples/test_notebook_execution.py:1305:    D-13 (08-08): the ollama leg probes through the ~60s retry window --
tests/examples/test_notebook_execution.py:1312:    ollama_ok, ollama_ev = _probe_http_with_retry(OLLAMA_URL)
tests/examples/test_notebook_execution.py:1314:    if ollama_ok and server_ok:
tests/examples/test_notebook_execution.py:1317:        f"ollama probe {OLLAMA_URL}: {'GREEN' if ollama_ok else 'down'} ({ollama_ev}); "
tests/examples/test_notebook_execution.py:1324:    if not ollama_ok:
tests/examples/test_notebook_execution.py:1325:        missing.append(f"ollama {OLLAMA_URL} unreachable")
tests/examples/test_notebook_execution.py:1427:    ("mcp_example/mcp_client_ollama_langchain_agents.ipynb", _gate_ollama_stack),
tests/examples/test_notebook_execution.py:1428:    ("mcp_example/mcp_client_ollama_pydantic_ai.ipynb", _gate_ollama_stack),
tests/examples/test_notebook_execution.py:1460:    "mcp_example/mcp_client_ollama_langchain_agents.ipynb",
tests/examples/test_notebook_execution.py:1461:    "mcp_example/mcp_client_ollama_pydantic_ai.ipynb",
tests/test_extras_guard.py:4:``langchain-ollama`` inside its throwaway venv, and the langchain mcp
tests/test_extras_guard.py:77:    def test_langchain_ollama_declared_in_mcp_extra(self) -> None:
tests/test_extras_guard.py:78:        """langchain-ollama>=1.1.0 is an exact member of the mcp extra."""
tests/test_extras_guard.py:80:        assert "langchain-ollama>=1.1.0" in mcp, (
tests/test_extras_guard.py:81:            f"langchain-ollama>=1.1.0 missing from the mcp extra (REPAIR-04): {mcp}"
tests/test_models_lock_contracts.py:53:ollama models only).
tests/test_runner_infra_contracts.py:3:Pins the in-repo ollama systemd unit's TWO Environment lines and the
tests/test_runner_infra_contracts.py:39:UNIT_FILE = REPO_ROOT / "scripts" / "runner" / "ollama.service"
tests/test_runner_infra_contracts.py:45:    """Return the in-repo ollama unit file text (the auditable source)."""
tests/test_runner_infra_contracts.py:68:    """D-06/D-12 Environment-line pins over scripts/runner/ollama.service."""
tests/test_runner_infra_contracts.py:73:            "D-06 runtime cut missing: scripts/runner/ollama.service must pin "
tests/test_runner_infra_contracts.py:82:            "D-12 access control missing: scripts/runner/ollama.service must keep "
tests/test_runner_infra_contracts.py:101:        assert "restart ollama" in text, (
tests/test_runner_infra_contracts.py:103:            "(sudo systemctl restart ollama) in the re-apply op"
tests/test_runner_infra_contracts.py:156:PYDANTIC_NB = "mcp_client_ollama_pydantic_ai.ipynb"
tests/test_runner_infra_contracts.py:157:LANGCHAIN_NB = "mcp_client_ollama_langchain_agents.ipynb"
tests/test_runner_infra_contracts.py:197:        assert any('"ollama:qwen3.5:4b"' in src for src in langchain_sources), (
tests/test_runner_infra_contracts.py:198:            f'{LANGCHAIN_NB} must carry "ollama:{NEW_MODEL}" in a code cell '
tests/test_runner_infra_contracts.py:221:            (DOCS_LANGCHAIN_MD, '"ollama:qwen3.5:4b"'),
tests/test_runner_infra_contracts.py:237:        assert "ollama pull qwen3.5:4b" in text, (
tests/test_runner_infra_contracts.py:239:            f"`ollama pull {NEW_MODEL}` ({SWAP_DECISION}) -- a stale pull "
```

## Ledger Reconciliation (OLLM-01)

**59 = 56 keep-verbatim + 3 comment-rewritten + 0 deleted-from-test-files.**

- 56 keep-verbatim: Group K (non-ollama contracts, 21) + Group S (ollama-execution surface / the skip mechanism itself, 35) — byte-identical to the plan's ledger
- 3 comment-rewritten (Group C): the old test_notebook_execution.py comment lines 1226/1227/1229 describing the deleted stage-3 machinery — replaced by the post-change D-07 contract prose (which deliberately avoids the lowercase token `ollama`)
- 0 deleted from test files — every deletion (Group X) lives in ci.yml only: the stage-3 step + banner, the stage3 junit audit-list member, the `-k "not mcp_example"` at both invocation sites, and the old triple
- Every disposition cites D-01 (skip, not remove), D-02 (only ollama-targeted machinery touched; every K contract preserved and green), D-03 (content frozen — notebooks, mirrors, `langchain-ollama>=1.1.0` in the mcp extra untouched), D-04 (mechanism = CI shape, not marks)

## Nightly Evidence Note (do NOT auto-dispatch)

Runner-side green evidence lands at the **next 05:30 UTC scheduled example-nightly run after merge** (or an owner dispatch). The executor did NOT dispatch the workflow: a ci.yml `workflow_dispatch` would also enqueue the dispatch-only `feas-spike` job on the self-hosted box (research A3). Standing evidence in lieu:

1. The local end-to-end proof under exact post-change conditions (Task 1): the pair typed-skips with live probe evidence and `audit_skips.py` exits 0
2. coverage-nightly already runs the full suite with no deselect and exercises this exact typed skip green today
3. A red first nightly falsifies OLLM-02 and must be triaged from the uploaded complete failure scene (D-14) before phase close — recorded as an `unrun-verify` entry (id 22) in `.planning/WINDOWS.md`

**D-05 owner-ops note:** stopping/disabling the runner host's ollama systemd service is an owner action outside the repo; the committed unit file `scripts/runner/ollama.service` is the restore path and stays pinned green (TestOllamaUnitPins). The skip fires via the MCP-server probe leg regardless (research A2).

## Files Created/Modified

- `.github/workflows/ci.yml` — stage 3 deleted (D-04 marker at the site), deselect dropped from stages 0.5/1, triple re-pinned on both gate lines, stage-4 list at 4 junits, stage-2.5 reframed, header + D-12 comments reconciled
- `tests/examples/test_notebook_execution.py` — D-07 stage-contract comment block rewritten to the post-change contract (only change in the file)
- `tests/expected_skips.yaml` — comment-only refresh on the `network-unavailable:` entry (matchers byte-identical)
- `.github/workflows/README.md` — Runner paragraph D-04 disposition, stage table rows 0.5/1/2.5 reworked + row 3 deleted (gap kept), deletion prose with restore path
- `scripts/runner/README.md` — num_ctx section stops claiming the nightly loads the model; all pin literals intact

## Decisions Made

- Recorded in the plan (D-01..D-05) and followed exactly; execution-time choices were measurement-driven only (the census triple pinned is the freshly measured value, matching the plan's expectation)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocker] expected_skips.yaml comment wording introduced a 57th residue line**
- **Found during:** Task 3 (residue equality check)
- **Issue:** The Task 2 comment refresh paraphrased the deletion as "with the stage-3 ollama batch deleted", adding one lowercase `ollama` token inside `tests/expected_skips.yaml` — tree-wide residue measured 57 (not 56) with a 6th file in the grep surface, failing the OLLM-03 gate
- **Fix:** Reworded to the plan's own sanctioned phrasing ("once no stage brings the stack up for it") — no `ollama` token; Task 2 commit amended (3fcfef7) so the atomic change stays one commit
- **Files modified:** tests/expected_skips.yaml
- **Verification:** tree-wide count back to exactly 56 across exactly the 5 swept files; matchers still byte-identical; comment still names Phase 13, D-04 and 2026-10-11; audit re-run OK
- **Committed in:** 3fcfef7 (amend of 53229c3)

**2. [Verify-mechanics note] Census idempotency check normalized per the stage-0.5 gate**
- **Found during:** Task 3
- **Issue:** The plan's idempotency verify compares raw `tail -1` lines, but pytest appends a run-dependent timing suffix (`in 0.89s` vs `in 0.82s`), so full-line equality can never hold — the truth being tested is the identical TRIPLE
- **Fix:** Compared the lines normalized with the exact sed the stage-0.5 CI gate itself uses — identical triple both runs (recorded above). No repo change.
- **Verification:** IDEMPOTENT: 216/217 tests collected (1 deselected)

**3. [Placement note] Workflows-README deletion prose lives in the topology paragraph, not at line 144**
- **Found during:** Task 2
- **Issue:** The plan suggested adding the D-04 deletion sentence "in the style used at line 144" (the Caches paragraph); placing a stage-topology deletion record inside the Caches paragraph would have been off-topic
- **Fix:** Same sentence content and decision-coded style, placed in the staged-serial topology intro directly above the stage table (where the stage-3 row was dropped); names `network-unavailable:` and the restore path as required
- **Verification:** LITERALS-OK composite grep (no row numbered 3, prefix named) passes

**4. [Precedent note] Plain-literal count of the re-pinned triple is 1, not 2**
- The acceptance criterion asks the literal to appear "on exactly 2 lines" — the grep-pattern line carries regex-escaped parens (`\(1 deselected\)`) as in the c99fa9d precedent, so a plain substring count sees only the FAIL message. Both triple-bearing lines (867 grep pattern, 868 FAIL message) are re-pinned; substance met.

---

**Total deviations:** 1 auto-fixed (Rule 3) + 3 verify/placement notes.
**Impact on plan:** All fixes necessary for the OLLM-03 residue gate to hold exactly; no scope creep — the amend kept Task 2 atomic and nothing outside the plan's file set was touched.

## Issues Encountered

None beyond the deviations above. Full fast lane 2487 passed / 2 skipped / 0 failed; all pin suites green; YAML validated parseable after surgery.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None — no code was added; the phase is deletion + re-pin + documentation. (WINDOWS.md entry 22 records the deferred nightly-green human-check as `unrun-verify`; entry 13 from Phase 05 remains open for owner/verifier adjudication with the Phase-8 execution evidence now on record.)

## Threat Flags

None — no security-relevant surface beyond the plan's threat model. The change REMOVES a network-listener start from CI (T-13-03 net surface reduction); the loopback pin, audit loop, and fail-soft summary are untouched (T-13-01/02 mitigations applied as planned; T-13-04 enforced by the matcher-diff gate).

## Next Phase Readiness

- Phase 13 complete (1/1 plan). Ready for the phase verifier; the sole human-check is the first post-merge example-nightly run (see Nightly Evidence Note)
- Non-ollama surfaces intact for later phases: Phase 16/17 research flags, Phase 18 extras surgery, Phase 19 docs sweep (which owns the known-stale `docs/user_guide/continuous_integration.md:17` ollama-pair sentence, deliberately untouched here)

---
*Phase: 13-ollama-surface-disposition*
*Completed: 2026-10-11*

## Self-Check: PASSED

- All 6 created/modified files exist on disk (5 production + this SUMMARY)
- Both task commits (ac4115f, 3fcfef7) are ancestors of HEAD
- The residue block reproduces byte-for-byte against a live `grep -Hn ollama` over the 5 swept files in ledger order (diff clean, 56 lines) — 4 indentation typos in the first transcription were caught by this check and fixed before commit

