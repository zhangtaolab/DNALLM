# Phase 13: Ollama Surface Disposition - Research

**Researched:** 2026-10-11
**Domain:** pytest skip-surface disposition + CI (GitHub Actions example-nightly) census/gate reconciliation — entirely in-repo mechanics; no new libraries
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Disposition Policy**

- **D-01: Tilt = skip, not remove.** Every ollama-targeted assertion carries a skip disposition with recorded rationale in the plan ledger — deletion is NOT the disposition for the live ollama surface. (Owner clarification, verbatim: "ollama test skip. 只针对 Ollama".) This resolves the research's open remove-vs-skip question in favor of skip everywhere the assertion is genuinely ollama-execution surface. — **Reversibility:** reversible — skips can be un-marked assertion by assertion later.
- **D-02: Only ollama-targeted refs are touched.** Non-ollama contracts sharing the swept files stay green and untouched: kernel-plot compat pins (`TestNotebookKernelPlotCompat`, ipython<9), extras brackets (`langchain-ollama>=1.1.0` in the mcp extra), docs-mirror content contracts (`"ollama:qwen3.5:4b"` in notebook + mirror), the models.lock docstring mention, and the gate-semantics matrix tests (they test gating logic via monkeypatch, not ollama). These kept-green refs form the sanctioned `grep ollama tests/` residue list recorded in the plan.
- **D-03: Ollama MCP content is frozen this milestone.** The two mcp_example notebooks, the `langchain-ollama` dependency, and the MCP server's ollama client capability are out of discussion — no edits, no removals, no re-gating of content.

**Skip Mechanism & Nightly Shape**

- **D-04: Skip lands via ci.yml stage-3 deletion + existing probe-then-execute gates (self-healing), NOT unconditional skip marks.** The example-nightly stage 3 (ollama batch that re-runs the two gated mcp tests with ollama + MCP server up) is deleted. With ollama no longer brought up on the nightly, `_gate_ollama_stack` fires its existing `environment-unavailable:` typed skip with live probe evidence — the `environment-unavailable:` prefix entry already exists in `tests/expected_skips.yaml`. On any box where ollama IS running (local dev), the tests still execute — self-healing by design, matching the research-recommended defensible state. — **Reversibility:** reversible — re-adding a stage-3-shaped batch restores nightly execution.
- **D-05: `scripts/runner/ollama.service` and its pin tests stay in-repo and green.** The `TestOllamaUnitPins` D-06/D-12 Environment-line assertions read the committed unit file only — no live ollama needed — and the unit file constitutes the explicit restore path. The GB10 host's systemd service itself may be stopped/disabled as an owner ops action (outside the repo; note it in the plan's ops section).

**Mechanical Consequences (locked by the above)**

- Census re-pin: the nightly P/S distribution shifts (the mcp pair moves from PASS in stage 3 to SKIP in the main batch); the D-03 census triple re-pins same-change; no pytest exit 5 (collection counts unchanged — the tests stay collected).
- ci.yml inline accuracy: stage-2.5 "pre-ollama" naming and the stage-3 block comments are corrected in the same change.
- README accuracy ride-along: `.github/workflows/README.md` and `scripts/runner/README.md` sentences describing stage 3 / the ollama loopback get minimal same-change accuracy fixes for deleted machinery; the full docs prose sweep stays in Phase 19 (DOCR-01).

### Claude's Discretion

- Per-assertion ledger format/numbering in the plan (assertion-level, not file-level — research Pitfall 5 warning sign).
- Exact census numbers after the re-pin (measured, not estimated).
- Whether the stage-2.5 hygiene step keeps a renamed role as end-of-run hygiene or its comment is simply reworded (kernel cleanup + memory floor is non-ollama discipline; do not delete the guard).
- Sanctioned residue-list formatting in the plan.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. (Full docs prose accuracy for runner/workflow READMEs beyond the minimal same-change fixes belongs to Phase 19's DOCR-01 sweep by design, not as a new deferral.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OLLM-01 | All ~59 ollama refs across the 5 inventoried test files dispositioned per-assertion as remove-or-skip with recorded rationale; non-ollama contracts sharing the surface preserved | Full per-ref inventory with line numbers and a three-group classification (keep / skip-via-gate / comment-accuracy) below; the skip needs ZERO test-file logic edits (the gate already emits it) |
| OLLM-02 | Census triple re-pinned, `expected_skips.yaml` updated, ci.yml example-nightly stages reconciled same-change — nightly stays green (no exit 5, no unexpected-skip audit failure) | Measured census triples (current reproduced, post-change measured); stage-by-stage edit map for ci.yml; end-to-end local proof: pair run under post-change conditions → 2 typed skips → `audit_skips.py` exit 0 |
| OLLM-03 | Post-sweep `grep ollama tests/` residue equals a sanctioned list recorded in the plan; the two mcp_example notebooks remain in-repo per WINDOWS #13 | Residue composition documented (which refs remain and why); D-03 freeze confirmed against pyproject mcp extra and notebook mirror contracts |
</phase_requirements>

## Summary

Phase 13 is an in-repo CI-shape and disposition phase with no new dependencies. The single load-bearing discovery of this research: **the D-04 skip mechanism already exists and is already green in CI — verified end-to-end this session.** `_gate_ollama_stack` probes `http://127.0.0.1:11434/api/tags` (30 attempts / ~60s window) and `http://localhost:8000/mcp`, and skips via `network_unavailable_skip` — `[VERIFIED: tests/examples/_execution.py:1230-1249]`, whose terminal line is verbatim `pytest.skip(f"network-unavailable: {action} ({evidence})")`. The prefix entry `- prefix: "network-unavailable:"` already exists in the allowlist `[VERIFIED: tests/expected_skips.yaml:29-30]`. coverage-nightly already runs the full suite with **no deselect** — the mcp pair already skips green there today (ci.yml's own comment documents this: `[VERIFIED: .github/workflows/ci.yml:575-580]`, "they skip deterministically as typed `\"network-unavailable:\"` skips"). I re-proved it locally under exact post-change conditions: the pair produced 2 skips carrying the full 30-attempt probe log, and `scripts/audit_skips.py` returned "OK: every skip ... matches the allowlist", exit 0, in 117s for both tests.

Two corrections to the CONTEXT's mechanical picture that the plan must carry: **(1) the typed prefix is `network-unavailable:`, not `environment-unavailable:`** — CONTEXT D-04 names the latter; the actual gate emits the former (both entries exist in expected_skips.yaml, so the mechanism works either way, but the plan ledger and any prose must name the right one). **(2) the mcp deselect is `-k "not mcp_example"`, which matches 8 tests, not 2** — the two gated execution tests plus 6 static checks (`test_notebook_json_validity`, `test_notebook_cell_syntax`, `test_notebook_imports` for both notebooks) `[VERIFIED: local collect-only this session: "8/217 tests collected (209 deselected)"]`. Removing the deselect therefore adds 6 always-green static tests AND the 2 skipping gated tests to stage 1, and moves the census triple from `208/217 tests collected (9 deselected)` (reproduced exactly this session) to **`216/217 tests collected (1 deselected)`** (measured this session with the exact stage-0.5 flags minus the `-k`). Three hard couplings in ci.yml must ride the same change or the nightly goes red: the stage-4 audit loop's junit list still names `pytest-junit-mcp-stage3.xml` (a missing junit records `=1` and the fail-soft summary greps it red — `[VERIFIED: .github/workflows/ci.yml:1054-1066]`); `_load_sse_probe_block` in `tests/test_runner_infra_contracts.py:54-63` slices ci.yml up to the literal `text.index("Stage 2.5", start)` — the step name must retain the `Stage 2.5` prefix or that test breaks; and stage 0.5's grep pattern + FAIL message carry the pinned triple.

**Primary recommendation:** one atomic change — delete ci.yml stage 3 (lines 1016-1048), drop `-k "not mcp_example"` from BOTH stage 0.5 (line 863) and stage 1 (line 884), re-pin the triple to the measured `216/217 tests collected (1 deselected)`, drop `pytest-junit-mcp-stage3.xml` from the stage-4 audit list, reword the stage-2.5 name suffix/comments while keeping the literal `Stage 2.5` prefix, apply the same-change comment/README accuracy fixes, and record the per-assertion ledger + sanctioned residue in the plan. **Zero functional edits to the 5 swept test files** — the disposition is realized entirely by CI shape; every test-file ollama ref either stays as-is (keep-contract or skip-machinery) or is a comment describing deleted machinery.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Skip disposition (mcp pair) | Test layer (`_gate_ollama_stack` probe-then-execute) | — | The gate IS the skip mechanism; D-04 forbids unconditional skip marks; self-heals where ollama+server exist |
| Nightly stage shape / stage-3 deletion | CI layer (`.github/workflows/ci.yml`) | — | Stage 3 is the only thing bringing ollama up on the nightly; deleting it is the disposition lever |
| Census integrity (D-03 triple) | CI layer (stage 0.5 hard gate) | Test layer (collection counts) | Pinned collect-only triple guards selector drift; re-pin is the designed bump-point |
| Skip honesty enforcement | CI layer (`scripts/audit_skips.py` + junit) | Test layer (expected_skips.yaml allowlist) | Fail-closed audit over junit skip messages; every skip must match an allowlist entry |
| Non-ollama contracts (extras/unit/docs-mirror pins) | Test layer (fast-leg tests) | — | Read committed files only (pyproject, unit file, notebooks, mirrors); no live ollama dependence |
| Runner-box ollama service state | Ops (outside repo) | — | Owner action (stop/disable); unit file stays in-repo as the restore path (D-05) |

## Standard Stack

### Core

No new packages. This phase installs nothing and adds no dependencies — it deletes a CI stage, re-pins gates, and edits comments/README prose. The entire "stack" is existing repo infrastructure:

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| pytest (repo venv) | installed in `.venv` | collect-only census measurement, gated-pair verification runs | Already the project's harness; `--collect-only -q` output line is the census artifact |
| `scripts/audit_skips.py` | in-repo | fail-closed junit skip audit | The enforcement gate OLLM-02 must satisfy; unit-tested by `tests/scripts/test_audit_skips.py` |
| `tests/expected_skips.yaml` | in-repo | skip allowlist (exact/prefix/reason_like matchers) | The typed-skip contract; `network-unavailable:` prefix already registered |
| GitHub Actions workflow syntax | existing ci.yml | stage surgery | YAML edit only; self-hosted runner untouched |

**Installation:** nothing — `npm`/`pip` installs are out of scope for this phase.

## Package Legitimacy Audit

> No external packages are installed, imported, or recommended by this phase. All edits target in-repo files (tests/, .github/workflows/ci.yml, READMEs).

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| — (none) | — | — | — | — | — | No gate applicable: zero packages |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

Post-change nightly data flow (the D-04 state):

```
schedule 05:30 UTC / workflow_dispatch
        │
        ▼
example-nightly job (self-hosted dnallm-nightly)
        │
        ├── stage 0    venv + extras ".[base,fla,dev,mcp]"  (mcp extra still needed: stage 2)
        ├── stage 0.5  collect-only census HARD gate
        │              selector: -m "not giants"            [deselect REMOVED]
        │              pinned triple: 216/217 collected (1 deselected)   [RE-PINNED]
        ├── stage 1    pytest tests/examples  (NO -k filter now)
        │              ├── 214 tests pass (incl. 6 static mcp_example checks, re-admitted)
        │              └── TestGatedNotebookExecution[mcp_example/*]
        │                      │ _gate_ollama_stack:
        │                      │   probe 127.0.0.1:11434/api/tags (30x, ~60s) → down
        │                      │   probe localhost:8000/mcp (1x) → down
        │                      │   both must be up to execute  [test_notebook_execution.py:1312-1315]
        │                      ▼
        │                  pytest.skip("network-unavailable: execute <nb> (…full probe log…)")
        │                  → junit <skipped message=…> (exit 0; collection unchanged → NO exit 5)
        ├── stage 1.5  hygiene HARD gate (kernel pkill + ≥35Gi floor)   [unchanged]
        ├── stage 2    MCP live-server probes :8000 (streamable + sse)  [unchanged; not ollama]
        ├── stage 2.5  hygiene HARD gate      [name keeps "Stage 2.5" literal; "pre-ollama" reworded]
        ├── (stage 3   DELETED — no server start, no ollama batch, no mcp-stage3 junit)
        └── stage 4    audit junit list: stage1, yaml, stage2×2   [stage3 entry REMOVED]
                       audit_skips.py per junit → "network-unavailable:" prefix match → OK
                       fail-soft summary: no '=[1-9]*$' ledger line → green

local dev box (ollama up + MCP server started manually)
        └── same gated tests: both probes GREEN → plain return → notebooks EXECUTE for real
            (self-healing; D-08-sanctioned ollama/VRAM coexistence)
```

### Recommended Project Structure

No new files. Touched files only:

```
.github/workflows/ci.yml            # stage 0.5 / 1 / 2.5 / 3 / 4 + header comments
.github/workflows/README.md         # stage table rows 0.5/1/2.5/3 + "Runner" paragraph sentence
scripts/runner/README.md            # minimal accuracy fixes (nightly-VRAM sentence)
tests/examples/test_notebook_execution.py  # comment block 1218-1232 only (stage-contract comment)
tests/expected_skips.yaml           # NO matcher change needed (optional comment note)
+ plan artifact: per-assertion ledger + sanctioned residue list
```

### Pattern 1: Disposition = CI shape, not test-file marks

**What:** The skip is realized by deleting the only CI machinery that satisfied the gate's probes. The gate, its tests, and its allowlist entry already exist.
**When to use:** Whenever a probe-then-execute gate's environment is withdrawn rather than the feature deleted.
**Why it satisfies OLLM-01:** the "skip disposition with recorded rationale" is the ledger entry naming the gate as mechanism; the "remove" arm of remove-or-skip applies to nothing in the test files under D-01 (removals are: ci.yml stage-3 block + stale comments).

### Pattern 2: Census re-pin (established bump-point discipline)

**What:** 2-line diff in the stage-0.5 hard gate, measured with the exact stage-0.5 flags, commit message records the measured triple.
**Precedent:** commit `c99fa9d` "fix(quick-261008-wfx): bump D-03 census triple 197/206 -> 208/217" — its message documents "Measured locally with the exact Stage 0.5 flags: 208/217 tests collected (9 deselected: 8 mcp + 1 giants)" [VERIFIED: git show c99fa9d this session]. That commit also independently confirms the 8-test deselect count.

### Pattern 3: Probe-then-execute with live evidence (v1.1 decision, preserved)

**What:** Skip messages carry verbatim probe results (every attempt), never bare reasons. "An ever-green skip is the same dishonesty class as a false-green gate" — but this skip is NOT ever-green: it fires only when a real probe really fails, carries the evidence, and self-heals.

### Anti-Patterns to Avoid

- **Adding `pytest.mark.skip`/`skipif` to the mcp pair** — D-04 explicitly rejects unconditional marks; the gate is the mechanism.
- **Deleting the gate or its matrix tests as "ollama surface"** — they are the skip mechanism and its regression proof (TestGateMatrix's `server-down`/`ollama-down`/`both-down` params are exactly the nightly's post-change states).
- **Marker-deselection instead of skip** (the `_GIANTS_GATED` shape) — rejected by owner policy for ollama: the runner environment class differs (giants: environment AVAILABLE, so a skip would lie; ollama post-change: environment genuinely withdrawn, so the typed skip is truthful). Keep the two mechanisms distinct in the ledger rationale.
- **Widening the census grep or adding an empty/wildcard allowlist entry** — both are the documented failure modes the gates exist to prevent.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Skip mechanism for the pair | New skip marks / new prefix (`removed-surface:` etc.) | Existing `_gate_ollama_stack` + `network-unavailable:` prefix | Already allowlisted, already proven green in CI (coverage-nightly), carries live evidence, self-heals |
| Census verification | Ad-hoc test counting | Stage-0.5 collect-only hard gate with re-pinned triple | The gate exists; re-pin is the designed bump-point (c99fa9d precedent) |
| Skip-honesty audit | New audit tooling | `scripts/audit_skips.py` + junit | Fail-closed, unit-tested, wired into 5 CI call sites |
| Residue enforcement | New lint script | Recorded sanctioned list in the plan + success-criterion grep at verification | OLLM-03's own definition; a script would be over-engineering for a one-time sweep |

**Key insight:** every mechanism this phase needs already shipped in v1.1 and is green in CI today. The phase is deletion + re-pin + documentation.

## Runtime State Inventory

> This phase changes CI/runtime shape (a stage that starts a server and depends on a host service is deleted) — inventory completed.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no databases or datastores reference the ollama test surface. Verified: the swept surface is pytest files + CI YAML only. | None |
| Live service config | The nightly workflow itself lives in git (ci.yml) — edited as code. The GB10 runner's LIVE ollama systemd unit (outside git, `scripts/runner/README.md` documents the 2026-10-05 drift to `0.0.0.0`) remains host state. | Owner ops note in plan: stopping/disabling the service is optional (skip fires via the server leg regardless); re-apply path stays documented (D-05) |
| OS-registered state | The runner host's `ollama.service` enabled-state and any actions-runner registration are outside the repo. On THIS dev box: `systemctl is-active ollama` → `inactive`, `is-enabled` → `not-found`, and `curl 127.0.0.1:11434/api/tags` → HTTP 000 [VERIFIED: local probe this session]. | Record as ops note; no repo change |
| Secrets/env vars | None — no secret or env-var name changes. `OLLAMA_HOST`/`OLLAMA_CONTEXT_LENGTH` live only inside the committed unit file, which stays. | None |
| Build artifacts | Nightly-run artifacts only: `pytest-junit-mcp-stage3.xml`, `mcp-server-stage3.log`, `stage3-mcp-pair.log` cease to be produced; the upload globs (`pytest-junit-*.xml`, `mcp-server-*.log`, `stage*.log`) absorb this without edit. | None (drop the stage3 junit from the stage-4 audit LIST — that is a code edit, covered in the edit map) |

**The canonical question answered:** after the repo edit, the only runtime system holding prior state is the runner host's ollama service itself — which the gate now probes and honestly reports down (or green, if left running and a server is up: then the pair executes, which is D-08-sanctioned).

## Common Pitfalls

### Pitfall 1: Stage-4 audit list orphans the deleted junit (nightly goes red on a green run)
**What goes wrong:** Deleting stage 3 removes `pytest-junit-mcp-stage3.xml` from disk. The stage-4 loop over junit filenames still lists it: `[VERIFIED: .github/workflows/ci.yml:1054-1056]` — verbatim list member `pytest-junit-mcp-stage3.xml`. The missing-file branch writes `stage4-audit-...=1` to `stage-results.txt` (lines 1057-1066), and the fail-soft summary's `grep -qE '=[1-9][0-9]*$'` (line 1095) fails the job.
**How to avoid:** Remove the stage3 junit from the audit list in the same edit. **Warning sign:** any nightly red whose stage-results ledger shows `stage4-audit-pytest-junit-mcp-stage3.xml=1 (junit missing)`.

### Pitfall 2: Renaming the stage-2.5 step breaks `TestExampleNightlySseProbe`
**What goes wrong:** `_load_sse_probe_block` slices ci.yml between the sse server-start line and the next occurrence of the literal step-name prefix: `[VERIFIED: tests/test_runner_infra_contracts.py:54-63]` — verbatim `end = text.index("Stage 2.5", start)`. If the hygiene step is renamed away from `Stage 2.5 …`, `str.index` raises `ValueError` and the fast-leg test fails.
**How to avoid:** Keep the `Stage 2.5:` name prefix; reword only the `(D-13, pre-ollama)` suffix and the comment body (the accuracy fix CONTEXT locks). Alternatively update the helper same-change — but keeping the prefix is the zero-risk minimal edit (and matches Claude's-discretion option "comment is simply reworded").

### Pitfall 3: Census selector and triple must move together, in BOTH places
**What goes wrong:** Re-pinning the triple without dropping `-k "not mcp_example"` from the stage-0.5 collect command pins a selector stage 1 no longer uses (the gate's own comment: "a selector set that omits the giants filter would guard nothing" — same logic for any drift between 0.5 and 1). Conversely dropping the `-k` from stage 1 only, without re-pinning, fails stage 0.5 first (hard gate, job-fatal).
**How to avoid:** One atomic change to lines 862-867 (selector + grep + FAIL message) and line 884. Measure with the exact new flags. **Warning sign:** "census collection mismatch (D-03)" at stage 0.5 — per repo doctrine, re-pin deliberately, never widen the grep.

### Pitfall 4: CONTEXT names the wrong skip prefix
**What goes wrong:** Plan prose or ledger says the nightly skip matches `environment-unavailable:`. The actual gate emits `network-unavailable:` `[VERIFIED: tests/examples/test_notebook_execution.py:1326-1329 calls network_unavailable_skip; _execution.py:1230-1249 pytest.skip(f"network-unavailable: {action} ({evidence})")]`; the gate-matrix test asserts it: line 738 verbatim `assert message.startswith("network-unavailable:")`. Both prefixes exist in the allowlist (lines 29-30 and 41-42), so the nightly is green either way — but a future auditor tracing D-04's wording would look for the wrong entry.
**How to avoid:** The plan ledger and README fixes name `network-unavailable:`. Optional: add a one-line comment beside the allowlist entry noting the example-nightly main batch now exercises it.

### Pitfall 5: The deselect is 8 tests, not 2 — stage 1 gains 6 static tests
**What goes wrong:** A plan assuming only the gated pair rejoins stage 1 under-counts the P/S shift and mis-pins the triple. Measured: `-k "mcp_example"` selects 8 — the 2 gated executions plus `test_notebook_json_validity` ×2, `test_notebook_cell_syntax` ×2, `test_notebook_imports` ×2 (static, fast, green on every fast leg today) [VERIFIED: local collect-only, ids listed this session].
**How to avoid:** Use the measured `216/217 tests collected (1 deselected)`; expect stage-1 junit to show +6 passes and +2 skips vs today; the D-12 measured-actuals comment (lines 657-658, "192P/1S/9 deselected") gets its numbers re-measured at execution time.

### Pitfall 6: The ~60s D-13 retry window now burns in stage 1
**What goes wrong:** With ollama down (connection-refused), each gated test spends ~60s in the retry window before skipping. Measured this session: the pair took **117s total** under post-change conditions. Worst case (dropped packets instead of refused: 30 × (5s timeout + 2s sleep)) approaches ~3.5 min per test.
**How to avoid:** Accept it (2 min against a 2700-min budget, and stage 3 previously cost ~87s+server-start anyway — net nightly time roughly neutral); note the measured figure in the ci.yml budget comment when updating the D-12 actuals. Do NOT shrink the retry window — it is the D-13 contract distinguishing a warming service from an absent one.

### Pitfall 7: Comment-contract drift (repo-specific)
**What goes wrong:** ci.yml's header comment block (lines 612-637) and the module comment in `tests/examples/test_notebook_execution.py:1218-1232` state "these module comments are the contract it [ci.yml] implements" — leaving stage-3 prose in either place after deletion makes the documented contract false. The repo treats these comments as auditable contract text (WINDOWS ledger discipline).
**How to avoid:** Same-change comment edits in both files. The historical run citations in `_execution.py:257,280` ("run 37406829738 stage 3 CellTimeoutError") are records of past runs — accurate history, safe to keep.

### Pitfall 8: Deletion creep into non-ollama contracts (the v1.3-research Pitfall 5)
**What goes wrong:** Keyword-classified removal deletes `TestOllamaUnitPins`, `TestMcpExampleModelSwap`, `EXPECTED_MCP_MEMBERS` neighborhood, or `TestNotebookKernelPlotCompat` — the guards for runner-box access control (loopback pin), mirror honesty, and the matplotlib/IPython regression.
**How to avoid:** D-02 already locks this; the ledger classification below makes each kept group explicit with line ranges. **Warning sign:** any diff touching assertions in `tests/test_runner_infra_contracts.py` or `tests/test_extras_guard.py`.

## Code Examples

### Census measurement (run before and after the edit)

```bash
# Current pinned selector — reproduces the pinned triple exactly
$ .venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"
=============== 208/217 tests collected (9 deselected) in 0.95s ================

# Post-change selector (stage 0.5 and stage 1 both drop -k "not mcp_example")
$ .venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"
=============== 216/217 tests collected (1 deselected) in 0.79s ================
```
[VERIFIED: both lines measured this session in the working tree at branch v1.3]

### Re-pin diff shape (precedent c99fa9d applied to this phase)

```diff
# .github/workflows/ci.yml, stage 0.5 (current lines 862-870)
-          .venv/bin/python -m pytest tests/examples --collect-only -q \
-            -m "not giants" -k "not mcp_example" 2>/dev/null | tee census-collect.txt | tail -1
+          .venv/bin/python -m pytest tests/examples --collect-only -q \
+            -m "not giants" 2>/dev/null | tee census-collect.txt | tail -1
...
-          if ! tail -1 census-collect.txt | grep -qE '^=* *208/217 tests collected \(9 deselected\) in '; then
-            echo "FAIL: census collection mismatch (D-03) — expected triple: 208/217 tests collected (9 deselected); measured line: $(tail -1 census-collect.txt)"
+          if ! tail -1 census-collect.txt | grep -qE '^=* *216/217 tests collected \(1 deselected\) in '; then
+            echo "FAIL: census collection mismatch (D-03) — expected triple: 216/217 tests collected (1 deselected); measured line: $(tail -1 census-collect.txt)"
```

### End-to-end post-change state proof (executed this session)

```bash
$ .venv/bin/python -m pytest tests/examples/test_notebook_execution.py -k "mcp_example" -q \
    --junitxml=/tmp/phase13-mcp-pair-junit.xml
================ 2 skipped, 70 deselected in 117.06s (0:01:57) =================

$ .venv/bin/python scripts/audit_skips.py /tmp/phase13-mcp-pair-junit.xml tests/expected_skips.yaml
Skip audit: 2 skipped test(s) in /tmp/phase13-mcp-pair-junit.xml
  allowed    …TestGatedNotebookExecution::test_gated_notebook_probes_then_executes[mcp_example/mcp_client_ollama_langchain_agents.ipynb] [network] -> 'network-unavailable: execute mcp_example/mcp_client_ollama_langchain_agents.ipynb (mcp client notebook) (dnallm MCP server http://localhost:8000/mcp unreachable; ollama http://127.0.0.1:11434/api/tags unreachable; … attempt 30: URLError: <urlopen error [Errno 111] Connection refused>); …)'
  allowed    …[mcp_example/mcp_client_ollama_pydantic_ai.ipynb] [network] -> 'network-unavailable: …'
OK: every skip in /tmp/phase13-mcp-pair-junit.xml matches the allowlist
audit_exit=0
```
This is the exact OLLM-02 acceptance state, demonstrated locally with no ollama and no MCP server — the same conditions stage 1 will have post-change.

## The Disposition Ledger Skeleton (input for the plan)

The planner owns format/numbering (Claude's discretion). Classification of every ollama ref in the 5 swept files, with dispositions fixed by D-01/D-02/D-04:

**Group K — KEEP, untouched (non-ollama contracts; D-02):**
- `tests/test_extras_guard.py` — all 5 refs: docstring line 4; `test_langchain_ollama_declared_in_mcp_extra` (lines 77-81, asserts verbatim `"langchain-ollama>=1.1.0" in mcp`). The mcp extra member stays declared `[VERIFIED: pyproject.toml:137 "langchain-ollama>=1.1.0"]`. `TestNotebookKernelPlotCompat` (line 112) is not ollama but shares the file — do not touch.
- `tests/test_runner_infra_contracts.py` — all 23 refs: module docstring (3-19), `UNIT_FILE` (39), `TestOllamaUnitPins` (67-103, reads committed unit only — D-05), `TestExampleNightlySseProbe` (112-147, stage-2 shape), `TestMcpExampleModelSwap` constants + 3 tests (156-252, committed notebook/mirror/README content pins incl. verbatim `"ollama:qwen3.5:4b"` and `"ollama pull qwen3.5:4b"`).
- `tests/test_models_lock_contracts.py` — 1 ref: docstring line 53 ("the mcp_example pair (local ollama models only)") — accurate prose, keep.

**Group S — SKIP via existing gate (ollama-execution surface; mechanism unchanged, zero code edits):**
- `tests/examples/test_notebook_execution.py`: `OLLAMA_URL` (1215), `_gate_ollama_stack` (1293-1329), `GATED_NOTEBOOKS` mcp rows (1427-1428), `_TIMEOUT_7200_GATED` mcp rows (1460-1461), `TestGateMatrix` (683-743), probe-retry semantics tests (~655-680), `TestKernelPlumbing` spec constants (758-759).
- `tests/examples/_execution.py`: `NOTEBOOK_EXEC_SPECS` mcp entries (245-280: cell_timeout 3600, langchain `kernel_name`), gated-layer comment (1237).

**Group C — COMMENT ACCURACY ONLY (prose describing deleted machinery):**
- `tests/examples/test_notebook_execution.py:1218-1232` — the D-07 stage-contract comment (stage-1 deselect description, "stage 3 OLLAMA BATCH", between-stage VRAM rationale).
- `.github/workflows/ci.yml` — header stage contract (612-637), D-12 actuals stage-3 rows (644-679), stage-1 step comment (872-879 area), stage-2.5 name+comment (990-995) and floor message (1012 "before the ollama batch"), plus the deletions/re-pins in the edit map below.

**Group X — DELETED (ci.yml machinery + its junit-list entry):** stage-3 step (1016-1048), stage3 junit in stage-4 list (1054-1056), `-k "not mcp_example"` at 863 and 884, old triple at 866-867.

**Sanctioned residue (OLLM-03):** post-change, `grep ollama tests/` (case-sensitive, 59 today) loses only the lowercase hits inside the Group C comment edits — the residue is otherwise the entire Group K + S surface. Exact final list is produced by running the grep at execution time and recording it verbatim in the plan; that record IS the success criterion.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PostHog-style "delete expired tests, never park them" as the default | Probe-then-execute typed skip for still-supported features (delete only dead surface) | v1.3 research (FEATURES §5, commit be31aad) | D-01/D-04 follow the feature-alive branch: the mcp client notebooks remain a supported local-execution feature, so the skip is the defensible state |
| Nightly brings ollama up (stage 3) and executes the pair | Nightly withdraws the environment; pair skips typed with live evidence; executes anywhere the environment exists | This phase | Restores the pre-08-09 posture for the pair while keeping every v1.1 honesty mechanism |

**Deprecated/outdated in-repo prose after this change:** stage-3 descriptions in ci.yml comments, `.github/workflows/README.md` (stage table rows + "Runner" paragraph sentence "stage 3 runs at the existing ~256k ctx, measured ~8 min"), `scripts/runner/README.md` nightly-VRAM sentence (line ~60), and — known, deferred to Phase 19 DOCR-01, NOT this phase — `docs/user_guide/continuous_integration.md:17` ("then the ollama-backed `mcp_example` pair").

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Stage-1 nightly P/S after the change ≈ prior 192P/1S + 6P + 2S (collection math verified; the run-record numbers are D-12-era and will be re-measured) | Pitfall 5, Edit Map | Comment states stale numbers — cosmetic; the hard gate uses collection, not P/S |
| A2 | This dev box is not (or is a standby for) the dnallm-nightly runner; its ollama state (inactive/not-found, port 000) may not reflect the runner host's | Runtime State Inventory | Ops note wording; zero repo impact — the plan's ops section should phrase the stop/disable as an owner action on the runner host |
| A3 | A dispatched (workflow_dispatch) example-nightly run is available at phase end for the Pitfall-6-mandated verification run | Open Questions | Verification falls back to owner-watched nightly + the local end-to-end proof already captured above |
| A4 | ~2 min added to stage 1 from the retry window is acceptable (117s measured locally; refused-connection case) | Pitfall 6 | If the runner host DROPs instead of refusing: up to ~7 min total — still trivial vs 2700-min budget; note in the budget comment |

All mechanical claims driving the plan (triple values, selector flags, junit list, prefix strings, helper line numbers, audit behavior) are `[VERIFIED]` by direct Read of the source file or by local execution this session, quoted inline above.

## Open Questions

1. **Stage-2.5 final naming** — Claude's discretion: recommended shape is keep the `Stage 2.5:` prefix (Pitfall 2 constraint), drop `pre-ollama`, reframe as "end-of-run hygiene before stage-4 audits". Alternatively keep the name fully and reword only the comment. Either is compliant; the plan just picks one.
2. **expected_skips.yaml "update"** — OLLM-02 requires it be "updated"; mechanically NO matcher change is needed (`network-unavailable:` entry pre-exists at lines 29-30). Recommended: a one-line comment refresh on that entry noting the example-nightly main batch now exercises it (2026-10-11, Phase 13 D-04) — satisfies the requirement honestly without touching matchers. The plan should state explicitly that adding any new entry would be wrong.
3. **Verification-run logistics** — who dispatches the post-change example-nightly (owner vs agent with gh access), and whether coverage-nightly's next natural run suffices as the green evidence (it already exercises the skip today).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Repo `.venv` with pytest | census measurement, local verification runs | ✓ | present at `.venv/bin/python` (all measurements this session ran on it) | — |
| ollama on `127.0.0.1:11434` | nothing (its ABSENCE is the tested state) | ✗ (down — HTTP 000) | — | Not needed: down is the post-change nightly condition |
| MCP server on `:8000` | nothing (absence is the tested state) | ✗ (HTTP 000) | — | Same |
| `scripts/audit_skips.py` + allowlist | OLLM-02 proof | ✓ | in-repo, exit 0 demonstrated | — |
| dnallm-nightly self-hosted runner | final nightly-green evidence | not probeable from this session | — | workflow_dispatch at phase end; local end-to-end proof already captured |
| git history precedent (c99fa9d) | re-pin pattern | ✓ | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** the nightly-runner verification (A3) — fallback documented.

## Security Domain

ASVS level 1 (config: `security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface changes |
| V3 Session Management | no | None |
| V4 Access Control | yes (preserved, not extended) | The ollama loopback bind (`Environment="OLLAMA_HOST=127.0.0.1:11434"`) is the access control for the unauthenticated model server; `TestOllamaUnitPins` keeps it pinned and stays green (D-05). Deleting stage 3 REMOVES a network listener start from CI — net surface reduction |
| V5 Input Validation | no | No new input parsing; audit_skips.yaml already fail-closed on malformed entries |
| V6 Cryptography | no | None |
| V10 Malicious Code | no | No new dependencies or download steps; the deleted stage contained no fetches |

### Known Threat Patterns for this change shape

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| CI workflow edit weakening gates (forever-green job) | Repudiation | The fail-soft summary + audit loop stay intact; only a genuinely-produced junit set is audited; stage-0.5 hard gate remains job-fatal |
| Skip-message forgery (allowlist gaming) | Tampering | No new allowlist entries; existing `network-unavailable:` entry demands real probe evidence in-message (30-attempt log verified this session) |
| Unauthenticated model server exposure on runner | Information Disclosure | Loopback pin kept in unit file + pin tests kept green (D-05); README live-drift notice (0.0.0.0, 2026-10-05) retained |

## Sources

### Primary (HIGH confidence — all verified this session by direct Read or execution)
- `tests/examples/test_notebook_execution.py` — gate implementation (1215-1329), gated parametrization (1426-1465), TestGateMatrix (683-743), stage-contract comment (1218-1232)
- `tests/examples/_execution.py` — skip helpers (1197-1249), mcp spec entries (245-280)
- `.github/workflows/ci.yml` — example-nightly job (596-1099): stage 0.5 (853-870), stage 1 (872-891), stage 2.5 (990-1014), stage 3 (1016-1048), stage 4 (1050-1099); coverage-nightly full-suite selector (581-594)
- `tests/expected_skips.yaml` — full allowlist (88 lines)
- `scripts/audit_skips.py` — full read (133 lines)
- `tests/test_runner_infra_contracts.py` — full relevant surface (252 lines)
- `tests/test_extras_guard.py`, `tests/test_models_lock_contracts.py`, `scripts/runner/ollama.service`, `pyproject.toml` (mcp extra), `.github/workflows/README.md`, `scripts/runner/README.md`
- Local executions: census collect-only (both selectors), mcp-pair run + audit (exit 0, 117s), endpoint probes
- git: `c99fa9d` re-pin precedent (diff read this session)

### Secondary (MEDIUM confidence — committed project research, sources embedded therein)
- `.planning/research/PITFALLS.md` §Pitfall 5/6/7 (commit be31aad)
- `.planning/research/FEATURES.md` §5 (PostHog/dataretrieval norms with upstream URLs)
- `.planning/research/SUMMARY.md` §Phase O; `.planning/codebase/TESTING.md`

### Tertiary (LOW confidence)
- None — no external web claims were needed; the phase is fully determined by in-repo evidence.

## Metadata

**Confidence breakdown:**
- Disposition mechanism (skip via gate, audit green): HIGH — proven end-to-end locally this session with pasted output
- Census numbers: HIGH — both triples measured with the exact flags; deselect membership enumerated (8 tests, ids listed)
- ci.yml edit map: HIGH — every line range Read this session; couplings (stage-4 list, Stage 2.5 literal) verified against the consuming test
- Nightly-box runtime behavior post-change: MEDIUM — inferred from gate logic + coverage-nightly precedent (the box itself not exercised this session; A2/A3)
- Ecosystem norms: inherited MEDIUM from committed research (not re-fetched; all search-provider flags disabled in config and no external question remained open)

**Research date:** 2026-10-11
**Valid until:** 2026-11-10 (stable: in-repo surfaces; re-verify census numbers if any test lands in tests/examples before this phase executes)
