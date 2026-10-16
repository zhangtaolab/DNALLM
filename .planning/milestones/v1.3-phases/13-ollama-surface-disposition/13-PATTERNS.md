# Phase 13: Ollama Surface Disposition - Pattern Map

**Mapped:** 2026-10-11
**Files analyzed:** 5 repo files to modify (0 new code files; plus plan artifacts — ledger + residue list — that live in PLAN.md, not in code)
**Analogs found:** 5 / 5

**Phase shape (drives everything below):** this is a deletion + re-pin + comment-accuracy phase. Zero functional edits to any test file. The skip mechanism (`_gate_ollama_stack` → `network_unavailable_skip` → `network-unavailable:` prefix) already exists and is already green in coverage-nightly. Consequently the "analog" for each edit is a **precedent commit that made the same kind of edit**, plus the in-file patterns those commits followed.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `.github/workflows/ci.yml` | config (CI workflow) | batch | commit `6218948` (step deletion + marker comments) + commit `c99fa9d` (census triple re-pin) + ci.yml's own coverage-nightly no-deselect census (lines 580-594) | exact |
| `.github/workflows/README.md` | docs | n/a (prose) | commit `6218948` README hunks + the current stage table (lines 127-140) | exact |
| `scripts/runner/README.md` | docs (ops) | n/a (prose) | commits `000c743` / `a072f0d` (narrative accuracy fixes under pin tests) + pin tests in `tests/test_runner_infra_contracts.py` | exact |
| `tests/examples/test_notebook_execution.py` | test (comment-only edit) | n/a | its own D-07 stage-contract comment block (lines 1218-1232) + `6218948`'s same-change header-comment rewrite | exact |
| `tests/expected_skips.yaml` | config (skip allowlist) | n/a (data) | its own entry-comment style (lines 32-36, 83-87) | exact |

Not code (no analog needed): the per-assertion disposition ledger and the sanctioned `grep ollama tests/` residue list — plan artifacts recorded in PLAN.md per OLLM-01/OLLM-03.

## Pattern Assignments

### `.github/workflows/ci.yml` (config, batch)

**Analogs:** commit `6218948` "ci(09-01): topology surgery — evo provisioning + models-cache deletions" (deletion shape), commit `c99fa9d` "fix(quick-261008-wfx): bump D-03 census triple 197/206 -> 208/217" (re-pin shape), and the file's own coverage-nightly census (the post-change stage-1 shape).

**Edit targets (current line numbers, all read this session):**

| Lines | What | Action |
|-------|------|--------|
| 612-637 | header D-07 stage-contract comment (stage-3 row at 632-633, stage-2.5 row at 630-631) | rewrite prose; drop stage-3 row; reword "before the ollama batch" |
| 644-679 | D-12 measured-actuals comment (stage-3 rows 664-670; stage-1 row 657-659 "192P/1S/9 deselected") | remove stage-3 rows, re-measure stage-1 numbers (Pitfall 5: +6P +2S) |
| 853-870 | stage 0.5 hard gate | drop `-k "not mcp_example"` (line 863); re-pin triple (lines 866-867) |
| 872-891 | stage 1 | drop `-k "not mcp_example"` (line 884); reword step comment (873-879) |
| 990-1014 | stage 2.5 | keep literal `Stage 2.5` name prefix; reword `(D-13, pre-ollama)` suffix + comment + floor message (line 1012 "before the ollama batch") |
| 1016-1048 | stage 3 (banner comment 1016-1026 + step 1027-1048) | DELETE; leave a deletion-marker comment (pattern below) |
| 1054-1056 | stage-4 junit audit list | remove `pytest-junit-mcp-stage3.xml` list member (Pitfall 1 — hard coupling) |

**Census re-pin pattern** (precedent `c99fa9d`, diff applied verbatim to lines 866-867; the commit message recorded "Measured locally with the exact Stage 0.5 flags"):

Current (lines 862-870):
```yaml
        run: |
          .venv/bin/python -m pytest tests/examples --collect-only -q \
            -m "not giants" -k "not mcp_example" 2>/dev/null | tee census-collect.txt | tail -1
          MEASURED=$(tail -1 census-collect.txt | sed -E 's/^=+ *//; s/ *=+$//; s/ in [0-9.]+s?$//')
          echo "census collection measured: ${MEASURED}"
          if ! tail -1 census-collect.txt | grep -qE '^=* *208/217 tests collected \(9 deselected\) in '; then
            echo "FAIL: census collection mismatch (D-03) — expected triple: 208/217 tests collected (9 deselected); measured line: $(tail -1 census-collect.txt)"
            cat census-collect.txt
            exit 1
          fi
```
Post-change triple (measured this session per RESEARCH): `216/217 tests collected (1 deselected)` — selector becomes `-m "not giants"` only. Both the grep (line 866) and the FAIL message (line 867) carry the triple; `c99fa9d` changed exactly these two lines.

**Post-change stage-1 shape analog** — coverage-nightly already runs the full suite with no deselect and documents the typed-skip outcome (ci.yml lines 580-589):
```yaml
      # NOTE: 27 tests carry the slow mark; 21 execute here. The 6 MCP
      # live-server probes ... target localhost:8000 and no CI job
      # starts a server there, so they skip deterministically as typed
      # "network-unavailable:" skips (allowlisted in tests/expected_skips.yaml).
      # Those probes are local-only.
      - name: Run gated full census (slow included)
        run: |
          source .venv/bin/activate
          .venv/bin/python -m pytest -ra --durations=0 --junitxml=pytest-junit-nightly.xml --cov -p no:cacheprovider -p no:progress
```
This is the model for what stage 1 becomes: no `-k` filter; the gated pair skips typed with probe evidence; the junit goes to the stage-4 audit which passes it.

**Step-deletion marker pattern** (precedent `6218948` — deleted steps leave a dated, decision-coded comment where they stood; three such markers are live in ci.yml today, e.g. lines 778-780 and 789-794):
```yaml
      # D-04: the evo isolated-venv provisioning step that stood here is
      # deleted with the evo/giants census exit (full note beside the
      # wheelhouse deletions below).
```
Stage-3's replacement marker should follow this shape (decision code D-04, date, one-line rationale pointing at the restore path: re-adding a stage-3-shaped batch, and at `scripts/runner/ollama.service` + the self-healing gate).

**Fail-soft ledger coupling** (unchanged machinery the deletion must not disturb — stage-4 audit list lines 1050-1068, summary lines 1086-1099):
```yaml
          for junit in pytest-junit-example-stage1.xml pytest-junit-yaml-stage1.xml \
            pytest-junit-mcp-stage2-streamable.xml pytest-junit-mcp-stage2-sse.xml \
            pytest-junit-mcp-stage3.xml; do
            ...
              echo "stage4-audit-$junit=1 (junit missing: pytest never wrote it)"
              echo "stage4-audit-$junit=1" >> stage-results.txt
```
```yaml
          if grep -qE '=[1-9][0-9]*$' stage-results.txt; then
            echo "FAIL: at least one stage item exited non-zero (D-08: this job may never be forever-green)"
```
A listed-but-missing junit writes `=1` and the summary's end-anchored grep turns the job red — hence the audit-list edit is mandatory in the same change (Pitfall 1). Also unchanged: upload globs `pytest-junit-*.xml`, `mcp-server-*.log`, `stage*.log` (lines 1079-1084) absorb the vanished artifacts with no edit.

---

### `.github/workflows/README.md` (docs)

**Analog:** commit `6218948`'s README hunks — the topology-surgery commit rewrote the Workflow Triggers section, the coverage-nightly Model Caches bullet ("now records the D-11 removal"), and added the example-nightly stage table in the same change as the ci.yml surgery. Deletions are recorded in prose with decision codes.

**Edit targets:**
- Line 123 (Runner paragraph): the sentence "The mcp pair's num_ctx cut is **DEFERRED** (owner decision 2026-10-06) — stage 3 runs at the existing ~256k ctx, measured ~8 min with no memory-floor pressure on the 128GB box." — stage-3 claim must go; replace with the D-04 disposition (pair skips typed in stage 1; executes only where ollama + server exist).
- Stage table rows (lines 127-136): row 0.5 (line 130) names `-k "not mcp_example"` in the pinned selector; row 1 (line 131) "with the two `mcp_example` ids deselected (stage 3 owns them)"; row 2.5 (line 134) "before the ollama batch"; row 3 (line 135) "fresh MCP server + the `mcp_example` pair (ollama loopback)". Rows 0/1.5/2/4 and the fail-soft / complete-failure-scene paragraphs (138-140) stay.
- Current row-1 text to rework (line 131):
```markdown
| 1 | torch-heavy example execution: `pytest tests/examples` with the two `mcp_example` ids deselected (stage 3 owns them) and the `giants`-marked evo execution test deselected (D-01 owner-policy exclusion ...) ... |
```

**Prose-recording pattern for deletions** (this README, line 144, written by `6218948` — the model sentence shape):
```markdown
... the evo provisioning steps (isolated venv, flash-attention wheelhouse,
giant-model prefetch) are deleted per D-04 — the flash-attn build-isolation
failure of run 37278002681 dissolved with that deletion.
```

---

### `scripts/runner/README.md` (docs, ops)

**Analog:** commits `000c743` ("reconcile num_ctx deferral narrative") and `a072f0d` ("drop retired model name from runner README narrative") — minimal narrative-accuracy fixes made under the fast-lane pin tests.

**Edit target (minimal):** the nightly-VRAM phrase in the "Why num_ctx 8192 (D-06)" section, lines 60-61 — "...dominates mcp-notebook latency and the nightly VRAM budget..." — the nightly no longer loads the model. The "How the D-13 readiness probe treats a down service" section (lines 90-99) already describes the exact post-change state and stays.

**HARD CONSTRAINT — pin tests read this file; these literals must survive any edit** (`tests/test_runner_infra_contracts.py`):
- `TestRunnerReadmeReapply::test_readme_documents_reapply_op_and_num_ctx_rationale` (lines 93-108): requires `"daemon-reload"` (line 96), `"restart ollama"` (line 101), and `"num_ctx"` + `"8192"` (line 105) present.
- `TestMcpExampleModelSwap::test_runner_readme_pull_and_verify_name_current_model` (lines 234-252): requires `"ollama pull qwen3.5:4b"` (line 237), `"must list qwen3.5:4b"` (line 243), and `OLD_MODEL_NAME` (`"qwen3.8"`) absent (line 248).

---

### `tests/examples/test_notebook_execution.py` (test, comment-only edit)

**Analog:** the comment block itself (lines 1218-1232) — the file's own D-07 stage-contract comment, which ci.yml's header comment explicitly mirrors ("these module comments are the contract it implements", line 1231-1232).

**Edit target — lines 1218-1232 ONLY; zero functional edits anywhere in the file:**
```python
# D-07 stage contract (08-08, MCP-02): within one example-nightly job the
# three GPU/VRAM/port consumers run STAGED SERIAL, never concurrently --
#   stage 1 TORCH-HEAVY: pytest tests/examples with the two mcp gated ids
#     DESELECTED (Pitfall 7: `--deselect` both mcp_example entries, else the
#     pair runs once as an honest skip here and once for real in stage 3,
#     polluting the junit audit);
#   stage 2 MCP LIVE SERVER: ...
#   stage 3 OLLAMA BATCH: with ollama up (D-13 probe below), re-run just the
#     two gated mcp tests (server up + ollama probed).
# BETWEEN stages: explicit kernel pkill + VRAM settle ... The ci.yml
# wiring lands in 08-09; these module comments are the contract it
# implements.
```
Rewrite to the post-change contract: no stage 1 deselect; the pair stays collected in stage 1 and typed-skips via the gate when ollama/server are down (executes where they exist — self-healing). Keep untouched: `OLLAMA_URL`/`MCP_ENDPOINT` (1215-1216), `_probe_http*`, `_gate_ollama_stack` (1293-1329), `TestGateMatrix`, `GATED_NOTEBOOKS` rows, and the historical run citations in `_execution.py` (257, 280 — accurate history).

**The skip mechanism that must NOT be edited** (lines 1312-1329 — this is what fires post-change):
```python
    ollama_ok, ollama_ev = _probe_http_with_retry(OLLAMA_URL)
    server_ok, server_ev = _probe_http(MCP_ENDPOINT)
    if ollama_ok and server_ok:
        return
    ...
    network_unavailable_skip(
        f"execute {nb_name} (mcp client notebook)",
        evidence=f"{'; '.join(missing)}; {evidence}",
    )
```

---

### `tests/expected_skips.yaml` (config, skip allowlist)

**Analog:** the file's own entry-comment style — each entry carries a dated, scope-explaining comment (e.g. the Phase-11 ClinVar entry, lines 32-36, and the Phase-12 JASPAR entry, lines 83-87).

**Edit: comment-only refresh on the existing entry (lines 26-30). NO matcher changes — adding any new entry is wrong (RESEARCH Open Question 2):**
```yaml
  # The 6 slow-marked MCP live-server probes (dnallm/mcp/tests/), rewritten
  # by FIX-03 to the typed helper skip_if_unreachable. Never collected on
  # the CI fast leg; serves local/full runs and the Phase-4 slow leg.
  - prefix: "network-unavailable:"
    category: network
```
Follow-up comment shape to copy (Phase-12 entry, lines 83-87):
```yaml
  # Typed network skip for the Phase-12 JASPAR client lane
  # (tests/interpret/test_motifs.py::TestJasparLive): fires when the canonical
  # jaspar.elixir.no host is unreachable on the slow leg.
```
NOTE for plan prose: the exercised prefix is `network-unavailable:` (this entry), NOT `environment-unavailable:` (lines 38-42) — CONTEXT D-04 names the wrong one; RESEARCH Pitfall 4 corrects it. Ledger and README fixes must name `network-unavailable:`.

## Shared Patterns

### Typed-skip enforcement chain (already shipped, zero edits)
**Source:** `tests/examples/_execution.py:1230-1249` (`network_unavailable_skip` → `pytest.skip(f"network-unavailable: {action} ({evidence})")`) → junit `<skipped message>` → `scripts/audit_skips.py` → `tests/expected_skips.yaml`.
**Apply to:** verification of every plan action. The audit (read in full, 133 lines) fails closed on absent/unparseable junit (lines 88-92), ignores `pytest.xfail` skips (100-104), and passes only when every skip message matches exactly one `exact`/`prefix`/`reason_like` entry (49-63). Post-change proof command (already demonstrated, RESEARCH lines 262-272):
```bash
.venv/bin/python -m pytest tests/examples/test_notebook_execution.py -k "mcp_example" -q --junitxml=/tmp/pair-junit.xml
.venv/bin/python scripts/audit_skips.py /tmp/pair-junit.xml tests/expected_skips.yaml   # → OK, exit 0
```

### Same-change contract sync (comment ↔ ci.yml ↔ READMEs)
**Source:** Pitfall 7 + the `6218948` precedent (ci.yml surgery, its header stage-contract comment, and the workflow README all moved in one commit).
**Apply to:** the stage-3 deletion must land atomically with: ci.yml header/D-12 comments, the test module comment (1218-1232), both READMEs, the triple re-pin, the `-k` drops (both lines 863 and 884), and the stage-4 audit-list edit.

### Invariants that consume ci.yml text (do not break)
**Source:** `tests/test_runner_infra_contracts.py`.
**Apply to:** every ci.yml edit.
```python
def _load_sse_probe_block() -> str:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    start = text.index("--transport sse > mcp-server-sse.log")
    end = text.index("Stage 2.5", start)
    return text[start:end]
```
(lines 54-64) — the stage-2.5 step name MUST retain the literal `Stage 2.5` prefix or `str.index` raises `ValueError` and the fast leg goes red. Likewise the runner-README literals listed above. The mcp-stage3 junit filename must leave the audit list when the step that produced it leaves the job.

### Census re-pin discipline
**Source:** commit `c99fa9d` (and the stage-0.5 comment itself, lines 853-860: "the pinned triple below is the deliberate bump-point").
**Apply to:** measure with the EXACT new stage-0.5 flags before editing; change only the two triple-bearing lines; record the measured triple in the commit message; never widen the grep.

### Deletion-marker comment discipline
**Source:** ci.yml lines 778-780, 789-794, 838-840 (all from `6218948`).
**Apply to:** the stage-3 deletion site — leave a short `# D-04 (2026-10-11): the stage-3 ... step that stood here is deleted ...` marker rather than a silent gap; name the restore path.

## No Analog Found

None — every edit in this phase has a direct in-repo precedent (5/5 matched). The only genuinely new artifacts are the plan-level per-assertion ledger and sanctioned residue list, which are documentation the planner formats at its own discretion (CONTEXT "Claude's Discretion"), not code needing a pattern.

## Metadata

**Analog search scope:** `.github/workflows/` (ci.yml full example-nightly job + coverage-nightly census), `tests/` (expected_skips.yaml, test_runner_infra_contracts.py full, test_notebook_execution.py gate section), `scripts/` (audit_skips.py full, runner/README.md full), git history (`c99fa9d`, `6218948`, `000c743`, `a072f0d` — all verified via `git show`; all named paths verified git-tracked via `git ls-files`).
**Files scanned:** 7 source files read + 4 precedent commits.
**Pattern extraction date:** 2026-10-11
