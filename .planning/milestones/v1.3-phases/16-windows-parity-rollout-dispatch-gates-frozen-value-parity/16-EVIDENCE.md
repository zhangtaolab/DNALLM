# Phase 16 Evidence Record — SC4 Nightly Re-execution + 5/5 SC Proof Matrix

**Recorded:** 2026-10-13/14 (executor run of plan 16-04; SC4 dispatched 2026-10-13T17:45Z, run concluded 2026-10-14T09:27Z — terminal record in section A; async-job manifest at `.planning/async-jobs/16-04-sc4-nightly.json` is terminal-resolved)
**Executor box:** forrest@edgexpert (GB10) — the same box that hosts the dnallm-nightly runner (id 22)
**Dev-venv:** `.venv/bin/python` (repo venv, fast lane 2550 passed this session)

Section order: A SC4 dispatch + terminal nightly record (concluded; rice-class caveat on two non-phase lanes, every phase-16 surface green), B re-check + capture commands, C SC proof matrix (5/5 — SC4 green-where-executable with recorded caveat), D phase-wide honesty ledger (all green), E planning-assumption dispositions.

---

## A. SC4 — nightly re-execution over the complete phase change

**Status: CONCLUDED 2026-10-14T09:27:42Z (run 38441258701, run-level conclusion `failure`) — every phase-16 surface GREEN on the dispatched ref; the only reds are the owner-accepted rice.uga.edu external-host class on the two genome-download lanes, neither of which reached phase-16 code (4th and 5th manifestations; phase-14 H.4/H.4.final precedent; owner-directed accept-with-caveat, section A.5).**

### A.1 Pre-flight (all verified live this session)

| Check | Result |
|-------|--------|
| Remote ref carries the phase commits | `git push origin v1.3` — `a45e61c..4863883` (17 commits, all of 16-01/02/03 + planning docs); post-push `origin/v1.3` tip = `4863883890fe62c845e3c81da06c7e07d4e8e282`, local ahead/behind = 0/0 |
| Runner online | self-hosted `dnallm-nightly` (repo runner id 22, `~/github-runners/dnallm-nightly`) — `Runner.Listener` PID 31376 alive since 2026-10-12 09:53:04 (post-power-resume start), worker diag log actively written at 17:42Z (job executing) — recovery manual NOT needed |
| gh CLI authenticated | `gh auth status` — forrestzhang, scopes incl. `workflow` (dispatch-capable) |
| Dispatched headSha = pushed phase head | run `38441258701` `headSha` = `4863883890fe62c845e3c81da06c7e07d4e8e282` = `git rev-parse HEAD` — the nightly checks out the COMPLETE 16-01/02/03 tree, no partial-tree evidence (T-16-09 mitigation) |
| D-06 stagger (no double-queue) | one in-flight job held the runner at dispatch time (see A.3); the dispatch added exactly one run to the queue |

### A.2 Dispatch record

```
$ gh workflow run ci.yml --repo zhangtaolab/DNALLM --ref v1.3
→ https://github.com/zhangtaolab/DNALLM/actions/runs/38441258701
```

- **Run id:** 38441258701
- **URL:** https://github.com/zhangtaolab/DNALLM/actions/runs/38441258701
- **Dispatched (created):** 2026-10-13T17:45:44Z (UTC)
- **Event / ref / headSha:** `workflow_dispatch` / `v1.3` / `4863883890fe62c845e3c81da06c7e07d4e8e282`
- **Job set on dispatch (D-19 gates working as designed):** `coverage-nightly`, `example-nightly`, `test-mamba` all queued (the manual dispatch fires BOTH nightlies — one dispatch is the full SC4 evidence source, AS-1); push/PR-gated jobs (`test` matrix, `test-cuda`, `test-windows`, `coverage-gate`, `deploy`) skipped

### A.3 Queue forensics at window close (18:20Z) — why the evidence is pending

The single `dnallm-nightly` runner serializes all self-hosted jobs. At dispatch time the queue held, in lease order:

1. **In-flight (not phase-16 work):** run `38354875692` (the phase-14 re-dispatch, ref v1.3 @ `2c73751` — a PRE-phase-16 tree; phase 14 already closed by owner accept-and-close 2026-10-13T13:2xZ) — its `example-nightly` job leased the runner 14:30:52Z and sat in Stage 1 (torch-heavy examples) from 16:11:16Z, ~2h08m at window close vs the 1:49:56 precedent stage-1 duration (the stretch is consistent with rice.uga.edu retry backoff; job timeout is 2700 min, so no self-termination guarantee).
2. **Queued ahead (foreign, created before this dispatch):** run `38381090082` (schedule/main, created 10:06:33Z — delayed 03:00-UTC cron) holding `coverage-nightly` + `test-mamba` queued (~4.5h serial work); run `38400225941` (schedule/main, created 12:49:58Z — delayed 05:30-UTC cron) holding `example-nightly` queued (~2.3h). Both run on `main`, which does NOT carry the phase change — they are pure queue latency for SC4, not evidence.
3. **This plan's run `38441258701`:** all three nightly jobs queued behind the above.

**Cancel attempt, recorded:** cancelling the two stale scheduled runs (queue hygiene per the phase-13 recovery manual's established cancel+re-dispatch pattern) was DENIED by the runtime permission classifier ("Interfere With Workloads" — cancelling runs this executor did not dispatch). Respected; no workaround attempted. Consequence: the first SC4 nightly job leases only after ~6h of foreign work (est. ~01:00Z+ 2026-10-14), full SC4 evidence est. ~08:00–09:00Z — beyond this executor's window. The scheduled runs re-fire on their daily cron, so nothing is lost by letting them drain.

**Lease-order note:** the phase-13 manual's example-first technique is moot here — SC4 needs BOTH nightly jobs green, so order among this run's three jobs does not change the evidence-complete time. No reorder cycles attempted.

### A.4 Terminal record — job conclusions and evidence rows (all live-recheckable)

**Run 38441258701** (https://github.com/zhangtaolab/DNALLM/actions/runs/38441258701), ref v1.3, headSha `4863883890fe62c845e3c81da06c7e07d4e8e282` (the COMPLETE 16-01/02/03 tree), concluded `failure` at 2026-10-14T09:27:42Z. Job outcomes (job ids live-recheckable via `gh run view --job <id> --json status,conclusion`):

| Job | Id | Conclusion | Window (UTC) |
|-----|----|------------|--------------|
| test-mamba (3.11) | 115448126580 | **success** | 2026-10-13T21:58:43Z → 22:03:25Z |
| example-nightly | 115448126933 | **failure** (1 rice-class red; every phase-16 lane green — rows below) | 2026-10-14T01:41:09Z → 05:42:37Z |
| coverage-nightly | 115448126935 | **failure** (2 rice-class reds; full census 2607 passed incl. both new root suites — rows below) | 2026-10-14T05:42:41Z → 09:27:41Z |

**example-nightly — durable junit artifact of record** (`gh run download 38441258701 -n example-nightly-junit`; captured 2026-10-14T09:32Z, archived at `/tmp/p16-sc4-example-junit/`):

- Stage-0.5 census hard gate **HELD**: `census-collect.txt` tail = `=============== 216/217 tests collected (1 deselected) in 1.22s ================` — the D-03 triple byte-identical, collected over the complete phase change (the job proceeded past the gate by construction).
- Suite header: `tests="216" failures="1" skipped="5" errors="0"` (timestamp 2026-10-14T01:57:14Z, hostname edgexpert-3eee); stage-results.txt: `stage1-examples=1`, all other stages + all four stage-4 audits `0`.
- **CRE band re-executed GREEN on the Linux truth chain:** `<testcase classname="tests.examples.test_plant_helixseek_showcase" name="test_cre_notebook_executes_within_selection_bands" time="254.961" />` — the parsed selection.md CRE jaccard band held over the dispatched cell-11 change (the dispatch did not leak).
- **NER notebook lane GREEN end-to-end on POSIX:** `test_notebook_executes_end_to_end[notebooks/finetune_NER_task/data_generation_and_inference.ipynb]` time=`4023.932` — the dispatched notebook (cells 0/1/2/8) executed the real pipeline on the bedtools+pybedtools chain, incl. its in-notebook selection.md evidence-key comparisons; sibling `finetune_NER_task.ipynb` also green (927.585).
- **The 5 skips = exactly the historical allowlisted set** (14-EVIDENCE section D classification, zero new skips): predict_data content skip; 2× mcp_example `network-unavailable:`; 2× pygenometracks `optional-dep:` — the Anno/combined band rows carry the pre-existing pygenometracks typed skip (identical to the phase-14 GREEN run of record; not a phase-16 surface).
- **The sole failure — rice.uga.edu fixture-seeding hang, BEFORE any phase-16 code ran:** `test_generate_bpe_dataset_produces_artifact` — traceback `test_script_execution.py:192 → _seed_rice_input (:110) → _download (:75) → shutil.copyfileobj` (ssl recv stuck; pytest-timeout 3600s). The script's dispatch seam never executed — the failure is in the fixture download helper, upstream of the code under test.

**coverage-nightly — full census incl. slow** (job log extracted from GitHub 2026-10-14T09:32Z, archived at `/tmp/p16-sc4-cov-log.txt`; the runner-workspace junit was cleaned by the next queued job — the log-extracted lines below are the durable record):

- Summary line (verbatim): `2 failed, 2607 passed, 17 skipped, 7 warnings, 23 subtests passed in 13390.29s (3:43:10)` (2026-10-14T09:27:29Z).
- **Both new root suites collected and GREEN in the full census — all 19 + 4 per-test PASSED rows present in the log**, including the strongest single evidence line of the phase: `tests/test_dispatch_direction.py::test_ner_script_win32_rows_match_live_pybedtools_loj PASSED` — the **live-parity branch EXECUTED on the nightly** (pybedtools installed + bedtools provisioned there), proving numpy == live bedtools-chain loj on CI over the complete change. Alongside: `test_ner_script_win32_rows_match_frozen_loj_truth`, `test_ner_adapter_textually_identical_script_and_cell8`, and all CRE/gate structure contracts PASSED.
- **Frozen-pair bands GREEN in the same census:** `test_frozen_cre_pair_jaccard_dual_oracle_and_band PASSED`, `test_frozen_anno_pair_exon_counts_rational_and_tight_band PASSED` (the parsed tight exon_f1 band held on Linux), `test_frozen_loj_pair_byte_parity_and_consecutive_null_sextet PASSED`, `test_band_parse_guard_fails_red_on_missing_row PASSED`.
- **The 2 failures — both the same rice.uga.edu class, neither a phase-16 surface:** (1) `test_notebook_executes_end_to_end[notebooks/finetune_NER_task/data_generation_and_inference.ipynb]` — `nbclient.exceptions.CellTimeoutError: A cell timed out while it was being executed, after 3600 seconds` on the genome-download cell (`3605.13s call`); (2) `test_generate_bpe_dataset_produces_artifact` — `Failed: Timeout (>3600.0s) from pytest-timeout` (same `_seed_rice_input` download hang as example-nightly). Both are input-download failures upstream of the phase's dispatch seams; the NER notebook lane that DID get its genome (example-nightly) passed for real at 4023s.

**Band-evidence summary (what "the nightly holds the parsed selection.md bands" means over this run):** CRE jaccard band — live re-executed GREEN (example-nightly); Anno exon_f1 band — GREEN via the frozen-pair suite parsing the same selection.md row (coverage-nightly 2607; the Anno notebook band test itself carries the pre-existing pygenometracks typed skip, unchanged since before the phase); negative-CRE/negative-Anno bands — inside the CRE-green / historically-skipped showcase tests; NER in-notebook evidence-key comparisons — GREEN in example-nightly's 4023s end-to-end pass. Tight frozen-pair band — GREEN (coverage-nightly). No band regressed anywhere in the run.

### A.5 Triage disposition — recorded, owner-directed (never a silent retry)

Per the plan's triage discipline, the red signatures were classified BEFORE any disposition: both failures sit in the `_seed_rice_input`/genome-download input path (rice.uga.edu), the **4th and 5th independent manifestations** of the class already owner-accepted in phase 14 (H.4: truncated FASTA KeyError Chr6; H.4.final: CellTimeoutError; owner accept-and-close 2026-10-13T13:2xZ with "NER genome-download fragility (no cache/timeout resilience on rice.uga.edu) recorded for phase-16 planning"). Zero phase-16 regression signal exists in the run — positively: the live-pybedtools parity test executed green, the CRE band re-executed green, the NER notebook passed end-to-end in example-nightly, both root suites went 19+4 green in the full census, and the census hard gate held byte-identical.

The plan's rice-class routing (record + single sanctioned re-dispatch) was superseded by the owner-side orchestration directive to accept-with-caveat per the phase-14 H.4.final precedent — the escalation endpoint that routing exists to reach; the class is already double-manifested historically, and a re-dispatch would re-roll the same external host (the phase-14 re-dispatch itself red on rice again). Evidence file records the failures AND the resolution (this section); WINDOWS #24 (the unrun-verify arm) is resolved with this record as the reason of record; the caveat is carried in 16-04-SUMMARY and STATE. The rice.uga.edu fragility remains an open external dependency owned upstream of this repo (no cache/timeout resilience on that host in the NER input path).

---

## B. Re-check and capture commands (live-recheckable, for the follow-up)

```bash
# Run-level conclusion (the SC4 verify step re-checks this live)
gh run view 38441258701 --repo zhangtaolab/DNALLM --json status,conclusion
# Per-job conclusions + step states
gh run view 38441258701 --repo zhangtaolab/DNALLM --json jobs \
  --jq '.jobs[] | {name, status, conclusion, startedAt, completedAt}'

# example-nightly evidence (junit artifact carries all log lines):
gh run download 38441258701 --repo zhangtaolab/DNALLM -n example-nightly-junit -D /tmp/p16-sc4-example-junit/
#   census-collect.txt tail        → the 216/217 triple (stage 0.5)
#   pytest-junit-example-stage1.xml→ PASS rows: CRE band, Anno band, NER notebook, BPE script lanes
#   stage1-examples.log            → the parsed-band assertion lines (jaccard/exon_f1 evidence keys)

# coverage-nightly evidence: the job uploads NO junit artifact (phase-14 H.4 precedent) —
# capture pytest-junit-nightly.xml from the runner workspace BEFORE the next serialized job's
# checkout cleans it (box-local path):
#   ~/github-runners/dnallm-nightly/_work/DNALLM/DNALLM/pytest-junit-nightly.xml
# or reconstruct from job logs: gh run view --job <job-id> --log
```

---

## C. SC proof matrix (4/5 green + SC4 pending)

Every green row's command was re-run fresh by this executor on 2026-10-13 and is re-runnable as written from the repo root. SC4's row is the only open one.

| SC | Criterion (ROADMAP verbatim, abridged) | Command (re-runnable) | Result |
|----|----------------------------------------|------------------------|--------|
| SC1 | acquisition-only fork; consumer byte-identical; install cell skips pybedtools on Windows | `.venv/bin/python -m pytest tests/test_dispatch_direction.py -q --no-header` → 19 passed; `.venv/bin/python -m py_compile example/notebooks/finetune_NER_task/generate_bpe_dataset.py` → OK; `grep -cE 'sys\.platform == "win32"' example/notebooks/finetune_NER_task/generate_bpe_dataset.py` → 2; consumer-untouched: `git diff ff95ef7..HEAD -- example/notebooks/finetune_NER_task/generate_bpe_dataset.py \| grep ^@@` → exactly 3 hunks (header seam / adapter block / 1-line call-site swap — nothing below the swap); notebook probe (16-02 Task-2 verify, cells 0/1/2/8 tokens) → NOTEBOOK SEAM CHECK OK | GREEN |
| SC2 | three gates platform-aware; monkeypatched Windows-branch contract tests; POSIX unchanged | `.venv/bin/python -m pytest tests/examples/test_script_execution.py::TestScriptGateContracts tests/examples/test_notebook_execution.py::TestActiveNotebookGates -q --no-header` → 5 passed; gate structure assertions inside the dispatch-direction suite green (guard-precedes-probe x2, registration untouched, skip-machinery ban); byte-equivalence hunk maps: `git diff ff95ef7..HEAD -- tests/examples/test_script_execution.py tests/examples/test_notebook_execution.py` → hunks ONLY in the gate regions (`_gate_script` @~130/`TestScriptGateContracts` @~152-174; `_gate_pybedtools` @~316-332 + the inventoried :1397 gate region; probe + typed skip + `_ACTIVE_NOTEBOOK_GATES` registration text-pinned untouched) | GREEN |
| SC3 | frozen pairs + pinned tolerances on both paths; selection.md single truth source | `.venv/bin/python -m pytest tests/test_frozen_pairs.py -q --no-header` → 4 passed; band-row probe (16-03 Task-2 corrected section isolation + both-direction collision) → `BAND ROW OK: \| Anno pooled exon-F1 (frozen pair) \| = 692/920 \| [0.752173912, 0.752173914] \| 0.7522 \|` (5 data rows, ≥6-decimal endpoints, all 4 existing labels single-hit); tight-band-consumed-here-only: `grep -rln "frozen pair" tests/ --include=*.py` → `tests/test_frozen_pairs.py` only; nightly assertions unchanged: showcase diff vs ff95ef7 = 2 hunks (`import sys` @34, gate guard @520-521) — the four drift bands + evidence key below the gate byte-preserved | GREEN |
| SC4 | dispatch-direction structure test same-change + Linux nightly re-execution holds the parsed bands | Structure test: `tests/test_dispatch_direction.py` 19 passed locally, AND all 19 PASSED in coverage-nightly's full census on the dispatched ref — incl. `test_ner_script_win32_rows_match_live_pybedtools_loj` (live truth-chain parity EXECUTED green on CI). Nightly re-execution (run 38441258701, section A.4): census hard gate `216/217 tests collected (1 deselected)` HELD; CRE jaccard band re-executed GREEN (`test_cre_notebook_executes_within_selection_bands` 254.961s, example-nightly); NER notebook end-to-end GREEN on POSIX (4023.932s); Anno exon_f1 band GREEN via the frozen-pair suite parsing the same selection.md row (`test_frozen_anno_pair_exon_counts_rational_and_tight_band PASSED`, coverage-nightly 2607); 5 skips = exactly the historical allowlisted set. **Caveat:** the two rice.uga.edu genome-download lanes red (4th/5th manifestations of the owner-accepted phase-14 class; failures sit in `_seed_rice_input`/genome-download, upstream of all phase-16 code — the BPE script dispatch never executed) — owner-directed accept-with-caveat per A.5 | GREEN (held where executable; rice-class caveat recorded) |
| SC5 | every example edit lands with its docs mirror byte-identical | `.venv/bin/python scripts/check_docs_sync.py` → `OK: docs/example/ is in sync with example/` exit 0; same-commit discipline (4/4, one line per example edit): `d05a3b9` CRE notebook + twin · `3ffa658` NER script + twin · `87b0c13` NER notebook + twin · `92ac413` selection.md + twin — every phase commit that touched an example file touched its docs twin in the same commit | GREEN |

**Matrix disposition: 5/5 — SC1/2/3/5 green outright; SC4 green where executable, with the rice.uga.edu caveat recorded in section A.4/A.5 (two genome-download lanes red on the owner-accepted external-host class, zero phase-16 regression signal; every phase-16 surface green in the run of record).**

---

## D. Phase-wide honesty ledger (guards — all green, measured this session)

- **Census triple byte-identical to phase entry:** `grep -oE '[0-9]+/[0-9]+ tests collected \([0-9]+ deselected\)' .github/workflows/ci.yml | head -1` → `216/217 tests collected (1 deselected)`; and `git diff --stat ff95ef7..HEAD -- .github/workflows/ci.yml` → EMPTY (ff95ef7 = phase-16 planning baseline, the last commit before any phase-16 execution code; the pinned range is exact).
- **expected_skips.yaml untouched:** `git diff --stat ff95ef7..HEAD -- tests/expected_skips.yaml` → EMPTY.
- **pyproject.toml untouched:** `git diff --stat ff95ef7..HEAD -- pyproject.toml` → EMPTY.
- **uv.lock untouched (recorded beyond the plan's guard):** `git diff --stat ff95ef7..HEAD -- uv.lock` → EMPTY — the no-new-packages / standard-stack claim (T-16-SC) is evidence-backed, not asserted.
- **Full fast lane:** `.venv/bin/python -m pytest -m "not slow" --no-cov -q` → `2550 passed, 1 skipped, 75 deselected in 102.93s` exit 0 (the 1 skip is the pre-existing allowlisted `test_notebook_imports[predict_data.ipynb]` content skip).
- **test-windows eligibility of both new root files re-confirmed:** `grep -c "pytest.mark" tests/test_dispatch_direction.py tests/test_frozen_pairs.py` → 0 / 0 (unmarked — collected and run on every leg incl. test-windows); no top-level pybedtools/subprocess dependency (the one `import pybedtools` at tests/test_dispatch_direction.py:479 is INSIDE the live-parity test, behind `find_spec` + `shutil.which` early-`return` — conditional contract-strengthening, not skip machinery).
- **Phase footprint (23 non-planning files over ff95ef7..HEAD):** 4 example files (CRE nb cell 11, NER script, NER nb cells 0/1/2/8, selection.md +1 row) + 4 docs mirrors + 4 gate/harness test files + 8 fixture files (6 BED/tsv pairs + PROVENANCE.md) + 2 tests/ root suites + .gitignore (the 16-03 Rule-3 scoped `!tests/fixtures/genomic_intervals/*.tsv` exception, already documented in 16-03-SUMMARY). Matches TRUTH-4's enumeration exactly — nothing else moved.

---

## E. Planning-assumption dispositions

| Assumption | Disposition |
|------------|-------------|
| 16-03 AS-1 (drift-honest encoding; anchor branch) | **Confirmed** — anchor branch ACTIVE: the regeneration reproduced ALL historical anchors exactly (jaccard 0.32472687098490116 / 15872 / 48878; counts 526/394/346/48/180; exon_f1 692/920; genes_above_floor 59); historical counts pinned as baked literals (16-03 PROVENANCE.md drift table) |
| 16-03 AS-2 (all pairs from the selection.md pipeline; no rice.uga.edu) | **Confirmed** — loj pair built from same-session CRE peaks (A) + truth DHS rows (B); zero rice.uga.edu access in 16-03 |
| 16-03 AS-3 (no empty-side pair committed) | **Confirmed** — real non-empty pairs (51/94, 394/526, 51/94); empty-side semantics stay oracle-pinned in phase-15's corpus tests |
| 16-03 AS-4 (band centers on the measured value) | **Confirmed** — [0.752173912, 0.752173914] (~2e-9 wide) centered on the measured 692/920 rational; no departure from the historical 0.7522 anchor (PROVENANCE.md records the zero-delta) |
| 16-03 AS-5 (distinct band label, OQ1) | **Confirmed** — 'Anno pooled exon-F1 (frozen pair)' collision-probed both directions (section C SC3 row, re-run green) |
| 16-04 AS-1 (manual dispatch is the SC4 verification step; scheduled runs additional) | **Confirmed-active** — the dispatch fired BOTH nightlies on the phase ref (one dispatch = full evidence source); the two queued scheduled runs are additional-but-blocking queue latency, not primary evidence |
| 16-04 AS-2 (runner ops; recovery manual applies if down) | **Confirmed with adjustment** — runner was ONLINE (no recovery needed); the adjustment: the single runner carried an in-flight foreign job + ~6h queued scheduled work, and cancelling foreign runs was denied by the permission classifier (A.3), so the SC4 evidence landed after the dispatching window (PENDING state reconciled 2026-10-14 via the async-jobs manifest; terminal record section A.4) |

---

*Phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity · 16-04 · SC4 record status: CONCLUDED (run 38441258701) — every phase-16 surface green on the dispatched ref; rice.uga.edu caveat on two non-phase input-download lanes recorded in A.4/A.5.*
