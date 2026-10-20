---
gsd_state_version: "1.0"
milestone: v1.3
status: Phase 21 complete
stopped_at: Phase 21 complete (ty hard gate CI-green; awaiting owner calls on pinned-stub-env + nightly resume)
last_updated: "2026-10-20T06:35:32.924Z"
last_activity: 2026-10-20
last_activity_desc: Phase 21 complete — ty hard flip CI-validated, mypy retired tree-wide
state_head: e60303f1c5073aed15e90c70cbfbe0666b5efcee
milestone_name: Platform Reach & Packaging Honesty
current_phase: 21
current_phase_name: Type-checker atomic swap — ty hard flip + mypy retirement
progress:
  total_phases: 9
  completed_phases: 8
  total_plans: 22
  completed_plans: 22
  percent: 89
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-11)

**Core value:** Make platform coverage and the distribution chain honest end-to-end — Windows-native execution parity, accurate packaging/install guidance, a rust-proof publish chain, a docs surface that matches the code — with the suite, coverage gate, and CI honest-green throughout.
**Current focus:** Phase 21 — Type-checker atomic swap: clear the 232 ty diagnostics to zero, then flip CI/pre-commit/check_code to ty and retire mypy in one atomic change (D-09)

## Current Position

Phase: Milestone v1.3 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-10-20 — Completed quick task 261020-gk4: TUI opportunity survey for v1.4 (6 ranked / 7 excluded vs dnallmmark; report .planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md, commit 3028f0c)

## Performance Metrics

**Velocity:**
- v1: 13 plans / 3 days · v1.1: 24 plans / 6 days · v1.2: 12 plans / 2 days
- Recent trend: Stable

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 13 P01 | 17 min | 3 tasks | 6 files |
| Phase 14 P01 | 14 min | 3 tasks | 2 files |
| Phase 14 P02 | 228 min | 3 tasks | 5 files |
| Phase 15 P01 | 26 min | 3 tasks | 3 files |
| Phase 16 P01 | 13 min | 2 tasks | 4 files |
| Phase 16 P03 | 42 min | 3 tasks | 12 files |
| Phase 16 P02 | 21 min | 3 tasks | 8 files |
| Phase 16 P04 | 70 min active (16h wall incl. CI run 38441258701) | 2 tasks | 2 files |
| Phase 17 P01 | 10 min | 2 tasks | 2 files |
| Phase 17 P02 | 21 min | 3 tasks | 3 files |
| Phase 18 P01 | 65 min | 3 tasks | 42 files |
| Phase 18 P02 | 24 min | 3 tasks | 6 files |
| Phase 19 P01 | 9 min | 2 tasks | 7 files |
| Phase 19 P02 | 20 min | 3 tasks | 31 files |
| Phase 19 P03 | 21 min | 3 tasks | 12 files |
| Phase 19 P04 | 27 min | 3 tasks | 18 files |
| Phase 19 P05 | 23 min | 3 tasks | 18 files |
| Phase 19 P06 | 22 min | 3 tasks | 9 files |
| Phase 19 P07 | 260 min active (~34h wall incl. nightly runs 38678158479 + 38695301631) | 3 tasks | 18 files |

## Accumulated Context

### Roadmap Evolution

- Phase 20 added (2026-10-15, post-audit option B): tech-debt closeout — CLI flag-only invocation crash (deferred-items #1) + residual docs mentions (faq legacy [cuda]/[rocm], task_guides hf route prose); owner human-gates #25/#27/#26 stay in the WINDOWS ledger outside this phase

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Roadmap decisions (2026-10-11):

- Roadmap: 7 phases (13–19) matching the research-converged ordering — O ollama sweep, G GAP-1, W-core evaluators, W-rollout dispatch/gates, S publish chain, P atomic extras surgery, D docs sweep last. Coarse-granularity compression already applied at the category level (WPAR split 2-not-4; PUBS+PKGH-08; PKGH+WPAR-07; DOCR+PKGH-05) — the five hard ordering constraints (census re-pin once, test_script_execution.py shared-file sequencing, evaluators-before-dispatch, smoke-before-extras, docs-last) fix the count at 7
- Cross-category mappings: PKGH-08 (mypy cap) → Phase 17 publish chain (the mypy failure class surfaced in publish run 38109158525); WPAR-07 (triton-windows) → Phase 18 atomic extras surgery (Pitfall 12: no half-changed extras surfaces); PKGH-05 (cuda130 default) → Phase 19 docs sweep (owner directive; describes the final extras state)
- Phase 17's FIRST act is the publish.yml upload guard/split — verified unguarded `workflow_dispatch`→`twine upload` today; it must land before any smoke trigger exists (PyPI filenames are immutable)
- Phase 13 executes the owner's 2026-10-11 ollama directive (all ollama pytest surface removed or skip-marked, per-assertion remove-vs-skip at planning time); WINDOWS #13 (mcp_example notebook execution) stays excluded — notebooks remain in-repo
- Linux truth chain stays bedtools-computed (no re-freeze); selection.md is the single truth source parsed on both platforms; tolerance lives in the comparison layer only
- Research flags for plan time: Phase 16 and Phase 17 warrant `--research-phase` (Phase 16: test-windows subset + time budget, tolerance-pin mechanics, marker registration, exon_f1 band pinning; Phase 17: `uv publish --dry-run` existence conflict vs installed uv 0.12.21, gh-action-pypi-publish input syntax); Phase 18 carries owner calls at discuss/plan time (base rename target + alias-or-break, rocm remove-vs-restore, triton-windows declare-vs-document)
- Commits carry no Co-Authored-By trailers (repo convention); one census-moving edit per phase (D-03 triple + expected_skips.yaml + ci.yml stages same-change)
- [Phase 13]: Skip disposition realized by CI shape: example-nightly stage-3 ollama batch deleted so the mcp pair typed-skips via the existing self-healing D-13 gate (network-unavailable: with probe evidence) — zero functional test-file edits (13-01, D-04)
- [Phase 13]: Census triple re-pinned 208/217 (9 deselected) to 216/217 (1 deselected) with the -k mcp deselect dropped from stages 0.5 and 1 — measured with the exact stage-0.5 flags, grep never widened (13-01)
- [Phase 13]: expected_skips.yaml touched comment-only: the pre-existing network-unavailable: prefix entry is the sole sanctioned match path for the nightly pair skips (13-01)
- [Phase 14]: GAPC-01 ladder measured at zero new rungs on the recorded run — promoter + BPE-leg both load and forward green on transformers 5.17 under the existing ladder; invert_attention_mask proven a dead branch via reachability (attribute inventory, not execution trace) (14-01)
- [Phase 14]: Owner disposition shim-zero-rung taken exactly once on the 14-EVIDENCE.md section F table via the D-04 blocking-human checkpoint — proceed to 14-02; WINDOWS #11/#12/#15 close fixed-with-probe-evidence (D-08) (14-01)
- [Phase 15]: Spike cores ported verbatim as specification (5008-case bedtools v2.31.1 parity record is the design authority); case-11 plan prose corrected against the oracle — NER driver is all-hits, null vector pinned in case 7 + pin table (15-01)
- [Phase 15]: Coverage proof at 100% via coverage-CLI form only (WINDOWS #20); module row visible = omit-list proof; baked corpus subset = one full 12-cycle per spike (the complete pattern x shuffle surface); census triple untouched — zero ci.yml/pyproject/expected_skips edits (15-01)
- [Phase 16]: [16-01] Platform dispatch pinned as the sys.platform == "win32" equality check at the seam only (CRE cell 11; inverse guard on the showcase gate) — POSIX branch byte-preserved, no try/except-ImportError ladder (WPAR-06 shape, now structure-tested)
- [Phase 16]: [16-01] Dispatch-direction runtime contracts live at tests/ root in test_dispatch_direction.py (monkeypatched platform + poisoned subprocess.run) — zero census movement, ci.yml triple untouched at 216/217
- [Phase 16]: [16-03] Frozen-pair regeneration reproduced ALL historical anchors exactly (jaccard 0.32472687098490116, 15872/48878, counts 526/394/346/48/180, exon_f1 692/920) — AS-1 anchor branch active, historical counts pinned as baked literals in the frozen suite
- [Phase 16]: [16-03] Tight exon_f1 band [0.752173912, 0.752173914] pinned in selection.md; consumed by tests/test_frozen_pairs.py ONLY (nightly keeps the four drift bands + evidence key — Pitfall 6)
- [Phase 16]: [16-03] Scoped .gitignore exception !tests/fixtures/genomic_intervals/*.tsv (Rule 3 — *.tsv line ignores the loj oracle; committed-fixtures-by-exception pattern)
- [Phase 16]: [16-02] NER dispatch = dual-truth parity: the always-on frozen 12-column literals (bedtools v2.31.1-measured, pre-verified against live pybedtools) carry the contract on every leg; the live pybedtools loj comparison runs wherever the truth chain is installed (find_spec + shutil.which gate) — no skip machinery, file stays platform-neutral for test-windows
- [Phase 16]: [16-02] Cell 8's inline adapter composed from the script's ast-extracted segments (identity by construction) and pinned by the whitespace-normalized textual-identity test (AS-1); notebook JSON round-trip needs ensure_ascii=False — the committed file stores raw-UTF-8 outputs, the default rewrote ~800 unrelated lines
- [Phase 16]: [16-02] Two Rule 3 same-change enablers: _strip_magic_lines substitutes magics with same-indent pass placeholders (guarded magics left bodyless branches after deletion — false syntax errors), and the four gate-contract tests pin sys.platform=linux so the win32 early-return cannot shadow the probe contracts on test-windows
- [Phase 16]: [16-04] SC4 concluded green-where-executable with owner-accepted rice.uga.edu caveat (run 38441258701 over the complete change): live-pybedtools parity executed green on CI, CRE band re-executed green, NER notebook green end-to-end, root suites 19+4 green in the full census, D-03 gate held 216/217 byte-identical; the 2 reds sit upstream of all phase code (4th/5th manifestations, phase-14 H.4/H.4.final precedent)
- [Phase 16]: [16-04] Phase-16 honesty ledger clean at close: census triple + expected_skips.yaml + pyproject.toml + uv.lock byte-identical to ff95ef7 across the whole phase; footprint = example seams/mirrors/gates/fixtures/2 root suites/.gitignore-tsv-exception only
- [Phase 17]: [17-02] WINDOWS #20 closed via PUBS-05's sanctioned coverage-CLI fallback: the live numpy==2.2.0 pin reprobe cleared the numpy facet but the torch overrides facet survived (independent, not a symptom) — numpy restored 2.5.3, bare --cov + coverage run forms are THE fix (both proven green)
- [Phase 17]: [17-02] mypy dev-extra cap >=1.15.0,<2.5 (one line, numstat-proven): 2.4.0 admitted (observed/tolerated publish-runner resolution), 2.5.0 excluded — the resolver class is frozen, not silenced
- [Phase 17]: [17-02] any future local numpy pin ceiling is <2.4 (2.4-series one-init guard) AND no numpy pin alone fixes the torch facet — both facts live-proven and recorded in WINDOWS row 20
- [Phase 18]: [18-01] Atomic extras surgery landed as ONE commit (399f9f1): base→full rename with one-cycle alias base=[dnallm[full]] (remove v1.4), cuda121 retired (cu121 index frozen torch 2.5.1), rocm removed (rocm7.1 restore path), triton-windows Windows-marker in fla; residue gates hold 2+3 verbatim sanctioned lines; census 216/217 untouched — 18-CONTEXT owner-locked decisions; pickaxe 3-way proof + smoke run 38578071738 publish-skipped/smoke-success
- [Phase 18]: [18-01] Pip-side alias full-closure resolve deferred (unrun-verify, WINDOWS ledger): pip urllib3 stalls against this box's network (5 attempts) while uv/gh/curl fine; mechanism probe-proven (pip 26.2.1) + METADATA alias line + guard + uv parity carry the evidence — box-network artifact, not repo defect; fix-attempt limit reached; exact re-run command recorded
- [Phase 19]: 19-01 strict-clean: 15 mkdocs warnings to zero via hand-edited page-relative See-Also tails + GitHub-URL precedent for the docs escape; anchors always confirmed against built-site id= (mkdocs slugify drops em-dashes); ledger rows same-commit as fixes
- [Phase 19]: :457 (Scenario 6 mamba combo) flipped to .[cuda130,mamba] — plan's count arithmetic (5 survivors + kept table row = band ceiling 6) overrode its 'keep' parenthetical; older-GPU lanes demoted to spaced 'CUDA 12.4-12.8' prose + matrix pointer where install-cmd gates forbid literal cuda124 commands
- [Phase 19]: [19-03] Model-name lock-alignment standard: docs examples citing models with no in-repo evidence (lgq12697/* modelscope ids) re-pointed to the lock-pinned/executed artifacts (kuleshov-group/PlantCAD2 + plantcad adapter, arcinstitute/evo2_1b_base, benchmark_config.yaml trio); names with no lock row and no contradiction left alone
- [Phase 19]: [19-03] train()/evaluate() metric contract enforced in docs: DNATrainer.train() returns HF train-loop metrics only; eval_* keys come exclusively from evaluate() — tutorial Task 2/3 prints fixed accordingly (eval_hamming_loss never existed)
- [Phase 19]: [19-04] Flag-only CLI invocation paths (dnallm-train/dnallm-inference without --config) crash with KeyError 'task' — documented as --config-required in docs, code bug routed via deferred-items.md #1 (out of sweep scope)
- [Phase 19]: [19-04] Lock-route alignment extended to runnable examples: 4 source="huggingface" sites aligned to lock-pinned modelscope routes; catalog/prose mentions stay per 19-03's distinction (task_guides.md flag noted in deferred-items.md #2)
- [Phase 19]: [19-04] Non-.md-link blind spot recorded: mkdocs --strict does not validate bare directory/URL-shaped links — docs/index.md docs/-prefix links and api/index.md models/special/ link had 404ed invisibly; sweep must check non-.md links against the built site tree
- [Phase 19]: [19-04] Fabricated support surfaces removed: lone repo-wide discord.gg link dropped (T-19-06 domain rule); landing-page '200+ test cases' moved to the 2556-based '2,500+' README parity
- [Phase 19]: :255 Scenario-6 pointer flipped to .[cuda130,mamba] (19-06): the pointer example must name the combination the scenario demonstrates — 19-02's survivor sanction was for its own count gates; this batch owns post-re-point coherence
- [Phase 19]: Vacuous-verification pattern (19-06): mamba verify commands tested mambapy (core) — a check that passes in every env proves nothing about the extra; both now import mamba-ssm, the extra's member
- [Phase 19]: Retired-extras vocabulary swept from wrapper prose (19-06): 'base/inference/benchmark extras' phrasing named extras that do not exist while the commands show .[full,cuda130] — prose may only name extras on the final surface
- [Phase 19]: [19-07] Nightly interpreter honesty (fix bdf6ba3, run 38678158479 repo-caused red): self-hosted nightly venvs pinned to setup-python's exact toolcache interpreter (`uv venv --python "$pythonLocation/bin/python"`) + mamba wheelhouse cache key py-tagged — closes the unpinned-uv-venv drift class (uv-managed 3.11 since 10-07, 3.13 since 10-14 silently replaced the labeled 3.12/3.11); disclosure recorded in 19-EVIDENCE.md; re-dispatched trio 38695301631 all green

### Pending Todos

Carried open ledger items from the v1.1/v1.2 handoffs (owner-scoped, outside v1.3 requirements): delete unreferenced doc_mocks.py entries; README:345 + server.py:97 `config/` path pattern; quick_start "NER" task-type line; MCP client `_call_tool` naming mismatch.

### Blockers/Concerns

Carried runner-box ops items (cache-quota, `$HOME` cleanup; ollama loopback cadence re-evaluated by Phase 13's disposition) live in `milestones/v1.1-MILESTONE-AUDIT.md`.
- RESOLVED 2026-10-14: 16-04 SC4 evidence concluded (run 38441258701 terminal record in 16-EVIDENCE.md A.4; WINDOWS #24 fixed) — green-where-executable with owner-accepted rice.uga.edu caveat; standing external note: rice genome downloads (NER input path) remain fragile, upstream of this repo

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261017-414 | scan all files for mypy remnants and systematically replace with ty (D-09 flip follow-up) | 2026-10-17 | 1c52e46 | [261017-414-scan-all-files-for-mypy-remnants-and-sys](./quick/261017-414-scan-all-files-for-mypy-remnants-and-sys/) |
| 261019-cr6 | Windows local CI replication + KDA declaration proof | 2026-10-19 | 6d14d9d | [261019-cr6-windows-local-ci-replication-kda-declara](./quick/261019-cr6-windows-local-ci-replication-kda-declara/) |
| 261020-gk4 | TUI opportunity survey for v1.4 planning (6 ranked / 7 excluded vs dnallmmark; analysis-only) | 2026-10-20 | 3028f0c | [261020-gk4-tui-dnallmmark-home-forrest-github-dnall](./quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/) |
| 4 | Append Part 5 addendum to 261020-gk4 TUI survey (pre-flight cluster + merge-or-build verdicts per owner directives 2026-10-20) | 2026-10-20 | e60303f | — |
*Rows archived 2026-10-16: 6 late v1.2 tasks -> milestones/v1.2-quick/ (README-v1.2-late-additions.md); 2 v1.3 spikes -> milestones/v1.3-quick/ (README.md)*

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first. The six 2026-10-11 post-close rows (cuda130 default + GPU matrix, OIDC migration, docs accuracy sweep, extras hygiene, publish smoke, mypy cap, mkdocs strict, #20 cov crash) were promoted into v1.3 active scope as PKGH/PUBS/DOCR/WPAR requirement IDs — see REQUIREMENTS.md.

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | MOTIF-01 golden fixture — HBG1/BCL11A Fig 4a owner inputs (issue #44; activation fixture-files-only) — now FUTR-04 | owner-deferred (option B) | 2026-10-10 | v1.2 |
| deferred_items | numpy 2.5.x cannot be instrumented by coverage on py3.13 (numpy ceiling call when the 1.26.4 leg retires) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | old-terminology prose hits outside check surface (pyproject description, 3 test docstrings, ui/ default, root cli/ shims — example/ configs now fall under DOCR-01) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | uat_gaps carried from v1.1 close (1 item) | acknowledged (prior close) | 2026-10-07 | v1.1 |
| deferred_items | 05: `uv run pytest` resolver failure under uv 0.12.20 (use `--no-sync`) | acknowledged | 2026-10-06 | v1.1 |
| deferred_items | 03: import-time `logs/dnallm.log` sink recreated under pytest cwd | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03: test_timeout.py fixed ~60s cost | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` — RESOLVED: fully implemented at data.py:1013 (verified 2026-10-15 during 19-03 docs sweep; was stale-recorded as a no-op) | resolved | 2026-10-15 | v1 |

## Session Continuity

Last session: 2026-10-20 (quick task 261020-gk4 via /gsd-quick)
Stopped at: Completed quick task 261020-gk4 — TUI opportunity survey (report: .planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md, commit 3028f0c); v1.4 TUI phases can be scoped from Part 3
Resume file: .planning/HANDOFF.json (one-shot) + .planning/.continue-here.md (paused_between_milestones); branch v1.4 tip 3028f0c

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
