---
gsd_state_version: "1.0"
milestone: v1.1
milestone_name: Example Execution Testing & Repair
current_phase: 09
status: completed
stopped_at: Phase 09 complete — all phases complete
last_updated: "2026-10-06T16:36:00.000Z"
last_activity: 2026-10-06
last_activity_desc: Phase 09 complete
state_head: a949b96c385788d655a4cf1b9f01eb539c4ef398
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 24
  completed_plans: 24
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-05)

**Core value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.
**Current focus:** Phase 09 — CI Wiring & Census Verification

## Current Position

Phase: 09
Plan: Not started
Status: All phases complete
Last activity: 2026-10-06 — Phase 09 complete
documented dependency (fla extra >=0.5.2,<0.6 in all + README + docs FAQ + 3 guard tests,
commit 2259573); owner-upgraded B+ decision landed in-phase before tail gates

Progress: [████████████████████] 20/20 plans ([██████████] 100% of planned; Phase 9 TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 24 (v1.1 Phase 05; v1 plans archived with the milestone)
- Average duration: ~53 min (Phase 05: 320 min across 6 plans)
- Total execution time: ~9.1 hours (v1) + ~5.3 hours (v1.1 Phase 05)

**By Phase (v1.1):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 05 | 6 | 320 min | ~53 min |
| 06 | 3 | - | - |
| 7 | 2 | - | - |
| 8 | 9 | - | - |
| 09 | 4 | - | - |

**Recent Trend:**
- Last 5 plans (v1 close): 44, 27, 52, 51, 39 min
- Trend: Stable

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 05 P01 | 11 min | 3 tasks | 5 files |
| Phase 05 P02 | 12 min | 3 tasks | 15 files |
| Phase 05 P03 | 94 min | 3 tasks | 12 files |
| Phase 05-04 P04 | 23 min | 2 tasks | 3 files |
| Phase 05-05 P05 | 29 min | 3 tasks | 8 files |
| Phase 05-06 P06 | 151min | 3 tasks | 5 files |
| Phase 06 P01 | 19 min | 2 tasks | 4 files |
| Phase 06 P02 | 16 min | 2 tasks | 3 files |
| Phase 06 P03 | 125 min | 3 tasks | 11 files |
| Phase 07 P01 | 38 min | 3 tasks | 19 files |
| Phase 07 P02 | 56 min | 3 tasks | 6 files |
| Phase 08 P01 | 97 min | 3 tasks | 4 files |
| Phase 08 P02 | 7h 27min | 3 tasks | 7 files |
| Phase 08 P03 | 20min | 3 tasks | 9 files |
| Phase 08 P04 | 115 min | 3 tasks | 10 files |
| Phase 08 P05 | 57 min | 2 tasks | 8 files |
| Phase 08 P06 | 20 min | 2 tasks | 5 files |
| Phase 08 P07 | 157 min | 2 tasks | 2 files |
| Phase 08 P08 | ~13h | 3 tasks | 5 files |
| Phase 08 P08 | ~13h | 3 tasks | 5 files |
| Phase 08 P09 | 3h 22min | 3 tasks | 3 files |
| Phase 09 P01 | 12 min | 3 tasks | 5 files |
| Phase 09 P03 | 15 min | 2 tasks | 3 files |
| Phase 09 P02 | 38 min | 2 tasks | 6 files |
| Phase 09 P04 | ~21h (4 runner cycles) | 3 tasks | 6 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work (v1.1 roadmap):

- Roadmap: 5 phases numbered 5–9 (continues v1, which ended at Phase 4); test-layer vertical (5, 8, 9) and example vertical (6, 7) touch disjoint files — 5 and 6 parallelizable
- Roadmap: harness + honest gates first (research consensus) — WR-08/09 and the docs-mirror drift closure land together in Phase 5 so every later repair rides an enforced lane
- Roadmap: truth agreement asserted as calibrated floors with tolerance bands, never exact outputs (transformers 4.49–5.x span)
- Roadmap: GB10 feasibility verdicts (evo-1/evo2/megaDNA/pyBigWig) precede execution-test authoring; `environment-unavailable:` typed skips only with recorded evidence
- Roadmap: separate example-execution nightly job pre-authorized by owner if total runtime exceeds the 900-min nightly (CI-06)
- Carried from v1 (AUDIT-04): kernel subprocesses are unmeasured by design — example execution must not move the 96.30% gate (documented in Phase 9)
- [Phase 05]: 05-01: nbclient 0.11.0 NotebookClient is not a context manager - harness uses plain execute() with shutdown_kernel=immediate; env overrides via os.environ save/restore (no env trait)
- [Phase 05]: 05-01: typed-skip prefixes environment-unavailable:/optional-dep: registered in expected_skips.yaml with zero callers - Phase 8 skip decisions inherit the allowlist contract
- [Phase 05]: 05-02: DOCS_ONLY_SUFFIXES relaxation scoped to right_only only - a both-sides .md (overview.md) must still match byte-for-byte, proven by injected-drift failure
- [Phase 05]: 05-02: docs-validation flipped honest only after all five workflow commands verified green locally (born green, D-01); mcp extra installed and README install line run verbatim before documenting
- [Phase 05]: 05-02: pre-existing uv-run resolver failure (mamba x cuda conflicts matrix, pyproject unchanged) logged to deferred-items.md instead of fixing - out of 05-02 scope
- [Phase 05]: evo-1 verdict FEASIBLE(small-variant): 131k remote code needs rotary_emb.pos_idx_in_fp32 (absent from every transformers >=4.49) but evo-1-8k-base runs end-to-end - Phase 8 executes the 8k variant and updates the notebook reference per D-06
- [Phase 05]: evo2 FEASIBLE(notebook-variant) only via the noFP8 config on GB10 - the FP8 auto-selection trap fired live (1b tier requires Transformer Engine; the empty TE meta package must stay absent because its RuntimeError escapes vortex's ImportError guard)
- [Phase 05]: pyBigWig environment-unavailable: default sdist build fails on the stock box (curl-config present, headers off the include path); CFLAGS deviation builds+round-trips green but a dev-extra line cannot encode it - pyproject untouched
- [Phase 05]: marimo flavor: export-html for Phase 8 (deterministic exit + HTML artifact); script-mode also terminates cleanly, binds no port (A4 resolved empirically)
- [Phase 05]: 05-04: D-07 ladder terminated at its designed rung - remote NT code needs removed 4.x PretrainedConfig defaults (is_decoder/add_cross_attention), not vendored-pure-helper territory; shim kept for the import fix, smoke = evidence-backed typed skip, benchmark notebook flagged census FAIL (REPAIR-03 stays PARTIAL, owner disposition per D-09)
- [Phase 05]: 05-04: native-ESM route (trust_remote_code=False) probed and refuted - FFN weight-shape mismatch (ckpt 4096x512 vs config.json 2048x512); remote esm_config.py is load-bearing, so no drop-in transformers-5 fix exists for this checkpoint
- [Phase 05]: 05-05: assert_tree_clean converted to delta-zero vs import-time baseline (owner's live IDE churn in tracked notebooks is not harness business; clean-checkout behavior identical to the original absolute check)
- [Phase 05]: 05-05: generate_bpe_dataset.py standalone defect (reads rice_annotation.bed it never wrote) repaired verbatim from the notebook; docs mirror resynced
- [Phase 05]: 05-05: GAP-1-class gap extends to plant-nucleotide-transformer-BPE (NER script + notebook; same EsmConfig.is_decoder rung after the import shim) — 05-04 ladder honored, script lane = self-healing typed skip; one owner disposition now covers three census items (D-09 hand-off)
- [Phase 05]: 05-06 census complete: 25/25 example items executed — 11 PASS / 12 FAIL (class-tagged repair queue: NT-REMOTE-STRUCTURAL x3-behind-one-disposition, BPE-TOKENIZER upstream artifacts, OTHER incl. benchmark.py:296 labels bug) / 2 deferred-owner (ollama probe green; Phase-8 plan required)
- [Phase 05]: 05-06 durable rollout: ACTIVE_NOTEBOOKS x8 (two real trainings included), 7 probe-then-execute gated tests with honest typed skips (mcp pair skips on the genuinely-down MCP endpoint with ollama-GREEN evidence in-message; both-up state fails loudly per T-05-16), 3 marimo apps; full tests/examples 107 passed/9 audit-matched skips in 47:26
- [quick 261003-csd]: D-08 closed — mcp client pair moved to the owner-approved EXECUTE state (T-05-16 sentinel retired 2026-10-03): both-up executes, any-down typed-skips with both live probe results, proven in both directions; langchain notebook runs under isolated kernelspec dnallm-mcp-langchain (VIRTUAL_ENV pinned to .scratch throwaway venv — project venv provably untouched)
- [quick 261003-csd]: two real MCP serving bugs fixed with same-change tests: single-flight inference (concurrent DataLoader forks + filelock = fork-unsafe deadlock; every multi-model predict used to time out) and dna_interpret mamba guard (captum backward on DNAMamba SIGKILLs the whole server, exit 137 repro); also discovered CLI --host/--port are dead flags (yaml always wins, deferred-items.md)
- [Phase 05 close 2026-10-03]: stale-digest re-verification passed 25/25 at 80b40a5 — NT smoke real-green (se3+sl7 shims), WR-04 rice network lane executed by the verifier; incremental review of the quick-task delta recorded 1C/1W/3I open (CR-01: single-flight asyncio lock releases on timeout cancellation → concurrent infer_seqs possible)
- [Phase 06]: 06-01: PlantHelixSeek label order frozen from upstream train_token_cls.py:78-96 (checkpoint configs carry none — Anno id2label is LABEL_i placeholders, CRE absent; re-proven live by the freeze run)
- [Phase 06]: 06-01: smoke-test forwards run under torch.no_grad() — autograd over the 8192 bp Anno eager-attention window OOM-kills the box; research memory budgets were measured no-grad
- [Phase 06]: 06-01: one-shot freeze tooling lives uncommitted in example/notebooks/plant_helixseek_shared/.scratch/ (owner constraint); only outcome artifacts committed (yaml + 2 tests + scratch-home .gitignore)
- [Phase 06]: [06-02] RED evidence via NotImplementedError stub committed in the test commit: suite fails at test level (RED_EVIDENCE_OK, 20 failed/exit 1) instead of collection-level ImportError (INVALID_RED per #3770)
- [Phase 06]: [06-02] normalize_chrom vocabulary: numeric token or organelle C/M after case-insensitive chr prefix, else bare digits; anything else raises (first draft over-accepted 'chromosome1' — caught by RED-authored tests)
- [Phase 06]: [06-02] fetch_sequence takes an open pyfastx.Fasta OR a path (function-local pyfastx import is load-bearing); half_open_to_gff1 rejects zero-width intervals — no 1-based closed form exists
- [Phase 06]: Owner decision B+ (06-03 Task-2 package checkpoint): flash-linear-attention 0.5.2 installed bare into .venv (no backend extra - may downgrade torch); 16-window probe through the unmodified dnallm route proved healthy DHS separation (0.7673 in-DHS vs 0.2230 non-DHS; dead fallback 0.0073/0.0087) - evidence in scratch fla-probe-0.5.2.md
- [Phase 06]: Owner upgraded the fla follow-up at 17:03 CST 2026-10-03 (no longer deferred): flash-linear-attention becomes a declared pyproject dependency + documented, landed as an in-phase quick task after 06-03; version direction 0.5.2, bounded range under discussion - 06-03 itself made no pyproject change
- [Phase 06]: 06-01 smoke tests validate shapes only - they passed with positionally-dead PlantHelixSeek outputs; value-level discrimination assertions (e.g. the DHS probe) are the CI follow-up class so silent semantic degradation is caught (06-03 checkpoint evidence)
- [Phase 06]: 06-03 selection: first-ranked tile Chr1:5100001-5300000 passed both floors on the first candidate (CRE jaccard 0.3247 >= 0.3; Anno pooled exon-F1 0.7522, 59/91 genes >= 0.8); intergenic negative asserts Anno genic fraction only (0.0000) - its CRE fraction 0.1200 is evidence-only, never jaccard-vs-empty
- [Phase 07]: 07-01: PlantHelixSeek truth GFF rows are already genomic — the FASTA-header offset applies to predictions only; found via the Pitfall-8 jaccard=0.0 signature while the local-coordinate flanking control reproduced 0.0325 exactly
- [Phase 07]: 07-01: showcase figures embed as compiled vega v6 object under application/vnd.vega.v6+json plus native vegalite v6 JSON string (nbformat rejects objects under bare .json mimes; altair 6.3's default and mimetype renderers both fail the D-10 requirement)
- [Phase 07]: 07-01: plant_helixseek_anno data dir mirrored in 07-01 (Pitfall 1 mirror-all-dirs + the plant_helixseek-scoped sync gate) ahead of 07-02's notebook; SHOW-07 denylist = every genome-wide line must carry the negated disclaimer
- [Phase 07]: 07-02: A4 closed — the B-L permutation was transcribed from the Phase-6 scratch select_loci.py (the code that produced the floors, citing upstream predict_genome_multigpu.py:97-101), cross-checked against 06-RESEARCH and BILOU semantics, and proven by exact reproduction of every selection.md value (exon_f1=0.7522, 59 genes, tp=346 fp=48 fn=180)
- [Phase 07]: 07-02: Anno nucleotide metrics are per-strand CDS base masks pooled across strands (evidence-only, unbanded); emitted GFF3 rows carry per-segment placeholder Parent ids since the frozen argmax decode does not group segments into genes
- [Phase 07]: 07-02: timeout-arithmetic hand-off for Phase 9 CI-07 — +2400s (CRE) + 5400s (Anno) marks push the coverage-nightly sum-of-ceilings comment past the 900-min job cap on paper (~970 min); D-14 budgets NOT shrunk, actuals ~15 min for both slow showcase tests; CI-06 pre-authorizes a separate example-execution nightly job if actuals overflow
- [Phase 08]: 08-01: pyBigWig aarch64 sdist links need LDFLAGS=-L$(sys.prefix)/lib in nightly installs (sysconfig bakes nonexistent /opt/hostedtoolcache); marimo exit-0 evidence = absence of .export.error.txt; example-nightly timeout 2700min sum-of-ceilings backstop
- [Phase 08]: D-17 executed shim-covered: 9-shim layer closes all NT rungs on a pristine snapshot; marker conversion stays dead fallback (08-02)
- [Phase 08]: ipython>=8.31,<9 pinned in notebook extra — pgt hard-caps matplotlib<3.9 so IPython 8 is the only kernel-plot lever (08-02)
- [Phase 08]: rice.uga.edu outage handled by census-cache seeding (cache-first dev box, cold cache keeps download+typed skip, 4xx re-raises WR-04)
- [Phase 08]: combined-notebook sibling gap fixed at harness seeding layer with a JSON-parsed ../ coverage contract test (08-02)
- [Phase 08]: 08-03: allow_patterns forwarding is conditional at BOTH layers (kwargs rebuilt per retry attempt) so exact-signature callers and the no-revision retry reset stay byte-identical
- [Phase 08]: 08-03: np.fromstring gate PROBES one binary-mode call instead of hasattr — numpy 2.x keeps the NAME as a raising ValueError stub, so presence gating would no-op on exactly the versions needing the shim
- [Phase 08]: 08-03: shim restores binary mode only (frombuffer semantics + writable copy); text mode refused with a loadtxt pointer; evo2 keeps its unfiltered 2.7GB fetch — only evo-1 passes the giants pattern set
- [Phase 08]: 08-03: np.fromstring gate PROBES one binary-mode call instead of hasattr (numpy 2.x raising stub)
- [Phase 08]: 08-04: DNATokenizer unknowns encode to id 1 (upstream encode_sequence rule) — checkpoint vocab stays six wide; None ids crashed tensor creation in _call_one
- [Phase 08]: 08-04: isolated megadna lane gates on a venv-targeted probe (cold = optional-dep skip, never auto-provision) so the runner rollout stays deliberate for 08-05/08-09
- [Phase 08]: 08-04: transformers 5.x no longer emits token_type_ids — MEGA-DNA column drop filters to present columns (4.49-5.x span repair, contract-tested)
- [Phase 08]: 08-04: three pre-existing mirror drifts resynced (Rule 3) to unblock the binary sync gate — check_notebook_md_sync 24/24 again
- [Phase 08]: [08-05] megaDNA sibling un-gating rides a reversible exact-version install into the project .venv (megadna @ cb2f5ab4 clone + MEGABYTE_pytorch==0.2.1) — the default-kernel pin is preserved; the installing finetune_generation keeps the isolated kernelspec (T-08-19)
- [Phase 08]: [08-05] demo-cell repair = pinned install cell immediately before the megaDNA demo load (census ImportError at megadna.py:146); generation_megaDNA's floating clone comment replaced by the same executable pinned form; D-21 stamps added, source= routes verified already D-15-aligned
- [Phase 08]: [08-05] content contracts check ACTIVE lines only — notebooks document alternative source= routes as comments; family lane 12 passed / 0 SKIPPED in 2885s, fast lane 1800/1, census megaDNA rows PASS (family CLOSED)
- [Phase 08]: [08-07] evo family CLOSED: 08-06's empty OPEN-RUNG ledger collapsed Task 1 to a confirmation run (5P/0S 49s; tests/utils 168P)
- [Phase 08]: [08-07] evo venv-only stack (flash_attn/stripedhyena/evo2) added to OPTIONAL_IMPORT_MODULES — 08-06's D-21 stamp cell failed the static import check in the project venv; fixed at the check's seam, not the notebook
- [Phase 08]: [08-07] A4/CI-05 verified at load time (offline load, evo-1 .pt-free, blobs unchanged 14,889 MiB); .pt-zero guarantee scoped to the evo-1 dir — evo2_1b_base.pt is its native checkpoint
- [Phase 08]: 08-08: lora+mcp families CLOSED on dev box (196P/1S census); LoRA-adapter repair at the spec-env seam (HF_ENDPOINT=hf-mirror.com, huggingface.co unreachable); D-13 retry probe + D-07 stage contract landed for 08-09 wiring
- [Phase 08]: 08-08: stage-boundary cleanup+assert discipline (owner directive): kill orphaned stage processes, verify port free, >=35Gi available before any heavy stage, before/after recorded in the census log

- [Phase 08 close 2026-10-05] Final census frozen: 196P/1S/0F identical across 08-08/08-09 (2:59:34); fast lane 1807P/1S zero new skips; verifier passed 8/8 must-haves + 11/11 reqs, 0 gaps (2e64237); example-nightly first full dispatch in flight
- [Phase 08 close 2026-10-05] Owner: models-cache layer dropped from CI — cold pulls accepted (65-min cold stage 1 proven vs 2700-min budget; lock-only cache 15.2GiB > 10GB quota and never actually saved); ci.yml edit lands in Phase 9
- [Phase 08 close 2026-10-05] Owner: hub cleanup executed — ~/.cache/huggingface/hub 84→19GB (evo-1-8k full-repo leftover 28GB + unreferenced legacy blobs 38GB removed; Qwen 19GB kept; giants tier + modelscope untouched)
- [Phase 08 close 2026-10-05] Owner directive 16:27 CST: 大模型(giants 类,evo family)不列入 pytest census — Phase 9 wires example-nightly WITHOUT the evo-lane pytest steps (deselect precedent: stage-1 mcp), giants-prefetch + evo-venv CI steps retire; evo tests stay in-repo as dispatch/manual lane; committed executed-notebook outputs remain the evidence; models.lock evo rows stay as provenance
- [Phase 08 close 2026-10-05] Owner rule: 本地 cache 一般情况下一律不清理 — local caches (hf/modelscope/giants/ollama) are the persistence layer that keeps nightlies download-free; cleanup only with explicit owner approval + dead-data evidence (2026-10-05 cleanup was the one approved exception; Phase 9 planning must NOT propose routine cleanup)
- [Owner decision 2026-10-05] Type-checking QC: **ty 升硬门,mypy 降级退休** — Phase 9 scope: triage E-family diagnostics to 0 (post-261003-0p0 baseline 165), then ty becomes a hard CI gate (fast leg + check_code.py + pre-commit); mypy exits pre-commit + CI + check_code.py ([tool.mypy] config retired last). Static checks stay OUT of pytest (lint lane, not behavior lane)
- [Owner decision 2026-10-05, A/A] Nightly runtime cuts (Phase 9 scope): (1) finetune_custom_head epochs 3→1 via TEST-SANDBOX-ONLY YAML patch (harness sandbox-patch step, ~31→~11min; committed notebook content unchanged; loop body identical so executability claim intact); (2) mcp_example pair: per-request ollama `options.num_ctx` ~8k at the D-13/probe layer (today's 256k kv-cache held 36GB and dominated both latency and the 14:07 VRAM trough; same model per D-11, same turns, notebook content unchanged). Both land with same-change tests in Phase 9
- [Phase 09]: [Phase 09] 09-01: giants marker registered + applied spec-derived via _GIANTS_GATED/_gated_test_param (composable marks; only the 1 evo execution test marked, 4 fast evo contract tests stay on every fast leg per OQ2); measured census triple 188/197 (9 deselected = 8 mcp + 1 giants) pinned by the new Stage 0.5 hard gate (D-03), the deliberate census-growth bump-point (09-02 bumps it to 193/202)
- [Phase 09]: [Phase 09] 09-01: D-04/D-11 topology surgery — all four evo provisioning blocks and both models.lock-keyed hub cache restores deleted (cold pulls by design; flash-attn build-isolation bug of run 37278002681 dissolved with deletion; local giants tier + $HOME hub caches remain, never cleaned per owner rule); D-19 cron-string gates on all three nightly jobs close the 05:30 double-trigger before phs merges to main; D-16 dispatch fired: run 37327398343 (outcome = 09-04/D-17 scope)
- [Phase 09]: 09-03: CI-08 guard landed with zero lock additions — first full scan triaged every non-lock candidate as non-model (MIME keys and data-file paths filtered generically; one local LoRA adapter path allowlisted with reason); lock additions stay deliberate review acts, the test never writes models.lock
- [Phase 09]: 09-03: quoted-only extraction misses unquoted YAML model paths (benchmark_config.yaml path: rows) — the CI-08 scanner applies quoted-token + YAML-value-position patterns to .yaml files (must-have: every example YAML config covered)
- [Phase 09]: 09-03: route-alignment guard covers 15 single-id/single-route live files (>=10 non-vacuity floor pinned); ambiguous multi-id/multi-route files, raw-AutoModel embedding_attention, marimo runtime dropdowns, and YAML configs are the documented skipped set
- [Phase 09]: 09-03: AUDIT-04 published user-visibly (D-15) — docs/user_guide/continuous_integration.md states kernel-subprocess coverage is unmeasured by design so the example lane does not move the 96.30% gate; out-of-docs links use blob URLs (relative paths to .github/ would 404 on the Pages site)
- [Phase 09]: 09-02: D-05 landed at both seams — seed_sandbox owns yaml_overrides application (fail-closed on missing file/section), the SPEC owns the value (yaml_patch key, finetune_custom_head epochs 3->1) forwarded by the gated fixture spec.get()-style; committed example content byte-identical (D-05 honesty contract)
- [Phase 09]: 09-02: D-03 census triple bumped 188/197 -> 193/202 (9 deselected unchanged: 8 mcp + 1 giants) in the SAME commit whose collected tests cause the growth (the designed bump-point); both ci.yml Stage 0.5 carriers updated from a fresh measurement with the exact stage-1 flags
- [Phase 09]: 09-02: D-06 landed as OLLAMA_CONTEXT_LENGTH=8192 in the in-repo ollama unit beside the byte-preserved 127.0.0.1 loopback pin; owner re-apply (09-USER-SETUP.md, PENDING) makes it live AND closes the live 0.0.0.0 bind drift (T-09-03); 09-04 Task 2 asserts the read-only systemctl evidence before its baseline dispatch
- [Phase 09]: 09-02: pre-existing fast-lane failure test_plot_for_regression proven pre-existing by a worktree run at plan-start HEAD (1825P/1F/1S -> 1833P/1F/1S = +8P/+0F/+0S); deferred-items.md + WINDOWS.md id 16; also: first D-16 dispatch ran stale remote phs (e9056c2) — cancelled, phs pushed, re-dispatched as 37335797121 at e85eb73b (push before dispatch, lesson for 09-04)
- [Phase 09]: D-06 seam re-decided by owner (option A, 2026-10-06 00:42 CST): num_ctx 8k cut moves from the ollama systemd unit server-default pin to PER-REQUEST options.num_ctx=8192 injection at the execution/probe layer — the original A/A decision form. The live ollama service is NOT reconfigured (owner declines the sudo re-apply); the runner box unit stays owner-managed. Rework lands as a quick task; 09-04 Task-2 precondition (D-06 re-apply) is voided and replaced by "rework merged". Stale run 37335797121 cancelled same day (owner default). — The systemd-unit seam required owner sudo on the runner host — CI-green could not prove the box was configured, and Phase 9 blocked on a human op (checkpoint 2026-10-05 16:07Z). Per-request injection is fully in-repo, CI-reproducible, and matches owner decision A/A (2026-10-05) verbatim.
- [Phase 09]: Owner decision (2026-10-06 00:52 CST, supersedes the 00:42 "option A" entry): num_ctx runtime cut is DEFERRED entirely. No rework (per-request injection quick task 261006-114 stopped pre-plan); live ollama service untouched; 09-02's in-repo unit pin (e85eb73) stays committed but inert until an owner applies it someday. 09-04 Task-2 precondition is voided — baseline dispatches at final HEAD WITHOUT the cut; D-12 measured budgets must record the un-cut reality and mark the cut deferred; USER-SETUP sudo re-apply item remains open-but-not-required-now. mcp_example pair keeps ~256k ctx kv-cache behavior (latency/VRAM cost accepted by owner for now). — Owner chose to postpone the num_ctx cut ("ollama 配置先不改了") after seeing the rework scope; the CI-06 budget risk of the un-cut mcp_example pair is accepted for now and will be visible in the D-12 measured numbers, giving the owner data to revisit the decision later.
- [Phase 09]: D-11 re-decision 2026-10-06 15:27 CST (quick task 261006-lhm) — same-model constraint lifted for the agent brain; committed mcp_example pair + all mirrors + runner README now reference qwen3.5:4b (4.2B Q4_K_M, ~3.3GB, default context 262144 recorded); dnallm MCP server side untouched; interplay: 00:52 num_ctx deferral stands (no ollama config change, in-repo unit pin stays inert), 3600s cell / 7200s outer timeouts stay (decision B); swap evidence = 15:24 CST capability probe PASSED (3-turn tool-calling via /api/chat, 34.8s cold / 6.1s / 5.1s warm); committed notebook outputs remain from the previous model's execution until the next full nightly re-execution; RED->GREEN pin = TestMcpExampleModelSwap; one ci.yml comment line updated under the narrow fence relaxation.
- [Phase 09]: 09-04: one workflow_dispatch fires all three D-19-gated nightly legs, so per-leg dispatch intent collapsed into single runs measuring every leg at one commit — adopted and recorded as the evidence model
- [Phase 09]: 09-04: coverage-nightly timeout decision (D-12/OQ3, measurement in hand) — KEEP 900 as documented override below both recomputed paper sums (~640min bind-set/~3920min all-marks); per-test marks primary, measured green path 1:42-1:58
- [Phase 09]: 09-04: SSE readiness must assert the received HTTP status, never curl's exit code — an exit-code probe is structurally ungreenable against a healthy SSE endpoint (run 37345067326: 17:58 blind polling over a serving server); pinned by TestExampleNightlySseProbe contract tests
- [Phase 09]: 09-04 close: D-17/D-18 green at the final wiring via run 37432001711 (example ledger zero; test-mamba 1840P/0F; coverage-nightly 1938P/0F @ 96.42%) — the phase's three criteria hold on runner evidence; num_ctx stays DEFERRED, D-05 epochs cut measured 566s

### Pending Todos

- [Owner decision 2026-10-04] BEFORE Phase 9 discuss/plan: refresh knowledge artifacts — run `/gsd-map-codebase` (refresh .planning/codebase/ maps, the direct planner/researcher input) then `/gsd-graphify` (rebuild .planning/graphs/). Trigger: Phase 8 execution completes + verification/transition done. Both maps are pre-v1.1-execution stale (2026-09-29 era); the plan-time drift gate will otherwise fire red at Phase 9 planning.
- Next notebook round (owner-scoped 2026-10-02, "门控的留在下一轮"): gated families — generation_evo_models, generation_megaDNA, finetune_custom_head, lora_finetune ×2 (evo2/megaDNA/mamba prerequisites per 05-FEASIBILITY.md); mcp_example ×2 DONE 261003-csd (both green in the gated lane); finetune_generation megaDNA half (now honestly gated via _gate_megadna)
- TypedDict pass for `load_config` (owner chose option A, 2026-10-02): `dnallm/configuration/configs.py:495` returns `dict[str, BaseModel]` → per-key TypedDict (task→TaskConfig etc.); coordinated update of `dict[str, BaseModel]` consumers (DNAInference/DNATrainer/cli) + tests in same change; kills IDE pyrefly `invalid-argument-type` on notebook `configs['task']` calls
- Typing special (merge with the TypedDict pass): ty baseline report at `.planning/research/ty-check-dnallm-2026-10-02.txt` (568 diagnostics, proven false positives in unresolved-import class, vendored files unexcluded) — configure `[tool.ty.src]` excludes, triage, polish the 15 shim annotations; CI stays mypy-advisory until then

### Blockers/Concerns

- [Phase 07] Advisory review findings open in 07-REVIEW-DISPOSITION.md (8 of 8 open; none failing a must-have): WR-01 check_docs_sync IGNORE still misses gitignored benchmark runtime dirt (local-only red, CI clean); WR-02 CRE cell-18 band-loop can raise bare KeyError if selection.md changes shape; 6 info (stale lock comment, _execution docstring count, anno.md per-gene-F1 prose, unused locus_key params, bedtools not in nightly CI image, Anno label-index assert suggestion) — `/gsd-code-review 7 --fix` addresses them if wanted
- Runtime budget risk: ~24 new slow tests with naive serial ceilings 14–48h vs the 900-min nightly job — measure per-artifact budgets during the Phase 5 pilot and Phase 8 rollout; escalation pre-authorized (CI-06)
- From v1 ship triage (still open, live in /gsd-ship ledger): WR-01 nightly test-mamba continue-on-error; WR-02 plot.py prepare_data drops task_type; WR-03 workflows README stale — note WR-08/09 are v1.1 Phase 5 scope, these three are not
- RESOLVED 2026-10-05 (owner decision): GitHub cache quota — models-cache layer dropped from CI entirely (cold pulls; 65-min cold stage 1 proven); box-side hub cleanup executed same day (84→19GB)
- [Phase 09 wiring] Owner directive 2026-10-05: evo/giants excluded from the pytest census — example-nightly deselects the evo lane (stage-1 mcp precedent); giants prefetch + evo venv CI steps retire in Phase 9; scope-fence it in 09-CONTEXT
- [Phase 08 leftover] example-nightly 05:30 cron double-triggers coverage-nightly + test-mamba via their own schedules — one-line job-gate fix pending owner call
- [Phase 08→09] example-nightly first full dispatch (run 37278002681) outcome lands per Phase-5 D-04 post-merge boundary — verify during Phase 9 census wiring
- [Phase 05] D-04 feasibility dispatch deferred post-merge — `workflow_dispatch` needs feasibility.yml on the default branch; fires after phs→dev→main integration (phs range unpushed, manual-push rule)
- [Phase 05] Open review findings in quick-task code (05-REVIEW-DISPOSITION.md): CR-01 and WR-01 both FIXED (261003-hhj commit 032b308; 261003-ij4 commit 3fe80bf) — no open critical/warning findings from the Phase 05 incremental review remain
- RESOLVED 2026-10-03 (owner decision B+): 06-03 Task 2 fla blocker — flash-linear-attention 0.5.2 installed into .venv; 16-window probe through the unmodified dnallm route confirmed healthy separation (0.7673/0.2230 vs dead fallback 0.0073/0.0087); 06-03 completed (commits e9d00df, b68e1a5). Follow-up upgraded by owner 17:03 CST: fla becomes a declared pyproject dependency (in-phase quick task after 06-03). Original evidence: example/notebooks/plant_helixseek_shared/.scratch/fla-fallback-diagnosis.md + fla-probe-0.5.2.md

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261002-se3 | Fix transformers 5.x remote-code compat: restore get_extended_attention_mask for trust_remote_code ESM models (benchmark notebook AttributeError), with pytest coverage | 2026-10-02 | fdc4915 | [261002-se3-fix-transformers-5-x-remote-code-compat-](./quick/261002-se3-fix-transformers-5-x-remote-code-compat-/) |
| 261002-sl7 | Run and fix the 6 non-gated census-failing notebooks to green: 5 promoted to ACTIVE lane (8-13), finetune_generation data-prep fixed + megaDNA half honestly gated; 5 transformers-5.x shims + 34 contract tests | 2026-10-02 | fa0386e | [261002-sl7-run-and-fix-the-5-non-gated-census-faili](./quick/261002-sl7-run-and-fix-the-5-non-gated-census-faili/) |
| 3 | gsd-fast: fix Benchmark.plot return annotation lie (-> None vs actual 2-tuple), kills ty not-iterable in benchmark notebook | 2026-10-02 | d352c0e | — |
| 261003-0p0 | Batch typing special: ty 570->165 (excludes + 44 audited suppressions + canonical renames + TypedDict + 41 ignore removals); E-family triage list emitted; fast lane 1703 green | 2026-10-02 | a0220d5 | [261003-0p0-batch-typing-special-configure-ty-baseli](./quick/261003-0p0-batch-typing-special-configure-ty-baseli/) |
| 261003-csd | Execute the 2 owner-deferred MCP client notebooks to green in the gated lane (D-08 closed; execute-state gate + 4xx probe + isolated langchain kernel; 2 dnallm serving fixes with tests: single-flight inference, mamba interpret guard; port 8000, fallback never fired; full lane 1716 green) | 2026-10-03 | 9453d23 | [261003-csd-execute-the-two-owner-deferred-mcp-clien](./quick/261003-csd-execute-the-two-owner-deferred-mcp-clien/) |
| 261003-hhj | Fix CR-01: MCP single-flight inference — threading.Lock inside the executor-submitted callable spans the orphaned thread lifetime (asyncio lock released on timeout cancellation); asyncio.wait_for-cancellation regression test; 3/3 single-flight tests + 220 tests/mcp green | 2026-10-03 | 032b308 | [261003-hhj-fix-cr-01-mcp-single-flight-inference-as](./quick/261003-hhj-fix-cr-01-mcp-single-flight-inference-as/) |
| 261003-ij4 | Fix WR-01: dna_interpret runs captum work in the default executor behind a dedicated `_interpret_thread_lock` (CR-01 pattern) — event loop stays responsive during long attributions, the 30s tool timeout actually fires, timeout→retry cannot stack concurrent interpretations; 3 red-then-green regression tests + 223 tests/mcp green | 2026-10-03 | 3fe80bf | [261003-ij4-fix-wr-01-dna-interpret-runs-blocking-ca](./quick/261003-ij4-fix-wr-01-dna-interpret-runs-blocking-ca/) |
| 261003-jpr | Fix IN-01: absence guards on the three sl7 patch installers in transformers_compat.py (configuration_utils/cache_utils/modeling_utils bare imports → standard try/except no-op guards); new TestTransformersAbsenceContract dynamically collects installers so future ones are auto-covered (11 items, roster pin + apply_patches survival); RED 4F/83P → GREEN 87P, tests/utils 129 green | 2026-10-03 | 56a72c9 | [261003-jpr-fix-in-01-the-three-new-patch-installers](./quick/261003-jpr-fix-in-01-the-three-new-patch-installers/) |
| 261003-r73 | Close Phase-06 review warnings WR-01/02/03: nightly CI legs install .[base,fla] + typed importorskip guards on the two slow smokes (WR-01, eb85f7e); CRLF-robust slice_gff_rows — strips \n/\r\n/\r terminators, raises on embedded \r, 2 same-change tests (WR-03, 8d6bd3b); _load_with_fallback exception classifier — env failures skip typed (byte-identical message), dnallm regressions fail, 7 fast tests (WR-02, 1219f0f); 06-REVIEW-DISPOSITION WR-01/02/03 flipped fixed, 35 fast tests green | 2026-10-03 | eb85f7e, 8d6bd3b, 1219f0f | [261003-r73-close-phase-06-review-warnings-wr-01-02-](./quick/261003-r73-close-phase-06-review-warnings-wr-01-02-/) |
| 261003-ryz | Close Phase-06 review info findings IN-01..06: fetch_sequence path branch releases pyfastx handles + cleans only a .fxi it created, 3 same-change tests (aaf6308); ASCII-only bare-numeric chrom digits (c19999a); import-purity test restores the package attribute + identity guard (3662ea5); Anno label_names re-quoted, one-line diff gate (7790920); fla extra asserted by exact bracket-member parse (2a0ba40); local .scratch/ ignore removed by live masked/unmasked check-ignore evidence — root .gitignore:60 covers it, 06-01 .py-coverage claim corrected (188a4f6); dispositions IN-01..06 fixed, open: 0; 36 fast tests green, branch pushed | 2026-10-03 | aaf6308, c19999a, 3662ea5, 7790920, 2a0ba40, 188a4f6 | [261003-ryz-close-phase-06-review-info-findings-in-0](./quick/261003-ryz-close-phase-06-review-info-findings-in-0/) |
| 261004-dyw | Showcase display enhancement: PNG mimes everywhere + pgt zoom windows + new combined notebook (window Chr1:5220001-5260000 +5kb flank; pygenometracks adopted, GPL override recorded; leaf-DNase bedGraph + truth GTF + region FASTA committed artifacts) | 2026-10-04 | 4c2e5bd | [261004-dyw-planthelixseek-showcase-notebook-vega-ve](./quick/261004-dyw-planthelixseek-showcase-notebook-vega-ve/) |
| 12 | gsd-fast: guard all FASTA header interval regex matches against None in the three showcase notebooks (ty Match\|None fix; 4 sites, RuntimeError guard style, mirrors synced, 19 fast tests green) | 2026-10-04 | cbc5735 | — |
| 13 | gsd-fast: DNAInference/DNATrainer config params dict->Mapping[str, Any] (ty TypedDict assignability for load_config output in notebooks; no mutation sites); 3 contract tests | 2026-10-05 | 86022f7 | — |
| 14 | gsd-fast: Mutagenesis/Benchmark/DNAInterpret config -> Mapping; Benchmark backfills into a private dict(config) copy (no caller aliasing); contract tests extended to 5 engines + no-alias pin (types.UnionType unwrap lesson); fast-subset 389P | 2026-10-05 | 16a9ffb | — |
| 261006-cum | Fix pre-existing test_plot_for_regression regression (Mapping fallout, WINDOWS id 16) | 2026-10-06 | 550d311 | [261006-cum-fix-pre-existing-test-plot-for-regressio](./quick/261006-cum-fix-pre-existing-test-plot-for-regressio/) |
| 261006-lhm | Direct-edit mcp_example notebooks' committed model id to qwen3.5:4b (owner decision 15:27 CST) | 2026-10-06 | 0a5ca0d | [261006-lhm-direct-edit-mcp-example-notebooks-commit](./quick/261006-lhm-direct-edit-mcp-example-notebooks-commit/) |
| 261006-uq7 | Apply dependabot ruff bump on phs with mamba red-line gates (owner 22:06) | 2026-10-06 | 7cf8458 | [261006-uq7-apply-dependabot-ruff-bump-on-phs-with-m](./quick/261006-uq7-apply-dependabot-ruff-bump-on-phs-with-m/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | 09/deferred-items.md: test_plot_for_regression fast-lane failure | acknowledged (resolved in artifact by 261006-cum `550d311`; WINDOWS id 16 fixed) | 2026-10-06 | v1.1 |
| deferred_items | 05/deferred-items.md: dnallm-mcp-server CLI `--host`/`--port` silently overridden by yaml config (server.py:1696-1700) | acknowledged (product decision: argparse sentinel defaults vs drop flags) | 2026-10-06 | v1.1 |
| deferred_items | 05/deferred-items.md: `uv run pytest` resolver failure under uv 0.12.20 universal resolution (mamba × cuda conflicts matrix) | acknowledged (working forms: `uv run --no-sync pytest` or venv pytest) | 2026-10-06 | v1.1 |
| quick_tasks | 261002-inq-raise-python-floor-to-3-12-drop-3-10-3-1 | missing (plan-only, never executed — Python floor raise not taken in v1.1) | 2026-10-06 | v1.1 |
| uat_gaps | 05/05-UAT.md | passed, 0 pending (feasibility.yml workflow_dispatch deferred until phs→dev→main) | 2026-10-06 | v1.1 |
| deferred_items | 03/deferred-items.md: import-time `logs/dnallm.log` sink recreated under pytest cwd every run (`DNALLMLogger._setup_handlers`, logger.py:57-60) | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03/deferred-items.md: test_timeout.py fixed ~60s cost from two full-30s-timeout waits (shorten `_tool_timeout_seconds` like `test_timeout_configurable` does) | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` no-op — `ds.map` result discarded (data.py:983), latent bug pinned as-is by test | acknowledged | 2026-10-01 | v1 |

## Session Continuity

Last session: 2026-10-06T21:35:00+08:00
Stopped at: Phase 09 complete — all phases complete
Resume file: None

## Deferred Verification

| Phase | State | Resume |
|-------|-------|--------|
| *(none — v1 phase 4 verification closed 2026-10-01, passed 10/10)* | | |

## Operator Next Steps

- Plan Phase 5 with `/gsd-plan-phase 5` (Phase 6 is an independent parallel track if desired)
