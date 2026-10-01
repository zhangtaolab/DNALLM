# Milestones

## v1 Test Suite Audit & Coverage Hardening (Shipped: 2026-10-01)

**Phases completed:** 4 phases, 13 plans, 34 tasks

**Delivered:** an untrustworthy 464-test suite at 45.92% (exit-code-masked, single-root collection) became 1,657 passing tests at **96.30%** line coverage behind a `fail_under=90` CI hard gate — red-proven end-to-end via probe PR #39, required-check branch protection on dev+main, nightly slow census green on the self-hosted GPU runner.

**Stats:** 3 days (2026-09-29 → 2026-10-01) · git range `fe1d4b4..ae5352a` (179 commits, 146 files, +32,174/−601) · closeout audit `tech_debt` verdict with 0 critical gaps (23/23 requirements, 8/8 integration seams, 2/2 E2E flows)

Known verification overrides: 3 newly acknowledged, 0 carried forward from a prior close (see STATE.md Deferred Items)

**Key accomplishments:**
- Single pytest config (pyproject.toml only), honest exit codes via pytest_sessionfinish cleanup, config-driven coverage with the pre-locked 7-entry omit denominator, and a permanent CI exit-code canary
- Full-suite audit under the Plan-01 harness: census 625/0/0/9 with skip reasons, baseline coverage 45.92% with a 43-file ranked gap worklist, cold/warm slow timings, and the subprocess-coverage scope decided on canary evidence
- Presence-guarded multiclass AUROC (matchable ValueError on absent-class batches, labels=-anchored macro-ovr) plus a guarded first-resolved-wins dispatch chain so the CrossDNA handler's result survives load_model_and_tokenizer — both formerly-skipped tests now run green
- Autouse tmp_path rebind of the PDF_OUTPUT_DIR module global redirects all 25 PDF-writing call sites with zero churn; class-level pdf markers make 53 writing tests selectable (was 0 of 65); .gitignore consolidated to one correct directory entry; 9 strays deleted — twice-run tree-clean proof green
- All 6 broad-except MCP network skips replaced by a typed httpx.TransportError + ExceptionGroup-unwrapping helper (stable network-unavailable: messages, all-leaves rule), dead string-matching skip scaffolding deleted, and an out-of-process enforcement pipeline (junit artifact -> 11-entry categorized YAML allowlist -> fail-closed audit script -> CI step) so an unexpected skip now fails CI instead of passing silently
- 279 new behavior tests closing the inference area from 1,575 to 176 missing lines (gate ≤250) via real-torch end-to-end engine paths, real captum attributions, and altair chart-contract assertions; five latent crashes fixed under Rule 1
- 317 fault-injection and real-torch behavior tests closing the models area from 1,209 to 88 missing lines (gate ≤ 240) via sentinel dispatch matrices, staged-failure tokenizer fallbacks, seven real head forwards, and sys.modules-stubbed special-family handler bodies
- 154 protocol and behavior tests closing the mcp area from 449 to 6 missing lines (gate ≤ 110) via a socket-free in-memory MCP round trip, ordered progress-contract assertions, patched-uvicorn transport construction shapes, and full lifecycle coverage — plus one Rule 1 fix to the multi-model success counter
- 104 content-level tests closing the datahandling/finetune area from 406 to 10 missing lines (gate ≤ 100) via tmp_path format round-trips, exact-id tokenization pipelines, boundary-mocked trainer wiring — and the suite crossed the milestone line to 91.24%, above the >90.5% target, one wave early
- The phase's named behavior-contract requirement proven on the live patched transformers class, all five CLI entry points driven in-process through CliRunner with asserted call args, both 60-line orphans closed to zero — and the FINAL GATE landed at 96.28% (7,124/7,399), strictly above the >90.5% target with audit-clean skips, pragma stability at exactly 3, and a diff-free denominator
- fail_under = 90 coverage ratchet live in pyproject (proven green at 96.30% and red at 78.92% locally), plus the two Wave-2 CI inputs: the 9-entry models.lock cache manifest and 7 per-test timeout marks beating the global 300s
- The amended two-job CI shape is live on dev: coverage-gate green at 96.27% on push run 36745734429 with all six matrix legs green under the same fail_under ratchet, nightly+deploy guards proven at runtime, codecov uploader gone (GATE-03), and calibration dispatch run 36747594207 healthy into its slow census with the models.lock cache seeding
- The gate's teeth are proven: probe PR #39 deleted tests/models, the real coverage-gate check on dev concluded FAILURE with the verbatim line `ERROR: Coverage failure: total of 79 is less than fail-under=90` in the CI log (all 1261 collected tests green — only the floor red), and the probe left zero residue — plus the WR-05 README now documents the two-job gate and the owner holds the branch-protection and nightly-runtime decisions as runnable artifacts

---
