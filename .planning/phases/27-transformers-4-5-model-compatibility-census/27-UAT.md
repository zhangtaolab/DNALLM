---
status: complete
phase: 27-Transformers 4/5 Model Compatibility Census
source: [27-VERIFICATION.md]
started: 2026-10-27T18:57:00+08:00
updated: 2026-10-28T21:35:00+08:00
---

## Current Test

[testing complete]

## Tests

### 1. Live dispatch of the census lane on the self-hosted runner
expected: gh workflow run census_compat.yml --ref v1.4 → 49 matched / exit 0 (or honest red on real drift)
result: issue
reported: "Owner delegated verification ('你来验证'). Verified on the runner box itself (edgexpert-296a = dnallm-nightly): gh dispatch returns HTTP 404 (workflow exists only on v1.4; GitHub registers dispatchable workflows from the default branch only, and main is frozen for the v1.4 cycle). Replicated the full lane command sequence locally on the same hardware/network (fresh uv venv, .[full] install, sanitized env): python scripts/census_compat.py --run --with-loads --verify --artifact <seeded copy> → 'UNEXPECTED (4 entr(ies) disagree)' → LANE-EXIT-CODE: 1. Deterministic, not world drift: all 5 newly-failing models are the mamba family (caduceus-ph/ps, PlantCaduceus, PlantCAD2, PlantBiMoE) failing at 'ModuleNotFoundError: No module named mamba_ssm' — the lane's 'uv pip install -e .[full]' lacks the mamba extra, while the committed census evidence was produced with mamba_ssm 2.3.2.post1 present. The lane as committed can never exit 0."
severity: blocker

### 2. Biologist-audience readability sign-off on the compatibility page
expected: Owner reviews docs/user_guide/model_compatibility.md and triages the 3 WARNING-class plain-language findings in 27-UI-REVIEW.md: (a) Failure Details cells leak engineering jargon on 6 "not working" rows, (b) footnote 1/2 lack inline row markers, (c) GENA-LM/MutBERT Details cells exceed 300 chars. Accept as-is or request a follow-up pass.
result: issue
reported: "Owner delegated verification. Evidence pack generated and verified (Material-theme screenshots at .scratch/p27-uat/shots/, HTML report served at http://192.168.130.8:8791/report.html): page renders correctly, intro is plain-language, counts sentence matches the table. All 3 UI warnings CONFIRMED present in the rendered page. Triage under the standing 2026-10-21 plain-language directive: (a) is a directive violation on the failure rows a worried reader checks first — recorded as a gap; (c) folds into the same generator change as (a); (b) accepted as-is for now (page-wide footnotes carry the caveat)."
severity: minor

## Summary

total: 2
passed: 0
issues: 2
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-27-1
  truth: "The census lane finishes green (49 matched, --verify exit 0) on unchanged inputs"
  status: failed
  status_note: "Code fix landed in Phase 27-05 (the lane now runs the sanctioned sha256-pinned scripts/install_mamba.sh between the .[full] install and the census run; the contract test pins it). The gap stays FAILED because its truth is a GREEN LIVE RUN, and that has never been observed — Phase 27.1's DISP-01 was owner-deferred (see 27.1-03-SUMMARY.md: GitHub protected-branch GH006 blocked registration; the defer-dispatch decision moved the single allowed dispatch to the post-merge operator step). Close this gap when that dispatch returns 49 matched / 'OK: census matches committed registry fields' / no UNEXPECTED / no VERSION DRIFT."
  reason: "User reported: 你来验证 — delegated live verification ran the exact lane command on the runner box: exit 1 with 4 UNEXPECTED (unmappable) entries; root cause is environmental, not world drift"
  severity: blocker
  test: 1
  root_cause: ".github/workflows/census_compat.yml installs only 'uv pip install -e \".[full]\"' — the full extra (dev,test,notebook,mcp) has no mamba-ssm, but the mamba-family remote code (kuleshov-group caduceus×3, PlantCAD2, lgq12697/PlantBiMoE) imports mamba_ssm at import-probe time. The committed census evidence was produced in the dev venv WITH mamba_ssm 2.3.2.post1 (installed via scripts/install_mamba.sh: sha256-pinned sdist + four -std=c++17→c++20 patches for torch≥2.11; plain .[mamba] extra cannot build on this box — uv source build fails, verified). Without mamba_ssm the lane deterministically reports 4 unmappable entries and exits 1."
  artifacts:
    - path: ".github/workflows/census_compat.yml"
      issue: "install step missing the sanctioned mamba-kernel install (scripts/install_mamba.sh) after the .[full] install"
    - path: "scripts/install_mamba.sh"
      issue: "the repo's sanctioned pinned installer the lane should call — already proven on this box (dev venv has mamba_ssm 2.3.2.post1); note it needs pip bootstrapped into the uv venv first (uv pip install --python <venv> pip)"
    - path: ".scratch/p27-uat/lane-sim/census-run.log"
      issue: "full evidence log of the red run (49 censused, 45 compared, 4 unmappable, exit 1)"
    - path: ".scratch/p27-uat/lane-sim/census-run-fixed.log"
      issue: "controlled green run after scripts/install_mamba.sh into the same replica venv: 49 matched, 'OK: census matches committed registry fields', LANE-EXIT-CODE: 0 — single-variable proof of the root cause and the fix direction (evidence ready for 27-05 Task 3)"
  missing:
    - "Add the mamba-kernel install step to census_compat.yml (bootstrap pip, then bash scripts/install_mamba.sh --python .venv/bin/python --index-url <mirror>) between the .[full] install and the census run"
  debug_session: ""

- gap_id: G-27-2
  truth: "Failure Details cells lead with a plain-language sentence a biologist can act on (technical cause demoted), per the 2026-10-21 plain-language directive"
  status: resolved
  resolved: "2026-10-28 by Phase 27.1 close-out — the plain-language lead layer landed in Phase 27-04 (generator + regenerated page + drift gate, verified byte-identical on re-emit) and the v1.4 milestone audit confirmed all 6 failure rows carry a plain-language lead with the technical cause demoted; Phase 27.1's IN-01 balance-aware truncation further tightened those cells without changing the lead wording."
  reason: "User reported: delegated verification confirmed UI-REVIEW warning (a) in the rendered page — 6 'not usable' rows carry raw engineering jargon (torch.Tensor/meta device, type_vocab_size, deepspeed import, input_shape slicing, olmo.config) with no plain-language lead; warning (c) same cells exceeding 300 chars"
  severity: minor
  test: 2
  root_cause: "scripts/census_compat.py --emit-docs emits the census evidence's technical failure cause verbatim as the entire Details cell; the generator has no plain-language lead-sentence layer for failure rows, and the drift gate (tests/models/test_compat_docs_consistency.py + census tests, 83 collected) pins the current wording byte-for-byte, so any rewording must extend generator and gate in lockstep"
  artifacts:
    - path: "scripts/census_compat.py"
      issue: "--emit-docs failure Details = raw technical cause, no plain-language lead"
    - path: "docs/user_guide/model_compatibility.md"
      issue: "6 failure rows leak jargon; GENA-LM/MutBERT cells >300 chars (generated file — fix belongs in the generator)"
    - path: "tests/models/test_compat_docs_consistency.py"
      issue: "drift gate pins current wording; must be extended in lockstep with the generator change"
  missing:
    - "Generator: failure Details cells lead with one plain-language sentence ('This model's published code uses a feature the new library version removed; it cannot be loaded until the model authors update it' pattern), technical cause demoted to a parenthetical/short suffix (also resolves the >300-char cells)"
    - "Regenerate docs/user_guide/model_compatibility.md via the generator (no manual edits) and extend the drift gate to pin the new wording"
  debug_session: ""

## Deferred Follow-Ups

- test: 2
  idea: "UI-REVIEW warning (b): footnote 1/2 (tested-on-5.17 / inferred-4) lack inline row markers — add per-row markers wired to the footnotes (requires generator + drift gate extension, same pattern as G-27-2)"
  deferred_at: 2026-10-27
