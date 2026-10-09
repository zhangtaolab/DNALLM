---
phase: 12-motif-matching-mcp-tools-milestone-closeout
plan: "03"
type: execute
wave: 2
depends_on: ["12-01", "12-02"]
files_modified:
  - docs/user_guide/fine_tuning/peft_adapters.md
  - docs/user_guide/continuous_integration.md
  - pyproject.toml
  - CHANGELOG.md
autonomous: true
requirements: [DOCS-01]
user_setup: []
estimate:
  tokens: 14000
  raw_tokens: 14000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - "The peft_adapters.md IA³ section is real usage documentation written from the shipped Phase-11 behavior (use_ia3 trainer branch, Ia3Config, Pydantic-time and trainer-init rejections, adapter save/reload roundtrip) — the 'coming in the next release' forward pointer is gone"
    - "docs/user_guide/continuous_integration.md and the pyproject.toml fail_under comment state the MEASURED post-Phase-12 coverage number from an actual coverage run in this plan — never a number copied from an older phase (D-07 honesty)"
    - "Every new module this phase (dnallm/interpret/motifs.py at minimum, plus dnallm/mcp/server.py's expanded surface) shows a measured coverage row at or above the 96% per-module standard"
    - "Every CHANGELOG ## [Unreleased] entry REV-01..REV-11 carries a commit-SHA link to its fix — the evidence chain is complete and traceable for the rebuttal letter (D-08/D-09)"
    - "The census collect line is verified to still read 208/217 tests collected (9 deselected) before closeout — re-pinned only if broken, minimally, per D-07"
    - "The full fast lane passes with 0 failures and every new skip typed and allowlisted"
  artifacts:
    - "docs/user_guide/fine_tuning/peft_adapters.md (completed IA³ chapter replacing the 174-186 stub)"
    - "docs/user_guide/continuous_integration.md (measured coverage expectation, honest update)"
    - "pyproject.toml (comment-only fail_under annotation update — no config value changes)"
    - "CHANGELOG.md (complete ## [Unreleased] evidence chain: REV-01..REV-11 with SHA links)"
  key_links:
    - "IA³ chapter prose <-> dnallm/finetune/trainer.py + dnallm/configuration/configs.py Ia3Config (docs must match shipped field validation exactly)"
    - "coverage-expectation docs <-> the coverage run performed in this plan (same-change measurement)"
    - "CHANGELOG SHA links <-> git log of the REV-tagged commits from Phases 10-12"
  prohibitions:
    - "No coverage number published that did not come from a measured run in this plan"
    - "No wholesale census re-freeze (D-07: re-pin only if the hard-asserted count is actually broken, and then minimally)"
    - "No CHANGELOG entry text rewrites — backfill adds links only, append-only discipline"
    - "No source-code changes (docs + comments + CHANGELOG only; if the fast lane is red, the fix belongs to the owning lane, escalated via SUMMARY — not patched here)"
    - "No example/ or tests/examples/ changes"
    - "No new dependencies; no dnallm/__init__.py changes"
---

<objective>
Milestone closeout (C3): complete DOCS-01's split delivery — the IA³ chapter finishing the
peft_adapters.md forward pointer — finalize the CHANGELOG evidence chain with SHA backfill
(REV-01..REV-11, D-08/D-09), and update the coverage-expectation docs honestly with the
measured post-Phase-12 number (D-07), verifying the census pin and full-fast-lane green as
the milestone's closing evidence.

Purpose: the v1.2 milestone closes fully green with an honest docs surface and a complete
rebuttal-letter evidence chain.
Output: completed docs chapter, measured coverage-expectation updates, finalized CHANGELOG,
and the milestone-closeout verification record.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-RESEARCH.md
@.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-CONTEXT.md
@docs/user_guide/fine_tuning/peft_adapters.md
@docs/user_guide/continuous_integration.md
@CHANGELOG.md
@dnallm/finetune/trainer.py
@dnallm/configuration/configs.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: IA³ chapter — replace the peft_adapters.md forward pointer with real usage documentation</name>
  <files>docs/user_guide/fine_tuning/peft_adapters.md</files>
  <read_first>
  - docs/user_guide/fine_tuning/peft_adapters.md:160-190 (the LoRA/QLoRA chapters above as the structural analog + the IA³ stub at 174-186 to replace)
  - CHANGELOG.md line ~17 (the REV-04 entry — the shipped-behavior summary to write from: real DNATrainer branch via peft.get_peft_model, Ia3Config peft-0.21.1 field parity, use_ia3 x use_qlora rejected at Pydantic time, LoRA x IA³ rejected at trainer init, adapter-kind-agnostic reload path)
  - dnallm/finetune/trainer.py (the IA³ branch actually shipped) and dnallm/configuration/configs.py (Ia3Config fields: target/feedforward modules, init, exclude_modules)
  - 12-RESEARCH.md Pitfall 8 (chapter contradicting shipped behavior; fenced blocks must survive the docs-validation ruff-format gate — 0.7.1 precedent)
  </read_first>
  <action>
  Replace the IA³ forward-pointer stub section at peft_adapters.md:174-186 (heading promises a future release; body says setting use_ia3 does not yet switch the trainer) with a real usage chapter per D-08, written from the shipped Phase-11 reality (Pitfall 8 guard: the stub text claims the trainer branch has not landed — it has). Mirror the LoRA/QLoRA chapter structure immediately above: prose tied to actual TrainingConfig/Ia3Config fields, a fenced YAML config block (finetune.use_ia3: true plus the ia3 section) and a fenced Python block (trainer init + adapter save/reload through the shared PeftModel path). Document the two rejections with their real matchable error behavior (use_ia3 x use_qlora at Pydantic config time; LoRA x IA³ at trainer init) and the from-Phase-11 acceptance facts (transformer + Mamba both supported; IA³ save/reload roundtrip). Keep every fenced block within ruff-format constraints (line length 100; the docs-validation gate formats fenced Python blocks — over-long lines trip CI). Do not touch the LoRA/QLoRA chapters above.
  </action>
  <verify>
    <automated>uv run --no-sync mkdocs build --strict</automated>
    <fails_when>nonzero exit — e.g. a fenced code block fails the docs-validation format constraints or the strict build reports a broken reference/warning</fails_when>
  </verify>
  <acceptance_criteria>
  - Content: the phrase "coming in the next release" no longer appears; the section documents use_ia3, the ia3 config section, both rejection behaviors, and the save/reload roundtrip
  - Accuracy: every field name and rejection described matches configs.py/trainer.py source (spot-checked against the shipped code, not the stub)
  - Build: mkdocs --strict green; fenced blocks within ruff-format line constraints
  - Scope: no edits outside the IA³ section
  </acceptance_criteria>
  <done>The IA³ chapter completes DOCS-01's split delivery: real, accurate usage documentation with green strict docs build; the forward pointer is gone.</done>
</task>

<task type="auto">
  <name>Task 2: Measure post-Phase-12 coverage + honest coverage-expectation docs update + census verify</name>
  <files>docs/user_guide/continuous_integration.md, pyproject.toml</files>
  <read_first>
  - 12-RESEARCH.md: Pitfall 6 (census re-pin trigger + the honest coverage-docs requirement), D-07 in 12-CONTEXT.md (minimal re-pin policy)
  - docs/user_guide/continuous_integration.md:24-40 (the "96.30%" lines 26/30/32 to update)
  - pyproject.toml:555-560 ([tool.coverage.report] fail_under comment carrying "96.30%")
  - .github/workflows/ci.yml:857-871 (the census pin block — verify-only reference)
  - The 12-01/12-02 SUMMARYs (which modules landed; their measured rows)
  </read_first>
  <precondition>Plans 12-01 and 12-02 are complete (their modules and tests landed) — this plan's wave-2 dependency.</precondition>
  <action>
  Measure, then write — never the reverse (D-07 honesty): run the full fast lane with coverage using the env-safe form (`uv run --no-sync coverage run -m pytest tests/ dnallm/mcp/tests -q -m "not slow"` then `uv run --no-sync coverage report`; `pytest --cov` is an env crash in this repo — see proven commands). Record (a) the global total, (b) the per-module rows for dnallm/interpret/motifs.py and dnallm/mcp/server.py — both must be >= 96% (the milestone's per-module standard); if either row is below standard, STOP: the deficit belongs to lane 12-01/12-02 — record it in the SUMMARY and escalate rather than papering over it here. Update docs/user_guide/continuous_integration.md lines ~26/30/32 replacing "96.30%" with the measured total plus a dated note (the number moved with the new Phase-12 modules; keep the fail_under=90 ratchet semantics prose intact and update the reference in the AUDIT-04 note paragraph). Update the pyproject.toml [tool.coverage.report] fail_under COMMENT (comment-only — the 90 value itself never changes) to the measured number. Verify the census: `uv run --no-sync pytest tests/ --collect-only -q | tail -3` must still read exactly `208/217 tests collected (9 deselected)`; if (and only if) it differs, apply the minimal re-pin at ci.yml:857-871 per D-07 and record why in the SUMMARY — research verified this should NOT trigger since the phase touches no example/ artifacts.
  </action>
  <verify>
    <automated>uv run --no-sync coverage report --include="dnallm/interpret/*,dnallm/mcp/server.py" --fail_under=96</automated>
    <fails_when>exit 2 — any included module's measured coverage row falls below the 96% per-module standard (the deficit escalates to the owning lane, not patched here)</fails_when>
  </verify>
  <acceptance_criteria>
  - Measurement: the numbers written to the docs come from the coverage run executed in this task (SUMMARY records the command + raw total)
  - Docs: continuous_integration.md carries the measured number with a date; the fail_under ratchet prose stays accurate; pyproject comment updated comment-only
  - Standard: dnallm/interpret/motifs.py and dnallm/mcp/server.py rows >= 96% (gate command proves it)
  - Census: collect line verified 208/217 (9 deselected) — re-pin only if actually broken, minimal, with reason recorded
  </acceptance_criteria>
  <done>Coverage-expectation docs are honest for post-Phase-12 reality, every new module verified at the per-module standard, census pin verified (or minimally re-pinned with cause), and the measurement evidence is in the SUMMARY.</done>
</task>

<task type="auto">
  <name>Task 3: CHANGELOG evidence-chain finalization — SHA backfill REV-01..REV-11 + full fast-lane closing run</name>
  <files>CHANGELOG.md</files>
  <read_first>
  - CHANGELOG.md ## [Unreleased] (all entries REV-01..REV-11 — re-read immediately before editing)
  - 12-CONTEXT.md D-08 (CHANGELOG finalization via the Phase-10 D-09 SHA-backfill mechanism)
  - git log of the revision branch (the REV-tagged commits: locate each entry's fix commit via git log --grep)
  - 12-01-SUMMARY.md and 12-02-SUMMARY.md (their recorded REV-10/REV-11 commit SHAs)
  </read_first>
  <precondition>Plans 12-01 and 12-02 complete with their REV-10/REV-11 entries already appended to ## [Unreleased] (their SHAs recorded in their SUMMARYs).</precondition>
  <action>
  Finalize the evidence chain per D-08/D-09: for every REV-01..REV-11 entry under ## [Unreleased], locate the commit that landed the fix (git log --grep per REV tag; the 12-01/12-02 SUMMARYs record their SHAs directly) and add the commit reference as a markdown link in that entry (short SHA linking to the zhangtaolab/DNALLM commit URL — consistent formatting across all eleven entries). Backfill adds links ONLY: no entry text is rewritten, no entries reordered (append-only discipline). Verify one entry exists per revision fix (eleven REV tags, none missing, none duplicated). Re-read CHANGELOG immediately before the edit (concurrent-surface discipline). Then run the closing full fast-lane evidence run: `uv run --no-sync pytest tests/ -q -m "not slow"` (the proven CI gate shape — 0 failures required; the pass count will exceed the pre-phase 2253 because Phase 12 added fast-lane tests, which is expected and recorded). Any failure here is escalated to the owning lane via the SUMMARY — this plan does not patch source code.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/ -q -m "not slow"</automated>
    <fails_when>nonzero exit or any F in the summary line — a red fast lane blocks milestone closeout and escalates to the owning lane</fails_when>
  </verify>
  <acceptance_criteria>
  - Evidence chain: every REV-01..REV-11 entry carries a working commit-SHA link; eleven tags present, none duplicated
  - Discipline: entry texts unchanged (diff shows link additions only)
  - Gate: full fast lane 0 failures; pass count recorded in the SUMMARY with the delta versus the pre-phase 2253 baseline
  - Traceability: the rebuttal-letter evidence chain is complete (each fix -> commit)
  </acceptance_criteria>
  <done>The milestone's CHANGELOG evidence chain is complete and link-traceable, and the full fast lane is green as the closing evidence — v1.2 is closable green pending phase verification.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| docs/CHANGELOG claims -> readers (rebuttal letter) | Documentation numbers and links are consumed as evidence by humans; no untrusted input crosses into this plan's code (none is written) |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-12-09 | Repudiation | coverage-expectation docs + CHANGELOG SHA links | medium | mitigate | Numbers written only from measured runs performed in this plan (Task 2 gate command); SHA links verified to resolve to the grep-located commits; no source changes allowed in this plan keeps the evidence surface append/update-only |
| T-12-SC | Tampering | package installs | high | accept | Zero package installs this phase (zero-new-dependencies milestone invariant) — nothing to gate |
</threat_model>

<verification>
- Docs: `uv run --no-sync mkdocs build --strict` green; the docs-validation workflow (ruff over fenced blocks) mirrors it in CI.
- Coverage honesty: the number in continuous_integration.md equals the coverage report total from this plan's run (SUMMARY records both).
- Evidence chain: eleven REV tags under ## [Unreleased], each with a commit link; `git diff` on CHANGELOG shows link-additions only.
- Closing gate: `uv run --no-sync pytest tests/ -q -m "not slow"` — 0 failures.
- Census: collect line 208/217 (9 deselected) verified (or minimal re-pin with recorded cause).
- Milestone invariants: `git diff origin/main..HEAD -- pyproject.toml` shows no dependency-list changes from this plan; dnallm/__init__.py untouched across the phase.
</verification>

<success_criteria>
- DOCS-01 completes its split delivery (IA³ chapter real and accurate; forward pointer gone).
- Coverage-expectation docs honest with the measured post-Phase-12 number; new modules at the >=96% per-module standard.
- CHANGELOG evidence chain complete (REV-01..REV-11 SHA-linked) per D-08/D-09.
- Census verified; full fast lane green; milestone v1.2 closable fully green.
</success_criteria>

## Artifacts this phase produces
- `docs/user_guide/fine_tuning/peft_adapters.md` — completed IA³ chapter (DOCS-01 split delivery closed)
- `docs/user_guide/continuous_integration.md` + `pyproject.toml` comment — honest measured coverage expectations
- `CHANGELOG.md` — finalized ## [Unreleased] evidence chain (REV-01..REV-11 SHA links; REV-10/REV-11 appended by lanes 12-01/12-02)
- The milestone-closeout verification record in this plan's SUMMARY (measured totals, census line, fast-lane count, phase artifact rollup)

<output>
Create `.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-03-SUMMARY.md` when done.
The SUMMARY is the milestone-closeout record: measured coverage numbers, census verification,
fast-lane pass count, the phase artifact rollup, and any escalations (per-module deficits,
red-lane failures, census re-pin causes) — plus the pending owner-input status for the
HBG1/BCL11A golden fixture reported by 12-01.
</output>
