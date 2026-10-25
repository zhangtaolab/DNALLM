---
phase: 261025-vvv
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - README.md
autonomous: true
must_haves:
  truths:
    - "README.md's hardware-group table (six rows: cpu/cuda124/cuda126/cuda128/cuda130/mamba) states PyTorch ceilings of 2.14, matching pyproject.toml's seven '<2.15' torch bounds raised by quick 261025-e4w (commit f00e90d); zero stale pre-raise ceiling cells remain (baseline at HEAD: 6)"
    - "README.md's Project Structure tree lists all twelve on-disk dnallm/ subpackages, including the three currently missing: interpret/, preflight/, tui/ (baseline at HEAD: 0 of the three present)"
    - "Every pre-existing docs link in README.md still resolves — this plan touches no link (verified at planning time: docs/getting_started/gpu_support.md, install_honesty.md, installation.md, both docs/pic assets, and both zhangtaolab.github.io user_guide URLs are live files present in mkdocs.yml nav)"
  artifacts:
    - README.md (hardware-group PyTorch table, lines ~199-204, ceilings updated to 2.14)
    - README.md (Project Structure tree, lines ~466-488, three subpackage rows added)
  key_links:
    - "README hardware-group table <-> pyproject.toml [project.optional-dependencies] torch bounds (pyproject is the single source of truth; cross-check anchor: 7 '<2.15' occurrences, baseline at HEAD)"
---

<objective>
Fix the stale references in README.md left behind by recent repo evolution.
Investigation at planning time (2026-10-25, v1.4 tip b188f6c) found exactly two
objective stalenesses and zero dead links:

1. The hardware-group PyTorch table (six rows) still caps every lane at the
   pre-raise ceiling. Quick 261025-e4w (commit f00e90d) raised all seven
   torch bounds in pyproject.toml from <2.12 to <2.15, so the installable
   range now runs through 2.14.x. The README table is the user-facing mirror
   of those bounds and is one release behind.
2. The Project Structure tree omits three dnallm/ subpackages that exist on
   disk: interpret/ (FIMO-convention motif scanner + JASPAR REST client),
   preflight/ (v1.4 Phase 23 validation core), and tui/ (v1.4 Phase 25
   Textual pre-flight wizard — the milestone's flagship feature, which the
   README itself showcases further down at the wizard.svg block).

Purpose: README is the front door for installs; a table that contradicts
pyproject misleads users pinning torch and erodes doc trust.
Output: README.md only — two localized edits, no code, no link changes.

Out of scope (queued follow-ups from the e4w session, tracked separately):
stale torch-2.12 claims in docs/getting_started/installation.md and
docs/faq/models_troubleshooting.md, and the pyproject.toml:180 comment nit.
Do NOT touch any docs/ file in this item.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@pyproject.toml
@README.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update hardware-group table torch ceilings to match pyproject &lt;2.15</name>
  <files>README.md</files>
  <action>In README.md's "Hardware groups (mutually exclusive, NOT included in `all`)" table (six rows: cpu, cuda124, cuda126, cuda128, cuda130, mamba; around lines 199-204), update the PyTorch column ceiling from the stale pre-raise value to 2.14 in all six rows, preserving each row's floor exactly as-is: cpu and cuda124 become 2.4.0 through 2.14; cuda126, cuda128, and mamba become 2.6.0 through 2.14; cuda130 becomes 2.9.0 through 2.14. Source of truth: pyproject.toml's seven torch specifiers all read "torch>=X.Y.0,<2.15" (raised by quick 261025-e4w, commit f00e90d); an exclusive <2.15 bound means 2.14.x is the highest installable series, hence 2.14 as the table ceiling. Change nothing else in the table — the Use Case column and the surrounding prose (CUDA lane recommendation at line ~166-171, the retired cu121/ROCm note at ~206) are already accurate. Do not edit any docs/ file; the installation.md / models_troubleshooting.md torch claims are separate queued follow-ups outside this item.</action>
  <verify>
    <automated>grep -cE '\| 2\.(4|6|9)\.0-2\.11 \|' README.md | grep -qx 0 &amp;&amp; grep -cE '\| 2\.(4|6|9)\.0-2\.14 \|' README.md | grep -qx 6 &amp;&amp; grep -c '&lt;2.15' pyproject.toml | grep -qx 7 &amp;&amp; echo TASK1-PASS</automated>
  </verify>
  <done>All six hardware-group rows read through 2.14; zero stale ceiling cells remain (baseline 6 -> 0); six 2.14 cells present (baseline 0 -> 6); pyproject anchor still shows 7 "<2.15" bounds, confirming README now mirrors the resolver truth.</done>
</task>

<task type="auto">
  <name>Task 2: Add the three missing dnallm subpackages to the Project Structure tree</name>
  <files>README.md</files>
  <action>In README.md's Project Structure tree (the fenced block under "## 🏗️ Project Structure", around lines 466-488), add three rows to the dnallm/ listing so it matches the twelve subpackages actually on disk. Insert, matching the existing row indentation (three spaces after the vertical bar, box-drawing branch character) and the surrounding comment-column alignment: a row for interpret/ directly after the inference/ row with comment "Motif analysis (FIMO scanner, JASPAR client)"; a row for preflight/ directly after the models/ row with comment "Pre-flight validation core"; a row for tui/ directly after the tasks/ row with comment "Terminal pre-flight wizard (dnallm preflight)". Keep mcp/ as the final row with its corner-branch character unchanged, and touch no other part of the tree — the root-level entries (cli/, example/, docs/, tests/, ui/, scripts/, .github/, pyproject.toml, README.md) are all still accurate. Comments must be plain-language for biologists per the owner directive (no engineering jargon beyond the tool names).</action>
  <verify>
    <automated>grep -cE '│   ├── (interpret|preflight|tui)/' README.md | grep -qx 3 &amp;&amp; grep -c 'interpret/\|preflight/\|tui/' README.md &gt;/dev/null &amp;&amp; echo TASK2-PASS</automated>
  </verify>
  <done>The tree lists all twelve on-disk dnallm/ subpackages (cli, configuration, datahandling, finetune, inference, interpret, models, preflight, tasks, tui, utils, mcp); the three new rows use the same branch character and alignment as their neighbors; no other tree line changed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| N/A | Repository documentation edit only. No code, config, dependency, or network surface changes; no untrusted input is processed. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vvv-01 | Tampering (documentation integrity) | README.md hardware-group table | low | mitigate | A README that contradicts pyproject.toml's resolver bounds misleads users pinning torch versions. Mitigated by Task 1's pyproject cross-check gate (7 "<2.15" occurrences must hold) keeping table and source of truth in lockstep within this change. |
| T-vvv-SC | Tampering (supply chain) | package installs | low | accept | No package-manager installs occur in this plan (docs-only edit); the package-legitimacy gate is not applicable. |

</threat_model>

<verification>
Run all gates against the working tree after both tasks:

1. Stale ceilings gone: grep -cE '\| 2\.(4|6|9)\.0-2\.11 \|' README.md returns 0 (baseline at HEAD: 6).
2. New ceilings present: grep -cE '\| 2\.(4|6|9)\.0-2\.14 \|' README.md returns 6 (baseline at HEAD: 0).
3. Source-of-truth anchor: grep -c '<2.15' pyproject.toml returns 7 (recorded at HEAD; must be unchanged by this docs-only plan).
4. Tree completeness: grep -cE '│   ├── (interpret|preflight|tui)/' README.md returns 3 (baseline at HEAD: 0).
5. No collateral link damage: git diff --stat README.md shows a single file with only additive/edit lines in the two regions; no link targets were modified (all README docs links verified live at planning time).

No pytest run is required: README.md is not dnallm/ library code, so the ships-with-pytest owner rule does not apply, and no test in tests/ asserts on README content.
</verification>

<success_criteria>
- README.md hardware-group table mirrors pyproject.toml's <2.15 torch bounds (six rows ending at 2.14).
- README.md Project Structure tree reflects all twelve on-disk dnallm/ subpackages.
- Zero changes outside these two README regions; zero docs/ files touched.
- Both tasks committed atomically on the v1.4 branch with no Co-Authored-By trailers.
</success_criteria>

<output>
Create `.planning/quick/261025-vvv-fix-stale-docs-reference-in-readme-md/261025-vvv-SUMMARY.md` when done
</output>
