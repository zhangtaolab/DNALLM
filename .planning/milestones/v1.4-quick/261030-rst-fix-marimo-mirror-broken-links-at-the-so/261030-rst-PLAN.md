---
phase: 261030-rst
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md
autonomous: true
requirements:
  - DOCS-12-MKDOCS-STRICT
must_haves:
  truths:
    - "The phase-12 mkdocs strict item is resolved with dated evidence for the fix that already landed and a fresh zero-warning build"
    - "The HBG1 golden-fixture and numpy coverage deferred items remain byte-for-byte unchanged and open"
    - "The marimo generator, all generated and hand-maintained documentation, and repository code remain unchanged"
  artifacts:
    - ".planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md — resolved mkdocs bullet with 2026-10-30 evidence and a regeneration-preservation warning"
  key_links:
    - "Canonical per-bullet resolved status -> stale mkdocs item no longer appears actionable"
    - "Commit 47b9bc2 plus fresh MkDocs 1.6.1 strict-build evidence -> auditable closure at baseline 8b224ea"
    - "Hand-maintained-mirror annotation -> future regeneration cannot silently clobber enriched prose"
---

<objective>
Close only the stale phase-12 mkdocs strict deferred item. Record that commit
47b9bc2 already fixed all 15 warnings and that a fresh build at baseline 8b224ea
passed with MkDocs 1.6.1, while preserving the unrelated HBG1 and numpy items.

Purpose: reconcile the planning record with the verified repository state without
making the source or mirror changes that the corrected scope explicitly forbids.

Output: one scoped deferred-item edit, a dated resolution annotation with strict-
build and hand-maintenance evidence, and one conventional commit containing only
that deferred-item change.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md
@mkdocs.yml

Planning baseline observed on branch v1.4 at 8b224ea: the worktree is clean and
`.venv/bin/mkdocs` is MkDocs 1.6.1. A live `.venv/bin/mkdocs build --strict`
completed successfully in 4.78 seconds with zero WARNING lines. Commit 47b9bc2,
`docs(19-01): strict-clean — 15 mkdocs warnings to zero`, already repaired the
14 marimo mirror links and the CONTRIBUTING link. A dry-run regeneration of the
benchmark mirror produced extensive non-link deletions because the generator is
bootstrap scaffolding and does not own the mirrors' enriched prose.

Source audit: corrected GOAL and REQ are the single stale mkdocs status closure,
covered end to end by Task 1. The corrected CONTEXT constraints are also covered:
only that bullet changes, the two unrelated bullets remain untouched, no source
or mirror regeneration occurs, and the deferred flip is committed. The superseded
generator/regeneration approach is explicitly excluded by the orchestrator's
corrected scope. There is no RESEARCH.md and no unplanned source item.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Resolve the stale mkdocs bullet with preservation evidence</name>
  <files>.planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md</files>
  <action>
    Use Edit for one scoped replacement in the existing deferred-items file. In
    only the section headed `[Out-of-scope discovery — 12-03 Task 1] mkdocs
    --strict fails on 15 pre-existing warnings`, replace `status: acknowledged`
    with `status: resolved`. Preserve the section's historical Found/Issue/Proof/
    Why/Suggested text as context; the new annotation supersedes its stale state.

    Add this annotation immediately after the resolved status, retaining bullet
    indentation and using English prose: `**Resolved 2026-10-30:** commit 47b9bc2
    (docs(19-01): strict-clean — 15 mkdocs warnings to zero) already fixed all
    15 warnings. Fresh verification at HEAD 8b224ea with MkDocs 1.6.1:
    .venv/bin/mkdocs build --strict exited 0 with zero WARNING lines. The three
    docs/example/marimo mirrors are hand-maintained; their Prerequisites,
    Overview, and See-Also prose lives only in the mirrors.
    scripts/generate_md_from_marimo.py is bootstrap scaffolding only, and a
    dry-run regeneration produced extensive non-link deletions. Do not
    regenerate without a preservation redesign.` Keep the annotation as one
    logical paragraph; wrapping its source lines is acceptable.

    Do not alter the HBG1/BCL11A section, the numpy section, either remaining
    `status: acknowledged`, or any heading. Do not modify
    scripts/generate_md_from_marimo.py, docs/example/marimo/, any other product
    documentation, repository code, tests, or configuration. Do not regenerate
    mirrors, install packages, run pytest, or clean caches.

    Run the complete verification gate before committing and inspect the scoped
    diff. Commit only the deferred-items file with subject
    `docs(quick-261030-rst): close stale mkdocs strict deferred item`; include no
    attribution trailers. The verification commands use fixed baseline 8b224ea,
    so they remain valid both before and after that commit.
  </action>
  <verify>
    <automated>.venv/bin/mkdocs --version && .venv/bin/python -I -c 'import re, subprocess; p=subprocess.run([".venv/bin/mkdocs", "build", "--strict"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT); print(p.stdout, end=""); assert p.returncode == 0, p.returncode; assert not re.search(r"(?m)^WARNING\s", p.stdout), "strict build emitted warnings"; print("PASS: strict build exited 0 with zero WARNING lines")' && .venv/bin/python -I -c 'from pathlib import Path; import subprocess; path=".planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md"; baseline=subprocess.check_output(["git", "show", "8b224ea:"+path], text=True); current=Path(path).read_text(); heading="## [Code review — Phase 12 IN-05] numpy 2.5.x cannot be instrumented by coverage on Python 3.13"; old="  status: acknowledged\n\n"+heading; note="  **Resolved 2026-10-30:** commit 47b9bc2"; assert baseline.count(old) == 1; mkdocs=current.split("## [Out-of-scope discovery — 12-03 Task 1] mkdocs --strict fails on 15 pre-existing warnings", 1)[1].split(heading, 1)[0]; assert "  status: resolved\n"+note in mkdocs; assert all(text in mkdocs for text in ("15 mkdocs warnings to zero", "HEAD 8b224ea", "MkDocs 1.6.1", "zero WARNING lines", "hand-maintained", "Prerequisites", "Overview", "See-Also", "bootstrap scaffolding only", "dry-run regeneration", "preservation redesign")); before_sections=baseline.split("\n## "); after_sections=current.split("\n## "); assert len(before_sections) == len(after_sections) == 4; changed=[before.split("\n", 1)[0] for before, after in zip(before_sections, after_sections, strict=True) if before != after]; assert changed == ["[Out-of-scope discovery — 12-03 Task 1] mkdocs --strict fails on 15 pre-existing warnings"], changed; assert before_sections[1] == after_sections[1] and before_sections[3] == after_sections[3]; assert current.count("status: acknowledged") == 2; print("PASS: only the mkdocs section changed; HBG1 and numpy sections are unchanged")' && git diff --check 8b224ea -- .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md && git diff --quiet 8b224ea -- docs scripts dnallm tests example mkdocs.yml pyproject.toml && rst_diff="$(git diff --unified=0 8b224ea -- .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md)" && printf '%s\n' "$rst_diff" | rg '^[+-][^+-]'</automated>
  </verify>
  <done>The mkdocs bullet alone has lowercase `status: resolved` and the complete dated evidence/preservation annotation; strict MkDocs exits 0 with no WARNING lines; fixed-base comparison proves both unrelated sections are unchanged; product code/docs/config have no diff; only the deferred-item change is committed.</done>
  <precondition>Run from the checkout root on branch v1.4 with the existing .venv and MkDocs 1.6.1; baseline 8b224ea and fix commit 47b9bc2 are present, and no product code or documentation edits are in progress.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| deferred-item record -> owner triage | per-bullet status and evidence determine whether resolved work is reopened |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-rst-01 | Tampering | Phase-12 shared deferred-items file | medium | mitigate | Compare all sections against fixed baseline 8b224ea and require the mkdocs section to be the sole changed section, leaving HBG1 and numpy byte-identical. |
| T-rst-02 | Tampering | Hand-maintained marimo mirrors | high | mitigate | Make no generator or mirror edits and record the dry-run-proven regeneration hazard in the closure annotation. |
| T-rst-SC | Tampering | npm/pip/cargo installs | low | accept | Reserved supply-chain row: this plan installs no packages and adds no dependencies. |
</threat_model>

<verification>
The single tracer task performs the complete evidence-only closure: it changes
the one stale status, adds the prior-fix and fresh-build evidence, and records
the preservation constraint that prevents future documentation loss.

The automated gate reruns the exact available MkDocs 1.6.1 command and rejects
both a nonzero exit and any WARNING line. A fixed-base whole-file section
comparison proves only the mkdocs section changed and that the HBG1 and numpy
sections remain untouched. Scoped diff checks reject whitespace errors and any
product code/docs/config change; the final changed-line listing provides the
requested concise diff evidence. No pytest or Ruff run is needed because only
.planning Markdown changes.
</verification>

<success_criteria>
- DOCS-12-MKDOCS-STRICT is resolved with commit 47b9bc2, baseline 8b224ea, MkDocs 1.6.1, and zero-warning strict-build evidence dated 2026-10-30.
- The annotation identifies the three mirrors as hand-maintained and forbids regeneration without a preservation redesign.
- The HBG1 and numpy bullets remain open and byte-for-byte unchanged.
- scripts/generate_md_from_marimo.py and all repository code, product docs, tests, examples, and configuration remain unchanged.
- Exactly one deferred-items file is committed with the specified conventional subject and no attribution trailers.
</success_criteria>

<output>
Create `.planning/quick/261030-rst-fix-marimo-mirror-broken-links-at-the-so/261030-rst-SUMMARY.md` when done. Record the resolved status, both evidence sources, the fresh strict-build result, fixed-base proof that only the mkdocs section changed, the deferred-flip commit hash, and confirmation that neither regeneration nor repository code/docs changes occurred.
</output>
