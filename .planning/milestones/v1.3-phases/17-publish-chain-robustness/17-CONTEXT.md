# Phase 17: Publish Chain Robustness - Context

**Gathered:** 2026-10-14
**Status:** Ready for planning

<domain>
## Phase Boundary

The publish chain exercised end-to-end on demand and on schedule with zero upload possibility: event-gated upload guard (first act regardless of position), never-uploading smoke chain (build → twine check → clean-venv install-from-dist → import probe reading dist/), OIDC Trusted Publishing migration, WINDOWS #20 cov crash remediation via local numpy pin, mypy dev-extra cap. Requirements PUBS-01..05 + PKGH-08.

</domain>

<decisions>
## Implementation Decisions

### 发布链决策 (owner-accepted 2026-10-14)
- Upload guard = EVENT GATING: dispatch/cron can never reach `twine upload`; real publishing only on the tag/push event path; the verify step reads `dist/`, never pip-installs from PyPI on non-publish runs
- Smoke schedule: WEEKLY cron (off-peak) + manual workflow_dispatch
- WINDOWS #20 fix: LOCAL NUMPY PIN (fix the cause) with the coverage-CLI workaround documented as the backup note — NOT a coverage version bump
- OIDC sequencing: workflow-side ready first (job-level id-token: write, pypa/gh-action-pypi-publish@release/v1, publisher registered for exact owner/repo/workflow-filename/environment, no reusable-workflow factoring, no token fallback); PyPI-side trusted-publisher REGISTRATION and PYPI_API_TOKEN deletion are OWNER human-gates — token deleted only after the first observed green OIDC release

### Carried constraints (roadmap)
- twine pinned >=7,<8; redundant build dep dropped; `uv build --no-sources`; mypy capped >=1.15.0,<2.5 in dev extra
- Smoke dispatches validate Phase 18's extras surgery (position note)

### Claude's Discretion
Workflow yaml mechanics, guard predicate shape, pin placement, smoke job structure within the locked contracts.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- .github/workflows/publish.yml (the current publish chain — WINDOWS #20's publish-run informational drift note references its env installs)
- The phase-14/16 CI dispatch patterns (event gates per cron-string precedent D-19)
- ci.yml coverage discipline (coverage-CLI form — the #20 workaround basis)

### Established Patterns
- Event gating by github.event_name / github.event.schedule (D-19 cron-string gates)
- WINDOWS ledger bookkeeping; owner human-gate tasks (16-01 checkpoint precedent)

### Integration Points
- PyPI trusted-publisher registration (owner account action); repo secrets (PYPI_API_TOKEN deletion after first green OIDC); dev extras (mypy cap); local venv (numpy pin)

</code_context>

<specifics>
## Specific Ideas

PyPI filenames are immutable — the upload guard must land BEFORE any smoke trigger exists. The verify step's current pip-install-from-PyPI behavior on non-publish runs is itself part of the guard surface.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
