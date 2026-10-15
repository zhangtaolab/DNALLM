# Phase 20: Tech-debt closeout — CLI flag-only crash + residual docs mentions - Context

**Gathered:** 2026-10-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the three in-repo-fixable debt items from the v1.3 milestone audit (option B), nothing else:

1. **CLI flag-only invocation crash (deferred-items #1):** `dnallm-train` / `dnallm-inference` (and the `dnallm/cli/cli.py` equivalents) invoked with flags but without `--config` construct a minimal config with no `task` segment and crash with `KeyError: 'task'` inside DNATrainer/DNAInference. Phase 19 documented the CLI as `--config`-required in docs; the code must now match the docs — a loud, actionable error instead of the KeyError.
2. **`docs/faq/models_troubleshooting.md:13` residual `[cuda]`/`[rocm]` extras mention** (integration-checker warning): the v0.5-version-pinned troubleshooting context names extras that do not exist on the v1.3 surface — align the wording to the phase-19 standard (legacy context may stay legible but must not present retired extras as current installables).
3. **`task_guides.md` prose-side hf-route flag (deferred-items #2):** apply the 19-03/19-04 lock-alignment standard to the flagged prose site.

Explicitly OUT of scope (stay in the WINDOWS ledger / their own tracks): owner human-gates #25 (PyPI registration) / #27 (Windows KDA smoke) / #26 (pip-side alias resolve, box-network); the temp `- v1.3` push-filter removal (bound to the actual merge event); any new capability beyond the three fixes.

</domain>

<decisions>
## Implementation Decisions

### CLI fix shape (auto-selected under the standing autonomous mandate; recommendation accepted pattern)
- **D-01:** The flag-only path gets a **loud `click.UsageError`-class exit naming `--config` as required** (message pointing at the config-file docs), replacing the KeyError crash — NOT a synthesized default `task`. Rationale: docs already state `--config`-required (19-04 disposition); inventing a default task would re-open a semantics question the docs closed. Applied at all three sites: `dnallm/cli/cli.py` (train + inference commands), `dnallm/cli/train.py`, `dnallm/cli/inference.py`. The existing partial guard ("--model, --data, and --output are required when not using --config") is superseded by the clear --config-required exit — **Reversibility:** reversible — local exit-path change, no published contract.
- **D-02:** A regression test lands with the fix (flag-only invocation exits with the actionable message and non-zero code, no traceback/KeyError), in the existing `tests/cli/` directory — this phase is the milestone's **single census-moving edit**: the D-03 triple in ci.yml (`216/217 tests collected (1 deselected)`) is re-pinned in the SAME change to the new measured count, per the established one-census-move-per-phase discipline (13-01 precedent; expected_skips.yaml stays untouched — no new skips).

### Docs residue fixes
- **D-03:** faq `models_troubleshooting.md` keeps its version-pinned legacy context but marks the `[cuda]`/`[rocm]` extras as not-on-the-current-surface with a pointer to the current GPU-matrix/installation page — mirror/gate discipline applies if the file is in a mirrored pair (check `check_docs_sync` scope; same-commit byte-identical mirror if so).
- **D-04:** `task_guides.md` flagged prose follows the 19-03 distinction: runnable-example routes align to the lock-pinned route; catalog/prose mentions that merely name a variant stay unless they contradict the lock.

### Claude's Discretion
Test placement/naming inside `tests/cli/`, exact error message wording (must name `--config` and point to docs), whether cli.py's superseded partial guard line is removed or folded into the new message.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Defect record and standards
- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/deferred-items.md` — items #1 (CLI KeyError 'task': dnallm/cli/cli.py train/inference + train.py:36 + inference.py:36) and #2 (task_guides hf route)
- `.planning/v1.3-MILESTONE-AUDIT.md` — the audit that scoped this phase (option B)
- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-EVIDENCE.md` — 19-03 model-name lock-alignment standard + docs ledger conventions
- `models.lock` — the pinned-route registry the docs prose must not contradict

### Census discipline
- `.github/workflows/ci.yml` lines ~892-893 — the D-03 census triple this phase re-pins same-change

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/cli/` — existing CLI test directory (pattern source for the regression test)
- `dnallm/cli/cli.py` — click-based commands; already has an `if config: … else:` split at both command sites, plus a partial flag-guard to supersede

### Established Patterns
- One census-moving edit per phase: D-03 triple + (only if needed) expected_skips.yaml + ci.yml stage flags, same-commit
- Docs edits mirrored byte-identical same-commit where a mirror pair exists (`check_docs_sync` gate)
- Commits carry no Co-Authored-By trailers (repo convention, owner directive)

### Integration Points
- `dnallm/cli/{cli,train,inference}.py` → `DNATrainer`/`DNAInference` config contract (`task` segment required)
- pyproject console-script entry points (`dnallm-train`, `dnallm-inference`)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard approaches per the referenced disciplines.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope (owner human-gates #25/#26/#27 explicitly out of scope, tracked in the WINDOWS ledger).

</deferred>

---
*Phase: 20-Tech-debt closeout*
*Context gathered: 2026-10-15*
