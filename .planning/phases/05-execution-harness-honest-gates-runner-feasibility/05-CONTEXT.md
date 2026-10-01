# Phase 5: Execution Harness, Honest Gates & Runner Feasibility - Context

**Gathered:** 2026-10-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Delivers the trustworthy foundation for all later v1.1 execution work: (1) the private nbclient execution harness in `tests/examples/` with tmp-sandbox cwd isolation, per-cell-inside-per-test timeout layering, context-managed kernel shutdown, and partial-notebook failure artifacts — proven on a 1–2 notebook pilot plus a deliberate-hang kill test; (2) WR-08/09 closed: every `continue-on-error` in docs-validation removed with the docs-mirror drift it was hiding, `mcp` extra installed, README fixed, and docs-validation promoted to a required branch-protection check; (3) the GB10 feasibility verdict matrix (evo-1 / evo2 / megaDNA / pyBigWig + marimo execution flavor) produced with real-forward evidence and recorded per family.

Requirements in scope: EXEC-01, EXEC-06, CI-01, CI-02, REPAIR-02, FEAS-01. Not in scope: full rollout of all 21 notebooks (Phase 8), registry/loci work (Phase 6), showcase notebooks (Phase 7), models.lock extension and cache tiers beyond what the spike itself needs (Phase 8), nightly census wiring (Phase 9).

</domain>

<decisions>
## Implementation Decisions

### WR-08 honesty scope
- **D-01:** ALL five `continue-on-error` steps in `docs-validation.yml` are flipped honest in this phase (check_docs_sync, validate_docs_snippets, validate_yaml, run example tests, run YAML load tests). If the snippets/YAML validators expose latent failures once unmasked, they are repaired to green within this phase — no advisory remnants. — **Reversibility:** costly — re-adding masking later would need to touch the workflow again and would re-open the exact false-green class the milestone exists to close.
- **D-02:** `docs-validation` is promoted to a required branch-protection check on dev+main, same tier as `coverage-gate`. — **Reversibility:** reversible — branch-protection rules are owner-runnable API edits.

### Notebook stale outputs
- **D-03:** The 19/21 notebooks with committed outputs keep them as-is; the docs mirror is resynced byte-identical from the current state (outputs included). Output refresh happens only when Phase 8 repairs a given notebook (and via the Phase 7 showcase write-back for the two PlantHelixSeek notebooks). No bulk output-stripping in this phase. GitHub render appearance is unchanged in the interim; stale outputs in the mirror for a few phases is accepted.

### GB10 feasibility spike
- **D-04:** Spike executes locally on the dev box first (same GB10 hardware class as the runner, verified via nvidia-smi) for fast iteration, then the resulting verdict matrix is re-confirmed once on the self-hosted nightly runner via `workflow_dispatch` to become the official verdict. The runner's schedule/dispatch-only security posture is untouched — PR-authored code never reaches it.
- **D-05:** Verdict depth uses the EXACT model variants the notebooks reference (e.g. `togethercomputer/evo-1-131k-base`, not a smaller evo-1): a family is FEASIBLE only after a real forward pass with the notebook variant succeeds, with time/VRAM/disk evidence recorded in the matrix. pyBigWig's entry needs import + a small real BigWig write/read. — **Reversibility:** costly — a verdict recorded against a different variant than the notebooks use would have to be re-derived before any Phase 8 skip decision.
- **D-06:** If a family's notebook variant fails on GB10, the same spike falls back to that family's smallest viable variant; if the small variant runs, Phase 8 executes with the small variant and the notebook's model reference is updated accordingly. Typed `environment-unavailable:` skip is used only when both variants fail, with recorded evidence.

### Carried-forward decisions (locked earlier, not re-discussed)
- nbclient used as a library inside parametrized pytest tests — no nbmake, no new test frameworks
- Timeout layering: nbclient per-cell timeout as the inner guard, `@pytest.mark.timeout(N)` as the outer backstop
- Harness mechanics live in `tests/examples/_execution.py` + a locally-scoped `tests/examples/conftest.py` — never root conftest, never inside `dnallm/`
- marimo apps execute via subprocess (export-html vs script-mode flavor decided by the Phase 5 pilot itself)
- Kernel management: context-managed `NotebookClient`, `shutdown_kernel="immediate"`, plus a nightly `pkill` hygiene step
- WR-08 flip and mirror-drift closure land in the same reviewable unit
- ModelScope-first model sourcing (applies to Phase 8 lock entries; spike downloads may use either hub)

### Claude's Discretion
- Choice of the 1–2 pilot notebooks (pick already-healthy, fast, no-huge-download ones)
- Typed-skip prefix naming details (`environment-unavailable:` vs `optional-dep:` assignment per family) as long as both are registered in `expected_skips.yaml` with the audit green
- Sandbox fixture mechanics (copy strategy, artifact layout) within the locked tmp-sandbox + cwd-redirect pattern
- Verdict-matrix document format and location (suggested: `.planning/` artifact + phase docs)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone research (grounding for every decision above)
- `.planning/research/SUMMARY.md` — synthesized stack/features/architecture/pitfalls; Phase-5-attributed pitfall mapping (kernel leaks, timeout arithmetic, hermeticity, mirror drift)
- `.planning/research/STACK.md` — nbclient/marimo verified semantics (traits, cwd handling, per-cell timeout, script-mode exit codes), extras placement (langchain-ollama → mcp, pyBigWig → dev)
- `.planning/research/ARCHITECTURE.md` — harness layout (`_execution.py` + conftest seam mirroring `dnallm/mcp/tests/_network_skip.py`), marker plan, verified mirror-drift findings
- `.planning/research/PITFALLS.md` — kernel-leak mechanisms (pytest-timeout issues #134/#159), cache quota, GB10 toolchain constraints, cwd false-repair trap

### Workflow and scripts under repair
- `.github/workflows/docs-validation.yml` — the five `continue-on-error` steps to flip (WR-08) and the `.[test,dev]` install missing `mcp` (WR-09)
- `scripts/check_docs_sync.py` — exits 1 today; needs wrapper-`.md` handling before the flip
- `README.md` "Local Testing" section — WR-09 stale install line to correct

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` — v1.1 REQ-IDs; Phase 5 = EXEC-01, EXEC-06, CI-01, CI-02, REPAIR-02, FEAS-01
- `.planning/ROADMAP.md` §"Phase 5" — goal and 4 success criteria

### Test-infra precedents
- `tests/examples/test_examples.py` — existing structural layer the new execution modules sit beside (untouched)
- `tests/expected_skips.yaml` + `scripts/audit_skips.py` — typed-skip allowlist the new prefixes must register into
- `.github/workflows/ci.yml` — coverage-gate/nightly census structure; branch-protection context "coverage-gate (py3.12, fast leg)" precedent for adding docs-validation as required

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/conftest.py` mock fixtures (mock_model, mock_tokenizer, …) — usable by harness unit tests without new scaffolding
- `dnallm/mcp/tests/_network_skip.py` — the private-helper + typed-skip seam the harness mirrors (`network-unavailable:` prefix matching already allowlisted)
- Root `conftest.py` session teardown (multiprocessing kill, CUDA cache clear, `os._exit(0)`) — already guards CI hangs; the harness adds kernel-level cleanup beneath it
- `pytest.mark.timeout` (pytest-timeout 2.4.0) — per-test marker overrides the 300s ini default; 7 existing per-test marks set precedent

### Established Patterns
- Slow-marker leg split (`-m "not slow"` fast / bare census nightly) — execution tests join the slow side; zero new fast-leg skips by construction
- Skip discipline: every skip typed with a prefix and matched against `expected_skips.yaml` by the out-of-process audit — new prefixes follow the same contract
- v1 Phase-2 PDF fix (autouse tmp_path rebind + twice-run tree-clean proof) — the exact pattern generalized to notebook sandboxes; `git status --porcelain` guard is the tripwire

### Integration Points
- `pyproject.toml [tool.pytest.ini_options]` — markers already registered; no ini changes expected beyond none needed for Phase 5
- `pyproject.toml [project.optional-dependencies]` — `mcp` extra gains `langchain-ollama` (WR-09); `notebook` extra already carries nbclient; `dev` gains pyBigWig only if the spike proves wheels on GB10
- Branch protection on dev+main — new required check added via owner-runnable `gh api` PUT (exact commands to be handed over as v1 did for coverage-gate)

</code_context>

<specifics>
## Specific Ideas

- Owner emphasis throughout: the example layer must eventually run with REAL models everywhere — the spike's job is to find the path to real execution (notebook variant → smallest variant → only then typed skip), never to reach for a skip first
- Owner on pushing: milestone v1.1 is manual-push-only; all workflow commits stay local with the unpushed range reported

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 5-Execution Harness, Honest Gates & Runner Feasibility*
*Context gathered: 2026-10-02*
