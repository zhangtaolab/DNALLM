# Phase 13: Ollama Surface Disposition - Context

**Gathered:** 2026-10-11
**Status:** Ready for planning

<domain>
## Phase Boundary

A per-assertion remove-or-skip disposition over the ~59 ollama pytest refs across the 5 inventoried test files (`tests/examples/test_notebook_execution.py` 36, `tests/test_runner_infra_contracts.py` 23, `tests/examples/_execution.py` 6, `tests/test_extras_guard.py` 5, `tests/test_models_lock_contracts.py` 1), with the D-03 census triple, `tests/expected_skips.yaml`, and the ci.yml example-nightly stages reconciled in the same change. The owner's clarified intent (2026-10-11): the ollama test surface gets honestly skip-marked — the ollama MCP feature/content itself is frozen and out of discussion for this milestone.

</domain>

<decisions>
## Implementation Decisions

### Disposition Policy

- **D-01: Tilt = skip, not remove.** Every ollama-targeted assertion carries a skip disposition with recorded rationale in the plan ledger — deletion is NOT the disposition for the live ollama surface. (Owner clarification, verbatim: "ollama test skip. 只针对 Ollama".) This resolves the research's open remove-vs-skip question in favor of skip everywhere the assertion is genuinely ollama-execution surface. — **Reversibility:** reversible — skips can be un-marked assertion by assertion later.
- **D-02: Only ollama-targeted refs are touched.** Non-ollama contracts sharing the swept files stay green and untouched: kernel-plot compat pins (`TestNotebookKernelPlotCompat`, ipython<9), extras brackets (`langchain-ollama>=1.1.0` in the mcp extra), docs-mirror content contracts (`"ollama:qwen3.5:4b"` in notebook + mirror), the models.lock docstring mention, and the gate-semantics matrix tests (they test gating logic via monkeypatch, not ollama). These kept-green refs form the sanctioned `grep ollama tests/` residue list recorded in the plan.
- **D-03: Ollama MCP content is frozen this milestone.** The two mcp_example notebooks, the `langchain-ollama` dependency, and the MCP server's ollama client capability are out of discussion — no edits, no removals, no re-gating of content.

### Skip Mechanism & Nightly Shape

- **D-04: Skip lands via ci.yml stage-3 deletion + existing probe-then-execute gates (self-healing), NOT unconditional skip marks.** The example-nightly stage 3 (ollama batch that re-runs the two gated mcp tests with ollama + MCP server up) is deleted. With ollama no longer brought up on the nightly, `_gate_ollama_stack` fires its existing `environment-unavailable:` typed skip with live probe evidence — the `environment-unavailable:` prefix entry already exists in `tests/expected_skips.yaml`. On any box where ollama IS running (local dev), the tests still execute — self-healing by design, matching the research-recommended defensible state. — **Reversibility:** reversible — re-adding a stage-3-shaped batch restores nightly execution.
- **D-05: `scripts/runner/ollama.service` and its pin tests stay in-repo and green.** The `TestOllamaUnitPins` D-06/D-12 Environment-line assertions read the committed unit file only — no live ollama needed — and the unit file constitutes the explicit restore path. The GB10 host's systemd service itself may be stopped/disabled as an owner ops action (outside the repo; note it in the plan's ops section).

### Mechanical Consequences (locked by the above)

- Census re-pin: the nightly P/S distribution shifts (the mcp pair moves from PASS in stage 3 to SKIP in the main batch); the D-03 census triple re-pins same-change; no pytest exit 5 (collection counts unchanged — the tests stay collected).
- ci.yml inline accuracy: stage-2.5 "pre-ollama" naming and the stage-3 block comments are corrected in the same change.
- README accuracy ride-along: `.github/workflows/README.md` and `scripts/runner/README.md` sentences describing stage 3 / the ollama loopback get minimal same-change accuracy fixes for deleted machinery; the full docs prose sweep stays in Phase 19 (DOCR-01).

### Claude's Discretion

- Per-assertion ledger format/numbering in the plan (assertion-level, not file-level — research Pitfall 5 warning sign).
- Exact census numbers after the re-pin (measured, not estimated).
- Whether the stage-2.5 hygiene step keeps a renamed role as end-of-run hygiene or its comment is simply reworded (kernel cleanup + memory floor is non-ollama discipline; do not delete the guard).
- Sanctioned residue-list formatting in the plan.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap & Requirements
- `.planning/ROADMAP.md` §"Phase 13: Ollama Surface Disposition" — goal, dependencies, 4 success criteria
- `.planning/REQUIREMENTS.md` §"Ollama Surface (OLLM)" — OLLM-01, OLLM-02, OLLM-03

### v1.3 Research (owner-directed; conclusions baked into decisions above)
- `.planning/research/PITFALLS.md` §"Pitfall 5 (O)" — the non-ollama-contract deletion hazard; per-assertion classification guidance and sanctioned residue rule
- `.planning/research/FEATURES.md` §5 — ecosystem norms (PostHog delete-don't-park vs probe-then-execute for supported features); anti-features list (no blanket skip, no capability removal, keep live-probe evidence)
- `.planning/research/SUMMARY.md` §"Implications for Roadmap" (Phase O section) — deliverables shape

### Test Infrastructure Contracts
- `.planning/codebase/TESTING.md` — typed-skip discipline, expected_skips.yaml matcher semantics, D-03 census gate, audit_skips fail-closed behavior
- `tests/expected_skips.yaml` — the live allowlist (note: `environment-unavailable:` prefix entry already present)
- The 5 swept files: `tests/examples/test_notebook_execution.py`, `tests/test_runner_infra_contracts.py`, `tests/examples/_execution.py`, `tests/test_extras_guard.py`, `tests/test_models_lock_contracts.py`

### CI & Runner Surface
- `.github/workflows/ci.yml` — example-nightly job: stage-2.5 hygiene step, stage-3 ollama batch, D-03 census assertion
- `scripts/runner/ollama.service` — the pinned unit file (stays in-repo per D-05)
- `scripts/runner/README.md` — runner ops doc (minimal same-change accuracy fix)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_gate_ollama_stack` + `_probe_http_with_retry` (`tests/examples/test_notebook_execution.py:1215-1325`) — the probe-then-execute gate that makes D-04 work with zero test-file logic changes
- Typed-skip helpers in `tests/examples/_execution.py` (`environment_unavailable_skip` etc.) — already emit the junit-greppable prefixes the audit matches
- `scripts/audit_skips.py` + junit skip audit — fail-closed enforcement that keeps the new nightly skips honest

### Established Patterns
- Probe-then-execute with live evidence carried in skip messages (v1.1 decision; "an ever-green skip is the same dishonesty class as a false-green gate" — D-04 preserves this: the skip fires with real probe evidence, and self-heals where ollama exists)
- One census-moving edit per phase; D-03 triple + expected_skips.yaml + ci.yml stages re-pinned in the same change (cross-cutting v1.3 invariant)
- Typed skips are the ONLY sanctioned skip paths; every skip message must match an allowlist entry

### Integration Points
- ci.yml example-nightly: stage-3 deletion + census re-pin + stage-2.5 comment rewording land together
- `.github/workflows/README.md` + `scripts/runner/README.md`: minimal accuracy fixes for deleted stage-3 machinery
- GB10 runner ops (outside repo): ollama loopback service stop/disable — recorded as an ops note, not a repo change

</code_context>

<specifics>
## Specific Ideas

Owner's framing, verbatim (2026-10-11 discussion, translated context): "ollama mcp 的内容本 milestone 不讨论， ollama test skip. 只针对 Ollama" — the ollama MCP content is not up for discussion this milestone; the ollama tests skip; the sweep targets only ollama. Also confirmed earlier in the discussion: NOT a blanket-skip-of-everything-ollama-named reading — non-ollama contracts survive (D-02), and the skip mechanism is the existing self-healing gate (D-04), chosen over unconditional skip marks.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Full docs prose accuracy for runner/workflow READMEs beyond the minimal same-change fixes belongs to Phase 19's DOCR-01 sweep by design, not as a new deferral.)

</deferred>

---

*Phase: 13-Ollama Surface Disposition*
*Context gathered: 2026-10-11*
