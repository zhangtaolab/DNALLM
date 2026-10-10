# Phase 3: Test Authoring to >90% Coverage - Context

**Gathered:** 2026-09-30
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — all 4 grey areas accepted as proposed

<domain>
## Phase Boundary

Line coverage on the agreed denominator exceeds 90%, closed biggest-gap-first with tests that verify observable behavior rather than merely executing lines.

In scope (requirements TEST-01..TEST-06): new tests for models+special dispatch (fault-injection, retry/reason-classification, tokenizer fallback), mcp/server transports + streaming + timeout errors, inference (engine, logits→predictions, interpret/mutagenesis/benchmark), datahandling/finetune, cli + compat shims (transformers_compat as behavior contract). Target: >90.5% (buffer above the gate).

Out of scope: the CI `fail_under` gate itself (Phase 4), vendored/unimportable-adapter coverage (excluded by denominator), new test frameworks.

Measured inputs (Phase 1 audit): baseline 45.92% (3,390/7,383 stmts); gap to 90% = 44.08 points ≈ 3,255 statements — inference 1,505 / models 1,210 (special 653) / mcp 449 / datahandling+finetune 441 / cli+utils 328. Ranked worklist: `01-AUDIT-REPORT.md` + `coverage-term-missing.txt` + `coverage.json`.

</domain>

<decisions>
## Implementation Decisions

### Sizing
- Single phase, ONE PLAN PER WAVE (~5 plans): models → mcp → inference → datahandling/finetune → cli/compat — the wave structure segments execution; fresh executor context per plan preserves fidelity; NO /gsd-phase split, NO roadmap renumbering
- Ordering authority: the ranked worklist (biggest-gap-first), NOT the ROADMAP's illustrative wave order — close inference's 1,505 before models' 1,210 if plan composition allows; wave dependency edges must stay acyclic
- Coverage re-measured after EACH wave (one command per Phase-1 tooling); course-correct early
- Landing target: >90.5% (buffer so minor refactors don't immediately trip the Phase-4 gate)

### MCP wave research (recorded STATE flag)
- Run a dedicated `gsd-plan-phase 3 --research-phase` pass BEFORE full planning — verifies transport/streaming/timeout patterns against the INSTALLED `mcp` 1.30.0 (the `server.py:1718+` seam)
- Transport tests: in-memory client/server with mocked transport (established `test_server_integration.py` pattern); NO live-network transport tests beyond existing typed network skips
- Streaming generator tests: consume generators to exhaustion, assert yielded sequence + final status, with fault-injection mid-stream

### Models-wave mock strategy
- Mock at the handler's own load calls (AutoModel.from_pretrained etc.) — exercises real dispatch/argument logic with lightweight fakes; no real tiny models
- Dispatch-chain coverage via sentinel fault-injection matrix (per-family selection + patch-all-but-one fall-through to generic loading)
- Tokenizer-fallback chain via staged failures (AutoTokenizer → PreTrainedTokenizerFast → DNAOneHotTokenizer), asserting which tier served

### Verification discipline & pragmas
- Every new test carries ≥1 observable-behavior assertion (TEST-06) — plan acceptance criteria + code review enforce; NO new enforcement tooling (framework ban)
- `# pragma: no cover` budget held at the recorded baseline of 3; additions require written justification in the wave SUMMARY
- Environment-bound branches (CUDA-only etc.): mock-based tests where behavior is assertable; otherwise documented as accepted-uncovered in the wave SUMMARY — never pragma'd
- Wave definition of done: area's ranked gaps closed AND full fast leg passes AND `scripts/audit_skips.py` exits 0 (allowlist absorbs any new intentional skips first)

### Claude's Discretion
Test names, file organization within the tests/ mirror, mock helper shapes — per codebase conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase-1 coverage tooling: config-driven `--cov`, `coverage report -m` / `coverage json` / term-missing; `01-AUDIT-REPORT.md` 43-row ranked worklist as the authoritative gap order
- Phase-2 typed-skip infrastructure (`_network_skip.py`, `expected_skips.yaml`, `scripts/audit_skips.py` + its 19 tests) — any new intentional skip must be allowlisted or the audit fails
- `tests/conftest.py` shared mocks; `tests/models/test_model.py` dispatch-test idioms (sentinel, `.to()` self-return); `dnallm/mcp/tests/test_server_integration.py` in-memory server pattern
- Fast leg at HEAD: 622 passed / 1 allowed skip / 78s

### Established Patterns
- One behavior per test, docstring on every test; `pytest.raises(match=...)`; mock at the import site; parametrize over task types; `Test*` class grouping
- English comments; ruff format (100 cols); absolute imports in tests

### Integration Points
- New test files under `tests/<subpackage>/` mirroring the package; possible new fixtures in `tests/conftest.py`
- `tests/expected_skips.yaml` must absorb any new intentional skips (audit is fail-closed)

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the accepted decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
