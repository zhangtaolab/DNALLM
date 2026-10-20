# Phase 13: Ollama Surface Disposition - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-11
**Phase:** 13-Ollama Surface Disposition
**Areas discussed:** Disposition tilt (skip vs remove), skip mechanism (stage-3 deletion vs unconditional marks), systemd unit fate, adjacent-contract handling, docs ride-along

---

## Language & Framing Clarification (pre-selection)

The initial four-area selection question was interrupted so the owner could (a) switch the discussion to Chinese and (b) correct a framing ambiguity. The owner's first free-text answer — "不是说 milestone 不讨论 ollama, 然后所有的ollama 有关的测试都要 skip。" — admitted two opposite readings (rejecting blanket-skip vs asserting blanket-skip), so Claude reflected both readings back against the recorded 2026-10-11 directive before proceeding.

**User's clarification:** "我的意思是 ollama mcp的 内容本 milestone 不讨论， ollama test skip. 只针对Ollama"
**Notes:** Resolved the ambiguity — the disposition targets only ollama assertions (skip, not remove); the ollama MCP feature/content (notebooks, dependency, server capability) is frozen and out of scope this milestone. This is NOT a blanket-skip-everything-ollama-named sweep; non-ollama contracts survive.

---

## Skip Mechanism (how "skip" lands)

| Option | Description | Selected |
|--------|-------------|----------|
| 删 stage 3，门控自愈 | Delete ci.yml stage 3; keep probe-then-execute gates. Nightly no longer brings ollama up → tests skip via `environment-unavailable:` with evidence; self-healing where ollama runs locally. Minimal test-file changes. | ✓ |
| 无条件 skip | Unconditional typed skip everywhere (never execute even with ollama up) + delete stage 3 + new expected_skips entries. Harder "parked" state but adds a forever-green skip class. | |

**User's choice:** 删 stage 3，门控自愈 (recommended option)
**Notes:** This also settles ci.yml stage-3 deletion and the nightly shape; the gate-semantics matrix tests stay green (they test gating logic, not ollama).

---

## Systemd Unit Fate

| Option | Description | Selected |
|--------|-------------|----------|
| 文件+pin 测试都留 | Keep `scripts/runner/ollama.service` and its D-06/D-12 pin tests in-repo and green (file-content assertions need no live ollama; the unit file is the restore path). GB10 host service may be stopped. | ✓ |
| 文件留，pin 也 skip | Keep the file, skip the pin tests too (literal "only ollama" reading). Cost: a green-for-nothing skip class. | |
| 文件+pin 全删 | Delete unit file + pins. Would be a content change the owner ruled out of discussion; restore path reduced to git history. | |

**User's choice:** 文件+pin 测试都留 (recommended option)
**Notes:** The GB10 service stop/disable is an owner ops action outside the repo — recorded as an ops note for the plan.

---

## Adjacent Contracts & Residue List

Not asked as a standalone question — settled by the clarification ("只针对Ollama") plus already-locked roadmap SC 2/SC 3: extras brackets, kernel-plot compat pins, docs-mirror contracts, models.lock docstring, and gate-semantics tests stay green and form the sanctioned `grep ollama tests/` residue list.

**User's choice:** (folded into the framing clarification)
**Notes:** Research Pitfall 5's per-assertion (not per-file) classification remains the planner's method.

---

## Docs Ride-Along

Not asked as a standalone question — Claude stated the default and the owner proceeded: minimal same-change accuracy fixes in `.github/workflows/README.md` and `scripts/runner/README.md` for text describing deleted stage-3 machinery; full prose sweep stays in Phase 19.

**User's choice:** accepted Claude's stated default
**Notes:** ci.yml inline comments (stage-2.5 "pre-ollama" naming, stage-3 block) are part of the same-change reconciliation regardless.

---

## Closure Gate

| Option | Description | Selected |
|--------|-------------|----------|
| 可以写 CONTEXT 了 | Decisions sufficient; write 13-CONTEXT.md. | ✓ |
| 再讨论其他方面 | Open more gray areas (census numbers, residue-list details, etc.). | |

**User's choice:** 可以写 CONTEXT 了

---

## Claude's Discretion

- Per-assertion ledger format in the plan (assertion-level granularity mandatory; numbering free)
- Exact post-sweep census numbers (measured at execution time)
- Stage-2.5 hygiene step's renamed role vs comment rewording (guard must survive)
- Sanctioned residue-list formatting

## Deferred Ideas

None — discussion stayed within phase scope.
