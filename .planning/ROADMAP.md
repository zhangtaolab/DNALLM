# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 Paper Revision Suite Support** — Phases 10–12 (shipped 2026-10-10) — [archive](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Platform Reach & Packaging Honesty** — Phases 13–21 (shipped 2026-10-16; reopened 2026-10-17 for owner-directed Phase 21) — [archive](milestones/v1.3-ROADMAP.md)

## Phases

- [ ] **Phase 21: Type-checker atomic swap — ty hard flip + mypy retirement (D-09)** — in progress (owner-directed post-ship addition, 2026-10-17)

**Goal:** ty (pinned `ty@0.0.84`) becomes the sole type gate and mypy retires in ONE atomic change (the D-08/D-09 discipline set in v1.1 Phase 9). The 232-diagnostic baseline clears to zero via real fixes plus line-level `# ty: ignore[code]` pragmas for library dynamics — no severity overrides, no new `[tool.ty.src]` excludes. Same change: the coverage-gate advisory ty step hardens (`|| true` dropped), the two test-job advisory mypy steps retire, the pre-commit mypy hook retires, `check_code.py` swaps mypy→ty, and the mypy dev-dep + `[tool.mypy]`/overrides + `.mypy_cache` gitignore entry are removed. `# type: ignore` comments stay — ty honors them live.

<details>
<summary>✅ v1.3 Platform Reach & Packaging Honesty (Phases 13–20) — SHIPPED 2026-10-16</summary>

- [x] Phase 13: Ollama Surface Disposition (1/1 plans) — completed 2026-10-13
- [x] Phase 14: GAP-1 Transformers 5.x Disposition (2/2 plans) — completed 2026-10-13
- [x] Phase 15: Windows Parity Core — Numpy Interval Evaluators (1/1 plans) — completed 2026-10-13
- [x] Phase 16: Windows Parity Rollout — Dispatch Gates & Frozen Value Parity (4/4 plans) — completed 2026-10-14
- [x] Phase 17: Publish Chain Robustness (3/3 plans; 17-03 halted at owner-deferred PyPI registration, WINDOWS #25) — completed 2026-10-15
- [x] Phase 18: Packaging & Install Honesty — Atomic Extras Surgery (3/3 plans; 18-03 halted at owner-deferred Windows KDA smoke, WINDOWS #27) — completed 2026-10-15
- [x] Phase 19: Docs Accuracy Sweep & Milestone Closeout (7/7 plans) — completed 2026-10-15
- [x] Phase 20: Tech-debt closeout — CLI flag-only crash + residual docs mentions (1/1 plan; post-audit option B) — completed 2026-10-16

Full phase details, requirements mapping, and success criteria: [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)

Closeout record: v1.3-MILESTONE-AUDIT.md (tech_debt verdict cleared by Phase 20; 27/29 requirements satisfied — PUBS-04/WPAR-07 acceptance halves owner-deferred as WINDOWS #25/#27, carried forward).

</details>

<details>
<summary>✅ v1.2 Paper Revision Suite Support (Phases 10–12) — SHIPPED 2026-10-10</summary>

- [x] Phase 10: Evaluation Contract Layer & Shared Scaffolding (4/4 plans) — completed 2026-10-09
- [x] Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities (5/5 plans) — completed 2026-10-09
- [x] Phase 12: Motif Matching, MCP Tools & Milestone Closeout (3/3 plans) — completed 2026-10-10

Full phase details, requirements mapping, and success criteria: [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)

</details>

<details>
<summary>✅ v1.1 Example Execution Testing & Repair (Phases 5–9) — SHIPPED 2026-10-07</summary>

- [x] Phase 5: Execution Harness, Honest Gates & Runner Feasibility (6/6 plans) — completed 2026-10-03
- [x] Phase 6: Model Registry & Showcase Data Curation (3/3 plans) — completed 2026-10-03
- [x] Phase 7: PlantHelixSeek Showcase Notebooks (2/2 plans) — completed 2026-10-04
- [x] Phase 8: Full Execution Rollout & Repair Loop (9/9 plans) — completed 2026-10-05
- [x] Phase 9: CI Wiring & Census Verification (4/4 plans) — completed 2026-10-06

Full phase details, requirements mapping, and success criteria: [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1 Test Suite Audit & Coverage Hardening (Phases 1–4) — SHIPPED 2026-10-01</summary>

- [x] Phase 1: Harness Integrity & Measured Baseline (2/2 plans) — completed 2026-09-30
- [x] Phase 2: Suite Hygiene & Known-Bug Fixes (3/3 plans) — completed 2026-09-30
- [x] Phase 3: Test Authoring to >90% Coverage (5/5 plans) — completed 2026-09-30
- [x] Phase 4: CI Gate Enforcement (3/3 plans) — completed 2026-10-01

Full phase details, requirements mapping, and success criteria: [milestones/v1-ROADMAP.md](milestones/v1-ROADMAP.md)

</details>

**Phase Numbering:**
- The next milestone continues from Phase 22 (numbering never restarts; Phase 21 consumed by the v1.3 reopen)
- Decimal phases (e.g. 21.1): urgent insertions (marked INSERTED)
