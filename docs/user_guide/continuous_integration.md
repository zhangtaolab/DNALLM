# Continuous Integration and Testing

This page summarizes how DNALLM is tested in continuous integration: the nightly job topology, what the coverage gate does (and deliberately does not) measure, the giant-model policy, and the typed-skip audit. The detailed, authoritative description of every workflow and job lives in [`.github/workflows/README.md`](https://github.com/zhangtaolab/DNALLM/blob/main/.github/workflows/README.md) — this page only summarizes it and explains the expectations behind it.

## Nightly Topology

Two schedule entries drive three nightly jobs on the self-hosted GPU runner (`dnallm-nightly`):

| Schedule (UTC) | Job | Content |
|----------------|-----|---------|
| 03:00 (`0 3 * * *`) | `coverage-nightly` | Full coverage census including the `slow` tests (real HF/ModelScope model downloads) under the 90% floor |
| 03:00 (`0 3 * * *`) | `test-mamba` | Native mamba-ssm / causal-conv1d kernel build and exercise (`.[mamba]` extra) |
| 05:30 (`30 5 * * *`) | `example-nightly` | Staged example-execution census: notebooks (nbclient kernels), marimo apps, the helper script, and every example YAML config through real `load_config()` |

Both cron entries trigger the whole workflow file (GitHub offers no per-cron routing), so each nightly job carries its own cron-string gate (`github.event.schedule == '<cron>'`) that selects exactly one entry — the 03:00 jobs never fire on the 05:30 entry and vice versa. Schedules fire only from the default branch. Every nightly job also admits manual `workflow_dispatch`, which is how calibration and one-off census runs are performed; the fast push/PR legs run the in-process suite only.

The example-nightly job runs its stages strictly in order inside one job (torch-heavy execution, then MCP live-server probes, then the ollama-backed `mcp_example` pair), with kernel cleanup and VRAM settle steps between stages, a hard collection-count assertion before any execution, and a fail-soft summary that turns the job red when any stage failed — no step can pass forever-green. See the [workflows README](https://github.com/zhangtaolab/DNALLM/blob/main/.github/workflows/README.md) for the full stage table and per-job timeouts.

## Coverage Expectation

Line coverage of the `dnallm` package is enforced by a hard gate in [`pyproject.toml`](https://github.com/zhangtaolab/DNALLM/blob/main/pyproject.toml):

```toml
[tool.coverage.report]
show_missing = true
fail_under = 90   # Phase 4 ratchet (GATE-01) — the floor, never the achievement.
                  # Fast-lane suite measured 96.72% at the v1.2 closeout (2026-10-10;
                  # was 96.30% at Phase 3) — the Phase 11/12 modules moved the number.
                  # Applies to every `--cov` invocation — use `--no-cov` for scoped runs.
```

The `fail_under = 90` value is a ratchet floor, not the achievement: the
in-process fast lane measured **96.72%** at the v1.2 milestone closeout
(2026-10-10; it was 96.30% when the gate landed in Phase 3 — the Phase 11/12
modules and their tests moved the number), and any `--cov` invocation — local
or in CI — whose total drops below 90 fails. Scoped runs of a subset of the
suite should drop `--cov` or pass `--no-cov`, because the floor applies to
every coverage invocation regardless of how many tests ran.

**What the gate measures — and what it does not (the AUDIT-04 design note).** The example-execution tests run their notebooks in *kernel subprocesses*: each notebook is executed by a separate IPython kernel process, and kernel subprocess coverage is not measured, by design. The example lane therefore **does not move the 96.72% coverage gate** — a green `example-nightly` run certifies that the notebooks executed end to end, not that additional package lines were counted. Conversely, the coverage total never says anything about notebook executability: that guarantee comes from the example census, which is its own lane with its own pass/fail contract. Do not expect example executions to raise the reported coverage total, and do not read the coverage total as evidence about them.

For how to run the censuses locally (fast leg, full census, scoped runs) see [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md).

## Giant Models: the Giants Policy

Tests that execute evo-class giant models are excluded from the `example-nightly` census by owner policy, via the dedicated `giants` pytest marker — the nightly deselects them with `-m "not giants"`. This is a policy exclusion, not an environment limitation: the runner can execute them, so they are marked rather than typed-skipped. The giant-model tests remain runnable through the dispatch/manual lane (`workflow_dispatch` plus an explicit `-m giants` invocation), and their committed executed-notebook outputs remain the evidence of record. Marker usage is documented in [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md).

## Typed Skips and the Skip Audit

Every skip the suite produces is audited: `scripts/audit_skips.py` compares the junit artifact against the typed allowlist in `tests/expected_skips.yaml`, and any skip not covered by an allowlist matcher fails the audit. The semantics are fail-closed in both directions — an unexpected skip fails the job, and a missing or malformed allowlist file fails the audit rather than widening it. If a test starts skipping in your environment, the audit output tells you which typed entry matched (or, if none did, that the skip is new and must be triaged: fixed, or deliberately allowlisted with a typed reason).

---

*For job-level details, stage tables, and timeout arithmetic, always refer to [`.github/workflows/README.md`](https://github.com/zhangtaolab/DNALLM/blob/main/.github/workflows/README.md).*
