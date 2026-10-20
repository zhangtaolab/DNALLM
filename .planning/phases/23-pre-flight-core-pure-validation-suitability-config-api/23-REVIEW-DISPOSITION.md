---
phase: 23
review: 23-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "Import-light invariant is empirically false — `import dnallm.preflight` pulls in torch, transformers, and datasets"
  - id: CR-02
    severity: critical
    disposition: open
    title: "`validate_data` crashes with a raw pandas ValueError on non-scalar cells (nested-array columns)"
  - id: WR-01
    severity: warning
    disposition: open
    title: "`edit_config` on a nonexistent path raises raw `FileNotFoundError`, breaking the zero-traceback contract"
  - id: WR-02
    severity: warning
    disposition: open
    title: "`generate_config(..., output_path=...)` writes an invalid config to disk even when generation produced errors"
  - id: WR-03
    severity: warning
    disposition: open
    title: "Probe memo is poisoned by injected-prober results — later default probes return the fake report"
  - id: WR-04
    severity: warning
    disposition: open
    title: "`test_availability.py` performs real network I/O at pytest collection time and depends on undeclared `httpx`"
  - id: WR-05
    severity: warning
    disposition: open
    title: "`ListingEntry.availability` typed as bare `object | None` instead of `AvailabilityReport | None`"
  - id: IN-01
    severity: info
    disposition: open
    title: "`label_dtype` value `\"int\"` is documented but unreachable — integer labels are reported as `\"float\"`"
  - id: IN-02
    severity: info
    disposition: open
    title: "Default label column name diverges from the parity reference (`\"label\"` vs `\"labels\"`)"
  - id: IN-03
    severity: info
    disposition: open
    title: "Reader opens files without `encoding=\"utf-8\"` (locale-dependent decoding)"
  - id: IN-04
    severity: info
    disposition: open
    title: "One 115-character line exceeds the project's ruff line-length 100"
  - id: IN-05
    severity: info
    disposition: open
    title: "`validate_data(min_length=0)` raises an undocumented raw pydantic `ValidationError`"
  - id: IN-06
    severity: info
    disposition: open
    title: "Local-cache detection runs `os.path.isdir` on the raw input alias"
  - id: IN-07
    severity: info
    disposition: open
    title: "Shared bounded-wait executor can report spurious PROBE_TIMEOUT while saturated"
  - id: IN-08
    severity: info
    disposition: open
    title: "New `validate_sections` docstring uses the legacy typed-param style"
open: 15
total: 15
recorded: 2026-10-20T17:56:58.388Z
---

# Phase 23: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| CR-02 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| WR-04 | warning | open | - |
| WR-05 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |
| IN-07 | info | open | - |
| IN-08 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
