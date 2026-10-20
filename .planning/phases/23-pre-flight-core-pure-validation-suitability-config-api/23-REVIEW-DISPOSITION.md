| IN-08 |  | fixed | 23-REVIEW-FIX.md || IN-04 |  | fixed | 23-REVIEW-FIX.md || IN-03 |  | fixed | 23-REVIEW-FIX.md || WR-05 |  | fixed | 23-REVIEW-FIX.md || WR-04 |  | fixed | 23-REVIEW-FIX.md || WR-03 |  | fixed | 23-REVIEW-FIX.md || WR-02 |  | fixed | 23-REVIEW-FIX.md || WR-01 |  | fixed | 23-REVIEW-FIX.md || CR-02 |  | fixed | 23-REVIEW-FIX.md || IN-07 |  | deferred | deferred — shared executor saturation PROBE_TIMEOUT nuance; bounded-wait semantics acceptable at current scale, revisit if TUI shows spurious timeouts. || IN-06 |  | deferred | deferred — isdir on raw alias is harmless in local-cache detection; defensive-only. || IN-05 |  | deferred | deferred — min_length<=0 ValidationError message improvement; cosmetic validation-message polish. || IN-02 |  | deferred | deferred — default label column name (label vs labels) is parity-locked by test_reader_parity; changing it alters documented behavior — owner call. || IN-01 |  | deferred | deferred — label_dtype documentation nuance; behavior change judgment call for owner (integer labels reported as float mirrors pandas dtype reality). || CR-01 |  | deferred | deferred — runtime import purity deferred to Phase 25 (owner call): 23-01 plan recorded the transitive-import discovered-constraint; AST/source-level enforcement is the planned scope. Docstring alignment applied in da4e44d; root-facade refactor unsanctioned this phase. |---
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
recorded: 2026-10-20T18:24:00.251Z
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
