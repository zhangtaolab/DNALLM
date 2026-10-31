---
phase: 23
review: 23-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: deferred
    title: "Import-light invariant is empirically false — `import dnallm.preflight` pulls in torch, transformers, and datasets"
  - id: CR-02
    severity: critical
    disposition: fixed
    title: "`validate_data` crashes with a raw pandas ValueError on non-scalar cells (nested-array columns)"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "`edit_config` on a nonexistent path raises raw `FileNotFoundError`, breaking the zero-traceback contract"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "`generate_config(..., output_path=...)` writes an invalid config to disk even when generation produced errors"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Probe memo is poisoned by injected-prober results — later default probes return the fake report"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "`test_availability.py` performs real network I/O at pytest collection time and depends on undeclared `httpx`"
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "`ListingEntry.availability` typed as bare `object | None` instead of `AvailabilityReport | None`"
  - id: IN-01
    severity: info
    disposition: deferred
    title: "`label_dtype` value `\"int\"` is documented but unreachable — integer labels are reported as `\"float\"`"
  - id: IN-02
    severity: info
    disposition: deferred
    title: "Default label column name diverges from the parity reference (`\"label\"` vs `\"labels\"`)"
  - id: IN-03
    severity: info
    disposition: fixed
    title: "Reader opens files without `encoding=\"utf-8\"` (locale-dependent decoding)"
  - id: IN-04
    severity: info
    disposition: fixed
    title: "One 115-character line exceeds the project's ruff line-length 100"
  - id: IN-05
    severity: info
    disposition: deferred
    title: "`validate_data(min_length=0)` raises an undocumented raw pydantic `ValidationError`"
  - id: IN-06
    severity: info
    disposition: deferred
    title: "Local-cache detection runs `os.path.isdir` on the raw input alias"
  - id: IN-07
    severity: info
    disposition: deferred
    title: "Shared bounded-wait executor can report spurious PROBE_TIMEOUT while saturated"
  - id: IN-08
    severity: info
    disposition: fixed
    title: "New `validate_sections` docstring uses the legacy typed-param style"
open: 0
total: 15
recorded: 2026-10-20T18:24:26.245Z
---

# Phase 23: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | deferred | deferred — runtime import purity is a Phase 25 owner call: 23-01 recorded the transitive-import discovered-constraint; source-level AST enforcement is the planned scope. Docstring alignment applied (da4e44d); root-facade refactor unsanctioned this phase. |
| CR-02 | critical | fixed | 23-REVIEW-FIX.md |
| WR-01 | warning | fixed | 23-REVIEW-FIX.md |
| WR-02 | warning | fixed | 23-REVIEW-FIX.md |
| WR-03 | warning | fixed | 23-REVIEW-FIX.md |
| WR-04 | warning | fixed | 23-REVIEW-FIX.md |
| WR-05 | warning | fixed | 23-REVIEW-FIX.md |
| IN-01 | info | deferred | deferred — label_dtype "int" documentation nuance; integer labels report "float" mirroring pandas dtype reality — owner judgment call. |
| IN-02 | info | deferred | deferred — default label column name ("label" vs data.py "labels") is parity-locked by test_reader_parity; renaming alters documented behavior — owner call. |
| IN-03 | info | fixed | 23-REVIEW-FIX.md |
| IN-04 | info | fixed | 23-REVIEW-FIX.md |
| IN-05 | info | deferred | deferred — min_length<=0 pydantic ValidationError message polish; cosmetic. |
| IN-06 | info | deferred | deferred — isdir on raw alias in local-cache detection is harmless; defensive-only. |
| IN-07 | info | deferred | deferred — shared executor saturation can report spurious PROBE_TIMEOUT; acceptable at current scale, revisit if the TUI shows spurious timeouts. |
| IN-08 | info | fixed | 23-REVIEW-FIX.md |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
