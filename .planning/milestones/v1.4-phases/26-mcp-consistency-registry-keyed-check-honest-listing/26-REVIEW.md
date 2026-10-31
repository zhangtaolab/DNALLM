---
phase: 26-mcp-consistency-registry-keyed-check-honest-listing
reviewed: 2026-10-22T01:47:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - dnallm/mcp/__init__.py
  - dnallm/mcp/consistency.py
  - dnallm/mcp/server.py
  - tests/mcp/test_config_registry_consistency.py
  - tests/mcp/test_server_streaming.py
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
---

# Phase 26: Code Review Report

**Reviewed:** 2026-10-22T01:47:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

Scope is the whole phase's production+test diff (`b09ebf2`…`f9fba20`): the new
`dnallm/mcp/consistency.py` checker module, the `dnallm/mcp/__init__.py` facade re-export, the
`_get_all_available_models` rewrite in `dnallm/mcp/server.py`, and both test files. The client
surface (`dnallm/mcp/client.py`, `tests/mcp/test_client_sdk.py`) is untouched and pinned green.

## Summary

The checker is a clean read-only collect-all implementation: safe-loader-only parsing, per-file
try/except producing ERROR findings instead of crashes, exact-equality comparisons, tier table
exactly as planned, 98% coverage. The listing rewrite is a faithful overlay composition from three
existing states with no new server state, and the error/tool contracts are preserved verbatim.
One real defect found in the never-raise contract (reproduced live below); two informational
notes.

## Warnings

### WR-01: Undecodable (non-UTF-8) config bytes crash the checker — UnicodeDecodeError escapes the never-raise contract

- **File:** `dnallm/mcp/consistency.py:152-154` (`_check_one_model_config`), same pattern at `:342-344` (`_check_one_server_config`)
- **Issue:** `path.read_text(encoding="utf-8")` raises `UnicodeDecodeError` (a `ValueError`) on
  binary/latin-1 bytes; the `except (OSError, yaml.YAMLError)` clause does not catch it, so the
  exception propagates out of `check_config_registry_consistency` — violating the D-09/must-have
  contract "never raises out of the checker" (an unreadable file must produce a CONFIG_UNREADABLE
  ERROR finding and the scan must continue).
- **Reproduction (live-proven):** a file containing `b"task: \xff\xfe binary garbage \x80"` named
  `bad_inference_config.yaml` raises `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff`
  out of the checker instead of returning a report.
- **Fix:** add `UnicodeDecodeError` to both except clauses (it is not an `OSError` subclass, so it
  must be named explicitly); add a regression test with undecodable bytes asserting a
  `CONFIG_UNREADABLE` finding and continued scan.
- **Severity rationale:** the committed configs are ASCII so CI is unaffected today, but the
  checker is a public API accepting any directory, and its core promise is findings-not-crashes.

## Info

### IN-01: Two server keys resolving to the same registry entry silently last-wins

- **File:** `dnallm/mcp/server.py` (overlay construction in `_get_all_available_models`)
- **Issue:** if two `models` entries resolve to the same registry entry name, the second
  overwrites the first in `overlay` (last-wins), and `configured_count` counts unique entries,
  not configured keys. Deterministic and harmless today (the committed configs have no
  duplicates), but an operator configuring the same model twice gets no signal.
- **Disposition suggestion:** acceptable for this phase; a future duplicate-detection INFO row on
  the listing or a checker finding could cover it.

### IN-02: Checker returns a vacuous pass on an empty config directory

- **File:** `dnallm/mcp/consistency.py` (`check_config_registry_consistency`)
- **Issue:** a directory with no `*_inference_config.yaml` and no `mcp_server_config*.yaml`
  yields `configs_checked=[]`, `findings=[]`, `passed=True`. "Nothing checked, nothing failed" is
  defensible semantics (and a wrong-directory typo is caught by the CI gate's explicit
  file-count assertions), but callers wanting a stronger signal get none.
- **Disposition suggestion:** document-only; no behavior change requested.

## Prior Findings (carried forward)

None — first review of this phase.
