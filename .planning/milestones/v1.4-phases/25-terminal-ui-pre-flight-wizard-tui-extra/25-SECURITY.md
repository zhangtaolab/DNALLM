---
phase: "25"
slug: "terminal-ui-pre-flight-wizard-tui-extra"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-22"
verified: "2026-10-22"
---

# Phase 25 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
>
> The register was **authored at plan time** — every `25-*-PLAN.md` carries a `## Threat Model`
> markdown table (27 numbered entries T-25-01…T-25-27 plus the per-gap T-25-SC). It is deliberately
> *not* the `<threat_model>` XML form, so a parser looking for that tag will mis-read this phase as
> unplanned; it was not. This audit re-verified each mitigation against the tree as it stands after
> the final gap-closure plan (25-13), which changed `dnallm/tui/app.py` CSS and the `compose()` order
> in `dnallm/tui/screens/config.py`.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| terminal ↔ wizard | Everything the user sees is rendered through a single markup-disabled helper; nothing from a data file or hub is interpreted as Textual markup | arbitrary user-supplied strings (file paths, model names, core error messages) |
| wizard ↔ Phase-23 core | Screens hold no domain logic and never open files; they hand paths to `dnallm.preflight` / `edit_config` and render the returned report | paths, reports, validation findings |
| core ↔ model hubs | Requests carry registry-derived repo ids only; no user data leaves the machine | repo ids, model names |
| package install ↔ environment | `tui` extra pulls exactly `textual` and `platformdirs`, both canonical upstream with no install scripts | package metadata |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-25-01 | Tampering (supply chain) | `textual`, `platformdirs` installs | high | mitigate | Owner-locked ranges in `pyproject.toml:149-152`; exact-set pin `tests/test_extras_guard.py:63-66,151-165`; RESEARCH legitimacy audit (canonical repos, no install scripts); phase-end human probe recorded pass (textual 8.2.8, platformdirs 4.12.1 import cleanly) | closed |
| T-25-02 | Tampering | pyproject extras surgery | medium | mitigate | `full` and `tui` pinned exactly in both directions (`tests/test_extras_guard.py:141-216`); bidirectional residue gates (269-400); `pyproject.toml:91-93,153-159` | closed |
| T-25-03 | Denial of service | worker lifecycle / app exit | medium | mitigate | `exit_on_error=False` on all 7 worker decorators (`dnallm/tui/workers.py:324,351,374,398,447-452,513,544`); `cancel_all()` + download cancel on unmount (`app.py:222-234`); failures render as typed `WorkerFailed` | closed |
| T-25-04 | Information disclosure | probe batch / download requests | low | accept | Only registry-derived repo ids reach the hubs; shortlist capped at 20 (`preflight/compatibility.py:129,593-608`); no user data in any payload — see Accepted Risks | closed (accepted) |
| T-25-05 | Tampering | lazy facade name resolution | high | mitigate | Lazy facades on all three `__init__`s; import-parity over every `__all__` name + `dir()` contract + star-import subprocess + attribute-cache assertion (`tests/test_package_facade.py:105-293`) | closed |
| T-25-06 | Spoofing (patch-order integrity) | transformers compat shim | high | mitigate | Shim imported first in every guard path (`dnallm/models/__init__.py:111-123`, five consumer `__init__`s); AST ordering contract + direct-import subprocess tests (`tests/test_package_facade.py:296-405`). Live CUDA CI legs are the remaining proof and were not re-run locally. | closed |
| T-25-07 | Denial of service | `collect_disk()` on unreadable/network roots | low | mitigate | Only cache roots enumerated (`utils/env_info.py:173-201`); every `disk_usage` wrapped, `OSError` degrades to an unknown row (204-248); runs off the event loop; degrade test `tests/utils/test_env_info.py:212-224` | closed |
| T-25-08 | Tampering | Textual markup rendering | high | mitigate | One `render_text(..., markup=False)` helper (`tui/app.py:201-216`) used by every non-literal Static; DataTable cells wrapped in Rich `Text`; bracketed-value Pilot proof `tests/tui/test_app.py:222-247` | closed |
| T-25-09 | Tampering | `--config` path handling | low | mitigate | `click.Path(exists=True)` (`cli/cli.py:345-352`); path stored, never opened in the app; loaded only through core `edit_config`; "deleted file still starts" test | closed |
| T-25-10 | Tampering | crafted/untrusted data file | high | mitigate | No pandas/csv/yaml import anywhere under `dnallm/tui/`; pickle refused, never unpickled (`preflight/validation.py:171-181,296-312`); report values rendered markup-free | closed |
| T-25-11 | Denial of service | very large data file in a worker | low | accept | Read runs in a thread worker; quit asks the worker to stop; 300 s pytest ceiling — see Accepted Risks | closed (accepted) |
| T-25-12 | Denial of service (rate limiting) | shortlist probing | medium | mitigate | 20-row cap (`preflight/compatibility.py:129,593-608`); bounded per-probe wait + memo; on-demand probes are single-row user actions; 429 → UNKNOWN (`availability.py:497-522`) | closed |
| T-25-13 | Spoofing (false availability claim) | availability column | high | mitigate | UNKNOWN always reads "hosting not verified"; hub named only when the core reported one; no row hidden; UNKNOWN stays selectable and says why (`screens/suitability.py:363-400,532-551`); tests `tests/tui/test_screens.py:626-700` | closed |
| T-25-14 | Tampering | unsafe YAML load / invalid config write | high | mitigate | `edit_config` safe-loader family only; writes gated on zero errors; screen never imports yaml; no-file-on-error tests | closed |
| T-25-15 | Tampering | output path handling | medium | mitigate | Exact user-chosen path, `expanduser` by the core, atomic temp+replace, no derived paths (`preflight/config_gen.py:214-247`); preview equals written bytes by construction | closed |
| T-25-16 | Tampering | downloaded model path display | medium | mitigate | Returned path rendered verbatim with markup off, never constructed (`screens/summary.py:74-110`); `download_model` called without `allow_patterns` (`workers.py:655-659`) | closed |
| T-25-17 | Denial of service | unbounded retries / unresponsive UI | medium | mitigate | `max_try` core-owned (default 10); dedicated daemon download thread, never joined; cooperative pre-attempt cancel; quit cancels; failure offers retry, never auto-loops | closed |
| T-25-18 | Denial of service (CI wall-time) | Pilot suite | medium | mitigate | Seams injected with fakes; `wait_for_complete` instead of polling; 300 s pytest ceiling; measured **199 passed in 46.00 s** against the re-adjudicated 50.0 s budget | closed |
| T-25-19 | Tampering | `copy.config_error_message` | medium | mitigate | Exact → generic → verbatim fallback (`tui/copy.py:544-657`); dotted/bare normalisation; every kind + fallback + no-dotted-keys tests (`tests/tui/test_copy.py:168-318`) | closed |
| T-25-20 | Denial of service (legibility at 80x24) | `#config-errors` detail toggle | low | accept | **Re-adjudicated 2026-10-22.** The recorded mitigation (`overflow-y: auto`) was deleted by 25-13 because it is inert on a `Static`; the live mechanism is `#config-errors { height: auto; }` (`tui/app.py:124`) inside the `#config-side` `VerticalScroll`. The strengthened 80x24 Pilot test proves the whole plain sentence — and, after the toggle, the raw detail — is laid out and reachable by scrolling (`tests/tui/test_screens.py:905-1014`). See Accepted Risks. | closed (accepted) |
| T-25-21 | Tampering | `TASK_TYPE_GLOSSES` vs core accepted values | medium | mitigate | One ordered 8-value mapping drives legend, hint and both error sentences; four-site completeness + drift against the live `TaskConfig` pattern (`tests/tui/test_copy.py:339-400`) | closed |
| T-25-22 | Spoofing | navigation sentences | low | mitigate | Each destination named by the constant that screen actually renders (`tui/copy.py:444-450,578-584`); substring assertions `tests/tui/test_copy.py:417-422` | closed |
| T-25-23 | Denial of service (legibility at 80x24) | ten-line suitability legend | low | accept | Legend inside `#suitability-body` `VerticalScroll`; 10 lines, longest 74 chars; 80x24 Pilot test proves legend, check action and use action each reachable by scrolling (`tests/tui/test_screens.py:1093-1120`). See Accepted Risks. | closed (accepted) |
| T-25-24 | Denial of service (legibility at 80x24) | LoRA target-modules hint | low | mitigate | Hint is exactly 226 characters (`tui/copy.py:519-523`); `<= 226` source pin (`tests/tui/test_copy.py:402-414`); containment in the 9-row form at 80x24 (`tests/tui/test_screens.py:1127-1212`) | closed |
| T-25-25 | Denial of service (legibility) | `#config-errors` pane inside `#config-side` | high | mitigate | Inert cap removed — `height: auto`, no `max-height`, no `overflow` (`tui/app.py:124`); laid-out height asserted against Textual's own `Content.get_height`, tail token in the last laid-out row, scroll containment at 80x24 **and** 100x40 (`tests/tui/test_screens.py:905-1014`) | closed |
| T-25-26 | Denial of service (legibility) | task-type hint vs `#config-form` viewport | medium | mitigate | Hint yielded after its input (`screens/config.py:270-289`); section top margin and output-row vertical margin dropped (`tui/app.py:119,131`); input contained in the form before scrolling in clean **and** error state, hint rows reachable | closed |
| T-25-27 | Tampering | 25-12 copy completeness contract | low | mitigate | `dnallm/tui/copy.py` and `tests/tui/test_copy.py` outside the 25-13 diff (last touched at `57bfde1`); completeness tests re-run unmodified and green | closed |
| T-25-SC | Tampering | npm/pip/cargo installs (per gap plan) | high | accept | No dependency or extras change in 25-11/12/13; `git log -- pyproject.toml` shows only the 25-01 commits. See Accepted Risks. | closed (accepted) |

*Status: open · closed*
*Severity: critical > high > medium > low — only open threats at or above `workflow.security_block_on` (`high`) count toward `threats_open`*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-25-04 | Probe and download requests carry only registry-derived repo ids; the shortlist is capped at 20 and no user data crosses the boundary. Accepted because the disclosure surface is registry metadata that is already public. | Phase 25 threat model (plan time) | 2026-10-21 |
| AR-02 | T-25-11 | A pathologically large data file can hold a thread worker busy. The read is off the event loop, quit requests cancellation, and the 300 s pytest ceiling bounds tests. Accepted at low severity: the UI stays responsive and the user can quit. | Phase 25 threat model (plan time) | 2026-10-21 |
| AR-03 | T-25-20 | Legibility at 80x24 for the error detail toggle. **Re-adjudicated 2026-10-22** after 25-13 removed the inert `overflow-y: auto`: the pane is now unbounded inside the scrolling side column, and the 80x24 test proves the full sentence and the raw detail are both laid out and reachable. Accepted at low severity on the new mechanism, not the stale text. | Phase 25 threat model; re-adjudicated by the phase security audit | 2026-10-21 / 2026-10-22 |
| AR-04 | T-25-23 | The ten-line suitability legend needs scrolling at 80x24. Bounded (10 lines, ≤74 chars) inside an existing scroll container with a reachability test. Accepted at low severity. | Phase 25 threat model (plan time) | 2026-10-21 |
| AR-05 | T-25-SC | The gap-closure plans add no dependency and change no `pyproject.toml` entry, so the phase's supply-chain audit from 25-10 remains in force. Accepted at high severity *because* the surface is unchanged, not because it was reviewed again. | Gap-closure plans 25-11…25-13 | 2026-10-21 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-22 | 28 | 28 | 0 | gsd-security-auditor (ASVS L1, block_on=high) |

---

## Notes for the next reader

- **Six summaries carry no `## Threat Flags` heading** (25-03, 25-04, 25-05, 25-06, 25-08, 25-11 — older
  template). Their absence is **not** positive confirmation that no new surface was declared; only the
  seven summaries that do have the section say "None beyond the plan's threat model".
- **`validate_data_worker` residual (inside AR-02's scope):** unlike `download_worker` (a dedicated
  daemon thread that is never joined), `validate_data_worker` is a Textual thread worker on the loop's
  default executor. A read wedged inside pandas will not be forcibly interrupted by `cancel_all()`, and
  interpreter exit joins the executor. The cancel flag the accepted rationale promises is present; the
  residual is why T-25-11 stays *accepted* rather than *mitigated*.
- **`#config-preview { max-height: 24 }` (`tui/app.py:128`)** is the same possibly-inert-overflow class
  as T-25-20 was. It is not in the threat register and has no security consequence (the side column
  scrolls), but it is a real deferred legibility defect: a 594-character edited config needs 43 rows and
  lays out 24. Tracked as UI-SPEC D-5, not a security item.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-22
