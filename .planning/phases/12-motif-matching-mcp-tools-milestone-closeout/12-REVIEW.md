---
phase: 12-motif-matching-mcp-tools-milestone-closeout
reviewed: 2026-10-10T05:42:39Z
depth: standard
files_reviewed: 24
files_reviewed_list:
  - CHANGELOG.md
  - dnallm/interpret/__init__.py
  - dnallm/interpret/motifs.py
  - dnallm/mcp/server.py
  - dnallm/mcp/start_server.py
  - docs/user_guide/continuous_integration.md
  - docs/user_guide/fine_tuning/peft_adapters.md
  - .github/workflows/ci.yml
  - .gitignore
  - pyproject.toml
  - tests/examples/test_examples.py
  - tests/expected_skips.yaml
  - tests/interpret/fixtures/cisbp_motif.txt
  - tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml
  - tests/interpret/fixtures/hbg1_bcl11a/synthetic_motif.meme
  - tests/interpret/fixtures/hbg1_bcl11a/synthetic_window.fasta
  - tests/interpret/fixtures/meme_motif.txt
  - tests/interpret/test_motifs.py
  - tests/mcp/test_client_sdk.py
  - tests/mcp/test_server_tools_v12.py
  - tests/mcp/test_server_transports.py
  - tests/mcp/test_start_server.py
  - tests/test_extras_guard.py
findings:
  critical: 1
  warning: 4
  info: 5
  total: 10
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-10-10T05:42:39Z
**Depth:** standard (targeted deep-dive on motifs.py algorithm, server.py tool bodies, start_server.py precedence; diff-driven on docs/config/CHANGELOG/fixtures)
**Files Reviewed:** 24
**Status:** issues_found

## Summary

Three lanes reviewed: the FIMO-convention motif scanner (`dnallm/interpret/motifs.py`), the three new MCP tools + host/port CLI-precedence fix (`dnallm/mcp/server.py`, `dnallm/mcp/start_server.py`), and the closeout/docs/CI-drift set. All 188 fast-lane phase tests pass; ruff lint and format are clean on every changed code file; the CHANGELOG evidence chain (11 REV tags, each exactly once, 10 unique SHAs all verified ancestors of HEAD, 19 insertions / 0 deletions — append-only) checks out.

The motif scanner's algorithm was independently re-derived and is sound: the DP convolution anchors in `TestPvalueTable` (uniform and skewed backgrounds) were recomputed by hand and match; the reverse-cumsum table, threshold-inversion edge semantics (tail exactly at 1e-4 is excluded under strict `p <`), strand coordinate mapping (`start = L - pos - w`), full-set BH arithmetic (`q = p*n/rank` values in `TestBhFdr` hold only for a single full-set call), E-value multiplier, GC background aggregation/floor, and the PSSM_RANGE=100 integer scaling with `score_bits = s/scale + w*offset` inversion are all internally consistent (scaled ints are used for both the DP table and position scoring — no float/int divergence). The JASPAR client's SSRF posture (matrix-id regex before URL construction, https-only base, redirects refused, size-capped reads, retried network errors vs immediate non-200) is correct, and the `# ruff: ignore[...]` suppressions in the test files were empirically confirmed to be honored by the pinned ruff.

The MCP tool bodies are validation-first with caps before engine access and never raise across the boundary — but adversarial probing found one Critical input-validation hole and several Warning-grade gaps, all empirically confirmed against the real tool bodies with mocked engines (probe outputs quoted in the findings). The host/port precedence fix is complete: both argparse entry points carry None sentinels, `_resolve_bind_address` implements CLI > transport-specific YAML (streamable-http only) > server YAML > documented default on both transports, neither starter re-reads host/port from YAML, and the flipped tests plus the full `TestHostPortPrecedence` matrix pin the new behavior. The IA³ chapter's claims were verified verbatim against `trainer.py`/`configs.py`/`lora_targets.yaml` (field set, both rejection messages, the exact `[Info] IA³ preset 'Plant DNABERT' selected (matched by name marker 'plant-dnabert')` log format). The interleaved CI fixes are accurate: pyarrow `<26` cap comment matches the numpy-1.26.4 leg rationale, the pydantic-ai `>=1.107.0,<2` floor comment matches the resolver-backtrack incident and the extras guard is synced to the same member, the exceptiongroup import is a module-level conditional (the only exceptiongroup import in the tree — no residual function-scope trap), and the notebook-import traceback-origin block correctly walks to the deepest frame.

The security test for the inline temp-VCF path (`test_zero_shot_security_no_record_derived_string_in_any_path`) genuinely pins what it claims: fixed basename `inline_variants.vcf`, `dnallm_zero_shot_` temp-dir prefix, and record strings absent from the path. However, it pins path safety only — record content can still escape its VCF field (CR-01).

Coverage note (honesty): the claimed per-module figures (motifs 99%, server 98%, interpret/__init__ 100%) could not be independently re-measured in this dev venv — numpy 2.5.3 + coverage 7.16.2 on py3.13 fails every `--cov` run at conftest import (`ImportError: cannot load module more than once per process` under all tracer cores; reproduced with coverage + numpy alone, no dnallm code involved). CI is protected by the explicit matrix pins; see IN-05.

## Critical Issues

### CR-01: `zero_shot_score` inline-variant `chrom` is unvalidated free text — VCF field/column injection defeats the validated fields, the variant cap, and the inline no-ClinVar guarantee

**File:** `dnallm/mcp/server.py:2430-2434` (chrom validation), `dnallm/mcp/server.py:2266-2271` (`_write_inline_vcf` interpolation)
**Issue:** `variants[i].chrom` is only checked for `isinstance(str)` and non-emptiness, then interpolated verbatim into a tab-separated VCF data line. `ref`/`alt` are pattern-validated (`^[ACGTacgt]+$`) and `pos` is an int — `chrom` is the one unvalidated field, and it can contain tabs and newlines. Empirically confirmed against the real tool body:

- Tab injection (`chrom = "chr1\t20\t.\tA\tT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided;CLNVC=single_nucleotide_variant"`, `pos=10, ref=A, alt=G`) materializes the row `chr1 20 . A T . . CLNSIG=Pathogenic;... 10 A G ...` — the client's injected POS/REF/ALT replace the validated ones, so a *different variant than requested* is scored, with a client-chosen ClinVar label.
- Newline injection (one variant dict whose chrom embeds 30 additional `\n`-separated fully-formatted rows) produced **31 data rows** in the temp VCF from a single accepted variant — the `ZERO_SHOT_MAX_VARIANTS = 500` cap (T-12-06) is bypassable, and injected `CLNSIG=Pathogenic`/`Benign` rows break the documented inline-mode contract ("inline variants carry no ClinVar annotation by construction ... metrics stay None by construction") by letting metrics be computed over attacker-chosen labels.

This defeats the tool's own validation contract and the phase's documented threat mitigations in a network-exposed tool; the existing security test covers path derivation only, not field containment.
**Fix:**
```python
# module level, next to the other tool-boundary constants
_CHROM_PATTERN = re.compile(r"^[A-Za-z0-9_.:<>|()-]+$")  # whitespace-free: cannot escape its VCF field

# in the per-variant validation loop (replaces the current chrom check)
chrom = variant.get("chrom")
if not isinstance(chrom, str) or not _CHROM_PATTERN.match(chrom):
    return {
        "error": (
            f"variants[{i}].chrom must be a whitespace-free chromosome name "
            f"(got {chrom!r})"
        ),
        "isError": True,
    }
```
Add regression tests: a tab-bearing chrom and a newline-bearing chrom must both return the matchable error dict, and (belt-and-braces) an assertion that the temp VCF data-line count equals `len(variants)`.

## Warnings

### WR-01: explicit `null` values in `clnsig_filter` bypass validation and crash into the generic non-matchable error

**File:** `dnallm/mcp/server.py:2206-2226` (`_build_clnsig_filter`)
**Issue:** Per-field checks treat `None` as "absent" (`if value is not None and ...`), but the constructor then uses `clnsig_filter.get("positive_labels", base.positive_labels)` — which returns `None` (not the default) when the key is *present* with a JSON `null` value. Confirmed empirically: `clnsig_filter={"positive_labels": None}` → `frozenset(None)` TypeError → caught by the outer `except Exception` → `{"content": [... "Zero-shot scoring failed. See server logs for details."], "isError": true}` — no matchable `error` text, violating the project's matchable-`ValueError`/error-dict convention and the tool's validation-first design. `variant_type=None` / `star_floor=None` pass through silently into the kernel (`variant_type=None` would exclude every row as `non_snv_clnvc`).
**Fix:** Resolve each field once and treat present-but-None as absent (or reject it matchably):
```python
positive = clnsig_filter.get("positive_labels")
negative = clnsig_filter.get("negative_labels")
built = ClinVarFilter(
    variant_type=clnsig_filter.get("variant_type") or base.variant_type,
    positive_labels=frozenset(positive) if positive is not None else base.positive_labels,
    negative_labels=frozenset(negative) if negative is not None else base.negative_labels,
    star_floor=clnsig_filter.get("star_floor") if clnsig_filter.get("star_floor") is not None else base.star_floor,
)
```
plus a test driving `{"positive_labels": None}` through the tool.

### WR-02: inline mode + any caller `clnsig_filter` silently drops the sentinel convention — every inline row is then excluded

**File:** `dnallm/mcp/server.py:2392-2398`
**Issue:** The pass-through sentinel filter (`variant_type="inline_variant"`, `positive_labels={"not_analyzed"}`) is applied only when the caller passed no `clnsig_filter` at all. Any override — e.g. a plausible `{"star_floor": 0}` "turn off filtering" attempt — replaces the sentinel wholesale, and `variant_type` reverts to the `ClinVarFilter()` default `"single_nucleotide_variant"` while every temp-VCF row carries `CLNVC=inline_variant`. Confirmed empirically (kernel received `variant_type="single_nucleotide_variant"`); with the real kernel every row lands in `exclusion_counts["non_snv_clnvc"]` and `evaluated == 0`. The only escape is knowing the private constant `_INLINE_SENTINEL_CLNVC`. The kernel's convention block does surface the exclusions, but the docstring frames `clnsig_filter` as "e.g. for non-ClinVar vcf_path input" without warning about inline mode, and no test pins the interaction.
**Fix:** In inline mode, either reject a `clnsig_filter` that omits `variant_type == _INLINE_SENTINEL_CLNVC` with a matchable error naming the required value, or merge unset fields with the sentinel instead of the D-17 defaults; at minimum document the interaction in the docstring and add a test asserting the exclusion accounting for the partial-override case.

### WR-03: `ism_scan`'s `positions` parameter is validated, capped, and echoed — but has no effect on the computation or the reported results

**File:** `dnallm/mcp/server.py:1771-1773` (contract), `1874-1892` (engine call), `1897-1973` (result assembly), `1970` (echo)
**Issue:** `_run_ism_scan` calls `mutagenesis.mutate_sequence(seq, ...)` with no position restriction — the engine mutates **every** position of every sequence (`Mutagenesis.mutate_sequence` has no positions parameter; `find_hotspots`-style subsetting is absent), and the reported `mutated_prediction.predictions` list is never filtered to the requested positions. A client passing `positions=[0]` on a 2000-base sequence gets the full ~6000-pass scan; `ISM_MAX_POSITIONS = 100` bounds only the echoed list, giving a false sense of bounded work (the real bound is solely `ISM_MAX_SEQUENCE_LENGTH`). The docstring ("positions of interest; validated ... and echoed in the response") does not say they are advisory-only, so the contract misleads.
**Fix:** Either filter the reported mutated entries to the requested positions (the engine's entry names encode `mut_{i}_{base}_{alt}`, so `i in positions` filters exactly), restrict the scan itself, or state explicitly in the docstring and the cap comment that `positions` is advisory/echo-only and the bound on work is the sequence cap.

### WR-04: `ism_scan` input-shape gaps: both-provided silently drops `sequence`; empty `sequences` list succeeds; non-string `sequence` fails non-matchably

**File:** `dnallm/mcp/server.py:1821-1839`
**Issue:** Three confirmed edge behaviors, all in the same validation block:
1. `sequence="AAAA", sequences=["ATGC"]` both provided → `sequences` wins, `sequence` silently ignored (probed: scanned `ATGC`, no error).
2. `sequences=[]` → success with `{"batch_results": [], "sequence_count": 0}` — inconsistent with the empty-`positions` rejection two checks earlier; an empty list is effectively "no input provided".
3. `sequence=12345` (non-string) → `dna_pattern.match(12345)` TypeError into the generic handler → non-matchable `"ISM scan failed. See server logs for details."`.
**Fix:**
```python
if sequence is not None and sequences is not None:
    return {"error": "Provide either sequence or sequences, not both", "isError": True}
if sequences is None:
    sequences = [sequence]
if not sequences or not all(isinstance(seq, str) and seq for seq in sequences):
    return {"error": "sequences must be a non-empty list of non-empty strings", "isError": True}
```

## Info

### IN-01: `_Pssm.x_threshold` is computed and stored but never read

**File:** `dnallm/interpret/motifs.py:443`, `458`
**Issue:** `_build_pssm` computes `x_threshold=_threshold_scaled(pv, p_threshold)` on every motif build, but neither `scan` nor `scan_single_strand` reads it — both filter on the raw `pv[s]` comparison (grep-verified: the only occurrences are the field declaration and the assignment). Dead state that suggests a threshold-fast-path that was never wired.
**Fix:** Drop the field, or use it as the position filter (`s >= pssm.x_threshold` instead of `p < p_threshold`) so the stored value and the filtering rule cannot drift apart.

### IN-02: `search_motifs` silently truncates at the 20-page bound

**File:** `dnallm/interpret/motifs.py:923-949`
**Issue:** When the loop exits because `page` reached `_JASPAR_MAX_PAGES` while `payload["next"]` is still non-null, the partial record list is returned with no signal — a hostile or merely large result set is silently cut at 2000 records. The bound itself is documented; the *truncation* is not surfaced.
**Fix:** After the loop, if the last page was full and `next` was non-null, either raise a matchable `ValueError` or log a warning stating the truncation.

### IN-03: `scan(windows, motifs=[])` returns an empty result with `n_motifs=0` instead of a matchable error

**File:** `dnallm/interpret/motifs.py:712`, `761-769`
**Issue:** The sibling entry points reject their empty inputs (`parse_meme` raises on no MOTIF blocks; `gc_background` raises on no ACGT bases), but `scan` with an empty motif list builds zero PSSMs and returns an all-zero `ScanResult` silently — an almost-certainly-mistaken call looks like "no hits".
**Fix:** `if not motifs: raise ValueError("scan requires at least one motif.")` (and consider the same for an empty `windows` list when `background` is None — `gc_background` already covers it).

### IN-04: stale comment miscounts the tool registration set next to the assertion that pins it

**File:** `tests/mcp/test_server_transports.py:38-44`
**Issue:** The comment says "ten timeout-wrapped tools plus the three streaming tools registered directly, plus the two Phase 12 analysis tools (ism_scan, hotspots)" — arithmetic 15, but `EXPECTED_TOOLS` (lines 45-62) contains 16 entries including `_zero_shot_score`, and line 173 asserts `len(names) == 16`. Someone "correcting" the set to match the comment would break the test.
**Fix:** "thirteen timeout-wrapped tools (including the three Phase 12 analysis tools: ism_scan, hotspots, zero_shot_score) plus the three streaming tools registered directly".

### IN-05: `numpy>=1.26.0` is unbounded while the local coverage toolchain cannot instrument numpy 2.5.x — fresh non-matrix resolves break every `--cov` run

**File:** `pyproject.toml:52`
**Issue:** Not introduced by this diff, but the same drift class as the pyarrow/pydantic-ai caps fixed mid-phase: this dev venv resolved numpy 2.5.3 (permitted by the floor), and coverage 7.16.2 on Python 3.13 then fails *every* `pytest --cov` invocation at conftest import — `ImportError: cannot load module more than once per process` from numpy's double-init guard, reproduced with coverage + numpy alone under all `COVERAGE_CORE` settings (no dnallm code involved). CI is protected by the explicit matrix pins (1.26.4 / 2.2.0), so only local/ungated resolves are exposed. This also blocked independent re-measurement of the phase's per-module coverage claims (see Summary).
**Fix:** When the matrix retires the 1.26.4 leg, consider an explicit `numpy<2.6` (or similar) ceiling co-located with the pyarrow cap comment, or pin the dev venv; at minimum record the incompatibility next to the numpy floor.

---

## Verified-sound (checked, no finding)

- **motifs.py algorithm**: DP convolution anchors (`{0:.25, 100:.5, 200:.25}` uniform; `{0:.12, 100:.46, 200:.42}` skewed) re-derived by hand and match the tests; pdf index bounds hold (`k+s <= w*100`); normalization guard and reverse cumsum correct; `_threshold_scaled` returns the minimal passing score under strict `p <` (tail exactly at 1e-4 excluded — pinned by `test_threshold_bits_tail_exactly_at_threshold_boundary`); scaled ints used consistently for DP, scoring, and the `s/scale + w*offset` inversion; minus-strand coordinate mapping verified; palindromic both-strands no-dedup semantics pinned by dedicated tests; BH over the full window×motif×strand vector with `zip(..., strict=True)`; pv[0] float drift clamped before scipy.
- **JASPAR client**: `MA\d{4}\.\d+$` validated before any URL construction (path-traversal probe rejected pre-network, test-pinned); https-only base; `_NoRedirectHandler` refusals test-pinned; `MAX_RESPONSE_BYTES` read cap verified via a recording fake; retry/backoff mirrors the `download_model` house pattern; non-200 raises without retry.
- **server.py tool boundary**: validation-first ordering confirmed (all caps and field checks precede `_ism_engine_guard`); timeout wrapper preserves signatures via `functools.update_wrapper`; CancelledError from `asyncio.wait_for` is not swallowed by `except Exception` and the `finally` still removes the temp VCF; `except ValueError` around the executor surfaces kernel messages matchably, everything else degrades to the generic dict — nothing raises across the protocol boundary (transport round-trip tests confirm over the wire).
- **start_server precedence**: both argparse entry points (`server.py:main`, `start_server.py:main`) use None sentinels with accurate help text; `_resolve_bind_address` implements the documented chain; neither starter re-reads host/port from YAML (`_start_http_server` comment matches code); no other consumer of host/port exists (`get_streamable_http_config`/`get_sse_config` have no in-package callers; `SSEConfig` has no host/port fields). The two flipped tests plus the 7-case `TestHostPortPrecedence` matrix pin the new behavior on both transports.
- **CHANGELOG**: 11 REV tags each exactly once (REV-01..REV-11); all 10 linked SHAs parse and are ancestors of HEAD; 19 insertions / 0 deletions in the range (append-only); REV-10/REV-11 SHAs match the commits whose messages claim those entries.
- **IA³ docs vs source**: `Ia3Config` field set (target_modules, exclude_modules, feedforward_modules, fan_in_fan_out, init_ia3_weights, modules_to_save, task_type) matches the doc's YAML 1:1; both rejection sites and message texts match (`configs.py:382-387` Pydantic-time, `trainer.py:399-400` trainer-init); the example preset log line matches the f-string at `trainer.py:448-450` with `matched_by = "name marker 'plant-dnabert'"` and the `Plant DNABERT` preset row (`lora_targets.yaml:50-57`).
- **CI-drift fixes**: pyarrow `>=15,<26` cap comment accurate (numpy 1.26.4 leg); pydantic-ai `>=1.107.0,<2` floor comment matches the incident and `EXPECTED_MCP_MEMBERS` is synced to the identical member string; exceptiongroup import is the module-level conditional at `tests/mcp/test_client_sdk.py:21-23` (grep-verified the only exceptiongroup import in the tree — no function-scope trap remains); the traceback-origin block (`test_examples.py:272-283`) correctly walks `__traceback__` to the deepest frame and appends ` [raised in file:line]`; ci.yml adds `revision` to the push trigger with an accurate comment.
- **Fixtures/skips/gitignore**: the golden stand-in window is exactly 60 bp with 15/15/15/15 counts and `AGTATACT` at offset 30 (manifest claims verified analytically); the palindromic MEME consensus matches the embedded site; `!tests/interpret/fixtures/**/*.fasta` negation is effective (files tracked, `git check-ignore` clean); `expected_skips.yaml` gained the typed `jaspar-unreachable:` prefix entry in the same change as the skip-producing tests.
- **Tooling**: ruff check and `ruff format --check` pass on all 10 changed code files; the `# ruff: ignore[rule-name]` comments were empirically confirmed to suppress their rules under the pinned ruff (probe with/without comment).
- **Test quality**: 188 fast-lane tests pass in 4.8s; no papering-over found — the security test reads the temp VCF from inside the kernel call before cleanup, the liveness test parks a real executor thread, and the BH/E-value tests assert exact hand-computed full-set arithmetic that a per-window correction would fail.

_Reviewed: 2026-10-10T05:42:39Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
