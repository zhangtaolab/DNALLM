# Phase 12: Motif Matching, MCP Tools & Milestone Closeout - Pattern Map

**Mapped:** 2026-10-10
**Files analyzed:** 11 (new + modified)
**Analogs found:** 11 / 11 (2 new-file lanes map to role analogs, not exact matches)

All analog paths verified git-tracked via `git ls-files`.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/interpret/motifs.py` (NEW) | service / utility | transform + network (request-response) | `dnallm/models/model.py` `download_model` (retry, lines 323-377) + `dnallm/inference/vep.py` `_load_reference` (parser, 451-501) | role-match |
| `dnallm/interpret/__init__.py` (NEW) | package init | — | `dnallm/datahandling/__init__.py` | exact (shape) |
| `tests/interpret/test_motifs.py` (NEW) | test | — | `tests/inference/` VEP tests + `tests/mcp/test_server_transports.py` structure | role-match |
| `tests/interpret/fixtures/` (NEW) | test assets | file I/O | `tests/` committed synthetic ClinVar fixture (Phase-11 precedent, see CHANGELOG REV-08) | role-match |
| `dnallm/mcp/server.py` (EDIT: 3 tools) | controller (MCP tool) | request-response | `dnallm/mcp/server.py` `_dna_mutagenesis` (1224-1451) | exact |
| `dnallm/mcp/server.py` (EDIT: host/port fix) | config/entry-point | request-response | same file, `start_server`/`_start_http_server` (1676-1890) | exact |
| `tests/mcp/test_server_transports.py` (EDIT) | test | — | itself: `TestStreamableHTTPConstruction` (276-307), `TestInMemoryProtocolRoundTrip` (155-246), `EXPECTED_TOOLS` (44-58) | exact |
| `tests/mcp/test_start_server.py` (EDIT) | test | — | itself: `TestMain` (66-145 per research) | exact |
| `tests/mcp/test_server_tools_v12.py` (NEW) | test | — | `tests/mcp/test_server_transports.py` (real_server fixture + patched uvicorn patterns) | role-match |
| `docs/user_guide/fine_tuning/peft_adapters.md` (EDIT) | documentation | — | its own LoRA chapter above line 160-172 | exact |
| `CHANGELOG.md` (EDIT) | documentation | — | its own `## [Unreleased]` REV-01..09 entries | exact |
| `docs/user_guide/continuous_integration.md` (EDIT) | documentation | — | existing coverage lines (26/30 per research) | exact |
| `tests/expected_skips.yaml` (EDIT, conditional) | test config | — | existing `prefix: "network-unavailable:"` entries | exact |
| `.github/workflows/ci.yml` census pin | config | — | lines 857-871 (research: NOT needed this phase) | n/a |

## Pattern Assignments

### `dnallm/interpret/motifs.py` (service, transform + network)

No existing network-client-with-parse module matches exactly; compose from two analogs.

**Analog A — retry/backoff house pattern:** `dnallm/models/model.py:323-377` (`download_model`). Extract the loop shape (attempt counter, `max_try`, sleep between attempts, final `ValueError` wrap):

```python
# model.py:351-358 (shape to copy)
cnt = 0
status = "incomplete"
while True:
    if cnt >= max_try:
        break
    cnt += 1
    try:
        ...
    except Exception as e:
        ...reason classification...
```
Tests patch `time.sleep` (house precedent). RESEARCH Code Examples gives the full `fetch_meme_motif` transcription (stdlib `urllib.request`, size-capped read, `raise ValueError(f"JASPAR fetch failed for {matrix_id}: {e}") from e`) — use it verbatim as the starting point.

**Analog B — strict file/text parser:** `dnallm/inference/vep.py:451-501` (`_load_reference`). Copy the parse-or-reject style: every malformed input raises a matchable `ValueError` naming the artifact (`raise ValueError(f"Reference FASTA '{path}' contains no sequences.")`). MEME/CIS-BP parsing must follow the same discipline (regex-anchored grammar, `alength=4` check, w cap, rows in [0,1], nsites >= 0).

**Module conventions** (from `dnallm/utils/sequence.py`, the utility nearest in spirit):
- Module docstring: summary + numbered features + honesty statement about the FIMO calibration (D-01 REQUIREMENTS acceptance) — Google style per CONTRIBUTING.
- Constants UPPER_SNAKE with provenance comments: `PSSM_RANGE = 100  # FIMO's internal integer-score granularity (MEME 4.8.1 src/pssm.h:13)`.
- Type hints, PEP 604 unions, relative imports inside the package (`from ..utils.sequence import reverse_complement`).
- `reverse_complement` at `dnallm/utils/sequence.py:37-66` — reuse, do NOT re-implement (note the lowercase-`n` mapping quirk: unknown chars pass through unchanged).

**No analog for the DP itself:** the FIMO exact-DP (scale → column convolution → reverse cumsum → threshold inversion) is genuinely new code; RESEARCH.md "Code Examples — FIMO-recipe calibration" is the authoritative reference with every constant source-cited. Hand-computable unit anchors listed there (uniform PWM → `ValueError`; 1-column motif analytic p=1.0; 2-column analytic convolution).

---

### `dnallm/interpret/__init__.py` (package init)

**Analog:** `dnallm/datahandling/__init__.py` (entire file, 7 lines):

```python
from .data import DNADataset, show_preset_dataset, load_preset_dataset

__all__ = [
    "DNADataset",
    ...
]
```

Keep it minimal (optionally just a docstring — CONTEXT discretion). **Constraint:** must NOT be re-exported from `dnallm/__init__.py` (facade byte-stable).

---

### `tests/interpret/test_motifs.py` + `tests/interpret/fixtures/` (test)

**Analog A — structure/marking:** any `tests/inference/` module; class-grouped `Test*` classes per function under test, `pytest.raises(ValueError, match=r"...")`, synthetic committed fixtures for the fast lane, `slow`-marked live network tests with typed skip reasons.

**Analog B — typed network skips:** `tests/expected_skips.yaml` header + entries:

```yaml
allowed:
  - prefix: "network-unavailable:"
    category: network
```
New JASPAR skips must add a `prefix: "jaspar-unreachable:"` (or similar) entry in the SAME change; never empty/wildcard entries.

**Fixture precedent:** Phase-11 committed synthetic ClinVar VCF fixture (CHANGELOG REV-08: "two-tier ClinVar tests ship a committed synthetic fixture (fast lane, network-free)") — the HBG1/BCL11A golden fixture follows it: committed FASTA windows + frozen JASPAR MEME file, network-free, release-pinned. Owner-blocked input (Fig 4a coordinates); everything else builds against synthetic fixtures meanwhile.

---

### `dnallm/mcp/server.py` — three new tools (controller, request-response)

**Analog:** `_dna_mutagenesis`, `dnallm/mcp/server.py:1224-1451`. Copy this exact skeleton:

1. **Registration** (from `_register_tools`, lines 289-291):
```python
self.app.tool()(self._with_timeout_wrapper(self._ism_scan, "ism_scan"))
self.app.tool()(self._with_timeout_wrapper(self._hotspots, "hotspots"))
self.app.tool()(self._with_timeout_wrapper(self._zero_shot_score, "zero_shot_score"))
```
Wire names carry the leading underscore (`functools.update_wrapper` at line 353).

2. **Validation-first body** (from 1258-1332): non-empty `model_name` check → allowed-values check → per-field input validation → cap enforcement (combo-limit precedent at 1316-1324: error dict states the cap) → `self.model_manager.get_inference_engine(model_name)`; `None` → `{"error": f"Model {model_name} not loaded", "isError": True}`. Regex `^[ACGTacgtNn]+$` content check at 1303.

3. **Response shape** (from 1425-1440): `{"content": [{"type": "text", "text": ...}], ...payload..., "model_name": model_name}`.

4. **Error boundary** (from 1441-1451): whole body in `try/except Exception` → logged with `exc_info=True` → generic `isError` dict; NEVER raise across the protocol boundary.

**Timeout wrapper** is inherited automatically (295-354): `asyncio.wait_for` → timeout dict with `suggestion`; the ISM input caps (Pitfall 4) must be enforced BEFORE the call so the `suggestion` is actionable.

**Blocking torch:** bridge through `self.model_manager.get_inference_engine` (already-loaded engines — no load inside the tool). If any sync-heavy work is added, the executor precedent is `dnallm/mcp/model_manager.py:99-107` (`loop.run_in_executor(None, self._load_model_sync, ...)`); the liveness test (concurrent `health_check` during a long call) goes in `test_server_tools_v12.py`.

**Wrapped surfaces (verified signatures):**
- `Mutagenesis(model, tokenizer, config)` → `.mutate_sequence(seq, replace_mut=..., delete_size=..., insert_seq=...)` → `.evaluate(do_pred=True)` (1349-1356) — `_ism_scan` mirrors this exactly.
- `Mutagenesis.find_hotspots(preds, strategy="maxabs", window_size=10, percentile_threshold=90.0) -> list[tuple[int, int]]` (`dnallm/inference/mutagenesis.py:517-581`) — `_hotspots` calls this on ISM preds.
- `vep.evaluate_vcf(model, tokenizer, vcf_path, reference, *, paradigm="mlm", context_window=200, clnsig_filter=None, alt_number=4, output_dir=None) -> VepResult` (`dnallm/inference/vep.py:734-745`) — `_zero_shot_score` kernel; skip/convention blocks from `VepResult` must surface verbatim in the tool JSON (locked). Reference loading via `vep._load_reference` (451-501) from a per-call `fasta_path` (recommended over pyfastx — dev-extra-only).
- Inline `{chrom,pos,ref,alt}` variants: temp-VCF materialization server-side with a fixed sanitized filename (research Open Question 3 recommendation); validate per-field with matchable error dicts.

---

### `dnallm/mcp/server.py` — host/port CLI-precedence fix

**Bug sites (edit targets):**
- `start_server` override, lines 1743-1747:
```python
server_config = self.config_manager.get_server_config()
if server_config:
    host = server_config.server.host
    port = server_config.server.port
```
- `_start_http_server` always-true conditional, lines 1853-1857:
```python
if streamable_http_config:
    if server_config and server_config.server.host == host:
        host = streamable_http_config.host
    if server_config and server_config.server.port == port:
        port = streamable_http_config.port
```
- argparse defaults `"0.0.0.0"`/`8000`, lines 1996-2008.

**Fix shape** (RESEARCH Pattern 6, sentinel-based): `host: str | None = None, port: int | None = None` on `start_server`; argparse `default=None`; resolve ONCE before dispatch (CLI-explicit > transport YAML `streamable_http`/`sse` > `server` YAML > documented default); delete both override blocks; both starters receive final values; stdio unaffected. Pick ONE documented default and note the `0.0.0.0`-vs-`127.0.0.1` divergence in the CHANGELOG entry.

---

### `tests/mcp/test_server_transports.py` (EDIT) + `tests/mcp/test_start_server.py` (EDIT) + `tests/mcp/test_server_tools_v12.py` (NEW)

**Analog (same file):** `tests/mcp/test_server_transports.py`.

- `EXPECTED_TOOLS` set, lines 44-58 → add `_ism_scan`, `_hotspots`, `_zero_shot_score`; flip `assert len(names) == 13` (line 169) → `16`. Same change as registration (Pitfall 3).
- New-tool round trips: copy `TestInMemoryProtocolRoundTrip.test_call_tool_health_check` (171-180) — `_client_session` helper (205-221), `_streamable_client` ASGI factory with localhost base_url (230-246, mcp 1.30.0 DNS-rebinding 421 guard), `json.loads(result.content[0].text)` assertions.
- Precedence tests: copy `TestStreamableHTTPConstruction.test_config_fields_assembled_from_streamable_http_block` (279-307) — patched `uvicorn.Config`/`uvicorn.Server`, assert `kwargs["host"]/["port"]`; this test FLIPS (explicit 8123 now beats YAML 8124) and gains the complementary yaml-only case. Add the SSE-side equivalents via `TestSSEConstruction._run_sse_start` (340-356).
- `test_start_server.py::TestMain::test_defaults_are_forwarded` (research: lines 123-142) changes from asserting `host="0.0.0.0", port=8000` forwarded to the `None`-sentinel chain.
- `real_server` fixture (134-147): zero models + `patch("dnallm.mcp.model_manager.load_model_and_tokenizer")` — reuse for `test_server_tools_v12.py` (mocked engines via `Mock()` inference engines).
- Unique basenames across the two collected roots (Pitfall 3/`tests/` vs `dnallm/mcp/tests/` no `__init__.py`).

---

### C3 documentation edits

**`docs/user_guide/fine_tuning/peft_adapters.md:174-186`** — replace the "IA³ (coming in the next release)" stub. Analog: the LoRA/QLoRA chapters immediately above (160-172): fenced YAML + Python blocks, prose tied to `TrainingConfig` fields. Write from Phase-11 reality per CHANGELOG line 17 (REV-04): real `DNATrainer` branch, `use_ia3 × use_qlora` rejected at Pydantic time, IA³ save/reload roundtrip. Fenced blocks must survive the docs-validation ruff-format gate (0.7.1 precedent: over-long fenced blocks tripped CI).

**`CHANGELOG.md`** — append REV-10/REV-11 to `## [Unreleased]` same-change, matching the existing entry shape exactly: one dense bullet per feature with inline `(REV-10, R1-3d)` tags (see lines 12-18). C3 then backfills commit SHAs as links (D-09 mechanism). Re-read immediately before edit; unique-anchor insert.

**`docs/user_guide/continuous_integration.md`** — update the "96.30%" coverage expectation (lines 26/30 per research) with the measured post-Phase-12 number; honest per D-07. Census re-pin NOT needed (208/217 verified current).

## Shared Patterns

### MCP error-dict convention (never raise across the boundary)
**Source:** `dnallm/mcp/server.py:1441-1451` (catch-all) and 336-350 (timeout dict).
**Apply to:** all three new tools. Validation errors return `{"error": <matchable message>, "isError": True}`; unexpected exceptions log `exc_info=True` and return the generic isError dict.

### Matchable ValueError for library code
**Source:** `dnallm/inference/vep.py:476` (`f"Reference FASTA not found at '{path}'."`), `500`.
**Apply to:** `motifs.py` parsing, JASPAR client failures, uniform-PWM guard. Tests assert with `pytest.raises(ValueError, match=r"...")`.

### Retry-with-backoff
**Source:** `dnallm/models/model.py:323-377`.
**Apply to:** JASPAR client (`max_try=3`, `time.sleep`, patched in tests).

### Typed network skips + expected_skips.yaml same-change
**Source:** `tests/expected_skips.yaml` (`prefix: "network-unavailable:"` entries).
**Apply to:** JASPAR live tests (`jaspar-unreachable:` prefix), MCP live probes; `slow` marker from birth; `models.lock` row only if a new model id is referenced.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| FIMO exact-DP core (`pvalue_table`, `threshold_bits`, scanner) in `dnallm/interpret/motifs.py` | service | transform | Genuinely new algorithm; use RESEARCH.md "Code Examples — FIMO-recipe calibration" verbatim as the reference (every constant cited to MEME 4.8.1 source) |
| HBG1/BCL11A golden test | test | — | Owner-blocked input (Fig 4a coordinates, motif ID, JASPAR release); build synthetic-fixture lanes meanwhile |

## Metadata

**Analog search scope:** `dnallm/mcp/`, `dnallm/inference/`, `dnallm/models/`, `dnallm/utils/`, `tests/mcp/`, `tests/expected_skips.yaml`, `docs/user_guide/`, `CHANGELOG.md`
**Files read:** 9 source/test/doc files (targeted ranges)
**Pattern extraction date:** 2026-10-10
