# Phase 18: Packaging & Install Honesty — Atomic Extras Surgery - Pattern Map

**Mapped:** 2026-10-14
**Files analyzed:** 12 (modified/new surfaces)
**Analogs found:** 12 / 12 (all analogs git-tracked, verified via `git ls-files`)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `pyproject.toml` (extras block :148-187, `[tool.uv]` index/sources/conflicts :200-270, fla marker, :116 comment) | config (packaging metadata) | — | the file itself; marker vocabulary from :89 `pygenometracks>=3.9; platform_system != 'Windows'` | exact |
| `.github/workflows/ci.yml` (9 install lines, :264-267 cuda block collapse, census pins :867-868) | config (CI) | — | the file itself; census re-pin = the existing D-03 triple at :867 | exact |
| `.github/workflows/publish.yml` (:59 install line only — NO trigger edits) | config (publish) | — | the file itself (17-01 invariant) | exact |
| `.github/workflows/feasibility.yml` (:67 install line) | config (CI) | — | the file itself | exact |
| `.github/workflows/README.md` (:81, :106, :155) | config docs | — | the file itself | exact |
| `tests/test_extras_guard.py` (EXTEND: full/alias/marker/absence guards + repo-scan grep gate) | test (tomllib bracket-member + repo-scan sanctioned residue) | file-I/O scan | itself (member guards) + `tests/test_models_lock_contracts.py` (repo-scan with reasoned allowlist) | exact |
| `tests/models/test_plant_helixseek_fla_kernels.py` (:45, :56, :61-62 literals `base`→`full`) | test (meta-extra parser contracts) | transform | the file itself | exact |
| `docs/getting_started/installation.md` (uv/pip lane split, extras table :198-226, deprecation notes) | docs | — | itself — the existing pip-lane section :121-165 + torch-LAST rule :139-165 | exact |
| NEW docs page: GPU support matrix (PKGH-06) | docs | — | `docs/getting_started/installation.md` section structure + verbatim 0542a6b STATE.md record (see 18-RESEARCH § GPU Support Matrix) | role-match |
| NEW docs page: cross-installer honesty matrix (PKGH-07) | docs | — | `installation.md:121-137` prose+table shape; content skeleton in 18-RESEARCH § Honesty Matrix | role-match |
| `README.md` (+ deprecation note; :76/:170/:178/:199/:204/:210 rows; :254 fla-upstream `[cuda]`/`[rocm]` stays) | docs | — | the file itself | exact |
| `docs/` + `example/` sweep (~49 `.[base` occurrences + mirrors) | docs mirror | — | byte-identical mirror discipline: `tests/test_runner_infra_contracts.py:209-232` + `tests/scripts/test_check_docs_sync.py` | exact |

Validation/mechanics (not files): fresh-venv per-extra resolves, smoke dispatch, pickaxe atomicity proofs — patterns below.

## Pattern Assignments

### `pyproject.toml` (config — the surgery table)

**Analog:** the file itself. Everything is an edit-in-place; the alias reuses the self-referential chain shape already proven daily.

**Extras block today (:148-156) — rename target:**
```toml
base = [
    "dnallm[dev,test,notebook,mcp]",
    "isort>=6.0.1",
    "types-transformers>=0.1.0",
]
all = [
    "dnallm[base,dev,test,notebook,docs,ui,mcp,fla]",
]
```
Recommended post-surgery shape: RESEARCH § Recommended alias shape (`full` carries base's exact members; `base = ["dnallm[full]"]` alias with deprecation comment; `all` swaps `base`→`full`, keep `fla` in the member list — Pitfall 8).

**Marker vocabulary — copy verbatim style (:89, :118):**
```toml
"pybedtools>=0.11.0; platform_system != 'Windows'",
"pygenometracks>=3.9; platform_system != 'Windows'",
```
New fla member: `"triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'"`.

**cuda121/rocm removal surface (exact lines read this session):** `cuda121 = [...]` (:159-161), `rocm = [...]` (:174-176), `[[tool.uv.index]] torch-cuda121` (:205-208) and `torch-rocm` (:230-233), sources line `{ index = "torch-cuda121", extra = "cuda121" }` (:238) + commented `torch-rocm` source (:243), conflicts rows `cpu×cuda121` (:248) and cuda121×{124,126,128,130} (:254-257) + commented rocm rows. Also rewrite the :116 comment's `.[base]` → `.[full]` so the residue gate stays clean. Keep the fla comment "do NOT use fla's own backend extras ([cuda]/[rocm])" — that is upstream-fla vocabulary, sanctioned (Pitfall 4).

### `.github/workflows/ci.yml` (config — rename + census re-pin)

**Analog:** the file itself; fresh grep at execution time (`grep -rn '\[base' .github/ pyproject.toml`) — line numbers drift (RESEARCH Pitfall 1; publish.yml's site moved :39→:59).

**Install-line shape to flip (e.g. :343):** `uv pip install -e ".[base,fla]"` → `".[full,fla]"`. 9 lines in ci.yml (:84, :169, :264, :266, :343, :410, :537, :744, :1127); cuda121 leg :264 dies with the matrix collapse (`cuda-version: ['12.4']`).

**Census pin pattern (:867-868) — re-pin SAME-CHANGE after adding guard tests:**
```bash
if ! tail -1 census-collect.txt | grep -qE '^=* *216/217 tests collected \(1 deselected\) in '; then
  echo "FAIL: census collection mismatch (D-03) — expected triple: 216/217 tests collected (1 deselected); measured line: $(tail -1 census-collect.txt)"
```
Measure with the exact stage flags (phase-13 discipline), update 216/217 to the new counts.

### `tests/test_extras_guard.py` (test — extend)

**Analog:** itself — copy the frozenset-member + `_load_*_extra` pattern exactly:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

EXPECTED_MCP_MEMBERS = frozenset({ ... })          # :40-48

def _load_mcp_extra() -> list[str]:                 # :60-64
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data["project"]["optional-dependencies"]["mcp"])
```
Add: `EXPECTED_FULL_MEMBERS`, `test_full_extra_members_preserved`, `test_base_alias_is_pure_self_reference` (`base == ["dnallm[full]"]` exactly), `test_triton_windows_marker_declared_in_fla` (exact-string), `test_cuda121_and_rocm_extras_absent` (also assert no `[[tool.uv.index]]` named torch-cuda121/torch-rocm) — shapes given in RESEARCH § Guard Tests Extension.

**Repo-scan grep-gate analog — `tests/test_models_lock_contracts.py:71-85` (reasoned allowlist over scanned tree):**
```python
REPO_ROOT = Path(__file__).resolve().parent.parent
# Non-model strings that survive the org/name shape filter; each entry is a
# (token, one-line reason) pair ...
_NON_MODEL_ALLOWLIST: tuple[tuple[str, str], ...] = ( ("...", "local LoRA adapter ... reason"), )
```
Build the `.[base]` sanctioned residue as an exact verbatim list (phase-13 OLLM-03 discipline: 0/short output = deletion creep = fail; extra/missing line = fail); second pattern `cuda121|torch-rocm|\[rocm\]` with near-empty list, sanctioning exactly README:254 and faq:13 (fla-upstream mentions). Scan scope: pyproject.toml, .github/, docs/, example/, README.md, tests/, scripts/ — NOT .planning/.

### `tests/models/test_plant_helixseek_fla_kernels.py` (test — literal flips)

**Analog:** the file itself. :45 `"dnallm[base,dev,test,notebook,docs,ui,mcp,fla]"` → `full`; :56 `dnallm[base, fla ]`; :61-62 `dnallm[base,flash-attn]` / `dnallm[base,fla-core,mamba-fla]` → `full`-prefixed. `test_fla_reachable_from_all` (:79-83) stays green as long as `fla` remains in the `all` member list (Pitfall 8).

### `docs/getting_started/installation.md` + new matrix pages (docs)

**Analog:** installation.md's existing pip-lane section (:121-137) + torch-LAST rule (:139-165) — extend/restructure, don't replace. Dependency Groups table (:198-226): base→full row + deprecation note, delete cuda121 row, delete rocm row + restore-path pointer (rocm7.1 facts — RESEARCH Pitfall 5). New pages compose with the pip-lane guidance. Every example edit lands with its `docs/` mirror byte-identical (WPAR-08).

**Mirror-discipline enforcement analog — `tests/test_runner_infra_contracts.py:209-216`:**
```python
source_bytes = (EXAMPLE_MCP_DIR / name).read_bytes()
mirror_bytes = (DOCS_MCP_MIRROR_DIR / name).read_bytes()
assert mirror_bytes == source_bytes, (
    f"docs/example/mcp_example/{name} drifted from its example/ "
    "source (check_docs_sync compares byte-for-byte) ...")
```

### Smoke dispatch + atomicity proofs (validation mechanics, 17-01 precedent)

**Analog:** `.planning/phases/17-publish-chain-robustness/17-01-SUMMARY.md`
- Smoke dispatch: `gh workflow run "Publish to PyPI" --ref <surgery-branch>`; assert `gh api runs/<id>/jobs` jq: `publish` conclusion `skipped` AND `smoke` conclusion `success` (run 38554913144 precedent). No publish.yml trigger edits (release-gated).
- Pickaxe atomicity: `git log -S '<string>'` on the alias declaration, a renamed install line, and the guard tests must yield the identical introducing hash ("ATOMIC-COMMIT-OK" phrasing, SUMMARY :43-56).
- Fresh-venv resolve: `W=$(ls dist/*.whl); uv pip install --python /tmp/xe-<extra>/bin/python "$W[<extra>]"` — concrete path REQUIRED; glob+bracket fails (RESEARCH Pitfall 3). Alias proof: assert `isort` present in the `base`-alias venv.

## Shared Patterns

### Atomic-commit / pickaxe discipline
**Source:** 17-01-SUMMARY.md :27-56. Apply to: the whole phase (one atomic commit: pyproject + workflows + guard tests + docs sweep + census re-pin). Pickaxe both the introduced and removed strings (`git log -S 'base = [' ` etc.).

### Sanctioned-residue grep gates
**Source:** `tests/test_models_lock_contracts.py:71-85` (reasoned allowlist) + phase-13 verbatim-record discipline (PLAN/SUMMARY reproduce the exact N-line grep output). Apply to: `.[base]` gate and `cuda121|torch-rocm|\[rocm\]` gate in `tests/test_extras_guard.py`.

### Extras member guards (tomllib)
**Source:** `tests/test_extras_guard.py:35-71` + `tests/models/test_plant_helixseek_fla_kernels.py:18-38` (`_META_EXTRA_RE` exact-member parsing, never substring). Apply to: all new guard tests.

### Environment markers (PEP 508)
**Source:** `pyproject.toml:89,118`. Apply to: triton-windows declaration in fla.

### Docs mirror byte-identity
**Source:** `tests/test_runner_infra_contracts.py:209-232` + `tests/scripts/test_check_docs_sync.py`. Apply to: every example/ ↔ docs/ pair touched by the ~69-occurrence sweep.

## No Analog Found

None — every surface is an edit-in-place of an existing tracked file or a new docs page whose structure analogs installation.md. The two NEW docs pages (GPU matrix, honesty matrix) have content fully pre-verified in 18-RESEARCH (verbatim 0542a6b record; live-probe matrix) — planner should paste from RESEARCH, not re-derive.

## Metadata

**Analog search scope:** pyproject.toml, tests/ (root + models + scripts), .github/workflows/, docs/getting_started/, README.md, .planning/phases/{16,17}
**Files scanned:** ~14
**Pattern extraction date:** 2026-10-14
