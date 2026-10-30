# Phase 28: Optional extras: deepspeed & flash-attn model support - Pattern Map

**Mapped:** 2026-10-29
**Files analyzed:** 15 (2 new, 13 modified)
**Analogs found:** 15 / 15 (research phase already read every touched file line-by-line; excerpts below verified against HEAD of branch v1.4)

All analog paths verified git-tracked (`git ls-files` non-empty). No mirror paths.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `pyproject.toml` (modify) | config | — | existing `mamba`/`fla` extras blocks (self-analog, lines 175-196) | exact |
| `scripts/install_flash_attn.sh` (NEW) | utility/script | file-I/O + build | `scripts/install_mamba.sh` | exact |
| `.github/workflows/extras_compat.yml` (NEW) | config/CI | batch | `.github/workflows/census_compat.yml` | exact |
| `.github/workflows/ci.yml` (modify) | config/CI | batch | existing `run_test_mamba` boolean block (self-analog, lines 61-64, 321) | exact |
| `dnallm/models/registry.py` (modify) | model/schema | — | existing `tf_compat_status` Literal (self-analog, lines 133-141) | exact |
| `dnallm/models/support.py` (modify) | model/vocab | — | existing `SUPPORT_WORDS`/`SUPPORT_WORD_BY_REPO` dicts (self-analog) | exact |
| `scripts/census_compat.py` (modify) | utility/tool | batch | `compat_verdict` branch chain (self-analog, lines 1479-1537) | exact |
| `dnallm/models/model.py` (modify) | model/loader | request-response | generic-chain wrap sites (self-analog, lines 1368-1369, 1438-1439) + `dnallm/models/special/gpn.py:23-33` | exact |
| `dnallm/models/special/crossdna.py` (modify) | model/handler | request-response | `dnallm/models/special/gpn.py:23-33` (D-10 interception shape) | exact |
| `tests/models/compat_census.json` (modify) | test-data | — | existing `disposition: {"skip": ...}` rows (self-analog) | exact |
| `tests/models/test_model_info_registry.py` (modify) | test | — | existing registry pins (self-analog) | exact |
| `tests/models/test_support.py` (modify) | test | — | existing word-table pins (self-analog) | exact |
| `tests/models/test_census_compat.py` (modify) | test | — | existing verdict-mapping pins (self-analog) | exact |
| `tests/models/test_compat_docs_consistency.py` (modify) | test | — | existing `Counter` cell-count gate (self-analog, lines 316-339) | exact |
| `tests/models/test_special/test_family_handlers.py` (modify) | test | — | existing family-handler tests (self-analog; note path is `tests/models/test_special/`) | exact |
| `docs/user_guide/optional_dependencies.md` (modify) | docs | — | current 27.2-worded sections (self-analog; GPN/Omni-DNA sections stay verbatim) | exact |
| `docs/user_guide/model_compatibility.md` (regen) | docs/generated | — | generator output (`census_compat.py --emit-docs`); never hand-edit | exact |
| `dnallm/models/model_info.yaml` (modify) | model/registry-data | — | writer-only: `census_compat.py --write-registry --apply` (never hand-edit YAML) | exact |

## Pattern Assignments

### `pyproject.toml` (config)

**Analog:** its own `mamba`/`fla` extras (lines 175-196) and the `all` meta-extra (lines 157-159).

Current shape to modify (lines 153-196, verbatim key excerpts):

```toml
all = [
    "dnallm[full,dev,test,notebook,docs,ui,mcp,fla,tui]",   # drop ,fla (D-09/D-13)
]
# mamba extra's comment block (lines 175-180) is the precedent for the
# deepspeed extra's mandatory band-cap comment citing the 0.19.7 removal:
mamba = [
    "causal_conv1d==1.7.0",
    "mamba-ssm==2.3.2.post1",
    "torch>=2.6.0,<2.15",
]
fla = [                                    # DELETE entire block (D-09); move
    "flash-linear-attention>=0.5.2,<0.6",  # these two lines VERBATIM into
    "triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'",
]                                          # [project] dependencies
```

New extras (research-measured bands, no platform markers per D-11, no meta-extra membership per D-13):

```toml
# deepspeed.ops.sparse_attention was REMOVED upstream in v0.19.7 (2026-09-16);
# GENA-LM-BigBird's remote code imports it — cap mandatory.
deepspeed = ["deepspeed>=0.19.6,<0.19.7"]
# flash-attn sdist has no pyproject.toml and imports torch in setup.py:
# end users must install with --no-build-isolation (documented, not marker-able).
flash-attn = ["flash-attn>=2.8.3.post2,<2.9"]
```

### `scripts/install_flash_attn.sh` (NEW — utility, file-I/O + build)

**Analog:** `scripts/install_mamba.sh` (fork wholesale; 347 lines, read header above).

Copy verbatim from analog:
- **Header/usage block** (lines 1-51): WHY/WHAT/SUPPLY-CHAIN comment structure, flag docs. Keep the exact flag set (`--default`, `--patch-only`, `--verify-file`, `--index-url`, `--python`, `--wheelhouse`, `--no-install`).
- **Pinned constants** (lines 56-61): swap in `FLASH_ATTN_VERSION="2.8.3.post2"`, `FLASH_ATTN_SDIST_SHA256="d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b"` (research-verified 2026-10-29).
- **Hash gate BEFORE extraction** — the entire reason pip is avoided until after verification: "pip builds sdist metadata by EXECUTING setup.py from the downloaded archive, which would run unverified bytes before the hash gate" (lines 25-27). Fetch via PEP 503 simple page + stdlib urllib (analog lines 164-204), never pip.
- **Occurrence-exact patch assertions**: mamba counts its 4 `-std=c++17` edits; flash-attn patches setup.py to add `-gencode arch=compute_121,code=sm_121` (upstream branches only cover 80/90/100/120) with the same occurrence-count discipline.
- **Build step** mirrors mamba step-5 shape (`pip wheel <extracted-tree>` — analog lines ~276-277) but MUST add `--no-build-isolation` (sdist setup.py imports torch unconditionally; isolated build dies `ModuleNotFoundError: torch`).

Deltas from analog: no `--no-deps` wheel-patch install pipeline of a second package (drop causal_conv1d handling); document `MAX_JOBS` cap (research A1: 1-3 h build on 20-core GB10).

### `.github/workflows/extras_compat.yml` (NEW — CI lane, batch)

**Analog:** `.github/workflows/census_compat.yml` (164 lines, read in full above).

Copy from analog:
- **Security posture** (lines 17-31): self-hosted runner gate. Analog is `on: workflow_dispatch` + `if: github.event_name == 'workflow_dispatch'`; this phase adds cron per D-06 (research Open Question 1 recommends `schedule:` + cron-string gate + `dispatch ∧ run_extras_compat`, inert until v1.4→main merge). Never push/pull_request reachable; `permissions: contents: read`; `runs-on: [self-hosted, dnallm-nightly]`.
- **GPU fail-closed check** (lines 37-54): gpu_absent marker + exit 1 + `if: always()` artifact upload.
- **uv install guard** (lines 60-80): skip curl|sh when `~/.local/bin/uv` already resolves (TUNA-configured box).
- **Env setup** (lines 82-108): `uv venv` + `uv pip install -e ".[full]"` + bootstrap pip + pinned installer script call — this lane swaps `install_mamba.sh` for `install_flash_attn.sh` (same flag shape: `--python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`) and adds the deepspeed extra via plain pip (`uv pip install 'dnallm[deepspeed]'` — band cap makes it safe; discretion default, no script).
- **Seed-from-committed-artifact + verify** (lines 147-151): `cp tests/models/compat_census.json census-artifacts/...` then run — this lane's probe command (D-07/D-08):

```bash
python scripts/census_compat.py --run --with-loads --verify \
  --entries AIRI-Institute/gena-lm-bigbird-base-sparse,chengCCC/CrossDNA_pretrain \
  --artifact census-artifacts/census-fresh.json
```
(Confirm exact `--entries` flag name against `census_compat.py` CLI at execution; research confirms single-entry mode exists.)

Second pass gate (D-08 gate 2): fresh records must match committed `extra-required-tf5` rows — the `--verify` drift gate covers this once the disposition seed (below) lands.

### `.github/workflows/ci.yml` (modify — CI lane wiring)

**Analog:** the existing three-boolean block, self-analog.

Boolean declaration (lines 51-64) — add `run_extras_compat` with identical shape:

```yaml
      run_test_mamba:
        description: "Run the test-mamba lane (mamba kernel build + census) on this dispatch"
        type: boolean
        default: true
```

Job-level gate (line 321) — the `inputs.<flag>` form, never `github.event.inputs`:

```yaml
    if: (github.event_name == 'workflow_dispatch' && inputs.run_test_mamba) || (github.event_name == 'schedule' && github.event.schedule == '0 3 * * *')
```

**Also (D-09 fallout, atomic with pyproject edit):** drop `,fla` from install lines at **ci.yml:382, :603, :819** (`.[full,fla]` → `.[full]`; `.[full,fla,dev,mcp]` → `.[full,dev,mcp]`) and review the comment block at ci.yml:361-368 (the `not slow chunk_kda import guard` note). uv fails loudly on unknown extras — miss one and the lane env-create goes red.

### `dnallm/models/registry.py` (modify — enum extension, D-01)

**Analog:** self — the `tf_compat_status` Literal (lines 133-141). Insertion-only: add `"extra-required-tf5"` to the list; change nothing else. `extra="forbid"` on the field (line 124) stays.

### `dnallm/models/support.py` (modify — vocabulary, D-14)

**Analog:** self — three dicts.

```python
SUPPORT_WORDS["extra-required-tf5"] = "works with the deepspeed or flash-attn extra"  # or per-override only
SUPPORT_WORD_BY_REPO["AIRI-Institute/gena-lm-bigbird-base-sparse"] = "works with the deepspeed extra"  # replace "needs deepspeed" (line 55)
SUPPORT_WORD_BY_REPO["chengCCC/CrossDNA_pretrain"] = "works with the flash-attn extra"  # NEW
DOCS_STATUS_CELLS["extra-required-tf5"] = "..."  # generated-page cell, e.g. "Works after installing an optional extra (tested)"
```

Key mechanics (support.py lines 98-121): `support_word()` falls back to `SUPPORT_WORDS.get(status, UNTESTED_WORD)` — the status key MUST exist in `SUPPORT_WORDS` or every non-overridden entry silently reads "untested". Override wins over status word (line 118-120). TUI/MCP/preflight consume this single source; change nothing in those three surfaces (27.2-05 tail-append already handles it). Note the docstrings at lines 49-51 and 58-61 reference "three repos"/"three strings" — update counts.

### `scripts/census_compat.py` (modify — disposition branch, D-02/D-04)

**Analog:** self — `compat_verdict` (lines 1479-1537, read above).

Insert ONE branch immediately BEFORE the `skip_reason` branch (line 1512) — order is load-bearing (a superseded `{"skip"}` seed would otherwise win):

```python
    extra = disposition.get("extra")
    if isinstance(extra, str) and extra:
        return "extra-required-tf5", f"{stamp}; {extra}"
```

Also extend `TF_COMPAT_STATUSES` frozenset (post-`run_census` region) with `"extra-required-tf5"`, and bump `DOCS_DATA_DATE = "2026-10-29"` constant (line ~1803) in the same change as the flip (D-15; pinned constant, never wall-clock).

Evidence strings (hand-authored into the seeded artifact rows; D-04 — BigBird names BOTH enablers):
- BigBird: `hf AIRI-Institute/gena-lm-bigbird-base-sparse@3a9028d36137341c10df771c3bd93de530d74594; load+forward verified via the deepspeed extra (deepspeed 0.19.6) + the _patch_apply_chunking_to_forward rung`
- CrossDNA: `hf chengCCC/CrossDNA_pretrain@cd118dc45eaa0f2debb05f03ed0290fe56a78ab1; load+forward verified via the flash-attn extra (flash-attn 2.8.3.post2)`

Why the disposition (not a `SHIM_RESTORED_LOADS`-style green-only dict): both rows are family reps; in the DEFAULT env load fails and, without a surviving disposition, `compat_verdict` raises ValueError (lines 1534-1537) → census lane red, violating D-02. `merge_records` (lines ~1296-1329) carries a disposition forward whenever status is unchanged (`import-ok` in both envs), and the green-drop exception drops only `skip` dispositions — replacing `{"skip": ...}` with `{"extra": ...}` keeps BOTH envs green through the same branch. Do NOT touch `merge_records`, `census_compat.yml`, or `SHIM_RESTORED_LOADS`.

### `dnallm/models/model.py` (modify — BigBird ImportError interception, D-17)

**Analog 1 (wrap sites):** self — the two `except Exception` wraps.

```python
# lines 1368-1369 (finetuned-checkpoint chain) and 1438-1439 (generic chain)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e
```

BigBird's deepspeed failure arrives as a construction-time `ModuleNotFoundError` inside `from_pretrained`, already caught here. Interception (implementer discretion on layer): before the generic wrap, inspect the cause chain — if any `__cause__`/`__context__` is a `ModuleNotFoundError` with `name == "deepspeed"` (or message containing `deepspeed.ops.sparse_attention`), raise the ImportError below instead.

**Analog 2 (error shape):** `dnallm/models/special/gpn.py:23-33` — the 27.2 D-10 precedent, replicate exactly (package + model + pip command + caveat + docs URL, chained `from e`):

```python
            except ImportError as e:
                raise ImportError(
                    f"gpn package is required for "
                    f"{model_name} but not installed. "
                    "Install it with: pip install gpn "
                    ...
                    "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/"
                ) from e
```

Phase-28 BigBird variant names `pip install 'dnallm[deepspeed]'` + the deepspeed `<0.19.7` cap caveat (module removed upstream). Plain language for biologists (owner directive).

### `dnallm/models/special/crossdna.py` (modify — CrossDNA interception, D-17)

**Analog:** `dnallm/models/special/gpn.py:23-33` (same shape as above).

Interception points: wrap the two `from_pretrained` calls at **crossdna.py:522** (`AutoModelForMaskedLM`) and **:574** (`AutoModelForSequenceClassification`). Catch `ImportError` whose message carries "Flash Attention" (fla's verbatim raise: `Please install Flash Attention via \`pip install flash-attn --no-build-isolation\` first`, fired inside `fla.layers.attn.Attention.__init__`) and re-raise with the appended plain-language line naming `pip install --no-build-isolation 'dnallm[flash-attn]'` + docs URL, `from e`. `ImportError` must NOT be swallowed by a broader except — keep it specific and re-raised.

### Test files (modify — offline gates)

All self-analogs; the pattern in each is "fail-closed pin re-derived from committed data, updated atomically with the flip":

| File | Change |
|---|---|
| `tests/models/compat_census.json` | Replace the two rows' `disposition: {"skip": ...}` with `{"extra": "<evidence text minus stamp>"}`; status stays `import-ok` |
| `tests/models/test_census_compat.py` | verdict-mapping pins learn the `extra` disposition branch → `extra-required-tf5` |
| `tests/models/test_support.py` | word-table pins: BigBird word flip, CrossDNA override added, status word/cell added |
| `tests/models/test_model_info_registry.py` | enum pin: Literal accepts `extra-required-tf5`; registry integrity pins for the two flipped rows |
| `tests/models/test_compat_docs_consistency.py` | the `Counter`-over-status cell-count gate (lines 316-339) becomes seven numbers; line 216 fail-closed on unknown status must accept the new value |
| `tests/models/test_special/test_family_handlers.py` | new D-17 interception tests (monkeypatched `from_pretrained` raising the verbatim fla/GENA errors → assert the ImportError message names the pip command) — note this file lives under `tests/models/test_special/` |

Owner rule: every `dnallm/` change ships with pytest in the same change, per-module ≥96% (plain `coverage run --include=`, pytest-cov broken).

### Docs (modify/regen)

- `docs/user_guide/optional_dependencies.md`: full affected-surface rewrite (D-16). Intro paragraph "what is an optional extra" in plain language; 4-row summary table (PlantHelixSeek → built-in, BigBird/CrossDNA → extra install, GPN/Omni-DNA unchanged); BigBird + CrossDNA sections become install guides with the exact commands `pip install 'dnallm[deepspeed]'` and `pip install --no-build-isolation 'dnallm[flash-attn]'` + Linux+CUDA platform statement + compile-time expectations. GPN/Omni-DNA sections stay byte-verbatim.
- `docs/user_guide/model_compatibility.md`: NEVER hand-edit — `python scripts/census_compat.py --emit-docs` after the registry flip + `DOCS_DATA_DATE` bump, one commit; byte-identity gate holds.

## Shared Patterns

### Supply-chain: sha256 gate before any byte executes
**Source:** `scripts/install_mamba.sh` lines 17-33, 56-61 (fetch via PEP 503 simple page + stdlib urllib, never pip, until after hash verification; no override flag/env bypass).
**Apply to:** `scripts/install_flash_attn.sh` (the only new executable surface in this phase).

### Fail-closed drift gates
**Source:** `census_compat.py --verify` (analog: census_compat.yml step at lines 147-151 + `verify_registry` UNEXPECTED diff, census_compat.py ~2257-2262).
**Apply to:** extras lane gate 2; every pytest pin file. Unknown/unclassifiable shapes raise, never pass.

### Honest-green evidence naming its enabler
**Source:** `compat_verdict` line 1521 (`shim-required-tf5, f"{stamp}; load+forward verified via rung {fix_rung}"`).
**Apply to:** both new evidence strings (D-04: BigBird names rung AND extra).

### Plain-language ImportError with pip command + docs URL, chained `from e`
**Source:** `dnallm/models/special/gpn.py:23-33` (omnidna.py:25-33 identical shape).
**Apply to:** model.py BigBird interception; crossdna.py CrossDNA interception.

### Single-source vocabulary
**Source:** `dnallm/models/support.py` (`SUPPORT_WORDS` / `SUPPORT_WORD_BY_REPO` / `DOCS_STATUS_CELLS`).
**Apply to:** docs generator, TUI suitability, MCP listing, preflight — all consume `support_word()`; no per-surface tables.

### Atomic landing
The pyproject fla deletion, the three ci.yml `,fla` removals (382/603/819), the registry flip, the artifact seed, the `DOCS_DATA_DATE` bump, and the docs regen each land in ONE change per group — split them and some gate goes red (Pitfalls 4/5/6 in 28-RESEARCH.md).

## No Analog Found

None. Every touched surface has an exact in-repo precedent; this phase is wiring, not tool-building.

## Metadata

**Analog search scope:** scripts/, .github/workflows/, dnallm/models/ (+special/), tests/models/, docs/user_guide/, pyproject.toml — all read directly this session or line-verified in 28-RESEARCH.md (2026-10-29).
**Files scanned:** ~20; excerpts verified against branch v1.4 HEAD.
**Pattern extraction date:** 2026-10-29
