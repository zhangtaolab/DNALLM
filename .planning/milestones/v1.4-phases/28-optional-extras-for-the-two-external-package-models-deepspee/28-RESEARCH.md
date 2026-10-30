# Phase 28: Optional extras: deepspeed & flash-attn model support - Research

**Researched:** 2026-10-29
**Domain:** Python optional-dependency architecture (pyproject extras + uv), CUDA extension source builds on aarch64/GB10, census/registry disposition tooling, GitHub Actions lane wiring
**Confidence:** HIGH (all load-bearing facts measured on this box or verified against primary registry/repo sources)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Census & Disposition Semantics

- **D-01: New registry status value `extra-required-tf5`**, mirroring the `shim-required-tf5` precedent (27.2 D-03): the verdict records BOTH the green load AND what it required (the extra) — never a "works as-is" claim. One new enum value; registry schema (`extra="forbid"`, insertion-only), docs status cells, and `support.py` all learn it.
- **D-02: The census keeps measuring the default install, forever.** `census_compat.yml` is byte-untouched; its default-install convention and 49-matched drift-gate semantics are unchanged. The two `extra-required-tf5` records keep measuring as failures in the default environment and are seeded into the drift gate as expected failures (the CR-01 unreachable-repo discipline). Extras-environment green evidence comes from separate probes that flip only these two entries.
- **D-03: Replica-grade evidence, in-phase flip** (27.2 `SHIM_RESTORED_LOADS` precedent): the local runner-box replica env (extras installed) runs the green probes (load + forward), and the committed `extra-required-tf5` flips land inside this phase. The workflow file lands on v1.4 simultaneously; the live lane becomes the durable guard after the v1.4→main merge. — **Reversibility:** reversible — registry rows flip back if a later probe goes red (green-drop rule symmetry).
- **D-04: Single status value; evidence names ALL dependencies.** BigBird's green load needs BOTH the `_patch_apply_chunking_to_forward` rung AND the deepspeed extra — its evidence string names both (27-03 precedent: rung name rides in evidence, never a new enum). CrossDNA's evidence names the flash-attn extra (fla is base after D-09). No combined enums.

#### CI Lane Architecture

- **D-05: New dedicated workflow `extras_compat.yml`** on the self-hosted GPU box, schedule/dispatch only, never PR-reachable (fork-PR safety rule). Single responsibility: install extras → run the two models' green probes → emit extras evidence. `census_compat.yml` gets zero edits.
- **D-06: Trigger cadence = nightly cron + `workflow_dispatch` + a fourth per-lane boolean** (`run_extras_compat`, default true) in the `ci.yml` nightly lane-selection block (261026-b34 pattern). Nightly guards upstream drift (model-repo changes, deepspeed/flash-attn upgrades); dispatch serves reproduction and the post-merge first live run.
- **D-07: Probes reuse `census_compat.py` single-entry mode** in the extras-equipped environment. Evidence format is identical to census records; the update-capable registry writer (27.2-01), drift gate, and docs generator need zero adaptation.
- **D-08: Two pass gates in the lane:** (1) both model probes green; (2) fresh extras-probe records match the committed registry `extra-required-tf5` records (mirror of the census drift gate, fail-closed) — upstream model-repo changes surface as red. Support-word/disposition consistency re-derivation stays in the pytest offline gate (27.1-01 pattern), not in the lane.

#### Extra Composition & Install Mechanics

- **D-09: `fla` is promoted to a base dependency** (owner directive: PlantHelixSeek must work out-of-the-box). `flash-linear-attention>=0.5.2,<0.6` plus the WPAR-07 Windows `triton-windows` marker line move verbatim into `[project]` dependencies; the `fla` extra definition is deleted; the `all` meta-extra drops its `fla` reference; `optional_dependencies.md`'s PlantHelixSeek section becomes "built-in". Breaking change accepted: `dnallm[fla]` syntax disappears (upgrading users get fla naturally on reinstall). — **Reversibility:** costly — undoing means re-publishing the install contract (base dep set shrink + re-introducing an extra) in a future release.
- **D-10: The CrossDNA extra is named `flash-attn`** and contains only flash-attn itself (fla is base after D-09). Named for the missing package, consistent with the `deepspeed` extra naming rule; no model-named extras.
- **D-11: No platform markers on either extra.** pip fails loudly on Windows/macOS at install time (compile failure) rather than silently installing an empty extra. Docs state Linux + CUDA as the supported platform for both extras.
- **D-12: CI installs flash-attn via a new sha256-pinned installer script** (`scripts/install_flash_attn.sh`) mirroring `install_mamba.sh` discipline exactly: pinned version + hash gate before any bytes are extracted or executed + mirror-source support; artifact is a pinned wheel URL if one exists for GB10 (aarch64 + cu130 + torch 2.11), else a pinned sdist with documented build flags. The lane calls only the script, never bare pip. The pyproject extra remains the end-user route. Wheel availability is a research-phase fact-finding item; the mechanism stance is locked.
- **D-13: Neither extra joins any meta-extra** (`all`/`full` stay without deepspeed/flash-attn). Both are heavy compile packages; `optional_dependencies.md` documents them as separately installed. `all` naturally slims when its `fla` reference drops (D-09).

#### Support Words & Docs Surface

- **D-14: Support words are `works with the deepspeed extra` / `works with the flash-attn extra`** — parallel in shape to the shim-required `works with built-in fixes` sentence. One vocabulary source shared across docs/TUI/MCP (27.2 D-16 discipline). The literal `pip install 'dnallm[deepspeed]'` command lives in docs detail text, never inside the short word (TUI/MCP column-width constraint, 27.2-05 tail-append layout).
- **D-15: `model_compatibility.md` regenerates with the flip; `DOCS_DATA_DATE` follows the evidence change.** Single-entry re-probe evidence suffices — no full 49-entry census re-run is required for a two-row change. Byte-identity regeneration gate keeps holding (quick-261029-c95 registry-data-change-drives-regen precedent).
- **D-16: `optional_dependencies.md` gets a full affected-surface rewrite:** a plain-language "what is an optional extra" intro paragraph; the summary table updated for four rows (PlantHelixSeek → built-in, BigBird/CrossDNA → extra install, GPN/Omni-DNA unchanged); BigBird and CrossDNA sections become practical install guides with the pip command + platform requirements (Linux + CUDA) + compile-time expectations. GPN/Omni-DNA sections stay verbatim (version-locked guidance cases).
- **D-17: Missing-extra errors are intercepted on the load chain** with one appended plain-language line naming the extra and the exact pip command (CrossDNA in its special handler; BigBird at the generic chain where the deepspeed `ImportError` is detected — exact interception layer is implementer discretion). Same shape as the GPN/Omni-DNA D-10 precedent from 27.2.

### Claude's Discretion

- Version bands for `deepspeed` and `flash-attn` in pyproject (research-measured against torch 2.11 / transformers 5.17)
- deepspeed lane install route — default: the plain `.[deepspeed]` extra via pip (regular PyPI wheel, band-pinned); escalate to a pinned script only if measurement shows a need
- Exact evidence-string format for the two flipped rows (follow the existing `hub repo@revision; cause` pattern)
- Exact seed mechanics for the two expected-failure rows in the drift gate
- Which pytest files host the new offline gates (existing census/compat test files preferred)
- The interception layer for BigBird's generic-chain deepspeed ImportError
- Whether the README supported-models statement needs an extras mention (only if it currently names the two models as broken)
- Final plan split and wave structure

### Deferred Ideas (OUT OF SCOPE)

- **GPN / Omni-DNA as extras** — permanently out (version-locked packages: gpn pins `transformers==5.15.0`, ai2-olmo pins `numpy<2`; they can never enter the main env). Roadmap-locked guidance cases; `optional_dependencies.md` keeps their verbatim guidance sections.
- **lgq12697 bulk ingestion (~130 unregistered ModelScope entries)** — separate v1.5 campaign (carried from 27.2).
- **Live dispatch of census + extras lanes** — both stay post-v1.4→main-merge operator steps (GitHub registers dispatchable workflows from the default branch only); replica evidence is the interim proof for both.
- **uv index pinning for flash-attn wheels** — rejected mechanism (would need new hosting infrastructure); revisit only if a future need emerges.
</user_constraints>

<phase_requirements>
## Phase Requirements

The phase carries no formal REQ IDs yet (owner decision 2026-10-29: option 1 chosen). The D-01..D-17 decisions are the requirement set.

| ID | Description | Research Support |
|----|-------------|------------------|
| D-01 | New enum value `extra-required-tf5` across registry/docs/support | Complete surface map (§ Enum Extension Surfaces) with verbatim current values from every file |
| D-02 | Default-install census keeps failing the two rows; seeded drift gate | Merge/disposition mechanics verified; symmetric seed design (§ Census Probe & Seed Mechanics) |
| D-03/D-04 | Replica-env green probes; evidence names all deps | Single-entry probe flow verified; both rows are family reps (loads run under `--with-loads`) |
| D-05–D-08 | `extras_compat.yml` lane + `run_extras_compat` boolean + two gates | Lane precedent anatomy (census_compat.yml + ci.yml boolean block) extracted |
| D-09 | fla → base dependency | Resolution analysis: no base-dep conflicts; 3 CI install lines must drop `,fla` (uv rejects unknown extras) |
| D-10–D-13 | Extra names, no markers, pinned installer, no meta-extras | flash-attn: no GB10 wheel exists → sdist branch of D-12 confirmed; deepspeed band capped by module removal (§ Standard Stack) |
| D-14–D-16 | Support words, docs regen, optional_dependencies rewrite | Word-source + docs-generator consumption paths mapped; DOCS_DATA_DATE mechanics confirmed |
| D-17 | Load-chain ImportError interception | Exact surfacing points verified for both models (construction-time, not module-import) |
</phase_requirements>

## Summary

Phase 28 converts the two fixable external-package models (GENA-LM-BigBird → deepspeed, CrossDNA → flash-attn) from "broken, install it yourself" into curated optional extras with CI-backed green evidence, and promotes `fla` (flash-linear-attention) to a base dependency so PlantHelixSeek works on a default install. This box is the runner-class GB10 dev box, so every environment fact below was measured locally: aarch64, torch 2.11.0+cu130, transformers 5.17.0, triton 3.6.0, fla 0.5.2 installed, deepspeed/flash-attn absent, nvcc at `/usr/local/cuda/bin/nvcc` (CUDA 13.0), GPU = NVIDIA GB10 sm_121, 20 cores, 121 GB RAM.

Three findings dominate the plan shape. **First: no prebuilt flash-attn artifact exists for GB10.** The v2.8.3.post2 GitHub release ships 48 wheels, all `linux_x86_64`, cu12, torch 2.4–2.8; PyPI is sdist-only. So D-12's wheel-vs-sdist branch settles on the **sha256-pinned sdist** (2.8.3.post2, sha256 `d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b`, 8.45 MB), built with documented flags. The build MUST run with `--no-build-isolation` (the sdist has no `pyproject.toml` and its `setup.py` imports torch unconditionally — an isolated PEP 517 build dies with `ModuleNotFoundError: torch`). **Second: deepspeed's `ops.sparse_attention` — the ONLY thing GENA's remote code imports — was removed upstream in v0.19.7 (2026-09-16); the last release shipping it is v0.19.6.** The extra must be capped `<0.19.7` or GENA still fails with deepspeed installed. The module itself is Triton-based (no `DS_BUILD_OPS`, no nvcc for this op; `triton._C.libtriton` imports clean under the installed triton 3.6.0). **Third: both missing-package failures fire at model CONSTRUCTION time, not module import** — GENA inside `BertSelfAttention.__init__` via `get_cls_by_name` → `importlib.import_module` (remote `modeling_bert.py:270/308`), CrossDNA inside `fla.layers.attn.Attention.__init__` (installed `fla/layers/attn.py:71`, verbatim raise below). Both statuses stay `import-ok` in the census artifact regardless of extras presence — the property that makes the disposition-carry seed work identically in default and extras environments.

**Primary recommendation:** Pin `deepspeed>=0.19.6,<0.19.7` and `flash-attn>=2.8.3.post2,<2.9` (both floors are the versions measured on the runner replica); fork `install_mamba.sh` into `install_flash_attn.sh` with the sdist sha256 above, `--no-build-isolation`, and an explicit sm_121 gencode patch (occurrence-counted, the mamba discipline); seed the two committed census rows with a new hand-authored disposition shape (e.g. `{"extra": "..."}`) and add one ordered branch in `compat_verdict` mapping it to `extra-required-tf5` — this single change keeps BOTH the default census lane and the new extras lane green without touching `census_compat.yml`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Extra declarations (`deepspeed`, `flash-attn`, fla promotion) | Package metadata (pyproject.toml) | CI install lines | The install contract is the deliverable; uv resolves extras, CI lanes prove them |
| flash-attn byte-integrity + build flags | `scripts/install_flash_attn.sh` (new) | extras_compat.yml lane | Supply-chain gate must precede any byte execution (install_mamba.sh discipline) |
| deepspeed install | pyproject extra (pip route, band-pinned) | — | Pure-python sdist builds without CUDA; no script needed unless measurement disagrees |
| Disposition flip + seeding | scripts/census_compat.py (`compat_verdict`, artifact) | registry writer (`--write-registry --apply`) | Measured evidence is the only flip authority (census discipline) |
| Status enum + support words | dnallm/models/registry.py + support.py | docs generator, TUI, MCP, preflight listing | Single-source vocabulary; all display surfaces consume `support_word()`/`DOCS_STATUS_CELLS` |
| Missing-extra error guidance | Load chain (model.py generic wrap; crossdna.py handler) | — | The ImportError must surface where construction happens, wrapped once with the pip command |
| Lane orchestration | .github/workflows/extras_compat.yml (new) | ci.yml boolean block | Self-hosted GPU box, dispatch/schedule only, fork-PR safety |
| Offline gates (word/disposition consistency) | tests/models/ (existing census/compat/support test files) | — | 27.1-01 pattern: gates re-derive from committed claims in pytest, not in the lane |

## Standard Stack

No new library enters `dnallm/` imports. The "stack" of this phase is two external install targets plus the existing tooling.

### Core

| Package | Version band (research-measured) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `deepspeed` | `>=0.19.6,<0.19.7` | GENA-LM-BigBird block-sparse attention (`deepspeed.ops.sparse_attention`) | Last release shipping `ops.sparse_attention` is 0.19.6; removed in 0.19.7 [VERIFIED: GitHub trees API bisect v0.18.5→v0.19.8]. `torch>=2.0.0` unpinned (no torch-2.11 conflict) [VERIFIED: PyPI JSON API requires_dist] |
| `flash-attn` | `>=2.8.3.post2,<2.9` | CrossDNA attention mixer via `fla.layers.attn.Attention` | Latest 2.x line (2.8.3.post2, 2026-10-13) [VERIFIED: PyPI JSON API]; fla 0.5.2 needs only `flash_attn_func` + `flash_attn_varlen_func`, both provided; `requires_dist: ['torch', 'einops']` unpinned [VERIFIED: PyPI JSON API] |
| `flash-linear-attention` (+`fla-core`) | `>=0.5.2,<0.6` (existing pin, moves to base) | PlantHelixSeek remote code (KDA kernels) | Already measured in-tree; bare requires pull no torch pin [VERIFIED: installed metadata — requires: `fla-core==0.5.2`, `transformers>=4.45.0`; fla-core bare requires: `einops`] |

### Supporting (existing, unchanged)

| Tool | Version (on box) | Purpose |
|------|------------------|---------|
| uv | 0.13.6 (`~/.local/bin/uv`, TUNA default index in `~/.config/uv/uv.toml`) | Lane env creation + installs |
| nvcc / CUDA toolkit | CUDA 13.0 at `/usr/local/cuda/bin/nvcc`; `torch.utils.cpp_extension` resolves `CUDA_HOME=/usr/local/cuda` | flash-attn source build |
| ninja | 1.13.2 (venv bin + base dep `ninja>1`) | Build backend for both pinned builds |
| triton | 3.6.0 (torch cu130 wheel dependency) | deepspeed sparse_attention kernels; `triton._C.libtriton` imports OK [VERIFIED: local venv probe] |
| pytest + plain `coverage run` | existing (pytest-cov broken locally on py3.13/numpy 2.5.3 — project convention) | Offline gates; per-module ≥96% for new code |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pinned-sdist flash-attn build | Third-party GB10 wheel repos (dgx-spark-llm-stack, mjun0812 prebuild wheels) | Rejected: uncurated hosting; D-12 locked the sha256-pinned-from-index mechanism; hash gate makes TUNA/PyPI route safe |
| `deepspeed` capped `<0.19.7` | Unpinned latest | Impossible: `ops.sparse_attention` gone in 0.19.7 — GENA's import fails even with deepspeed installed |
| Explicit sm_121 gencode patch in installer | Rely on default arch list (sm_120 SASS; binary-compat loads on sm_121) | Default build embeds 4 arch families (80/90/100/120) → ~4x compile time; the 120→121 load relies on the same-major rule [CITED, derived] — explicit 121 is deterministic; both routes documented, execution measures |

**Installation (end-user route, documented in optional_dependencies.md):**

```bash
# deepspeed — plain pip works (sdist builds pure-python; ops this model needs are Triton)
pip install 'dnallm[deepspeed]'

# flash-attn — the sdist imports torch at build time with no pyproject.toml,
# so build isolation fails; the documented command MUST carry --no-build-isolation
pip install --no-build-isolation 'dnallm[flash-attn]'
```

**Version verification (2026-10-29):**
- `deepspeed` PyPI latest 0.19.8 (2026-10-20) — sdist only, 2,020,482 bytes, sha256 `9e491abd0827ea5c389dd702e15e11e13d83d904bac4a20d6c2a4cb37207a9af` [VERIFIED: PyPI JSON API; identical sha on TUNA]. TUNA carries 0.19.6/0.19.7/0.19.8 sdists.
- `flash-attn` PyPI latest 2.8.3.post2 (2026-10-13) — sdist only, 8,453,127 bytes, sha256 `d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b`. TUNA carries `flash_attn-2.8.2/2.8.3/2.8.3.post1/2.8.3.post2.tar.gz`.
- Release-asset matrix (settles D-12's branch): v2.8.3.post2 = 48 assets, all `flash_attn-2.8.3.post2+cu12torch{2.4..2.8}cxx11abi{FALSE,TRUE}-cp{39..313}-*_linux_x86_64.whl`. Zero aarch64, zero cu13, zero torch≥2.9 assets; fa4 beta tags (v4.0.0.beta36/37, Oct 2026) carry 2 non-wheel assets each. [VERIFIED: GitHub releases API]

## Package Legitimacy Audit

> Gate run via `gsd-tools query package-legitimacy check --ecosystem pypi deepspeed flash-attn flash-linear-attention` (2026-10-29).

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| deepspeed | PyPI | latest publish 2026-10-20; project since 2020 | unknown (gate signal null) | github.com/deepspeedai/DeepSpeed | SUS | Flagged — see resolution below |
| flash-attn | PyPI | latest publish 2026-10-14; project since 2022 | unknown (gate signal null) | github.com/Dao-AILab/flash-attention | SUS | Flagged — see resolution below |
| flash-linear-attention | PyPI | latest publish 2026-10-23; project since 2023 | unknown (gate signal null) | github.com/fla-org/flash-linear-attention | SUS | Flagged — see resolution below |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** all three — but the verdicts are artifacts of the gate's recency heuristic (it reads the *latest release* publish date, all within October 2026, and PyPI download stats returned null). Mitigating evidence gathered this session: each package resolves to its canonical source repo [VERIFIED: PyPI JSON API repo URLs]; each has a multi-year release history cross-checked against canonical GitHub tags (deepspeed v0.16.x→v0.19.8 line bisected; flash-attn v2.7.x→2.8.3.post2; fla version line pinned in-repo since Phase 25 era); the sdist sha256 for deepspeed 0.19.8 is byte-identical on PyPI and TUNA. Strongest of all: **the owner explicitly named all three packages in locked decisions D-09/D-10/D-12** — the human verification the checkpoint would seek is already recorded in CONTEXT.md. Recommendation: no `checkpoint:human-verify` install tasks needed; the sha256 gate on flash-attn and the band cap on deepspeed are the real integrity controls.

## Architecture Patterns

### System Architecture Diagram

```
                        ┌────────────────────────────────────────────────┐
                        │ pyproject.toml (install contract)              │
                        │  base += flash-linear-attention (D-09)          │
                        │  extra: deepspeed>=0.19.6,<0.19.7               │
                        │  extra: flash-attn>=2.8.3.post2,<2.9            │
                        │  [fla] extra DELETED; all drops fla             │
                        └──────┬───────────────────────────┬─────────────┘
                               │                           │
              default install  │              extras install│(pip / install_flash_attn.sh)
                               ▼                           ▼
        ┌────────────────────────────────┐   ┌──────────────────────────────────┐
        │ census_compat.yml (BYTE-       │   │ extras_compat.yml (NEW)          │
        │ UNTOUCHED, default env)        │   │  uv venv + .[full]               │
        │  .[full] (fla now rides base)  │   │  + deepspeed band via pip        │
        │  + install_mamba.sh            │   │  + scripts/install_flash_attn.sh │
        │  49-entry census + --verify    │   │    (sha256 gate → patch → build) │
        └──────────────┬─────────────────┘   │  probes: census_compat.py        │
                       │                     │   --run --entries <2 repos>      │
                       │                     │   --with-loads --verify          │
                       ▼                     └──────────────┬───────────────────┘
        ┌────────────────────────────────┐                  │
        │ tests/models/compat_census.json│                  │
        │  2 rows: status=import-ok      │◄──── merge_records carries the new  │
        │  disposition {"extra": ...}    │      {"extra"} disposition on the  │
        │  (seed, replaces {"skip"})     │      unchanged import-ok status    │
        └──────────────┬─────────────────┘                  │
                       │ compat_verdict: disposition.extra  │
                       │  → "extra-required-tf5" (new branch,│
                       │  ordered BEFORE skip/green)         │
                       ▼                    ▼                ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ --write-registry --apply → model_info.yaml (2 rows flip)     │
        │ registry.py Literal + census TF_COMPAT_STATUSES gain the     │
        │ value; support.py words/cells; --emit-docs regenerates       │
        │ model_compatibility.md (+DOCS_DATA_DATE bump, D-15)          │
        └──────────────┬───────────────────────────────────────────────┘
                       │ offline pytest gates (word/docs/registry pins)
                       ▼
        TUI suitability column / MCP listing / preflight ListingEntry.support
        (27.2-05 tail-append; _AVAILABILITY_COLUMN stays 4)
```

Trace: a user's default install → census artifact keeps recording the two load failures → seeded `{"extra"}` disposition maps both to `extra-required-tf5` → registry flip on replica-measured green → all three display surfaces show "works with the deepspeed/flash-attn extra" → the nightly extras lane re-proves the green and the drift gate holds both worlds honest.

### Pattern 1: The symmetric disposition seed (D-02 + discretion "exact seed mechanics")

**What:** Both target rows currently sit in the committed artifact as `status: "import-ok"`, `load.ok=false`, `disposition: {"skip": "<reason>"}`, `is_family_rep: true` [VERIFIED: tests/models/compat_census.json — BigBird record and CrossDNA record]. In BOTH environments (default: load fails; extras: load greens) the import status stays `import-ok` — fla only *warns* at import when flash-attn is absent, and GENA's deepspeed import is function-local inside `__init__`. `merge_records` carries a disposition forward whenever `previous.get("status") == record.get("status")` [VERIFIED: scripts/census_compat.py:1303-1329], and its green-drop exception drops ONLY a `skip` disposition on a fresh green load+forward (`prior_is_skip`) [VERIFIED: scripts/census_compat.py:1311-1323].

**When to use:** Replace the two committed `{"skip": ...}` dispositions with a hand-authored `{"extra": "<plain reason naming the extra>"}` shape, and add one branch in `compat_verdict` before the skip branch. Result: default census lane → record (import-ok, load fail, disposition.extra) maps to `extra-required-tf5` = committed → green; extras lane → record (import-ok, load green, disposition.extra carried) maps through the SAME branch → green. Symmetric, zero `census_compat.yml` edits, zero `merge_records` changes.

**Why not `EXTRA_REQUIRED_LOADS` alone (the `SHIM_RESTORED_LOADS` pattern):** a repo→extra dict consulted only on green load [cf. census_compat.py:1522-1526] leaves the DEFAULT-env record (load fail, no surviving disposition once `skip` is replaced) unmappable — `compat_verdict` raises `ValueError` ("carries no green load and no skip disposition") because both rows are family reps [VERIFIED: census_compat.py:1527-1537] → the default census lane goes red, violating D-02. The disposition branch is required; the dict is optional sugar on top.

**Drift semantics preserved:** if a transformers bump or model-repo change flips `status` away from `import-ok`, `merge_records` drops the stale disposition and stamps `previous_status` [VERIFIED: census_compat.py:1296-1302] → verdict disagrees with committed → lane red. Exactly the upstream-drift tripwire D-06 wants.

### Pattern 2: sha256-pinned installer fork (D-12)

Fork `scripts/install_mamba.sh` (read in full this session) into `install_flash_attn.sh`, preserving: pinned-constants header, `--verify-file` / `--patch-only`-style flags, `--index-url` mirror support (fetch from the PEP 503 simple page, never pip resolution — pip executes setup.py for sdist metadata), hash gate BEFORE extraction, occurrence-exact patch assertions. Deltas for flash-attn: no `--no-deps` wheel-patch pipeline (the build is the point), plus `pip wheel <extracted-tree> --no-build-isolation --no-deps` mirroring the mamba step-5 shape [VERIFIED: install_mamba.sh:276-277].

### Pattern 3: Load-chain interception (D-17), GPN/Omni-DNA precedent shape

The 27.2 precedent raises a plain-language `ImportError` with: package name + model name + install command + measured caveat + docs URL, chained `from e` [VERIFIED: dnallm/models/special/gpn.py:24-33, omnidna.py:25-33 — quoted verbatim in Code Examples]. For this phase the failures arrive as construction-time errors inside `from_pretrained`, already wrapped by the generic chain's `except Exception as e: raise ValueError(f"Failed to load model: {e}") from e` [VERIFIED: dnallm/models/model.py:1368-1369 and 1438-1439] — interception must inspect the cause chain for `ModuleNotFoundError` named `deepspeed` (BigBird, generic chain) and for the fla `ImportError` message (CrossDNA, inside `_handle_crossdna_models` around its two `from_pretrained` calls at crossdna.py:522/574).

### Anti-Patterns to Avoid

- **Bare `pip install 'dnallm[flash-attn]'`** — PEP 517 isolation builds the sdist without torch → `ModuleNotFoundError: torch` in the build env. The extra exists to express the dependency contract; docs and lane must use `--no-build-isolation` (fla's own error message teaches exactly this).
- **Leaving `,fla` in any CI install line after deleting the extra** — uv fails loudly on unknown extras; `.[full,fla]` at ci.yml:382/603 and `.[full,fla,dev,mcp]` at ci.yml:819 ALL need the `,fla` dropped.
- **A `SHIM_RESTORED_LOADS`-only seed** — reds the default census lane (Pattern 1 analysis).
- **Unpinned `deepspeed` in the extra** — any resolve ≥0.19.7 installs a package whose `ops.sparse_attention` no longer exists; GENA stays broken and the honest-green claim becomes a lie.
- **Wall-clock dates in regenerated docs** — `DOCS_DATA_DATE` is a pinned constant [VERIFIED: census_compat.py:1803 `DOCS_DATA_DATE = "2026-10-29"`]; bump it as part of the flip change, never via `date`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Green-evidence artifact + verdict mapping | New probe tool | `census_compat.py --run --entries <repo> --with-loads` (D-07 locked) | Merge/verify/disposition machinery is measured-green (49 matched on this box's replica, 27.2-03) |
| Registry write-back | YAML surgery | `census_compat.py --write-registry --apply` (update-capable writer, 27.2-01: replaces changed canonical tf_compat triples in place) | Hand-edited YAML risks non-canonical layouts the writer fail-closes on |
| Docs page regen | Manual table edits | `census_compat.py --emit-docs` + byte-identity gate | quick-261029-c95 precedent: registry-data-change-drives-regen |
| Byte-integrity of downloaded sdists | ad-hoc curl+check | The install_mamba.sh gate skeleton (sha256 before extraction, index-declared-hash precheck) | pip executes sdist setup.py during metadata resolution — unverified bytes run before pip reports success |
| Support-word mapping in TUI/MCP/preflight | Per-surface word tables | `support_word(entry)` + `SUPPORT_WORD_BY_REPO` overrides (single source, 27.2 D-16) | Three surfaces already consume it; a fourth mapping invites drift |

**Key insight:** every mechanism this phase needs — single-entry probes, disposition carrying, green-drop, update-capable writer, drift gate, docs generator — already exists and is test-pinned. The phase is wiring, not tool-building.

## Runtime State Inventory

> The fla-extra→base promotion is a published-install-contract migration; inventory taken per protocol.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `tests/models/compat_census.json`: 2 records get new dispositions (BigBird @revision 3a9028d36137341c10df771c3bd93de530d74594, CrossDNA @revision cd118dc45eaa0f2debb05f03ed0290fe56a78ab1); `dnallm/models/model_info.yaml`: 2 rows flip tf_compat_* | Committed-data edit via the census writer (code edit; no external data migration) |
| Live service config | None — no external service stores the extra names (verified: extras exist only in pyproject + workflows + docs; MCP/TUI derive from the registry at runtime) | None |
| OS-registered state | None — no scheduled tasks/launchd/systemd units reference extras; GitHub workflow registration is default-branch-only (deferred live dispatch, D-05) | None (post-merge dispatch is the recorded operator step) |
| Secrets/env vars | None — no secret or env var keys reference `fla`/`deepspeed`/`flash-attn` extras | None |
| Build artifacts | Dev venv has fla 0.5.2 already (no action — base promotion adds nothing new); lane venvs are ephemeral per run; no persistent wheel caches keyed by extra name | None — do NOT clean local model caches (owner standing directive) |

## Common Pitfalls

### Pitfall 1: flash-attn build dies in isolation (`ModuleNotFoundError: torch`)
**What goes wrong:** `pip install flash-attn` / `uv pip install 'dnallm[flash-attn]'` builds the sdist in an isolated env; setup.py's unconditional `import torch` fails (sdist has NO `pyproject.toml`, so isolation installs only implicit setuptools+wheel).
**Why:** upstream deliberately does not declare torch as a build dep (arch detection needs the target env's torch).
**How to avoid:** `--no-build-isolation` everywhere (lane script + docs command). Build deps must pre-exist: torch, ninja, setuptools, wheel, packaging — ALL already in dnallm base deps (`ninja>1`, `wheel>=0.45.0`) [VERIFIED: pyproject.toml:24-73].
**Warning signs:** `ModuleNotFoundError: No module named 'torch'` during `Getting requirements to build wheel`.

### Pitfall 2: deepspeed band ceiling (module removed upstream)
**What goes wrong:** an unpinned or `>=0.19.7` resolve silently installs a deepspeed without `ops.sparse_attention`; GENA's `get_cls_by_name`/`from deepspeed.ops.sparse_attention import SparseSelfAttention` still fails.
**Why:** upstream removed the module in v0.19.7 (2026-09-16).
**How to avoid:** cap `<0.19.7` with a pyproject comment citing the removal; the nightly extras lane catches any future re-resolution drift.
**Warning signs:** `ModuleNotFoundError: No module named 'deepspeed.ops.sparse_attention'` despite deepspeed installed.

### Pitfall 3: sm_121 absent from flash-attn's gencode branches
**What goes wrong:** setup.py's arch handling has no "121" case: `os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120")` with branches only for 80/90/100/120 [VERIFIED: Dao-AILab setup.py v2.8.3.post2 lines 69-70, 179-191]. A default build emits compute_80/90/100/120 gencodes; loading on sm_121 then rests on the same-major binary-compat rule (sm_120 cubin runs on sm_121; the reverse is false) [CITED: NVIDIA Hopper Compatibility Guide same-major rule; PyTorch forum staff confirm 120↔121 compat; the rule's application to this pair is derived, not doc-stated].
**How to avoid:** the installer patches setup.py to add `-gencode arch=compute_121,code=sm_121` (occurrence-counted assertions) — the route a measured GB10 deployment writeup used (with `MAX_JOBS=4`, `FLASH_ATTENTION_FORCE_BUILD=TRUE`) [CITED: jishuzhan GB10 writeup, third-party MEDIUM]. Alternatively build defaults and let binary compat carry it — cheaper to script, weaker guarantee. Decide by measurement: the probe's forward IS the falsification.
**Warning signs:** build succeeds, first forward dies with `no kernel image is available for execution on the device`.

### Pitfall 4: uv rejects unknown extras after the `fla` deletion
**What goes wrong:** `uv pip install -e ".[full,fla]"` errors ("extra `fla` is not defined") the moment the extra disappears.
**Why:** three CI lines still name it [VERIFIED: ci.yml:382, ci.yml:603, ci.yml:819] plus the comment block at ci.yml:361-368.
**How to avoid:** edit all three in the SAME change as the pyproject edit (Phase 25's "CI/extra wiring lands atomically" precedent).
**Warning signs:** lane env-create step red on the first post-change run.

### Pitfall 5: census default-lane red after the flip (asymmetric seed)
**What goes wrong:** flipping the registry rows without seeding the committed artifact makes `verify_registry` compute `remote-code-broken` (skip disposition, load fail) vs committed `extra-required-tf5` → `UNEXPECTED ... committed=extra-required-tf5 fresh=remote-code-broken` [VERIFIED: census_compat.py:2257-2262].
**How to avoid:** Pattern 1's disposition seed lands in the same commit as the registry flip; run `census_compat.py --verify` dry (committed artifact) locally before committing.
**Warning signs:** local dry `--verify` not reporting 49 matched.

### Pitfall 6: docs byte-identity vs DOCS_DATA_DATE bump
**What goes wrong:** regenerating `model_compatibility.md` without bumping `DOCS_DATA_DATE` leaves the page claiming the old data date; bumping it without regenerating reds the byte-identity gate; regenerating with wall-clock would flap.
**Why:** the date is a pinned constant tied to the data change (D-15, quick-261029-c95 precedent).
**How to avoid:** one commit: writer flip + `DOCS_DATA_DATE` bump + `--emit-docs` regen; the docs-consistency pytest gate re-derives counts from the registry (`Counter` over `tf_compat_status`, currently "all six numbers" — becomes seven) [VERIFIED: tests/models/test_compat_docs_consistency.py:316-339, 216 fail-closed on unknown status].
**Warning signs:** docs gate red on `unknown tf_compat_status 'extra-required-tf5'` — every fail-closed surface must learn the value in the same change.

### Pitfall 7: deepspeed sparse kernels under triton 3.6 (unmeasured)
**What goes wrong:** `import` of `deepspeed.ops.sparse_attention` succeeds (verified: `triton._C.libtriton` imports under 3.6.0), but its `@triton.jit` kernels descend from 2020-era torch-blocksparse code; first kernel launch may fail under a modern triton compiler.
**Why:** ancient Triton API surface (`tl.load(header + 0)`-style pointer arithmetic) [VERIFIED: deepspeed v0.19.6 matmul.py source].
**How to avoid:** nothing preventive — the green probe (load + REAL forward, D-03) is the measurement. If the forward reds on kernel compile, the disposition does NOT flip (census discipline); the phase then needs a documented fallback decision from the owner.
**Warning signs:** probe log shows `triton.compiler.errors` / `CompilationError` after a successful model construction.

### Pitfall 8: runner ruff/ty drift on new code
**What goes wrong:** new `dnallm/` lines (support words, interception) trip pinned ruff 0.17.1 (repo hook runs 0.16.10 — 27.2-05 caught exactly this) or the ty hard gate.
**How to avoid:** run `ruff format` + `ruff check` with the pinned version and `ty check` on touched files before commit; every `dnallm/` change ships with pytest in the same change, per-module ≥96%.

## Code Examples

### Enum surface — every value that must learn `extra-required-tf5` (verbatim current state)

```python
# Source: dnallm/models/registry.py:133-141 (RegistryEntry, extra="forbid" at :124)
    tf_compat_status: (
        Literal[
            "verified-tf5",
            "import-verified-tf5",
            "shim-required-tf5",
            "remote-code-broken",
            "census-blocked",
        ]
        | None
    ) = None

# Source: scripts/census_compat.py (TF_COMPAT_STATUSES frozenset, post-run_census region)
# "The five-value registry verdict vocabulary (mirrors RegistryEntry's Literal)."
TF_COMPAT_STATUSES = frozenset({
    "verified-tf5",
    "import-verified-tf5",
    "shim-required-tf5",
    "remote-code-broken",
    "census-blocked",
})
```

### Support-word surface (current values, verbatim)

```python
# Source: dnallm/models/support.py:39-47, 52-56, 62-66
SUPPORT_WORDS: dict[str, str] = {
    "verified-tf5": "works",
    "import-verified-tf5": "family-tested",
    "shim-required-tf5": "works with built-in fixes",
    "remote-code-broken": "not working with transformers 5",
    "census-blocked": "untested",
}

SUPPORT_WORD_BY_REPO: dict[str, str] = {
    "songlab/gpn-brassicales": "needs pip install gpn",
    "zehui127/Omni-DNA-116M": "needs pip install ai2-olmo",
    "AIRI-Institute/gena-lm-bigbird-base-sparse": "needs deepspeed",   # → "works with the deepspeed extra" (D-14)
}
# CrossDNA has NO override today (rides the generic broken word) → ADD
#   "chengCCC/CrossDNA_pretrain": "works with the flash-attn extra"

DOCS_STATUS_CELLS: dict[str, str] = {
    "verified-tf5": "Tested working",
    "import-verified-tf5": "Tested loading its code; full run inferred from the family model",
    "shim-required-tf5": "Works with DNALLM's built-in compatibility fixes (tested)",
}
# Add an "extra-required-tf5" cell; support_word() falls back
# SUPPORT_WORDS.get(status, UNTESTED_WORD) (support.py:121) — the status key must exist
# there too (or every non-overridden entry silently reads "untested").
```

### fla 0.5.2's exact failure mode (installed source, verbatim)

```python
# Source: .venv/lib/python3.13/site-packages/fla/layers/attn.py (fla 0.5.2), module level:
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
# ... and in Attention.__init__ (the census-recorded failure, line 71):
        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
```

### GENA remote code's deepspeed consumption (cached snapshot, verbatim)

```python
# Source: ~/.cache/huggingface/modules/transformers_modules/AIRI_hyphen_Institute/
#   gena_hyphen_lm_hyphen_bigbird_hyphen_base_hyphen_sparse/3a9028d.../modeling_bert.py
# Construction time (line ~270) — the census-recorded ModuleNotFoundError site:
        sparse_config_cls_name = getattr(config, 'sparse_config_cls', None)
        if sparse_config_cls_name:
            self.is_sparse = True
            sparse_config_cls = get_cls_by_name(sparse_config_cls_name)   # importlib.import_module
# (line ~308-312) — the function-local import:
        if self.is_sparse:
            try:
                from deepspeed.ops.sparse_attention import SparseSelfAttention
            except ImportError as e:
                logger.error(f'DeepSpeed is required for Sparse Ops: {e}')
                raise
            self.sparse_self_attention = SparseSelfAttention(self.sparse_config, max_seq_length=self.max_seq_len)
# fp16 constraint is handled INSIDE the remote code (auto-casts q/k/v to half and the
# context back to float) — an fp32 census-style probe will NOT trip it.
```

### D-17 interception shape (GPN/Omni-DNA precedent, verbatim)

```python
# Source: dnallm/models/special/gpn.py:23-33 (the 27.2 D-10 shape to replicate)
            except ImportError as e:
                raise ImportError(
                    f"gpn package is required for "
                    f"{model_name} but not installed. "
                    "Install it with: pip install gpn "
                    "(measured caveats, gpn 0.9.0 wheel: requires Python 3.13 "
                    "through 3.13.x only, pins transformers==5.15.0, and does not "
                    "provide the gpn.model module this handler imports - use a "
                    "separate environment). See "
                    "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/"
                ) from e
# Phase-28 variants: BigBird — at the generic chain's wrap sites
# (dnallm/models/model.py:1368-1369, 1438-1439), inspect the cause chain for
# ModuleNotFoundError with name "deepspeed" and re-raise ImportError naming
#   pip install 'dnallm[deepspeed]'
# CrossDNA — inside _handle_crossdna_models around the two from_pretrained calls
# (dnallm/models/special/crossdna.py:522, 574), catch ImportError whose message
# carries "Flash Attention" (fla's verbatim raise above) and append
#   pip install --no-build-isolation 'dnallm[flash-attn]'
```

### compat_verdict seed branch (insertion point)

```python
# Source: scripts/census_compat.py:1479-1537 — current branch order:
#   census-blocked → disposition.skip → disposition.fix+green → green(+SHIM_RESTORED_LOADS)
#   → family-rep → raise ValueError
# Insert BEFORE the skip branch:
#     extra = disposition.get("extra")
#     if isinstance(extra, str) and extra:
#         return "extra-required-tf5", f"{stamp}; {extra}"
# Evidence examples (implementer discretion on exact text; D-04 requires BigBird's
# to name BOTH enablers):
#   BigBird:   "hf AIRI-Institute/gena-lm-bigbird-base-sparse@3a9028d36137341c10df771c3bd93de530d74594; "
#              "load+forward verified via the deepspeed extra (deepspeed 0.19.6) + the "
#              "_patch_apply_chunking_to_forward rung"
#   CrossDNA:  "hf chengCCC/CrossDNA_pretrain@cd118dc45eaa0f2debb05f03ed0290fe56a78ab1; "
#              "load+forward verified via the flash-attn extra (flash-attn 2.8.3.post2)"
```

### installer pinned constants (starting values; re-verify at fork time)

```bash
# flash-attn 2.8.3.post2 sdist [VERIFIED: PyPI JSON API 2026-10-29]
FLASH_ATTN_VERSION="2.8.3.post2"
FLASH_ATTN_SDIST_SHA256="d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b"
# deepspeed 0.19.6 sdist sha256 (if measurement ever forces a script route):
#   31593acffbe4794c74446094122b1bc29d8c73a137e37aff899ca4a6480df2d8  [VERIFIED: TUNA simple page]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| deepspeed ships `ops/sparse_attention` (Triton) | Module deleted upstream | v0.19.7, 2026-09-16 [VERIFIED: GitHub trees API bisect] | The extra MUST cap `<0.19.7`; pyproject comment should cite this |
| flash-attn GitHub wheels cover current stacks | Wheel matrix stopped at cu12/torch 2.8/x86_64 (2.8.3.post2, 2026-10-13); fa4 betas ship no wheels | observed 2026-10-29 | GB10 (aarch64/cu130/torch 2.11) = source-build territory |
| Census statuses: 5 values; BigBird/CrossDNA = remote-code-broken | `extra-required-tf5` (6th value), 2 rows flip | this phase | Six fail-closed surfaces must learn the value atomically |
| `fla` an optional extra (Phase 25 era) | base dependency (owner directive: lab models out-of-the-box) | this phase | 3 CI install lines + `all` slim + census lane consistency (see below) |

**Notable side effect (plan should record it):** `census_compat.yml:88` installs `.[full]` only — fla today. The live census lane (deferred post-merge) would therefore lack fla and redden on PlantHelixSeek import probes. D-09's promotion fixes this latent gap automatically (fla rides base into every `.[full]` env), and also gives the `not slow chunk_kda import guard` (ci.yml:366, WR-01) its import on every lane. [VERIFIED: census_compat.yml:82-88; ci.yml:361-368, 382, 603, 819]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | flash-attn source build on 20-core GB10 takes roughly 1–3 h with MAX_JOBS capped (~4–8) | Pitfalls/Environment | Lane timeout mis-set (not measured — execution measures; mamba precedent shows the box compiles CUDA extensions) |
| A2 | deepspeed 0.19.6 sparse-attention Triton kernels LAUNCH (not just import) under triton 3.6.0 / torch 2.11 / sm_121 | Pitfall 7 | If the forward reds, BigBird's disposition cannot flip; phase degrades to CrossDNA-only flip + owner decision |
| A3 | sm_120 SASS loads on sm_121 by the same-major rule (derived from documented rule; PyTorch staff statement supports) | Pitfall 3 | Default-arch build unusable → explicit 121 patch mandatory (already the recommended route) |
| A4 | deepspeed 0.19.6 sdist builds green under uv/pip PEP 517 (setup.py degrades without torch in build env, as 0.19.8's does — 0.19.6 assumed same shape) | Standard Stack | Lane falls back to `--no-build-isolation` pip install (build deps already present) — same posture as flash-attn |
| A5 | `pip download`-free claims: no py3.13 incompatibility in deepspeed 0.19.6 (0.19.8 declares `requires_python: None`; 0.16+ line added py3.13) | Standard Stack | Resolver picks nothing on py3.13 → widen floor down (e.g. `>=0.16.5,<0.19.7`) after checking 0.19.6 metadata at execution |
| A6 | The `run_extras_compat` boolean wires into the extras workflow's dispatch gate the same way the three existing booleans gate ci.yml lanes (CONTEXT leaves the exact wiring to the planner) | Patterns | Mechanical only — both precedent shapes extracted below |

## Open Questions

1. **Extras-lane cron residency** — D-06 says "nightly cron" but cron schedules fire only from the default branch, and D-05/deferred say live activation is post-merge. Does `extras_compat.yml` carry its `schedule:` entry now (inert until merge, like census lane's dispatch-only posture) or start dispatch-only and gain cron at merge? Recommendation: mirror the ci.yml pattern (cron + cron-string gate + dispatch∧boolean), accepting inertness until merge — matches D-06's wording exactly.
2. **BigBird probe dtype** — the census load program pins `torch_dtype=torch.float32`; the remote code auto-handles the sparse fp16 cast, so no conflict is expected, but the extras-env probe is the first-ever GREEN BigBird forward — treat any dtype surprise as new evidence (census discipline), not a probe bug.
3. **Does `install_flash_attn.sh` also own the deepspeed install?** — D-12 scopes the script to flash-attn; discretion defaults deepspeed to the plain extra. Recommendation: keep the script single-purpose (mirror of install_mamba.sh's single responsibility); deepspeed needs no patch pipeline unless A4/A5 falsify.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| GPU (GB10, sm_121) | probes + builds | ✓ | NVIDIA GB10, CC (12,1), 1 device [VERIFIED: local torch probe] | — |
| nvcc / CUDA toolkit | flash-attn build | ✓ | CUDA 13.0 at `/usr/local/cuda/bin/nvcc`; torch resolves `CUDA_HOME=/usr/local/cuda` [VERIFIED: local] | — |
| ninja | builds | ✓ | 1.13.2 (venv + base dep `ninja>1`) | — |
| triton | deepspeed sparse kernels | ✓ | 3.6.0 (torch dep) | — |
| torch (cu130, aarch64) | everything | ✓ | 2.11.0+cu130 [VERIFIED: venv] | — |
| transformers | probes | ✓ | 5.17.0 [VERIFIED: venv] | — |
| fla 0.5.2 | CrossDNA/PlantHelixSeek | ✓ installed (dev venv) | flash-linear-attention 0.5.2 + fla-core 0.5.2 | — |
| deepspeed | BigBird probe | ✗ not installed | — | install band `>=0.19.6,<0.19.7` at execution (TUNA sdist verified available) |
| flash-attn | CrossDNA probe | ✗ not installed | — | pinned-sdist build at execution (per D-12; build NOT run during research, per scope) |
| uv + TUNA index | lane installs | ✓ | uv 0.13.6, `~/.config/uv/uv.toml` defaults to `https://pypi.tuna.tsinghua.edu.cn/simple` [VERIFIED: local file] | — |
| TUNA: flash_attn sdists | installer fetch | ✓ | 2.8.2 / 2.8.3 / 2.8.3.post1 / 2.8.3.post2 listed [VERIFIED: TUNA simple page] | PyPI route |
| TUNA: deepspeed sdists | extra install | ✓ | 0.19.6 / 0.19.7 / 0.19.8 listed, sha256 matches PyPI for 0.19.8 [VERIFIED] | PyPI route |
| RAM / cores | compile budget | ✓ | 121 GB total, 20 cores [VERIFIED: free/nproc] | MAX_JOBS cap for headroom |

**Missing dependencies with no fallback:** none blocking — deepspeed/flash-attn installs ARE the execution phase.
**Missing dependencies with fallback:** none.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface added |
| V3 Session Management | no | — |
| V4 Access Control | yes (CI boundary) | Fork-PR safety: extras lane is dispatch/schedule-only on `[self-hosted, dnallm-nightly]`, `permissions: contents: read`, never PR-reachable (census_compat.yml posture, D-05) |
| V5 Input Validation | yes (artifact integrity) | `compat_census.json` records validated by `read_artifact`; `compat_verdict` fails closed on unclassifiable shapes (`ValueError`); `_docs_cell` fails closed on table-breaking characters |
| V6 Cryptography | yes (byte integrity) | sha256 pin gate BEFORE any sdist byte is extracted or executed — no override flag, no env bypass (install_mamba.sh discipline; fork it verbatim) |
| V14 Config | yes | `permissions: contents: read` least-privilege on the new workflow; no repo-default token inheritance |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious/mirrored sdist substitution (supply chain) | Tampering | Pinned sha256 + index-declared-hash precheck before download + local hash gate before extraction; any index route then fails closed on altered bytes |
| PR-authored code reaching the self-hosted GPU box | Elevation | Dispatch/schedule triggers only; `if: github.event_name == 'workflow_dispatch'` style gates; workflow registration is default-branch-only |
| Remote model-code execution at probe time (trust_remote_code) | Tampering/Info disclosure | Pre-existing accepted posture (census lane already runs these two repos); the extras lane runs the SAME repos at pinned revisions already stamped in the artifact |
| Unverified bytes executed via pip sdist metadata resolution | Tampering | Installer fetches via PEP 503 simple page + stdlib urllib, never pip, until after the hash gate (install_mamba.sh:164-204 pattern) |

## Sources

### Primary (HIGH confidence)
- Local measurement (this box = runner-class GB10): torch 2.11.0+cu130, transformers 5.17.0, triton 3.6.0 + `triton._C.libtriton` import OK, fla 0.5.2 installed, deepspeed/flash-attn absent, GPU sm_121, 20 cores/121 GB, nvcc CUDA 13.0, uv 0.13.6 + TUNA config
- PyPI JSON API (2026-10-29): flash-attn file/sha/requires; deepspeed 0.19.8 file/sha/requires_dist
- GitHub API (2026-10-29): flash-attention v2.8.3.post2 asset matrix (48 x86_64/cu12 wheels); DeepSpeed tree bisect v0.18.5→v0.19.8 locating sparse_attention removal at v0.19.7; deepspeed v0.19.6 sparse_attention file list + matmul.py/__init__.py source
- flash-attention v2.8.3.post2 setup.py (raw source): `FLASH_ATTN_CUDA_ARCHS` default + gencode branch table
- Repo sources read this session: pyproject.toml, scripts/install_mamba.sh, scripts/census_compat.py (CLI/merge/verdict/verify/docs regions), tests/models/compat_census.json (both records), .github/workflows/ci.yml + census_compat.yml, dnallm/models/registry.py, support.py, model.py (wrap sites), special/{crossdna,gpn,omnidna}.py, docs/user_guide/optional_dependencies.md, cached GENA modeling_bert.py + CrossDNA modeling_crossdna.py, installed fla layers/attn.py, tests/models/test_{support,model_info_registry,census_compat,compat_docs_consistency}.py (pin regions)

### Secondary (MEDIUM confidence)
- [NVIDIA Hopper Compatibility Guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html.md) — same-major/minor-≥ cubin rule (120→121 derived application)
- [PyTorch forum: DGX Spark GB10 CUDA 13.0 sm_121](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744/5) — staff confirm sm_121 ↔ sm_120 binary compat
- [GB10 flash-attn build writeup (jishuzhan)](https://jishuzhan.net/article/2032483049980624898) — explicit sm_121 gencode patch + MAX_JOBS=4 recipe (third-party)
- [jetson-containers flash-attention recipes](https://github.com/dusty-nv/jetson-containers/tree/2cdc926d091a6e093c336c5b68e0704ce02d78ec/packages/llm/flash-attention) — Jetson build precedent
- [Blackwell compute-capability overview (Qualiteg)](https://journal.qualiteg.com/cuda-compute-capability-numbering/) — GB10 = 12.1, RTX 50 = 12.0 mapping
- gsd-tools package-legitimacy gate output (SUS verdicts + mitigations, § Package Legitimacy Audit)

## Metadata

**Confidence breakdown:**
- Extra version bands / install mechanics: HIGH — registry + upstream source verified; only kernel-launch behavior (A2) unmeasured by design
- Census/seed mechanics: HIGH — merge/verdict/verify code read line-by-line; both artifact records inspected; symmetric design derived from read rules
- CI lane wiring: HIGH — both precedent workflows extracted verbatim
- flash-attn build outcome on sm_121: MEDIUM — flags/paths verified, compile+load unmeasured (execution phase, per scope)

**Research date:** 2026-10-29
**Valid until:** 2026-11-29 (stable: upstream removals/wheel matrices move slowly; re-check the deepspeed cap and flash-attn release page if execution slips past a month)
