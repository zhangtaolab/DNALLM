# Phase 18: Packaging & Install Honesty — Atomic Extras Surgery - Research

**Researched:** 2026-10-14
**Domain:** Python packaging (pyproject extras / PEP 621 + uv index machinery), installer-behavior verification (pip/uv), CI workflow surgery, install-lane documentation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (owner-accepted 2026-10-14)

- base → **`full`** (dev/test/notebook/mcp superset; self-referential `all` chain intact; one-cycle deprecated alias so pip's warn-and-exit-0 cannot silently under-install; grep gate holds stale `.[base]` residue to a sanctioned list)
- rocm: **REMOVED with documented restore path** (no rocm wheel exists for torch>=2.10 — keeping it would be a false promise)
- WPAR-07 triton-windows: **DECLARED** behind `platform_system == 'Windows'` marker in the fla extra (`>=3.7.1.post27,<3.8`, research-recommended); fla KDA smoke re-verified on Windows under the resolved torch
- Validation depth: fresh-venv resolve per changed extra + smoke dispatch against the post-surgery tree (each extras change validated)

Carried constraints (roadmap):

- cuda121 retired same atomic discipline (extra + uv index/sources/conflicts + docs in one change, index frozen at torch 2.5.1)
- 5 CI/publish call sites + extras guard tests + docs in the same atomic change as the rename
- Install docs: explicit uv vs pip lanes stating the [tool.uv.sources] asymmetry; GPU matrix from pre-verified boundaries (Pascal cpu-only + min-GPU line, 20xx→cu128, 30/40xx all extras, 50xx cu128+); cross-installer honesty matrix page live

### Claude's Discretion

Alias mechanics, grep-gate implementation, docs structure within the locked contracts.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PKGH-01 | `base` rename/re-scope with one-cycle deprecated alias; atomic across pyproject (`all` chain), 5 CI/publish call sites, extras guard tests, docs; grep gate on `.[base]` residue | Complete rename map (§ Rename Map) — 11 install lines across 3 workflow files, not 5; live-probed alias mechanism (§ Deprecated-Alias Mechanism); guard-test extension pattern (§ Guard Tests) |
| PKGH-02 | cuda121 retired — extra + uv index/sources/conflicts + docs in one atomic change (index frozen at torch 2.5.1) | Exact removal list with line numbers (§ cuda121 Retirement); cu121 freeze live-reverified today |
| PKGH-03 | rocm disposition executed on verified facts (remove + restore path documented) | **Facts have MOVED since 2026-10-11** — rocm7.1 now ships torch 2.11.0 (§ rocm Disposition); removal still executes per owner lock; restore-path content updated to rocm7.1 |
| PKGH-04 | Install docs split uv/pip lanes stating the `[tool.uv.sources]` asymmetry | Existing partial pip lane found in installation.md:121-165 (§ Install-Lane Docs); uv docs verbatim quote on uv-only sources (in-repo STACK.md, cited) |
| PKGH-06 | GPU support matrix published from pre-verified boundaries | Verbatim pre-verified matrix recovered from git 0542a6b (§ GPU Support Matrix) — content ready to paste |
| PKGH-07 | Cross-installer honesty matrix page (command × installer → what you get) | Live-probe matrix supplies exact warn/exit semantics per installer (§ Deprecated-Alias Mechanism + § Honesty Matrix) |
| WPAR-07 | triton-windows declared behind Windows marker in fla extra (`>=3.7.1.post27,<3.8`); fla KDA smoke re-verified on Windows | Exact marker syntax + registry verification + legitimacy audit (§ triton-windows); KDA re-verification shape from WINDOWS #18 precedent |
</phase_requirements>

## Summary

The surgery surface is fully enumerable and small: `pyproject.toml` (extras block, 7 `[[tool.uv.index]]` entries, `[tool.uv.sources]`, `[tool.uv] conflicts`), 11 `uv pip install` lines across `ci.yml`/`publish.yml`/`feasibility.yml` (the "5 CI/publish call sites" in PKGH-01 is an **undercount** — it names only the bare `.[base]` lines; the six compound `.[base,...]` lines carry the same rename), 2 guard-test files, ~69 `.[base` doc/example/README occurrences, and the cuda121/rocm/triton-windows rows in every extras table. The highest-risk design decision — the deprecated-alias mechanism — was settled by **live probe** (synthetic wheel, pip 26.2.1 + uv 0.13.1, clean venvs): the self-referential alias `base = ["dnallm[full]"]` resolves the full set and exits 0 under both installers, from a local wheel path (the smoke's exact scenario). Critical honest finding: **no installer emits any warning for an extra that exists** — the "warns-and-exit-0" behavior belongs to the post-alias state (pip: `WARNING: dnallm X.Y.Z does not provide the extra 'base'`, exit 0; uv: `warning: The package ... does not have an extra named 'base'`, exit 0). During the alias cycle the loud surface must be docs/README/release-notes, because there is no native extra-deprecation metadata channel in pip/uv/setuptools (PEP 685 only normalizes names).

One factual correction to the locked record: the 2026-10-11 claim "no rocm wheel exists for torch>=2.10" is **stale as of today** — `download.pytorch.org/whl/rocm7.1/` now lists `torch-2.10.0+rocm7.1` and `torch-2.11.0+rocm7.1` (cp310–cp314, manylinux x86_64, Linux-only). The owner's REMOVE decision still stands on its other legs (the lane is inert in-repo today — sources/conflicts commented out; never execution-tested on AMD; no AMD card in the verified matrix), but PKGH-03's rationale wording and the documented restore path must carry today's facts (restore path: manual `--index-url https://download.pytorch.org/whl/rocm7.1`, torch 2.10/2.11, resolution-tested-only). Everything else verified live and matches the pre-verified records: cu121 index frozen at torch 2.5.1 (re-fetched today); triton-windows 3.7.1.post27 on PyPI with win_amd64 wheels cp310–314 under the triton-lang org.

**Primary recommendation:** one atomic commit performs the whole surgery (pyproject extras + uv machinery + workflows + guard tests + docs rename sweep), validated same-change by fresh-venv per-extra resolves from the built wheel (concrete wheel path + bracket extras — the glob form fails) and a publish-smoke dispatch against the surgery branch; the grep gate is a phase-13-style verbatim sanctioned-residue record enforced by a repo-scan test.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Extras declaration + alias + markers | Packaging metadata (pyproject.toml) | — | PEP 621 `optional-dependencies` is the only surface installers read; no code runtime exists for extras |
| uv index/sources/conflicts cleanup | Packaging metadata (`[tool.uv]` tables) | — | uv-only resolution config, never published in wheel metadata |
| CI/publish install-line rename | CI workflows (ci.yml, publish.yml, feasibility.yml) | — | The rename's mechanical consumers; must never drift from pyproject (atomicity contract) |
| Residue enforcement (grep gate) | Test suite (pytest repo-scan) | Plan/SUMMARY verbatim record | Phase-13 precedent: residue equals an exact recorded list; pytest keeps it enforced post-phase |
| Extras guard tests | Test suite (tests/test_extras_guard.py, tests/models/test_plant_helixseek_fla_kernels.py) | — | tomllib-based bracket-member assertions; the existing extendable pattern |
| Install-lane + GPU matrix + honesty matrix docs | docs/ (installation.md primary) | README, docs/index.md | PKGH-04/06/07 deliverable surface; lane split already partially present |
| Fresh-venv per-extra validation | Local execution (GB10 box) | Publish smoke dispatch (CI) | CONTEXT-locked validation depth: local resolves prove extras; smoke proves the built artifact end-to-end |
| Windows fla KDA re-verification | Owner's Windows box (manual) | WINDOWS ledger record | WPAR-07 acceptance lives on real Windows hardware (RTX 5080 precedent) |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| triton-windows (NEW, marker-gated) | `>=3.7.1.post27,<3.8; platform_system == 'Windows'` | Make `.[fla]` functional on Windows (fla-core's `fla.ops.*` needs triton; torch Windows wheels declare no triton dep) | Only dependency this phase adds. [VERIFIED: pypi.org JSON fetched 2026-10-14] — latest 3.8.0.post29, `requires_python: <3.15,>=3.10`, homepage `https://github.com/triton-lang/triton-windows` (fork lives under the triton-lang org), 3.7.1.post27 ships 5 win_amd64 wheels cp310–cp314. Version history 3.5.1.post24→3.8.0.post29 matches the v1.3 STACK record. `<3.8` cap: fork's guarantee table pairs 3.7 with torch 2.12/2.13, 3.8 with torch 2.14/2.15 — both above the repo's `torch<2.12`; the exact 3.7.1.post27 is the RTX 5080 sm_120 empirical proof (WINDOWS #18) |

Everything else this phase touches is already declared (flash-linear-attention, torch, uv, twine) — the surgery is configuration, deletion, rename, and one marker-gated declaration.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Self-referential alias `base = ["dnallm[full]"]` | Empty extra `base = []` | Rejected by live probe: empty extra is **completely silent** under pip AND uv (exit 0, no warning, installs nothing) — the pure silent-under-install hazard [VERIFIED: live probe 2026-10-14] |
| Self-referential alias | Hard removal (no alias) | pip warns `WARNING: dnallm X.Y.Z does not provide the extra 'base'` + exit 0, uv `warning: ... does not have an extra named 'base'` + exit 0 — a single scroll-past line; under-installs silently. Alias over-installs (safe direction) for one cycle instead |
| Declaring triton-windows | Document-only manual step | STACK-research rejected alternative: keeps `.[fla]`-on-Windows succeeding-then-breaking-at-import — the exact dishonesty class this milestone removes |

**Installation (the single pyproject dependency delta):**

```toml
fla = [
    "flash-linear-attention>=0.5.2,<0.6",
    "triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'",
]
```

Marker style matches the existing in-repo vocabulary verbatim (`pybedtools>=0.11.0; platform_system != 'Windows'`, `pygenometracks>=3.9; platform_system != 'Windows'` — pyproject.toml:89,118) [VERIFIED: pyproject.toml read this session].

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| triton-windows | PyPI | ~1.5 yr line (3.5.1.post24 era → 3.8.0.post29) | not fetched (seam returned null) | github.com/triton-lang/triton-windows | SUS* | Flagged — planner gates the Windows KDA re-verification as the human-verify checkpoint (it already IS a success criterion) |
| flash-linear-attention | PyPI | years (latest rel 2026-07-27) | not fetched | github.com/fla-org/flash-linear-attention | SUS* | Approved — already a shipped dependency of this repo (fla extra); no new install decision this phase |

\* **Seam-verdict caveat (report honestly):** `gsd-tools query package-legitimacy check` returned SUS for both with **all-null signals** (`exists: null, publishedAt: null, repoUrl: null` — its metadata fetch failed/timed out), i.e. "no observation", not negative evidence. Direct registry reads (pypi.org JSON, fetched this session) confirm existence, repo URLs, wheel inventories, and version history; triton-windows additionally carries the in-repo WINDOWS #18 empirical execution proof (KDA chunk kernel compiled + executed on RTX 5080 sm_120 at exactly 3.7.1.post27) and the triton-lang org residency + attested-builds-since-post27 record from STACK.md [CITED: .planning/research/STACK.md:32-34]. Per protocol the SUS flag stays visible; the WPAR-07 Windows execution re-verification is the natural checkpoint that retires it.

Python wheels run no postinstall scripts (no npm-style script surface), so the postinstall-script check does not apply.

**Packages removed by the surgery:** cuda121/rocm are extra *names* removed from this repo's metadata — not registry packages; no external package is un-declared except the `torch` index pins behind those extras.

## Current Anatomy (verbatim, pre-surgery)

All line numbers from `pyproject.toml` read this session (head d3d57eb).

### The extras block [VERIFIED: pyproject.toml:148-187]

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

`base` today = the dev+test+notebook+mcp superset plus isort + types-transformers — the everything-for-development aggregate mislabeled as a minimal install. The self-referential chain shape (`dev = ["dnallm[test,notebook]"]` at :83, `base`, `all`) is the exact resolver pattern the deprecated alias reuses — proven daily by every CI `.[base]`/`.[base,fla]` install.

### The cuda121 footprint [VERIFIED: pyproject.toml:159-161, 205-208, 238, 248, 254-257]

- :159-161 — `cuda121 = ["torch>=2.2.0,<2.12"]`
- :205-208 — `[[tool.uv.index]] name = "torch-cuda121"`, `url = "https://download.pytorch.org/whl/cu121"`, `explicit = true`
- :238 — `{ index = "torch-cuda121", extra = "cuda121" }` in `[tool.uv.sources]`
- Conflicts rows (5 total, exactly as STACK.md counted): :248 `[ { extra = "cpu" }, { extra = "cuda121" } ]`; :254-257 cuda121×{cuda124, cuda126, cuda128, cuda130}
- ci.yml:222 — matrix `cuda-version: ['12.1', '12.4']`; ci.yml:263-267 — the if/elif install block (`".[base,cuda121]"` / `".[base,cuda124]"`)
- Docs: installation.md:117-118, :218 (table row "cuda121 | 2.2.0-2.12"); quick_start.md:101-102, :329; README:170, :178, :199 (table row); docs/index.md:79

### The rocm footprint [VERIFIED: pyproject.toml:174-176, 230-233, 243, 253, 264-266]

- :174-176 — `rocm = ["torch>=2.5.0,<2.12"]` (extra still declared)
- :230-233 — `[[tool.uv.index]] name = "torch-rocm"`, `url = "https://download.pytorch.org/whl/rocm6.2"`, `explicit = true` (index still ACTIVE)
- :243 — commented source line `#   { index = "torch-rocm", extra = "rocm" },`
- :253, :264-266 — four commented conflicts rows
- Net effect today: `.[rocm]` resolves torch from the default PyPI index — the lane is inert; README's ROCm claim (:76, :204) promises something no installer delivers
- Docs: README:76, :178, :204; docs/index.md:49, :79; installation.md:213, :223; quick_start.md:329
- **Do NOT sweep:** README:254 and docs/faq/models_troubleshooting.md:13 mention fla's *own upstream* `[cuda]`/`[rocm]` extras ("install the package bare — since v0.5 its `[cuda]`/`[rocm]` extras pin their own torch") — those are flash-linear-attention extras, not dnallm's rocm extra; they stay

### The smoke (phase-17 asset) [VERIFIED: .github/workflows/publish.yml:81-127 read this session]

`smoke` job: `if: github.event_name == 'workflow_dispatch' || (schedule && cron '0 8 * * 0')` → checkout → setup-uv → `uv build --no-sources` → `twine check dist/*` → `uv venv /tmp/smoke-venv` + `uv pip install --python /tmp/smoke-venv/bin/python dist/*.whl` → import probe. The `publish` job is release-gated (`if: github.event_name == 'release'`), so a dispatch can never upload. Dispatch proof pattern from 17-01: `gh api runs/<id>/jobs` → assert publish `skipped` AND smoke `success` (run 38554913144 precedent).

## Deprecated-Alias Mechanism (live-probed)

**Method:** synthetic package `probeextras` 0.1.0 with extras `full = ["six"]`, `base = ["probeextras[full]"]`, `empty = []`; built with `uv build --no-sources`; installed into clean seeded venvs under pip 26.2.1 (py3.12) and uv 0.13.1 (py3.12), local wheel path.

| Case | pip 26.2.1 | uv 0.13.1 |
|---|---|---|
| Alias `wheel[base]` → `pkg[full]`, clean venv, first install | **Installs the full set (six) + pkg, exit 0, NO warning.** Self-reference matched to the local wheel by name | **`Installed 2 packages ... + six==1.17.0`, exit 0, NO warning** |
| Missing extra `wheel[nonexistent]`, clean venv | `WARNING: probeextras 0.1.0 does not provide the extra 'nonexistent'` — printed after "Successfully installed", **exit 0** | `warning: The package 'probeextras @ ...' does not have an extra named 'nonexistent'` — printed after install, **exit 0** |
| Empty extra `wheel[empty]`, clean venv | Silent success, exit 0, installs nothing extra | Silent success, exit 0, installs nothing extra |

All six cells [VERIFIED: live probe 2026-10-14, output pasted above]. pip's warn-and-exit-0 on missing extras is stable across pip 23.2.1→26.x (community multi-version test evidence: [lint_extras.py](https://github.com/nousergon/crucible-predictor/blob/main/scripts/lint_extras.py)) [CITED].

**Consequences for the design:**

1. **The alias works and is the safe shape.** `base = ["dnallm[full]"]` installs the exact full set under both installers, from local wheel paths (the smoke scenario) — and by the same name-match mechanism from PyPI (the in-repo `all`/`dev` self-references prove the pattern end-to-end in every CI leg today). Over-install (safe direction), never under-install.
2. **No installer warns on an alias.** There is no PEP/PyPA extra-deprecation metadata (PEP 685 only normalizes names; `base` and `full` are single lowercase words — zero normalization hazard). The CONTEXT's "must WARN and exit 0" behavior is what pip/uv natively do **once the alias is dropped** — which is exactly why the alias must exist for a cycle: it converts that warn-and-exit-0 (which silently under-installs) into a correct install while docs carry the deprecation loudly. The plan's acceptance criteria should phrase it this way; a test asserting "pip warns on the alias" would be unsatisfiable.
3. **The loud surface during the cycle is docs** (README deprecation note + installation.md + release notes). The grep gate enforces that the alias + its note are the ONLY sanctioned `base` residue in-repo.
4. Version-consistency: `pip install 'dnallm[base]<1.3'` pins an old version whose own `base` is the old kitchen sink — self-consistent by construction; no migration needed.

### Recommended alias shape (Claude's discretion area — recommended, not locked)

```toml
# pyproject.toml [project.optional-dependencies]
full = [
    "dnallm[dev,test,notebook,mcp]",
    "isort>=6.0.1",
    "types-transformers>=0.1.0",
]
# Deprecated one-cycle alias (owner 2026-10-14): base renamed to full; remove
# this alias in the next minor release -- pip then warns "does not provide the
# extra 'base'" and exits 0, by design.
base = [
    "dnallm[full]",
]
all = [
    "dnallm[full,dev,test,notebook,docs,ui,mcp,fla]",
]
```

`full` carries `base`'s exact former members (owner: "dev/test/notebook/mcp superset" — same content, honest name); `all` swaps `base`→`full` in the member list (its `dev,test,notebook,mcp` members become redundant-but-harmless — keep to minimize diff, or drop to the non-redundant set; either resolves identically since extras are set-union).

## Rename Map: base → full (complete)

### Workflow install lines — 11, not 5 [VERIFIED: grep this session]

The PKGH-01 "5 CI/publish call sites" originates from STACK.md:42's list of the **bare** `.[base]` installs. The full rename surface:

| # | File:line (today's HEAD) | Line content (verbatim) |
|---|---|---|
| 1 | ci.yml:84 | `uv pip install -e ".[base]"` (test job) |
| 2 | ci.yml:169 | `uv pip install -e ".[base]"` (test-windows) |
| 3 | ci.yml:264 | `uv pip install -e ".[base,cuda121]"` (test-cuda 12.1 leg — dies with cuda121) |
| 4 | ci.yml:266 | `uv pip install -e ".[base,cuda124]"` (test-cuda 12.4 leg) |
| 5 | ci.yml:343 | `uv pip install -e ".[base,fla]"` (test-mamba) |
| 6 | ci.yml:410 | `uv pip install -e ".[base]"` (coverage-gate) |
| 7 | ci.yml:537 | `uv pip install -e ".[base,fla]"` (coverage-nightly) |
| 8 | ci.yml:744 | `uv pip install -e ".[base,fla,dev,mcp]"` (example-nightly) |
| 9 | ci.yml:1127 | `uv pip install -e ".[base,docs]"` (docs-validation lane in ci.yml) |
| 10 | publish.yml:59 | `uv pip install --system -e '.[base]'` (publish job deps) |
| 11 | feasibility.yml:67 | `uv pip install -e ".[base]"` |

Prose/comment references in workflows (same sweep): ci.yml:290, :326, :471 comments; `.github/workflows/README.md`:81, :106, :155. NOTE: STACK.md recorded the publish.yml site as :39 — **line numbers drift; the plan must instruct a fresh grep at execution time**, not trust any recorded line number (including this table's).

### pyproject self-references [VERIFIED: pyproject.toml:115-118, :154]

- :154 — the `all` member list `dnallm[base,dev,test,notebook,docs,ui,mcp,fla]`
- :117 — comment inside the notebook extra: "dies with AttributeError ... on the test-windows .[base] install" (comment mentions `.[base]` — rewrite to `.[full]` so the residue gate stays clean)

### Guard tests (the extend surface) [VERIFIED: tests read this session]

- `tests/test_extras_guard.py` — mcp/notebook member guards (unchanged by rename); ADD full-extra member guards + alias-shape guard (see § Guard Tests)
- `tests/models/test_plant_helixseek_fla_kernels.py:51` — hardcoded expected member list `"dnallm[base,dev,test,notebook,docs,ui,mcp,fla]"` (the `all` chain verbatim) and :60-62 example strings `dnallm[base,flash-attn]` — flip `base`→`full` in the expected literals
- `tests/test_runner_infra_contracts.py`, `tests/test_models_lock_contracts.py` — grep-verified: no `base` references

### Docs/examples sweep [VERIFIED: grep counts this session]

49 `.[base` occurrences across docs/+example/+README: installation.md ×11, README ×3 (incl. `.[base,mamba]` at :210), example/notebooks/overview.md ×2 + its docs mirror ×2, quick_start.md ×2, end_to_end_workflow/getting_started/index/faq ×1 each, ~19 notebook mirror files ×1 each. 20 more matches across .github/ + pyproject (install lines, comments, workflows README). Every example `.md` edit must land with its `docs/` mirror byte-identical (`check_docs_sync` green — WPAR-08 invariant). No `dnallm[base` published-form usage exists in docs.

## cuda121 Retirement

Removal list (all in the one atomic change) — see § Current Anatomy for exact lines. Live re-verification today: `https://download.pytorch.org/whl/cu121/torch/` tops at `torch-2.5.1` (cp310–cp313; win_amd64 wheels exist for cp310-312 only) [VERIFIED: index fetched 2026-10-14]. The extra's `torch>=2.2.0,<2.12` spec can never resolve past 2.5.1 on its own index — a dead lane under the `torch>=2.4.0,<2.12` core floor.

Post-retirement shape: test-cuda matrix becomes `cuda-version: ['12.4']` (or re-pointed per owner taste — the matrix leg removal halves that job), the if/elif block collapses to a single `uv pip install -e ".[full,cuda124]"`; conflicts shrink to the pairwise set among {cpu, cuda124, cuda126, cuda128, cuda130} (6 rows); sources drop the cuda121 line; the `torch-cuda121` index entry goes.

## rocm Disposition — FACTS CORRECTION

**The locked rationale is stale; the decision stands.**

- 2026-10-11 record (STACK.md:44): "rocm6.3 AND rocm6.4 both top out at torch 2.9.1 ... ROCm 7.x pairs with torch ≥2.13, above the <2.12 cap" and REQUIREMENTS PKGH-03: "no rocm wheel exists for torch>=2.10".
- **Live re-fetch 2026-10-14** [VERIFIED: download.pytorch.org indexes fetched today]:

| Index | torch versions (top) | Platforms |
|---|---|---|
| rocm6.3 / rocm6.4 | ≤ 2.9.1 (confirms the 10-11 record) | Linux manylinux x86_64 only |
| rocm6.2.4 | ≤ 2.7.1 | Linux |
| rocm7.0 | ≤ 2.10.0 | Linux manylinux_2_28 x86_64 |
| **rocm7.1** | **2.10.0, 2.11.0** (cp310–cp314 + cp313t/cp314t) | Linux only, no Windows |
| rocm7.2 / rocm7.14 | 2.12.1+ / 2.14.x | Linux |

`torch-2.11.0+rocm7.1` wheels EXIST — a rocm lane within the `torch<2.12` cap is now resolvable, contradicting "no rocm wheel exists for torch>=2.10". The REMOVE decision remains correct on its other legs (owner-locked): the lane is inert in-repo TODAY (sources/conflicts commented out — `.[rocm]` installs default PyPI torch), it has never been execution-tested on AMD hardware, and no AMD card exists in the verified matrix. **The plan must (a) correct the rationale wording wherever it repeats the stale claim, and (b) write the restore path against today's facts.**

**Restore-path documentation content (recommended):**

> ROCm is not offered as an extra: the lane was never execution-tested on AMD hardware. AMD MI-series users can manually install torch from AMD's wheel lane — modern: `uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.1` (torch 2.10/2.11, Linux x86_64 only, resolution-tested-only); older series: `rocm6.4` tops out at torch 2.9.1. ROCm 7.2+ requires torch ≥2.12, above dnallm's cap. This is a manual lane — dnallm makes no AMD claim.

(Supersedes STACK.md's rocm6.4 restore suggestion for the modern lane; 6.4 mention optional.)

## triton-windows Declaration (WPAR-07)

**Exact spec (append to the fla extra):** `"triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'"`

- Registry [VERIFIED: pypi.org JSON 2026-10-14]: 3.7.1.post27 present with 5 win_amd64 wheels cp310–cp314 (covers the whole CI matrix 3.11–3.13 and the Windows box); latest is 3.8.0.post29 — excluded by the `<3.8` cap exactly as designed; `requires_python >=3.10,<3.15` compatible with the repo floor `>=3.11`
- Provenance note for the docs: upstream triton ships no win_amd64 wheels; the fork moved to the triton-lang org (github.com/triton-lang/triton-windows), maintainers woct0rdho + jammm, wheels reproducible and attested since post27 [CITED: STACK.md:32, fetched 2026-10-11]; homepage confirmed on PyPI today [VERIFIED]
- On Linux the marker evaluates false — the declaration changes nothing on any Linux leg (fresh-venv fla resolve proves the marker rides the wheel metadata without pulling the package)
- **Windows fla KDA smoke re-verification (WPAR-07 acceptance):** on the owner's Windows box under the resolved torch — (1) `pip show triton-windows` shows 3.7.1.post27 pulled by the marker; (2) `from fla.ops.kda.chunk import chunk_kda` resolves (the repo test `test_chunk_kda_importable_when_fla_installed`, tests/models/test_plant_helixseek_fla_kernels.py:89, gates import); (3) compile + execute a KDA chunk kernel — the #18 precedent proof was manual execution on RTX 5080 sm_120. WINDOWS #18 closes here (ledger row: quick-261008-env)

## GPU Support Matrix (pre-verified content, verbatim from git)

The verified matrix was recorded at v1.2 close (commit 0542a6b, STATE.md deferred row, recovered from git history this session) [VERIFIED: `git show 0542a6b:.planning/STATE.md`]:

> "FINAL FLOOR (all boundaries verified 2026-10-11): cuda130 default = Ampere (30xx) and newer; 20xx Turing served by cuda121–cuda128 (sm_75 IS in cu128 — torch 2.11 dropped only Volta sm_70; source pytorch#172352) but NOT cu130 (local verify: arch list sm_80+, no PTX); 30xx/40xx run every extra (cu130 via sm_80 same-major binary compat); 50xx needs cuda128+; Pascal (10xx) is NOT GPU-supported anywhere in the extras — document explicitly as cpu-only with the minimum-GPU-line stated (owner decision 2026-10-11: no 1080 support, follow upstream envelope); ... forward note: cu128 wheels deprecated from torch 2.12, so the 20xx ceiling will freeze gradually; optionally add a driver-based chooser line (cuda124–cuda130)"

Local verification basis (same record): "Local torch: 2.11.0+cu130, arch list [sm_80, sm_90, sm_100, sm_110, sm_120]".

**Phase-18 adjustments when publishing:** the 20xx row narrows to **cuda124–cuda128** once cuda121 retires (recommended lane cu128 per PKGH-06's phrasing); state the cuda130 default = Ampere+ (30xx/40xx/50xx classes: 30/40xx all extras, 50xx cu128+); Pascal cpu-only with the minimum-GPU-line (Turing) explicit. Optional driver-based chooser line (cuda124–cuda130) per the record.

## Cross-Installer Honesty Matrix (PKGH-07 content skeleton)

Grounded in this session's live probes + the uv-docs asymmetry fact. Command × installer → what you actually get:

| Command | uv (`uv pip install`, editable or wheel) | plain pip |
|---|---|---|
| `dnallm[full]` | full dev aggregate (dev/test/notebook/mcp + isort + stubs) | identical set — no index asymmetry involved |
| `dnallm[base]` (during alias cycle) | resolves alias → full set, exit 0, **no warning** (probe-verified) | identical, exit 0, no warning |
| `dnallm[base]` (after alias removal) | `warning: The package ... does not have an extra named 'base'`, installs core only, **exit 0** | `WARNING: dnallm X.Y.Z does not provide the extra 'base'`, installs core only, **exit 0** |
| `dnallm[cuda130]` | torch from the pinned `torch-cuda130` index (`[tool.uv.sources]` respected) | **default PyPI torch** — Linux: cu130-bundled default; Windows: CPU-only build. The index pin is uv-only ("Sources are only respected by uv" — uv docs, cited via STACK.md:46) |
| `dnallm[fla]` on Windows | pulls triton-windows via the marker (post-surgery) | identical — the marker is standard PEP 508 metadata, honored by both |

Plus the pip-lane rule already in installation.md:139-165 ("Install torch LAST" + verify `torch.version.cuda` suffix) — the honesty page composes with, not replaces, that guidance.

## Install-Lane Docs (PKGH-04)

installation.md ALREADY has a partial pip lane ("GPU Support with plain pip", :121-137 + torch-LAST rule :139-165 + Windows cu130 section :167-190) — the PKGH-04 split is an extension/restructure of an existing section, not a from-scratch page: state the `[tool.uv.sources]` asymmetry explicitly at the top of the extras table (plain pip never sees the cuda index pins and gets the default torch build), lane-label every install command (uv lane / pip lane), and link the honesty matrix page. The Dependency Groups table (:198-226) needs the rename (base→full row + deprecation note), cuda121 row deletion, rocm row deletion + restore-path pointer.

## Validation Mechanics (CONTEXT-locked depth)

### Fresh-venv per-extra resolves (local, GB10 box)

For each CHANGED extra — `full`, `base` (alias), `all`, `fla` (Linux marker-false proof), plus one unchanged control (e.g. `cpu` or `cuda124`):

```bash
uv build --no-sources
W=$(ls dist/*.whl)   # concrete path REQUIRED
uv venv /tmp/xe-<extra> --python 3.12
uv pip install --python /tmp/xe-<extra>/bin/python "$W[<extra>]"
/tmp/xe-<extra>/bin/python -c "import dnallm"
```

- [VERIFIED: live probe] `"$W[base]"`-style concrete-path+bracket works under pip and uv; the quoted glob form **fails**: `uv pip install 'dist/*.whl[full]'` → `error: The wheel filename '*.whl' is invalid: Must have a version` [VERIFIED: probed 2026-10-14]
- Budget note: each resolve pulls torch (multi-GB on cuda variants; the default-PyPI torch on full/all/cpu lanes). The box's uv cache mitigates; sequence resolves; fla/full/all share most of the tree. cuda-extras resolves are uv-index-routed and conflict-pairwise — one venv each.
- The `base`-alias resolve doubles as the alias proof: assert `isort` present in the venv (a full-only member reaching the alias install).

### Smoke dispatch against the post-surgery tree

`gh workflow run "Publish to PyPI" --ref <surgery-branch>` → smoke job runs the never-uploading chain against that ref. **Honest scope note:** the smoke's install-from-dist step installs `dist/*.whl` WITHOUT extras — it proves core metadata + core deps resolve from the built artifact + twine check + import probe; the extras resolution proof is the per-extra fresh-venv resolves above. The smoke is the end-to-end artifact gate; the local resolves are the extras gate (this matches CONTEXT's "fresh-venv resolve per changed extra + smoke dispatch" — two complementary proofs, and no smoke edit is required). If the planner wants the smoke to also prove one extra: extend with `W=$(ls dist/*.whl); uv pip install --python ... "$W[full]"` in a second venv (concrete path, per the probe) — optional, adds install weight to the weekly cron.

Assert via `gh api runs/<id>/jobs` jq: `publish` conclusion `skipped`, `smoke` conclusion `success` (17-01 D2 pattern, run 38554913144).

### Census discipline (cross-cutting invariant)

New/extended guard tests ADD collected counts → the D-03 census triple (216/217 stage pins in ci.yml) and any skip-audit surfaces move SAME-CHANGE; this phase gets its one census-moving edit (roadmap invariant: one per phase). Extend tests/test_extras_guard.py + test_plant_helixseek_fla_kernels.py, then re-measure and re-pin the ci.yml stage counts.

## Grep-Gate Design (sanctioned-residue pattern)

Phase-13 precedent (OLLM-03): the residue grep equals an exact verbatim N-line record reproduced in the PLAN and SUMMARY — non-empty by construction (0 or short output = deletion creep = phase failure), re-runnable byte-for-byte, with the match token and file order fixed explicitly.

Recommended gate for `.[base]`:

- **Scan scope:** `pyproject.toml`, `.github/`, `docs/`, `example/`, `README.md`, `tests/`, `scripts/` — NOT `.planning/` (historical records are immutable by convention)
- **Sanctioned residue (expected shape, finalize at execution):**
  1. pyproject.toml — the alias line `"dnallm[full]"` is NOT a `[base]` bracket hit; the only pyproject hit is the alias declaration line `base = [` itself + its deprecation comment (depends on the exact grep pattern — define the pattern to match `\[base\]|\[base,` install-bracket forms, which the alias declaration does NOT match; then pyproject has ZERO hits and the gate is pure install-command residue)
  2. README deprecation note + installation.md deprecation note (prose "formerly `base`, renamed in 1.3; the alias installs `full` and will be removed in the next minor release")
  3. Nothing else — all 69 doc/example + 11 workflow occurrences flip to `full` (or die with cuda121)
- **Enforcement vehicle (Claude's discretion):** a pytest repo-scan test (tomllib/grep in Python, placed in tests/test_extras_guard.py alongside the member guards — keeps it inside the existing census-pinned suite) asserting the residue equals the sanctioned list verbatim; the plan records the list; the phase-13 edge discipline applies (empty/short = fail, extra line = fail, missing line = fail)
- Same gate family catches cuda121/rocm stragglers: a second pattern `cuda121|torch-rocm|\[rocm\]` over the same scope with its own (near-empty) sanctioned list — excluding the fla-upstream `[rocm]` mentions (README:254, faq:13) by exact-line sanction

## Guard Tests Extension

Existing pattern (tests/test_extras_guard.py): tomllib → `data["project"]["optional-dependencies"]` → frozenset member assertions. Extend with:

```python
EXPECTED_FULL_MEMBERS = frozenset({
    "dnallm[dev,test,notebook,mcp]",
    "isort>=6.0.1",
    "types-transformers>=0.1.0",
})

def test_full_extra_members_preserved(self):       # rename lost nothing
def test_base_alias_is_pure_self_reference(self):  # base == ["dnallm[full]"] exactly
def test_triton_windows_marker_declared_in_fla(self):
    # exact-string assertion: "triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'"
def test_cuda121_and_rocm_extras_absent(self):     # retirement cannot regress
    # plus: no [[tool.uv.index]] named torch-cuda121/torch-rocm in [tool.uv]
```

And in tests/models/test_plant_helixseek_fla_kernels.py — flip the :51 expected `all` member list literal and the :60-62 example strings `base`→`full`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deprecation warning on extra use | sentinel/dummy PyPI package, install-time hooks, manufactured resolver conflicts | The alias + docs deprecation note | Wheels run no code at install; manufactured conflicts corrupt resolution; no native mechanism exists (probe-verified) — docs IS the loud channel |
| Extras resolution proof | Bespoke resolver scripts | Fresh-venv `uv pip install "$W[extra]"` per changed extra | The real resolver against the real artifact is the only honest proof; the pattern is probe-verified |
| Residue enforcement | One-off grep in CI yaml | pytest repo-scan with verbatim sanctioned list | Keeps the gate inside the census-pinned suite; phase-13 pattern has 4 designed edge dispositions already |
| GPU matrix content | Re-derive boundaries from vendor charts | The 0542a6b verified record (paste + adjust for cuda121 retirement) | "Publishing matrix rows without verification provenance" is a recorded anti-feature; the provenance exists |

**Key insight:** every dishonesty this phase removes was created by metadata claiming something installers don't do — the fixes are all metadata/docs truth-telling plus resolver-verified proofs, no new machinery.

## Runtime State Inventory

> This phase includes a rename (base→full) — inventory taken.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | PyPI published metadata for dnallm ≤1.2.1 carries `Provides-Extra: base` (old kitchen-sink semantics) — immutable, version-pinned forever | None — resolution is version-consistent (old versions serve their own old `base`); the alias ships as NEW metadata |
| Live service config | GitHub environment `pypi` (17-03, pending registration) binds workflow filename `publish.yml` + environment — the surgery renames no workflow file | None — verify no workflow-file renames sneak into the phase |
| OS-registered state | None — verified by grep: no tasks/services/embedded registrations reference the `base` extra string | None |
| Secrets/env vars | `PYPI_API_TOKEN` (deletion gated on first green OIDC release, WINDOWS #25 — carried item, not this phase); runner uv caches keyed by `hashFiles('**/pyproject.toml')` rotate automatically on the surgery commit | None — cache keys self-invalidate |
| Build artifacts / installed packages | `dnallm.egg-info/` + `build/` in tree (stale metadata regenerates on next build — delete before fresh-venv proofs so no stale extras shadow); local `.venv` editable dist-info reads **dnallm-0.7.1** (stale vs pyproject 1.2.1); Windows box `dnallm-cuda` env has triton-windows 3.7.1.post27 installed manually (WINDOWS #18 remedy) — the declaration aligns metadata with that reality | Rebuild clears egg-info; local .venv re-sync is routine (not part of the phase's proof — fresh venvs install from the built wheel); Windows env needs no migration, only the KDA re-verification pass |

**Canonical question answer:** after every repo file is updated, the systems still holding the old string are: PyPI's immutable ≤1.2.1 metadata (by design, version-consistent), stale local egg-info/build dirs (rebuild clears), and the uv caches (key-rotate). None require data migration; all are code-edit-or-no-op.

## Common Pitfalls

### Pitfall 1: Trusting the "5 call sites" count
**What goes wrong:** the rename lands on the 5 bare `.[base]` lines, leaving `.[base,fla]`-class compound lines on ci.yml:343/537/744 and `.[base,docs]` :1127 partially renamed or untouched — a half-changed extras surface (the exact Pitfall-12 class from the milestone research).
**Why:** PKGH-01/ROADMAP say "the 5 CI/publish call sites", inherited from STACK.md:42's bare-install enumeration.
**How to avoid:** the rename map in this doc lists all 11 install lines + 3 workflow comments + workflows README prose; instruct executors to re-grep (`grep -rn '\[base' .github/ pyproject.toml`) at execution time — line numbers drift (publish.yml's site moved :39→:59 since the 10-11 research).
**Warning signs:** post-surgery grep returns compound-bracket hits outside the sanctioned list.

### Pitfall 2: An unsatisfiable "pip warns on the alias" acceptance test
**What goes wrong:** a success criterion or test asserts `pip install dnallm[base]` produces a deprecation WARNING while the alias exists — no installer does this (probe-verified both pip and uv); the phase can never honestly pass it.
**Why:** the CONTEXT phrasing "must WARN and exit 0" describes the mechanism's INTENT (surface the hazard loudly, don't hide it), but the warn-and-exit-0 behavior is what installers do AFTER the alias is dropped.
**How to avoid:** phrase acceptance as (a) alias resolves full-set exit-0 both installers (probe-proven), (b) docs carry the deprecation note during the cycle, (c) post-alias-removal behavior is the native warning (documented in the honesty matrix so the eventual removal is pre-explained).
**Warning signs:** any task action asserting warning text during the alias cycle.

### Pitfall 3: The glob+extras wheel install
**What goes wrong:** `uv pip install 'dist/*.whl[full]'` fails — `error: The wheel filename '*.whl' is invalid: Must have a version` [VERIFIED: probed 2026-10-14].
**Why:** uv does not expand shell-globs inside a requirement string with extras.
**How to avoid:** concrete path: `W=$(ls dist/*.whl); uv pip install "$W[full]"` (probe-proven form).

### Pitfall 4: Sweeping fla's upstream [rocm]/[cuda] mentions into the dnallm-rocm removal
**What goes wrong:** README:254 and docs/faq/models_troubleshooting.md:13 (flash-linear-attention's own backend extras — "install the package bare... its [cuda]/[rocm] extras pin their own torch") get deleted/mangled as "rocm residue", corrupting load-bearing install guidance.
**How to avoid:** the rocm sanctioned-list gate names those two exact lines as sanctioned; the removal targets only dnallm's extra/index/table rows.

### Pitfall 5: Publishing the stale rocm rationale
**What goes wrong:** the restore-path doc or SUMMARY repeats "no rocm wheel exists for torch>=2.10" — false as of today (rocm7.1 ships 2.10/2.11); a reader with an MI card tries rocm6.4 (torch ≤2.9.1) believing 2.11 impossible.
**How to avoid:** use this session's index table; restore path cites rocm7.1 (2.10/2.11, Linux-only, resolution-tested-only) with the honest "never executed on AMD" framing.

### Pitfall 6: Docs mirrors drifting on the rename sweep
**What goes wrong:** example `.md` files edited without regenerating `docs/` mirrors (or vice versa) — `check_docs_sync` red.
**How to avoid:** every example edit lands mirror-same-change (WPAR-08 invariant, standing).

### Pitfall 7: Census movement without re-pinning
**What goes wrong:** new guard tests shift collected counts; ci.yml stage pins (216/217) and the skip audit go red or — worse — silently stale.
**How to avoid:** the census triple re-pin is part of the same change; measure with the exact stage flags (phase-13 discipline).

### Pitfall 8: The `all` chain edit breaking the fla-reachability test
**What goes wrong:** re-scoping `all`'s member list (e.g. dropping now-redundant `dev,test,notebook,mcp`) trips `test_fla_reachable_from_all` (parses `_meta_extra_names` over `all`'s members expecting `fla`).
**How to avoid:** keep `fla` in the member list (any shape); the base→full swap itself is safe — but update the :51 hardcoded literal either way.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Undead pinned-minor CUDA extras (cuda121-class) | Retire + document the frozen index | spaCy removed its cuda121/122 extras (discussion #13567) — direct precedent | PKGH-02 follows ecosystem practice |
| rocm extra with commented-out pins | Remove + manual AMD lane doc | rocm7.0/7.1 indexes appeared ~Oct 2026 (new since the 10-11 research) | Restore path content updated to rocm7.1 |
| triton-windows as community fork (woct0rdho) | triton-lang org, attested builds since post27 | 2026-06 (3.7.1.post27) | Fork-legitimacy objection to WPAR-07 declaration is retired |
| `pip install pkg[extra]` silently ignoring unknown extras | same behavior today (warn + exit 0, pip 23.2.1→26.x) | unchanged | The alias exists precisely because this didn't improve |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Self-referential alias resolves identically from the PyPI registry (probe covered local-wheel paths; the in-repo `all`/`dev` self-refs prove the editable+PyPI pattern daily, but `pip install dnallm[base]` against the published post-surgery wheel was not directly executed — no such wheel exists yet) | Deprecated-Alias Mechanism | Low — same name-match resolver path; first fresh-venv resolve of the published wheel would catch it |
| A2 | The next release carrying the surgery is the "one cycle"; alias removal lands the following minor (timing is an owner call at plan time) | Alias shape | Low — gate sanctions the alias until removed; removal timing recorded in the deprecation note |
| A3 | Windows box availability for the KDA re-verification during execution (box exists per WINDOWS #18; scheduling is owner-side) | triton-windows | Blocks WPAR-07 closure only, not the Linux-side surgery |
| A4 | cuda124 stays the test-cuda matrix's surviving leg and the recommended-lane examples after cuda121 retires (PKGH-05's cuda130-default push is Phase 19's surface) | cuda121 Retirement | Low — matrix shape is a one-line planner choice |
| A5 | Fresh-venv resolve budget on the GB10 box is acceptable for full/all/cpu/fla lanes (torch multi-GB downloads; uv cache mitigates) | Validation Mechanics | Time cost only; sequence resolves to keep the phase fast |

## Open Questions

1. **Should the smoke carry an extras proof?**
   - What we know: the smoke installs `dist/*.whl` bracketless (core-only); per-extra fresh-venv resolves are the CONTEXT-locked extras proof; the probe proves the `"$W[full]"` extension works.
   - What's unclear: whether the owner wants the weekly cron to also exercise one extra from the wheel (heavier cron) or keep the smoke core-only.
   - Recommendation: keep the smoke core-only this phase (matches the locked validation depth); note the one-line extension for later.
2. **Alias cycle boundary (A2):** which release removes the `base` alias — record the exact version/timing in the deprecation note at landing.
3. **test-cuda matrix post-retirement shape:** single 12.4 leg vs re-pointing the leg to a newer CUDA variant — planner's one-line choice (A4).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| uv | builds, fresh-venv resolves, smoke | ✓ (local box) | 0.13.1 (aarch64) | CI uses `setup-uv@v7` "latest" — same 0.13.x line |
| pip | alias-behavior probes, docs accuracy | ✓ (conda base) | 26.2.1 | probes already executed; no runtime dep |
| Network → pypi.org + download.pytorch.org | registry/index verification, resolves | (fetched today) | — | n/a |
| gh CLI | smoke dispatch + run assertions | ✓ (17-01 pattern in active use) | — | web UI dispatch (manual) |
| Windows box (RTX 5080) | WPAR-07 KDA re-verification | owner-side | — | none — requirement gates on it (A3) |
| mkdocs | docs build check (if lane pages added) | ✓ (repo toolchain, docs-validation lane) | — | n/a |

**Missing dependencies with no fallback:** none for the Linux-side surgery; the Windows KDA re-verification is owner-scheduled (A3).

## Security Domain

ASVS level 1 (`security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (metadata/docs/CI config only) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | marginal | PEP 508 marker/environment-marker declarations validated by the resolver itself; tomllib parses pyproject (no eval) |
| V6 Cryptography | no | — |
| V10 Malicious Code (supply chain) | **yes** | The one supply-chain-relevant act: a third-party fork (triton-windows) enters declared, published dependencies. Mitigations already in the design: exact version band `>=3.7.1.post27,<3.8` (no floating), marker-gated to Windows only, triton-lang org residency + attested builds since post27, registry-verified wheels (cp310–314 win_amd64), and the real-hardware execution proof (WPAR-07) as the human-verify checkpoint |

### Known Threat Patterns for {Python packaging / CI workflows}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Dependency confusion / slopsquatting on the new declaration | Tampering/Elevation | Registry JSON verified this session (exact files enumerated); version band caps drift; SUS flag carried with the Windows execution checkpoint |
| Workflow_dispatch abuse to publish | Elevation | Inherited from Phase 17: job-level `if: github.event_name == 'release'` gate — this phase must NOT weaken it (no publish.yml trigger edits) |
| Cache poisoning via stale keys | Tampering | uv cache keys include `hashFiles('**/pyproject.toml')` — surgery commit rotates them automatically |
| Over-broad grep-gate scope reading untrusted dirs | Information disclosure | Gate scans repo-tracked code/docs only; excludes `.planning/` and any downloaded content |

## Sources

### Primary (HIGH confidence — fetched/executed this session)
- Live probes (2026-10-14, synthetic `probeextras` wheel, clean venvs): pip 26.2.1 + uv 0.13.1 behavior matrix for alias/missing/empty extras; glob+bracket failure — outputs pasted in-line above
- download.pytorch.org index inventories (fetched 2026-10-14): `/whl/cu121/torch/` (≤2.5.1), `/whl/rocm6.3|6.4|6.2.4|7.0|7.1|7.2|7.14/torch/` version tables
- pypi.org/pypi/triton-windows/json: metadata, version history, wheel inventory, triton-lang org homepage
- In-repo reads (this session): pyproject.toml, tests/test_extras_guard.py, tests/models/test_plant_helixseek_fla_kernels.py, .github/workflows/{ci,publish,feasibility}.yml + README, docs/getting_started/installation.md, .planning/WINDOWS.md row 18, `git show 0542a6b:.planning/STATE.md` (verified GPU matrix verbatim)

### Secondary (MEDIUM confidence — authoritative, fetched 2026-10-11 by the v1.3 research; load-bearing rows re-verified live today where marked)
- .planning/research/STACK.md — uv-only `[tool.uv.sources]` quote, triton-windows fork history/torch-pairing table, cuda121/rocm index facts (rocm portion superseded by today's fetch)
- .planning/research/FEATURES.md §2 — uv/pip lane asymmetry grounding, spaCy/jax extra-rename precedents
- .planning/research/PITFALLS.md Pitfall 12 — half-changed extras surfaces

### Tertiary (LOW confidence)
- pip missing-extra behavior across versions 23.2.1–25.0: community lint script evidence ([lint_extras.py](https://github.com/nousergon/crucible-predictor/blob/main/scripts/lint_extras.py)) — consistent with, and rendered non-load-bearing by, the live 26.2.1 probe
- seam `package-legitimacy` verdicts (null-signal SUS — treated as no-observation, see audit caveat)

## Metadata

**Confidence breakdown:**
- Alias mechanism + installer behavior: HIGH — live-probed both installers, all shapes, clean venvs
- Rename/retirement surfaces: HIGH — every line enumerated by grep + read this session
- rocm facts: HIGH for today's index state (live fetch); the RECORD CORRECTION is the headline
- triton-windows: HIGH on registry facts; MEDIUM on fork-governance claims (carried from 10-11 research, homepage re-confirmed)
- GPU matrix: HIGH — verbatim verified record recovered from git, boundaries owner-accepted

**Research date:** 2026-10-14
**Valid until:** 2026-11-14 (stable surfaces) — but the pytorch.org rocm/cu index state and triton-windows latest-version drift monthly; re-fetch before relying on version tables at execution time
