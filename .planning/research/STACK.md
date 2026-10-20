# Stack Research

**Domain:** Windows platform parity, packaging/extras honesty, and publish-chain robustness for an existing uv-built PyPI package (`dnallm` 1.2.1)
**Milestone:** v1.3 "Platform Reach & Packaging Honesty"
**Researched:** 2026-10-11
**Confidence:** MEDIUM overall — headline claims verified against official docs (docs.pypi.org, docs.astral.sh, download.pytorch.org index pages, PyPI release pages) and live probes of the installed dev venv (transformers 5.17.0, mypy 2.4.0, coverage 7.16.2, torch 2.11.0+cu130, numpy 2.5.3); per-claim tiers below. GPU support matrix boundaries, the numpy-vs-bedtools parity spikes, and existing validated capabilities were NOT re-researched (owner directive).

## Headline Verdict

**One new dependency is actually needed: `triton-windows` behind a Windows marker in the `fla` extra (and even that is a declare-vs-document owner decision). Everything else in v1.3 is configuration, deletion, CI workflow authoring, and vendored code following in-repo precedent.** The numpy interval evaluators need zero additions (numpy ≥2.0 already a hard floor). The cuda121 extra retires on verified facts (index frozen at torch 2.5.1, gone from the official selector). The rocm extra should be removed rather than restored (no rocm wheel exists for torch 2.10/2.11; the lane can only ever deliver torch ≤2.9.1 and was never execution-tested on AMD hardware). The publish chain moves to `pypa/gh-action-pypi-publish` (OIDC Trusted Publishing) with a scheduled smoke workflow that stops at `uv build --no-sources` + `twine check` + a dist-install probe (optionally TestPyPI via a separate trusted-publisher registration). GAP-1 needs no dependency — a live probe of transformers 5.17.0 against the NT promoter remote code shows every import resolves natively or via the existing `transformers_compat.py` rungs except one method (`invert_attention_mask`), i.e. the established vendored-rung ladder is the right tool.

---

## Recommended Stack

### A. Windows platform parity (numpy interval evaluators)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **numpy (already installed)** | `>=2.0.0` floor (dev venv 2.5.3; CI matrix 1.26.4/2.2.0 legs retire with the 1.x lane) | Jaccard + loj-intersect interval evaluators as library code | The parity spikes (261010-wx2 jaccard, 261011-0a3 loj-intersect) already proved numpy equals `bedtools` v2.31.1 on the frozen selection pair and randomized property cases. Interval arithmetic (`np.searchsorted`, boolean masks, `argsort` for ascending-order join traversal) is decade-stable core API — safe on both CI numpy legs. **Do not add any interval library** (see What NOT to Use) |
| Platform detection (stdlib) | `platform.system() == "Windows"` or `sys.platform == "win32"` | Runtime dispatch in `generate_bpe_dataset.py`, the NER notebook, and the CRE notebook | No dependency; matches the existing `platform_system != 'Windows'` marker vocabulary already in `pyproject.toml`. Pick ONE idiom for example code (recommend `sys.platform == "win32"` — cheapest and import-free of `platform`) and use it consistently |
| pytest.mark.skipif / typed-skip infra (existing) | pytest ≥8.4 (installed) | Platform-aware test gates at the three inventoried sites (`test_script_execution.py:128` `_gate_script_pybedtools`, `test_notebook_execution.py:1397` `_gate_pybedtools`, `test_plant_helixseek_showcase.py:523` `which('bedtools')` fail-loud) | The milestone's own machinery (typed skips, `audit_skips.py`, allowlist) already exists; on Windows the gates become numpy-path passes instead of skips, so `expected_skips.yaml` needs no growth |
| `pytest.approx` (stdlib of pytest) | — | Frozen-value tolerance pins | The ledger already observed trailing float deltas (numpy `0.3247270000001` vs frozen `0.324727`); `pytest.approx(..., abs=1e-6)` on the band assertions in `tests/examples/` absorbs them without loosening the selection.md bands |
| **pybedtools (existing, marker unchanged)** | `>=0.11.0; platform_system != 'Windows'` in dev extra (latest upstream 0.12.1) | Linux keeps bedtools; frozen truth chain stays bedtools-computed | Official docs (0.12.1): **"Windows is not supported"** (untested under Cygwin/conda compilers; community answer is WSL). Verified MEDIUM. Confirms the platform-split decision and the existing marker — no change, no Windows wheel will ever appear (bedtools CLI itself has no native Windows build) |

**Integration points (from the WINDOWS #19 change-surface inventory — do not re-inventory):** new evaluator module (precedent: `dnallm/utils/genomic_coords.py`) consumed by `generate_bpe_dataset.py:7,249` + `data_generation_and_inference.ipynb` cell 2/cell 8 (cell 0's `!uv pip install pyfastx pybedtools` must skip pybedtools on Windows), and `plant_helixseek_cre.ipynb` cell 11 jaccard; test-windows CI leg becomes the numpy-path executor (leg already installs `.[base]` via `uv pip install -e`, needs no new step).

### B. triton-windows declaration (#18 decision)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **triton-windows** (PyPI; import name `triton`) | **`>=3.7.1.post27,<3.8; platform_system == 'Windows'`** inside the `fla` extra | Make `pip install dnallm[fla]` actually functional on Windows (fla-core imports top-level but `fla.ops.*` raises `ModuleNotFoundError` without triton; torch Windows wheels declare no triton dependency) | **Declare it — the risk picture changed in the project's favor.** Verified MEDIUM (GitHub releases + PyPI): the fork moved from `woct0rdho/triton-windows` to the **`triton-lang` org** (github.com/triton-lang/triton-windows; maintainers woct0rdho + jammm, AMD/Intel contributions, the pytorch issue #178368 discussion on its future). Since `3.7.1.post27` (Jun 21 2026) wheels are **reproducible and attested**. PyPI metadata: `requires-python >=3.10,<3.15`, win_amd64 wheels cp310–314 (covers the whole 3.11–3.13 CI matrix), MIT license, ~50 MB wheel. The exact version `3.7.1.post27` is the one empirically proven on the owner's RTX 5080 (sm_120) — KDA chunk kernel compiled and executed (WINDOWS #18) |

**Version pin rationale:** cap `<3.8` because the fork's own guarantee table pairs 3.7 with torch 2.12/2.13 and 3.8 with torch 2.14/2.15 — both above the project's `torch<2.12` ceiling; letting the resolver float to 3.8.0.post29 (latest, Sep 28 2026) leaves the guaranteed pairing zone. torch 2.11 is unlisted in the fork's table (Linux upstream pairs torch 2.11 ↔ triton 3.6 — the dev box proves it: triton 3.6.0 with torch 2.11.0+cu130); the Windows-side evidence is the executed 3.7.1.post27 proof, so pin exactly that line and re-validate in the landing phase (one-command smoke: import `fla.ops` + compile a KDA chunk kernel on the Windows box).

**Alternative (rejected):** document-only (manual `pip install triton-windows==3.7.1.post27` step in the fla/Windows docs). Defensible pre-triton-lang-org; now strictly worse — it keeps the `.[fla]`-on-Windows install lying (succeeds, then breaks at import) which is exactly the dishonesty class this milestone exists to remove.

### C. Packaging & extras hygiene

| Item | Recommendation | Evidence / Rationale |
|------|---------------|---------------------|
| **`base` extra rename** | Rename to **`dev-all`** (or `ci` — owner taste); update the 5 workflow call sites (`ci.yml:84`, `ci.yml:169`, `ci.yml:410`, `feasibility.yml:67`, `publish.yml:39`), the `all` extra composition (`dnallm[base,dev,...]` → new name), and the docs sweep (installation.md, quick_start, faq, README, example mirrors — inventoried by grep 2026-10-11) | `base` today = `dnallm[dev,test,notebook,mcp] + isort + types-transformers` — it is the everything-for-development aggregate, not a base install. Nothing about the stack changes; this is pure honesty + rename cascade. Keep a one-release note in docs ("formerly `base`") |
| **cuda121 extra** | **Retire**: remove the extra, the `torch-cuda121` `[[tool.uv.index]]` entry, the `tool.uv.sources` line, and its 5 conflict rows | Verified MEDIUM (direct index fetch 2026-10-11): `/whl/cu121` is **frozen at torch 2.5.1** (cp313 Linux-only; no Windows cp313 wheel), absent from the official pytorch.org Get Started selector (now cu126/cu128/cu130 + ROCm). The extra can never resolve a torch newer than 2.5.1 while the package floor is `torch>=2.4.0,<2.12` — a dead lane pretending to be a choice. GPU matrix (pre-verified) needs nothing from it |
| **rocm extra + index pin** | **Remove** (opinionated): delete the `rocm` extra, the commented `torch-rocm` index/sources/conflict lines; document a manual AMD lane (`uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.4`) for MI-series users | Verified MEDIUM (direct index fetches): rocm6.3 AND rocm6.4 both top out at **torch 2.9.1** (Linux manylinux only; no Windows); ROCm 7.x pairs with torch ≥2.13, above the `<2.12` cap. Restoring the pin would create a "supported" lane that (a) caps two minors below the default cuda130 lane, (b) has never been execution-tested on AMD hardware (no AMD card in the verified matrix). Honesty favors deletion + docs. If the owner insists on restoring: pin `torch-rocm6.4` (adds cp314 over 6.3, same 2.7.0–2.9.1 range), restore conflicts vs cpu + all cuda extras, and label it resolution-tested-only |
| **cuda130 as default** | No dependency change (`cuda130` extra exists, `torch>=2.9.0,<2.12`); docs change only | Plain-PyPI Linux `pip install torch` already delivers the cu130 build (sm75+/Turing and up — verified MEDIUM via the pytorch compatibility index); Windows PyPI torch wheels are CPU-only. This fact IS the uv-vs-pip lane story (below) |
| **uv-vs-pip lane docs** | Document two lanes explicitly; no pyproject change can fix pip | Verified MEDIUM (uv docs, fetched): *"Sources are only respected by uv. If another tool is used, only the definitions in the standard project tables will be used."* `[tool.uv.sources]` index pins never reach published wheel metadata — `pip install dnallm[cuda130]` on Windows silently gets CPU torch. Honest pip lane: `pip install torch --index-url https://download.pytorch.org/whl/cu130` FIRST, then `pip install dnallm[fla]` etc. (or `--extra-index-url`, with the usual resolution caveat). Also adopt **`uv build --no-sources`** in the publish path — uv docs recommend it to prove the build matches what pip/build would see |
| **`langchain-ollama` in mcp extra** | Keep (no change now); flag for the ollama-disposition phase | The floor (`>=1.1.0`) exists solely for the ollama notebook cell; the notebooks stay in-repo under the v1.3 directive (only pytest surface is removed/skip-marked, and the mcp_example notebooks are already excluded per commits 8c25a5a/c308079), so the dep remains load-bearing for manual runs. Revisit only if the owner later deletes the notebooks |

### D. Publish chain (smoke + Trusted Publishing)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **pypa/gh-action-pypi-publish** | `@release/v1` (latest v1.14.x — v1.14.2 per newreleases tracking; LOW-MEDIUM on the exact patch, HIGH on the v1 line) | Replace `twine upload` + `PYPI_API_TOKEN` with OIDC Trusted Publishing in `publish.yml` | The PyPA-blessed path (docs.pypi.org "Publishing with a Trusted Publisher" names it directly). Recipe — verified MEDIUM: job-level `permissions: id-token: write` (mandatory; job-level strongly recommended), **remove** `username`/`password` inputs, `environment: pypi` on the job (optional, strongly encouraged — enables required manual approvals via the GitHub environment), and register the publisher on the EXISTING PyPI project: pypi.org → Your projects → dnallm → Manage → **Publishing** → owner `zhangtaolab`, repo `DNALLM`, workflow filename `publish.yml`, environment `pypi`. v1.13.0+ generates PEP 740 **attestations** natively (a feature `uv publish` lacks) and fixed GHSA-vxmw-7h4f-hqxh |
| **uv (build + optional publish)** | 0.12.x (repo uses `astral-sh/setup-uv@v7`, installed 0.12.21) | `uv build --no-sources` produces sdist+wheel (already the build step in publish.yml — add `--no-sources`) | Already the project toolchain. `uv publish` is a viable drop-in alternative (credential-free on GitHub Actions once a trusted publisher is registered — verified MEDIUM via uv packaging guide) but it only **uploads** PEP 740 attestations, it cannot generate them; the pypa action wins on attestations. Keep `uv build` either way |
| **twine (check only)** | `>=7,<8` (latest 7.0.0, Jul 27 2026; prior 6.2.0) | `twine check dist/*` — metadata/readme rendering validation in the smoke lane | The publish.yml line `uv pip install --system build twine` is **unpinned across a major bump** (6.x→7.0.0, now requires Python ≥3.10) — itself a rust vector. Either pin `twine>=7,<8` or drop `build` entirely (uv build replaces it) and keep twine solely for `check`. Do NOT keep twine as the upload path once OIDC lands |
| **publish-smoke.yml (new workflow)** | GitHub Actions, `workflow_dispatch` + `schedule` (suggest monthly cron) | Rust-proof the chain: checkout → setup-uv → `uv build --no-sources` → `twine check dist/*` → fresh-venv `pip install dist/*.whl` + `import dnallm` probe → (optional) TestPyPI upload | **There is no `uv publish --dry-run`** (verified MEDIUM — not documented in the uv packaging guide), so a smoke that never uploads must stop at build+check+install-from-dist. For an upload-exercising smoke: register a SECOND trusted publisher on the **TestPyPI** `dnallm` project (same owner/repo, workflow `publish-smoke.yml`; OIDC audience is `testpypi`) and use the pypa action with `repository-url: https://test.pypi.org/legacy/` + `skip-existing: true`. Never register the smoke workflow as a producer on production PyPI — the real publisher stays bound to `publish.yml` |
| **PYPI_API_TOKEN retirement** | Delete the secret after the first green OIDC release | Standard migration hygiene; PyPI docs don't mandate it (not covered on the fetched pages — LOW, operational judgment), but keeping it is a standing phishing surface with zero remaining function. Multi-publisher per project is explicitly supported, so a rollback path (re-register token) always exists |

### E. mypy cap + #20 coverage crash (dev-extra hygiene)

| Item | Recommendation | Evidence / Rationale |
|------|---------------|---------------------|
| **mypy cap** | `mypy>=1.15.0,<2.5` in the dev extra | Verified MEDIUM: mypy release timeline is 2.0 (2026-05-06), 2.1 (2026-05-11), 2.2 (2026-07-07), 2.3 (2026-07-13), **2.4 (2026-10-01 = current latest)**. The dev venv is green on exactly 2.4.0. An uncapped floor lets fresh CI resolves jump minors on a cadence of weeks; 2.4 itself flipped the default to the native Ruff-based parser (legacy parser removal planned early 2027), so 2.5 is a plausible behavior-change release. Cap at the validated series; raise deliberately |
| **#20 (local-venv `pytest --cov` crash)** | No tool addition possible: coverage 7.16.2 (Sep 27, 2026) is BOTH the latest release and the installed version — there is no upstream fix to bump into | The crash is coverage `source_pkgs` importing dnallm under the tracer → torch → numpy 2.5.3 tripping numpy's "cannot load module more than once" C-extension guard (reproduced in WINDOWS #20; CI legs on numpy 1.26.4/2.2.0 unaffected). Remediation lanes: pin the local dev venv numpy to 2.2.x (matches the CI matrix leg) or keep the recorded `coverage run --include=` CLI workaround. Decision is env-policy, not stack |

### F. GAP-1 — NT promoter checkpoint on transformers 5.17+ (windows #11/#12)

**No new dependency.** The tool is the in-repo `dnallm/utils/transformers_compat.py` absence-gated vendored-rung ladder (10 rungs already shipped on this pattern). New evidence from this research — live probe of installed transformers 5.17.0 against the actual remote file (`zhangtaolab/nucleotide-transformer-v2-100m-promoter/modeling_esm.py`, fetched 2026-10-11):

| Remote-code requirement | transformers 5.17.0 status | Covered by existing rungs? |
|---|---|---|
| `from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward` | **module EXISTS, all three names present** (probed live — eliminates the biggest import-time risk) | n/a |
| `from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, …PoolingAndCrossAttentions, MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput` | all five present (probed) | n/a |
| `find_pruneable_heads_and_indices`, `prune_linear_layer` (from modeling_utils/pytorch_utils) | absent / present natively (probed) | **YES** — `_patch_remote_code_pruning_helpers` |
| `self.get_extended_attention_mask(...)` | absent (probed) | **YES** — `_patch_get_extended_attention_mask` |
| `self.get_head_mask(...)` | absent (probed) | **YES** — `_patch_get_head_mask` |
| `self.init_weights()` entries + `post_init()` | 5.x bookkeeping shape | **YES** — `_patch_legacy_init_weights_bookkeeping` |
| `config.is_decoder` / `config.add_cross_attention` reads | no defaults (probed; rung probes fresh `PretrainedConfig()`) | **YES** — `_patch_pretrained_config_legacy_defaults` |
| **`self.invert_attention_mask(...)`** (called in `EsmModel.forward` when cross-attention is active) | **ABSENT and NOT shimmed** (probed) | **NO — the one known gap** |

**Recommended approach:** Option A (compat shim) — add a vendored `invert_attention_mask` rung (upstream tag v4.49.0, `modeling_utils.py` `ModuleUtilsMixin.invert_attention_mask`, same absence-gated attach-on-`PreTrainedModel` shape as the `get_head_mask` rung), then load the checkpoint on 5.17 and extend rung-by-rung on the actual tracebacks (the established D-07 ladder; note `_set_gradient_checkpointing` signature drift is a benign-override candidate if it surfaces). Option B (removal) — drop the third model from the benchmark notebook / skip the pkl leg permanently. Owner decides at planning time per PROJECT.md; the probe table above says Option A is small: exactly one known-missing method plus whatever the load run surfaces. If shimming succeeds, closure of #15 rides along per the milestone context.

---

## Installation

```bash
# pyproject.toml deltas (the only dependency-tree changes in v1.3):

# 1. fla extra — declare triton-windows on Windows only (#18)
[project.optional-dependencies]
fla = [
    "flash-linear-attention>=0.5.2,<0.6",
    "triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'",
]

# 2. dev extra — mypy cap
mypy>=1.15.0,<2.5   # was: mypy>=1.15.0

# 3. retire: cuda121 extra + [[tool.uv.index]] torch-cuda121 + sources line + 5 conflict rows
# 4. remove (recommended): rocm extra + commented torch-rocm index/sources/conflicts
# 5. rename: base -> dev-all (+ all-extra composition, 5 CI call sites, docs sweep)
```

```yaml
# publish.yml — OIDC migration (job-level)
permissions:
  id-token: write   # mandatory for Trusted Publishing; job-level strongly recommended
  contents: read
# environment: pypi   # optional, strongly encouraged (required reviewers via GH environment)
# steps: keep checkout/setup-python/setup-uv; uv build --no-sources; twine check dist/*;
#        - uses: pypa/gh-action-pypi-publish@release/v1   # NO username/password inputs
# delete after first green OIDC release: secrets.PYPI_API_TOKEN
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| numpy evaluators (Windows) | pyranges1 / ruranges | Never in v1.3 — rejected 2026-10-10 with evidence (WINDOWS #19): ruranges 0.2.7 has no win_amd64 wheels, needs ≥3.12, Rust msvc source build on Windows; the marker would simply transfer. Revisit only if ruranges ships win wheels |
| numpy evaluators (Windows) | WSL-scoped bedtools provisioning in CI/local | Contradicts the native-parity goal (the classifier claims OS Independent; the dev machine is Windows 11); WSL-on-hosted-runner adds a second OS surface to keep green. Rejected |
| Declare triton-windows | Document-only manual step | Only if the owner weighs "third-party fork in declared deps" over install honesty even post-triton-lang-org/attested builds — but that objection is now much weaker (see section B) |
| pypa/gh-action-pypi-publish (upload) | `uv publish` (trusted publishing) | If the project wants a single-toolchain rule (uv everywhere) more than PEP 740 attestation generation. `uv publish` is otherwise equivalent: credential-free in Actions once the publisher is registered |
| Keep token + dry-run smoke only | Full OIDC migration | The milestone already records the Trusted-Publishing migration as deferred owner implementation-decision; the smoke can ship first (build+check+dist-install) with the token untouched, and OIDC lands as a second change. Sequencing choice, not either/or |
| Vendored compat rungs (GAP-1) | Remove the NT third model from examples | If the load run after the `invert_attention_mask` rung surfaces a disproportionate ladder (several more removed APIs), removal is the honest exit; single-rung cost argues for the shim |
| Remove rocm extra | Restore pin at torch-rocm6.4 | Only if an AMD user materializes or the owner values the lane existing; it caps at torch 2.9.1 and stays resolution-tested-only |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| pyranges1 / ruranges / any genomic-interval library | Windows wheel gap + Python floor + the parity spikes already proved numpy suffices; a new dep also violates the minimal-footprint invariant the last two milestones held | Pure-numpy evaluators in `dnallm/utils/` |
| Upstream `triton` on Windows | No win_amd64 wheels exist upstream; the Windows gap is the entire reason triton-windows exists | `triton-windows` behind the marker |
| `pip install dnallm[cuda13x]` documented as GPU-complete | `[tool.uv.sources]` is uv-only (documented verbatim by uv); pip users on Windows silently get CPU torch — the exact silent-lie this milestone removes | Two-lane install docs (uv one-liner / pip torch-index-first) |
| `uv build` (without `--no-sources`) in the publish path | uv docs explicitly recommend `--no-sources` to prove the published artifact builds as pip/build would see it | `uv build --no-sources` |
| twine as the upload path post-OIDC / manual `mint-token` OIDC | PyPI docs call the manual mint-token flow an unstable internal interface; keeping token upload after migrating keeps a phishable secret alive for no benefit | pypa/gh-action-pypi-publish (or uv publish) |
| Registering the smoke workflow as a production PyPI publisher | Widens the OIDC trust surface to a scheduled, lower-scrutiny workflow | Smoke stops at build+check+dist-install, or uploads to TestPyPI under a separate registration |
| Unpinned `twine` / `build` installs in CI | twine crossed a major (6.2.0 → 7.0.0, Jul 2026) — unpinned resolves drift majors silently; `build` is redundant under `uv build` | Pin `twine>=7,<8` (check only), drop `build` |
| New test frameworks / assertion libs for tolerance pins | Suite invariant: pytest + pytest-cov only | `pytest.approx`, existing skipif/typed-skip machinery |
| bumping coverage to chase #20 | 7.16.2 (Sep 27 2026) is both latest and installed — no fix exists to bump into | Local venv numpy 2.2.x pin or the recorded coverage-CLI workaround |
| mcp 2.x / torch ≥2.12 / numpy <2 floors | Orthogonal upgrades with zero v1.3 feature need; each breaks a verified matrix boundary | Existing floors |

## Stack Patterns by Variant

**If installing on Windows (uv lane):** `uv pip install -e ".[dev-all,fla]"` — the marker pulls triton-windows 3.7.1.post27 automatically; torch comes from whichever cuda extra/index the user resolved first (Windows PyPI default is CPU-only — the docs must say so).

**If installing on Windows (pip lane):** `pip install torch --index-url https://download.pytorch.org/whl/cu130` FIRST (or cu128 for 20xx/50xx per the verified matrix), then `pip install dnallm[fla]` — the extras cannot route pip to the torch index (uv-only pins).

**If publishing a release:** tag → `publish.yml` on release published → OIDC upload with attestations (environment `pypi` if adopted). **If smoking the chain monthly:** `publish-smoke.yml` dispatch/cron → `uv build --no-sources` → `twine check` → dist-install probe → (optional) TestPyPI.

**If a Windows user needs the NER/CRE examples:** no bedtools/pybedtools obtainable — the numpy path executes transparently behind the platform check; exon_f1=0.7522 and jaccard=0.3247 hold in band on both paths (parity pre-proven; tolerance pins absorb trailing float deltas).

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| triton-windows 3.7.1.post27 | torch 2.12/2.13 (fork's guarantee); **proven on the owner's torch-2.11-era Windows box (sm_120)** | 3.6 ↔ torch 2.10 (Linux upstream pairs 2.11↔3.6); 3.8 ↔ torch 2.14/2.15 — both outside the `torch<2.12` cap; cap `<3.8` |
| triton-windows wheels | CPython 3.10–3.14 win_amd64 (+ ARM64 3.11–3.14) | Covers CI matrix 3.11–3.13 and the Windows 11 dev box |
| cu121 index | torch ≤2.5.1, frozen | Retirement evidence; cp313 Linux-only, no Windows cp313 |
| rocm6.3 / rocm6.4 indexes | torch 2.7.0–2.9.1, Linux only | No rocm wheel for torch 2.10/2.11 anywhere; ROCm 7.x starts at torch ≥2.13 (above the cap) |
| mypy 2.4.x | dev venv (validated green) | 2.4 = current latest (2026-10-01); cap `<2.5` |
| coverage 7.16.2 | numpy 1.26.4 / 2.2.0 (CI, fine); numpy 2.5.3 local venv (`--cov` source_pkgs crash, WINDOWS #20) | No newer coverage exists; fix is the venv numpy pin |
| transformers 5.17.0 | NT promoter remote code | All imports resolve natively or via existing rungs except `invert_attention_mask` (probed 2026-10-11) |

## Sources

- docs.pypi.org/trusted-publishers/using-a-publisher/ — OIDC permissions, environment recommendation, pypa action usage, TestPyPI audience (MEDIUM, official docs, fetched 2026-10-11)
- docs.pypi.org/trusted-publishers/adding-a-publisher/ — existing-project registration steps, fields, multi-publisher support (MEDIUM, official docs, fetched)
- github.com/pypa/gh-action-pypi-publish/releases + newreleases.io — v1.14.x line, v1.13.0 attestations/GHSA fix (MEDIUM for the v1 line, LOW-MEDIUM for exact patch v1.14.2)
- docs.astral.sh/uv/guides/package/ — uv publish trusted publishing, no dry-run flag, `--check-url`, attestation upload-not-generation, `uv build --no-sources` (MEDIUM, official docs, fetched)
- docs.astral.sh/uv/concepts/projects/dependencies/ — "Sources are only respected by uv" (pip-ignored), marker/extra fields on source entries (MEDIUM, official docs, fetched)
- github.com/triton-lang/triton-windows/releases + pypi.org/project/triton-windows/ — org move, release history (3.6.0.post26 … 3.8.0.post29), attested builds since post27, requires-python, wheel matrix (MEDIUM, fetched both)
- pytorch.org/get-started/locally/ + docs.pytorch.org — current selector (cu126/cu128/cu130 + ROCm; no cu121) (MEDIUM)
- download.pytorch.org/whl/cu121/torch/, /whl/rocm6.3/torch/, /whl/rocm6.4/torch/ — direct index inventories: cu121 frozen at 2.5.1; rocm tops at 2.9.1 (MEDIUM, primary source, fetched)
- mypy-lang.org + mypy-lang.blogspot.com — 2.0–2.4 timeline, 2.4 native-parser default (MEDIUM)
- pypi.org/project/coverage/ (7.16.2, Sep 27 2026) + pypi.org/project/twine/ (7.0.0, Jul 27 2026) — latest-release checks (MEDIUM, primary source)
- daler.github.io/pybedtools/main.html — "Windows is not supported", 0.12.1 (MEDIUM, official docs)
- huggingface.co/zhangtaolab/nucleotide-transformer-v2-100m-promoter/resolve/main/modeling_esm.py — remote-code import/attr inventory (MEDIUM, primary artifact, fetched)
- Local live probes (2026-10-11, dev venv): transformers 5.17.0 hasattr/import checks (file_utils, modeling_outputs, PreTrainedModel methods, pytorch_utils); installed mypy 2.4.0 / coverage 7.16.2 / torch 2.11.0+cu130 / triton 3.6.0 / numpy 2.5.3; grep inventories of `.[base]` call sites and bedtools/jaccard surfaces (empirical, reproducible)

---
*Stack research for: dnallm v1.3 — Platform Reach & Packaging Honesty*
*Researched: 2026-10-11*
