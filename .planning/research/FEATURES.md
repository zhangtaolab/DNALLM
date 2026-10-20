# Feature Research

**Domain:** Platform parity, packaging honesty, publish-chain robustness, and docs accuracy for a distributed DNA-LM toolkit (dnallm v1.3 "Platform Reach & Packaging Honesty")
**Researched:** 2026-10-11
**Confidence:** HIGH for internal repo facts (pyproject extras inventory, publish.yml, WINDOWS ledger #11/#12/#18/#19/#20, parity-spike verdicts — read directly); MEDIUM for ecosystem claims cross-verified across ≥2 independent sources (official docs + secondary); individually noted LOW where only single/indirect sources exist

"Features" here are the v1.3 target capabilities from PROJECT.md §Current Milestone. "Users" are (a) Windows-local researchers running the example tree natively, (b) pip/uv installers choosing an install lane, (c) PyPI consumers of the published wheel, (d) maintainers operating the release chain, and (e) the paper/benchmark pipeline needing the frozen metrics (jaccard=0.3247, exon_f1=0.7522) to keep meaning the same thing on every platform. Every section answers: how do comparable mature packages implement this behavior, what is table stakes vs differentiator vs anti-feature, how complex is it, and which existing dnallm machinery does it sit on — so REQUIREMENTS.md can scope each item testable and atomic.

---

## 1 · Windows platform parity (numpy evaluators, platform dispatch, #18, GAP-1 #11/#12)

### How established tools do it

**The pybedtools/bedtools ecosystem has no native Windows story at all.** Official pybedtools docs state plainly: "Windows: may work with conda compilers or Cygwin but this is untested. Windows is not supported" ([pybedtools installation docs](https://daler.github.io/pybedtools/main.html)); Bioconda has no Windows channel, so `conda install -c bioconda pybedtools` fails with `PackagesNotFoundError` on Windows ([Stack Overflow 48441023](https://stackoverflow.com/questions/48441023/unable-to-install-bioconda-packages-in-conda-environments)), and downstream packages simply inherit the gap ([GLUE issue #95](https://github.com/gao-lab/GLUE/issues/95)). The canonical ecosystem answers are WSL or Docker — i.e., "don't run natively." A genuine native-Windows execution path is therefore already above the bar pybedtools itself sets.

**The dominant precedent for pure-Python interval engines is bioframe** (open2c): interval operations implemented directly on numpy/pandas — chromosome `groupby`, `numpy.lexsort` sorting, `numpy.searchsorted` bisection for overlap detection — with no compiled extension and no subprocess ([bioframe paper, bioRxiv](https://www.biorxiv.org/content/10.1101/2022.02.16.480748v1.full.pdf), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10903647/)). Crucially, bioframe treats bedtools equivalence as a *documented contract*, not an accident: the ["Bioframe for bedtools users" guide](https://bioframe.readthedocs.io/en/latest/guide-bedtools.html) maps each bedtools command to its bioframe call, including exactly the operation dnallm needs: `bedtools intersect -loj` → `bf.overlap(A, B, how='left')` (and `intersect -wa -wb` → `how='inner'`, `-c` → `count_overlaps`, `-v` → `setdiff`). dnallm's chosen direction — a numpy evaluator used where bedtools cannot install, with the loj join semantics given their own parity check — is the bioframe pattern applied narrowly (jaccard + loj-intersect only) instead of as a full engine. Note bioframe's trade data: it is ~2× pybedtools memory (64-bit coords) and roughly PyRanges-speed — irrelevant at showcase scale (≤200kb loci), so no performance concern gates this feature.

**Parity verification as a continuous contract, not a one-time spike.** bioframe grounds equivalence in documented mappings + property tests; dnallm already went further with measured cross-implementation spikes (261010-wx2: numpy vs bedtools v2.31.1 jaccard on the frozen selection pair 0.324727 + randomized property cases — VERDICT PARITY; 261011-0a3: loj-intersect — VERDICT PARITY). The ecosystem norm for float-producing dual implementations is tolerance-bounded comparison (`pytest.approx` / `math.isclose` with documented absolute tolerance), never exact equality across arithmetic orders — exactly the trailing-delta class dnallm observed (0.3247270000001). Platform detection idioms are settled: `platform.system()` / `sys.platform` at runtime in example code, `platform_system != "Windows"` environment markers in packaging (dnallm already uses these for pybedtools and pygenometracks), `pytest.mark.skipif` in gates.

**GAP-1 (#11/#12) is a known, named ecosystem failure class.** transformers v5 refactored config classes to dataclasses (v5.3→v5.4: keyword-only, re-based on a huggingface_hub validated dataclass) and undeclared kwargs now raise `TypeError: ...__post_init__() got an unexpected keyword argument ...` in old remote code — the [OpenELM discussion](https://huggingface.co/apple/OpenELM-1_1B-Instruct/discussions/8) is the canonical instance: works on transformers 4.57.1, fails on 5.x; `from_legacy_cache`/`to_legacy_cache`/`seen_tokens` also removed ([MIGRATION_GUIDE_V5.md](https://raw.githubusercontent.com/huggingface/transformers/c472755e79aac54d675845bff5e5c821c21260af/MIGRATION_GUIDE_V5.md)). The NT-promoter and plant-Nucleotide-Transformer-BPE checkpoints are this exact class. Resolution paths observed in the wild: (a) pin a transformers 4.x ceiling (contradicts dnallm's 4.49–5.x span invariant), (b) wait for/patch upstream remote code, (c) drop the checkpoint, or (d) shim the removed defaults — dnallm already owns the shim precedent in `dnallm/utils/transformers_compat.py` (spanning 4.49–5.x).

**#18 triton-windows.** Upstream triton ships no win_amd64 wheels. The community fork `triton-windows` (woct0rdho; development now hosted under the [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows) org) provides prebuilt wheels; Blackwell sm_120 needs triton≥3.3 + torch≥2.7 + CUDA 12.8, and wheels bundle a minimal CUDA toolchain since 3.2.0.post11 ([README](https://raw.githubusercontent.com/woct0rdho/triton-windows/master/README.md)). unsloth's official install scripts recommend the fork ([unsloth README](https://raw.githubusercontent.com/unslothai/unsloth/refs/tags/September-2025-v3/README.md)) — but as a documented install step, not a declared package dependency. Declaring a third-party compiler fork in package metadata is a trust decision each project makes individually; the mainstream pattern is documentation + optional marker-gated extra.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Runtime platform dispatch in `generate_bpe_dataset.py` + `data_generation_and_inference.ipynb` (Linux keeps `pybedtools.BedTool.intersect(loj=True)`, Windows dispatches to numpy equivalent; install cell skips pybedtools on Windows) | owner directive 2026-10-10; #19 change-surface inventory is exhaustive (2 files, one code path + cell 0 install) | LOW-MED | spike 261011-0a3 numpy loj code; `dnallm.utils.genomic_coords` precedent |
| Table stakes | numpy jaccard evaluator behind the CRE notebook's bedtools call on Windows | spike 261010-wx2 | LOW | spike code; `tests/examples/test_plant_helixseek_showcase.py` band harness |
| Table stakes | Continuous parity tests (numpy vs bedtools/pybedtools) executed on **Linux** CI where the reference binary exists — parity is re-proven every run, not asserted from the spike | bioframe documented-equivalence norm; owner precondition "ships only under parity" | MED | bedtools on Linux runners; property-case generators from spikes |
| Table stakes | Frozen-value tolerance pins: band assertions compare with documented absolute tolerance (trailing-delta class), bands parsed from selection.md unchanged | numeric-testing norm (`pytest.approx`); D-05/D-06 single-source-of-truth rule | LOW | existing `_parse_floors`/`_parse_bands` |
| Table stakes | Platform-aware test gates: the three fail-loud gates (`test_script_execution.py:128`, `test_notebook_execution.py:1397`, `test_plant_helixseek_showcase.py:523`) become platform-conditional — pass via numpy path on Windows, bedtools path on Linux | skipif/platform-conditional idiom; #19 inventory | LOW-MED | typed-skip machinery unchanged (these are gates, not skips) |
| Table stakes | test-windows CI leg executes the numpy path end-to-end and asserts exon_f1=0.7522 in band on both paths | owner directive; Windows leg already exists in matrix | MED | test-windows leg; example-nightly census discipline (re-pin counts) |
| Table stakes | Docs mirrors byte-synced with the platform-split example code | check_docs_sync.py invariant | LOW | existing byte-identity gate |
| Table stakes | GAP-1 disposition decided and executed: compat-shim (confined to `transformers_compat.py`, regression-tested, revives #11+#12) **or** removal of the NT checkpoint/script leg with typed-skip retired | OpenELM-class ecosystem norm: decide, don't leave owner-pending forever | MED (shim) / LOW (removal) | `transformers_compat.py`; `example/notebooks/benchmark/benchmark.ipynb` third model; `tests/examples/test_script_execution.py` |
| Table stakes | #18 decided: either declare `triton-windows; sys_platform == 'win32'` in the fla extra (owner approves third-party fork) **or** document the manual step and make the fla test typed-skip honest on Windows | unsloth documents rather than declares; current state (silent breakage) is the only unacceptable option | LOW either way | pyproject fla extra; fla import test |
| Differentiator | Dual-engine evaluators with the frozen truth chain staying bedtools-computed (no re-freeze) — provenance preserved while adding reach | bioframe has equivalence docs but no frozen-value contracts | — | selection.md frozen contract |
| Differentiator | Native-Windows bio-ML example execution where the pybedtools ecosystem's official answer is "use WSL" | pybedtools docs | — | whole v1.1 example-nightly investment |
| Anti-feature | Replacing bedtools/pybedtools on Linux/macOS (re-freezing the truth chain on numpy) | destroys frozen-metric provenance; owner explicitly rejected | — | — |
| Anti-feature | Platform-split logic inside library `dnallm/` evaluator code | the bedtools dependence lives in example code; the library is already platform-neutral (scikit-allel Windows-verified); moving logic in widens the audited surface for zero user gain | — | — |
| Anti-feature | Third interval engine (pyranges/ruranges) | rejected 2026-10-10: no win_amd64 wheels, py≥3.12, zero Windows gain | — | — |
| Anti-feature | Exact float equality across evaluator paths | trailing-delta class makes it permanently flaky; tolerance is the norm | — | — |
| Anti-feature | Global monkeypatch of `PretrainedConfig.__init__` for GAP-1 (shim applied at package import to all models) | OpenELM-class shims must be scoped to the affected remote-code load path only; a global patch perturbs every unrelated model load and contradicts the facade-stability discipline | — | — |
| Anti-feature | Pinning the whole project to transformers<5 for GAP-1 | contradicts the 4.49–5.x compat span; strands the suite on an aging major | — | — |

**#18 decision input (owner):** declaring the fork makes `pip install dnallm[fla]` self-healing on Windows at the cost of a third-party compiler fork in declared metadata and a maintenance exposure if the fork stalls; documenting keeps metadata clean but leaves the silent-breakage failure mode one forgotten README away. Ecosystem weight (unsloth, Wan2GP guides) treats the fork as the de-facto standard — the trust question is real but the fork is now under the triton-lang org umbrella.

---

## 2 · Packaging & install honesty (extras hygiene, install lanes, GPU matrix)

### How established tools do it

**The uv/pip lane asymmetry is the load-bearing fact.** `[tool.uv.sources]` and `[tool.uv] conflicts` are project metadata applied only by uv when resolving from source — they are **not part of the published wheel metadata, so pip ignores them entirely** ([dataeval-flow docs](https://pypi.org/project/dataeval-flow/0.2.3/#2) document exactly this pattern and its fix). Consequence for dnallm today: `pip install dnallm[cuda128]` does not route to the cu128 index — it installs whatever torch PyPI serves by default (Linux: bundled-CUDA default; Windows: CPU-only). The cudaXXX extras are honest only under uv-from-source. This is precisely why "uv-vs-pip lane docs" is table stakes and not polish. uv's own answer for pip-style installs is `--torch-backend=auto` (infers the PyTorch index from the GPU/driver; `uv pip` CLI only, ignored by `uv add`/`uv sync`) — adopted in install docs by [vLLM](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/39511/overview) and unsloth; known pitfalls: multi-GPU driver parse failures, WSL lacking `/sys/module/nvidia/version`, and PEP 440 ignoring `+cpu`/`+cuXXX` local version labels so a stale CPU torch can silently persist ([pydevtools handbook](https://pydevtools.com/handbook/explanation/installing-cuda-python-packages.md)).

**How torch communicates the matrix.** The [Get Started selector](https://docs.pytorch.org/get-started/locally/) exposes OS × package manager × compute platform (CUDA 12.6/12.8/13.0/13.2, ROCm, CPU) and emits the exact lane command (`pip3 install torch --index-url https://download.pytorch.org/whl/cu128`, or `uv add torch --default-index .../cu128`). Lane EOLs are announced publicly — e.g., [CUDA 12.6 wheels stop with PyTorch 2.15 (drops Maxwell/Pascal/Volta)](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) — and the runtime itself warns users on dropped architectures ([PR #158478](https://github.com/pytorch/pytorch/pull/158478): sm_50/sm_60 removed from cu128 builds in 2.8). Support facts that bracket dnallm's pre-verified matrix: cu121 wheels are no longer built (last ≈torch 2.5, the standard refuge for Maxwell/Pascal/Volta per [compatibility cheat sheets](http://llmlaba.com/articles/cuda-pytorch-compatibility.html) and the [CUDA wheel matrix](https://pozzettiandrea.github.io/cuda-wheels/matrix/)); CUDA 13 drops everything below Turing sm_75 ([NVIDIA blog](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)); plain-Linux `pip install torch` now defaults to a CUDA 13.0 wheel supporting CC 7.5+. dnallm's pre-verified boundaries (Pascal→cpu-only, 20xx→cu128, 30/40xx→all extras, 50xx→cu128+) sit exactly on these ecosystem facts — the docs just have to say so.

**Honest GPU extras under pip exist — when the wheels live on PyPI.** spaCy ships `cuda11x`/`cuda12x`/`cuda-autodetect` extras that genuinely work under plain pip because they pull `cupy-cuda12x`/`cupy-wheel` from PyPI ([spaCy install setup](https://deepwiki.com/explosion/spaCy/1.2-installation-and-setup)) — and notably spaCy **removed** its overly-specific `cuda121`/`cuda122` extras, breaking users who asked for them ([discussion #13567](https://github.com/explosion/spaCy/discussions/13567)) — a direct precedent for retiring a frozen minor-version CUDA extra rather than keeping it undead. jax is the gold standard: `jax[cuda12]`/`jax[cuda13]` pull plugin wheels (`jax-cuda12-plugin` + NVIDIA runtime wheels) that live on PyPI, so the extra is honest under every installer; the old `jax[cuda12_pip]` was removed and names dash-normalized per [PEP 685](https://peps.python.org/pep-0685/) ([jax v0.6.0 notes](https://www.gitclear.com/open_repos/google/jax/release/jax-v0.6.0)). dnallm cannot fully reach the jax shape (torch CUDA wheels are not on PyPI), so the honest intermediate is: keep the uv extras machinery as the primary lane, retire the undead lane, and document the pip `--index-url` lane explicitly.

**On `base` (and meta-extras generally).** No PEP or PyPA spec blesses or bans extra *names* like `base`/`all` — PEP 685 only fixes normalization; the criticism found in standards discussions targets default/meta extras generally (dependency bloat, "you may also like" UX — [PEP 771 discussion](https://discuss.python.org/t/pep-771-default-extras-for-python-software-packages-round-2/94905/14)). The dishonesty in dnallm's `base` is semantic, not syntactic: `Provides-Extra: base` is public on PyPI, and a user reasonably reads "base" as the minimal runtime when it actually installs dev+test+notebook+mcp plus isort and stubs (pyproject lines 145-149). The fix direction (fold into `dev`, or re-scope `base` to mean the actual runtime default) is a LOW-complexity rename with a one-release deprecation note; spaCy's cuda-extra removal shows users survive extra renames at 0.x/1.x.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | `cuda121` extra retired (or explicitly frozen-and-labeled legacy) | cu121 wheels stopped ≈torch 2.5 — the lane resolves to an EOL torch while `<2.12` pretends otherwise; spaCy removed its cuda121-class extras | LOW | pyproject + uv index/sources/conflicts cleanup |
| Table stakes | rocm pin decision: restore the commented-out `torch-rocm` index/conflict rows **or** remove the `rocm` extra entirely — a declared extra that resolves to plain PyPI torch on every installer is dishonest | same uv/pip asymmetry fact | LOW | pyproject |
| Table stakes | Install-lane documentation: one page/table stating, per extra, what pip actually installs vs what uv installs, with the explicit pip `--index-url https://download.pytorch.org/whl/cuXXX` lane commands | dataeval-flow pattern; torch get-started is the reference shape | LOW-MED | docs pipeline |
| Table stakes | cuda130 as the documented default recommendation | torch's own Linux default is now the cu130 wheel (CC 7.5+); project runs torch 2.11 cu130 in dev | LOW | docs |
| Table stakes | Published GPU support matrix (Pascal cpu-only, 20xx→cu128, 30/40xx all extras, 50xx cu128+) with the pre-verification evidence linked | torch/bitsandbytes-class packages publish matrices; boundaries pre-verified 2026-10-11 | MED (writing + evidence chain; verification done) | docs; pre-verification records |
| Table stakes | `base` extra rename/re-scope so no PyPI-visible extra misdescribes its contents | PyPA surface-visibility fact; meta-extra bloat criticism | LOW | pyproject; publish smoke proves no consumer breakage |
| Table stakes | Extra-name hygiene check (PEP 685 normalized, no near-collisions) after the rename | PEP 685 | LOW | — |
| Differentiator | Cross-installer honesty matrix: a table of "command × installer → what you actually get" | rare; most projects document only their blessed lane | LOW | same lane docs, phrased comparatively |
| Differentiator | Per-GPU-generation **verified** matrix (each row backed by a live install/execution proof on the actual hardware classes, not vendor charts) | torch publishes vendor-level support; nobody publishes tested-on-our-matrix evidence at small-project scale | — | GB10 box + RTX 5080 Windows env |
| Differentiator | Runtime lane warning (torch 2.8-style): detect dropped-arch GPU at import and point at the right lane doc | torch PR #158478 precedent | MED | version/device probe utilities |
| Anti-feature | Trying to make cudaXXX extras pip-routable via dependency tricks (torch local-version pins, sdists with dynamic deps) | PEP 440 local versions ignored by resolvers — the unsloth silent-CPU-training trap; complexity with zero honesty gain | — | — |
| Anti-feature | Deleting the uv index machinery and going docs-only | the uv lane is the project's primary dev lane and works; jax-level pip-honest extras are unreachable for torch | — | — |
| Anti-feature | Publishing matrix rows without verification provenance | turns the honesty milestone into its own violation; every row cites its proof | — | — |
| Anti-feature | Keeping `cuda121` "for legacy users" while the torch floor has moved | undead lane = the exact dishonesty being fixed | — | — |

---

## 3 · Publish chain robustness (smoke, OIDC, #20, mypy cap)

### How established tools do it

**Trusted Publishing (OIDC) is the recommended modern flow; tokens still work.** PyPI introduced Trusted Publishing in April 2023 ([PyPI blog](https://blogpypiorg.readthedocs.io/posts/2023-04-20-introducing-trusted-publishers/)) — short-lived (≈15-min) OIDC tokens exchanged per-CI-run, no stored credentials. Setup: job-level `permissions: id-token: write` (workflow-level discouraged), publisher registered on PyPI with owner/repo/workflow-filename, and a GitHub **environment** binding which is optional but strongly recommended — it can require manual approval and blocks fork-originated publishes ([PyPI docs: using a publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/), [safeguard.sh adoption guide](https://safeguard.sh/resources/blog/pypi-trusted-publishing-adoption-guide)). `pypa/gh-action-pypi-publish` has produced PEP 740 attestations **by default** since v1.11 (Nov 2024) ([PyPI attestations docs](https://docs.pypi.org/attestations/producing-attestations/)). No API-token sunset has been announced (LOW confidence on any removal date), but 2FA has been mandatory for all uploading accounts since 2024-01-01 ([PyPI blog](https://blog.pypi.org/posts/2023-12-13-2fa-enforcement/)) and OIDC is the direction; staged publishing went GA 2026-05-22, enabling staging-only trusted publishers ([konishi note](https://hidekazu-konishi.com/entry/publishing_npm_and_pypi_packages_without_long_lived_tokens.html)) — relevant only if the smoke lane later wants server-side rehearsal. Since dnallm already builds with `uv build`, note `uv publish` natively supports `--trusted-publishing automatic|always|never` (auto-detects GitHub OIDC with `id-token: write`) plus `--dry-run` and `--index testpypi` ([uv publish reference](https://mintlify.wiki/astral-sh/uv/cli/publish), [uv PR #7475](https://github.com/astral-sh/uv/pull/7475)).

**Publish smoke is a recognized pre-publish validation chain.** The recurring pattern across release guides: `uv build` → `check-wheel-contents dist/*.whl` (wheel-contents lint) → `twine check dist/*` (metadata/README rendering) → **install the built wheel in a clean venv and import-smoke it** → optional TestPyPI rehearsal (`uv publish --index testpypi --dry-run`) → real publish; the publish job downloads already-validated artifacts rather than rebuilding ([PyKMP releasing guide](https://gertvdijk.github.io/PyKMP/releasing/), NVIDIA-NeMo and Ananke-Plexus release workflows). dnallm's current `publish.yml` post-hoc "Verify installation" (`pip install dnallm` from PyPI after upload) is the race-y inverted variant: propagation delays can serve a stale version, and it validates nothing before the artifact is public. The honest variant validates the built artifact in a clean venv *before* upload. A `workflow_dispatch`-triggered dry-run lane running this chain periodically (no credentials needed if it stops before upload) catches resolver/registry/toolchain drift between releases — rare in the ecosystem (most projects validate only at release time), which makes it a genuine differentiator rather than a borrowed norm.

**#20 (local-venv `pytest --cov` crash)** is internal tooling: coverage 7.16.2 `source_pkgs` imports dnallm under the tracer, tripping numpy 2.5.x's load-once C-extension guard on the py3.13 local venv (CI legs on numpy 1.26.4/2.2.0 unaffected; WINDOWS #20). The feature framing is "dev-env reproducibility": either a documented numpy ceiling for the local lane or a coverage invocation that doesn't import-at-start. **mypy cap** for the dev extra follows the widespread practice of bounding type-checker versions in dev dependencies so a new mypy release cannot flip CI red overnight (MEDIUM confidence — consistent with how scipy/numpy pin mypy in dev requirements; not separately researched).

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | workflow_dispatch publish-smoke: `uv build` → `check-wheel-contents` → `twine check` → clean-venv install-from-wheel + import/version smoke, exiting before any upload | PyKMP/NeMo/Ananke pattern | MED | existing uv build step in publish.yml |
| Table stakes | Real publish path reuses the same validation steps (one reusable chain; publish downloads validated artifacts) | release-workflow norm | LOW-MED | publish.yml refactor |
| Table stakes | Trusted Publishing migration (if chosen): job-level `id-token: write`, PyPI publisher registration + `pypi` environment, `PYPI_API_TOKEN` secret retired | PyPI docs; uv publish `--trusted-publishing automatic` | MED | publish.yml; PyPI-side registration (owner) |
| Table stakes | mypy upper cap in dev extra (`mypy>=1.15.0,<X`) | dev-tool bounding practice | LOW | pyproject dev extra |
| Table stakes | #20 fix or documented local-lane pin so `pytest --cov` works repo-wide in the dev venv | WINDOWS #20 | MED (debugging-heavy) | coverage config in pyproject |
| Differentiator | *Periodic* (scheduled, not just pre-release) smoke dispatch — catches drift between releases | rare in ecosystem | LOW (once dry-run exists) | smoke workflow + cron |
| Differentiator | PEP 740 attestations on published artifacts | free with OIDC; supply-chain credibility for a scientific package | — | rides the migration |
| Anti-feature | Smoke that uploads anywhere real (PyPI or TestPyPI) without explicit promotion semantics | version-collision noise; dry-run exists precisely to avoid this | — | — |
| Anti-feature | Post-upload PyPI install as the *only* install smoke (current shape) | propagation race validates the wrong artifact; clean-venv from the built wheel is the honest form | — | — |
| Anti-feature | Keeping API token and OIDC both live long-term | two standing credential classes after migrating defeats the point | — | — |
| Anti-feature | Full test suite inside the release job | release job bloat; the gate already ran on the exact commit — validate artifacts, not tests, at publish time | — | — |

**OIDC decision input (owner, may subsume the smoke):** migration cost is workflow surgery + one PyPI registration; benefit is no stored token, environment-gated publishing, default attestations. The dry-run smoke is orthogonal and worth having regardless of which credential mechanism wins — the token flow can host the dry-run lane today.

---

## 4 · Docs accuracy review (docs/ + example/ per-file sweep)

### How established tools do it

**Executable documentation is the ecosystem's strongest anti-drift tool, and dnallm already exceeds the norm.** The standard ladder: doctest / [pytest-doctestplus](https://github.com/scientific-python/pytest-doctestplus/) (numpy/astropy; FLOAT_CMP directives, rst support) → [sybil](https://skills.rest/skill/sybil) (executes prose-embedded code samples inside pytest — audeering's auglib [migrated to it](https://github.com/audeering/auglib/pull/51) to catch docs drift) → sphinx-gallery (scikit-learn/matplotlib: every example script executed at docs build) → notebook validation via nbval/nbpages ([LSST SQR-032](https://raw.githubusercontent.com/lsst-sqre/sqr-032/tickets/DM-20939/index.rst)). dnallm's example-nightly already executes the entire example tree and keeps docs mirrors byte-identical (`check_docs_sync.py`) — a stronger guarantee than sphinx-gallery gives its users. What the ecosystem adds on top is *periodic prose hygiene*: scheduled link checks and quarterly example reviews (e.g., [yt-cli docs gates](https://yt-cli.readthedocs.io/_/downloads/en/v0.13.5/pdf/)), and mkdocs `--strict` (warnings-are-errors) as the table-states build gate — dnallm carries 15 pre-existing `--strict` warnings as ledgered debt.

**What "systematic review" means at mature projects:** a per-file pass with an explicit checklist (does every named API exist; does every install command still resolve; is every version claim current; is every number traceable), fixes landing in the same pass, and a regression mechanism so the fixed state stays fixed. For dnallm the regression mechanisms already exist (byte-sync gate, example-nightly, docs-validation.yml); the milestone feature is the *sweep itself* plus closing the warnings debt. The post-numpy2/py311 staleness class (old install commands, pre-float64-change idioms, removed-API references) is exactly what doctest/sybil-style execution catches mechanically — but only for code; prose claims need the per-file human-verifiable pass the milestone defines.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Per-file accuracy sweep of docs/ + example/ prose vs code, with same-pass fixes and per-file disposition records | milestone definition; periodic-review norm | MED-HIGH (breadth: full docs surface) | docs-validation.yml; check_docs_sync.py |
| Table stakes | Install-command correctness re-verified **after** the packaging changes land (ordering dependency) | lane-honesty is the point of the matrix docs | LOW if sequenced after | packaging phase outputs |
| Table stakes | mkdocs `--strict` zero-warnings (15 ledgered warnings dispositioned/fixed) | warnings-are-errors is the table-stakes gate | MED | mkdocs config |
| Table stakes | Showcase/selection.md wording reconciled with the platform split (frozen-convention cells stay bedtools-worded; numpy path documented as parity-verified alternate) | #19 inventory classes the docs-only surface | LOW | selection.md contract |
| Differentiator | Every repaired docs claim carries the executing test that proves it (accuracy with evidence chain) | matches the project's verification culture; no ecosystem norm demands it | — | example-nightly |
| Differentiator | Recorded per-file review evidence (auditable "this doc was verified against this code on this date") | docs-audit trails are rare | LOW | planning artifacts |
| Anti-feature | Restructuring docs (Diátaxis re-org, theme changes) inside the accuracy pass | accuracy ≠ reorganization; scope creep doubles the diff under review | — | — |
| Anti-feature | LLM-bulk-rewritten docs without per-claim verification | hallucination risk; violates the truth→docs direction the project enforces elsewhere | — | — |
| Anti-feature | Fixing drift by editing code to match stale docs | direction is truth → docs, always | — | — |

---

## 5 · Ollama test-surface removal (~59 refs, remove-vs-skip per item)

### How established tools do it

**The consensus is unambiguous: permanently-skipped tests are dead code — delete them; git history is the archive.** PostHog's test-maintenance guidance is the crispest statement: "Delete expired tests. Do not leave them skipped," with deletion requiring that the covered behavior be nameable and no supported path needs it ([PostHog skill](https://github.com/PostHog/posthog/blob/master/.agents/skills/maintaining-python-tests/SKILL.md)). The middle-ground pattern (USGS [dataretrieval-python PR #259](https://github.com/DOI-USGS/dataretrieval-python/pull/259)) allows a module-level skip **only inside a bounded deprecation window** with a removal date, after which the file is deleted. The recurring pathology named across sources: skip lists monotonically grow because "a skipped test is indistinguishable from a deleted one after the first sprint" ([quarantine-pattern writeup](https://dev.to/multigrid/quarantining-a-flaky-llm-test-without-deleting-it-2k1j)). dnallm is better armed than most: typed-skip prefixes + `expected_skips.yaml` + `scripts/audit_skips.py` already make every skip enumerated and CI-enforced — the ledger prevents the invisible growth pathology by construction.

Mapping to the directive: refs covering surface that will **not** be supported (dead integration paths, probes for services the runner will never host again) → delete per the PostHog rule; refs covering surface that **remains a supported feature** but is environment-gated (the two mcp_example client notebooks' live-ollama/MCP tests, WINDOWS #13) → the existing probe-then-execute typed skip with live evidence is the defensible state, since the feature is alive and the skip is self-healing by design. Per-item disposition at planning time is exactly the ecosystem-correct shape.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Every one of ~59 ollama refs dispositioned remove-or-skip with recorded rationale per item | owner directive 2026-10-11; PostHog/dataretrieval norms | MED (breadth: 5 test files + 2 notebooks) | typed-skip machinery; audit_skips gate |
| Table stakes | `expected_skips.yaml` + census counts re-pinned after removal (audit exit 0, no unexpected skips) | existing CI gate discipline | LOW | audit_skips.py; D-03 census triple |
| Table stakes | Removed-surface tests deleted outright (not zombie-skipped) | "permanently-skipped tests are dead code" consensus | LOW | — |
| Differentiator | Disposition ledger documenting why each ref was removed vs kept — the audit-skips philosophy extended to deletions | matches project culture; not an ecosystem norm | LOW | WINDOWS ledger / planning artifacts |
| Anti-feature | Blanket skip-everything "to be safe" | creates the forever-skipped zombie class the consensus warns about; the directive explicitly forbids it | — | — |
| Anti-feature | Removing the ollama MCP-client *capability* from the library/MCP server | directive scope is pytest surface + notebooks, not the feature; library surface stays | — | — |
| Anti-feature | Deleting live-probe evidence for the remaining supported mcp_example pair | the both-up green state (6 live-server probes) is v1.1-earned truth; keep unless the notebooks themselves go | — | — |

---

## Feature Dependencies

```
[numpy evaluators (jaccard + loj)]
    └──requires──> [parity tests on Linux CI]          (spikes proved it once; tests keep it true)
                          └──gates──> [platform dispatch in example code]
                                            └──requires──> [platform-aware test gates]
                                                                  └──requires──> [test-windows numpy-path executor]
                                                                                         └──asserts──> [exon_f1 band on BOTH paths]

[packaging extras hygiene (base/cuda121/rocm)]
    └──orders──> [install-lane docs + GPU matrix docs]     (docs must describe the final extras shape)
                       └──orders──> [docs accuracy sweep]  (install-command re-verification sequenced last)

[publish smoke (dry-run chain)]
    ├──independent of──> [OIDC migration]                  (either credential flow can host the dry-run)
    └──proves──> [base-extra rename consumer safety]

[ollama disposition] ──independent── (re-pins census counts consumed by CI gates)

[GAP-1 shim-or-remove] ──independent── (touches benchmark notebook + script test only)
[#18 triton-windows declaration] ──independent── (pyproject + one test + docs row)
```

### Dependency Notes

- **Platform split requires continuous parity:** the owner precondition ("ships only under parity") is met by the spikes but must survive future bedtools/numpy releases — hence parity tests on Linux CI precede the split landing in the roadmap, and the Windows executor's band assertion is the end-to-end closure.
- **Packaging precedes docs sweep:** install-lane/matrix docs describe the post-hygiene extras shape; the accuracy sweep re-verifies install commands last so it isn't done twice.
- **Publish smoke enhances (does not require) the OIDC migration:** the dry-run chain is credential-free; the migration changes only the upload step's auth. The owner note "may subsume" means one workflow can carry both.
- **Ollama disposition conflicts with nothing** but re-pins the census/gate counts other features assert against — land it early so later phases pin against the final counts.
- **GAP-1 shim conflicts with facade stability if implemented globally** — hence the anti-feature row; a scoped shim inside the affected load path composes with the compat-span invariant instead.

## MVP Definition

### Launch With (v1.3 — all in scope; these are the P1 core)

- [ ] Platform dispatch + numpy evaluators + parity tests + platform-aware gates + test-windows numpy executor + tolerance pins — the milestone's headline; exon_f1=0.7522 in band on both paths is the acceptance
- [ ] Extras hygiene: `base` rename/re-scope, cuda121 retirement, rocm restore-or-remove — the dishonest metadata rows go first
- [ ] Install-lane docs + GPU support matrix + cuda130 default recommendation
- [ ] Publish smoke (workflow_dispatch dry-run chain with clean-venv install-from-wheel)
- [ ] Ollama surface: per-item remove-vs-skip disposition executed, gates re-pinned
- [ ] GAP-1 #11/#12 and #18 decided and executed (shim-or-remove; declare-or-document)

### Add After Validation (same milestone if capacity, else carry)

- [ ] OIDC Trusted Publishing migration — well-bounded workflow surgery; may subsume the smoke workflow per owner note
- [ ] mkdocs `--strict` 15-warning cleanup — ledgered debt the docs sweep surfaces naturally
- [ ] mypy cap, #20 cov-crash fix — dev-lane robustness items
- [ ] Scheduled (periodic, not just dispatch) publish smoke cadence

### Future Consideration (post-v1.3)

- [ ] Runtime lane/architecture warning at import (torch 2.8-style) — nice reach, not honesty-critical
- [ ] Staged publishing / TestPyPI rehearsal lane with promotion semantics — only if release frequency grows
- [ ] pyranges/ruranges re-evaluation — only if win_amd64 wheels ever ship (owner trigger condition already recorded)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Platform split + parity tests + Windows executor | HIGH (unlocks native Windows) | MEDIUM (surface fully inventoried) | P1 |
| Extras hygiene (base/cuda121/rocm) | HIGH (installs do what they claim) | LOW | P1 |
| Install-lane docs + GPU matrix | HIGH (first thing every user hits) | LOW-MED | P1 |
| Publish smoke dry-run | MEDIUM-HIGH (rust-proofs the chain) | MEDIUM | P1 |
| Ollama disposition | MEDIUM (suite honesty, runner stability) | MEDIUM (breadth) | P1 |
| GAP-1 disposition | MEDIUM (recovers or retires blocked surface) | MEDIUM (shim) / LOW (remove) | P1 |
| #18 triton-windows decision | MEDIUM (Windows fla users) | LOW | P1 |
| OIDC migration | MEDIUM (security posture, attestations) | MEDIUM | P2 |
| mkdocs --strict cleanup | MEDIUM (docs gate honesty) | MEDIUM | P2 |
| mypy cap + #20 fix | LOW-MEDIUM (maintainer lane) | LOW / MEDIUM | P2 |
| Periodic smoke cadence | LOW-MEDIUM | LOW | P2 |
| Runtime GPU warning | LOW | MEDIUM | P3 |
| Staged publishing lane | LOW | MEDIUM | P3 |

## Comparable Analysis

| Dimension | Comparable | Their approach | dnallm v1.3 plan |
|-----------|-----------|----------------|------------------|
| Windows interval ops | pybedtools / bioframe | pybedtools: unsupported, WSL/Docker; bioframe: pure-numpy engine with documented bedtools-equivalence map (`-loj` → `overlap(how='left')`) | platform-split: keep bedtools where it installs, numpy evaluator only where it can't — bioframe's pattern scoped to jaccard + loj, with parity *tests* (bioframe ships mappings, dnallm ships measured parity) |
| GPU extras that work under pip | spaCy (`cuda12x`, `cuda-autodetect`) / jax (`cuda12`/`cuda13` via PyPI plugin wheels) | extras pull GPU wheels that live on PyPI → honest under every installer; spaCy deleted its frozen `cuda121`-class extras | torch CUDA wheels aren't on PyPI → keep uv lanes primary, retire the undead cuda121 lane, document the pip `--index-url` lane explicitly |
| Install-lane communication | PyTorch get-started matrix + dev-discuss lane-EOL notices + runtime arch warnings | OS × platform × lane selector emitting exact commands; public lane deprecation notices | lane docs + verified per-generation matrix; cuda130 default; (P3) runtime warning |
| uv/pip asymmetry handling | dataeval-flow docs, vLLM (`--torch-backend=auto`) | explicitly document that `[tool.uv.sources]` is uv-only; give pip users index commands | same: cross-installer honesty table |
| Publish auth | pypa/gh-action-pypi-publish / `uv publish --trusted-publishing` | job-level `id-token: write`, environment binding, attestations default-on | OIDC migration (P2), token retired |
| Publish validation | PyKMP / NVIDIA-NeMo release guides | build → check-wheel-contents → twine check → clean-venv install-from-wheel smoke → optional TestPyPI dry-run | workflow_dispatch dry-run with exactly this chain, periodic cadence as differentiator |
| Docs anti-drift | sphinx-gallery / sybil / pytest-doctestplus; periodic link+example reviews | execute everything; schedule prose hygiene | already exceeds (example-nightly + byte-sync); adds the per-file sweep + `--strict` zero-warnings |
| Dead test surface | PostHog ("delete expired tests"), dataretrieval (bounded skip window) | delete removed-feature tests; skip only in dated deprecation windows | per-item remove-vs-skip with ledgered rationale; typed-skip allowlist enforces the survivors |
| Stale remote-code checkpoints | OpenELM-on-v5 class | pin ceiling, upstream fix, or drop | scoped compat-shim inside transformers_compat.py, or removal — decided at planning |

## Sources

- pybedtools installation docs (Windows unsupported): https://daler.github.io/pybedtools/main.html — MEDIUM (official docs, single source for the negative claim but unambiguous)
- bioframe paper + "Bioframe for bedtools users" + benchmarks: https://www.biorxiv.org/content/10.1101/2022.02.16.480748v1.full.pdf, https://bioframe.readthedocs.io/en/latest/guide-bedtools.html, https://pmc.ncbi.nlm.nih.gov/articles/PMC10903647/ — MEDIUM (cross-verified, official docs + peer-reviewed)
- Bioconda no-Windows / downstream breakage: https://stackoverflow.com/questions/48441023/, https://github.com/gao-lab/GLUE/issues/95 — MEDIUM
- torch get-started lane matrix + commands: https://docs.pytorch.org/get-started/locally/ — MEDIUM (official)
- cu126 wheel EOL notice (2.15, drops Maxwell/Pascal/Volta): https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432 — MEDIUM (official mailing list)
- sm_50/sm_60 removal warnings in 2.8 cu128: https://github.com/pytorch/pytorch/pull/158478 — MEDIUM
- cu121 wheels stopped / CUDA 13 drops pre-Turing: https://pozzettiandrea.github.io/cuda-wheels/matrix/, https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/ — MEDIUM (cross-checked against cheat sheets)
- `[tool.uv.sources]` invisible to pip; lane docs pattern: https://pypi.org/project/dataeval-flow/0.2.3/#2 — MEDIUM (single project's docs but states the mechanism plainly; consistent with uv docs)
- uv `--torch-backend` semantics + pitfalls: https://app.semanticdiff.com/gh/astral-sh/uv/pull/12070/overview, https://pydevtools.com/handbook/explanation/installing-cuda-python-packages.md — MEDIUM
- spaCy GPU extras + cuda121-class removal: https://deepwiki.com/explosion/spaCy/1.2-installation-and-setup, https://github.com/explosion/spaCy/discussions/13567 — MEDIUM
- jax cuda extras / PEP 685: https://www.gitclear.com/open_repos/google/jax/release/jax-v0.6.0, https://peps.python.org/pep-0685/ — MEDIUM (PEP is HIGH-legitimacy primary)
- PyPI Trusted Publishing + attestations + 2FA timeline: https://docs.pypi.org/trusted-publishers/using-a-publisher/, https://docs.pypi.org/attestations/producing-attestations/, https://blogpypiorg.readthedocs.io/posts/2023-04-20-introducing-trusted-publishers/, https://blog.pypi.org/posts/2023-12-13-2fa-enforcement/ — MEDIUM (official docs, cross-verified)
- uv publish flags (dry-run, trusted-publishing, testpypi): https://mintlify.wiki/astral-sh/uv/cli/publish, https://github.com/astral-sh/uv/pull/7475 — MEDIUM
- Publish smoke chain: https://gertvdijk.github.io/PyKMP/releasing/ — MEDIUM (single guide but corroborated by NVIDIA-NeMo/Ananke workflow shapes in the same results)
- Staged publishing GA (2026-05-22): https://hidekazu-konishi.com/entry/publishing_npm_and_pypi_packages_without_long_lived_tokens.html — LOW (single secondary source)
- Docs testing tools: https://github.com/scientific-python/pytest-doctestplus/, https://skills.rest/skill/sybil, https://github.com/audeering/auglib/pull/51, https://raw.githubusercontent.com/lsst-sqre/sqr-032/tickets/DM-20939/index.rst — MEDIUM (cross-verified)
- transformers v5 config dataclass breakage + OpenELM precedent: https://huggingface.co/apple/OpenELM-1_1B-Instruct/discussions/8, https://raw.githubusercontent.com/huggingface/transformers/c472755e79aac54d675845bff5e5c821c21260af/MIGRATION_GUIDE_V5.md — MEDIUM (official guide + live incident)
- triton-windows fork status + requirements: https://github.com/triton-lang/triton-windows, https://raw.githubusercontent.com/woct0rdho/triton-windows/master/README.md, https://raw.githubusercontent.com/unslothai/unsloth/refs/tags/September-2025-v3/README.md — MEDIUM
- Test deletion vs skip consensus: https://github.com/PostHog/posthog/blob/master/.agents/skills/maintaining-python-tests/SKILL.md, https://github.com/DOI-USGS/dataretrieval-python/pull/259, https://dev.to/multigrid/quarantining-a-flaky-llm-test-without-deleting-it-2k1j — MEDIUM (three independent sources agreeing)
- Internal (HIGH, read directly): `.planning/PROJECT.md`, `.planning/WINDOWS.md` (#11/#12/#13/#18/#19/#20), `pyproject.toml` extras + uv tables, `.github/workflows/publish.yml`, parity-spike verdicts recorded in PROJECT.md

**Gaps to flag for phase-time research:** exact pypa/gh-action-pypi-publish input syntax for the combined environment+OIDC workflow (README not fetched — the action README should be read at plan time); whether PyPI API tokens gain any sunset date (no announcement found — treat as "recommended migration," not forced); mypy capping practice not separately researched (MEDIUM from training; verify at plan time if the cap choice matters); bioframe's own test suite internals not inspected (its bedtools-parity *tests* vs documented mappings — dnallm's spike already exceeds this bar locally).

---
*Feature research for: v1.3 Platform Reach & Packaging Honesty (Windows parity, packaging honesty, publish chain, docs accuracy)*
*Researched: 2026-10-11*
