# Stack Research

**Domain:** Paper-revision feature suite for an existing PyPI-published DNA-LM toolkit (fine-tuning / inference / benchmarking / MCP server)
**Researched:** 2026-10-09
**Confidence:** HIGH (headline claims verified by direct execution against the installed venv — peft 0.21.1, transformers 5.17.0, torch 2.11.0, scipy 1.18.1, sklearn 1.9.1, numpy 2.5.3 — plus live JASPAR API probes; web sources cross-checked)

## Headline Verdict

**Zero new runtime dependencies. Zero new dev/test dependencies. Every REV-01…REV-11 feature is satisfiable with the existing pinned stack plus the Python standard library.** The only two questions that could have forced an addition (VCF parsing, JASPAR/PWM handling) resolve to stdlib because (a) the only mature VCF libraries have no Windows wheels and this project ships a Windows CI leg, and (b) the JASPAR REST API natively returns directly-parseable text formats. The recommended "addition" for this milestone is therefore **discipline**: requirements should encode a what-NOT-to-add list (below) so the minimal footprint survives phase planning.

## Recommended Stack

### Core Technologies (all already pinned in `pyproject.toml` — no changes)

| Technology | Version (pin → installed) | Purpose in v1.2 | Why Recommended |
|------------|---------------------------|-----------------|-----------------|
| peft | `>=0.14.0` → 0.21.1 (latest 0.21.2) | REV-04 IA³ adapter | `IA3Config` has existed since peft 0.4.0 (Dec 2023) — the floor pin already covers it with 10 minor versions of margin. Verified locally: full config surface, bitsandbytes dispatch file `peft/tuners/ia3/bnb.py` present, end-to-end `get_peft_model` + forward passed under transformers 5.17.0 on both a BERT-style transformer **and** a mambapy Mamba backbone (REV-04's two acceptance targets) |
| scipy | `>=1.15.2` → 1.18.1 | REV-09 bootstrap CI; REV-10 FDR | `scipy.stats.bootstrap` (since 1.7) covers ci95-bootstrap aggregation; `scipy.stats.false_discovery_control` (since 1.11, BH + BY methods) covers motif-scan multiple testing. Both verified working on the installed version. The 1.15.2 floor is comfortably above both introduction versions |
| scikit-learn | `>=1.4.0` → 1.9.1 | REV-07 frozen-embedding probing | `LogisticRegression`, `MLPClassifier`, `StandardScaler`, `train_test_split`, `roc_auc_score`, `average_precision_score` — the entire probing component is one `sklearn` import block. Nothing beyond what's installed |
| transformers | `>=4.49.0,<6` → 5.17.0 | REV-06 `random_init=True` | Canonical from-scratch path is `AutoConfig.from_pretrained` + `AutoModel*.from_config(config)` — stable across the whole 4.49–5.x span; `from_config` exercised locally under 5.17 in the IA³ smoke test. Also REV-01's `TrainingArguments` semantics (`eval_strategy`, `load_best_model_at_end`) are Trainer-side config, no library change |
| numpy | `>=1.26.0` → 2.5.3 local / 1.26.4 & 2.2.0 CI matrix | REV-08 scoring math; REV-10 PWM log-odds scan; REV-07 `.npz` embedding cache | Vectorized PWM scanning (log-odds via strided windows or `numpy.lib.stride_tricks`), VEP Δlog-likelihood arithmetic, `savez` cache. The 1.26.4 CI leg is safe: everything used here is decade-stable numpy core API |
| mcp SDK | `>=1.3.0,<2` → 1.30.0 (latest 2.x = 2.3.0, deliberately not taken) | REV-11 three new tools | New tools are plain methods registered via the existing `self.app.tool()(self._with_timeout_wrapper(fn, name))` pattern (`dnallm/mcp/server.py:264-291`). The `app.tool()` registration API is stable across the 1.x pin. mcp 2.x exists but upgrading is orthogonal risk with zero feature need for REV-11 — do not bundle it into this milestone |
| pyyaml | `>=6.0` → installed | REV-05 presets table; REV-02 alias table | `configs/presets/lora_targets.yaml` (per-family target_modules + recommended r) and any metric-alias YAML are plain `yaml.safe_load` documents |

### Standard Library Components (the actual "additions" — all `import x` with no install)

| Component | Module | Purpose | Integration Point |
|-----------|--------|---------|-------------------|
| Minimal VCF reader | `gzip` + `str` splitting | REV-08 `evaluate_vcf`: parse CHROM/POS/ID/REF/ALT + INFO (e.g. ClinVar `CLNSIG`) from text VCF | New `dnallm/inference/vep.py`, ~100 lines: skip `##` meta-lines, read `#CHROM` header, split fixed fields on tab, split multi-allelic ALT on comma, extract label from INFO by key. **Critical fact verified:** bgzipped VCF (ClinVar's distribution format) is multi-member gzip — stdlib `gzip.open` reads it sequentially, no index needed for a scoring sweep |
| JASPAR fetch + PWM parser | `urllib.request` + ~50-line parser | REV-10 motif ingestion | New `dnallm/interpret/motifs.py`. **Live-verified 2026-10-09:** canonical host has moved — `jaspar.genereg.net` 301s to **`jaspar.elixir.no`**; `GET /api/v1/matrix/{MA_ID}?format=pfm\|meme\|transfac\|jaspar\|json\|yaml\|bed`. `format=pfm` returns raw ACGT count rows; `format=meme` returns MEME4 letter-probability with `strands: + -`, background frequencies, `nsites`, and E-value. Prefer `meme` (single fetch carries strands + background metadata needed for correct log-odds); compute GC-matched background from query sequences when uniform 0.25 is not acceptable |
| Benjamini-Hochberg FDR | `scipy.stats.false_discovery_control` | REV-10 FDR over scan hits | Already in scipy floor (≥1.11 < 1.15.2 pin); verified locally |
| Bootstrap CI | `scipy.stats.bootstrap` | REV-09 `aggregate_seeds` → mean/sd/ci95 | New `dnallm/finetune/sweep.py`, pure function; mean/sd via `numpy` (`std(ddof=1)`) |
| Parameter hashing | `hashlib` | REV-06 "randomly initialized" proof (parameter hash differs from pretrained) | `dnallm/models/model.py` `load_model_and_tokenizer(..., random_init=True)` |
| Metric registry | plain dict + YAML aliases | REV-02 canonical names + alias resolution | New module — **placement warning:** do NOT put it at `dnallm/tasks/metrics/registry.py` as the intake plan sketches. `dnallm/tasks/metrics/` is the vendored dir omitted from `[tool.coverage.run]`, ruff, and mypy — a registry there is invisible to the 90% hard gate and unlinted, exactly what this project's ethos forbids. Place at `dnallm/tasks/metric_registry.py` (or `dnallm/tasks/registry.py`), outside the omit glob |

### Supporting Libraries (existing, referenced by new features)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | `>=2.4.0,<2.12` → 2.11.0+cu130 | `torch.nn.init` re-init for REV-06; scoring kernels for REV-08 already exist in `mutagenesis.py:258/312` (`mlm_evaluate`/`clm_evaluate`) and `inference.py:1746` (`scoring`) | REV-08 reuses these kernels — no new scoring code paths |
| peft `prepare_model_for_kbit_training` | 0.21.1 | IA³ + QLoRA combination | If `use_ia3` is combined with 4-bit: peft ships `tuners/ia3/bnb.py` (Linear4bit/Linear8bitLt variants registered for IA³), and the correct call order — `prepare_model_for_kbit_training(model)` **before** `get_peft_model` — is already what `trainer.py:157-164` does for LoRA. IA³ rides the identical path; no new ordering code |
| click | existing | REV-08 CLI entry (`dnallm-vep` or subcommand in `dnallm/cli/cli.py` with lazy import per house convention) | Follow `dnallm/cli/cli.py` lazy-import pattern |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| pytest + pytest-cov (existing) | Tests for all new modules behind the `fail_under=90` gate | No new frameworks (hard constraint). New pure-Python modules (vep reader, registry, sweep, motifs parser) are unit-testable offline with committed fixtures: a tiny synthetic VCF (plain + gzipped), a committed JASPAR `meme`-format snippet, constructed seed arrays |
| Typed network skips (existing pattern) | Live JASPAR fetch tests | Follow the repo's `network-unavailable:` typed-skip + `expected_skips.yaml` allowlist machinery; deterministic offline tests parse committed fixtures, one live probe test behind the gate |

## Installation

```bash
# NOTHING TO INSTALL. The milestone adds zero dependencies.
# Existing floors already exceed every feature's requirement:
#   peft>=0.14.0   (IA3Config needs >=0.4.0)
#   scipy>=1.15.2  (false_discovery_control needs >=1.11; bootstrap needs >=1.7)
#   scikit-learn>=1.4.0, numpy>=1.26.0, pyyaml>=6.0, mcp>=1.3.0,<2
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Stdlib VCF reader (~100 lines) | **cyvcf2** (0.34.0) | Only if random-access via `.tbi`/`.csi` indexes or BCF (binary) input ever becomes a requirement, **and** Windows support is dropped or the dep is made a Linux/macOS-only optional extra. cyvcf2 has Python 3.13 + numpy-2 wheels — but Linux/macOS only |
| Stdlib VCF reader | **pysam** (0.23.3/0.24.0) | Same conditions. pysam's maintainers explicitly do not support Windows ("Have you tried WSL?"); PyPI can't publish their MSVC-incompatible builds. Both alternatives would break the `test-windows` lane or force conditional deps — disqualifying for a published package with this CI matrix |
| `urllib.request` + custom parser for JASPAR | **biopython** `Bio.motifs` (1.88) | Only if JASPAR/TRANSFAC/MEME format zoo handling grows beyond the 2 formats needed. Biopython is a C-extension package added to parse a format the API already serves in trivially-splitting text — wrong footprint trade |
| `urllib.request` + custom parser | **jaspar-fetch** client / **meme.io** | Never for this milestone — thin wrappers over a one-endpoint REST API; meme.io is a web tool, not a library fit |
| `scipy.stats.false_discovery_control` | **statsmodels** `multipletests` (0.15.0) | Only if non-BH/BY procedures (e.g. Holm step-down with specific conventions, or a need to match statsmodels output byte-for-byte in a comparison) become a reviewer requirement. BH via scipy is the standard citation-grade answer |
| `scipy.stats.bootstrap` | Manual bootstrap loop in numpy | Neither needs adding; scipy's version is battle-tested, seeded, and offers percentile/BCa — prefer it. (A manual loop is acceptable only if the aggregation JSON must record the exact resample indices, an exotic need) |
| Existing mcp 1.x pin | mcp 2.3.0 | Only as a separate, dedicated milestone with its own regression scope — the server, transports, and example notebooks all currently run on 1.30.0. REV-11 needs nothing from 2.x |
| `AutoConfig.from_pretrained` + `AutoModel*.from_config` | `from_pretrained(..., state_dict={})` hacks or manual `post_init` | Never — `from_config` is the documented from-scratch path across 4.49–5.x |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| cyvcf2 / pysam as core deps | No Windows wheels (htslib/MSVC mismatch); breaks the Windows CI leg; ~30 MB C baggage for 5 fields of a text format | Stdlib reader in `dnallm/inference/vep.py` |
| biopython | C-extension dependency added for a ~50-line parse of a text format the JASPAR API serves directly | `urllib.request` + custom `meme`/`pfm` parser |
| statsmodels | One function (`multipletests`) duplicating `scipy.stats.false_discovery_control` already inside the floor pin | scipy |
| mcp SDK 2.x upgrade inside this milestone | Zero feature need; server/transports/notebooks validated on 1.30.0; unrelated regression surface during a time-boxed revision cycle | Stay on `mcp>=1.3.0,<2` |
| Any VEP framework (Ensembl VEP, gpn, dart-eval as deps) | Suite-side scoring is Δlog-lik / log-odds over existing kernels; frameworks are heavyweight, GPL/service-bound, or model-specific | Reuse `mutagenesis.py:258/312` + `inference.py:1746` kernels behind the new alignment layer |
| torchmetrics / ignite | Metric emission already has a house path (`dnallm/tasks/metrics.py` → REV-02 registry); a second metrics framework fragments the contract the milestone exists to tighten | REV-02 registry over sklearn/scipy functions |
| Putting REV-02 registry inside `dnallm/tasks/metrics/` | That directory is the vendored-code omit in `[tool.coverage.run]`, ruff, and mypy — new code there escapes the 90% gate and lint, silently | `dnallm/tasks/metric_registry.py` outside the omit glob |
| New test frameworks | Hard project constraint (pytest + pytest-cov only) | Existing pytest config/markers |

## Stack Patterns by Variant

**If IA³ is combined with QLoRA 4-bit (REV-04 × existing `use_qlora`):**
- Call order `prepare_model_for_kbit_training(model)` → `get_peft_model(model, ia3_config)` (peft ships `ia3/bnb.py`; verified present in 0.21.1)
- Watch the known gradient-checkpointing `use_reentrant` interaction — same caveat as the existing LoRA path, not IA³-specific; keep the trainer's current handling

**If IA³ targets a Mamba/hybrid family (REV-04 acceptance requires one Mamba model):**
- Works mechanically — IA³ wraps `nn.Linear` by name. Smoke-proven on mambapy `Mamba` with `target_modules=['in_proj','x_proj','out_proj'], feedforward_modules=['x_proj']` (448 trainable params, forward OK)
- The authoritative per-family module lists belong in the REV-05 presets YAML (derived from each model's `config.json`/module names, per the intake plan's "no guessing" rule) — not hardcoded in trainer.py

**If the JASPAR fetch must run in CI or offline:**
- Default to committed fixture PWMs for unit tests; live fetches go behind the typed network-skip pattern. Host must be `jaspar.elixir.no` (genereg.net now 301s); make the base URL a parameter for mirror/proxy environments (the project already runs CI with `HF_ENDPOINT=hf-mirror.com` precedent)

**If seed count is small (n=3) for REV-09 bootstrap:**
- Use `method='percentile'` (BCa can degenerate at tiny n); fix `n_resamples` (e.g. 10,000) and `random_state` in the protocol so the JSON `statistics` block is reproducible; document the method choice in the sweep docstring (reviewers will ask)

**If REV-06 `random_init=True` meets a special family (EVO, Enformer, …):**
- `from_config` is the generic-transformers path; special-family loaders (`dnallm/models/special/*`) have bespoke load flows — restrict `random_init` initially to generic `Auto*` families and raise `ValueError` with a clear message for special families, rather than half-supporting re-init inside handlers (scope control, matches house error conventions)

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| peft `IA3Config` (needs ≥0.4.0) | floor `peft>=0.14.0` | Margin of 10 minors. Verified on 0.21.1 under transformers 5.17.0 + torch 2.11.0; also on mambapy backbone |
| `scipy.stats.false_discovery_control` (needs ≥1.11) | floor `scipy>=1.15.2` | Pure-Python over numpy — no ABI concerns on either numpy CI leg |
| `scipy.stats.bootstrap` (needs ≥1.7) | floor `scipy>=1.15.2` | Same |
| sklearn probing APIs (LogisticRegression/MLPClassifier) | floor `scikit-learn>=1.4.0`, numpy 1.26.4 & 2.2.0 legs | On the numpy-2.2.0 leg the resolver naturally picks sklearn ≥1.5 (numpy-2-compatible); 1.4.x self-caps below numpy 2 — the floor self-corrects, no pin change needed |
| transformers 4.49–5.x `from_config` + Trainer eval semantics | existing `>=4.49.0,<6` span | `from_config` exercised locally under 5.17; Trainer `eval_strategy`/`load_best_model_at_end` are config-level (REV-01 is pure dnallm code) |
| mcp `app.tool()` registration | `mcp>=1.3.0,<2` (1.30.0 installed) | Stable across 1.x; 2.3.0 exists but is explicitly out of scope |
| JASPAR API `?format=` | live, versioned `/api/v1/` | Host migration verified 2026-10-09 (genereg.net → elixir.no); pin nothing, parameterize base URL |
| Python 3.11/3.12/3.13 matrix + requires-python ≥3.10 | all stdlib components used (`gzip`, `urllib.request`, `hashlib`) | stdlib API is stable since 3.10; nothing new constrains the floor |

## Sources

- **Local execution against installed venv (highest confidence, 2026-10-09):** `IA3Config` field surface + instantiation (peft 0.21.1); IA³ end-to-end smoke on BERT-style transformer (transformers 5.17.0, `SEQ_CLS`, trainable 546/36,292 params) and on mambapy `Mamba` (448 trainable params); `peft/tuners/ia3/bnb.py` presence; `scipy.stats.bootstrap` (BCa) and `scipy.stats.false_discovery_control` working calls; sklearn probe estimator imports; `AutoModelForSequenceClassification.from_config` under transformers 5.17
- **Live JASPAR API probe (first-party, HIGH):** `https://jaspar.elixir.no/api/v1/matrix/MA0004.1?format=pfm` and `?format=meme` responses captured 2026-10-09; API format list (json/jsonp/jaspar/meme/transfac/pfm/yaml/bed) from `https://jaspar.genereg.net/api/`
- [peft 0.4.0 release notes](https://newreleases.io/project/pypi/peft/release/0.4.0) — IA³ introduced alongside QLoRA support (cross-checked with [IA3Config source at v0.19.0](https://github.com/huggingface/peft/blob/v0.19.0/src/peft/tuners/ia3/config.py) and [HF peft IA3 docs](https://huggingface.co/docs/peft/package_reference/ia3)) — MEDIUM-HIGH
- [peft IA3 bnb dispatch listing](https://leeroopedia.com/index.php/?title=Environment:Huggingface_Peft_BitsAndBytes_Quantization&oldid=27093) (IA3 among bnb-supported methods) + local file check — HIGH (cross-verified)
- [pysam GitHub issue #1137](https://github.com/pysam-developers/pysam/issues/1137) (maintainer: no Windows support), [pysam release notes](https://pysam.readthedocs.io/en/v0.24.0/release.html) (3.13/3.14 wheels Linux/macOS), [cyvcf2 changelog](https://data.safetycli.com/packages/pypi/cyvcf2/changelog?page=1) (numpy-2 fix in 0.31.1, wheels through cp314, no Windows) — MEDIUM (multi-source consistent)
- PyPI JSON API live checks (2026-10-09): latest versions — mcp 2.3.0, peft 0.21.2, statsmodels 0.15.0, biopython 1.88 — HIGH (first-party registry)

---
*Stack research for: DNALLM v1.2 Paper Revision Suite Support (REV-01…REV-11)*
*Researched: 2026-10-09*
