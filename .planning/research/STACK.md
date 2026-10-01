# Stack Research

**Domain:** Example-execution testing (real-model .ipynb/marimo execution under pytest) + genomics track I/O and visualization for PlantHelixSeek-CRE/-Anno showcase notebooks, on an existing pytest/torch/HF Python toolkit
**Researched:** 2026-10-01
**Confidence:** MEDIUM overall — every recommendation is cross-verified against the installed venv (deterministic local introspection) AND official docs/PyPI JSON; the GSD source-hierarchy seam assigns LOW to single-channel webfetch claims and MEDIUM to websearch-verified ones, so web-only items are flagged inline. Nothing below rests on training-data recall alone.

**Scope guard:** the pre-validated stack (pytest 8.4+/pytest-cov/pytest-timeout/pytest-asyncio, markers, torch 2.11 cu130, transformers 4.49–5.x compat, HF/ModelScope loading, `notebook` extra, self-hosted `dnallm-nightly` GPU runner) is NOT re-researched. This file covers only additions for the three new capabilities.

## Recommended Stack

### Core Technologies (new capabilities)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **nbclient** | 0.11.0 (latest; already installed) | Execute the ~21 `.ipynb` files as pytest tests | It *is* the execution engine under both `nbconvert --execute` and nbmake — use it directly as a library (in-scope per the no-new-test-frameworks constraint). Verified on the installed dist: `NotebookClient` traitlets `allow_errors=False` (raises `CellExecutionError` on the first failing cell = fail-loud, exactly the milestone's "real execution finds real errors" goal), `timeout=None`, `kernel_name=''` (falls back to notebook metadata → `python3`), `startup_timeout=60`. `nbclient.execute(nb, cwd=...)` convenience exists; `resources={"metadata": {"path": nb_dir}}` sets the kernel cwd so notebooks that load `./inference_config.yaml` relative to their own directory work (verified in installed `client.py:431,535`). |
| **marimo `export html` CLI** | 0.25.0 (latest == installed) | Headless execution of the 3 marimo apps | `marimo export html app.py -o out.html` **runs** the app headlessly (help text: "Run a notebook and export it as an HTML file" — verified locally on the installed CLI). Subprocess invocation from a pytest test gives full engine semantics (marimo runtime, UI elements, app-level `--sandbox`/args), kernel isolation mirroring the nbclient approach, exit-code-based assertion, and an HTML artifact for debugging. `marimo export session` also executes (snapshots; `--continue-on-error` default). Zero new dependencies — marimo is already in the `notebook` extra. |
| **pyBigWig** | 0.3.26 (latest) | Write BigWig signal tracks from sliding-window CRE scores | The standard write-capable bigWig library (C extension, MIT). **Verified wheel coverage:** manylinux_2_27/2_28 x86_64 wheels for cp310–cp313 on PyPI — no compiler needed on the Linux x86_64 GPU runner. Write API per official README: `bw = pyBigWig.open(p,"w")` → `bw.addHeader([("Chr1", len), ...])` (ordered chrom/length list) → `bw.addEntries(chroms, starts, ends=..., values=...)` (bedGraph-style; sorted order required; `validate=True` default) → `bw.close()` (builds index + up-to-10 zoom levels; `maxZooms=0` produces an IGV-breaking intervals-only file — do not use). numpy arrays accepted for `values`. |
| **pyfastx** | 2.3.1 (already in `dev` extra) | FASTA region slicing for Arabidopsis genome windows | Already the project's FASTA library (the committed `.fxi` index in `example/notebooks/finetune_generation/` proves the precedent). C extension + sqlite index → indexed random access into the (gitignored) full TAIR10 FASTA without loading it. Killer property for this milestone: `fa.fetch(name, (start, end))` is **1-based inclusive — identical to GFF3 coordinates** — so region extraction and GFF3 comparison share one coordinate convention with no off-by-one translation. `strand="-"` gives reverse complement in the same call. |
| **stdlib GFF3 reader/writer + interval math** | Python stdlib (no version) | Write predicted Anno gene models; parse predicted + TAIR10 truth; compute agreement | GFF3 is nine tab-separated columns with 1-based inclusive coords, `##gff-version 3` first line, percent-encoded attributes (spec v1.26 verified — see Sources). The showcase scale is ≤200 kb loci → tens-to-hundreds of features; a ~60-line strict reader (dataclass + `str.split("\t", 8)`) plus sorted-list/bisect overlap math does prediction-vs-truth comparison with **zero dependencies**, and the identical parser feeds the agreement metrics *and* the track rendering (see Track Display). Writing is f-string formatting. gffutils/BCBio.GFF add dependency weight for queries this scale never needs. |
| **altair + vl-convert-python (already present)** | altair 6.3.0 / vl-convert-python 1.9.0 | Static track + gene-model figures inside the showcase notebooks | **Zero new dependency — verified:** dnallm depends on `altair[all]`, whose `all` extra pins `vl-convert-python>=1.9.0`; `vl_convert` 1.9.0 is importable in the dev venv right now, and `dnallm/inference/plot.py:380` already calls `chart.save(...)` — the headless-save code path is already exercised by the suite. `chart.save("track.png")` produces `image/png` outputs that survive nbconvert→HTML and the mkdocs-jupyter docs mirror deterministically (Rust renderer, no browser, no kernel comm). |
| **ollama (runner service — not a pip package)** | current stable, via official `install.sh` | Local LLM backend for the 2 MCP client notebooks | One-time runner bootstrap: `curl -fsSL https://ollama.com/install.sh | sh` installs the `ollama.service` systemd unit listening on `127.0.0.1:11434`. Verified against official docs: OpenAI-compatible base URL `http://localhost:11434/v1/` supports `/v1/chat/completions` and `/v1/models` with an ignored-but-required API key — exactly the `OllamaProvider(base_url="http://localhost:11434/v1")` + `OpenAIChatModel` path the pydantic-ai notebook already uses. Model provisioning: `ollama pull qwen3.6:latest` (the tag both notebooks reference) or `POST /api/pull`; readiness probe `curl -sf localhost:11434/api/tags`. Keep it out of pyproject — it is runner infrastructure, like the HF cache. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **ipykernel** | 7.3.0 installed (latest 7.4.0) | Provides the `python3` kernel nbclient launches | Already transitive via the `jupyter` metapackage in the `notebook` extra. The venv ships its kernelspec at `{sys.prefix}/share/jupyter/kernels/python3` with `argv: ["python", ...]` — correct as long as the venv `bin` is on PATH (CI's `uv run` satisfies this; assert in the test if paranoid). No new pin needed; 7.4.0 requires Python ≥3.11, harmless for the 3.11–3.13 matrix (uv resolves 6.x for 3.10). |
| **langchain-ollama** | 1.1.0 | The actually-missing import of `mcp_client_ollama_langchain_agents.ipynb` | The notebook currently shell-magics `!uv pip install -U langchain-ollama` in a cell (side-effecting, upgrade-mutating, network-dependent — hostile to hermetic CI). Add `langchain-ollama>=1.1.0` to the `mcp` extra (deps: `langchain-core>=1.2.21,<2` + `ollama>=0.6.1,<1` — compatible with installed langchain 1.4.3) and repair the notebook cell to a plain import. This is the concrete "missing mcp extra" repair (WR-09-adjacent). |
| **pytest-timeout (existing)** | 2.4.0 | Per-test timeout override for slow execution tests | Verified: `pytest.mark.timeout` marker is available in the installed 2.4.0. The global `--timeout=300` stays; real-model notebook tests get `@pytest.mark.timeout(3600)` (signal method works on the Linux runner). This replaces nbmake's `--nbmake-timeout` entirely. |
| openai (already installed, 3.20.0) | — | Readiness/assertion calls against ollama's `/v1` endpoint in tests | Only if a test wants to assert the LLM actually answered; plain `curl` in the workflow or stdlib `urllib` on `/api/tags` suffices for the skip-guard. Do not add the `ollama` pip package — no example imports it. |

### Development / Runner Tools (environment, not packages)

| Tool | Purpose | Notes |
|------|---------|-------|
| `MPLBACKEND=Agg` + `MPLCONFIGDIR=$(mktemp -d)` | Deterministic headless matplotlib | Official docs: Agg is the non-interactive backend auto-selected on Linux without X/Wayland; `MPLBACKEND` overrides any matplotlibrc; isolated `MPLCONFIGDIR` avoids font-cache races/writes in `$HOME`. Set in the execution-test fixture (and/or nightly workflow env). matplotlib is 3.11.2 here — no pin interaction. |
| ollama systemd drop-in | Persistent, re-install-safe model cache | `sudo systemctl edit ollama.service` → `[Service] Environment="OLLAMA_MODELS=/opt/cache/ollama"` then `daemon-reload && systemctl restart ollama`. Direct unit edits are overwritten by re-running `install.sh`; drop-ins are the documented-safe mechanism. Runner-ops memory note applies: restart services with sanitized env (`env -i`) on `dnallm-nightly`. |
| Typed skip guard for ollama | Keep the skip-audit gate honest | Execution tests for the 2 MCP notebooks gate on `curl -sf localhost:11434/api/tags` and otherwise skip with the project's typed-prefix discipline (e.g. `network-unavailable: ollama service not reachable`) and an `expected_skips.yaml` entry — matching the existing census machinery. Same pattern if VRAM contention with torch models forces ordering constraints. |

## Track Display for PlantHelixSeek Notebooks (owner-added scope)

**Hard gate (owner requirement): the chosen option must support GFF3** — the Anno pipeline outputs GFF3 gene models and truth is TAIR10 GFF3; side-by-side gene-model display is the core showcase.

Evaluation axes: (a) headless execution under nbclient without hanging; (b) rendered result survives nbconvert→HTML into the mkdocs docs mirror (mkdocs-jupyter); (c) dependency footprint; (d) license compat (dnallm is MIT); (e) interactive value live vs static publication value.

| Option | GFF3 (gate) | (a) Headless nbclient | (b) mkdocs-mirror survival | (c) Footprint | (d) License | (e) Value | Verdict |
|---|---|---|---|---|
| **Static altair from parsed GFF3** (altair 6.3 + vl-convert 1.9, both already installed) | **By construction** — the notebook must parse predicted + TAIR10 GFF3 anyway for the agreement metrics; the same DataFrame draws boxes/arrows | Deterministic: pure Python + Rust vl-convert, no browser, no kernel comm, no hang surface | **Yes** — `chart.save("png")` embeds `image/png`, rendered by any template incl. mkdocs-jupyter | **Zero new deps** | MIT-compatible stack | Interactive in live Jupyter (altair tooltips/zoom), static in docs | **RECOMMENDED PRIMARY** |
| igv-notebook 0.6.2 (igv.js 3.1.4) | Native — `format: "gff3"` annotation tracks; tabix `.tbi` indexing strongly recommended for anything nontrivial (MEDIUM, websearch-verified) | Unverified — README has no headless/CI guidance; emits frontend JS with no browser present | **No** — `to_svg()` is documented "Jupyter Notebook only" (not JupyterLab); widget/JS output does not survive nbconvert→HTML into the mirror | Light (ipykernel/ipython/requests, MIT) | MIT | Best-in-class interactivity live | **Optional interactive add-on, OUTSIDE gated cells** |
| jbrowse-anywidget 0.3.0 | Native — bigWig + tabix-GFF3 + DataFrame tracks | Not pytest-proven — README itself: "pytest never opens one"; their headless runner needs puppeteer + a sibling jbrowse-components checkout | No — GPU anywidget needs a live widget frontend | **Not on PyPI** (verified — PyPI lookup fails); git-only `pip install jbrowse-anywidget @ git+...` | Apache-2.0 | Highest (GPU view, region sync) | **RULED OUT this milestone** — no PyPI pin possible = CI non-reproducible; labeled Prototype |
| pyGenomeTracks 3.9 | **Fails the gate as documented** — track list says "bed/gtf", GFF3 not documented; needs a GFF3→GTF/bed12 conversion step | Yes (matplotlib-based, CI-proven in the community) | Yes (PNG/PDF/SVG) | Heavy: `matplotlib<3.9` pin (**hard conflict** with installed 3.11.2), pysam, hicmatrix, bx-python, pybedtools, gffutils + **external bedtools binary since 3.5** (verified absent from the runner PATH) | **GPL-3.0** — real contamination concern for an MIT project's published extras | Publication-grade static tracks (the field's standard look) | **RULED OUT** — GPL + matplotlib pin + bedtools binary + GFF3 conversion friction |

**Recommendation: primary = static altair rendering; optional add-on = igv-notebook, only in cells excluded from gated execution.**

Rationale: the notebook must already hold predicted-vs-truth GFF3 as DataFrames to compute the "substantially consistent" agreement asserts — rendering gene models (rect marks for genes/exons, arrow/text for strand) and per-bin CRE scores from those DataFrames is incremental code, not a new subsystem. Side-by-side tracks are two `vconcat` charts sharing the x-scale (truth on top, prediction below). The BigWig + GFF3 **files** remain the interchange artifacts for users who want a real genome browser — the visualization is a view, not the product. If interactive browsing is wanted for demos, add an igv-notebook appendix cell (or a companion non-executed markdown snippet) referencing the same BigWig/GFF3 outputs; do not put it in the CI-gated path, and expect tabix-indexed (`bgzip` + `tabix -p gff`) files if it loads the full truth track. Re-evaluate jbrowse-anywidget when it lands on PyPI with a stable tag.

## Installation

```bash
# pyproject.toml changes (extras only — no new frameworks, no new runners)
[project.optional-dependencies]
notebook = [
    "jupyter>=1.1.1",
    "marimo>=0.16.3",
    "nbclient>=0.10",                      # ADD: make the (today transitive) dep explicit; tests import it
]
mcp = [
    # ... existing ...
    "langchain-ollama>=1.1.0",             # ADD: import used by mcp_client_ollama_langchain_agents.ipynb
]
dev = [
    # ... existing ...
    "pyBigWig>=0.3.26; platform_system != 'Windows'",   # ADD: precedent = pybedtools marker;
                                                          # wheels are linux x86_64 only (cp310-313)
]

# One-time GPU-runner bootstrap (NOT pyproject — runner infrastructure)
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl edit ollama.service        # [Service] Environment="OLLAMA_MODELS=/opt/cache/ollama"
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama pull qwen3.6:latest                # tag referenced by both MCP notebooks
```

Where each addition lands (quality-gate ask): `nbclient` → `notebook` (used by tests via the dev→notebook chain, but it is a notebook-runtime lib); `pyBigWig` → `dev` (bio tooling lives there per pyfastx/pybedtools precedent; needed at notebook runtime on the runner, which installs dev); `langchain-ollama` → `mcp` (it is the MCP-example import). Nothing goes into `test` — the execution harness is plain pytest + existing plugins.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| nbclient as a library | **nbmake / pytest-nbmake 1.5.5** (`pytest --nbmake --nbmake-timeout=N`) | If you wanted zero harness code and per-cell timeouts from a flag. Rejected: it is a pytest *plugin* adding collection semantics on top of a suite that already has strict-markers, a global `--timeout=300`, typed-skip audits, and a coverage denominator — duplicate timeout machinery and a second execution-config surface for no capability nbclient lacks. |
| nbclient as a library | **`nbconvert --execute` CLI** (nbconvert 7.17.1 installed) | nbconvert's `--execute` is a subprocess CLI around the very same `NotebookClient`; pytest sees only a process exit code, stack traces are buried in converted-notebook output, and per-notebook cwd/resources control is clumsier. Use nbconvert only for *rendering* executed notebooks to HTML for the docs mirror. |
| nbclient as a library | **papermill 2.7.0** | Parametrized notebook *pipelines* (parameters cell, cloud I/O). No parametrization need here; adds a dependency for nothing. |
| `marimo export html` subprocess | **In-process `marimo.App.run(defs=None)`** (signature verified on installed 0.25.0: returns `(outputs, defs)`) | If a test must assert on specific marimo defs, `App.run()` after importing the app module works and skips a subprocess. Primary stays the CLI because it exercises marimo's own full runtime path (UI elements, app wiring) and is the documented headless command; the official pytest guide documents only reactive test cells, not `App.run` — so in-process is the less-proven route. |
| stdlib GFF3 handling | **gffutils 0.14** (pure Python; pulls pyfaidx, argh, argcomplete, simplejson) | If later phases need a sqlite `FeatureDB` with interval queries over whole-genome annotations. At ≤200 kb loci it is dependency weight for nothing. |
| stdlib GFF3 handling | **BCBio.GFF / bcbio-gff 0.7.1** (pure Python parser/writer) | A reasonable middle ground if hand-parsing is rejected in review; still an external dep for a TSV. |
| pyfastx | **pyfaidx 0.9.0.4** (pure Python, samtools-compatible `.fai`, 0-based python slicing / 1-based `get_seq`) | If a no-C-extension constraint ever appears (pyfaidx compiles nothing). Otherwise redundant with an existing, already-indexed dependency. |
| runner ollama service | **`ollama` pip package 0.6.3** | Only if Python-level orchestration of pulls/chats is wanted; the notebooks use OpenAI-compat HTTP + langchain-ollama, and the workflow can `curl`/`ollama pull`. |
| no action | **`altair_saver`** | Deprecated upstream since altair 5; vl-convert-python (already installed) supersedes it. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| pyGenomeTracks as a dependency | GPL-3.0 in an MIT project's published extras; `matplotlib<3.9` pin conflicts with installed 3.11.2 (resolver downgrade would destabilize seaborn/logomaker/dnallm plots); requires external `bedtools` binary (absent from the runner, verified); GFF3 not a documented track type (needs GFF3→GTF/bed12 conversion) | altair static rendering; note in docs that our BigWig/BED/GFF3 outputs render fine in pyGenomeTracks for users who have it |
| jbrowse-anywidget | Not on PyPI (git-only), labeled Prototype, no pytest-proven headless path, puppeteer + sibling checkout for its own runner | Watchlist; revisit on PyPI release. altair meanwhile |
| igv-notebook in CI-gated cells | `to_svg()` classic-Notebook-only; output does not survive nbconvert→HTML/mkdocs mirror; no headless guarantees documented | Optional interactive appendix outside gated execution, pointing at the same output files |
| nbmake/pytest-nbmake | New pytest plugin semantics + duplicate timeout machinery; adds nothing over direct nbclient | nbclient called from parametrized pytest tests |
| `ollama` pip client | Nothing imports it; REST/OpenAI-compat endpoints already cover readiness + chat | curl / openai client / langchain-ollama |
| Kernel auto-resolution assumptions in CI | The venv `python3` kernelspec uses bare `python` from PATH; a workflow that runs pytest without the venv on PATH would launch a *different* interpreter's kernel | Ensure `uv run` (or explicit PATH) in the nightly job; optionally pass `kernel_name="python3"` and assert `jupyter kernelspec list` in-test |
| `maxZooms=0` when writing BigWig | Produces an intervals-only file that breaks IGV and other zoom-dependent viewers | Default zoom levels (built on `close()`) |
| Notebook cells that shell-install (`!uv pip install -U ...`) | Mutates the env mid-run, upgrades unrelated pins, needs network at cell-execution time | Declare deps in extras; repair the cell to a plain import |

## Stack Patterns by Variant

**If the job is the fast PR leg (`coverage-gate`, no GPU):**
- Execution tests are `slow`-marked and deselected there (existing mechanism). Nothing new needed; pyBigWig/nbclient still install fine.

**If the job is the nightly GPU census (`coverage-nightly` on `dnallm-nightly`):**
- Run execution tests with `@pytest.mark.timeout(3600)` overrides, `resources.metadata.path` set per notebook dir, `MPLBACKEND=Agg`, `MPLCONFIGDIR` tmp; HF models from the models.lock-keyed cache; ollama service pre-started with `qwen3.6:latest` pre-pulled into `OLLAMA_MODELS` cache; MCP server fixture bound to `:8000/mcp` for the 2 client notebooks.
- Sequencing pitfall: ollama and torch models share GPU VRAM — run the 2 MCP notebooks after (or apart from) heavy model tests, or cap ollama parallelism.

**If the platform is Windows (ungated matrix leg) or aarch64 Linux:**
- `pyBigWig` is excluded by the `platform_system != 'Windows'` marker (no wheels → sdist would need MSVC+libcurl); execution tests are skipped there anyway (no GPU/no runner services). GFF3/FASTA/altair paths stay cross-platform.

**If a notebook needs its executed form in the docs mirror:**
- Execute with nbclient (in-place), then `nbconvert --to html` (or write the nbformat node) for the mirror; embedded `image/png` outputs from matplotlib/vl-convert render everywhere. Widget/JS outputs do not.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| pyBigWig 0.3.26 | Python 3.9–3.13 via manylinux_2_27/2_28 **x86_64 wheels** | No Windows/aarch64 wheels → platform marker mandatory; numpy support present (README-documented array `values`) — works under the matrix numpy 1.26.4/2.2.0 (C extension is numpy-version-tolerant via its own bindings; flagged MEDIUM — confirm in phase spike if the matrix pins bite) |
| langchain-ollama 1.1.0 | langchain-core ≥1.2.21,<2 (installed langchain 1.4.3 OK); pulls `ollama` 0.6.x client | Python ≥3.10 — matches requires-python |
| nbclient 0.10+ | Python ≥3.10; jupyter-client 8.x (installed 8.10.0) | No pinned ceiling needed; 0.11.0 is current |
| ipykernel 7.4.0 | Python ≥3.11 | Matrix is 3.11–3.13 → fine; 3.10 users resolve 6.x via uv (library baseline unaffected) |
| marimo 0.25.0 | Python ≥3.10 | `notebook` extra already `>=0.16.3`; runner installs latest — CLI surface verified on 0.25.0 |
| vl-convert-python 1.9.0 | ships as `altair[all]`/`[save]` extra content | Already installed; Rust binary wheel, no browser/node needed |
| pytest-timeout 2.4.0 marker | existing `--timeout=300` global | Per-test marker overrides upward for slow notebook tests (signal method on Linux) |

## Sources

- PyPI JSON API (authoritative for versions/wheels; fetched 2026-10-01): nbclient 0.11.0, nbmake 1.5.5, pyBigWig 0.3.26 (+ wheel file list), gffutils 0.14 (+ requires_dist), bcbio-gff 0.7.1, pyfaidx 0.9.0.4, pyfastx 2.3.1, ollama 0.6.3, marimo 0.25.0, ipykernel 7.4.0, vl-convert-python 1.9.0, papermill 2.7.0, langchain-ollama 1.1.0, pygenometracks 3.9 (GPL + matplotlib<3.9 pin + pybedtools dep), igv-notebook 0.6.2 (MIT, ipykernel/ipython/requests), anywidget 0.11.0; jbrowse-anywidget **absent from PyPI** — HIGH (deterministic API check)
- Local introspection of the installed venv (deterministic — HIGH): nbclient traits (`allow_errors=False`, `timeout=None`, `kernel_name=''`, `startup_timeout=60`) + `resources.metadata.path` handling (`client.py:431,535`) + in-place execute; marimo 0.25.0 CLI help (`export html` "Run a notebook…", `export session`, `run --headless`) + `App.run(defs=None)` signature; venv `share/jupyter/kernels/python3/kernel.json` content; vl-convert-python 1.9.0 importable; altair 6.3.0 `all`-extra contents; `pytest.mark.timeout` available; bedtools NOT on runner PATH; langchain-ollama NOT installed
- pyBigWig official README (github.com/deeptools/pyBigWig) — write API, zoom levels, validate/sorted-order, close() semantics — MEDIUM (single webfetch channel; API additionally matches upstream deeptools docs convention)
- altair saving-charts docs (altair-viz.github.io) — vl-convert requirement, ppi/scale_factor, deprecated altair_saver — MEDIUM
- matplotlib backends FAQ (matplotlib.org, stable) — MPLBACKEND / Agg / matplotlib.use precedence — MEDIUM
- jupyter_client kernels docs (jupyter-client.readthedocs.io, stable) — kernelspec search paths, kernel.json format — MEDIUM
- GFF3 spec v1.26 (Sequence Ontology Specifications, gff3.md) — columns, 1-based inclusive, escaping, directives — MEDIUM
- ollama official docs (docs.ollama.com/openai — verified; install/FAQ via docs.ollama.com/linux + /faq) — `curl -fsSL https://ollama.com/install.sh | sh`, systemd `ollama.service`, `OLLAMA_MODELS` drop-in pattern, `http://localhost:11434/v1/` endpoints — MEDIUM
- marimo docs (docs.marimo.io/guides/testing/, /pytest/) — reactive test cells; no `App.run` documented there (CLI behavior verified locally instead) — MEDIUM
- nbmake (github.com/treebeardtech/nbmake via websearch) — `--nbmake`, `--nbmake-timeout`, nbclient-based — MEDIUM
- pyfastx README (github.com/lmdu/pyfastx) — `fetch` 1-based inclusive, strand, indexing — MEDIUM; pyfaidx README (github.com/mdshw5/pyfaidx) — pure Python, 0-based slicing — MEDIUM
- pyGenomeTracks (github.com/deeptools/pyGenomeTracks README + readthedocs) — GPL-3.0, bedtools required since 3.5, pdf/png/svg outputs, "bed/gtf" track list — MEDIUM
- igv-notebook README (github.com/igvteam/igv-notebook) — 0.6.2/igv.js 3.1.4, to_svg "Jupyter Notebook only", JupyterLab local-path restriction — MEDIUM; igv.js GFF3+tabix support via igv.js wiki/issues (websearch) — MEDIUM
- jbrowse-anywidget README (github.com/GMOD/jbrowse-anywidget) — Prototype label, git-only install, puppeteer headless runner, "pytest never opens one", Apache-2.0 — MEDIUM

Open items for phase-level spikes (flagged, not blockers): pyBigWig under matrix numpy 1.26.4 on the runner; whether marimo demo apps containing `mo.ui` elements behave identically under `marimo export html` vs live `marimo run` (spot-check one app first).

---
*Stack research for: example-execution testing + PlantHelixSeek genomics showcases*
*Researched: 2026-10-01*
