# Phase 6: Model Registry & Showcase Data Curation - Research

**Researched:** 2026-10-02
**Domain:** Model-registry freeze + genomics ground-truth curation for the PlantHelixSeek showcase (dnallm generic loader route; TAIR10/PlantDHS data pipelines)
**Confidence:** HIGH — nearly every load-bearing claim was verified this session either by reading repo source, fetching the authoritative remote artifact (HF/ModelScope/GitHub/plantdhs.org), or running an empirical probe on the actual dev box (GB10, transformers 5.17.0). Residual uncertainty is confined to the Assumptions Log.

## Summary

Phase 6's two deliverables are now de-risked empirically. First, both PlantHelixSeek checkpoints **load and forward through dnallm's generic task-type route on the transformers 5.17 dev environment** — this was probed live on the GB10 box this session (download → `load_model_and_tokenizer` → forward, both the HuggingFace and the ModelScope sourcing routes, id2label override verified post-load). The single most important discovery refining the freeze mechanics: **neither checkpoint's `config.json` carries semantic label names.** The Anno config's `id2label` is the transformers-default placeholder `"0": "LABEL_0" … "16": "LABEL_16"`, and the CRE config has no `id2label` at all. The authoritative 17-BILOU index order lives in the upstream training script (`scripts/gene_annotation/train_token_cls.py` lines 78-96, `LABEL_NAMES`), which I fetched and quote verbatim below. The freeze script therefore reads the checkpoint config only to *prove the placeholder/absent pattern and the head shape* (17/2 columns), and freezes the semantic order from the upstream training definition — the "read config.id2label and write the yaml in that exact order" decision is honored in spirit (never hand-transcribed from the HF card, which is grouped and unindexed) but the mechanism must be label-source-aware or it would freeze `LABEL_0…LABEL_16`.

Second, the data side is fully mapped: the only network fetch (`TAIR10_DHSs.gff.zip` from `plantdhs.org/static/download/`) was downloaded and inspected this session (39,523 `DHSs` rows, TAIR `ChrN` naming, coordinate-sorted); the owner's local TAIR10 inputs were inspected (`TAIR10_chr1.fas` is uppercase, unmasked, `>Chr1`, 80-col; `TAIR10_GFF3_genes.gff` has gene/mRNA/exon/CDS/UTR rows with a trailing-semicolon `Parent=` quirk on 197,160 CDS rows); pyfastx's 1-based-inclusive `fetch()` semantics were verified against a fixture; bedtools jaccard's sorted-input requirement was re-confirmed; and scan timing was measured (CRE 500bp windows: 83 ms/window at batch 4 → a 200kb locus scans in ~5.4 min; Anno 8192bp windows: 7.9 s/window eager → a 200kb both-strand locus scan in ~12.4 min) making the truth-based-rank + model-verify-top-K selection design comfortably feasible inside an hour of GPU time.

**Primary recommendation:** Plan three workstreams in dependency order — (1) registry entries + freeze script + smoke tests (fast-leg structure test + slow-leg real-download smoke, both routes proven), (2) the shared normalization helper in `dnallm/utils/` with tiny committed fixtures, (3) `scripts/showcase/select_loci.py` curation run producing committed `.fas`/`.gff`/`.gff3` slices + `selection.md` — and watch the three planted traps documented below: the gitignored `*.fa` extension, the `attn_implementation: "eager"` throughput ceiling, and the label-source reality in the freeze script.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Locus selection methodology (SHOW-01)**
- One CRE locus + one Anno locus + one shared negative-control set: a flanking low-signal segment adjacent to the selected locus AND one intergenic region (two failure modes covered)
- Automated scan of the Chr1 front ~10–20Mb window; prefer regions with dense DHS signal plus complete gene annotation; the selection doc records coordinates, scan window, and thresholds
- Substantial-agreement selection thresholds: CRE — predicted peaks vs DHS peaks Jaccard ≥ 0.3 on the selected locus; Anno — ≥3 gene models with exon-level F1 ≥ 0.8; Phase-7 test assertions reuse these floors with tolerance bands
- Guarantee is "substantially consistent" (基本一致), verified during selection and asserted by later tests — never exact outputs

**Registry entries & smoke-load (REG-01..03)**
- CRE/Anno entries in the `finetuned:` section of `model_info.yaml`, following the existing zhangtaolab entry shape (name / describe / task_type / num_labels / label_names)
- Smoke-load on the SLOW leg (`@pytest.mark.slow` + timeout mark, real download); the fast leg gets a registry-entry structure assertion only (YAML parses, fields present, no network)
- Anno `label_names` frozen FROM the checkpoint at execution time (one-shot script reads `config.id2label` and writes the yaml in that exact order) — never hand-transcribed from the HF page
- Smoke environment: the dev box (transformers 5.17 — the REG-03 compat gate); record the version; on failure, one fallback attempt on 4.57 with evidence, else typed skip + upstream issue

**Data acquisition & landing shape (SHOW-01 mechanics)**
- Selection script `scripts/showcase/select_loci.py`; scratch dir under the showcase example dir, gitignored; script STARTS by probing `~/Downloads/` for already-downloaded inputs and copies them into scratch (owner downloaded `TAIR10_chr1.fas` 30.8MB and `TAIR10_GFF3_genes.gff` 44.1MB on 2026-10-02 — both present and reusable)
- Genome + TAIR10 GFF3 are LOCAL (reuse from `~/Downloads/`); the only remaining network fetch is PlantDHS `TAIR10_DHSs.gff.zip` (CRE truth) — direct plantdps.org URL with browser-UA retry; on failure the script prints a manual-placement instruction into `.scratch/` and exits non-zero
- Committed artifacts: per-locus FASTA segment + two GFF3 slices (DHS truth + TAIR10 annotation) + `selection.md` rationale doc; everything else stays in gitignored scratch
- Chromosome naming normalized to TAIR `Chr1` style (PlantDHS's native style); the normalization helper converts both directions (`1` ↔ `Chr1`)

**Carried forward (locked earlier — not re-decided)**
- ModelScope-first model sourcing; generic task-type route (NO special handler — research-verified einops-only hard dep, fla/flash_attn guarded)
- id2label equality asserted after load (dnallm rebuilds id2label from `label_names` — a permuted list silently permutes predictions)
- ≤200kb committed budget; prediction-vs-truth guarantee asserted by tests; intermediates gitignored; altair static rendering (Phase 7)

### Claude's Discretion
- Exact scan-window internals of select_loci.py (stride, candidate ranking), the normalization helper's API shape and module location, test file layout, YAML field phrasing within the established entry shape

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-01 | `model_info.yaml` finetuned-section entries for `PlantHelixSeek-CRE` (binary, num_labels 2) and `PlantHelixSeek-Anno` (token, num_labels 17) loadable through the existing generic task-type route (no special handler) | Exact entry shape extracted verbatim (existing entries + token-task precedent); dispatch-chain fall-through to `_load_model_by_task_type` verified by reading model.py:862-883; both models **empirically loaded and forwarded** through the route this session on transformers 5.17 |
| REG-02 | Anno `label_names` frozen to the checkpoint's exact `config.id2label` order; the execution test asserts `model.config.id2label` equality after load (silent-permutation guard) | Checkpoint config.id2label is placeholder `LABEL_i` (fetched verbatim) — freeze mechanism refined: upstream `LABEL_NAMES` (quoted verbatim) is the semantic source, checkpoint read proves the identity-placeholder pattern + `[17,512]` head; post-load equality assert verified working in the probe |
| REG-03 | Smoke-load of both checkpoints via the dnallm route succeeds on the transformers 5.x dev environment (compat risk gate) | **Empirically proven this session**: CRE and Anno load + forward via dnallm route on transformers 5.17.0/torch 2.11.0+cu130/GB10, HF route AND ModelScope route, with timings |
| SHOW-01 | Loci selection produces committed in-repo Arabidopsis fragments ≤200kb per region where predictions are substantially consistent with experimental truth, plus truth slices, selection-rationale doc, one negative-control locus; all download intermediates gitignored | PlantDHS file downloaded + format mapped; local TAIR10 inputs inspected; pyfastx slicing verified; scan timing measured (CRE 5.4 min/locus, Anno 12.4 min/locus); ranking data (DHS density, gene completeness) quantified; gitignore traps identified (`*.fa`!) |
| SHOW-02 | A shared, unit-tested coordinate/chrom-name normalization helper (0-based half-open ↔ 1-based closed; `Chr1` ↔ `1`) is used by all genomics code paths, with non-emptiness assertions against silent-empty results | Helper API + module location recommendation designed; the silently-empty failure mode is concretely documented (milestone Pitfall 11 + this session's chrom-name audit: both truth files already use `ChrN`, so the helper's value is robustness + the conversion contract); fixture design from milestone Pitfall 11 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Registry entries (`model_info.yaml`) | Packaged data (`dnallm/models/`) | — | Metadata registry shipped in the wheel; **no runtime consumer** (verified — see Pattern 1), correctness asserted by tests |
| Checkpoint loading / smoke proof | `dnallm/models/model.py` generic route | tests (slow leg) | The loader already forwards `trust_remote_code` and builds id2label from `label_names`; tests prove it, not modify it |
| Label-order freeze | One-shot script (repo `scripts/`) | checkpoint config + upstream training script as sources | Freeze is a curation act recorded in git, not runtime behavior |
| Locus ranking (truth-only) | `scripts/showcase/select_loci.py` | local truth files in `.scratch/` | Pure data curation; no dnallm runtime code involved |
| Prediction verification during selection | Selection script → dnallm public API | GPU (GB10) | Must use the SAME public route the showcase uses (`load_model_and_tokenizer` + tokenizer + forward) so floors measured here are the floors Phase 7 asserts |
| Coordinate/chrom normalization | `dnallm/utils/` (new small module) | unit tests | Shared by selection script, Phase-7 notebooks, Phase-8 tests — must be importable everywhere without path hacks, so it lives in the package, not under `tests/` or `example/` |
| Committed showcase data | `example/notebooks/plant_helixseek_*/data/` | docs mirror (Phase 7) | Follows milestone-research structure; fragments are committed inputs the nightly tests consume |
| Ground-truth acquisition | Selection script (probe `~/Downloads/` → plantdhs.org fetch) | manual placement fallback | One-time curation; tests never re-download (milestone Pitfall 12) |

## Standard Stack

**No new packages.** This phase installs nothing — the constraint "no new test frameworks" holds trivially. Everything used is already present (verified this session in `.venv`):

### Core (existing, verified this session)

| Tool | Version (verified) | Purpose in this phase |
|------|--------------------|----------------------|
| transformers | 5.17.0 (`.venv`) | REG-03 compat gate environment; `trust_remote_code` custom arch loading |
| torch | 2.11.0+cu130, CUDA available (`.venv`) | Selection-scan inference on GB10 (121.6 GiB visible) |
| pydantic (`TaskConfig`) | v2 (per pyproject `>=2.10.6`) | Config objects for smoke loads |
| pyfastx | 2.3.1 (`.venv`, dev extra) | FASTA region slicing — `fetch(chrom, (start, end))` **verified 1-based inclusive** this session |
| bedtools | v2.31.1 (`/home/linuxbrew/.linuxbrew/bin/bedtools`) | `jaccard` metric for CRE floor (CONTEXT: smoke verified 2026-10-02; sorted-input requirement re-confirmed this session) |
| PyYAML (`yaml.safe_load`) | existing core dep | Fast-leg registry structure test + freeze script |
| einops | 0.8.2 (`.venv`) | The remote code's only hard third-party dep (milestone-verified; import exercised by every probe load this session) |

### Supporting

| Tool | Purpose | When to use |
|------|---------|-------------|
| stdlib `zipfile`/`urllib` or `curl` via subprocess | PlantDHS zip fetch + magic-byte-ish validation (zip magic `PK`) | Inside select_loci.py with browser-UA retry per CONTEXT |
| `huggingface_hub.snapshot_download` | Warm-cache downloads in the slow-leg smoke test (route proven this session) | Only inside slow-marked tests / freeze script |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pyfastx | BioPython SeqIO slice | BioPython loads whole record; pyfastx is indexed random access — already a dev extra, verified semantics |
| bedtools jaccard | pure-Python interval intersection | CONTEXT decision: bedtools primary (field standard), pure-Python as cross-check; both viable — keep bedtools since verified installed |
| gffutils / BCBio | stdlib GFF3 parsing | Milestone research already decided: ~60-line strict parser beats the dependencies at ≤200kb scale |
| upstream Viterbi decode (port) | simple argmax BILOU collapse | See Open Question 2 — recommended simple decode for floor calibration; viterbi port adds a 74KB-file port + `transition_probs.npz` provenance question |

## Package Legitimacy Audit

**This phase installs no external packages** — no `pyproject.toml` dependency changes are in scope (constraint: "no new test frameworks"; nothing else is needed). All tools above are pre-existing project dependencies or system tools whose presence was verified this session (`pyfastx 2.3.1`, `einops 0.8.2`, `bedtools v2.31.1`, `.venv` transformers/torch). Therefore:

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| *(none — no installs)* | — | — | — | — | — | N/A |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

## Architecture Patterns

### System Architecture Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │ WORKSTREAM 1: REGISTRY (REG-01..03)          │
                    │                                              │
  upstream GitHub ──▶ one-shot freeze script (scripts/showcase/   │
  train_token_cls.py  freeze_registry.py or inside select flow)   │
  LABEL_NAMES {0..16} │  ├─ reads checkpoint config.json           │
                     │  │    (prove LABEL_i placeholder pattern,  │
                     │  │     17 entries, [17,512] head)          │
                     │  └─ appends 2 entries to model_info.yaml   │
  HF/ModelScope ─────▶  finetuned: (packaged data, no runtime     │
  zhangtaolab/         consumer)                                  │
  PlantHelixSeek-{CRE,Anno}                                       │
        │                     ┌───────────────────────────────────┘
        ▼                     ▼
  ┌──────────────┐   ┌─────────────────────┐
  │ SLOW leg     │   │ FAST leg           │
  │ smoke test   │   │ registry structure │
  │ (real DL,    │   │ test (yaml parse,  │
  │ 1800s mark)  │   │ fields, 17-order)  │
  └──────┬───────┘   └─────────────────────┘
         │ load_model_and_tokenizer(repo_id, TaskConfig,
         │   source="modelscope"|"huggingface")
         ▼   → generic _load_model_by_task_type (NO handler)
              → assert model.config.id2label == frozen dict
              → forward (CRE 500bp / Anno 8192bp)

  ┌────────────────────────────────────────────────────────────┐
  │ WORKSTREAM 2+3: DATA CURATION (SHOW-01, SHOW-02)           │
  │                                                           │
  ~/Downloads/TAIR10_chr1.fas ─┐                              │
  ~/Downloads/TAIR10_GFF3_.. ──┤ probe-and-copy (FIRST)       │
  plantdhs.org/static/download/│                             │
    TAIR10_DHSs.gff.zip ───────┘ fetch w/ UA retry → .scratch/│
         │                     (gitignored)                   │
         ▼                                                   │
  scripts/showcase/select_loci.py                            │
   1. RANK (no model): 200kb tiles in Chr1:0–20Mb by          │
      DHS density + gene completeness (truth files only)      │
   2. VERIFY (model, top-K): CRE 500/50 scan → mean-class-1   │
      bin track → mean+1.5σ peaks → bedtools jaccard ≥0.3?    │
      Anno 8192/4096 both strands → BILOU argmax → CDS spans  │
      → exon-F1 ≥0.8 on ≥3 genes?                             │
   3. NEGATIVE: flanking low-DHS segment + intergenic region  │
      (0 genes, 0 DHS) — model must predict ≈ nothing         │
         │                                                   │
         ▼  committed (≤200kb seq per region)                 │
  example/notebooks/plant_helixseek_*/data/                   │
    <locus>.fas (NOT .fa — gitignored!)                       │
    TAIR10_DHSs_<locus>.gff / TAIR10_GFF3_<locus>.gff3        │
    selection.md (coords, scan window, floors, observed vals) │
         │                                                   │
  dnallm/utils/genomic_coords.py (SHOW-02) ◀── used by every  │
    chrom-name + coordinate conversion above + unit-tested    │
    on tiny committed fixtures                                │
  └───────────────────────────────────────────────────────────┘
         ▼ consumed later by Phase 7 notebooks & Phase 8 tests
```

### Recommended Project Structure

```
dnallm/utils/genomic_coords.py            # NEW — SHOW-02 helper (ships in wheel)
tests/utils/test_genomic_coords.py        # NEW — unit tests + tiny fixtures
                                          #   (fixture FASTA/GFF rows inline or under
                                          #    tests/utils/fixtures/)
dnallm/models/model_info.yaml             # MODIFIED — 2 finetuned entries appended
scripts/showcase/select_loci.py           # NEW — curation script (probe ~/Downloads,
                                          #   fetch plantdhs zip, rank, verify, emit)
tests/models/test_plant_helixseek_registry.py   # NEW — fast-leg structure assertions
                                          #   (layout name is planner's discretion;
                                          #    tests/ mirrors package layout)
tests/models/test_plant_helixseek_smoke.py      # NEW — slow-leg smoke (real download,
                                          #   @pytest.mark.slow + timeout(1800))
example/notebooks/plant_helixseek_cre/data/     # NEW — committed CRE locus fragments
example/notebooks/plant_helixseek_anno/data/    # NEW — committed Anno locus fragments
example/notebooks/plant_helixseek_*/.gitignore  # NEW — .scratch/ + generated outputs
                                          #   (+ negation patterns if .fa used; see Pitfall 3)
selection.md location: see Open Question 5 (shared negative control is shared data)
```

### Pattern 1: Registry-first onboarding — the yaml is data, tests are the validator

**What:** `model_info.yaml` is packaged registry data with **no runtime consumer** — verified this session: a repo-wide grep for `model_info` across `dnallm/`, `ui/`, `cli/`, `scripts/` finds only MCP `get_model_info` tooling (unrelated) and a comment at `dnallm/mcp/server.py:673` ("This would integrate with model_info.yaml"); the hard-coded `MODEL_INFO` dict at `dnallm/models/modeling_auto.py:43` is a separate pretrained-models structure that does not read the yaml. [VERIFIED: repo-wide grep + Read of modeling_auto.py:43 context, 2026-10-02]

**Consequences for the plan:**
- The yaml edit cannot break runtime — the fast-leg structure test parses the file directly with `yaml.safe_load` (via `Path(dnallm.__file__).parent / "models" / "model_info.yaml"` or `importlib.resources`).
- REG-01's "loadable through the generic route" is proven by the smoke test using the repo id *from the entry*, not by any loader-registry integration.
- Entry shape to follow verbatim [VERIFIED: dnallm/models/model_info.yaml:164-172, Read this session]:

```yaml
finetuned:
  - name: "Plant DNABERT BPE promoter"
    model: "zhangtaolab/plant-dnabert-BPE-promoter"
    task:
      describe: "Predict whether a DNA sequence is a core promoter in plants by using Plant DNABERT model with BPE tokenizer."
      task_type: "binary"
      num_labels: 2
      label_names: ["Not promoter", "Core promoter"]
      threshold: 0.5
```

  A token-task precedent with a BILOU list exists (single-quoted inline list, 7 labels) [VERIFIED: model_info.yaml:1445-1447 grep-located]: `label_names: ['O','B-Intron', 'I-Intron', 'B-tRNA', 'I-tRNA', 'B-anti', 'I-anti']`. `base_model:` is an optional field used by PlantCAD2 entries (model_info.yaml:1451) — including it (`zhangtaolab/PlantHelixSeek`) follows that precedent; the milestone research's draft entries include it. Note existing entries use `task_type: "classification"` in places, which is *not* a valid TaskConfig enum value — the yaml has no validator; our entries must use `binary` / `token` (both valid TaskConfig values) per CONTEXT.

- **Mutating the yaml:** the `finetuned:` section runs to EOF (its last entries are the PlantCAD2 trio). Safest edit is a surgical text append of two formatted blocks — not parse-modify-dump, which would reformat all 1,500+ lines. [VERIFIED: tail of model_info.yaml Read/grep this session]

### Pattern 2: The label freeze — checkpoint config proves constraints, upstream training code supplies semantics

**The empirical reality (fetched verbatim this session):**

- Anno `config.json` `id2label` is placeholders [VERIFIED: huggingface.co/zhangtaolab/PlantHelixSeek-Anno/raw/main/config.json, fetched 2026-10-02]:

```json
"id2label": {
    "0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2", "3": "LABEL_3", "4": "LABEL_4",
    "5": "LABEL_5", "6": "LABEL_6", "7": "LABEL_7", "8": "LABEL_8", "9": "LABEL_9",
    "10": "LABEL_10", "11": "LABEL_11", "12": "LABEL_12", "13": "LABEL_13",
    "14": "LABEL_14", "15": "LABEL_15", "16": "LABEL_16"
}
```

- CRE `config.json` has **no** `id2label`/`label2id` keys at all [VERIFIED: same endpoint for -CRE, fetched 2026-10-02 — keys absent between `"hidden_size": 512` and `"initializer_range": 0.02` where Anno carries them].
- Head shapes from safetensors headers (range-requested, no full download) [VERIFIED: model.safetensors header parse this session]: Anno `classifier.weight [17, 512]`, `classifier.bias [17]`; CRE `score.weight [2, 512]`.
- The HF Anno model card documents label *groups* (a table of `O` / CDS / Intron / 5'UTR / 3'UTR tag groups) but **no index order** [VERIFIED: README.md fetched this session] — insufficient as a freeze source, exactly as the CONTEXT decision anticipated ("never hand-transcribed from the HF page").
- The authoritative indexed order — the training script's definition, where classifier column *i* is `LABEL_NAMES[i]` (class weights are built as `[train_counts[LABEL_NAMES[i]] for i in range(NUM_LABELS)]`) [VERIFIED: raw.githubusercontent.com/zhangtaolab/PlantHelixSeek/main/scripts/gene_annotation/train_token_cls.py:78-99 + :280, fetched and Read this session]:

```python
LABEL_NAMES: dict[int, str] = {
    0: "O",
    1: "B-CDS",
    2: "I-CDS",
    3: "L-CDS",
    4: "U-CDS",
    5: "B-INTRON",
    6: "I-INTRON",
    7: "L-INTRON",
    8: "U-INTRON",
    9: "B-UTR5",
    10: "I-UTR5",
    11: "L-UTR5",
    12: "U-UTR5",
    13: "B-UTR3",
    14: "I-UTR3",
    15: "L-UTR3",
    16: "U-UTR3",
}

NUM_LABELS = 17
LABEL_PAD = -100
```

**Refined freeze-script contract ( honoring the locked decision's intent):**
1. Load the checkpoint config (the model itself via the dnallm route, warm cache, or `AutoConfig.from_pretrained(..., trust_remote_code=True)` — both equivalent for reading `id2label`; the probe used the full dnallm load).
2. Assert the constraint facts: Anno config `id2label` matches the identity-placeholder pattern (`id2label[str(i)] == f"LABEL_{i}"` for i in 0..16) and has 17 entries; classifier head is `[17, 512]`; CRE config has no id2label and head is `score.weight [2, 512]`. These assertions *prove the config carries no ordering information*, which is precisely why the upstream training script is the semantic source and why a post-load behavioral backstop matters.
3. Write the yaml entries with the frozen lists (Anno: the 17 names above in order; CRE: `["Not CRE", "CRE"]` — index 1 = CRE verified from the upstream CRE README: bin score is "arithmetic mean of **class-1 probabilities**" [CITED: upstream scripts/cis_regulatory/README.md line 55]).
4. Record provenance in the entry comment or selection.md: upstream file, line numbers, checkpoint commit shas (captured this session: CRE `7093de3baf64bf59ac147be3971482a13238aabd`, Anno `6d39386ab562b2a9a3d3581ebe28e235383c8d3c` [VERIFIED: HF API `?blobs=true` response this session]) — this doubles as the `trust_remote_code` provenance record milestone Pitfall 8 asks for.

**Why the post-load assert still works (REG-02's test target):** dnallm rebuilds id2label from our list and passes it into `from_pretrained`, which overrides the checkpoint's mapping. Verified in source [VERIFIED: dnallm/models/model.py:495-512, Read this session]:

```python
    label_names = task_config.label_names
    if label_names is None:
        # Default empty mappings for tasks without labels
        return {}, {}
    id2label = dict(enumerate(label_names))
    label2id = {label: i for i, label in enumerate(label_names)}
    return id2label, label2id
```

and the token route passes them through [VERIFIED: dnallm/models/model.py:610-617]:

```python
    elif task_type == "token":
        model = modules["AutoModelForTokenClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **model_load_kwargs,
        )
```

Empirically confirmed after load this session: Anno `id2label[1] == 'B-CDS'` with the full dict equal to `dict(enumerate(labels))`; CRE `id2label == {0: 'Not CRE', 1: 'CRE'}`. A permuted yaml list would produce a permuted post-load dict → the equality assert catches it. The *behavioral* backstop (truth agreement on the selected locus) is what actually catches a wrong-vs-training order, since the checkpoint config cannot.

**TaskConfig requirements for the smoke test** [VERIFIED: dnallm/configuration/configs.py:97-131, Read this session]: `task_type` pattern includes `binary` and `token`; `num_labels: int | None = Field(default=2)`; `label_names: list[str] | None = None`; `threshold: float = Field(default=0.5)`. Only `binary` gets default label_names (`["negative", "positive"]`) — `token` does **not**, so the Anno config/yaml must always carry the 17 names or `_create_label_mappings` returns `{}`. Also `_safe_num_labels` raises `ValueError` if `num_labels is None` for token [VERIFIED: model.py:671-678]:

```python
    if num_labels is None and task_type in [
        "binary",
        "multiclass",
        "multilabel",
        "regression",
        "token",
    ]:
        raise ValueError(f"num_labels is required for task type '{task_type}' but is None")
```

### Pattern 3: Smoke-load reality — both sourcing routes proven, timings measured

**Empirical probe results (this session, dev box: GB10 121.6 GiB visible to torch, transformers 5.17.0, torch 2.11.0+cu130, CUDA available):** [VERIFIED: live probe run 2026-10-02, script + logs in /tmp — re-runnable]

| Stage | Result | Time | Notes |
|---|---|---|---|
| HF download CRE | OK | 84.9 s | 1,883 MB safetensors ≈ 22 MB/s |
| HF download Anno | OK | 86.1 s | 1,883 MB |
| dnallm-route load CRE (`source="huggingface"`) | OK | 12.1 s warm | `HelixSeekForSequenceClassification`, `id2label={0:'Not CRE',1:'CRE'}`, on `cuda:0` |
| CRE forward 500bp bs=1 | OK | 0.59 s | peak 3.83 GB |
| CRE batch ladder 500bp | bs=4: **83 ms/win, 2.5 GB** | — | bs=8: 112 ms/4.1GB; bs=16: 169 ms/10.5GB; bs=32: 285 ms/35.8GB; bs=48: 400 ms/77.7GB; **bs=64: CUDA OOM** (30.76 GiB alloc attempt in eager delta-attention) |
| dnallm-route load Anno | OK | 12.4 s warm | `num_labels=17`, `id2label[1]='B-CDS'` |
| Anno forward 8192bp bs=1 (eager) | OK | 7.91 s | peak 12.99 GB, logits `(1, 8194, 17)` |
| ModelScope route (`source="modelscope"`) CRE: download+load+forward | OK | 122 s total | caches under `~/.cache/modelscope/hub/models/zhangtaolab/PlantHelixSeek-CRE` |

Implications:
- **REG-03 is de-risked**: the compat gate (transformers 5.x + this custom code) passes on the dev box, both routes. The locked ModelScope-first decision is executable — the ModelScope repos carry all remote-code files (file listing verified via ModelScope API this session: `attention.py`, `cache.py`, `config.json`, `configuration.py`, `layers.py`, `mlp.py`, `model.py`, `model.safetensors` 1883.00 MB, tokenizer files, `utils.py`) [VERIFIED: modelscope.cn/api/v1/models/zhangtaolab/PlantHelixSeek-CRE/repo/files].
- **Eager attention is the throughput tax, not a blocker.** dnallm forces it [VERIFIED: dnallm/models/model.py:556-559]: `model_load_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}`. The shipped remote code *does* implement sdpa (`HelixSeek_ATTENTION_CLASSES = {"eager": HelixSeekAttention, "flash_attention_2": HelixSeekFlashAttention2, "sdpa": HelixSeekSdpaAttention}` [VERIFIED: attention.py fetched from the Anno repo this session]; remote `model.py` `_supports_sdpa = True` and reads `config._attn_implementation`) — so *upstream's* scripts (which don't force eager and default to sdpa) can use batch 256 for CRE. Through the dnallm route, use **small batches**: bs=4 is optimal for CRE at 500bp (83 ms/window — throughput *degrades* superlinearly per window as batch grows). Do not plan bs>16 through the dnallm route at fp32.
- Scan arithmetic for the selection script: CRE 200kb locus at stride 50 = 3,901 windows ≈ **5.4 min** (bs=4). Anno 200kb locus: 47 windows/strand × 2 ≈ 94 forwards ≈ **12.4 min**. Verifying K=5 candidates CRE-only ≈ 27 min; Anno only on CRE-passing candidates (1-2) ≈ 12-25 min. Total GPU budget ≈ 40-60 min — comfortably feasible.
- Benign runtime warnings observed (do not let tests assert on stderr): `HelixSeek requires an initialized cache to return a cache. None was provided…`, transformers-5 `use_return_dict is deprecated` (ModelScope route).
- Slow-leg test budget: cold-cache worst case = 2×85 s downloads + 2×13 s loads + forwards ≈ 5 min; use `@pytest.mark.timeout(1800)` (existing ladder: 900 downloads / 1800 inference class / 3600-7200 finetune) [VERIFIED: grep of tests/ timeout marks this session].

### Pattern 4: Data acquisition — verified formats, one network fetch, magic-byte validation

**Owner-local inputs (probe first, per CONTEXT):** [VERIFIED: files inspected this session]
- `~/Downloads/TAIR10_chr1.fas` — 30,812,908 bytes; header `>Chr1 CHROMOSOME dumped from ADB: Jun/20/09 14:53; last updated: 2009-02-02` (id = first token `Chr1`); 80-char lines; **entirely uppercase** (full-file census: A 9,709,674 / T 9,697,113 / C 5,435,374 / G 5,421,151 / N 163,958 / IUPAC W,Y,M,K,R,S ≈ 401 total) — unmasked, so `.upper()` is a no-op here but still required defensively; rare IUPAC letters tokenize to `<unk>` (vocab has only `ACGTN` + 6 special tokens; tokenizer is `EsmTokenizer` per `tokenizer_config.json`) — negligible (~1 in 76 kb) and consistent with upstream (both upstream inference scripts call `.upper()` and do nothing else).
- `~/Downloads/TAIR10_GFF3_genes.gff` — 44,139,005 bytes, 590,264 lines, chroms `Chr1..Chr5, ChrC, ChrM`; feature census (first 200k lines): exon 73,249 / CDS 66,739 / mRNA 11,818 / protein 11,818 / gene 9,612 / five_prime_UTR 11,600 / three_prime_UTR 10,189 + ncRNA/TE/pseudogene types. ID/Parent convention: `ID=AT1G01010` (gene) → `ID=AT1G01010.1;Parent=AT1G01010` (mRNA) → `Parent=AT1G01010.1` (exon/CDS/UTR, no ID); CDS rows list two parents `Parent=AT1G01010.1,AT1G01010.1-Protein;`. **Parser hazard: 197,160 CDS rows end with a trailing `;` after the Parent value** — attribute splitting must drop empty tail segments. Coordinates 1-based closed; the `chromosome` feature row for Chr1 gives length 30,427,671.

**The one network fetch:** [VERIFIED: downloaded and inspected this session]
- URL: `https://plantdhs.org/static/download/TAIR10_DHSs.gff.zip` (scraped from the `/Download` page href `/static/download/TAIR10_DHSs.gff.zip`; the page serves static HTML to curl — a browser-UA header was sent and worked; note the CONTEXT typo "plantdps.org" — the real domain is **plantdhs.org**).
- Zip: 493,051 bytes → single `TAIR10_DHSs.gff`, 3,007,760 bytes, internal date 2015-06-02 (a decade-stable static resource; direct file path, not an SPA route — but still validate: zip magic `PK\x03\x04` then GFF text, per milestone Pitfall 12).
- GFF content: **no `#` header lines at all**; tab-delimited; source `jianglab`; feature type `DHSs` (with the s); chroms `Chr1..Chr5` (already TAIR style — same convention as TAIR10 GFF3 and the FASTA, so in-repo data needs no renaming; the `1 ↔ Chr1` helper direction exists for user-supplied Ensembl-style inputs); sorted by coordinate within each chrom (file starts with Chr5); strand/score/frame are `.`; `Name=TAIR10_Chr1:3064-3280` duplicates coords in the name.
- Density numbers driving locus ranking: 39,523 DHS rows total; Chr1 = 10,278; **Chr1 front 10 Mb = 3,936; front 20 Mb = 6,314**; width stats (front 20 Mb): min 50 / median 322 / max 3,366 bp. Gene side: Chr1 has 7,509 genes total, 4,627 genes and 5,752 mRNAs in the front 20 Mb — a 200 kb gene-dense tile contains ~40-50 genes, far above the "≥3 gene models" floor.

**Committed-slice sizes (quantified on a real gene-dense 200 kb example, Chr1:2.0-2.2 Mb):** DHS slice = 73 rows / 5,402 bytes; TAIR10 slice = 1,384 rows / 101,331 bytes; FASTA at 80-col ≈ 202.5 KB per 200 kb. **Full-200 kb locus ≈ 308 KB committed bytes**; the negative-control flanking segment + intergenic region add smaller FASTA+slice files. See Open Question 1 for the budget-interpretation question (kb-of-sequence vs KB-of-bytes).

### Pattern 5: Selection script design — truth-rank, model-verify, negative-control

**Rank (no model):** tile the scan window (Chr1:0-20 Mb per CONTEXT "front ~10-20Mb") into candidate regions (200 kb tiles, optionally strided 50 kb for boundary sensitivity — discretion), score each by (a) DHS site density (rows overlapping the tile; front-20 Mb mean ≈ 0.32 sites/kb, gene-dense tiles run higher) and (b) gene-model completeness (mRNAs with full exon+CDS+UTR feature sets inside the tile). Rank descending; take top K (K=3-5 — each CRE verification is ~5.4 min, so K is cheap).

**Verify (model, through the dnallm route only):**
- CRE: replicate the upstream contract on the candidate — window 500 / stride 50 / bin 50; each bin score = arithmetic mean of class-1 probabilities over covering windows (≤10 windows cover a bin at 500/50) [CITED: upstream cis_regulatory README lines 53-56]; peak-call with `mean + 1.5σ` (upstream defaults: `threshold_method mean_std`, `threshold_params 1.5`, `min_length 50`, `max_length 5000`, `min_score 0.6`, `merge_gap 50`) [CITED: upstream README call-peaks table + verified in call_peaks_from_bigwig.py this session]; write BEDs (0-based half-open!), `sort -k1,1 -k2,2n`, then `bedtools jaccard` — output columns `intersection union jaccard n_intersections`, parse the 3rd [VERIFIED: bedtools smoke this session; unsorted input errors with "Sorted input specified, but the file … out of order"]. Floor: **Jaccard ≥ 0.3** (CONTEXT).
- Anno: window 8192 / stride 4096, both strands, upstream middle-region stitching (margin = (8192-4096)/2 = 2048 dropped per side per window; "stitched directly (no probability averaging)" [CITED: upstream gene_annotation README line 56]); token alignment — logits carry BOS/EOS (`(1, 8194, 17)` measured), take `logits[:, 1:window+1, :]` [VERIFIED: upstream predict_genome_multigpu.py:308 this session]; reverse-strand labels must swap B↔L: `_B_SWAP_L = np.array([0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13, 16], dtype=np.int32)` [VERIFIED: upstream predict_genome_multigpu.py:97-101, fetched this session]. Decode: argmax per base → collapse CDS BILOU spans → predicted CDS segments; compare against truth CDS features (`Parent=mRNA`) — recommended match rule: reciprocal overlap ≥ 0.5 on the same strand; pooled exon-level F1 across the locus; floor: **≥3 gene models with exon-F1 ≥ 0.8** (CONTEXT). Pin the exact F1 definition in `selection.md` so Phase 7 reuses it verbatim (see Open Question 2 for the decode-choice dependency).
- Only candidates passing CRE get Anno verification (saves ~12 min per rejected candidate).

**Negative control (CONTEXT: two failure modes):**
1. Flanking low-signal segment: within ±100 kb of the selected locus, pick the 10-20 kb window with the lowest DHS count. Expected: the model calls ≈ no peaks there.
2. Intergenic region: a window with 0 gene rows in TAIR10 (pericentromeric TE-rich territory around Chr1 ~14-18 Mb is inside the scan window and gene-poor; or any intergenic gap). Expected: ≈ no predicted genic bases.
- Assertion shape (calibrated at selection, recorded with observed values): predicted-peak base fraction (CRE) / predicted genic base fraction (Anno) on the negative locus ≈ 0 with a small tolerance band — *not* bedtools-jaccard-against-empty (jaccard vs an empty truth BED is degenerate 0/0). This is the Pitfall-13 device: proves the metric distinguishes signal from absence.

### Pattern 6: The SHOW-02 normalization helper

**Location recommendation:** `dnallm/utils/genomic_coords.py` (new module; `dnallm/utils/sequence.py` holds sequence-string ops — coordinates are a distinct concern). Rationale: it must be importable by (a) `scripts/showcase/select_loci.py`, (b) Phase-7 notebooks (which import only `dnallm*` in their sandboxed kernels), and (c) Phase-8 tests — an example- or tests-local helper fails (b). Cost: ships in the wheel and enters the coverage denominator — fine, SHOW-02 demands unit tests anyway, and a ~60-line pure module will sit near 100%.

**API sketch (discretion area — planner finalizes):**

```python
def normalize_chrom(name: str, *, style: str = "tair") -> str:
    """'1'/'chr1'/'Chr1' -> 'Chr1' (tair) or '1' (ensembl); ChrC/ChrM pass through."""

def gff1_to_half_open(start: int, end: int) -> tuple[int, int]:
    """GFF3 1-based closed -> BED 0-based half-open: (start - 1, end)."""

def half_open_to_gff1(start0: int, end0: int) -> tuple[int, int]:
    """BED 0-based half-open -> GFF3 1-based closed: (start0 + 1, end0)."""

def fetch_sequence(fa, chrom: str, start: int, end: int, *, uppercase: bool = True) -> str:
    """pyfastx fetch + chrom normalization + non-empty assertion.
    Raises ValueError on unknown chrom or empty result (the silent-empty guard)."""

def parse_gff_attributes(attrs: str) -> dict[str, list[str]]:
    """GFF3 column-9 parser that tolerates the trailing ';' and comma-joined Parents."""

def slice_gff_rows(rows, chrom: str, start: int, end: int, *, require_nonempty: bool = False):
    """Filter GFF/GFF3 rows to a 1-based closed locus; optional non-empty assertion."""
```

Non-emptiness is enforced *inside* the helpers (raise `ValueError`) plus caller-side asserts on any extracted array — that is SHOW-02's "non-emptiness assertions against silent-empty results" made structural. `pyfastx` fetch semantics verified this session against a fixture: `fa.fetch('Chr1', (10, 20))` returns 11 bases equal to `seq[9:20]` — 1-based inclusive, identical to GFF3. [VERIFIED: pyfastx 2.3.1 probe this session]

**Fixtures:** tiny committed FASTA (200 bp) + a handful of GFF3 rows (one DHS, one exon/CDS pair) with unit tests covering: both chrom-name directions, organelle pass-through, both coordinate conversions round-trip, the trailing-semicolon attribute parse, the empty-result `ValueError`, and a wrong-chrom-name lookup raising (the Pitfall-11 scenario).

### Anti-Patterns to Avoid

- **Freezing `config.id2label` strings verbatim** — they are `LABEL_0…LABEL_16`; the freeze must take semantics from the upstream `LABEL_NAMES` and use the checkpoint read only for constraint proof (Pattern 2).
- **Committing fragments as `.fa`** — globally gitignored (Pitfall 3 below); use `.fas` or a negation pattern.
- **Big batches through the dnallm route** — eager attention OOMs at bs=64 (CRE 500bp) and per-window throughput *worsens* with batch; plan bs=4.
- **bedtools jaccard on unsorted BEDs** — errors; sort first (verified this session).
- **Measuring Anno floors with a different decode than Phase 7 will assert** — the floor and the decode must be frozen together in `selection.md`.
- **Re-deriving architecture here** — milestone `.planning/research/ARCHITECTURE.md` already owns the two-leg test design, sandboxing, models.lock, mirror handling; this phase plugs into it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FASTA random-access slicing | full-file readers / BioPython load | `pyfastx` `fetch()` (1-based inclusive, verified) | indexed access on a 30.8 MB chromosome; semantics already proven identical to GFF3 coords |
| Interval-overlap metric | custom overlap math for the floor | `bedtools jaccard` (v2.31.1 installed) + pure-Python cross-check only | field-standard; sorted-input contract verified |
| Peak calling logic | novel thresholding | replicate upstream `mean + 1.5σ` + merge_gap 50 + min_length 50 (+ min_score 0.6) defaults | SHOW-03 already pins this contract; upstream parameters documented and verified |
| Coordinate/chrom conversion | scattered `-1` adjustments | the SHOW-02 helper (one place, unit-tested) | Pitfall 11: three conventions meet; the failure mode is silence |
| GFF3 attribute parsing | naive `split(';')` without empty-tail handling | helper `parse_gff_attributes` | 197,160 CDS rows carry trailing `;` (quantified this session) |
| Model loading | any special-handler / direct-transformers path for these models | dnallm generic route (`_load_model_by_task_type`) | locked decision; empirically proven this session; dispatch chain verified fall-through (model.py:862-883 — crossdna/dnabert2 substring guards don't match `planthelixseek`) |

**Key insight:** every conversion or metric in this phase has an upstream or field-standard definition already documented — Phase 6's job is *transcription with provenance + tests*, not design.

## Common Pitfalls

### Pitfall 1: The label-source trap (REG-02)
**What goes wrong:** A freeze script that literally copies `config.id2label` writes `LABEL_0…LABEL_16` into the yaml — order-correct but semantically void; Phase 7's BILOU decode then has no tag semantics, or someone "fixes" the names later in a different order and silently permutes predictions.
**Why:** The checkpoint predates semantic naming (upstream trains and infers purely in index space — their own predict script never passes id2label).
**How to avoid:** Pattern 2's contract: placeholder-pattern assert + upstream `LABEL_NAMES` freeze + post-load equality + behavioral backstop.
**Warning signs:** yaml containing `LABEL_`; a smoke test asserting against `LABEL_i`.

### Pitfall 2: Eager-attention memory ceiling through the dnallm route
**What goes wrong:** CRE scan at batch ≥64 OOMs (measured: 30.76 GiB single-allocation failure in eager delta-attention; bs=48 needs 77.7 GB peak); conversely, copying upstream's `--batch_size 256` default (which assumes sdpa) into a dnallm-route script crashes.
**Why:** `_load_model_by_task_type` forces `attn_implementation: "eager"` (model.py:556-559); the remote code implements sdpa but only selects it when config asks.
**How to avoid:** batch 4 for CRE 500bp windows (measured optimum, 2.5 GB); Anno 8192 at bs=1 (12.99 GB peak). If Phase 7 ever needs more headroom, that is a *documented tradeoff decision*, not an incidental change.
**Warning signs:** OOM inside `attention.py … attn = decay_weights * qk`; throughput decreasing as batch grows.

### Pitfall 3: Committed fragments silently gitignored
**What goes wrong:** `git add example/notebooks/plant_helixseek_cre/data/locus.fa` adds nothing — root `.gitignore` lines 69-70 ignore `*.fasta` and `*.fa` globally; the commit looks complete, the checkout is missing the showcase data.
**Why:** blanket data-file ignores predate the committed-fragments design.
**How to avoid (verified via `git check-ignore` this session):** commit fragments as `.fas` (NOT ignored — `test.fas` produced no check-ignore match; consistent with `TAIR10_chr1.fas` itself), or add a negation (`!data/*.fa`) in the per-dir `.gitignore` (deeper .gitignore wins). `.gff`/`.gff3`/`.md` are not ignored (verified). Conversely the scratch dir needs an explicit `.scratch/` entry because `.fas` and `.gff` copies inside it are NOT covered by root patterns.
**Warning signs:** `git status` clean right after "committing" fragments; nightly tests failing with file-not-found on a fresh checkout.

### Pitfall 4: GFF3 attribute parsing on TAIR10's trailing semicolons
**What goes wrong:** `dict(kv.split('=') for kv in attrs.split(';'))` yields `{'Parent': ['…'], '': ['']}`-style ghosts or drops the last Parent on 197,160 CDS rows.
**How to avoid:** the helper's attribute parser (strip empties; split Parent on comma).
**Warning signs:** exon counts off by exactly the CDS-row count; empty attribute keys.

### Pitfall 5: strand handling in Anno verification
**What goes wrong:** reverse-strand windows decoded without the B↔L swap produce garbage spans; or logits indexed `[:, 0:8192]` (BOS offset) shifting every coordinate by one.
**Why:** logits include BOS/EOS (`(1, 8194, 17)` measured); BILOU tags are direction-relative.
**How to avoid:** follow upstream exactly: `logits[:, 1:window+1, :]` and `_B_SWAP_L` on reverse-strand labels (both quoted in Pattern 5).
**Warning signs:** reverse-strand genes scoring ~0 F1 while forward-strand genes pass; coordinates off by one base.

### Pitfall 6: PlantDHS URL domain and zip validation
**What goes wrong:** the CONTEXT spells the domain "plantdps.org" — the real host is **plantdhs.org**; also, an SPA/error page saved as the zip crashes later with a confusing zipfile error.
**How to avoid:** exact verified URL `https://plantdhs.org/static/download/TAIR10_DHSs.gff.zip`, browser-UA retry, validate zip magic (`PK\x03\x04`) and that the inner GFF starts with `Chr` rows, per the CONTEXT failure contract (print manual-placement instruction into `.scratch/`, exit non-zero).
**Warning signs:** downloaded file ≈ HTML-sized; `unzip -l` failing.

### Pitfall 7: Scope creep into Phase 7/8
**What goes wrong:** the selection script growing notebooks, track rendering, or models.lock edits.
**How to avoid:** Phase 6 commits *data + rationale + helper + registry + smoke tests*; notebooks are Phase 7; models.lock/CI wiring is Phase 8 (CI-04). Registry smoke via ModelScope route is Phase 6 (proven feasible); lock entries are not.

## Code Examples

### Registry entries (ready for the freeze script to emit — provenance in comments)

```python
# Source of label order: zhangtaolab/PlantHelixSeek GitHub,
# scripts/gene_annotation/train_token_cls.py:78-96 (LABEL_NAMES), fetched 2026-10-02.
# Checkpoint constraints verified 2026-10-02: Anno config.id2label == {str(i): f"LABEL_{i}"}
# (identity placeholders, 17 entries), classifier.weight [17,512]; CRE config has no
# id2label, score.weight [2,512]; class-1 == CRE per upstream cis_regulatory README.
CRE_LABELS = ["Not CRE", "CRE"]
ANNO_LABELS = [
    "O", "B-CDS", "I-CDS", "L-CDS", "U-CDS",
    "B-INTRON", "I-INTRON", "L-INTRON", "U-INTRON",
    "B-UTR5", "I-UTR5", "L-UTR5", "U-UTR5",
    "B-UTR3", "I-UTR3", "L-UTR3", "U-UTR3",
]
```

```yaml
# appended to dnallm/models/model_info.yaml finetuned: section
  - name: "PlantHelixSeek CRE"
    model: "zhangtaolab/PlantHelixSeek-CRE"
    base_model: "zhangtaolab/PlantHelixSeek"
    task:
      describe: "Predict cis-regulatory elements (open chromatin) in plant genomes by using PlantHelixSeek model."
      task_type: "binary"
      num_labels: 2
      label_names: ["Not CRE", "CRE"]
      threshold: 0.5
  - name: "PlantHelixSeek Anno"
    model: "zhangtaolab/PlantHelixSeek-Anno"
    base_model: "zhangtaolab/PlantHelixSeek"
    task:
      describe: "Predict gene structure (CDS/intron/5'UTR/3'UTR) per nucleotide by using PlantHelixSeek model."
      task_type: "token"
      num_labels: 17
      label_names: ['O', 'B-CDS', 'I-CDS', 'L-CDS', 'U-CDS', 'B-INTRON', 'I-INTRON', 'L-INTRON', 'U-INTRON', 'B-UTR5', 'I-UTR5', 'L-UTR5', 'U-UTR5', 'B-UTR3', 'I-UTR3', 'L-UTR3', 'U-UTR3']
      threshold: 0.5
```

### Slow-leg smoke skeleton (shape proven by this session's probe)

```python
# tests/models/test_plant_helixseek_smoke.py (layout name = planner discretion)
import pytest
from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer

@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_planthelixseek_cre_smoke_load():
    cfg = TaskConfig(task_type="binary", num_labels=2,
                     label_names=["Not CRE", "CRE"], threshold=0.5)
    model, tok = load_model_and_tokenizer(
        "zhangtaolab/PlantHelixSeek-CRE", cfg, source="modelscope")  # ModelScope-first (locked)
    assert model.config.id2label == {0: "Not CRE", 1: "CRE"}
    seq = "ACGT" * 125  # 500 bp, uppercase (vocab is ACGTN-only)
    enc = tok([seq], return_tensors="pt", padding=True)
    out = model(input_ids=enc["input_ids"].to(model.device),
                attention_mask=enc["attention_mask"].to(model.device))
    assert tuple(out.logits.shape) == (1, 2)

@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_planthelixseek_anno_smoke_load():
    from tests.models.test_plant_helixseek_registry import ANNO_LABELS  # frozen list
    cfg = TaskConfig(task_type="token", num_labels=17,
                     label_names=ANNO_LABELS, threshold=0.5)
    model, tok = load_model_and_tokenizer(
        "zhangtaolab/PlantHelixSeek-Anno", cfg, source="modelscope")
    assert model.config.num_labels == 17
    assert model.config.id2label == dict(enumerate(ANNO_LABELS))  # REG-02 assert
    # 8192bp forward measured at 7.9s / 13GB peak on GB10 eager
```

### Fast-leg registry structure test skeleton

```python
# no network; parses the packaged yaml directly (it has no runtime consumer)
import yaml
from pathlib import Path
REGISTRY = Path(__file__).parents[2] / "dnallm" / "models" / "model_info.yaml"

def _finetuned_by_model(repo_id):
    data = yaml.safe_load(REGISTRY.read_text())
    return next(e for e in data["finetuned"] if e["model"] == repo_id)

def test_registry_entries_structure():
    cre = _finetuned_by_model("zhangtaolab/PlantHelixSeek-CRE")
    anno = _finetuned_by_model("zhangtaolab/PlantHelixSeek-Anno")
    assert cre["task"]["task_type"] == "binary" and cre["task"]["num_labels"] == 2
    assert len(cre["task"]["label_names"]) == 2
    t = anno["task"]
    assert t["task_type"] == "token" and t["num_labels"] == 17
    assert t["label_names"] == ANNO_LABELS and len(set(t["label_names"])) == 17
```

### CRE verification core (selection script; contracts verified this session)

```python
# bin track: upstream contract — mean class-1 prob over covering windows (<=10 per bin)
import numpy as np
probs = softmax_class1  # per 500bp window at stride 50
n_bins = locus_len // 50
score_sum, coverage = np.zeros(n_bins), np.zeros(n_bins)
for win_start, p in zip(starts, probs):          # starts = range(0, locus_len-500+1, 50)
    b0, b1 = win_start // 50, min((win_start + 500 - 1) // 50 + 1, n_bins)
    score_sum[b0:b1] += p
    coverage[b0:b1] += 1
assert coverage.min() > 0                        # SHOW-02 non-emptiness
bin_scores = np.divide(score_sum, coverage, where=coverage > 0)
# peaks: values >= mean + 1.5*std -> merge_gap 50, min_length 50 (upstream defaults)
# then: sort -k1,1 -k2,2n both BEDs -> bedtools jaccard -> parse column 3 (verified)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Upstream monkey-patches attention to SDPA ("ChaobaV2 OOMs at 8192") | Shipped remote code natively registers `sdpa` attention class + `_supports_sdpa` | checkpoint remote code (2026-07) | SDPA is available *if* the loader requests it; dnallm forces eager, so the patch is unnecessary but the eager ceiling (Pitfall 2) is real |
| Anno decode = heuristic | Upstream default `viterbi+orf` (transition matrix `transition_probs.npz` + ORF correction) | upstream repo current | Phase 6 floor calibration should use the simple decode it can freeze reproducibly (Open Question 2); upstream viterbi is the reference for "best results" |
| CRE per-window hard threshold examples | `mean_std k=1.5` peak calling on the 50bp-bin track | upstream default | SHOW-03/selection must use the mean±1.5σ contract (already locked in REQUIREMENTS wording) |
| transformers 4.49-only remote code | Loads clean on transformers 5.17.0 via dnallm route (this session) | verified 2026-10-02 | REG-03's risk is retired for the dev box; the 4.57 fallback path stays as belt-and-braces per CONTEXT |

**Deprecated/outdated:** nothing in-repo goes stale in this phase; the milestone-level deprecations (docs mirror repair etc.) were Phase 5 scope.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | "≤200kb per region" means ≤200,000 *bases of committed sequence* per region set (REQUIREMENTS wording "fragments ≤200kb per region"), not ≤200KB total file bytes | Budget arithmetic (Pattern 4) | If the owner means bytes: a full 200kb locus (~308 KB with truth slices) violates it → cap locus at ~120-150 kb; cheap to honor either way if decided before selection runs |
| A2 | Phase-6 floor calibration uses the simple argmax+BILOU-collapse decode (not the upstream viterbi+orf), and Phase 7 asserts floors with the same decode | Pattern 5 / Open Q2 | If Phase 7 adopts viterbi decode, floors calibrated on simple decode may be under-conservative or over-conservative → re-run selection or calibrate both |
| A3 | Exon-F1 match rule "predicted CDS span ↔ truth CDS feature, reciprocal overlap ≥ 0.5, same strand, pooled per locus" is acceptable as the floor metric definition | Pattern 5 | A stricter rule (exact match) would make ≥0.8 on ≥3 genes harder to hit; selection-time measurement reveals which is attainable — pin the definition in selection.md |
| A4 | The negative-control assertion metric is predicted-signal base fraction ≈ 0 (tolerance band set at selection time), not jaccard-vs-empty | Pattern 5 | If owner prefers a different negative metric, only selection.md + the Phase-7 assertion shape change |
| A5 | Committed fragments use the `.fas` extension (verified not gitignored) rather than `.fa` + negation patterns | Pitfall 3 / Structure | Cosmetic; negation-pattern route equally valid |
| A6 | The helper lives at `dnallm/utils/genomic_coords.py` (discretion area — recommendation) | Pattern 6 | Relocating later is a rename; low cost but touches imports in 3 consumers |
| A7 | Upstream GitHub `train_token_cls.py` LABEL_NAMES reflects the released checkpoints' training order (evidence: class-weight indexing `train_counts[LABEL_NAMES[i]]`, pred_to_gff3 genic_mask `!= 0`, both fetched this session; both checkpoints' remote code + upstream scripts are same-org, same-date) | Pattern 2 | If wrong, the behavioral backstop (Anno exon-F1 on the selected locus) collapses → selection would fail visibly, forcing a label-order investigation; no silent failure path remains |
| A8 | ModelScope repos stay in sync with HF for these checkpoints (both verified published 2026-10-02; owner-controlled org) | Pattern 3 | Route divergence would surface as load failure at smoke time; HF fallback is the one-line `source=` change |

## Open Questions (RESOLVED)

All five resolved by the owner on 2026-10-02 — recorded in 06-CONTEXT.md § "Post-research decisions (2026-10-02)". Per-question outcomes:

1. **The 200kb budget unit (A1)** — What we know: full-200kb locus ≈ 308 KB of committed files (202.5 KB FASTA + 5 KB DHS + 101 KB TAIR10, measured on a real gene-dense example); REQUIREMENTS says "fragments ≤200kb per region" (sequence reading). What's unclear: whether the owner also cares about total bytes. Recommendation: sequence reading; record byte totals in `selection.md`; if the owner wants a byte cap, select a 120-150 kb locus — all thresholds still attainable (more genes than needed at Chr1 front-20Mb density).
   → RESOLVED: budget unit = sequence bases; byte totals recorded in `selection.md` (owner decision, absorbed in fdcccb3).
2. **Anno decode for floor calibration (A2)** — simple argmax (free, self-contained) vs porting upstream viterbi (74KB file port + `transition_probs.npz` provenance/license question). Recommendation: simple decode, documented; viterbi can appear in Phase 7 as an *illustrative* improvement beyond the asserted floor.
   → RESOLVED: simple argmax decode for calibration; viterbi deferred to Phase 7 as illustrative only.
3. **Shared-data layout** — the negative-control set and `selection.md` serve both notebooks. Options: (a) a shared `example/notebooks/plant_helixseek_shared/data/` dir; (b) duplicate copies per task dir; (c) keep everything under the CRE dir and reference from Anno. Recommendation: (a) — one source of truth, matches "one shared negative-control set" in the locked decisions.
   → RESOLVED: option (a) shared `plant_helixseek_shared/data/` dir.
4. **Smoke-test sourcing route on the slow leg** — ModelScope-first is the locked model-sourcing decision and both routes are proven; the smoke test could run `source="modelscope"` only, or both. Recommendation: modelscope-only for the nightly budget (the HF route differing only in cache path), HF fallback attempt only if MS fails.
   → RESOLVED: ModelScope-only smoke with one HF fallback attempt on MS failure.
5. **Freeze-script placement** — standalone `scripts/showcase/freeze_registry.py` vs a `--freeze-registry` mode of `select_loci.py`. Recommendation: standalone (registry work is dependency-free and parallelizable; the selection script stays data-only).
   → RESOLVED: standalone script — and per the binding intermediate-tooling constraint it lives under the gitignored scratch home, never `scripts/`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| GB10 GPU (121.6 GiB) | selection-scan inference, smoke forwards | ✓ | cc 12.1, torch 2.11.0+cu130 | CPU (hours-scale — not viable for Anno; de-scope to smaller locus) |
| transformers (dev env) | REG-03 gate | ✓ | 5.17.0 | one attempt on 4.57 per CONTEXT, else typed skip |
| HuggingFace network | smoke downloads | ✓ | ~22 MB/s measured (1.9 GB ≈ 85 s) | ModelScope route (proven, 122 s incl. load) |
| ModelScope network | ModelScope-first sourcing | ✓ | API + full download verified 2026-10-02 | HF route (proven) |
| plantdhs.org | TAIR10_DHSs.gff.zip (the one fetch) | ✓ | verified 2026-10-02 (493 KB zip) | manual placement into `.scratch/` + non-zero exit (CONTEXT contract) |
| `~/Downloads/TAIR10_chr1.fas` | genome source | ✓ | 30.8 MB, verified content | arabidopsis.org URL (browser-UA; SPA risk — milestone Pitfall 12) |
| `~/Downloads/TAIR10_GFF3_genes.gff` | Anno truth | ✓ | 44.1 MB, verified content | same as above |
| pyfastx | FASTA slicing | ✓ | 2.3.1 (dev extra) | none needed |
| bedtools | jaccard metric | ✓ | v2.31.1 (linuxbrew) | pure-Python interval math cross-check (CONTEXT allows as fallback) |
| einops | remote-code hard dep | ✓ | 0.8.2 | none needed (proven sufficient — fla/flash_attn guarded upstream) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none currently missing; the two documented fallback ladders (plantdhs manual placement; HF↔MS sourcing) are contractual contingencies, not gaps.

## Security Domain

ASVS level 1 (`security_asvs_level: 1`, `security_block_on: high`). This phase adds no auth/session/crypto surface; it does add **external-data ingestion** and **remote-code execution** touchpoints.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no new endpoints/daemons) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | All downloaded/committed ground-truth files validated: zip magic bytes + GFF structural parse via the SHOW-02 helper with non-emptiness assertions (raises on silent-empty); registry yaml parsed with `yaml.safe_load` (never `yaml.load`); locus coords bounds-checked against the chromosome row (Chr1 length 30,427,671) |
| V6 Cryptography | no | — |
| V12 File Handling | **yes** (L1-relevant) | Selection script writes only inside gitignored `.scratch/` and the designated `data/` dirs; tree-clean discipline from Phase 5 (`assert_tree_clean` pattern) applies to the curation run; no executable content in committed data |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious/compromised data download (plantdhs.org zip swapped for something else) | Tampering | magic-byte + structural validation before parse; files are data-only (never executed); committed artifacts are human-reviewed in the PR |
| `trust_remote_code` execution of model repo code on the self-hosted GPU box (milestone Pitfall 8) | Elevation/Execution | checkpoints are owner-org published (zhangtaolab), commit shas recorded in this research and to be pinned in selection.md/models.lock (Phase 8 revision pinning); no unpinned floating refs in tests |
| Prompt-injection-style content inside fetched files (GFF attributes are untrusted text) | Tampering | parser treats column 9 as opaque data; no eval/format-execution on it; untrusted-input boundary honored (external content inspected as data this session) |
| YAML deserialization | Tampering | `yaml.safe_load` only |

## Sources

### Primary (HIGH confidence)
- Live probes on the dev box (GB10, transformers 5.17.0, torch 2.11.0+cu130), 2026-10-02: HF+MS downloads, dnallm-route loads (both models, both sources), CRE batch ladder, Anno 8192 forward, pyfastx coordinate fixture, bedtools jaccard smoke — scripts/logs retained under /tmp (`phs_smoke_probe.py`, `phs_batch_ladder.py`, `phs_ms_probe.py`)
- `dnallm/models/model.py` (Read: 380-495 source dispatch, 495-631 `_create_label_mappings`/`_load_model_by_task_type`, 661-716 `_safe_num_labels`, 719-901 dispatch chain), `dnallm/models/tokenizer.py:256-305`, `dnallm/configuration/configs.py:97-131`, `dnallm/models/model_info.yaml:164-172` + token precedent + tail
- HuggingFace raw/API fetches 2026-10-02: both `config.json` files, `?blobs=true` file listings (sizes + shas), safetensors header range-requests (head shapes), both READMEs, `vocab.txt`, `tokenizer_config.json`, remote `attention.py`/`model.py`
- GitHub raw fetches 2026-10-02: `train_token_cls.py` (LABEL_NAMES + SDPA patch + training config), `predict_genome_multigpu.py` (window/stitch/B-swap/batch), `predict_cre.py` (window/stride/bin/batch), `call_peaks_from_bigwig.py` (thresholds), both pipeline READMEs
- ModelScope API fetches 2026-10-02: model metadata + repo file listings for both checkpoints
- plantdhs.org: `/Download` page scrape + `TAIR10_DHSs.gff.zip` download and full content census
- Local data inspection 2026-10-02: `~/Downloads/TAIR10_chr1.fas` (header, line width, full case/N census), `~/Downloads/TAIR10_GFF3_genes.gff` (feature census, ID/Parent conventions, trailing-semicolon count, gene densities)
- `.gitignore` behavior: `git check-ignore -v` on candidate artifact paths

### Secondary (MEDIUM confidence)
- Milestone research layer (`.planning/research/ARCHITECTURE.md`, `PITFALLS.md` — repo-verified 2026-10-01) — used for the two-leg test design, sandbox patterns, Pitfalls 11-13 mapping; not re-derived here
- Upstream README-documented defaults (bin aggregation wording, peak thresholds, viterbi+orf recommendation) — authoritative for pipeline contracts but describing their scripts, which we re-implement on the dnallm API

### Tertiary (LOW confidence)
- None — no claim in this document rests on an unverified single web source

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — everything verified installed this session; no new packages
- Registry/freeze mechanics: HIGH — checkpoint configs, head shapes, upstream label order, dnallm rebuild path all captured verbatim + empirically exercised
- Data curation: HIGH — all three data sources downloaded/inspected or locally verified; slice sizes and densities measured
- Selection feasibility: HIGH — timing measured on the actual box; the selection *outcomes* (which locus wins, observed Jaccard/F1) are inherently execution-time facts and are not predicted here

**Research date:** 2026-10-02
**Valid until:** 2026-11-01 (stable domain: pinned static data files + owner-org checkpoints; the only volatile fact is plantdhs.org availability, which has a contractual fallback)

---
*Phase: 6-Model Registry & Showcase Data Curation*
*Research layer: implementation detail below `.planning/research/*.md` (milestone architecture layer incorporated by reference, not re-derived)*
