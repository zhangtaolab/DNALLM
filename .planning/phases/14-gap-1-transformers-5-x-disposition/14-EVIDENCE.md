# Phase 14 Evidence Record — GAP-1 Transformers 5.x Disposition (GAPC-01/GAPC-02)

**Recorded:** 2026-10-13 (executor run of plan 14-01, on the dev venv)
**Dev-venv versions** (`.venv/bin/python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"` → `5.17.0 2.11.0+cu130`):

```
$ .venv/bin/python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
5.17.0 2.11.0+cu130
```

Section order is fixed by plan 14-01 (edge GAPC-01/ordering): A load run, B BPE-leg, C reachability, D nightly junit, E census baseline, F evidence table + options, G decision record.

## A. GAPC-01 recorded load run

**Command (exact):**

```
.venv/bin/python -m pytest tests/models/test_model_remote_code.py::TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards -m slow -q --junitxml=/tmp/gapc14-smoke-junit.xml 2>&1 | tee /tmp/gapc14-smoke-run.log
```

(`-m slow` re-includes the `@pytest.mark.slow` test — default fast-lane markers deselect it. The run executes the REAL load plus forward through `load_model_and_tokenizer` with `source="modelscope"` — the exact route `benchmark_config.yaml` uses. D-01: the evidence is execution, not inventory.)

**Verbatim output — last 15 lines of /tmp/gapc14-smoke-run.log:**

```
============================= test session starts ==============================
platform linux -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/forrest/Github/DNALLM
configfile: pyproject.toml
plugins: timeout-2.4.0, cov-7.1.0, asyncio-1.4.0, logfire-sdk-6.0.0b8, anyio-4.15.1, langsmith-0.14.7, progress-1.4.0, platformdirs-4.12.4
timeout: 300.0s
timeout method: signal
timeout func_only: False
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item

tests/models/test_model_remote_code.py .                                 [100%]

--------------- generated xml file: /tmp/gapc14-smoke-junit.xml ----------------
============================== 1 passed in 6.92s ===============================
```

**Machine-readable artifact:** `/tmp/gapc14-smoke-junit.xml` — exactly one `<testsuite>` carrying `errors="0" failures="0" skipped="0" tests="1"` with the single testcase `classname="tests.models.test_model_remote_code.TestRemoteCodeCheckpointCompat" name="test_nt_v2_promoter_loads_and_forwards" time="3.562"` (timestamp 2026-10-13T02:04:59Z, hostname edgexpert-3eee).

**Environment:** transformers `5.17.0`, torch `2.11.0+cu130` (recorded verbatim in the header above), modelscope snapshot cache `~/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/snapshots/master/` (warm).

**Ladder sizing (measured from this recorded run):** `new rungs required: 0`

The rung ladder as it exists today (`apply_patches()` in `dnallm/utils/transformers_compat.py:1228-1246`, including the `_patch_pretrained_config_legacy_defaults` rung at 489-510 that answers `is_decoder`/`add_cross_attention` over the closed default map) sufficed for the full load + real `(1, 2)`-logits forward; no traceback occurred. Per D-01 rungs extend only on real tracebacks, and none was produced.

## B. BPE-leg load probe

**Command (exact):**

```
.venv/bin/python /tmp/gapc14-bpe-probe.py 2>&1 | tee /tmp/gapc14-bpe-probe.log
```

The probe (`/tmp/gapc14-bpe-probe.py`, mirroring the research session's `/tmp/gapc01-probe/bpe_leg_probe.py`) imports `dnallm.utils.transformers_compat`, builds `TaskConfig(task_type="token", num_labels=7, label_names=["O", "B-UTR", "I-UTR", "B-CDS", "I-CDS", "B-INT", "I-INT"])` — mirroring `example/notebooks/finetune_NER_task/ner_task_config.yaml` exactly — calls `load_model_and_tokenizer("zhangtaolab/plant-nucleotide-transformer-BPE", ..., source="modelscope")`, prints the model class, roundtrips one 64 nt sequence through the tokenizer, and runs a no-grad forward.

**HARD CONSTRAINT honored (Pitfall 3):** `task_type` is exactly `token` — any other value falls into the `dnallm/models/model.py:653-664` else-branch, resolves native `EsmModel` with the remote `EsmConfig`, and fails on `rope_theta`; that failure would be probe error, not ladder evidence.

**Verbatim output:**

```
transformers: 5.17.0
Downloading Model from https://www.modelscope.cn to directory: /home/forrest/.cache/modelscope/hub/models/zhangtaolab/plant-nucleotide-transformer-BPE
02:08:19 - dnallm.utils.support - INFO - Model files are stored in /home/forrest/.cache/modelscope/hub/models/zhangtaolab/plant-nucleotide-transformer-BPE
Loading weights:   0%|          | 0/336 [00:00<?, ?it/s]Loading weights: 100%|████████████| 336/336 [00:00<00:00, 9688.01it/s]
[transformers] [1mEsmForTokenClassification LOAD REPORT[0m from: /home/forrest/.cache/modelscope/hub/models/zhangtaolab/plant-nucleotide-transformer-BPE
Key                       | Status     |
--------------------------+------------+-
lm_head.dense.weight      | UNEXPECTED |
lm_head.dense.bias        | UNEXPECTED |
lm_head.bias              | UNEXPECTED |
lm_head.layer_norm.bias   | UNEXPECTED |
lm_head.decoder.weight    | UNEXPECTED |
lm_head.layer_norm.weight | UNEXPECTED |
classifier.bias           | MISSING    |
classifier.weight         | MISSING    |

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
=== LOAD OK ===
model class: EsmForTokenClassification
tokenizer class: EsmTokenizer
config.is_decoder resolved: False
config.add_cross_attention resolved: False
tokenizer roundtrip: '<cls> ACGT ACGT ACGT ACGT ACGT ACGT ACGT ACGT ACGT ACGT ACGT'
[transformers] `use_return_dict` is deprecated! Use `return_dict` instead!
forward logits shape: (1, 17, 7)
sequence length (input_ids.shape[1:]): (17,)
=== PROBE COMPLETE: BPE-leg checkpoint load + forward succeed ===
```

**Reading:** model class `EsmForTokenClassification` (the remote-code class, not native), forward logits `(1, 17, 7)` with the probed sequence length 17 (BPE tokenization of `"ACGT"*16`). The `use_return_dict is deprecated!` warning is benign (the 4.x-era remote code passing the legacy kwarg; transformers 5.x maps it). The LOAD REPORT's UNEXPECTED `lm_head.*` keys and MISSING `classifier.*` weights are the expected shape when a masked-LM checkpoint head is swapped for a freshly-initialized token-classification head — the standard transformers adaptation report, not an error.

## C. invert_attention_mask reachability analysis

**Verbatim guard block** from the cached remote `~/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/snapshots/master/modeling_esm.py` (lines 1029-1044) — the ONLY call site of `invert_attention_mask` in the file, wrapped by this conditional, with the else-arm assigning `None`:

```python
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None
```

**Checkpoint config.json key-absence check:**

```
$ .venv/bin/python -c "import json; c = json.load(open('/home/forrest/.cache/modelscope/models/zhangtaolab--nucleotide-transformer-v2-100m-promoter/snapshots/master/config.json')); print('KEYS-ABSENT' if 'is_decoder' not in c and 'add_cross_attention' not in c else 'KEY-PRESENT')"
KEYS-ABSENT
```

**Three-step conclusion (the recorded justification for ladder size zero):**

1. The `_patch_pretrained_config_legacy_defaults` rung (`dnallm/utils/transformers_compat.py:489-510`, the closed `_LEGACY_PRETRAINED_CONFIG_DEFAULTS` map) answers `is_decoder` as `False` on every 5.x `PretrainedConfig` — confirmed live in both recorded runs (section A smoke; section B probe line `config.is_decoder resolved: False`).
2. The checkpoint's `config.json` never carries `is_decoder` or `add_cross_attention` as explicit keys (KEYS-ABSENT above), so nothing ever sets `is_decoder` truthy for this checkpoint.
3. dnallm passes `encoder_hidden_states=None` on every binary/token load route it exercises (`load_model_and_tokenizer` forwards through the model with tokenizer outputs only — no cross-attention encoder inputs anywhere on these routes).

Therefore `self.config.is_decoder and encoder_hidden_states is not None` is always False on every binary/token route dnallm loads — the `invert_attention_mask` call site is unreachable, and the else-arm (`encoder_extended_attention_mask = None`) always executes. The milestone research's "exactly `invert_attention_mask` missing" was an attribute inventory over a dead branch, not an execution trace (Pitfall 1); no rung is required for it.

## D. Example-nightly junit forensics (run 38308972028)

**Provenance (gh API, queried 2026-10-13):** run `38308972028` — workflow "CI", event `workflow_dispatch`, branch `v1.3`, created `2026-10-12T21:10:38Z`, run-level conclusion **cancelled**. Job conclusions:

```
example-nightly (py3.12, example execution census)  -> success
coverage-nightly (py3.12, full suite incl. slow)    -> cancelled
test-mamba (3.11)                                   -> failure
test-windows (py3.12) / test / test-cuda / coverage-gate / deploy -> skipped
```

**Do not misread the run status:** the run-level `cancelled` reflects the co-enqueued coverage-nightly cancellation during the 2026-10-12 power-outage recovery; the example-nightly JOB itself concluded **success**. The artifact `example-nightly-junit` (id 11778219413) was re-downloaded fresh via `gh run download 38308972028 -n example-nightly-junit -D /tmp/gapc14-nightly-junit/` this session and is **byte-identical** to the research session's copy (all 12 files `cmp`-identical), so the rows below are the artifact of record.

**Stage-1 census line (verbatim from the artifact's `stage1-examples.log` tail):**

```
===== 211 passed, 5 skipped, 1 deselected, 1 warning in 6596.90s (1:49:56) =====
```

JUnit suite header (`pytest-junit-example-stage1.xml`): `tests="216" failures="0" errors="0" skipped="5"` (timestamp 2026-10-12T21:26:46Z, hostname edgexpert-3eee); all stage-4 audits exited 0 (`stage-results.txt`: 8/8 zero).

**PASS row — BPE script lane:**

```
<testcase classname="tests.examples.test_script_execution.TestExampleScriptExecution" name="test_generate_bpe_dataset_produces_artifact" time="102.244" />
```

**PASS row — benchmark notebook (third model) lane:**

```
<testcase classname="tests.examples.test_notebook_execution.TestNotebookExecution" name="test_notebook_executes_end_to_end[notebooks/benchmark/benchmark.ipynb]" time="48.714" />
```

**Classification of all 5 skips (from the junit `<skipped>` rows):**

| # | Test id | Skip message (lead) | Class |
|---|---------|---------------------|-------|
| 1 | `tests.examples.test_examples.TestNotebookExamples::test_notebook_imports[notebooks/data_prepare/predict/predict_data.ipynb]` | `No import statements found` | content |
| 2 | `tests.examples.test_notebook_execution.TestGatedNotebookExecution::test_gated_notebook_probes_then_executes[mcp_example/mcp_client_ollama_langchain_agents.ipynb]` | `network-unavailable: ... dnallm MCP server http://localhost:8000/mcp unreachable` | mcp network-unavailable |
| 3 | `tests.examples.test_notebook_execution.TestGatedNotebookExecution::test_gated_notebook_probes_then_executes[mcp_example/mcp_client_ollama_pydantic_ai.ipynb]` | `network-unavailable: ... dnallm MCP server http://localhost:8000/mcp unreachable; oll...` | mcp network-unavailable |
| 4 | `tests.examples.test_plant_helixseek_showcase::test_anno_notebook_executes_within_selection_bands` | `optional-dep: ... (pygenometracks zoom cell) (pygenometracks CLI not on PATH (Windows-excluded notebook extra, 261007-mhz...` | pygenometracks optional-dep |
| 5 | `tests.examples.test_plant_helixseek_showcase::test_combined_notebook_executes_with_display_figure` | `optional-dep: ... (pygenometracks zoom cell) (pygenometracks CLI not on PATH (Windows-excluded notebook extra, 261007-mhz...` | pygenometracks optional-dep |

**Explicit statement (per-item #15 evidence, D-08; Pitfall 5):** none of the 5 skips is a #15-class skip — the #15 items are the evo/megaDNA optional-dep gates, the megaDNA demo cell, and the PlantCAD lora pair / mamba_ssm gates, and NONE of them appears in this junit's skip set (the two pygenometracks rows belong to the #19/showcase surface). Every #15 gate executed for real in this green run.

## E. Census baseline (pre-edit)

**Command (the exact stage-0.5 flags), run twice:**

```
.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" 2>/dev/null | tail -1
```

Raw lines (run-dependent timing suffix, identical triple):

```
run 1: =============== 216/217 tests collected (1 deselected) in 0.89s ================
run 2: =============== 216/217 tests collected (1 deselected) in 0.82s ================
```

Normalized with the stage-0.5 gate's own sed (`sed -E 's/^=+ *//; s/ *=+$//; s/ in [0-9.]+s?$//'`):

```
run 1: 216/217 tests collected (1 deselected)
run 2: 216/217 tests collected (1 deselected)
IDEMPOTENT: 216/217 tests collected (1 deselected)
```

Both normalized lines are identical and read `216/217 tests collected (1 deselected)` — the pre-edit baseline plan 14-02 proves against. This plan touches no collected test surface (no test file is edited), so the triple must hold unchanged through the disposition. Cross-reference: the nightly artifact's own census line (section D, `census-collect.txt` tail: `=============== 216/217 tests collected (1 deselected) in 1.25s ================`) carries the same triple.

## F. Evidence table and disposition options

**Ladder sizing (measured, section A):** `new rungs required: 0`

**Reachability conclusion (section C):** the milestone probe's one known gap (`invert_attention_mask`) is an attribute inventory over a dead branch — unreachable on every binary/token route dnallm loads.

### Evidence table

| Surface | Recorded outcome | Evidence section |
|---------|------------------|------------------|
| Promoter load (`zhangtaolab/nucleotide-transformer-v2-100m-promoter`, binary, source=modelscope) | GREEN — real load + real forward, `(1, 2)` logits, `1 passed in 6.92s`, junit `tests="1" failures="0" errors="0" skipped="0"` | A |
| BPE-leg load (`zhangtaolab/plant-nucleotide-transformer-BPE`, token, num_labels 7) | GREEN — remote `EsmForTokenClassification`, tokenizer roundtrip, real forward logits `(1, 17, 7)` | B |
| BPE script lane (`generate_bpe_dataset.py`) | PASS for real in example-nightly run 38308972028 — `test_generate_bpe_dataset_produces_artifact` time=102.244 | D |
| Benchmark notebook third model (`benchmark.ipynb` + `benchmark_config.yaml`) | PASS for real in example-nightly run 38308972028 — `test_notebook_executes_end_to_end[notebooks/benchmark/benchmark.ipynb]` time=48.714 | D |
| `invert_attention_mask` (the milestone probe's one known gap) | DEAD BRANCH — the only call site is guarded by `self.config.is_decoder and encoder_hidden_states is not None`; `is_decoder` resolves False via the legacy-defaults rung, config.json carries no such keys, `encoder_hidden_states` is None on every route | C |

### Disposition options (verbatim from the locked decisions D-03/D-04/D-05/D-08)

**Option shim-path — expected: zero new rungs.** Keep the NT surface; execute plan 14-02 (comment-only prose resolution, local gates, WINDOWS #11/#12/#15 fixed with probe evidence per D-08, CI re-verification). Closure semantics: shim path closes #11/#12/#15 **all fixed with probe evidence** (D-08). Keeps nightly-green functionality on all three surfaces (benchmark third model, BPE script lane, models-lane smoke). The shim already lives in the existing ladder per D-05 (`dnallm/utils/transformers_compat.py`, version-keyed, absence-gated rungs) — zero code change to the ladder is required; the already-landed rungs suffice (section A measured).

**Option removal — honest removal of the third benchmark model plus the BPE pkl leg.** Atomic scope: third benchmark model + BPE pkl leg removed **together** per D-03 (#11/#12 same GAP-1 root, never separately). Closure semantics: closes #11/#12 as **removed-by-decision** with #15 closed **on per-item probe evidence** per D-08. Deletes working nightly-green coverage (benchmark third model, BPE pkl production lane); benchmark coverage shrinks from 3 models to 2; blast radius spans ~9 files (model_info.yaml, modeling_auto.py, dataset_auto.py + data.py, mcp/server.py, lora_targets.yaml, the benchmark example config, generate_bpe_dataset.py, and their tests) with census re-pin per D-07. Requires a fresh planning pass before any execution.

**D-02 threshold (restated):** at most 3 additional rungs, public-API absence only; touching private transformers internals or behavior emulation flips the disposition to removal. Measured requirement this phase: 0 — far inside the threshold.

**Supply-chain anchor (models.lock lines 32-33):**

```
ms  zhangtaolab/plant-nucleotide-transformer-BPE@e66bf2535aa2fc3297f677195d6b7e122c3a0510  # example notebook finetune_NER_task + generate_bpe_dataset.py script (source=modelscope)
ms  zhangtaolab/nucleotide-transformer-v2-100m-promoter@c138b8c50e0013354a3aac0cef722be608bcffd0  # example benchmark_config.yaml third model (source=modelscope)
```

**Recommendation (on the recorded evidence):** shim-path / zero new rungs — the gap is measured closed, both nightly lanes pass for real, the one known missing attribute is a dead branch, and removal would delete working, nightly-green functionality. The choice remains the owner's per D-04 (Task 3 checkpoint).

## G. Decision record

**Selected option (verbatim):** `shim-zero-rung`
**Decided:** 2026-10-13 (owner selection taken via the D-04 blocking-human decision checkpoint; recorded 2026-10-13T02:15Z)
**Question as presented:** "Execute the GAP-1 disposition via the shim path (zero new rungs) or via honest removal of the third benchmark model plus the BPE pkl leg, on the recorded evidence in 14-EVIDENCE.md section F" — one question, exactly two options (`shim-zero-rung` "Shim path — zero new rungs" / `removal` "Honest removal — third model + BPE pkl leg"), with the section F evidence table, the section A measured sizing line (`new rungs required: 0`), and the section C reachability conclusion rendered in the question body.

**Consequence (one line):** proceed to plan 14-02 (shim-path disposition: comment-only prose resolution, local gates, WINDOWS #11/#12/#15 fixed with probe evidence per D-08, CI re-verification).

**Closure semantics this selection locks (GAPC-02 edge, D-08):** WINDOWS #11/#12/#15 close as fixed-with-probe-evidence (the recorded sections A-D are that evidence; #15 per-item via the section D nightly classification — zero #15-class skips in the green run).
