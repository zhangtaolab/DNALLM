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

_(pending — filled by Task 2)_

## C. invert_attention_mask reachability analysis

_(pending — filled by Task 2)_

## D. Example-nightly junit forensics (run 38308972028)

_(pending — filled by Task 2)_

## E. Census baseline (pre-edit)

_(pending — filled by Task 2)_

## F. Evidence table and disposition options

_(pending — filled by Task 2)_

## G. Decision record

_(pending — filled by the Task 3 owner checkpoint)_
