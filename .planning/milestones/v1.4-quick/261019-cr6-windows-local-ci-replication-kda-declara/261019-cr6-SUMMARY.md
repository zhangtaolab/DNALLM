---
phase: 261019-cr6
plan: 01
subsystem: ci-replication
tags: [windows, ci-parity, kda, triton-windows, ty, cuda130, evidence]
status: complete
gate_failures: [ty-hard-gate]
commits: 2
plan_head_before: 7bc9ecf8b95abc3698b664d993ac47312b55d31c
plan_head_after: 6d14d9d104e8caf34480a7e6de9785a030b65909
key-files:
  created:
    - .planning/quick/261019-cr6-windows-local-ci-replication-kda-declara/261019-cr6-EVIDENCE.md
  modified: []
decisions:
  - ty hard gate is platform-dependent (pyfaidx via Linux-only pygenometracks) — documented RED, no source fix per owner boundary
  - torch 2.10.0+cu130 is the CI-faithful cuda130 resolution under current pyproject (<2.12); plan's 2.11.x expectation recorded as deviation
actuals:
  tokens: 4611   # chars/4 over the realized diff (18444-byte EVIDENCE.md)
  tasks: 4
  commits: 2     # measured: git rev-list --count 7bc9ecf..HEAD
metrics:
  duration: 4h 20m wall (02:49Z-07:09Z; ~2h of it collision forensics + long installs/GPU leg)
  completed: 2026-10-19
---

# Quick Task 261019-cr6: Windows local CI replication + KDA declaration — Summary

One-liner: All five CI detection-surface gates replicated in fresh venvs on the owner's real Windows machine — four GREEN (ruff x2, fast leg 96% cov + audit + canary, GPU test-cuda leg on RTX 5080), one RED with root cause (ty hard gate: pyfaidx unresolved-import, platform-dependent by construction); WINDOWS #27 KDA fresh-venv declaration proven (triton-windows 3.7.1.post27 auto-resolved, import OK True) and combined with dnallm-cuda kernel evidence into a committed six-item evidence file.

## Gate Results (per task)

| # | Task | Gate command | Exit | Key output |
|---|------|--------------|------|------------|
| 1 | Fresh py3.12 .venv + `.[full]` | `uv pip install -e ".[full]"` (venv seeded from MinerU 3.12.11) | 0 | `Resolved 347 packages / Installed 347`; ruff exactly 0.16.10; torch 2.10.0+cpu |
| 1 | ruff format | `PYTHONUTF8=1 .venv/Scripts/ruff.exe format --check .` | 0 | `312 files already formatted` |
| 1 | ruff lint | `PYTHONUTF8=1 .venv/Scripts/ruff.exe check . --statistics` | 0 | no output, no violations |
| 1 | ty stub env install | `uv pip install -e ".[full]" -c .github/workflows/ty-stub-constraints.txt` | 0 | all 12 pins exact (torch 2.10.0, transformers 5.17.0, datasets 4.8.4, tokenizers 0.23.3, peft 0.21.2, accelerate 1.15.0, evaluate 0.4.6, altair 6.3.0, modelscope 1.40.2, numpy 2.2.0, pandas 3.0.6, pydantic 2.14.1) |
| 1 | **ty hard gate** | `uvx ty@0.0.84 check dnallm/ --python $TEMP/ty-stub-env-cr6` | **1 — RED** | `error[unresolved-import]: Cannot resolve imported module 'pyfaidx'` at `dnallm/models/special/enformer_model/data.py:112:14`; `Found 1 diagnostic` (reproduced twice, identical) |
| 2 | CPU-torch precondition | `python -c "...sys.exit(0 if not torch.cuda.is_available() else 1)"` | 0 | `2.10.0+cpu`, cuda unavailable |
| 2 | Fast leg | `python -m pytest -m "not slow" --cov --junitxml=pytest-junit.xml` | 0 | `2571 passed, 4 skipped, 75 deselected, 23 subtests passed in 500.97s`; junit `tests="2598" errors="0" failures="0" skipped="4"`; `TOTAL 9809 351 96%` (fail_under=90 held via bare --cov) |
| 2 | Skip audit | `python scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml` | 0 | 4 skipped, all allowed; `OK: every skip in pytest-junit.xml matches the allowlist` |
| 2 | Exit-code canary | exact ci.yml lines 192-208 shape | 0 (OK branch) | `Canary OK: pytest exited non-zero as expected`; `1 failed in 0.18s`; transient file + log deleted |
| 3 | KDA Step 1 install | `uv pip install -e ".[fla]"` into fresh py3.11 .venv-kda | 0 | `+ triton-windows==3.7.1.post27` auto-resolved from the Windows platform marker — zero manual triton installs |
| 3 | KDA Step 2 | `uv pip show triton-windows` | 0 | `Version: 3.7.1.post27` |
| 3 | KDA Step 4 import | `from fla.ops.kda.chunk import chunk_kda; print('import OK:', callable(chunk_kda))` | 0 | `import OK: True` |
| 4 | GPU self-check | `nvidia-smi --query-gpu=name,driver_version,memory.total` | 0 | RTX 5080, 610.88, 16303 MiB |
| 4 | cuda130 install | `uv pip install -e ".[full,cuda130]"` into fresh py3.11 .venv-cuda | 0 | `+ torch==2.10.0+cu130` via torch-cuda130 index routing (351 packages) |
| 4 | Device record | `print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))` | 0 | `2.10.0+cu130 13.0 True NVIDIA GeForce RTX 5080` |
| 4 | GPU test leg | `python -m pytest tests/ -v -m "not slow" --tb=short` (CI-exact: no cov/junit/audit) | 0 | `2519 passed, 4 skipped, 68 deselected, 23 subtests passed in 347.94s` — real device, strictly stronger than CI's device-less hosted leg |

Verify chains: T2-GREEN, T3-GREEN, T4-GREEN (T1's chain exits 1 at the ty step — the documented RED gate).

## Gate Failures

**ty hard gate (coverage-gate job shape) — RED, deterministic, root-caused, NOT fixed (owner-ratified boundary).**

- Command (CI-exact): `uvx ty@0.0.84 check dnallm/ --python <12-pin stub env>` → exit 1, `error[unresolved-import]: Cannot resolve imported module 'pyfaidx'` at `dnallm/models/special/enformer_model/data.py:112` (lazy `from pyfaidx import Fasta  # optional dep, raise-on-use`).
- Root cause (proven this session): `pyfaidx` is declared nowhere in pyproject; it reaches CI's Linux env only transitively via `pygenometracks`, which pyproject gates `platform_system != 'Windows'` (line 116). Dry-run proof: `pygenometracks>=3.9` resolves `+ pyfaidx==0.9.0.4`; bare `pybedtools` resolves `pysam` not pyfaidx (pygenometracks is the sole carrier). ty analyzes the function-body import statically, so the gate diverges by platform: green on CI Linux only because a Linux-only transitive dependency satisfies the import. Any Windows CI-parity replica of this gate is RED by construction.
- Consequence for the plan's must_haves: "All five local gates green" does NOT hold (4/5 green); the failing one is captured verbatim in EVIDENCE.md section C with the reproduction, root cause, and CI-divergence statement. No source fix applied (`# ty: ignore` pragma or pyfaidx declaration would be source changes — out of scope per constraints).

## Deviations from Plan

1. **[Execution collision — documented, not auto-fixed]** Two prior interrupted executor sessions raced this plan on the same machine (session 1 ≈02:12-02:30Z; session 2 ≈02:51-03:29Z, overlapping this session's 02:49Z start; both dead by 03:29Z; neither committed). Their leftovers: a pre-replaced .venv (hence the stale-venv inventory differed from the plan's expectation — found py3.12.11/ruff 0.16.10/torch 2.10.0+cpu), a pre-built .venv-kda, a stale pytest-junit.xml, an untracked EVIDENCE.md draft, and a stray task-dir scratch file `kda-install-transcript.txt`. Mitigations: every gate re-executed from scratch under this session's instrumentation; .venv-kda destroyed and rebuilt synchronously in the foreground (two raced background rebuilds at 12:39-12:41Z produced provenance-unattributable state); the draft and stray file superseded/removed; full disclosure embedded in EVIDENCE.md.
2. **[Evidence-integrity finding]** The dead sessions' ty-gate claims ("All checks passed!", exit 0 — session-1 scratch log and session-2 draft) are NOT reproducible against the CI-exact stub env on this machine; a pass requires a stub env deviating from CI shape (e.g. pyfaidx present) or a non-CI-exact invocation. Session 1's own `cr6-resolve-linux.txt` shows it had investigated the pyfaidx platform gap. This session's instrumented RED result supersedes those claims; flagged prominently in EVIDENCE.md section C so WINDOWS #27/#18 closeout consumers cannot mistake the draft for evidence.
3. **[External actor deleted .venv-cuda mid-task]** The first .venv-cuda install (12:39-12:41Z, exit 0, torch 2.10.0+cu130) was silently deleted before its device record could be taken (found gone at 14:31Z; no process of this session deleted it; consistent with the concurrent actor executing plan-cleanup steps). Rebuilt synchronously; all section-G results are from the rebuilt venv. The GPU pytest leg additionally self-protects (running interpreter locks venv binaries).
4. **[torch version vs plan expectation]** Plan expected `torch 2.11.x+cu130` (inherited from the older dnallm-cuda env); current pyproject says `torch>=2.4.0,<2.12` and uv deterministically resolved `2.10.0+cu130` in both installs — identical to what CI's test-cuda job resolves from the same pyproject + index routing. The routing mechanism the gate proves (+cu130 via torch-cuda130 index) holds exactly. Recorded in EVIDENCE.md section G.
5. **[uv.lock does not exist]** The constraint list required uv.lock byte-identical; the repo tracks no uv.lock (verified: `git ls-files | grep uv.lock` empty). Trivially satisfied.
6. **[Local substitutions noted in evidence]** conda seeds for setup-python; `python -m pytest` / `ruff.exe` by path for activated-venv bare commands; `$TEMP` stub env for CI's `/tmp`; Windows-native `rmdir` used for venv deletion after MSYS `rm -rf` hit read-only/ACL failures on the inherited venvs.

## Commits

- 00c6128 — `docs(planning): quick 261019-cr6 evidence — windows-local CI replication + KDA fresh-venv declaration (WINDOWS #27 items 1/2/4/6)` (EVIDENCE.md sections A-F, 300 insertions)
- 6d14d9d — `docs(planning): quick 261019-cr6 evidence addendum — test-cuda push leg on real RTX 5080` (section G + F update, 66 insertions)
- Both: no attribution trailers; nothing pushed; `git diff 7bc9ecf..HEAD --stat` = EVIDENCE.md only (zero source-tree changes).

## Evidence File

`D:/Github/DNALLM/.planning/quick/261019-cr6-windows-local-ci-replication-kda-declara/261019-cr6-EVIDENCE.md` (committed, 366 lines): sections A (machine identity + collision disclosure), B (test-windows shape), C (ty hard gate RED + root cause + prior-claim discrepancy), D (fast leg + audit + canary), E (KDA six-item: fresh items 1/2/4/6 + dnallm-cuda 2026-10-19 items 3/5 — torch 2.11.0+cu130, RTX 5080 sm_120, maxdiff 0.000788 — Steps 3/5 explicitly NOT re-run in the fresh venv), F (closing state), G (test-cuda real-GPU leg with the strictly-stronger-than-CI comparison).

## Known Stubs

None. No stubs, no skipped verifications, no unrun verify chains (T2/T3/T4 all GREEN; T1's chain failure is the documented ty gate itself).

## Self-Check: PASSED

- EVIDENCE.md exists at the planned path (committed in 00c6128 + 6d14d9d).
- Both commit hashes verified as ancestors of HEAD.
- `.venv-kda` and `.venv-cuda` both absent; ty stub env removed from $TEMP; `git status --porcelain` empty.
- Source tree byte-identical: only EVIDENCE.md changed across the plan's commits.
