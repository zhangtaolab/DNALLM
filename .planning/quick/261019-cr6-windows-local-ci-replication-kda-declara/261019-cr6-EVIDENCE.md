# 261019-cr6 EVIDENCE — Windows local CI replication + KDA fresh-venv declaration (WINDOWS #27)

All gates below ran on the owner's real Windows machine (DESKTOP-JEIOI9A) from the repo root
`D:\Github\DNALLM` in Git Bash on **2026-10-19**, branch `v1.3`, HEAD
`7bc9ecf8b95abc3698b664d993ac47312b55d31c`. Every command, exit code, and key output line in
sections B-F is quoted verbatim from THIS executor session's captured logs (`$TEMP/*-cr6*.log`).

> **Execution-collision disclosure (read first).** This plan was raced by two earlier
> interrupted executor sessions on the same machine (session 1 ≈02:12-02:30Z, session 2
> ≈02:51-03:29Z, overlapping this session's 02:49Z start; both dead by 03:29Z, nothing
> committed by either). Their leftovers contaminated the working state this session found:
> a pre-replaced `.venv`, a pre-built `.venv-kda`, a stale `pytest-junit.xml`, and an
> untracked EVIDENCE.md draft whose ty-gate claim ("All checks passed! exit 0") is **not
> reproducible** against the CI-exact stub-env shape (see section C). This session
> re-executed every gate from scratch under its own instrumentation, destroyed and rebuilt
> `.venv-kda` synchronously in the foreground after two raced background rebuilds
> (12:39-12:41Z) produced provenance-unattributable state, and replaced the draft entirely.
> Only this session's outputs are quoted below.

## A) Machine identity + repo state

| Fact | Value |
| --- | --- |
| Hostname | DESKTOP-JEIOI9A (Windows 11, Git Bash execution) |
| GPU / driver | `NVIDIA GeForce RTX 5080, 610.88` (`nvidia-smi --query-gpu=name,driver_version --format=csv`) |
| Seed interpreter 3.12 | `Python 3.12.11` at `D:/miniconda3/envs/MinerU/python.exe` |
| Seed interpreter 3.11 | `Python 3.11.13` at `D:/miniconda3/envs/py311/python.exe` |
| uv / uvx | `uv 0.13.1 (62faed335 2026-10-13 x86_64-pc-windows-msvc)` |
| Repo | branch `v1.3`, HEAD `7bc9ecf`, `git status --porcelain` empty at session start |
| Date | 2026-10-19 (all runs this file reports) |

- **Stale `.venv` replaced** (plan Task 1 step 2): the inventory actually found pre-replacement
  was `Python 3.12.11 / ruff 0.16.10 / torch 2.10.0+cpu` — NOT the plan-recorded
  `py3.13.15 / ruff 0.15.8 / torch 2.6.0+cpu`, because a prior interrupted session had already
  replaced it (see disclosure). Recorded as found, then deleted and rebuilt fresh (section B).
- **Local substitutions vs CI** (deterministic equivalents, noted once each): CI's
  `setup-python 3.12` / `setup-python 3.11` replaced by explicit conda seeds in `uv venv`;
  CI's activated-venv bare `pytest`/`ruff` invoked as `.venv/Scripts/python.exe -m pytest` /
  `.venv/Scripts/ruff.exe` (same interpreter/module, no PATH mutation); CI's `/tmp` stub env
  became `$TEMP/ty-stub-env-cr6` (`C:/Users/forre/AppData/Local/Temp/ty-stub-env-cr6`).
- **CRLF parity note**: the working tree is CRLF (local `core.autocrlf=true`, index LF; CI
  checks out with `core.autocrlf false`). Planning-time pre-verification with the repo-pin
  ruff 0.16.10 showed `ruff format --check` green on this exact tree (ruff's auto
  line-ending policy preserves CRLF), so the CRLF/LF difference vs CI is evidence-neutral.
  The checkout was NOT renormalized.

## B) test-windows shape replication (fresh py3.12 `.venv`, `.[full]`, ruff gates)

Venv creation and install (this session, foreground-captured background job, exit 0):

```
$ uv venv .venv --python D:/miniconda3/envs/MinerU/python.exe
Using CPython 3.12.11 interpreter at: D:\miniconda3\envs\MinerU\python.exe
Creating virtual environment at: .venv

$ UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv pip install --python .venv/Scripts/python.exe -e ".[full]"
Resolved 347 packages in 22.70s
Installed 347 packages in 1m 30s
EXIT_CODE=0
```

Version record (dev extra pins `ruff==0.16.10` — resolved exactly; torch is the PyPI Windows
default **CPU** wheel, required for Task 2 skip parity):

```
$ .venv/Scripts/python.exe --version
Python 3.12.11
$ .venv/Scripts/ruff.exe --version
ruff 0.16.10
$ .venv/Scripts/python.exe -c "import torch; print(torch.__version__)"
2.10.0+cpu
```

Gate 1 — format (exit 0):

```
$ PYTHONUTF8=1 .venv/Scripts/ruff.exe format --check .
312 files already formatted
EXIT_CODE=0
```

Gate 2 — lint (exit 0, no output, no violations):

```
$ PYTHONUTF8=1 .venv/Scripts/ruff.exe check . --statistics
EXIT_CODE=0
```

## C) ty hard gate (coverage-gate job shape, pinned stub env) — RED, 1 diagnostic

Stub env at `C:/Users/forre/AppData/Local/Temp/ty-stub-env-cr6` (Python 3.12.11), CI-exact
install (exit 0):

```
$ UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv pip install --python "$TY_STUB/Scripts/python.exe" -e ".[full]" -c .github/workflows/ty-stub-constraints.txt
Resolved 347 packages in 11.41s
Installed 347 packages in ...
EXIT_CODE=0
```

All 12 constraint pins honored exactly (`uv pip show`, this session):

```
torch -> Version: 2.10.0        altair -> Version: 6.3.0
transformers -> Version: 5.17.0 modelscope -> Version: 1.40.2
datasets -> Version: 4.8.4      numpy -> Version: 2.2.0
tokenizers -> Version: 0.23.3   pandas -> Version: 3.0.6
peft -> Version: 0.21.2         pydantic -> Version: 2.14.1
accelerate -> Version: 1.15.0
evaluate -> Version: 0.4.6
```

Gate — **FAILED** (reproduced twice this session; identical output both times):

```
$ uvx ty@0.0.84 check dnallm/ --python "C:/Users/forre/AppData/Local/Temp/ty-stub-env-cr6"
error[unresolved-import]: Cannot resolve imported module `pyfaidx`
   --> dnallm\models\special\enformer_model\data.py:112:14
    |
112 |         from pyfaidx import Fasta  # optional dep, raise-on-use
    |              ^^^^^^^
info: Searched in the following paths during module resolution:
info:   1. D:\Github\DNALLM (first-party code)
info:   2. vendored://stdlib (stdlib typeshed stubs vendored by ty)
info:   3. C:\Users\forre\AppData\Local\Temp\ty-stub-env-cr6\Lib\site-packages (site-packages)
info:   [... win32/pythonwin editable paths elided]
Found 1 diagnostic
EXIT_CODE=1
```

**Root cause (verified this session, no source changes applied — owner-ratified boundary):**
`pyfaidx` is not declared anywhere in `pyproject.toml`; it reaches CI's **Linux** env only as
a transitive dependency of `pygenometracks`, which is Windows-excluded:

```
pyproject.toml:116:  "pygenometracks>=3.9; platform_system != 'Windows'",
```

Dry-run proof on this machine (`uv pip install --dry-run "pygenometracks>=3.9"` resolves
`+ pyfaidx==0.9.0.4` among others; a bare `pybedtools` dry-run pulls `pysam`, NOT pyfaidx —
pygenometracks is the sole carrier). `uv pip show pyfaidx` in both this session's `.venv` and
the stub env: "Package(s) not found". The import at `data.py:112` is lazy (function-body,
raise-on-use) but ty analyzes it statically, so the gate outcome diverges by platform:
**the CI ty hard gate is green on Linux only because a Linux-only transitive dependency
satisfies a statically-analyzed optional import. The Windows replica of the CI-exact command
against the CI-exact stub-env shape is deterministically RED (exit 1).**

**Discrepancy note vs prior sessions:** session 1's scratch log
(`$TEMP/ty-check-suppl-cr6.log`, 02:14Z) and session 2's untracked EVIDENCE draft both claim
`All checks passed!` / exit 0. That result is not reproducible against the CI-exact stub env
on this machine (two runs this session, identical RED output); a pass necessarily involved a
stub env deviating from the CI shape (e.g. `pyfaidx` present) or a non-CI-exact invocation.
Session 1's own `$TEMP/cr6-resolve-linux.txt` (a full Linux-target resolution dump containing
pyfaidx) shows it had investigated exactly this platform gap. This session's instrumented
RED result supersedes those claims.

## D) Fast leg + skip audit + exit-code canary (CPU-torch `.venv` only)

Skip-parity precondition (exit 0 — GPU-less windows-runner shape):

```
$ .venv/Scripts/python.exe -c "import torch, sys; print(torch.__version__); sys.exit(0 if not torch.cuda.is_available() else 1)"
2.10.0+cpu
PRECONDITION_EXIT=0
```

Fast leg (this session, full console at `$TEMP/fastleg-cr6.log`; bare `--cov` activates
pyproject `fail_under = 90` — exit 0 proves the coverage gate held):

```
$ PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -m "not slow" --cov --junitxml=pytest-junit.xml
= 2571 passed, 4 skipped, 75 deselected, 7 warnings, 23 subtests passed in 500.97s (0:08:20) =
FASTLEG_EXIT=0 WALL_SECONDS=517
TOTAL                                   9809    351    96%
```

pytest-junit.xml (this session's run, timestamp 2026-10-19T11:28:18+08:00, hostname
DESKTOP-JEIOI9A — overwrote the prior session's stale artifact):

```
<testsuite name="pytest" errors="0" failures="0" skipped="4" tests="2598" time="499.806" ...>
```

Skip audit (exit 0; skip set identical to CI's):

```
$ PYTHONUTF8=1 .venv/Scripts/python.exe scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml
Skip audit: 4 skipped test(s) in pytest-junit.xml
  allowed    tests.examples.test_examples.TestNotebookExamples::test_notebook_imports[notebooks\\data_prepare\\predict\\predict_data.ipynb] [content] -> 'No import statements found'
  allowed    tests.models.test_plant_helixseek_fla_kernels.TestChunkKdaImportPath::test_chunk_kda_importable_when_fla_installed [environment] -> 'environment-unavailable: flash-linear-attention not installed (PlantHelixSeek KDA kernels absent, models silently degrade, see docs/faq)'
  allowed    tests.utils.test_cuda_compat::test_nvjitlink_soname_registered_on_cuda13 [environment] -> 'SONAME check is Linux only'
  allowed    tests.utils.test_genomic_coords::test_fetch_sequence_path_branch_leaks_no_file_descriptors [environment] -> 'fd accounting needs /proc (Linux only)'
OK: every skip in pytest-junit.xml matches the allowlist
AUDIT_EXIT=0
```

Exit-code canary (exact ci.yml lines 192-208 shape; transient file created and deleted inside
the step, never staged):

```
$ if PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest tests/test_ci_exitcode_canary.py -q -p no:cacheprovider > canary.log 2>&1; then ...
Canary OK: pytest exited non-zero as expected
E   AssertionError: intentional canary failure
FAILED tests/test_ci_exitcode_canary.py::test_ci_exitcode_canary - AssertionE...
============================== 1 failed in 0.18s ==============================
CANARY_RESULT=OK
```

Post-canary: `tests/test_ci_exitcode_canary.py` and `canary.log` deleted; `git status
--porcelain` showed no tracked changes (pytest-junit.xml / .coverage / canary.log are
gitignored). Task 2 verify chain: `T2-GREEN` (exit 0).

## E) KDA six-item Step-6 evidence (WINDOWS #27 declaration proof)

Runbook `WINDOWS-KDA-VERIFY.md` Steps 1/2/4 executed fresh in a `.venv-kda` built
**synchronously in the foreground by this session** (after destroying the prior sessions'
contaminated venvs — see disclosure; the transcript below is this session's
`$TEMP/kda3-install-cr6.log`, a single uninterrupted foreground command chain). Steps 3/5
are cited from the dnallm-cuda run of 2026-10-19 (runbook Step 5 revision at `42b08eb`).
Substitutions vs the PowerShell-worded runbook, noted once: Git Bash equivalents throughout;
the runbook's bare `--python 3.11` was seeded explicitly from
`D:/miniconda3/envs/py311/python.exe` (3.11.13) for determinism.

**Step 1 — fresh venv install (NO manual triton install of any kind; the platform marker
must and does pull it):**

```
$ uv venv .venv-kda --python D:/miniconda3/envs/py311/python.exe
Using CPython 3.11.13 interpreter at: D:\miniconda3\envs\py311\python.exe
Creating virtual environment at: .venv-kda

$ UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv pip install --python .venv-kda/Scripts/python.exe -e ".[fla]"
Using Python 3.11.13 environment at: .venv-kda
Resolved 192 packages in 16.72s
Installed 192 packages in 1m 32s
 + fla-core==0.5.2
 + flash-linear-attention==0.5.2
 + torch==2.11.0
 + triton-windows==3.7.1.post27
KDA3_FULL_EXIT=0
```

The `.[fla]` Windows platform marker (`triton-windows>=3.7.1.post27,<3.8; platform_system ==
'Windows'`, pyproject.toml:189) auto-resolved `triton-windows==3.7.1.post27` — visible in
the install resolution above; no other command in this session's transcript installs any
triton package.

**Step 2 — triton-windows auto-resolution version (exit 0):**

```
$ uv pip show --python .venv-kda/Scripts/python.exe triton-windows
Name: triton-windows
Version: 3.7.1.post27
Location: D:\Github\DNALLM\.venv-kda\Lib\site-packages
```

**Step 4 — KDA import contract (exit 0; the two UserWarnings are the known benign
Windows fla/torch import notices):**

```
$ .venv-kda/Scripts/python.exe -c "from fla.ops.kda.chunk import chunk_kda; print('import OK:', callable(chunk_kda))"
import OK: True
```

Fresh-venv torch build, for the record: `2.11.0+cpu` (CPU — expected and sufficient; this
venv exists to prove the declaration, not kernel execution).

**EXPLICIT six-item combination statement (runbook Step 6).** The six required items are
satisfied by TWO deliberately combined sources, stated plainly:

| Item | Source | Evidence |
| --- | --- | --- |
| 1. venv path + creation command | **fresh `.venv-kda` (this session)** | `uv venv .venv-kda --python D:/miniconda3/envs/py311/python.exe` (Step 1 above) |
| 2. triton-windows version | **fresh `.venv-kda` (this session)** | `Version: 3.7.1.post27`, auto-resolved (Steps 1-2 above) |
| 3. torch version + CUDA build | **dnallm-cuda, 2026-10-19** | `torch 2.11.0+cu130`, RTX 5080 `sm_120` |
| 4. import result | **fresh `.venv-kda` (this session)** | `import OK: True` (Step 4 above) |
| 5. kernel evidence | **dnallm-cuda, 2026-10-19** | fused-gate chunk_kda compile+execute probe, `maxdiff 0.000788` vs fp32 reference (runbook Step 5 revision at `42b08eb`) |
| 6. date | both | 2026-10-19 |

Steps 3/5 were **NOT** re-run in the fresh `.venv-kda`: the fresh venv intentionally carries
CPU torch (it exists to prove the platform-marker declaration, not kernel execution), and the
kernel/`+cu130` proofs were already established on this same machine and GPU in dnallm-cuda
on 2026-10-19 per the runbook's Step 5 revision note. Items 1/2/4/6 (declaration: marker
auto-pulls triton-windows and the import contract holds in a stock fresh venv) + items 3/5
(kernels: torch/CUDA build and compiled-kernel correctness on the RTX 5080) together close
the six-item contract.

## F) Closing state (Task 3 close)

- Deleted after evidence capture: `.venv-kda` (gitignore-gap venv, delete-after-capture
  discipline) and the ty stub env `C:/Users/forre/AppData/Local/Temp/ty-stub-env-cr6`; the
  prior sessions' untracked scratch copy `kda-install-transcript.txt` in this task directory
  was removed (its content was the raced session 2 rebuild, superseded by section E).
- Kept: `.venv` (gitignored name; standing local env), `pytest-junit.xml` / `.coverage`
  (gitignored Task 2 artifacts from this session's run).
- `git status --porcelain` lists only this evidence file before its commit; zero
  modifications to `pyproject.toml`, `.github/**`, `dnallm/**`, `tests/**`, or `scripts/**`
  (`uv.lock` does not exist in this repo). Working tree CRLF untouched.
- Task 4 (test-cuda leg on the real RTX 5080) appends section G and updates this section in
  a follow-up commit.
