---
phase: quick-261001-ith
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/ci.yml
autonomous: true
requirements: []
user_setup: []
estimate:
  tokens: 30000
  raw_tokens: 30000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - "The test-mamba job executes its install and pytest steps on the [self-hosted, dnallm-nightly] runner when triggered by schedule or workflow_dispatch (live dispatch run shows gpu-check has_gpu=true and the mamba install step running) — it no longer no-op skips every step on GPU-less hosted runners"
    - "On push and pull_request events test-mamba is skipped and no protected job changes behavior — coverage-gate and coverage-nightly are structurally identical to their pre-change versions (automated deep-compare)"
    - "The deploy job still deploys docs on push to main/master — its needs list is [test, test-cuda] with the now event-gated test-mamba removed"
    - "A hung or pathological CUDA kernel build cannot hold the single self-hosted runner indefinitely — job-level timeout-minutes backstop of 180"
    - "No pull_request-triggered job runs on the self-hosted runner — fork and same-repo PR code never executes on the GPU box (guard excludes pull_request)"
  artifacts:
    - ".github/workflows/ci.yml — test-mamba job header edited (runs-on, if, timeout-minutes, rationale comments); deploy needs trimmed"
  key_links:
    - "deploy needs = [test, test-cuda] — GitHub skips dependents of skipped needed jobs, so leaving the event-gated test-mamba in deploy's needs would silently stop gh-pages doc deploys"
    - "test-mamba runs-on labels [self-hosted, dnallm-nightly] — the same runner/labels already serving coverage-nightly in this file"
---

<objective>
Move the CI `test-mamba` job in `.github/workflows/ci.yml` onto the self-hosted GPU runner
(`[self-hosted, dnallm-nightly]`) so it actually executes instead of no-op skipping on GPU-less
hosted runners, and add a `timeout-minutes` backstop appropriate for the CUDA kernel build.

Purpose: today `test-mamba` runs on `ubuntu-latest`, where the `nvidia-smi` gpu-check gate makes
every step skip — the job provides zero signal. The self-hosted GB10 (aarch64) box already serves
`coverage-nightly` under the same labels in this same file.

Output: a minimal, behavior-faithful edit of `.github/workflows/ci.yml` plus a live
`workflow_dispatch` validation run proving the job executes on the GPU runner.

**Design point resolved: nightly cadence (`schedule || workflow_dispatch`), not `push || pull_request`.**
Justification:
1. Security posture — this repo's documented invariant (see the comment block on
   `coverage-nightly`) is that PR jobs stay on hosted runners so PR-authored code (including
   forks) never executes on the self-hosted box with its warm caches and live GPU. Keeping
   `pull_request` in the guard would break that invariant.
2. Architecture decision GATE-02 (amended) — PR/push is the fast leg; heavy/long work belongs to
   nightly cadence. The `.[mamba]` install (`--no-cache-dir --no-build-isolation` into a fresh
   `uv venv`) compiles mamba-ssm + causal_conv1d CUDA kernels from source on *every* run, not
   just the first — far too heavy for per-push cadence on the single runner box.
3. `workflow_dispatch` preserves a manual calibration trigger, the same pattern used by the
   Phase 04-02 nightly calibration (dispatch run 36747594207).
The GPU-check step stays as a fail-safe no-op per the task brief: if the box ever loses its GPU,
the job green-no-ops instead of failing the nightly run.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.github/workflows/ci.yml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Re-target test-mamba to the self-hosted runner with nightly cadence, timeout backstop, and deploy needs fix</name>
  <files>.github/workflows/ci.yml</files>
  <action>
Edit ONLY two regions of `.github/workflows/ci.yml` — the `test-mamba` job header (job starts at
the `test-mamba:` key, currently ~line 183) and the `deploy` job's `needs` line (currently
~line 390). Do not touch any step of test-mamba (gpu-check, setup-python, uv install, venv +
mamba install, continue-on-error pytest step, artifact upload all stay byte-identical) and do not
touch the coverage-gate or coverage-nightly jobs at all.

In the `test-mamba` job header, make exactly three key changes plus comments:
1. `runs-on: ubuntu-latest` becomes `runs-on: [self-hosted, dnallm-nightly]` — the same label
   set `coverage-nightly` already uses for this box.
2. The `if:` guard `github.event_name == 'push' || github.event_name == 'pull_request'` becomes
   `github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'` (nightly
   cadence, per the design-point resolution in the objective).
3. Add `timeout-minutes: 180` immediately after the `if:` line. Sizing rationale for the
   comment: the recurring per-run CUDA kernel source build is the long pole (the
   `--no-cache-dir` flag plus a fresh `uv venv` per run bypass all wheel caching, so the
   mamba-ssm/causal_conv1d compile repeats every run, not just the first); 180 minutes covers a
   worst-case GB10 build plus the `.[test,dev]` install plus the `not slow` census with
   headroom. Self-hosted jobs have no platform time cap, so this explicit backstop is the only
   protection against a hung build monopolizing the single runner that also serves
   coverage-nightly.
4. Add a short comment block above `runs-on:` in the style of the coverage-nightly comment,
   stating: self-hosted GPU box on nightly cadence per GATE-02 amended (heavy recurring kernel
   build belongs to nightly, PR/push stays the fast leg); schedule/dispatch-only preserves the
   posture that PR code (including forks) never executes on this runner; the gpu-check step
   below stays as a fail-safe no-op.

In the `deploy` job, change `needs: [test, test-cuda, test-mamba]` to
`needs: [test, test-cuda]`. WHY (this edit is required, not optional): GitHub Actions skips a
dependent job when a needed job fails OR IS SKIPPED unless the dependent uses an
`always()`-style conditional — and `deploy`'s `if:` does not. Once test-mamba is event-gated to
schedule/dispatch it is skipped on every push, which would silently stop the gh-pages docs
deployment on push to main/master. Since test-mamba can no longer produce push-event signal,
keeping it in deploy's needs is both broken and semantically empty. The deploy job is not one
of the protected gate/nightly census jobs, so this edit is in scope. Branch protection
(verified live on main and dev) requires only `coverage-gate (py3.12, fast leg)` — test-mamba is
not a required check, so re-cadencing it cannot block PR merges.

Run the verify BEFORE committing so the deep-compare compares the working tree against the
pre-change HEAD.
  </action>
  <verify>
    <automated>python3 - &lt;&lt;'PY'
import subprocess, yaml
new = yaml.safe_load(open('.github/workflows/ci.yml'))
old = yaml.safe_load(subprocess.run(['git','show','HEAD:.github/workflows/ci.yml'],capture_output=True,text=True).stdout)
tm = new['jobs']['test-mamba']
assert tm['runs-on'] == ['self-hosted', 'dnallm-nightly'], tm['runs-on']
assert tm['timeout-minutes'] == 180, tm['timeout-minutes']
cond = tm['if']
assert "schedule" in cond and "workflow_dispatch" in cond, cond
assert "push" not in cond and "pull_request" not in cond, cond
assert new['jobs']['deploy']['needs'] == ['test', 'test-cuda'], new['jobs']['deploy']['needs']
assert set(new['jobs']) == set(old['jobs']), 'job set changed'
for j in old['jobs']:
    if j not in ('test-mamba', 'deploy'):
        assert new['jobs'][j] == old['jobs'][j], f'{j} changed'
strip = lambda d: {k: v for k, v in d.items() if k != 'jobs'}
assert strip(new) == strip(old), 'top-level workflow keys (triggers, permissions) changed'
print('OK: test-mamba header moved, deploy needs trimmed, all other jobs + triggers identical')
PY</automated>
  </verify>
  <done>
ci.yml parses as YAML; test-mamba has runs-on [self-hosted, dnallm-nightly],
if restricted to schedule/workflow_dispatch, timeout-minutes 180; every test-mamba step and both
protected jobs (coverage-gate, coverage-nightly) plus all top-level workflow keys deep-equal
their pre-change versions; deploy needs is exactly [test, test-cuda].
  </done>
</task>

<task type="auto">
  <name>Task 2: Live dispatch validation — prove test-mamba executes on the GPU runner</name>
  <files>(none — validation only; records evidence in the SUMMARY)</files>
  <precondition>
Task 1's change is committed and pushed to origin/dev (quick-flow default), and `gh auth status`
shows an authenticated account (forrestzhang) able to dispatch workflows on
zhangtaolab/DNALLM.
  </precondition>
  <action>
Trigger a manual run and prove the defect is fixed — the job must EXECUTE on the self-hosted
runner rather than no-op skip.

1. Trigger: run `gh workflow run ci.yml --ref dev` (the workflow declares `workflow_dispatch`
   with no inputs, so no `-f` flags). Note: a dispatch also queues coverage-nightly (same event
   guard), and the single runner serializes jobs — test-mamba may sit queued behind the nightly
   census first.
2. Identify the run: poll `gh run list --workflow=ci.yml --event=workflow_dispatch --branch=dev -L 1 --json databaseId,status,createdAt` until a run newer than your trigger appears; capture the run id.
3. Poll for execution (bounded wait, ~60s interval, up to 60 minutes, early-exit on success):
   `gh api repos/zhangtaolab/DNALLM/actions/runs/<run_id>/jobs` — find the job whose name
   starts with `test-mamba` (matrix renders it as `test-mamba (3.11)`). Success criteria: the
   job's `labels` include `self-hosted` and `dnallm-nightly`, its status reaches `in_progress`,
   the `Check for GPU` step completed with `has_gpu=true` in its output, and the venv/mamba
   install step started. Once the install step is verifiably running, the original defect
   (always fully skipped on GPU-less hosted runners) is proven fixed — do NOT block the task on
   full census completion (first kernel build can run close to the 180-minute ceiling).
4. If the 60-minute window expires while the job is still queued behind coverage-nightly:
   record the run id and URL in the SUMMARY as pending-live-proof and note the nightly schedule
   will exercise it within 24h — the static guarantees from Task 1 still hold.
5. Failure handling: if the job FAILS at the CUDA source build (e.g. toolchain or compute-
   capability issue on GB10/sm_121), that is real signal this task surfaces — capture
   `gh run view <run_id> --log-failed | head -100` in the SUMMARY for follow-up. Do not attempt
   to fix the mamba build in this task (out of scope; the pytest step is continue-on-error and
   the artifact upload captures pytest.log).
6. Record in the SUMMARY: run id, run URL, observed job states for the whole run (expected on
   dispatch: test-mamba and coverage-nightly non-skipped; test, test-cuda, coverage-gate,
   deploy skipped by their event guards), runner labels, and whether the install step was
   observed running.
  </action>
  <verify>
    <automated>RUN_ID=$(gh run list --workflow=ci.yml --event=workflow_dispatch --branch=dev -L 1 --json databaseId --jq '.[0].databaseId') && echo "run=$RUN_ID" && gh api repos/zhangtaolab/DNALLM/actions/runs/$RUN_ID/jobs --jq '.jobs[] | select(.name | startswith("test-mamba")) | {name: .name, status: .status, conclusion: .conclusion, labels: .labels}' && ! gh api repos/zhangtaolab/DNALLM/actions/runs/$RUN_ID/jobs --jq '.jobs[] | select(.name | startswith("test-mamba")) | .conclusion' | grep -qx skipped</automated>
  </verify>
  <done>
A workflow_dispatch run on dev shows the test-mamba job assigned to a runner with labels
self-hosted + dnallm-nightly and NOT skipped — either in_progress/completed with the GPU-check
step reporting has_gpu=true and the mamba install step started, or (timeout-window case) queued
behind coverage-nightly with the run id and URL recorded in the SUMMARY. Any build failure is
captured with logs for follow-up rather than silently dropped.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| GitHub event plane → self-hosted runner | Workflow/test code executes on the owner's GPU box with warm HF/ModelScope caches in $HOME and a live GPU |
| Shared single-runner resource | One box (dnallm-nightly) serialized between coverage-nightly and test-mamba |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261001-01 | Elevation of privilege (untrusted code execution) | test-mamba event guard | critical | mitigate | Guard restricts to schedule + workflow_dispatch only; pull_request/push excluded so PR-authored code (incl. forks) never executes on the self-hosted box — matches the documented coverage-nightly posture. Verified live: branch protection requires only coverage-gate, so no required-check bypass. |
| T-261001-02 | DoS (resource exhaustion) | recurring CUDA kernel build on the single shared runner | medium | mitigate | timeout-minutes: 180 bounds a hung build (no platform cap on self-hosted); nightly-only cadence keeps the box available for coverage-nightly; gpu-check fail-safe no-ops the job if the GPU disappears. |
| T-261001-03 | Tampering / silent regression | deploy job needs + protected jobs | medium | mitigate | deploy needs trimmed to [test, test-cuda] (skipped-need would silently stop gh-pages deploys); Task 1 automated deep-compare proves coverage-gate and coverage-nightly plus all top-level workflow keys identical to pre-change. |
</threat_model>

<verification>
- Static (Task 1): YAML parses; structural asserts on runs-on/if/timeout-minutes/deploy-needs;
  every other job and all top-level workflow keys deep-equal the pre-change HEAD version.
- Live (Task 2): workflow_dispatch run on dev shows test-mamba on the self-hosted
  dnallm-nightly runner, gpu-check has_gpu=true, mamba install step executing.
- The repo's `python3` has PyYAML (verified) and `gh` is authenticated (verified).
</verification>

<success_criteria>
- test-mamba executes for real on the GB10 self-hosted runner at nightly/dispatch cadence with a
  180-minute backstop — no more permanent no-op skipping.
- coverage-gate and coverage-nightly are byte-identical in behavior (automated deep-compare).
- Docs deploys on push to main/master keep working (deploy needs trimmed).
- PR/fork code never executes on the self-hosted runner.
</success_criteria>

<output>
Create `.planning/quick/261001-ith-move-test-mamba-ci-job-to-self-hosted-gp/261001-ith-SUMMARY.md` when done
</output>
