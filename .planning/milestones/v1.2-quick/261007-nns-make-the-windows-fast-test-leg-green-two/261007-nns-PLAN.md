---
phase: quick-261007-nns
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/models/test_special/test_megadna.py
  - tests/expected_skips.yaml
autonomous: true
requirements:
  - QUICK-261007-NNS-01
user_setup: []

estimate:
  tokens: 11000
  raw_tokens: 7000
  tasks: 2
  confidence: med

must_haves:
  truths:
    - "All 9 parametrized cases of TestMegadnaCheckpointSelection::test_member_selects_its_intended_checkpoint pass on Windows AND Linux because the expected torch.load path is built with os.path.join (OS-native separators), matching how the library builds it"
    - "tests/expected_skips.yaml allowlists the Linux-only skip reason of tests/utils/test_genomic_coords.py::test_fetch_sequence_path_branch_leaks_no_file_descriptors, so scripts/audit_skips.py exits 0 on the Windows fast leg once the 9 assertions pass"
    - "tests/utils/test_genomic_coords.py is untouched; on Linux the fd-accounting test still runs and passes (not skipped)"
    - "tests/scripts/test_audit_skips.py contract tests stay green (the new entry is structurally valid: exactly one matcher key plus category)"
  artifacts:
    - "tests/models/test_special/test_megadna.py: import os added to the import block; the checkpoint assertion reads os.path.join(\"/snapshot\", expected_checkpoint) with a one-line comment noting separators are an OS detail"
    - "tests/expected_skips.yaml: new documented entry reason_like: \"fd accounting needs /proc\" with category environment, placed after the SONAME entry, comment mirroring house style"
  key_links:
    - "test expected path <-> dnallm/models/special/megadna.py:155 os.path.join(downloaded_model_path, full_model_name) (both sides OS-native, so equality holds on posix and nt)"
    - "Windows junit <skipped message> 'fd accounting needs /proc (Linux only)' <-> allowlist reason_like substring matcher <-> scripts/audit_skips.py main() exit 0 (test-windows audit step)"
---

<objective>
Make the Windows fast-test leg green by fixing the two remaining Windows-only failures in one quick task: (1) the 9 parametrized checkpoint-selection assertions in tests/models/test_special/test_megadna.py hardcode the posix separator in an f-string join of the snapshot fixture path and the checkpoint filename, while the library correctly builds the torch.load path with os.path.join (backslash on Windows is valid local-file IO — the library must NOT change); (2) the Linux-only skip in tests/utils/test_genomic_coords.py fires on the Windows leg with no tests/expected_skips.yaml entry, so scripts/audit_skips.py would fail the job right after the 9 assertions turn green.

Purpose: The Windows leg (test-windows, py3.12) currently fails at the "Run fast tests" step with 9 failed / 1921 passed (run 37593859969), all `AssertionError: assert ['/snapshot\\...'] == ['/snapshot/...']`. The install step was already fixed by quick task 261007-mhz and ruff steps are green; this task closes the fast-tests and skip-audit steps so the leg's full pipeline passes.

Output: A two-file change. The megadna assertion becomes OS-native via os.path.join (identical string on Linux — all 9 stay green locally), and the skip allowlist gains one documented reason_like entry. The library file dnallm/models/special/megadna.py is NOT touched.

Planner decision (delegated in the task brief, allowlist entry vs typed-prefix reword): ADD THE ALLOWLIST ENTRY. Rationale: (a) exact in-file precedent for this class — the other Linux-only skipif in tests/utils (test_cuda_compat.py:26-27, "SONAME check is Linux-specific") is allowlisted via reason_like/category environment, not reworded; (b) the typed `environment-unavailable:` prefix belongs to the Phase-8 evidence-backed skip helpers in tests/examples/_execution.py (D-06 contract: typed skips only with recorded evidence) — borrowing it for a plain OS-property skipif would dilute that convention; (c) smaller diff: one data entry, zero test-behavior change.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@tests/models/test_special/test_megadna.py
@tests/expected_skips.yaml

Live-tree facts observed at planning time (2026-10-07, branch phs):

- tests/models/test_special/test_megadna.py: import block is lines 18-22 (`import torch`, `import pytest`, `from unittest.mock import Mock, patch`, blank, `from dnallm.models.special.megadna import _handle_megadna_models`) — `os` is NOT imported. The failing assertion is line 178 inside `TestMegadnaCheckpointSelection::test_member_selects_its_intended_checkpoint` (parametrize at lines 146-158, 9 cases; the `_snapshot` fixture path `("/snapshot", None)` at line 172).
- dnallm/models/special/megadna.py:155 builds `downloaded_model_path = os.path.join(downloaded_model_path, full_model_name)` before torch.load at lines 165/167 — the library is correct on every OS; DO NOT EDIT IT.
- Baseline on this box: `.venv/bin/pytest tests/models/test_special/test_megadna.py::TestMegadnaCheckpointSelection -q` = 9 passed in 3.72s (posix: `os.path.join("/snapshot", "x")` == `"/snapshot/x"`, so the fix is string-identical on Linux).
- tests/utils/test_genomic_coords.py:261-263 carries `@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="fd accounting needs /proc (Linux only)")` on `test_fetch_sequence_path_branch_leaks_no_file_descriptors` (def at line 264). /proc exists on Linux so the test RUNS there; it skips only on Windows. The file is NOT edited by this plan.
- tests/expected_skips.yaml: `allowed:` list ends at lines 68-70 with the structural twin of this case — `reason_like: "SONAME check is Linux-specific"` / `category: environment` under the comment `# tests/utils/test_cuda_compat.py:26 — non-Linux legs only (CI is linux).` The new entry mirrors that shape.
- scripts/audit_skips.py: `reason_like` is substring containment (entry_matches, line 63); `load_allowlist` rejects any entry without exactly one non-empty matcher key plus category (lines 37-45). Planning-time proof on this box: the proposed entry matches the bare reason AND decorated junit forms ("SKIPPED [1] ... : fd accounting needs /proc (Linux only)"), no existing entry matches it (non-redundant), and the current 13-entry allowlist loads clean. tests/scripts/test_audit_skips.py contract tests use only tmp_path fixtures — they never load the real yaml, so a new entry cannot redden them.
- Scope check: only two Linux-only skipifs exist in tests/ (`grep -rn "proc/self\|sys.platform\|platform.system" tests/`); the cuda_compat one is already allowlisted, the genomic-coords one is this task's entry — no third gap hides behind these two fixes.
- Out of scope (no edits): dnallm/models/special/megadna.py (library correct), tests/utils/test_genomic_coords.py (skip reason text stays as-is per the allowlist decision), .github/workflows/ci.yml (the Windows leg already runs fast tests + audit; nothing to rewire).
- Remote proof: test-windows runs on push to phs; no local Windows runtime exists on this box, so the leg itself closes on the next push (baseline failing run 37593859969, job test-windows py3.12, step "Run fast tests": 9 failed / 1921 passed).
</context>

<tasks>

<task type="auto">
  <name>Task 1: Make the megadna checkpoint-path assertion OS-native (test-side only)</name>
  <files>tests/models/test_special/test_megadna.py</files>
  <action>
    In tests/models/test_special/test_megadna.py, add `import os` to the import block as its first line (above `import torch` at line 18) — the file currently does not import os. Replace the assertion at line 178 — the f-string equality that hardcodes a forward-slash join of the /snapshot fixture directory and expected_checkpoint (the sole remaining such literal in the file; the Task 1 verify gate asserts its absence afterwards) — with the OS-native equality `assert loaded_paths == [os.path.join("/snapshot", expected_checkpoint)]`, and add one English comment line directly above the assert noting that path separators are an OS detail because the handler builds the torch.load path with os.path.join (Windows uses backslashes). This mirrors the library's own construction at dnallm/models/special/megadna.py:155, so both sides of the equality are OS-native and the assertion holds on posix and nt alike. Do NOT touch dnallm/models/special/megadna.py — the backslash path it produces on Windows is valid local-file IO and the library is correct (the 9 CI failures are the TEST hardcoding the posix separator, per run 37593859969: `assert ['/snapshot\\...'] == ['/snapshot/...']`). Change nothing else in the file (parametrize rows, fixtures, other test classes all stay byte-identical); keep ruff line-length 100 and double quotes.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/models/test_special/test_megadna.py -q && .venv/bin/python -c 'import ntpath, posixpath; win = ntpath.join("/snapshot", "megaDNA_phage_145M.pt"); lin = posixpath.join("/snapshot", "megaDNA_phage_145M.pt"); assert win == "/snapshot" + chr(92) + "megaDNA_phage_145M.pt", win; assert lin == "/snapshot/megaDNA_phage_145M.pt", lin; print("ntpath form:", win, "| posixpath form:", lin)' && ! grep -qF 'f"/snapshot/{' tests/models/test_special/test_megadna.py</automated>
  </verify>
  <done>
    All 20 tests in tests/models/test_special/test_megadna.py pass on Linux (the 9 checkpoint-selection cases included — os.path.join is string-identical to the old f-string under posixpath); the ntpath/posixpath proof prints both separator forms and asserts the Windows form is backslash-joined, establishing that the assertion now expects exactly what the handler builds on Windows; the hardcoded posix f-string is gone from the file; dnallm/models/special/megadna.py is untouched (git diff shows only the test file).
  </done>
</task>

<task type="auto">
  <name>Task 2: Allowlist the Linux-only fd-accounting skip for the Windows audit step</name>
  <files>tests/expected_skips.yaml</files>
  <action>
    In tests/expected_skips.yaml, append one entry at the end of the `allowed:` list (immediately after the "SONAME check is Linux-specific" block at lines 68-70), mirroring that entry's house style — a comment line citing the source test and leg, then the matcher and category: comment `# tests/utils/test_genomic_coords.py:261 — non-Linux legs only (fd accounting` / `# reads /proc/self/fd; fires on the test-windows fast leg).` followed by `- reason_like: "fd accounting needs /proc"` and `category: environment`. Use reason_like (substring) rather than exact because junit may decorate skipif reasons (file header, Pitfall 6), and do NOT reword the test's skip reason to the `environment-unavailable:` typed prefix — that prefix is reserved for the Phase-8 evidence-backed skip helpers in tests/examples/_execution.py per the D-06 contract (planner decision recorded in the objective). This is the smaller diff and follows the structural twin precedent two lines above. Change no other entry; the file remains a reviewable suppression record (no wildcards, no empty matchers).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/scripts/test_audit_skips.py -q && .venv/bin/python -m pytest "tests/utils/test_genomic_coords.py::test_fetch_sequence_path_branch_leaks_no_file_descriptors" -q && .venv/bin/python -c 'import importlib.util; spec = importlib.util.spec_from_file_location("audit_skips", "scripts/audit_skips.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); entries = m.load_allowlist("tests/expected_skips.yaml"); msg = "fd accounting needs /proc (Linux only)"; hits = [e for e in entries if m.entry_matches(msg, e)]; assert len(hits) == 1 and hits[0]["reason_like"] == "fd accounting needs /proc" and hits[0]["category"] == "environment", hits; print("exactly one allowlist entry matches:", hits[0])' && .venv/bin/python -c 'import importlib.util, pathlib, tempfile; q = chr(34); spec = importlib.util.spec_from_file_location("audit_skips", "scripts/audit_skips.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); xml = f"<?xml version={q}1.0{q}?><testsuite name={q}pytest{q} tests={q}1{q} skipped={q}1{q}><testcase classname={q}tests.utils.test_genomic_coords{q} name={q}test_fetch_sequence_path_branch_leaks_no_file_descriptors{q} time={q}0.001{q}><skipped type={q}pytest.skip{q} message={q}fd accounting needs /proc (Linux only){q}/></testcase></testsuite>"; p = pathlib.Path(tempfile.mkdtemp()) / "win-junit.xml"; p.write_text(xml); assert m.main(str(p), "tests/expected_skips.yaml") == 0, "Windows-leg skip not allowlisted"; print("synthetic Windows junit audit: exit 0")'</automated>
  </verify>
  <done>
    tests/scripts/test_audit_skips.py passes (contract tests unaffected); the fd-accounting test still runs and passes on Linux (no skip, no behavior change); load_allowlist accepts the edited file and exactly one entry — reason_like "fd accounting needs /proc" / category environment — matches the skip reason; the synthetic Windows junit (one skipped testcase carrying the decorated reason) audits clean through scripts/audit_skips.py main() with exit 0, proving the test-windows audit step passes once the fast tests are green; tests/utils/test_genomic_coords.py is untouched (git diff shows only tests/expected_skips.yaml).
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CI gate data | tests/expected_skips.yaml decides which junit skips the CI hard gate (scripts/audit_skips.py) treats as expected |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261007-nns-01 | Tampering | tests/expected_skips.yaml allowlist breadth | low | mitigate | The entry uses the narrowest practical matcher for one static skipif reason (reason_like on the specific substring "fd accounting needs /proc", tied to tests/utils/test_genomic_coords.py:261); load_allowlist structurally rejects malformed/wildcard entries; the verify step proves exactly one entry matches the message, so the gate is widened by precisely one deterministic skip and no other message silently becomes allowed |
| T-261007-nns-02 | Tampering | tests/models/test_special/test_megadna.py checkpoint assertion | low | mitigate | Both sides of the equality now construct the path with the same OS-native os.path.join, so the assertion can no longer encode a platform assumption that passes on Linux while failing on Windows; the full 20-test file staying green pins the checkpoint-selection semantics (WR-02) unchanged |

No packages are installed or declared by this plan (two test-tree files only), so the package-legitimacy gate and the reserved `T-{phase}-SC` row do not apply.
</threat_model>

<verification>
- Task 1: `.venv/bin/python -m pytest tests/models/test_special/test_megadna.py -q` = 20 passed on Linux (includes the 9 previously Windows-failing cases; os.path.join is string-identical to the old f-string under posixpath); the ntpath/posixpath one-liner proves the expected value now equals the OS-native form on both platforms; `git diff --name-only` shows only the test file.
- Task 2: audit contract tests green; the fd-accounting test still passes (not skipped) on Linux; the exactly-one-match proof and the synthetic Windows junit through `main()` prove the audit step exits 0 for this skip before the push.
- Full-file grep confirms no other hardcoded posix snapshot literal remains: `grep -n "snapshot" tests/models/test_special/test_megadna.py` shows the fixture path `("/snapshot", None)` and the os.path.join assertion only.
- Remote proof (no local Windows box): the next push to phs runs test-windows — expected outcome: fast-tests step 1930 passed / 0 failed (1921 + 9 repaired) and the subsequent audit step exits 0; baseline failing run 37593859969.
</verification>

<success_criteria>
- tests/models/test_special/test_megadna.py builds its expected checkpoint path with `os.path.join("/snapshot", expected_checkpoint)` behind a one-line separator comment, with `import os` added; all 20 tests in the file pass locally on Linux; the library file dnallm/models/special/megadna.py is byte-identical.
- tests/expected_skips.yaml gains exactly one documented entry (reason_like "fd accounting needs /proc", category environment) mirroring the SONAME precedent; the allowlist loads clean; the synthetic Windows junit audits exit 0.
- tests/utils/test_genomic_coords.py and .github/workflows/ci.yml are untouched.
- Two atomic commits (one per task); the Windows fast-test leg is expected green on the next phs push.
</success_criteria>

<output>
Create `.planning/quick/261007-nns-make-the-windows-fast-test-leg-green-two/261007-nns-SUMMARY.md` when done
</output>
