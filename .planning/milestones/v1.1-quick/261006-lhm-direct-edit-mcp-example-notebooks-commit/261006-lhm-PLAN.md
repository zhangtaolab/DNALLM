---
phase: 261006-lhm
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/test_runner_infra_contracts.py
  - example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_pydantic_ai.md
  - docs/example/mcp_langchain.md
  - scripts/runner/README.md
  - .github/workflows/ci.yml
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
autonomous: true
requirements:
  - D-11-SWAP
estimate:
  tokens: 30000
  raw_tokens: 16000
  tasks: 3
  confidence: med

must_haves:
  truths:
    - "Both committed notebooks reference the new model: example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb has a code cell containing model_name='qwen3.5:4b' and its markdown preamble names qwen3.5:4b in both the availability prose and the pull command; example/mcp_example/mcp_client_ollama_langchain_agents.ipynb has a code cell containing \"ollama:qwen3.5:4b\"."
    - The string qwen3.8 appears in NONE of: the two example notebooks, the two docs/example/mcp_example/*.ipynb copies, the two md pages, scripts/runner/README.md (verified at planning time that committed OUTPUTS contain zero such lines, so a whole-file assertion is safe).
    - ".venv/bin/python scripts/check_notebook_md_sync.py exits 0 (24/24 pairs) and .venv/bin/python scripts/check_docs_sync.py exits 0 after the change."
    - New contract class TestMcpExampleModelSwap in tests/test_runner_infra_contracts.py shows RED (fails at test level against the pre-swap tree) then GREEN (passes after Task 2), with both runs recorded in the SUMMARY per the 06-02 RED_EVIDENCE_OK convention.
    - ".venv/bin/python -m pytest tests/mcp tests/test_runner_infra_contracts.py -q is fully green (kernel-free; no notebook execution, no live server needed)."
    - scripts/runner/README.md ops path names qwen3.5:4b in the pull step and the /api/tags verify comment, its num_ctx narrative cites the new model metadata (4.2B Q4_K_M, ~3.3GB pull, default context length 262144 = 256k-class), and the D-12 loopback / never-0.0.0.0 wording is byte-preserved.
    - .github/workflows/ci.yml differs from its pre-task state by EXACTLY one comment line (line ~975, the qwen3.8 17GB settle note); no step logic, keys, or other lines change; the edit is skipped (and recorded as skipped) if the file is not porcelain-clean immediately before editing.
    - No systemd/ollama live configuration is touched, no model caches are deleted, no GH runs are dispatched or cancelled, the 15:24 CST capability probe is NOT re-run (cited, not reproduced).
    - Branch phs is pushed and origin/phs equals local HEAD at the end.
  artifacts:
    - tests/test_runner_infra_contracts.py — new TestMcpExampleModelSwap class (3 tests) + one model-swap note line in the module docstring.
    - The six mirror-set content files (2 example notebooks, 2 docs ipynb byte-copies, 2 docs md pages) all carrying qwen3.5:4b.
    - scripts/runner/README.md — swapped pull/verify ops and restated num_ctx/loopback narrative.
    - Comment-only updates at tests/examples/_execution.py (~243-258), tests/examples/test_notebook_execution.py (~1030, ~1066, ~1223).
    - STATE.md Phase-09 decision bullet recording the D-11 re-decision; 261006-lhm-SUMMARY.md with output-provenance note.
  key_links:
    - example notebook code literal -> docs/example/mcp_example/*.ipynb byte-identical copy (filecmp gate in scripts/check_docs_sync.py) -> docs md page python block (AST statement gate in scripts/check_notebook_md_sync.py).
    - scripts/runner/README.md pull/verify op -> the runner box's pulled model -> nightly stage-3 gated mcp pair (tests/examples/test_notebook_execution.py GATED_NOTEBOOKS) executing against qwen3.5:4b.
    - TestMcpExampleModelSwap raw-text/JSON assertions -> every content file in this change (the pin that makes a future editorial revert fail in seconds, D-07 same-change test contract).
---

<!-- planner-discipline-allow: qwen3.8 -->
<!-- Rationale: the old model id is the literal this plan replaces, so it must be nameable in
     action prose; every negative grep below is region-scoped to repo content files, never to
     this plan or to test/comment prose. -->

<objective>
Swap the committed MCP client example notebooks' agent-brain model from qwen3.8:latest to
qwen3.5:4b everywhere the repo faces it (owner decision 2026-10-06 15:27 CST; supersedes a
discarded sandbox-seam plan — this is a DIRECT committed-content edit), keep both docs sync
gates green, align the harness narrative, pin the swap with a RED->GREEN contract test, record
the D-11 re-decision, and push phs.

Purpose: the owner pulled qwen3.5:4b (4.2B Q4_K_M, ~3.3GB, default context length 262144)
on-box and ordered the model swap in committed content; every stale qwen3.8 reference would
otherwise mislead the next runner rebuild (wrong pull), the ops doc, and the harness rationale.

Output: the 11 files in files_modified (ci.yml: exactly one comment line), a STATE.md decision
bullet, and 261006-lhm-SUMMARY.md.

## Owner-approved evidence (executor cites, does NOT re-derive or re-run)

- Capability probe PASSED 2026-10-06 15:24 CST (run by the orchestrator, before planning):
  3-turn tool-calling via /api/chat — turn-1 34.8s cold with perfectly-formed args, warm turns
  6.1s / 5.1s, correct chained tool->result->tool->result->summary behavior.
- Model metadata: 4.2B Q4_K_M, ~3.3GB pull, default context length 262144 (256k-class default
  present but empirically not a blocker). Do NOT re-run the probe; do NOT re-pull (already on-box).
- Verified-at-planning inventory (grep, full repo): the ONLY files carrying the old model string
  are the 11 in files_modified. Occurrence map — pydantic notebook: 2 (markdown cell 1: prose +
  pull command on one line; code cell 3: model_name literal); langchain notebook: 1 (code cell 6);
  docs ipynb copies: same as their sources (byte-identical pairs); docs/example/mcp_pydantic_ai.md
  line ~45; docs/example/mcp_langchain.md line ~76; scripts/runner/README.md lines ~23/~27/~58/
  ~64/~66 (pull, tags-verify, 256k kv-cache rationale, Modelfile num_ctx status, re-probe
  command) plus a stale "17GB loaded model" size in the Why-loopback-only section; the three
  test files (comments/docstrings only — verified: NO executable model-name probe exists; the
  harness OLLAMA_URL probe is HTTP-reachability-only over /api/tags); .github/workflows/ci.yml
  line 975 (one comment).
- Notebook JSON format fact (verified): both notebooks roundtrip byte-identically with
  stdlib json.dumps(nb, indent=1, ensure_ascii=False) — a load/replace/dump rewrite is
  byte-stable apart from the edited source lines.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@scripts/check_docs_sync.py
@scripts/check_notebook_md_sync.py
@scripts/runner/README.md
@tests/test_runner_infra_contracts.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED — add TestMcpExampleModelSwap pin tests, prove them failing</name>
  <files>tests/test_runner_infra_contracts.py</files>
  <behavior>
    - Test 1 (test_both_committed_notebooks_pin_qwen35): json-load each example/mcp_example notebook; assert some code cell's joined source contains model_name='qwen3.5:4b' (pydantic) and "ollama:qwen3.5:4b" (langchain); assert the old string appears nowhere in either file's raw text.
    - Test 2 (test_docs_ipynb_mirrors_and_md_pages_carry_the_swap): docs/example/mcp_example/*.ipynb read_bytes() equal to their example/ sources; docs/example/mcp_pydantic_ai.md and docs/example/mcp_langchain.md contain qwen3.5:4b and not the old string.
    - Test 3 (test_runner_readme_pull_and_verify_name_current_model): scripts/runner/README.md contains an "ollama pull qwen3.5:4b" line and a tags-verify comment naming qwen3.5:4b, and the old string appears nowhere in the README.
  </behavior>
  <action>
    Append a new contract class TestMcpExampleModelSwap to tests/test_runner_infra_contracts.py,
    following the file's existing style (module-level REPO_ROOT Path helpers, plain asserts with
    explanatory failure messages citing the 2026-10-06 15:27 CST owner swap decision). Reuse the
    existing _load_runner_readme() helper for Test 3; add small local loaders for the notebook and
    md paths. Keep every test kernel-free, network-free, and fast (pure file reads). Do NOT touch
    the existing TestOllamaUnitPins / TestRunnerReadmeReapply / TestExampleNightlySseProbe
    assertions — the unit pins (OLLAMA_CONTEXT_LENGTH=8192, loopback host) stay exactly as they
    are (num_ctx deferral 2026-10-06 00:52 CST stands; the in-repo unit stays inert).
    Then run the new class and record the RED evidence: all three tests must FAIL at test level
    (assertion failures against the still-unswapped tree), not at collection/import level, per
    the 06-02 RED_EVIDENCE_OK convention. Commit the failing test first
    (test(quick-261006-lhm): pin mcp example notebooks model swap (RED)).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/test_runner_infra_contracts.py::TestMcpExampleModelSwap -q; echo "exit=$? (expect 3 failed / exit non-zero at RED stage)"</automated>
  </verify>
  <done>
    TestMcpExampleModelSwap exists with the three tests above; the RED run shows exactly 3
    test-level failures (no collection error); the existing 5 contract tests in the file still
    pass; committed test-first.
  </done>
</task>

<task type="auto">
  <name>Task 2: GREEN — swap the model in notebooks, all mirrors, runner README, one ci.yml comment</name>
  <files>example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb, example/mcp_example/mcp_client_ollama_langchain_agents.ipynb, docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb, docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb, docs/example/mcp_pydantic_ai.md, docs/example/mcp_langchain.md, scripts/runner/README.md, .github/workflows/ci.yml</files>
  <action>
    Notebooks (JSON-aware, structure-preserving): with a small stdlib-json script, load each
    example/mcp_example notebook, walk cells, and replace the old model token with qwen3.5:4b
    inside the matching source line strings — pydantic notebook: markdown cell 1 (two occurrences
    on one line: the availability prose and the inline pull command, keeping the sentence shape
    "Please make sure qwen3.5:4b is available. if not, please run `ollama pull qwen3.5:4b`...")
    and code cell 3 (model_name='qwen3.8:latest' -> model_name='qwen3.5:4b'); langchain
    notebook: code cell 6 ("ollama:qwen3.8:latest" -> "ollama:qwen3.5:4b", keep the trailing
    "# Local LLM model via Ollama" comment). The script must assert per-file replacement counts
    (pydantic 2, langchain 1) and change NOTHING else — outputs, execution_count, metadata, cell
    ids, and JSON layout all byte-preserved (dump with json.dumps(nb, indent=1,
    ensure_ascii=False) plus the original trailing-newline state; planning verified this is the
    file's exact format). Committed OUTPUTS stay as-executed evidence from the previous model's
    run — do NOT fabricate, refresh, or clear any output. Prove minimality: git diff on each
    notebook shows only the intended source-line changes.
    Mirrors: cp the two updated notebooks onto their docs/example/mcp_example/ counterparts
    (check_docs_sync compares byte-for-byte via filecmp). Edit the md pages' python-block lines
    to the same literals (docs/example/mcp_pydantic_ai.md ~line 45; docs/example/mcp_langchain.md
    ~line 76, keep the inline comment). Do not regenerate the md pages with
    generate_md_from_notebook.py — hand-edit the single line so the rest of the curated page is
    untouched.
    Runner README (scripts/runner/README.md): update the pull step to ollama pull qwen3.5:4b and
    its size note (17.74GB -> ~3.3GB); update the verify comment to "must list qwen3.5:4b";
    restate the Why-num_ctx-8192 narrative for the new model — 4.2B Q4_K_M, ~3.3GB, default
    context length 262144 (still 256k-class, so the ~36GB kv-cache-per-request-at-native-ctx
    rationale and the env-precedence discussion stay valid); update the Modelfile re-probe
    command to the new model name; add one sentence dating the swap (owner decision 2026-10-06
    15:27 CST, notebooks' literal reference changed; num_ctx cut remains DEFERRED per
    2026-10-06 00:52 CST, so the server default env pin stays as committed). Sweep the README's
    remaining stale size prose ("a 17GB loaded model" in Why-loopback-only -> ~3.3GB). Do NOT
    weaken or reword the D-12 loopback access-control wording or the never-0.0.0.0 notice.
    ci.yml (narrow fence relaxation, comment-only): immediately before editing, run
    git status --porcelain .github/workflows/ci.yml — if it prints ANYTHING, skip this edit and
    record the skip in the SUMMARY (a concurrent executor owns the file). If clean, change only
    line 975's comment so it reads that ipykernel children die and memory settles before
    qwen3.5:4b's ~3.3GB load — no step logic, keys, or any other line may change, and no other
    .github/ file may be touched.
    Finish by proving the sweep is closed: a scoped grep for the old model token over
    example/mcp_example, docs/example/mcp_pydantic_ai.md, docs/example/mcp_langchain.md,
    docs/example/mcp_example, and scripts/runner/README.md returns nothing. Commit
    (feat(quick-261006-lhm): swap mcp example notebooks model to qwen3.5:4b (GREEN)).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/test_runner_infra_contracts.py -q && .venv/bin/python scripts/check_notebook_md_sync.py && .venv/bin/python scripts/check_docs_sync.py && ! grep -R "qwen3.8" example/mcp_example docs/example/mcp_example docs/example/mcp_pydantic_ai.md docs/example/mcp_langchain.md scripts/runner/README.md && echo SWAP-SWEEP-CLEAN</automated>
  </verify>
  <done>
    TestMcpExampleModelSwap is GREEN alongside the 5 pre-existing contract tests (8 passed);
    check_notebook_md_sync reports 24/24 pairs in sync; check_docs_sync exits clean; the scoped
    old-token grep is empty; git diff --stat shows only the 8 files of this task (ci.yml present
    only if the porcelain precondition held, and then as a one-line comment change).
  </done>
</task>

<task type="auto">
  <name>Task 3: harness narrative alignment, D-11 record, final battery, push</name>
  <files>tests/examples/_execution.py, tests/examples/test_notebook_execution.py, tests/test_runner_infra_contracts.py, .planning/STATE.md</files>
  <action>
    Comment/docstring-only edits (no executable change — verified at planning that no
    model-name probe exists in code): in tests/examples/_execution.py's two mcp_example
    NOTEBOOK_EXEC_SPECS entries (~lines 243-258), restate the budget comments for the swap —
    model now qwen3.5:4b (owner decision 2026-10-06 15:27 CST; 4.2B Q4_K_M, ~3.3GB), probe
    evidence 15:24 CST (3-turn tool-calling, 34.8s cold / 6.1s / 5.1s warm), cell_timeout 3600
    and the 7200s outer override STAY (owner decision B), num_ctx cut still DEFERRED (00:52 CST)
    so the 256k-class default context still governs the un-cut latency tail; keep the historical
    run-37406829738 evidence attributed to the previous model. In
    tests/examples/test_notebook_execution.py update the three narrative sites: ~1030 (VRAM
    settle between stages now waits out the ~3.3GB weights load), ~1066 (retry-window rationale:
    the smaller model loads faster, but the ~60s D-13 window contract is unchanged and the probe
    remains reachability-only), ~1223 (_TIMEOUT_7200_GATED rationale: add the swap decision and
    that the new model's default context is 262144 — same 256k class, budgets unchanged,
    revisit only if the num_ctx cut is un-deferred). In tests/test_runner_infra_contracts.py's
    module docstring (~line 11) add one short note that the notebooks' model reference became
    qwen3.5:4b on 2026-10-06 (256k-class default context 262144 keeps the D-06 rationale
    intact) while the pinned unit Environment lines are unchanged and remain inert pending the
    owner re-apply. Then append the STATE.md Phase-09 decision bullet (Decisions section, after
    the 00:52 num_ctx entry): D-11 re-decision 2026-10-06 15:27 CST — same-model constraint
    lifted for the agent brain; committed mcp_example pair + all mirrors + runner README now
    reference qwen3.5:4b (4.2B Q4_K_M, ~3.3GB, default context 262144 recorded); dnallm MCP
    server side untouched; interplay: 00:52 num_ctx deferral stands (no ollama config change,
    in-repo unit pin stays inert), 3600s cell / 7200s outer timeouts stay (decision B); swap
    evidence = 15:24 CST probe PASSED (3-turn tool-calling, 34.8s cold / 6.1s / 5.1s warm);
    committed notebook outputs remain from the previous model's execution until the next full
    nightly re-execution; RED->GREEN pin = TestMcpExampleModelSwap; one ci.yml comment line
    updated under the narrow fence relaxation. Finally write
    .planning/quick/261006-lhm-direct-edit-mcp-example-notebooks-commit/261006-lhm-SUMMARY.md
    including: output provenance (outputs not fabricated; nightly re-execution refreshes them),
    RED and GREEN evidence lines, the ci.yml precondition outcome, and the push evidence. Run
    the final battery, audit the diff scope, then push: git push origin phs (retry once or twice
    on egress flake), then verify git rev-parse origin/phs matches HEAD and record both hashes.
    The diff-scope audit runs as its own step (git diff --stat against the pre-task origin
    state, eyeballed against the 11-file allowlist plus .planning records) before the push, so
    no fallible git stage is masked by a later pipeline stage.
    Do NOT dispatch or cancel any GitHub runs (the orchestrator sequences re-dispatch).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/mcp tests/test_runner_infra_contracts.py -q && .venv/bin/python scripts/check_notebook_md_sync.py && .venv/bin/python scripts/check_docs_sync.py && test "$(git rev-parse HEAD)" = "$(git rev-parse origin/phs)" && echo PUSHED-AND-GREEN</automated>
  </verify>
  <done>
    tests/mcp plus tests/test_runner_infra_contracts.py fully green (kernel-free); both sync
    gates green; the pushed diff vs origin's pre-task state touches only the 11 listed files plus
    this task's .planning records; STATE.md carries the D-11 re-decision bullet; SUMMARY exists
    with provenance + evidence; origin/phs == local HEAD; no GH run was dispatched or cancelled.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| runner box -> loopback ollama service | unauthenticated model server; access control = the D-12 loopback bind this change's README edit must not weaken |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-lhm-01 | Tampering | scripts/runner/README.md ops path | medium | mitigate | TestMcpExampleModelSwap pins the pull/verify lines to qwen3.5:4b, so an editorial revert or wrong-model pull instruction fails the fast lane in seconds; the edit explicitly preserves the never-0.0.0.0 / loopback-is-access-control wording (T-09-03 class exposure stays documented). |
| T-lhm-02 | Denial of Service | nightly stage-3 gated mcp pair | low | accept | Budgets (3600s cell / 7200s outer) are kept unchanged for the smaller, faster model; worst case is headroom, and the D-12 measured-budget record will capture the faster reality on the next nightly run. |
</threat_model>

<verification>
1. RED->GREEN chain: Task 1's recorded 3 test-level failures flip to 8 passed in tests/test_runner_infra_contracts.py after Task 2.
2. Both sync gates green: scripts/check_notebook_md_sync.py (24/24 pairs) and scripts/check_docs_sync.py exit 0.
3. Kernel-free battery green: .venv/bin/python -m pytest tests/mcp tests/test_runner_infra_contracts.py -q.
4. Scoped sweep: the old model token is absent from example/mcp_example, docs/example/mcp_example, the two docs md pages, and scripts/runner/README.md.
5. Diff-scope audit: git diff --stat against the pre-task origin state is confined to the 11 files_modified plus .planning records; ci.yml appears only as the single comment line and only if its porcelain precondition held.
6. Push: origin/phs == local HEAD, hashes recorded in the SUMMARY.
</verification>

<success_criteria>
- Both committed notebooks (and every mirror) reference qwen3.5:4b with zero stale tokens in the swept set; outputs untouched (provenance noted).
- The pin test class makes any future revert fail in seconds; sync gates and the kernel-free mcp battery are green.
- STATE.md records the D-11 re-decision with the deferral/timeout interplay and probe evidence; SUMMARY carries RED/GREEN + push evidence.
- Standing fences honored: no .github step logic (one comment line at most), no systemd/ollama config change, no cache deletion, no GH dispatch/cancel, probe not re-run.
</success_criteria>

<output>
Create .planning/quick/261006-lhm-direct-edit-mcp-example-notebooks-commit/261006-lhm-SUMMARY.md when done; push origin phs and verify origin == HEAD.
</output>
