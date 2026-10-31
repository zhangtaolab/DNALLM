"""Runner-infrastructure contract tests (D-06/D-12, 09-02; SSE probe, 09-04).

Pins the in-repo ollama systemd unit's TWO Environment lines and the
owner re-apply op documented in the runner README, so an editorial revert
of either pin fails on the fast lane in SECONDS instead of at the next
~25-minute real execution (D-07 same-change test contract). 09-04 adds
the example-nightly stage-2 SSE readiness-probe shape contract (see
:class:`TestExampleNightlySseProbe`):

* ``OLLAMA_CONTEXT_LENGTH=8192`` -- the D-06 server-default runtime cut.
  The qwen3.8:latest model (17.74GB) allocates a ~36GB kv-cache per
  request at its native 256k context, dominating mcp-pair latency and the
  nightly VRAM trough. The server default covers BOTH mcp client stacks
  (the pydantic_ai sibling talks OpenAI-compat ``/v1``, which has no
  per-request context parameter); the env is read at server start, so the
  owner must re-apply the unit + restart the service.
* ``OLLAMA_HOST=127.0.0.1:11434`` -- the D-12 loopback bind, which IS the
  access control for this unauthenticated model server. The LIVE unit on
  the runner was probed at ``OLLAMA_HOST=0.0.0.0:11434`` on 2026-10-05
  (real LAN exposure, T-09-03); the owner re-apply documented in the
  README restores the loopback pin and closes that exposure.

The unit file is the auditable source (its own header contract: "auditable
and rebuildable by diff"); these tests keep it honest between re-applies.

2026-10-06 model swap (owner decision 15:27 CST): the notebooks' model
reference became qwen3.5:4b (4.2B Q4_K_M, ~3.3GB; default context 262144
-- still 256k-class, so the D-06 rationale above stays intact), pinned by
:class:`TestMcpExampleModelSwap`; the pinned unit Environment lines are
unchanged and remain inert pending the owner re-apply.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT_FILE = REPO_ROOT / "scripts" / "runner" / "ollama.service"
RUNNER_README = REPO_ROOT / "scripts" / "runner" / "README.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CENSUS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "census_compat.yml"


def _load_unit_text() -> str:
    """Return the in-repo ollama unit file text (the auditable source)."""
    return UNIT_FILE.read_text(encoding="utf-8")


def _load_runner_readme() -> str:
    """Return the runner README text (the owner re-apply op source)."""
    return RUNNER_README.read_text(encoding="utf-8")


def _load_sse_probe_block() -> str:
    """Return the example-nightly stage-2 SSE step body from ci.yml.

    Sliced between the step's unique ``--transport sse`` server-start line
    and the following ``Stage 2.5`` step, so the assertions below read
    exactly the readiness-probe wiring (09-04, D-17 repair).
    """
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    start = text.index("--transport sse > mcp-server-sse.log")
    end = text.index("Stage 2.5", start)
    return text[start:end]


class TestOllamaUnitPins:
    """D-06/D-12 Environment-line pins over scripts/runner/ollama.service."""

    def test_unit_pins_context_length_8192(self) -> None:
        """The D-06 server-default cut is present as an Environment line."""
        assert 'Environment="OLLAMA_CONTEXT_LENGTH=8192"' in _load_unit_text(), (
            "D-06 runtime cut missing: scripts/runner/ollama.service must pin "
            'Environment="OLLAMA_CONTEXT_LENGTH=8192" (the num_ctx 8k server '
            "default covering both mcp client stacks) -- restore the line and "
            "see scripts/runner/README.md for the owner re-apply op"
        )

    def test_unit_still_pins_loopback_host(self) -> None:
        """The D-12 loopback access-control pin survives (live unit drifted to 0.0.0.0)."""
        assert 'Environment="OLLAMA_HOST=127.0.0.1:11434"' in _load_unit_text(), (
            "D-12 access control missing: scripts/runner/ollama.service must keep "
            'Environment="OLLAMA_HOST=127.0.0.1:11434" -- loopback binding is the '
            "access control for the unauthenticated model server; the LIVE runner "
            "unit was probed at 0.0.0.0:11434 on 2026-10-05 (T-09-03), never let "
            "the in-repo source drift too"
        )


class TestRunnerReadmeReapply:
    """The README owner re-apply op + num_ctx rationale (D-06)."""

    def test_readme_documents_reapply_op_and_num_ctx_rationale(self) -> None:
        """README carries the re-apply op (daemon-reload + restart) and the why-8192 section."""
        text = _load_runner_readme()
        assert "daemon-reload" in text, (
            "scripts/runner/README.md must document the owner re-apply op "
            "(systemctl daemon-reload + restart) -- the env is read at server "
            "start, so copying the unit alone changes nothing"
        )
        assert "restart ollama" in text, (
            "scripts/runner/README.md must name the service restart "
            "(sudo systemctl restart ollama) in the re-apply op"
        )
        assert "num_ctx 8192" in text or ("num_ctx" in text and "8192" in text), (
            "scripts/runner/README.md must carry the Why num_ctx 8192 (D-06) "
            "rationale section beside Why loopback-only"
        )


class TestExampleNightlySseProbe:
    """The example-nightly stage-2 SSE readiness probe shape (09-04, D-17).

    An SSE endpoint answers 200 + ``text/event-stream`` and then holds the
    body open forever, so a curl EXIT-CODE success probe (`curl -m 2 ... &&
    break`) always fails at the max-time cap even against a healthy server:
    the first live exercise of the stage (run 37345067326) polled a fully
    serving server for 17 minutes and failed the stage on probe shape alone.
    Readiness must be asserted on the RECEIVED HTTP status (``%{http_code}``
    prints 200 once headers arrive, 000 when nothing answered).
    """

    def test_sse_probe_asserts_received_http_status(self) -> None:
        """The /sse readiness check compares %{http_code} to 200, not curl's exit code."""
        block = _load_sse_probe_block()
        assert "%{http_code}" in block, (
            "the stage-2 SSE readiness probe must read curl's %{http_code} "
            "output -- an SSE body never completes under -m, so curl's exit "
            "code can never signal readiness (run 37345067326: healthy server "
            "polled for 17 minutes, stage recorded failed)"
        )
        assert '[ "$CODE" = "200" ]' in block, (
            "the stage-2 SSE readiness probe must gate on a received 200 "
            "status; 000 (nothing answered) is the only not-ready signal"
        )

    def test_sse_probe_never_requires_curl_exit_success(self) -> None:
        """The pre-repair exit-code probe shapes are absent from the SSE step."""
        block = _load_sse_probe_block()
        old_shapes = (
            "curl -s -o /dev/null -m 2 http://localhost:8000/sse && break",
            "if curl -s -o /dev/null -m 2 http://localhost:8000/sse; then",
        )
        for shape in old_shapes:
            assert shape not in block, (
                f"stage-2 SSE probe regressed to the always-fail shape ({shape!r}): "
                "SSE responses never complete under -m, so curl exits 28 at the "
                "cap even when the server is serving -- assert %{http_code} instead"
            )


# --- Committed-content model-swap pins (261006-lhm, D-11 re-decision) ---

OLD_MODEL_NAME = "qwen3.8"
NEW_MODEL = "qwen3.5:4b"
PYDANTIC_NB = "mcp_client_ollama_pydantic_ai.ipynb"
LANGCHAIN_NB = "mcp_client_ollama_langchain_agents.ipynb"
EXAMPLE_MCP_DIR = REPO_ROOT / "example" / "mcp_example"
DOCS_MCP_MIRROR_DIR = REPO_ROOT / "docs" / "example" / "mcp_example"
DOCS_PYDANTIC_MD = REPO_ROOT / "docs" / "example" / "mcp_pydantic_ai.md"
DOCS_LANGCHAIN_MD = REPO_ROOT / "docs" / "example" / "mcp_langchain.md"
SWAP_DECISION = "owner model-swap decision 2026-10-06 15:27 CST"


def _load_example_notebook_raw(name: str) -> str:
    """Return the raw committed text of an example/mcp_example notebook."""
    return (EXAMPLE_MCP_DIR / name).read_text(encoding="utf-8")


def _load_example_notebook_code_sources(name: str) -> list[str]:
    """Return each code cell's source as one joined string (JSON-aware read)."""
    notebook = json.loads(_load_example_notebook_raw(name))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


class TestMcpExampleModelSwap:
    """Committed-content pins for the qwen3.5:4b agent-brain swap (261006-lhm).

    The owner swapped the MCP client example notebooks' agent-brain model
    from qwen3.8:latest to qwen3.5:4b (2026-10-06 15:27 CST) everywhere the
    repo faces it: both committed notebooks, their docs mirrors, the docs
    md pages, and the runner README ops path. These pins make an editorial
    revert -- or a wrong-model pull instruction -- fail the fast lane in
    seconds instead of misleading the next runner rebuild (D-07
    same-change test contract).
    """

    def test_both_committed_notebooks_pin_qwen35(self) -> None:
        """Each committed notebook's model literal is qwen3.5:4b, old token absent."""
        pydantic_sources = _load_example_notebook_code_sources(PYDANTIC_NB)
        assert any("model_name='qwen3.5:4b'" in src for src in pydantic_sources), (
            f"{PYDANTIC_NB} must carry model_name='{NEW_MODEL}' in a code cell "
            f"({SWAP_DECISION}) -- the committed literal is what the nightly "
            "gated mcp pair executes against"
        )
        langchain_sources = _load_example_notebook_code_sources(LANGCHAIN_NB)
        assert any('"ollama:qwen3.5:4b"' in src for src in langchain_sources), (
            f'{LANGCHAIN_NB} must carry "ollama:{NEW_MODEL}" in a code cell '
            f"({SWAP_DECISION}) -- the committed literal is what the nightly "
            "gated mcp pair executes against"
        )
        for name in (PYDANTIC_NB, LANGCHAIN_NB):
            raw = _load_example_notebook_raw(name)
            assert OLD_MODEL_NAME not in raw, (
                f"{name} still references the retired {OLD_MODEL_NAME} model "
                f"({SWAP_DECISION}) -- sweep every occurrence to {NEW_MODEL}"
            )

    def test_docs_ipynb_mirrors_and_md_pages_carry_the_swap(self) -> None:
        """docs ipynb mirrors stay byte-identical and both md pages carry the swap."""
        for name in (PYDANTIC_NB, LANGCHAIN_NB):
            source_bytes = (EXAMPLE_MCP_DIR / name).read_bytes()
            mirror_bytes = (DOCS_MCP_MIRROR_DIR / name).read_bytes()
            assert mirror_bytes == source_bytes, (
                f"docs/example/mcp_example/{name} drifted from its example/ "
                "source (check_docs_sync compares byte-for-byte) -- re-copy "
                "the swapped notebook onto its docs mirror"
            )
        for md_path, literal in (
            (DOCS_PYDANTIC_MD, "model_name='qwen3.5:4b'"),
            (DOCS_LANGCHAIN_MD, '"ollama:qwen3.5:4b"'),
        ):
            text = md_path.read_text(encoding="utf-8")
            assert literal in text, (
                f"{md_path.name} must carry {literal} in its python block "
                f"({SWAP_DECISION}) -- check_notebook_md_sync asserts the md "
                "block matches the notebook source"
            )
            assert OLD_MODEL_NAME not in text, (
                f"{md_path.name} still references the retired {OLD_MODEL_NAME} "
                f"model ({SWAP_DECISION}) -- sweep the python-block line"
            )

    def test_runner_readme_pull_and_verify_name_current_model(self) -> None:
        """Runner README ops path pulls and verifies qwen3.5:4b, old token absent."""
        text = _load_runner_readme()
        assert "ollama pull qwen3.5:4b" in text, (
            "scripts/runner/README.md pull step must read "
            f"`ollama pull {NEW_MODEL}` ({SWAP_DECISION}) -- a stale pull "
            "instruction misleads the next runner rebuild into pulling the "
            "retired 17.74GB model"
        )
        assert "must list qwen3.5:4b" in text, (
            "scripts/runner/README.md /api/tags verify comment must name "
            f"{NEW_MODEL} ({SWAP_DECISION}) -- the verify step is what proves "
            "the pull landed on the runner box"
        )
        assert OLD_MODEL_NAME not in text, (
            "scripts/runner/README.md still references the retired "
            f"{OLD_MODEL_NAME} model ({SWAP_DECISION}) -- sweep the pull, "
            "verify, num_ctx-narrative, and re-probe lines"
        )


# --- Census-lane install-sequence pins (Phase 27, UAT G-27-1) ---

# The exact lines the lane's install step must carry, in order.
CENSUS_FULL_INSTALL = 'uv pip install -e ".[full]"'
CENSUS_PIP_BOOTSTRAP = "uv pip install --python .venv/bin/python pip"
CENSUS_MAMBA_INSTALL = (
    "bash scripts/install_mamba.sh "
    "--python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple"
)
CENSUS_RUN_LINE = "python scripts/census_compat.py --run --with-loads --verify"

# The unpinned installer pipeline, kept ONLY as the wiped-box fallback (IN-05).
CENSUS_UV_INSTALLER = "curl -LsSf https://astral.sh/uv/install.sh | sh"

# CR-01 (27.1 review): the skip branch must persist the provisioned uv's
# directory to $GITHUB_PATH, exactly as the curl installer it replaces did
# (the served install.sh appends its install dir when GITHUB_PATH is set) --
# a run step's `export PATH` dies with its own shell, so without this write
# the next step's bare `uv venv` cannot resolve uv on the provisioned box.
CENSUS_UV_GITHUB_PATH_WRITE = 'echo "$HOME/.local/bin" >> "$GITHUB_PATH"'


def _load_census_workflow_text() -> str:
    """Return the committed census lane workflow text (the auditable source)."""
    return CENSUS_WORKFLOW.read_text(encoding="utf-8")


class TestCensusLaneInstallSequence:
    """Census-lane mamba-kernel install pins (Phase 27, UAT G-27-1).

    The lane's ``.[full]`` install carries no mamba-ssm, but the
    mamba-family remote code imports ``mamba_ssm`` at import-probe time, so
    as originally committed the fresh census reported those entries as
    unmappable and the lane exited 1 deterministically (red-run replica
    evidence on the runner box: 45 matched / 4 disagreeing entries, exit 1).
    These pins hold the environmental fix: an editorial revert -- or masking
    the kernel build -- re-creates the permanently-red lane from UAT G-27-1
    and fails this fast lane in seconds instead of at the next 4-hour
    dispatch.
    """

    def test_install_sequence_full_then_bootstrap_then_mamba_then_census(self) -> None:
        """pip bootstrap + install_mamba.sh sit between the .[full] install and the run."""
        text = _load_census_workflow_text()
        full = text.index(CENSUS_FULL_INSTALL)
        bootstrap = text.index(CENSUS_PIP_BOOTSTRAP)
        mamba = text.index(CENSUS_MAMBA_INSTALL)
        census = text.index(CENSUS_RUN_LINE)
        assert full < bootstrap < mamba < census, (
            "census_compat.yml must install the mamba kernels BETWEEN the "
            f".[full] install and the census run: {CENSUS_FULL_INSTALL!r} < "
            f"{CENSUS_PIP_BOOTSTRAP!r} < install_mamba.sh < {CENSUS_RUN_LINE!r} "
            "(UAT G-27-1: without the kernels the mamba-family remote code "
            "fails its import probe, the census reports those entries as "
            "unmappable, and the lane is permanently red -- an editorial "
            "revert or reordering re-creates exactly that)"
        )

    def test_mamba_installer_invocation_is_the_sanctioned_shape(self) -> None:
        """The installer is called exactly as the proven ci.yml mamba lane calls it."""
        text = _load_census_workflow_text()
        assert CENSUS_MAMBA_INSTALL in text, (
            "census_compat.yml must invoke the repo's sanctioned pinned "
            f"installer as `{CENSUS_MAMBA_INSTALL}` -- the plain .[mamba] "
            "extra cannot build on this box (uv source build fails, "
            "verified; the release wheel matrix stops at torch 2.10), so "
            "any other invocation shape re-creates the permanently-red lane "
            "from UAT G-27-1"
        )

    def test_no_masked_failure_in_the_lane(self) -> None:
        """No ``|| true`` masking in any executed line: a failed build must redden the lane."""
        text = _load_census_workflow_text()
        # Only executed lines count -- the lane's own comments name the
        # forbidden shape as prose ("NO `|| true` masking"), which is fine.
        executed = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert "|| true" not in executed, (
            "census_compat.yml must never mask a failed step with '|| true' "
            "(UAT G-27-1 posture): a masked kernel-build failure would green "
            "the lane while the census still reports the mamba family as "
            "unmappable -- a silent red-to-green lie"
        )

    def test_no_continue_on_error_anywhere(self) -> None:
        """No job- or step-level continue-on-error: a failed build must redden the lane."""
        doc = yaml.safe_load(_load_census_workflow_text())
        job = doc["jobs"]["census-compat"]
        steps = job.get("steps", [])
        assert "continue-on-error" not in job, (
            "job-level continue-on-error would green a failing lane"
        )
        flagged = [s.get("name", s.get("uses")) for s in steps if "continue-on-error" in s]
        assert not flagged, f"steps carrying continue-on-error: {flagged}"

    def test_uv_install_skips_provisioned_binary(self) -> None:
        """Install uv keeps the box's provisioned uv and only installs on a wiped box.

        The persistent runner carries a mirror-configured uv under
        ``$HOME/.local/bin`` (runner ops, owner directive); an unconditional
        ``curl | sh`` both fetches unpinned code on every dispatch and
        silently replaces that binary (IN-05). The guard must check
        ``command -v uv`` BEFORE any installer fetch, print the provisioned
        version, keep the installer pipeline only in the else branch, and
        persist the uv directory to ``$GITHUB_PATH`` so later steps' bare
        ``uv`` resolves (CR-01, 27.1 review).
        """
        doc = yaml.safe_load(_load_census_workflow_text())
        steps = doc["jobs"]["census-compat"]["steps"]
        uv_steps = [s for s in steps if s.get("name") == "Install uv"]
        assert len(uv_steps) == 1, (
            f"census_compat.yml must carry exactly one Install uv step (found {len(uv_steps)})"
        )
        step = uv_steps[0]
        assert step.get("if") == "steps.gpu-check.outputs.has_gpu == 'true'", (
            "the Install uv step must keep its GPU guard -- IN-05 changes the run body only"
        )
        run = step.get("run", "")
        assert 'export PATH="$HOME/.local/bin:$PATH"' in run, (
            "the Install uv guard must export $HOME/.local/bin first (runner-"
            "ops uv lives there; A1 in 27.1-RESEARCH.md)"
        )
        assert "command -v uv" in run, (
            "the Install uv step must probe `command -v uv` and skip the "
            "installer when the box already carries uv (IN-05)"
        )
        assert "$(uv --version)" in run, (
            "the skip branch must echo the provisioned uv version so the "
            "dispatch log names the toolchain actually used"
        )
        assert CENSUS_UV_INSTALLER in run, (
            "the wiped-box fallback installer pipeline must stay in the else "
            f"branch: {CENSUS_UV_INSTALLER!r}"
        )
        assert run.index("command -v uv") < run.index(CENSUS_UV_INSTALLER), (
            "the `command -v uv` guard must precede the installer fetch -- an "
            "unconditional install replaces the box's mirror-configured uv "
            "on every census dispatch (IN-05)"
        )
        assert CENSUS_UV_GITHUB_PATH_WRITE in run, (
            "the skip branch must append the provisioned uv's directory to "
            f"$GITHUB_PATH as `{CENSUS_UV_GITHUB_PATH_WRITE}` -- a run step's "
            "export PATH dies with its own shell, and the curl installer this "
            "guard replaces performed exactly this write (verified in the "
            "served install.sh), so without it the next step's bare `uv venv` "
            "fails with 'uv: command not found' on the provisioned box and "
            "reddens the single allowed DISP-01 dispatch for an environment "
            "reason (CR-01, 27.1 review)"
        )
        assert (
            run.index("command -v uv")
            < run.index(CENSUS_UV_GITHUB_PATH_WRITE)
            < run.index(CENSUS_UV_INSTALLER)
        ), (
            "the $GITHUB_PATH write belongs INSIDE the skip branch (between "
            "the `command -v uv` guard and the else-branch installer): the "
            "wiped-box fallback gets its GITHUB_PATH write from the installer "
            "itself, while the provisioned-box path this guard adds has no "
            "other writer (CR-01, 27.1 review)"
        )
        assert "UV_VERSION" not in run, (
            "UV_VERSION does not exist in the served astral installer "
            "(refuted in 27.1-RESEARCH.md); do not plan around it"
        )

    def test_dispatch_only_posture_unchanged(self) -> None:
        """Triggers, permissions, runner, guard, and timeout are exactly the D-08 posture."""
        doc = yaml.safe_load(_load_census_workflow_text())
        # pyyaml gotcha: a bare `on:` key parses as boolean True, and its
        # bare-scalar value parses as a plain string (not a dict/list).
        triggers = doc.get(True) or doc.get("on")
        assert triggers is not None, "census_compat.yml carries no trigger block"
        if isinstance(triggers, str):
            names = {triggers}
        else:
            names = set(triggers)
        assert names == {"workflow_dispatch"}, (
            "census_compat.yml must stay dispatch-ONLY (D-08): triggers are "
            f"{sorted(names)} -- adding push/pull_request/schedule would let "
            "PR-authored code run on the self-hosted GPU box"
        )
        assert doc.get("permissions") == {"contents": "read"}, (
            "census_compat.yml must keep permissions at contents: read "
            "(least privilege on the self-hosted runner)"
        )
        job = doc["jobs"]["census-compat"]
        assert job.get("if") == "github.event_name == 'workflow_dispatch'", (
            "the census job must keep its event-name guard (D-08 security "
            "boundary on the self-hosted runner)"
        )
        assert job.get("runs-on") == ["self-hosted", "dnallm-nightly"], (
            "the census job must stay on [self-hosted, dnallm-nightly] -- "
            "the box the committed census evidence was produced on"
        )
        assert job.get("timeout-minutes") == 240, (
            "the census job must keep its 240-minute timeout (49 fetches "
            "plus family-representative loads; UAT G-27-1's kernel install "
            "adds build time but stays inside this budget)"
        )


# --- Extras-lane contracts (Phase 28, D-05..D-08/D-12; owner ruling A) ---

EXTRAS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "extras_compat.yml"
EXTRAS_JOB_NAME = "extras-compat"
EXTRAS_NIGHTLY_CRON = "0 4 * * *"

EXTRAS_BIGBIRD_REPO = "AIRI-Institute/gena-lm-bigbird-base-sparse"
EXTRAS_CROSSDNA_REPO = "chengCCC/CrossDNA_pretrain"
EXTRAS_ENTRIES_VALUE = f"{EXTRAS_BIGBIRD_REPO},{EXTRAS_CROSSDNA_REPO}"

# The exact executed lines the extras lane must carry (install order is
# load-bearing: the pinned installer needs pip bootstrapped AND torch from
# the .[full] install ahead of it).
EXTRAS_FULL_INSTALL = 'uv pip install -e ".[full]"'
EXTRAS_PIP_BOOTSTRAP = "uv pip install --python .venv/bin/python pip"
EXTRAS_DEEPSPEED_INSTALL = 'uv pip install -e ".[deepspeed]"'
EXTRAS_FLASH_ATTN_INSTALLER = (
    "bash scripts/install_flash_attn.sh "
    "--python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple"
)

# D-07 length fact (28-01 measured): sparse BigBird requires
# seq_len % block_size(64) == 0, so the committed tool's
# LOAD_SEQUENCE_LENGTH=32 can only ever forward-red on divisibility and mask
# the real kernel verdict. The lane probes through a one-line copy with the
# constant at 64; the committed tool itself is never invoked with --run here.
EXTRAS_LEN64_SED = "s/^LOAD_SEQUENCE_LENGTH = 32$/LOAD_SEQUENCE_LENGTH = 64/"
EXTRAS_LEN64_PROBE = ".lane-tools/census_compat_len64.py --run --with-loads"
EXTRAS_SEED_COPY = "cp tests/models/compat_census.json census-artifacts/extras-fresh.json"
EXTRAS_VERIFY_LINE = (
    "python scripts/census_compat.py --verify --artifact census-artifacts/extras-fresh.json"
)

# Bare package-index flash-attn installs: forbidden shapes (D-12 -- the
# sha256-pinned installer script is the ONLY CI flash-attn route; the
# pyproject extra stays the end-user route).
EXTRAS_FORBIDDEN_FLASH_ATTN_INSTALLS = (
    "pip install flash-attn",
    "pip install flash_attn",
    "uv pip install flash-attn",
    "uv pip install flash_attn",
    'uv pip install -e ".[flash-attn]"',
)

EXTRAS_PROBE_STEP_NAME = "Seed and probe the two extras entries"
EXTRAS_GATE1_STEP_NAME = "Gate 1: fresh probe evidence matches the ruling-A expected shape"
EXTRAS_GATE2_STEP_NAME = "Gate 2: fresh verdicts match the committed registry"


def _load_extras_workflow_text() -> str:
    """Return the committed extras lane workflow text (the auditable source)."""
    if not EXTRAS_WORKFLOW.exists():
        pytest.fail(
            f"{EXTRAS_WORKFLOW.relative_to(REPO_ROOT)} is missing: the Phase 28 "
            "extras lane (D-05) must exist as a dedicated workflow file"
        )
    return EXTRAS_WORKFLOW.read_text(encoding="utf-8")


def _load_extras_doc() -> dict:
    """Return the parsed extras lane workflow document."""
    return yaml.safe_load(_load_extras_workflow_text())


def _extras_step_run(step_name_prefix: str) -> str:
    """Return the concatenated run body of the first step named with the prefix."""
    steps = _load_extras_doc()["jobs"][EXTRAS_JOB_NAME]["steps"]
    for step in steps:
        if str(step.get("name", "")).startswith(step_name_prefix):
            return str(step.get("run", ""))
    pytest.fail(f"extras_compat.yml carries no step named {step_name_prefix!r}*")


class TestExtrasLaneContract:
    """Extras-lane structural pins (Phase 28, D-05..D-08/D-12; ruling A).

    The dedicated extras lane answers a DIFFERENT question than the default
    census (D-02): census_compat.yml keeps measuring the default install and
    its missing-extra failures, while extras_compat.yml proves the same two
    models still load and forward when their curated packages are installed.
    These pins hold that split: the lane's secure self-hosted posture, the
    D-12 install order (pinned flash-attn installer only), the D-07 length-64
    probe route, and the TWO independent pass gates -- an editorial revert of
    any of them re-creates either a PR-reachable GPU lane or a lane whose
    green could ride a stale seeded verdict instead of tonight's probe.
    """

    def test_triggers_are_schedule_dispatch_and_call_only(self) -> None:
        """schedule/workflow_dispatch/workflow_call only -- never push or pull_request."""
        doc = _load_extras_doc()
        # pyyaml gotcha: a bare `on:` key parses as boolean True (the same
        # normalization TestCensusLaneInstallSequence applies).
        triggers = doc.get(True) or doc.get("on")
        assert triggers is not None, "extras_compat.yml carries no trigger block"
        names = {triggers} if isinstance(triggers, str) else set(triggers)
        assert names == {"schedule", "workflow_dispatch", "workflow_call"}, (
            "extras_compat.yml triggers must be exactly schedule/workflow_dispatch/"
            f"workflow_call (found {sorted(names)}): push/pull_request would let "
            "PR-authored code run on the self-hosted GPU box, and workflow_call "
            "is not an independent public trigger (T-28-09)"
        )
        assert triggers["schedule"] == [{"cron": EXTRAS_NIGHTLY_CRON}], (
            "the extras lane must own exactly one nightly cron, 0 4 * * * "
            "(D-06: 04:00 UTC, queue-serialized after the 03:00/05:30 ci.yml "
            "lanes on the single dnallm-nightly runner)"
        )
        assert "push" not in names
        assert "pull_request" not in names

    def test_permissions_runner_and_timeout(self) -> None:
        """contents:read, the self-hosted nightly runner, and a 360-minute bound."""
        doc = _load_extras_doc()
        assert doc.get("permissions") == {"contents": "read"}, (
            "extras_compat.yml must stay at contents: read (least privilege on "
            "the self-hosted runner; no secrets are ever needed)"
        )
        job = doc["jobs"][EXTRAS_JOB_NAME]
        assert job.get("runs-on") == ["self-hosted", "dnallm-nightly"], (
            "the extras job must run on [self-hosted, dnallm-nightly] -- the box "
            "the measured extras evidence was produced on"
        )
        assert job.get("timeout-minutes") == 360, (
            "the extras job must carry a 360-minute timeout: the uncached "
            "flash-attn source build measured ~64 min at MAX_JOBS=8 (28-01), "
            "and self-hosted jobs have no platform cap, so this is the only "
            "whole-job bound against a hung build monopolizing the single runner"
        )

    def test_event_gate_admits_only_sanctioned_shapes(self) -> None:
        """The job gate ANDs schedule with its exact cron string; dispatch and call pass."""
        gate = _load_extras_doc()["jobs"][EXTRAS_JOB_NAME].get("if", "")
        assert (
            f"github.event_name == 'schedule' && github.event.schedule == '{EXTRAS_NIGHTLY_CRON}'"
            in gate
        ), (
            "the schedule arm must be ANDed with its exact cron string (D-06 "
            "cron-string gate discipline): a second cron entry added later must "
            "fail closed instead of silently reaching the runner"
        )
        assert "github.event_name == 'workflow_dispatch'" in gate, (
            "standalone manual dispatch must stay admitted (reproduction + the "
            "post-merge first live run)"
        )
        assert "github.event_name == 'workflow_call'" in gate, (
            "the internal reusable call from ci.yml's dispatch-gated job must "
            "stay admitted (D-06 fourth selector wiring)"
        )

    def test_uv_step_fails_closed_without_installer_fallback(self) -> None:
        """Install uv requires the runner-provisioned binary and never curls.

        Unlike the census lane (whose curl installer stays as the wiped-box
        fallback, T-27.1-06), the extras lane fails closed when uv is absent:
        its long flash-attn build must never start from an unpinned fetched
        installer, and the box this lane is pinned to carries a runner-ops
        mirror-configured uv under $HOME/.local/bin.
        """
        doc = _load_extras_doc()
        steps = doc["jobs"][EXTRAS_JOB_NAME]["steps"]
        uv_steps = [s for s in steps if s.get("name") == "Install uv"]
        assert len(uv_steps) == 1, (
            f"extras_compat.yml must carry exactly one Install uv step (found {len(uv_steps)})"
        )
        run = str(uv_steps[0].get("run", ""))
        assert 'export PATH="$HOME/.local/bin:$PATH"' in run, (
            "the Install uv step must export $HOME/.local/bin first (the runner-ops uv lives there)"
        )
        assert run.index("command -v uv") < run.index("$GITHUB_PATH"), (
            "the `command -v uv` probe must precede the $GITHUB_PATH write"
        )
        assert 'echo "$HOME/.local/bin" >> "$GITHUB_PATH"' in run, (
            "the provisioned-uv branch must persist $HOME/.local/bin to "
            "$GITHUB_PATH so later steps' bare `uv` resolves (CR-01 precedent)"
        )
        assert "curl" not in run, (
            "the extras lane must carry NO curl installer fallback: a wiped "
            "box fails closed with a clear message instead of fetching an "
            "unpinned installer ahead of a 64-minute build"
        )
        assert "exit 1" in run, "uv absence must fail the step (fail-closed posture)"

    def test_install_order_full_pip_deepspeed_installer_probe(self) -> None:
        """.[full] -> pip bootstrap -> .[deepspeed] -> pinned installer -> probe."""
        text = _load_extras_workflow_text()
        full = text.index(EXTRAS_FULL_INSTALL)
        bootstrap = text.index(EXTRAS_PIP_BOOTSTRAP)
        deepspeed = text.index(EXTRAS_DEEPSPEED_INSTALL)
        installer = text.index(EXTRAS_FLASH_ATTN_INSTALLER)
        probe = text.index(EXTRAS_LEN64_PROBE)
        assert full < bootstrap < deepspeed < installer < probe, (
            "extras_compat.yml install order is load-bearing (D-12): the "
            "installer script needs pip bootstrapped and torch resolved ahead "
            "of it (the sdist's setup.py imports torch with --no-build-"
            "isolation), and the deepspeed extra must be installed before the "
            "probes run -- any reordering redds the lane deterministically"
        )

    def test_flash_attn_installs_only_through_the_pinned_script(self) -> None:
        """The exact pinned installer invocation; no bare package-index flash-attn install."""
        text = _load_extras_workflow_text()
        assert EXTRAS_FLASH_ATTN_INSTALLER in text, (
            "extras_compat.yml must invoke the sha256-pinned installer exactly "
            f"as `{EXTRAS_FLASH_ATTN_INSTALLER}` (D-12: the only CI flash-attn "
            "route -- no aarch64/cu130/torch-2.14 wheel exists, and the sdist "
            "needs the sm_121 gencode + c++20 patches the script applies)"
        )
        for shape in EXTRAS_FORBIDDEN_FLASH_ATTN_INSTALLS:
            assert shape not in text, (
                f"forbidden flash-attn install shape {shape!r}: D-12 forbids "
                "every bare package-index route in CI (the plain extra install "
                "cannot patch setup.py, so it either fails or builds wrong "
                "architectures on this box)"
            )

    def test_probe_selects_both_repos_in_single_entry_mode(self) -> None:
        """The --entries value names exactly BigBird and CrossDNA, comma-joined."""
        probe = _extras_step_run(EXTRAS_PROBE_STEP_NAME)
        assert f"--entries {EXTRAS_ENTRIES_VALUE}" in probe, (
            "the probe step must run census single-entry mode over exactly "
            f"{EXTRAS_ENTRIES_VALUE} (D-07): the two external-package models "
            "the curated extras exist for"
        )

    def test_probe_runs_bigbird_at_a_64_multiple(self) -> None:
        """The probe goes through the diff-proven length-64 tool copy, never the committed --run."""
        text = _load_extras_workflow_text()
        probe = _extras_step_run(EXTRAS_PROBE_STEP_NAME)
        assert "grep -c '^LOAD_SEQUENCE_LENGTH = 32$' scripts/census_compat.py" in probe, (
            "the length-64 copy must be occurrence-asserted before the sed "
            "swap: a renamed constant must fail loudly instead of silently "
            "probing at 32 (the divisibility red that masks the kernel verdict)"
        )
        assert EXTRAS_LEN64_SED in probe, (
            "the copy must be produced by the exact one-line sed swap "
            f"`{EXTRAS_LEN64_SED}` -- BigBird requires seq_len % 64 == 0 "
            "(28-01 measured; D-07)"
        )
        assert "diff scripts/census_compat.py .lane-tools/census_compat_len64.py" in probe, (
            "the copy must be diff-proven to differ from the committed tool "
            "(the 28-01 scratch-copy discipline, now lane-resident)"
        )
        assert EXTRAS_LEN64_PROBE in probe, (
            "the two-entry probe must execute through the length-64 copy"
        )
        assert "scripts/census_compat.py --run" not in text, (
            "the committed census tool must never be invoked with --run by the "
            "extras lane: its LOAD_SEQUENCE_LENGTH=32 is the default lane's "
            "contract and structurally cannot forward sparse BigBird"
        )

    def test_fresh_probes_are_forced_not_carried(self) -> None:
        """The seeded load evidence is stripped so both entries re-measure tonight.

        apply_loads keeps a prior load that already passed (it carries the old
        evidence instead of re-fetching), and the committed seed carries
        CrossDNA's Plan-01 measured green load. Without this strip the lane
        would assert LAST cycle's (or the committed artifact's) green instead
        of tonight's -- a green that cannot catch a broken extras env.
        """
        probe = _extras_step_run(EXTRAS_PROBE_STEP_NAME)
        assert EXTRAS_SEED_COPY in probe, (
            "the fresh artifact must be seeded from the committed census "
            "(D-02/D-08: the 49-record roster and the hand-authored "
            "dispositions live there; gate 2 diffs against all of them)"
        )
        assert 'record["load"] = None' in probe, (
            "the seed-prep must null the two entries' load evidence so "
            "apply_loads re-probes both models fresh every run"
        )
        assert EXTRAS_BIGBIRD_REPO in probe
        assert EXTRAS_CROSSDNA_REPO in probe

    def test_two_distinct_gates_green_then_verify(self) -> None:
        """Gate 1 (fresh evidence) and gate 2 (registry agreement) are separate steps, in order."""
        doc = _load_extras_doc()
        steps = doc["jobs"][EXTRAS_JOB_NAME]["steps"]
        names = [str(s.get("name", "")) for s in steps]
        gate1 = next(i for i, n in enumerate(names) if n.startswith("Gate 1"))
        gate2 = next(i for i, n in enumerate(names) if n.startswith("Gate 2"))
        assert gate1 < gate2, (
            "gate 1 (independent fresh-evidence assertion) must precede gate 2 "
            "(the --verify registry diff): a --verify-only lane could green on "
            "a carried seeded verdict (D-08)"
        )
        gate1_run = str(steps[gate1].get("run", ""))
        gate2_run = str(steps[gate2].get("run", ""))
        assert "extras-fresh.json" in gate1_run, "gate 1 must parse the fresh artifact itself"
        assert "forward_ok" in gate1_run, (
            "gate 1 must read the raw load fields, not derived verdicts"
        )
        assert EXTRAS_VERIFY_LINE in gate2_run, (
            "gate 2 must run the committed census tool's fail-closed --verify "
            f"drift gate as `{EXTRAS_VERIFY_LINE}`"
        )

    def test_gate1_reads_raw_load_evidence_not_dispositions(self) -> None:
        """Gate 1 asserts load.ok/forward_ok booleans; the extra disposition can never satisfy it."""
        gate1_run = _extras_step_run(EXTRAS_GATE1_STEP_NAME)
        for banned in ("compat_verdict", "extra-required-tf5", "disposition"):
            assert banned not in gate1_run, (
                f"gate 1 must not consult {banned!r}: the extra disposition "
                "alone can never satisfy gate 1 on a failed forward -- the "
                "gate exists to require tonight's real load+forward outcome"
            )

    def test_gate1_asserts_the_ruling_a_expected_shape(self) -> None:
        """CrossDNA fully green; BigBird load-green with the structural forward-red named."""
        gate1_run = _extras_step_run(EXTRAS_GATE1_STEP_NAME)
        assert EXTRAS_CROSSDNA_REPO in gate1_run, (
            "gate 1 must address the CrossDNA probe record individually"
        )
        assert EXTRAS_BIGBIRD_REPO in gate1_run, (
            "gate 1 must address the BigBird probe record individually"
        )
        assert "superblock" in gate1_run, (
            "gate 1 must pin BigBird's forward failure to the known structural "
            "cause (triton libtriton.superblock absence, owner ruling A): a "
            "failure at any OTHER cause -- including a length-32 divisibility "
            "error -- means the probe did not reach the real kernel verdict"
        )
        assert 'forward_ok"] is True' in gate1_run, (
            "gate 1 must assert CrossDNA forward_ok=True (the flash-attn "
            "extra delivers load+forward)"
        )
        assert 'forward_ok"] is False' in gate1_run, (
            "gate 1 must assert BigBird forward_ok=False (the ruling-A split: "
            "load works with the deepspeed extra, forward is structurally "
            "red) -- never one blanket assertion over both"
        )

    def test_no_masked_failure_or_continue_on_error(self) -> None:
        """No ``|| true`` on executed lines; no continue-on-error at any level."""
        text = _load_extras_workflow_text()
        executed = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert "|| true" not in executed, (
            "extras_compat.yml must never mask a failed step with '|| true' "
            "(T-28-10): a masked install or probe failure would green the lane "
            "over missing evidence"
        )
        doc = _load_extras_doc()
        job = doc["jobs"][EXTRAS_JOB_NAME]
        assert "continue-on-error" not in job, (
            "job-level continue-on-error would green a failing lane"
        )
        flagged = [
            s.get("name", s.get("uses")) for s in job.get("steps", []) if "continue-on-error" in s
        ]
        assert not flagged, f"steps carrying continue-on-error: {flagged}"

    def test_gpu_absence_fails_closed_with_marker(self) -> None:
        """GPU absence writes the marker artifact and exits nonzero (T-28-10 fail-closed)."""
        doc = _load_extras_doc()
        steps = doc["jobs"][EXTRAS_JOB_NAME]["steps"]
        gpu_steps = [s for s in steps if s.get("name") == "Check for GPU"]
        assert len(gpu_steps) == 1, "exactly one Check for GPU step"
        run = str(gpu_steps[0].get("run", ""))
        assert "census-artifacts/gpu_absent.marker" in run, (
            "GPU absence must write census-artifacts/gpu_absent.marker so the "
            "always-upload still delivers the diagnosis"
        )
        assert "exit 1" in run, "GPU absence must fail the lane (zero-evidence green forbidden)"

    def test_evidence_upload_always_runs(self) -> None:
        """The extras-fresh-evidence upload is unconditional and carries both files."""
        doc = _load_extras_doc()
        steps = doc["jobs"][EXTRAS_JOB_NAME]["steps"]
        uploads = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
        assert len(uploads) == 1, (
            f"expected exactly one artifact upload step (found {len(uploads)})"
        )
        upload = uploads[0]
        assert upload.get("if") == "always()", (
            "the evidence upload must run on red and green alike -- a red run "
            "with evidence beats a green run without it"
        )
        with_block = upload.get("with", {})
        assert with_block.get("name") == "extras-fresh-evidence", (
            "the artifact name must be extras-fresh-evidence (the documented operator surface)"
        )
        path = str(with_block.get("path", ""))
        assert "census-artifacts/extras-fresh.json" in path, "the fresh evidence must upload"
        assert "census-artifacts/gpu_absent.marker" in path, (
            "the GPU-absence marker must ride the same upload"
        )


class TestExtrasDispatchSelection:
    """ci.yml's fourth nightly-lane selector and reusable extras-lane call (D-06).

    ``run_extras_compat`` follows the 261026-b34 per-lane dispatch pattern:
    a boolean defaulting true beside the existing three selectors, consumed
    through the ``inputs`` context (booleans stay booleans there, while
    ``github.event.inputs`` renders them as strings). The caller reaches
    the lane only through the dispatch-gated reusable call -- extras_compat.yml
    owns the nightly cron, so nightly and manual dispatch each run the lane
    exactly once.
    """

    def test_fourth_selector_declared_after_run_test_mamba(self) -> None:
        """run_extras_compat is the fourth boolean, exact shape, default true."""
        ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        inputs = ci[True]["workflow_dispatch"]["inputs"]
        assert list(inputs) == [
            "run_coverage_nightly",
            "run_example_nightly",
            "run_test_mamba",
            "run_extras_compat",
        ], (
            "the dispatch-selection block must declare exactly four booleans with "
            "run_extras_compat immediately after run_test_mamba (D-06)"
        )
        assert inputs["run_extras_compat"] == {
            "description": (
                "Run the extras-compat lane (deepspeed + flash-attn probes) on this dispatch"
            ),
            "type": "boolean",
            "default": True,
        }

    def test_reusable_call_gate_uses_inputs_boolean(self) -> None:
        """The extras-compat job gate reads inputs.run_extras_compat, dispatch-only."""
        ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        job = ci["jobs"]["extras-compat"]
        assert (
            job["if"] == "github.event_name == 'workflow_dispatch' && inputs.run_extras_compat"
        ), (
            "the reusable call must be gated on the dispatch event ANDed with the "
            "inputs-context boolean (never github.event.inputs string comparison)"
        )
        assert job["uses"] == "./.github/workflows/extras_compat.yml", (
            "the caller must reach the lane through the local reusable workflow path"
        )

    def test_reusable_call_passes_no_secrets_or_environment(self) -> None:
        """The call passes no secrets, no environment, and contents:read only."""
        ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        job = ci["jobs"]["extras-compat"]
        assert "secrets" not in job, "the extras lane needs no secrets (T-28-09)"
        assert "environment" not in job, "the extras lane declares no environment"
        assert job.get("permissions") == {"contents": "read"}, (
            "the caller must pass least-privilege permissions to the reusable lane"
        )

    def test_caller_keeps_no_extras_schedule_arm(self) -> None:
        """ci.yml's schedule block is unchanged; extras_compat.yml owns the 04:00 cron."""
        ci_text = CI_WORKFLOW.read_text(encoding="utf-8")
        ci = yaml.safe_load(ci_text)
        assert ci[True]["schedule"] == [{"cron": "0 3 * * *"}, {"cron": "30 5 * * *"}], (
            "ci.yml must keep exactly its two committed cron entries -- adding the "
            "extras cron here would double-run the lane nightly"
        )
        assert EXTRAS_NIGHTLY_CRON not in ci_text, (
            "the 04:00 UTC extras cron must live ONLY on extras_compat.yml (D-06: "
            "nightly coverage occurs exactly once)"
        )
        extras = yaml.safe_load(EXTRAS_WORKFLOW.read_text(encoding="utf-8"))
        assert extras[True]["schedule"] == [{"cron": EXTRAS_NIGHTLY_CRON}], (
            "extras_compat.yml retains sole ownership of the nightly extras cron"
        )

    def test_caller_gate_has_no_schedule_arm(self) -> None:
        """The reusable-call job's gate admits dispatch only -- no nightly duplication."""
        ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        gate = ci["jobs"]["extras-compat"]["if"]
        assert "github.event_name == 'schedule'" not in gate, (
            "the ci.yml caller must have no schedule arm: the nightly run comes "
            "from extras_compat.yml's own cron, so a schedule arm here would "
            "queue the lane twice a night on the single dnallm-nightly runner"
        )
