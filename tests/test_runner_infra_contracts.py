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
        version, and keep the installer pipeline only in the else branch.
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
