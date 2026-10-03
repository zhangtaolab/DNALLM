"""Real end-to-end execution tests for example notebooks.

Census-driven rollout layer of the v1.1 example execution (D-08): the
parametrized test runs whole notebooks through the private nbclient
harness inside a tmp sandbox, and the kill test proves a hung kernel is
cleaned up (EXEC-06).  Every kernel-spawning test here is slow-marked so
hosted fast legs never spawn kernels (the :class:`TestSeedSandbox`
unit tests below run kernel-free in the fast lane).  :data:`NOTEBOOK_EXEC_SPECS`
carries budgets for all 21 notebooks; the :data:`ACTIVE_NOTEBOOKS` list
below holds the census-green notebooks (8 from the 05-06 campaign plus
the 5 repaired by quick task 261002-sl7: benchmark, finetune_data,
embedding_attention, finetune_NER_task, inference_for_tRNA -- 13 total),
and :class:`TestGatedNotebookExecution` covers the environment-gated set
with probe-then-execute typed skips -- the rollout never widens
silently (census FAIL items stay census rows and the Phase 8 repair
queue; finetune_generation joined the gated lane with the megaDNA gate,
its data-prep half repaired and evidenced 261002-sl7).  The mcp client
pair moved to its owner-approved EXECUTE state 261003-csd (D-08: both
endpoints up executes for real, the langchain sibling routed through
the isolated ``dnallm-mcp-langchain`` kernelspec, any endpoint down
typed-skips with both live probe results -- the T-05-16
never-auto-execute sentinel is retired).
"""

from __future__ import annotations

import importlib.util
import inspect
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import nbformat
import nbformat.v4 as nbf
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

from tests.examples._execution import (
    EXAMPLE_DIR,
    LANGCHAIN_KERNEL_NAME,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    ensure_isolated_kernel,
    network_unavailable_skip,
    optional_dep_skip,
    run_notebook,
    seed_sandbox,
)

# Census rollout list: every notebook here executes for real on each run.
# Grown by 05-06 from the pilot to exactly the census-green set (D-08):
# every entry PASSED real end-to-end execution in the 05-06 campaign
# (manifest evidence under .scratch/census-out/).  Census FAIL items stay
# census rows and the Phase 8 repair queue -- never silently rolled in.
# 261002-sl7 appended the five repaired census failures (owner
# instruction 2026-10-02): each PASSED a fresh post-repair execution in
# the 261002-sl7 campaign (evidence under .scratch/sl7/).
ACTIVE_NOTEBOOKS = [
    EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb",
    EXAMPLE_DIR / "notebooks" / "generation" / "inference.ipynb",
    EXAMPLE_DIR / "notebooks" / "in_silico_mutagenesis" / "in_silico_mutagenesis.ipynb",
    EXAMPLE_DIR / "notebooks" / "interpretation" / "interpretation.ipynb",
    EXAMPLE_DIR / "notebooks" / "data_prepare" / "predict" / "predict_data.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_binary" / "finetune_binary.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_multi_labels" / "finetune_multi_labels.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_NER_task" / "data_generation_and_inference.ipynb",
    EXAMPLE_DIR / "notebooks" / "benchmark" / "benchmark.ipynb",
    EXAMPLE_DIR / "notebooks" / "data_prepare" / "finetune" / "finetune_data.ipynb",
    EXAMPLE_DIR / "notebooks" / "embedding_attention.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_NER_task" / "finetune_NER_task.ipynb",
    EXAMPLE_DIR / "notebooks" / "inference_for_tRNA" / "inference.ipynb",
]


# Cross-directory sandbox inputs (261002-sl7): notebooks whose configs or
# code reach outside their own directory via cwd-relative paths.  Keys are
# POSIX ids relative to EXAMPLE_DIR (the parametrization id form); values
# are seed_sandbox (src, dest-relative-to-sandbox) tuples.  The benchmark
# config's ``path: ../inference/test.csv`` is the root cause this table
# fixes: without the sibling seeding, DNAInference.generate_dataset
# silently treats the missing path STRING as one "sequence", building a
# labels-less one-row dataset that crashes Benchmark.run at
# dnallm/inference/benchmark.py:296 (KeyError on the label column).
_NOTEBOOK_EXTRA_INPUTS: dict[str, list[tuple[Path, str]]] = {
    "notebooks/benchmark/benchmark.ipynb": [
        (EXAMPLE_DIR / "notebooks" / "inference" / "test.csv", "../inference/test.csv"),
    ],
}


# The sandbox fixture lives in this module (not a tests/examples/conftest.py):
# tests/ is not a package, so a second conftest.py would win the bare
# ``conftest`` module-name race and break the three existing test files that
# do ``from conftest import ...`` (test_trainer/test_benchmark/test_dna_dataset).
@pytest.fixture
def notebook_sandbox(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """Yield a seeded tmp sandbox of the executed notebook's directory.

    Seeds the parametrized notebook's parent directory over ``tmp_path``
    and asserts the repo tree stayed clean on teardown -- belt-and-braces
    beyond the in-test guard.  Generalized over the expanded spec dict
    (05-05): the seeded directory follows ``nb_path`` from the test's
    parametrization, so every census notebook gets a faithful sandbox.
    Cross-directory inputs ride along through the per-notebook
    :data:`_NOTEBOOK_EXTRA_INPUTS` table (261002-sl7).
    """
    nb_path = Path(request.node.callspec.params["nb_path"])
    extras = _NOTEBOOK_EXTRA_INPUTS.get(nb_path.relative_to(EXAMPLE_DIR).as_posix(), [])
    yield seed_sandbox(nb_path.parent, tmp_path, extra_inputs=extras)
    assert_tree_clean()


def _kernel_count() -> int:
    """Count live ipykernel_launcher processes.

    In-pytest use is self-match-safe: the pytest argv never contains the
    pattern, unlike the enclosing ``bash -c`` of a CI step, which would
    self-match (05-RESEARCH.md Pitfall 1).  Callers therefore compare
    against a captured pre-test baseline (delta-zero), never against
    absolute zero -- unrelated jupyter servers on the box are none of
    this test's business.
    """
    # ruff: ignore[start-process-with-partial-path]
    result = subprocess.run(
        ["pgrep", "-f", "ipykernel_launcher"],
        capture_output=True,
        text=True,
        check=False,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


@pytest.mark.slow
@pytest.mark.timeout(7200)
class TestNotebookExecution:
    """Real end-to-end notebook execution through the private harness."""

    @pytest.mark.parametrize(
        "nb_path",
        ACTIVE_NOTEBOOKS,
        ids=lambda p: str(p.relative_to(EXAMPLE_DIR)),
    )
    def test_notebook_executes_end_to_end(
        self,
        nb_path: Path,
        tmp_path: Path,
        notebook_sandbox: Path,
    ) -> None:
        """Execute the whole notebook in a tmp sandbox and prove it stays error-free."""
        spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
        sandbox = notebook_sandbox
        artifacts = tmp_path / "artifacts"
        nb = run_notebook(
            nb_path,
            sandbox,
            cell_timeout=spec["cell_timeout"],
            artifact_dir=artifacts,
        )

        # Structure-only invariants: every non-empty code cell ran to
        # completion with no error output (values are checked by the
        # notebook's own assertions, not duplicated here).
        code_cells = [
            cell
            for cell in nb.cells
            if cell.cell_type == "code"
            and (
                "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            ).strip()
        ]
        errored = [
            (index, output)
            for index, cell in enumerate(code_cells)
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errored, f"{nb_path.name} executed with error outputs: {errored}"
        assert_tree_clean()


class TestKernelLifecycle:
    """Deliberate-hang proof that the harness kills a stuck kernel (EXEC-06)."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)  # outer backstop: kernel start + 3s cell + kill + poll
    def test_hung_kernel_is_killed_and_cleaned_up(self, tmp_path: Path) -> None:
        """Prove a cell sleeping past the per-cell timeout gets killed with delta-zero kernels."""
        nb = nbf.new_notebook(
            cells=[
                nbf.new_code_cell('print("ok")'),
                nbf.new_code_cell("import time; time.sleep(300)"),
            ]
        )
        before = _kernel_count()
        client = NotebookClient(
            nb,
            timeout=3,
            kernel_name="python3",
            shutdown_kernel="immediate",
            resources={"metadata": {"path": str(tmp_path)}},
        )
        with pytest.raises(CellTimeoutError):
            client.execute()
        # Delta-zero, polled: the kill is async-ish and CI boxes differ
        # from the GB10 probe (which showed the kernel gone immediately),
        # so poll up to 15s in 0.5s steps before asserting.
        deadline = time.time() + 15
        while time.time() < deadline and _kernel_count() > before:
            time.sleep(0.5)
        assert _kernel_count() == before, "hung kernel survived the harness"


class TestPartialFailureArtifacts:
    """A failing notebook leaves its partial-execution artifacts behind (EXEC-01)."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_cell_error_captures_artifacts_and_reraises(self, tmp_path: Path) -> None:
        """A raising cell re-raises AND writes the executed node + error text to artifact_dir."""
        nb_path = tmp_path / "failing.ipynb"
        nbformat.write(
            nbf.new_notebook(
                cells=[
                    nbf.new_code_cell('marker = "cell-1-ran"'),
                    nbf.new_code_cell("raise RuntimeError('deliberate failure')"),
                    nbf.new_code_cell('print("never reached")'),
                ]
            ),
            nb_path,
        )
        artifacts = tmp_path / "artifacts"
        with pytest.raises(CellExecutionError):
            run_notebook(nb_path, tmp_path, cell_timeout=120, artifact_dir=artifacts)

        executed = artifacts / "failing.executed.ipynb"
        error_txt = artifacts / "failing.error.txt"
        assert executed.is_file(), "partial-execution notebook artifact missing"
        assert error_txt.is_file(), "error text artifact missing"
        partial = nbformat.read(executed, as_version=4)
        sources = [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in partial.cells
            if c.cell_type == "code"
        ]
        assert 'marker = "cell-1-ran"' in sources, "pre-error cell missing from partial artifact"
        assert "deliberate failure" in error_txt.read_text(encoding="utf-8")
        assert_tree_clean()


class TestSeedSandbox:
    """Kernel-free unit contract for :func:`seed_sandbox` extra inputs (261002-sl7).

    Covers both accepted shapes -- a bare ``Path`` copying into the sandbox
    root (the unchanged pilot contract) and a ``(src, dest-relative)``
    tuple copying to a cwd-relative sibling position (the benchmark
    ``../inference/test.csv`` root-cause fix) -- plus the T-sl7-03 guard:
    a destination escaping the pytest ``tmp_path`` is rejected with
    ``ValueError`` before anything is written outside the sandbox tree.
    """

    @staticmethod
    def _seed_dir(tmp_path: Path) -> Path:
        """Create a tiny fake example dir holding one notebook and one sibling input."""
        src_dir = tmp_path / "fake_example"
        src_dir.mkdir()
        (src_dir / "tiny.ipynb").write_text("{}", encoding="utf-8")
        (src_dir / "sibling.csv").write_text("sequence,label\nAT,1\n", encoding="utf-8")
        return src_dir

    def test_bare_path_extra_copies_into_sandbox_root(self, tmp_path: Path):
        """A bare Path extra lands beside the notebook inside the sandbox."""
        src_dir = self._seed_dir(tmp_path)
        extra = tmp_path / "outside.csv"
        extra.write_text("sequence\nAT\n", encoding="utf-8")

        sandbox = seed_sandbox(src_dir, tmp_path / "run", extra_inputs=[extra])

        assert (sandbox / "tiny.ipynb").is_file()
        assert (sandbox / "sibling.csv").is_file()
        assert (sandbox / "outside.csv").read_text(encoding="utf-8") == "sequence\nAT\n"

    def test_tuple_extra_copies_to_relative_sibling_position(self, tmp_path: Path):
        """A (src, '../inference/test.csv') tuple seeds the cwd-relative cross-dir input."""
        src_dir = self._seed_dir(tmp_path)
        extra_src = tmp_path / "repo_side_test.csv"
        extra_src.write_text("sequence,label\nGC,0\n", encoding="utf-8")

        sandbox = seed_sandbox(
            src_dir,
            tmp_path / "run",
            extra_inputs=[(extra_src, "../inference/test.csv")],
        )

        seeded = tmp_path / "run" / "inference" / "test.csv"
        assert seeded.is_file(), "the sibling-position destination must exist after seeding"
        assert seeded.read_text(encoding="utf-8") == "sequence,label\nGC,0\n"
        # the sandbox cwd itself is unchanged by the sibling seeding
        assert not (sandbox / "test.csv").exists()

    def test_tuple_extra_escape_beyond_tmp_path_is_rejected(self, tmp_path: Path):
        """A destination resolving outside tmp_path raises ValueError before any write."""
        src_dir = self._seed_dir(tmp_path)
        extra_src = tmp_path / "escape.csv"
        extra_src.write_text("x\n", encoding="utf-8")

        with pytest.raises(ValueError, match="outside the pytest tmp dir"):
            seed_sandbox(
                src_dir,
                tmp_path / "run",
                extra_inputs=[(extra_src, "../../../etc/evil.csv")],
            )

        escaped = (tmp_path.parent / "etc" / "evil.csv").resolve()
        assert not escaped.exists(), "the guard must reject before anything is written outside"


class TestProbeHonesty:
    """4xx-honesty contract for :func:`_probe_http` (261003-csd, fact 4).

    MCP streamable-http endpoints answer a bare GET with a 4xx (session /
    method semantics) and urllib raises ``HTTPError`` on 4xx, so folding
    any 4xx into "down" made a genuinely-up server probe unreachable
    forever.  Any HTTP answer below 500 must count as reachable;
    transport failures stay verbatim down evidence.
    """

    def test_http_405_answer_counts_as_reachable(self) -> None:
        """A local 405-returning http.server probes (True, evidence citing 405)."""

        class _Always405(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # http.server API name
                self.send_response(405)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return  # keep the test output silent

        server = HTTPServer(("127.0.0.1", 0), _Always405)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/mcp"
            ok, evidence = _probe_http(url, timeout_s=5.0)
        finally:
            server.shutdown()
            server.server_close()
        assert ok is True, (
            f"a 405 answer proves a server is bound there (got evidence {evidence!r})"
        )
        assert "405" in evidence

    def test_unbound_port_probes_down_with_verbatim_evidence(self) -> None:
        """A freed local port probes (False, non-empty transport evidence)."""
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ok, evidence = _probe_http(f"http://127.0.0.1:{port}/", timeout_s=5.0)
        assert ok is False
        assert evidence, "the down direction must carry verbatim transport evidence"


class TestGateMatrix:
    """Execute-state gate semantics for the ollama/mcp stack (D-08, 261003-csd).

    Owner decision D-08/T-05-16 (2026-10-03) retired the never-auto-execute
    sentinel: both endpoints up is now the owner-approved EXECUTE state
    (ollama/VRAM coexistence sanctioned).  Any endpoint down stays an
    honest typed skip carrying BOTH live probe results, with no
    deferred-coexistence clause and no pytest.fail path anywhere.
    """

    OLLAMA_UP = (True, "ollama-fake-evidence")
    OLLAMA_DOWN = (False, "ollama-down-evidence")
    SERVER_UP = (True, "server-fake-evidence")
    SERVER_DOWN = (False, "server-down-evidence")

    @staticmethod
    def _patch_probes(
        monkeypatch: pytest.MonkeyPatch,
        ollama: tuple[bool, str],
        server: tuple[bool, str],
    ) -> None:
        """Point both gate probes at canned (reachable, evidence) pairs."""

        def fake_probe(url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
            return ollama if url == OLLAMA_URL else server

        # Patch the RUNNING module object, not a re-imported copy: tests/
        # is not a regular package, so a dotted-string target would import
        # a second module object and leave the gate reading real probes.
        monkeypatch.setattr(sys.modules[__name__], "_probe_http", fake_probe)

    def test_both_up_executes_with_plain_return(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both probes green -> the gate returns None (the caller executes)."""
        self._patch_probes(monkeypatch, self.OLLAMA_UP, self.SERVER_UP)
        assert _gate_ollama_stack("nb.ipynb") is None

    @pytest.mark.parametrize(
        ("ollama", "server"),
        [(OLLAMA_UP, SERVER_DOWN), (OLLAMA_DOWN, SERVER_UP), (OLLAMA_DOWN, SERVER_DOWN)],
        ids=["server-down", "ollama-down", "both-down"],
    )
    def test_any_down_skips_typed_with_both_probe_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ollama: tuple[bool, str],
        server: tuple[bool, str],
    ) -> None:
        """Any endpoint down -> network-unavailable skip carrying both evidence strings."""
        self._patch_probes(monkeypatch, ollama, server)
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _gate_ollama_stack("nb.ipynb")
        message = str(excinfo.value.args[0])
        assert message.startswith("network-unavailable:"), message
        assert "ollama-fake-evidence" in message or "ollama-down-evidence" in message, message
        assert "server-fake-evidence" in message or "server-down-evidence" in message, message
        assert "deferred pending" not in message, (
            "the retired Phase-8 coexistence-deferral clause must not appear"
        )


class TestKernelPlumbing:
    """Spec-level kernel routing for the isolated langchain lane (261003-csd).

    The langchain notebook executes under the dedicated ``dnallm-mcp-langchain``
    kernelspec (kernel.json env ``VIRTUAL_ENV`` pinned to the throwaway
    ``.scratch/mcp-example-venvs/langchain`` venv, so the notebook's own
    ``!uv pip install -U`` cells never touch the project venv); the
    pydantic_ai sibling has no install cells and keeps the project-venv
    ``python3`` kernel; ``run_notebook`` defaults to ``python3`` so every
    existing caller is unchanged.
    """

    LANGCHAIN_SPEC = str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_langchain_agents.ipynb")
    PYDANTIC_SPEC = str(EXAMPLE_DIR / "mcp_example" / "mcp_client_ollama_pydantic_ai.ipynb")

    def test_langchain_spec_pins_isolated_kernel(self) -> None:
        """The langchain mcp spec routes execution to the isolated kernelspec."""
        assert NOTEBOOK_EXEC_SPECS[self.LANGCHAIN_SPEC]["kernel_name"] == "dnallm-mcp-langchain"

    def test_pydantic_ai_spec_keeps_default_project_kernel(self) -> None:
        """The pydantic_ai mcp spec carries no kernel override (python3 default)."""
        assert "kernel_name" not in NOTEBOOK_EXEC_SPECS[self.PYDANTIC_SPEC]

    def test_run_notebook_kernel_name_defaults_to_python3(self) -> None:
        """run_notebook exposes kernel_name with the project-kernel default."""
        params = inspect.signature(run_notebook).parameters
        assert params["kernel_name"].default == "python3"


# --------------------------------------------------------------------------
# Gated census layer (D-05/D-06 ladder terminals; 05-06 Task 3)
# --------------------------------------------------------------------------
# Notebooks whose real execution is environment-gated.  Mirroring
# dnallm/mcp/tests/_network_skip.py: the probe runs at TEST time and the
# typed skip carries the live probe evidence; when the environment does
# provide the prerequisites the notebook executes for real (D-05
# execute-first).  The mcp pair's probe-green-but-forbidden sentinel
# (T-05-16) was retired by owner decision D-08 of 2026-10-03: both-up is
# now the owner-approved execute state, with the langchain sibling
# running under its isolated kernelspec.  Non-qualifying execution
# failures always re-raise -- never converted to skips.

OLLAMA_URL = "http://localhost:11434/api/tags"
MCP_ENDPOINT = "http://localhost:8000/mcp"


def _probe_http(url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    """GET *url*; return (reachable, evidence) without side effects.

    Any HTTP answer below 500 counts as REACHABLE (261003-csd): MCP
    streamable-http endpoints answer a bare GET with a 4xx (session /
    method semantics) and urllib raises ``HTTPError`` on 4xx, so folding
    4xx into "down" made a genuinely-up server probe unreachable forever.
    URLError / timeout / generic exceptions remain (False, verbatim
    evidence).
    """
    try:
        # ruff: ignore[suspicious-url-open-usage]  # probe constants only
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            return True, f"HTTP {exc.code} ({exc.reason})"
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # probe reports any transport failure verbatim
        return False, f"{type(exc).__name__}: {exc}"


def _probe_module(name: str) -> tuple[bool, str]:
    """find_spec probe; return (installed, evidence)."""
    spec = importlib.util.find_spec(name)
    return spec is not None, f"find_spec({name!r}) {'resolved' if spec else 'is None'}"


def _gate_ollama_stack(nb_name: str) -> None:
    """Gate for the two mcp client notebooks (D-08 execute state, 261003-csd).

    The notebooks talk to ollama at :11434 AND the dnallm MCP server at
    :8000/mcp; the langchain sibling runs under the isolated
    ``dnallm-mcp-langchain`` kernelspec so its ``uv pip install`` cells
    never touch the project venv.  Both endpoints up -> plain return: the
    notebook executes for real (owner decision D-08/T-05-16 of 2026-10-03
    retired the never-auto-execute sentinel; ollama/VRAM coexistence is
    owner-sanctioned).  Any unreachable endpoint is an honest
    network-unavailable skip carrying both live probe results.
    """
    ollama_ok, ollama_ev = _probe_http(OLLAMA_URL)
    server_ok, server_ev = _probe_http(MCP_ENDPOINT)
    if ollama_ok and server_ok:
        return
    evidence = (
        f"ollama probe {OLLAMA_URL}: {'GREEN' if ollama_ok else 'down'} ({ollama_ev}); "
        f"dnallm MCP server {MCP_ENDPOINT}: {'up' if server_ok else 'unreachable'} "
        f"({server_ev})"
    )
    missing = []
    if not server_ok:
        missing.append(f"dnallm MCP server {MCP_ENDPOINT} unreachable")
    if not ollama_ok:
        missing.append(f"ollama {OLLAMA_URL} unreachable")
    network_unavailable_skip(
        f"execute {nb_name} (mcp client notebook)",
        evidence=f"{'; '.join(missing)}; {evidence}",
    )


def _gate_optional_deps(nb_name: str, modules: tuple[str, ...]) -> None:
    """Gate for prerequisite-gated notebooks: optional-dep skip when absent.

    All prerequisites present -> return (the caller executes the notebook
    for real, D-05).  Any missing -> the typed skip listing exactly which
    probes failed.
    """
    results = {name: _probe_module(name) for name in modules}
    missing = [name for name, (ok, _ev) in results.items() if not ok]
    if missing:
        evidence = "; ".join(f"{results[name][1]}" for name in missing)
        optional_dep_skip(f"execute {nb_name} (prerequisites install-gated)", evidence=evidence)


def _gate_evo(nb_name: str) -> None:
    """Evo legs: stripedhyena (evo-1) + evo2 package, both 05-06-proven."""
    _gate_optional_deps(nb_name, ("stripedhyena", "evo2"))


def _gate_megadna(nb_name: str) -> None:
    """megaDNA: pinned clone importable as ``megaDNA`` + MEGABYTE_pytorch."""
    _gate_optional_deps(nb_name, ("megaDNA", "MEGABYTE_pytorch"))


def _gate_mamba(nb_name: str) -> None:
    """PlantCAD remote code requires the mamba_ssm optional extra."""
    _gate_optional_deps(nb_name, ("mamba_ssm",))


# Gated parametrization: notebook id (POSIX relative to EXAMPLE_DIR) plus
# its gate.  Budgets come from NOTEBOOK_EXEC_SPECS as everywhere else; the
# per-test timeout marks come from the class ladder plus the explicit
# overrides below (the class mark is the plan's 3600s default).
GATED_NOTEBOOKS: list[tuple[str, object]] = [
    ("mcp_example/mcp_client_ollama_langchain_agents.ipynb", _gate_ollama_stack),
    ("mcp_example/mcp_client_ollama_pydantic_ai.ipynb", _gate_ollama_stack),
    ("notebooks/generation_evo_models/inference.ipynb", _gate_evo),
    ("notebooks/generation_megaDNA/inference.ipynb", _gate_megadna),
    (
        "notebooks/finetune_custom_head/finetune.ipynb",
        _gate_megadna,
    ),
    (
        "notebooks/finetune_generation/finetune_generation.ipynb",
        _gate_megadna,
    ),
    ("notebooks/lora_finetune_inference/lora_finetune.ipynb", _gate_mamba),
    ("notebooks/lora_finetune_inference/lora_inference.ipynb", _gate_mamba),
]

# Gated entries whose per-cell budget (NOTEBOOK_EXEC_SPECS cell_timeout) is
# 3600s: the outer pytest-timeout mark must stay strictly ABOVE the cell
# timeout (run_notebook's contract -- an outer kill at the cell budget would
# preempt nbclient's clean CellTimeoutError handling and the partial-failure
# artifact capture), so these carry the 7200s override instead of the
# class-level 3600s mark.
_TIMEOUT_7200_GATED: frozenset[str] = frozenset({
    "notebooks/finetune_custom_head/finetune.ipynb",
    "notebooks/finetune_generation/finetune_generation.ipynb",
    "notebooks/lora_finetune_inference/lora_finetune.ipynb",
})


@pytest.mark.slow
@pytest.mark.timeout(3600)
class TestGatedNotebookExecution:
    """Probe-then-execute for environment-gated notebooks (D-05/D-06)."""

    @pytest.fixture
    def gated_sandbox(self, tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
        """Seed the gated notebook's directory and assert the tree stays clean."""
        nb_path = EXAMPLE_DIR / request.node.callspec.params["gated_id"]
        yield seed_sandbox(nb_path.parent, tmp_path)
        assert_tree_clean()

    @pytest.mark.parametrize(
        "gated_id",
        [
            pytest.param(nb_id, marks=pytest.mark.timeout(7200))
            if nb_id in _TIMEOUT_7200_GATED
            else nb_id
            for nb_id, _gate in GATED_NOTEBOOKS
        ],
        ids=str,
    )
    def test_gated_notebook_probes_then_executes(
        self,
        gated_id: str,
        tmp_path: Path,
        gated_sandbox: Path,
    ) -> None:
        """Probe the gate; skip typed when gated, else execute for real."""
        gate = dict(GATED_NOTEBOOKS)[gated_id]
        gate(gated_id)
        nb_path = EXAMPLE_DIR / gated_id
        spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
        kernel = spec.get("kernel_name", "python3")
        if kernel == LANGCHAIN_KERNEL_NAME:
            # Provision (idempotent) BEFORE spawning; a failure here raises
            # -- and an unprovisioned spec would raise NoSuchKernel before
            # any cell runs, so the project venv is unreachable by
            # construction (T-mcp1-01).
            ensure_isolated_kernel()
        run_notebook(
            nb_path,
            gated_sandbox,
            cell_timeout=spec["cell_timeout"],
            artifact_dir=tmp_path / "artifacts",
            kernel_name=kernel,
        )
        assert_tree_clean()
