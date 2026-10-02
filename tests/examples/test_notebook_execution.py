"""Real end-to-end execution tests for example notebooks.

Census-driven rollout layer of the v1.1 example execution (D-08): the
parametrized test runs whole notebooks through the private nbclient
harness inside a tmp sandbox, and the kill test proves a hung kernel is
cleaned up (EXEC-06).  Every test here is slow-marked so hosted fast
legs never spawn kernels.  :data:`NOTEBOOK_EXEC_SPECS` carries budgets
for all 21 notebooks; the :data:`ACTIVE_NOTEBOOKS` list below holds
exactly the 05-06 census-green notebooks, and
:class:`TestGatedNotebookExecution` covers the environment-gated set
with probe-then-execute typed skips -- the rollout never widens
silently (census FAIL items stay census rows and the Phase 8 repair
queue).
"""

from __future__ import annotations

import importlib.util
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import nbformat
import nbformat.v4 as nbf
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

from tests.examples._execution import (
    EXAMPLE_DIR,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
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
ACTIVE_NOTEBOOKS = [
    EXAMPLE_DIR / "notebooks" / "inference" / "inference.ipynb",
    EXAMPLE_DIR / "notebooks" / "generation" / "inference.ipynb",
    EXAMPLE_DIR / "notebooks" / "in_silico_mutagenesis" / "in_silico_mutagenesis.ipynb",
    EXAMPLE_DIR / "notebooks" / "interpretation" / "interpretation.ipynb",
    EXAMPLE_DIR / "notebooks" / "data_prepare" / "predict" / "predict_data.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_binary" / "finetune_binary.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_multi_labels" / "finetune_multi_labels.ipynb",
    EXAMPLE_DIR / "notebooks" / "finetune_NER_task" / "data_generation_and_inference.ipynb",
]


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
    """
    nb_path = Path(request.node.callspec.params["nb_path"])
    yield seed_sandbox(nb_path.parent, tmp_path)
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


# --------------------------------------------------------------------------
# Gated census layer (D-05/D-06 ladder terminals; 05-06 Task 3)
# --------------------------------------------------------------------------
# Notebooks whose real execution is environment-gated.  Mirroring
# dnallm/mcp/tests/_network_skip.py: the probe runs at TEST time and the
# typed skip carries the live probe evidence; when the environment does
# provide the prerequisites the notebook executes for real (D-05
# execute-first), and a probe-green-but-forbidden state fails loudly
# instead of skipping.  Non-qualifying execution failures always
# re-raise -- never converted to skips.

OLLAMA_URL = "http://localhost:11434/api/tags"
MCP_ENDPOINT = "http://localhost:8000/mcp"


def _probe_http(url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    """GET *url*; return (reachable, evidence) without side effects."""
    try:
        # ruff: ignore[suspicious-url-open-usage]  # probe constants only
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return True, f"HTTP {response.status}"
    except Exception as exc:  # probe reports any transport failure verbatim
        return False, f"{type(exc).__name__}: {exc}"


def _probe_module(name: str) -> tuple[bool, str]:
    """find_spec probe; return (installed, evidence)."""
    spec = importlib.util.find_spec(name)
    return spec is not None, f"find_spec({name!r}) {'resolved' if spec else 'is None'}"


def _gate_ollama_stack(nb_name: str) -> None:
    """Gate for the two mcp client notebooks (T-05-16: never auto-execute).

    The notebooks talk to ollama at :11434 AND the dnallm MCP server at
    :8000/mcp; the langchain sibling additionally begins with ``uv pip
    install`` cells that must never run against the project venv.  Any
    unreachable endpoint is an honest network-unavailable skip carrying
    both live probe results; both endpoints up is the Phase-8-only state
    (ollama/VRAM coexistence plan + isolated install env), which fails
    loudly as an owner decision rather than silently skipping.
    """
    ollama_ok, ollama_ev = _probe_http(OLLAMA_URL)
    server_ok, server_ev = _probe_http(MCP_ENDPOINT)
    evidence = (
        f"ollama probe {OLLAMA_URL}: {'GREEN' if ollama_ok else 'down'} ({ollama_ev}); "
        f"dnallm MCP server {MCP_ENDPOINT}: {'up' if server_ok else 'unreachable'} "
        f"({server_ev})"
    )
    if not server_ok:
        network_unavailable_skip(
            f"execute {nb_name} (mcp client notebook)",
            evidence=evidence + "; execution additionally deferred pending the Phase-8 ollama/VRAM "
            "coexistence plan (owner decision, D-08/T-05-16)",
        )
    if not ollama_ok:
        network_unavailable_skip(f"execute {nb_name} (mcp client notebook)", evidence=evidence)
    pytest.fail(
        f"{nb_name}: both local endpoints are up, but executing the mcp notebooks in "
        "Phase 5 is forbidden (uv pip install cells would mutate the project venv, "
        "T-05-16) -- needs the Phase-8 ollama coexistence plan (owner decision)"
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
# its gate.  Budgets come from NOTEBOOK_EXEC_SPECS as everywhere else;
# the custom_head entry carries the 7200s mark matching its spec (the
# class mark is the plan's 3600 default for the rest).
GATED_NOTEBOOKS: list[tuple[str, object]] = [
    ("mcp_example/mcp_client_ollama_langchain_agents.ipynb", _gate_ollama_stack),
    ("mcp_example/mcp_client_ollama_pydantic_ai.ipynb", _gate_ollama_stack),
    ("notebooks/generation_evo_models/inference.ipynb", _gate_evo),
    ("notebooks/generation_megaDNA/inference.ipynb", _gate_megadna),
    (
        "notebooks/finetune_custom_head/finetune.ipynb",
        _gate_megadna,
    ),
    ("notebooks/lora_finetune_inference/lora_finetune.ipynb", _gate_mamba),
    ("notebooks/lora_finetune_inference/lora_inference.ipynb", _gate_mamba),
]


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
            if nb_id == "notebooks/finetune_custom_head/finetune.ipynb"
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
        run_notebook(
            nb_path,
            gated_sandbox,
            cell_timeout=spec["cell_timeout"],
            artifact_dir=tmp_path / "artifacts",
        )
        assert_tree_clean()
