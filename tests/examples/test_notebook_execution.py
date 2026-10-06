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
import os
import json
import re
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
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

from tests.examples._execution import (
    EVO_KERNEL_NAME,
    EXAMPLE_DIR,
    LANGCHAIN_KERNEL_NAME,
    MEGADNA_KERNEL_NAME,
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    ensure_isolated_kernel,
    ensure_megadna_kernel,
    ensure_evo_kernel,
    evo_prerequisites_installed,
    megadna_prerequisites_installed,
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
#
# 08-02 (rice.uga.edu outage 2026-10-04): data_generation_and_inference
# downloads its rice inputs in-notebook via ``wget -c``, which skips
# complete files -- so seeding complete copies from the 05-06 census cache
# (gitignored .scratch/, dev-box only) makes the notebook input-outage
# proof while CI/fresh checkouts keep the in-notebook download as the
# only path.  Built by :func:`_rice_cache_extras` (unit-tested below).
RICE_CACHE_DIR = EXAMPLE_DIR.parent / ".scratch" / "census-out" / "inputs"
RICE_CACHE_NAMES = ("osa1_r7.asm.fa.gz", "osa1_r7.all_models.gff3.gz")


def _rice_cache_extras(cache_dir: Path = RICE_CACHE_DIR) -> list[tuple[Path, str]]:
    """Build (cached-file, sandbox-name) seed tuples for the rice inputs.

    Returns one tuple per COMPLETE cached file (dev-box mirror of the
    notebook's documented rice.uga.edu URLs); empty on a cold cache so the
    notebook falls back to its own ``wget -c`` download cell.
    """
    return [
        (cached, cached.name)
        for name in RICE_CACHE_NAMES
        if (cached := cache_dir / name).is_file() and cached.stat().st_size > 0
    ]


_NOTEBOOK_EXTRA_INPUTS: dict[str, list[tuple[Path, str]]] = {
    "notebooks/benchmark/benchmark.ipynb": [
        (EXAMPLE_DIR / "notebooks" / "inference" / "test.csv", "../inference/test.csv"),
    ],
    "notebooks/finetune_NER_task/data_generation_and_inference.ipynb": _rice_cache_extras(),
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
    # Mirror gated_sandbox (D-05, 09-02): forward any spec-driven yaml_patch
    # so a future cut on an ACTIVE notebook is never silently ignored --
    # specs without a yaml_patch key seed unchanged (None).
    spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
    yield seed_sandbox(
        nb_path.parent, tmp_path, extra_inputs=extras, yaml_overrides=spec.get("yaml_patch")
    )
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
            env=spec.get("env"),
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


class TestSeedSandboxYamlOverrides:
    """Kernel-free contract for the D-05 sandbox-only YAML patch seam (09-02).

    The finetune_custom_head nightly runtime cut (``num_train_epochs`` 3 -> 1,
    owner A/A decision 2026-10-05) must land ONLY in the seeded sandbox copy:
    committed example content -- and the census executability claim resting on
    it -- stays byte-identical, and the loop body the notebook runs is
    identical (only the epoch count changes).  These tests pin the
    :func:`seed_sandbox` ``yaml_overrides`` contract (fail-closed on a missing
    target file or missing section, no-op by default) and the
    NOTEBOOK_EXEC_SPECS ``yaml_patch`` key that drives it from the spec layer
    (the spec-env precedent).
    """

    @staticmethod
    def _seed_dir(tmp_path: Path) -> Path:
        """Create a tiny fake example dir holding one notebook and one finetune YAML."""
        src_dir = tmp_path / "fake_finetune_example"
        src_dir.mkdir()
        (src_dir / "tiny.ipynb").write_text("{}", encoding="utf-8")
        (src_dir / "finetune_config.yaml").write_text(
            'task:\n    task_type: "binary"\nfinetune:\n    num_train_epochs: 3\n    seed: 42\n',
            encoding="utf-8",
        )
        return src_dir

    def test_override_patches_sandbox_copy_only(self, tmp_path: Path) -> None:
        """D-05: the SANDBOX copy loads num_train_epochs 1 while the SOURCE still reads 3."""
        src_dir = self._seed_dir(tmp_path)

        sandbox = seed_sandbox(
            src_dir,
            tmp_path / "run",
            yaml_overrides={"finetune_config.yaml": {"finetune": {"num_train_epochs": 1}}},
        )

        patched = yaml.safe_load((sandbox / "finetune_config.yaml").read_text(encoding="utf-8"))
        source = yaml.safe_load((src_dir / "finetune_config.yaml").read_text(encoding="utf-8"))
        assert patched["finetune"]["num_train_epochs"] == 1, (
            "the sandbox copy must carry the patched epoch count (the D-05 cut)"
        )
        assert source["finetune"]["num_train_epochs"] == 3, (
            "the committed-side source file must never be mutated by a sandbox "
            "override (D-05 honesty contract)"
        )

    def test_missing_override_target_file_raises(self, tmp_path: Path) -> None:
        """Fail-closed: an override naming a file absent from the sandbox raises ValueError."""
        src_dir = self._seed_dir(tmp_path)

        with pytest.raises(ValueError, match=re.escape("absent_config.yaml")):
            seed_sandbox(
                src_dir,
                tmp_path / "run",
                yaml_overrides={"absent_config.yaml": {"finetune": {"num_train_epochs": 1}}},
            )

    def test_missing_override_section_raises(self, tmp_path: Path) -> None:
        """Fail-closed: an override naming an absent section raises ValueError naming it."""
        src_dir = self._seed_dir(tmp_path)

        with pytest.raises(ValueError, match="no_such_section"):
            seed_sandbox(
                src_dir,
                tmp_path / "run",
                yaml_overrides={
                    "finetune_config.yaml": {"no_such_section": {"num_train_epochs": 1}}
                },
            )

    def test_finetune_custom_head_spec_pins_the_epochs_cut(self) -> None:
        """The spec yaml_patch key equals the D-05 patch -- guards editorial removal."""
        nb_path = EXAMPLE_DIR / "notebooks" / "finetune_custom_head" / "finetune.ipynb"

        spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]

        assert spec.get("yaml_patch") == {
            "finetune_config.yaml": {"finetune": {"num_train_epochs": 1}}
        }, (
            "the finetune_custom_head entry must carry the D-05 sandbox YAML patch -- "
            "without it the seeded copy runs the committed 3 epochs (~31 min nightly)"
        )

    def test_no_overrides_leaves_sandbox_yaml_byte_identical(self, tmp_path: Path) -> None:
        """Default no-op: without yaml_overrides the sandbox YAML stays byte-identical."""
        src_dir = self._seed_dir(tmp_path)

        sandbox = seed_sandbox(src_dir, tmp_path / "run")

        assert (sandbox / "finetune_config.yaml").read_bytes() == (
            src_dir / "finetune_config.yaml"
        ).read_bytes(), "existing callers (no yaml_overrides) must be unaffected"


class TestRiceCacheExtras:
    """Contract for :func:`_rice_cache_extras` (08-02 rice.uga.edu outage).

    Warm cache -> one seed tuple per complete file (name, size order from
    RICE_CACHE_NAMES); cold/partial cache -> fewer or none, so the
    notebook keeps its own ``wget -c`` download cell as the only path.
    """

    def test_warm_cache_yields_both_tuples(self, tmp_path: Path) -> None:
        """Both complete cached files become (path, name) seed tuples."""
        for name in RICE_CACHE_NAMES:
            (tmp_path / name).write_bytes(b"x" * 16)
        extras = _rice_cache_extras(tmp_path)
        assert [(src.name, dest) for src, dest in extras] == [
            ("osa1_r7.asm.fa.gz", "osa1_r7.asm.fa.gz"),
            ("osa1_r7.all_models.gff3.gz", "osa1_r7.all_models.gff3.gz"),
        ]

    def test_cold_cache_yields_nothing(self, tmp_path: Path) -> None:
        """No cache dir -> no extras (notebook downloads in-cell)."""
        assert _rice_cache_extras(tmp_path / "cold") == []

    def test_zero_byte_cache_entry_is_skipped(self, tmp_path: Path) -> None:
        """A truncated/zero-byte cached file must not seed a sandbox."""
        (tmp_path / RICE_CACHE_NAMES[0]).write_bytes(b"")
        (tmp_path / RICE_CACHE_NAMES[1]).write_bytes(b"full")
        extras = _rice_cache_extras(tmp_path)
        assert [src.name for src, _dest in extras] == [RICE_CACHE_NAMES[1]]


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

    # -- D-13 retry-window contract (08-08) --------------------------------

    def test_retry_probe_succeeds_first_try(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reachable-first-try url returns green on attempt 1/30, no sleeps."""
        sleeps: list[float] = []
        monkeypatch.setattr(
            sys.modules[__name__], "_probe_http", lambda url, timeout_s=5.0: (True, "HTTP 200")
        )
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        ok, evidence = _probe_http_with_retry("http://127.0.0.1:1/tags")
        assert ok is True
        assert "attempt 1/30" in evidence
        assert sleeps == [], "a first-try success must never sleep"

    def test_retry_probe_succeeds_after_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A service green from attempt 3 returns green; sleeps 1 and 2 happened."""
        calls: list[int] = []

        def warmup_probe(url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
            calls.append(1)
            return (True, "HTTP 200") if len(calls) >= 3 else (False, "URLError: warming")

        monkeypatch.setattr(sys.modules[__name__], "_probe_http", warmup_probe)
        ok, evidence = _probe_http_with_retry("http://127.0.0.1:1/tags", attempts=5)
        assert ok is True
        assert "attempt 3/5" in evidence

    def test_retry_probe_caps_retries_and_logs_every_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A persistently-down url stops at the cap; evidence carries every attempt."""
        sleeps: list[float] = []
        monkeypatch.setattr(
            sys.modules[__name__],
            "_probe_http",
            lambda url, timeout_s=5.0: (False, "URLError: refused"),
        )
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        ok, evidence = _probe_http_with_retry("http://127.0.0.1:1/tags", attempts=4, interval_s=2.0)
        assert ok is False
        assert "unreachable after 4 attempts" in evidence
        assert evidence.count("URLError: refused") == 4, (
            "every attempt's verbatim result must ride the evidence"
        )
        assert sleeps == [2.0] * 3, "retries pause between attempts but not after the last"


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
        # The ollama leg probes through the D-13 retry window (08-08); a
        # down fake would otherwise burn 30x2s of real sleep here.
        monkeypatch.setattr(time, "sleep", lambda _s: None)

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


class TestSpecEnvOverrides:
    """Per-notebook kernel env overrides (08-03, giants tier CI-05/D-14).

    A spec entry's optional ``env`` dict is merged over the global
    ``_ENV_OVERRIDES`` for that notebook's kernel execution only; the
    save/update/restore sandwich in ``run_notebook`` must return every
    touched key -- spec-only keys included -- to its prior state.
    """

    EVO_SPEC = str(EXAMPLE_DIR / "notebooks" / "generation_evo_models" / "inference.ipynb")

    def test_evo_spec_carries_runtime_expanded_giants_hf_hub_cache(self) -> None:
        """The evo notebook points its kernel at the giants hub, outside the quota cache."""
        env = NOTEBOOK_EXEC_SPECS[self.EVO_SPEC]["env"]
        assert env["HF_HUB_CACHE"] == os.path.expanduser("~/models-giants/hub")

    def test_run_notebook_env_defaults_to_none(self) -> None:
        """run_notebook exposes env with a None default so pre-existing callers are unchanged."""
        params = inspect.signature(run_notebook).parameters
        assert params["env"].default is None

    def test_spec_env_keys_set_during_execution_and_restored_after(
        self, monkeypatch, tmp_path
    ) -> None:
        """Spec env keys are live for the kernel and fully restored afterwards.

        Uses a fake NotebookClient (kernel-free, fast lane) that snapshots
        os.environ at execute() time: the spec-only key and the spec-wins
        override must both be visible during execution, and the sandwich
        must restore pre-existing values / remove absent ones afterwards.
        """
        from tests.examples import _execution

        seen: dict[str, str | None] = {}

        class _FakeClient:
            def __init__(self, nb, **kwargs):
                self.nb = nb

            def execute(self):
                for key in ("GSD_SPEC_ONLY", "GSD_SPEC_WINS", "MPLBACKEND"):
                    seen[key] = os.environ.get(key)

        monkeypatch.setattr(_execution, "NotebookClient", _FakeClient)
        monkeypatch.setenv("GSD_SPEC_WINS", "old-value")
        monkeypatch.setenv("MPLBACKEND", "original-backend")
        monkeypatch.delenv("GSD_SPEC_ONLY", raising=False)

        nb_path = tmp_path / "fake.ipynb"
        nbformat.write(nbf.new_notebook(cells=[nbf.new_code_cell("pass")]), nb_path)

        run_notebook(
            nb_path,
            tmp_path,
            cell_timeout=60,
            env={
                "GSD_SPEC_ONLY": "spec-only-value",
                "GSD_SPEC_WINS": "spec-wins-value",
                "MPLBACKEND": "Agg",
            },
        )

        # During kernel execution: spec-only key set, spec key wins over
        # both the prior value and the global override.
        assert seen == {
            "GSD_SPEC_ONLY": "spec-only-value",
            "GSD_SPEC_WINS": "spec-wins-value",
            "MPLBACKEND": "Agg",
        }
        # After: every touched key back to its prior state.
        assert os.environ["GSD_SPEC_WINS"] == "old-value"
        assert os.environ["MPLBACKEND"] == "original-backend"
        assert "GSD_SPEC_ONLY" not in os.environ


class TestLoraMirrorEndpoint:
    """Spec-level mirror endpoint for the lora pair (08-08 repair).

    The dev box cannot reach huggingface.co directly while hf-mirror.com
    (the endpoint dnallm's own ``use_mirror`` toggle installs) serves the
    family's repos; without the override the UNCACHED PlantCAD2 LoRA
    adapter fails download after retries while the cached base model
    silently passes -- splitting the pair.  The pin keeps the override
    from being silently dropped (an editorial revert fails in seconds on
    the fast lane, not at the next 25-minute real execution).
    """

    LORA_SPECS = (
        str(EXAMPLE_DIR / "notebooks" / "lora_finetune_inference" / "lora_finetune.ipynb"),
        str(EXAMPLE_DIR / "notebooks" / "lora_finetune_inference" / "lora_inference.ipynb"),
    )

    def test_both_lora_specs_pin_the_mirror_endpoint(self) -> None:
        """Both lora specs carry HF_ENDPOINT=hf-mirror.com for the kernel."""
        for spec in self.LORA_SPECS:
            assert NOTEBOOK_EXEC_SPECS[spec]["env"]["HF_ENDPOINT"] == "https://hf-mirror.com", spec


class TestMegadnaIsolatedLane:
    """Spec-level wiring for the isolated megaDNA lane (08-04).

    finetune_generation executes under the dedicated ``dnallm-megadna``
    kernelspec (kernel.json env ``VIRTUAL_ENV`` pinned to the throwaway
    ``.scratch/megadna-venvs/megadna`` venv); its gate probes THAT venv's
    interpreter for the FEASIBILITY-locked prerequisites instead of the
    running one.  The two read-only megaDNA siblings keep the
    project-venv gate until their own family rollout (08-05).
    """

    FGEN_SPEC = str(EXAMPLE_DIR / "notebooks" / "finetune_generation" / "finetune_generation.ipynb")
    SIBLING_SPECS = (
        str(EXAMPLE_DIR / "notebooks" / "generation_megaDNA" / "inference.ipynb"),
        str(EXAMPLE_DIR / "notebooks" / "finetune_custom_head" / "finetune.ipynb"),
    )

    def test_finetune_generation_spec_pins_megadna_kernel(self) -> None:
        """The finetune_generation spec routes to the isolated kernelspec."""
        assert NOTEBOOK_EXEC_SPECS[self.FGEN_SPEC]["kernel_name"] == MEGADNA_KERNEL_NAME

    def test_megadna_siblings_keep_default_project_kernel(self) -> None:
        """The sibling megaDNA notebooks carry no kernel override yet."""
        for spec in self.SIBLING_SPECS:
            assert "kernel_name" not in NOTEBOOK_EXEC_SPECS[spec], spec

    def test_gate_green_when_venv_prerequisites_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A green venv probe -> plain return (the caller executes)."""
        # Patch the RUNNING module object, not a re-imported copy (the
        # established _probe_http idiom in this file).
        monkeypatch.setattr(
            sys.modules[__name__],
            "megadna_prerequisites_installed",
            lambda: (True, "fake venv green"),
        )
        assert _gate_megadna_isolated("nb.ipynb") is None

    def test_gate_skips_typed_when_venv_cold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cold venv -> optional-dep typed skip carrying the probe evidence."""
        monkeypatch.setattr(
            sys.modules[__name__],
            "megadna_prerequisites_installed",
            lambda: (False, "fake venv cold evidence"),
        )
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _gate_megadna_isolated("nb.ipynb")
        message = str(excinfo.value.args[0])
        assert message.startswith("optional-dep:"), message
        assert "fake venv cold evidence" in message, message
        assert "isolated megaDNA venv" in message, message


class TestEvoIsolatedLane:
    """Spec-level wiring for the isolated evo giants lane (08-06).

    The evo notebook executes under the dedicated ``dnallm-evo-kernel``
    kernelspec (kernel.json env ``VIRTUAL_ENV`` pinned to the throwaway
    ``.scratch/evo-venvs/evo`` venv) with its ``HF_HUB_CACHE`` pointed at
    the giants dir (CI-05); the gate probes THAT venv's interpreter for
    the FEASIBILITY-locked prerequisites instead of the running one.
    """

    EVO_SPEC = str(EXAMPLE_DIR / "notebooks" / "generation_evo_models" / "inference.ipynb")

    def test_evo_spec_pins_isolated_kernel_and_giants_env(self) -> None:
        """The evo notebook routes to the isolated kernelspec + giants hub."""
        spec = NOTEBOOK_EXEC_SPECS[self.EVO_SPEC]
        assert spec["kernel_name"] == EVO_KERNEL_NAME
        assert spec["env"]["HF_HUB_CACHE"] == os.path.expanduser("~/models-giants/hub")
        # Hermetic lane (08-06): the prefetched giants tier serves both model
        # legs; revision resolution never head-calls the Hub.
        assert spec["env"]["HF_HUB_OFFLINE"] == "1"

    def test_gate_green_when_venv_prerequisites_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A green venv probe -> plain return (the caller executes)."""
        monkeypatch.setattr(
            sys.modules[__name__],
            "evo_prerequisites_installed",
            lambda: (True, "fake evo venv green"),
        )
        assert _gate_evo("nb.ipynb") is None

    def test_gate_skips_typed_when_venv_cold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cold venv -> optional-dep typed skip carrying the probe evidence."""
        monkeypatch.setattr(
            sys.modules[__name__],
            "evo_prerequisites_installed",
            lambda: (False, "fake evo venv cold evidence"),
        )
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _gate_evo("nb.ipynb")
        message = str(excinfo.value.args[0])
        assert message.startswith("optional-dep:"), message
        assert "fake evo venv cold evidence" in message, message
        assert "isolated evo venv" in message, message


class TestFinetuneGenerationContentContracts:
    """Fast JSON-level contracts for the repaired finetune_generation (08-04).

    The notebook executed end-to-end under the isolated dnallm-megadna
    kernelspec; these pin the content invariants the execution proved, so a
    future editorial revert fails in seconds on the fast lane instead of at
    the next 14-minute real execution (REPAIR-01 same-commit regression).
    """

    NB_PATH = EXAMPLE_DIR / "notebooks" / "finetune_generation" / "finetune_generation.ipynb"

    @classmethod
    def _code_cells(cls) -> list[str]:
        nb = json.loads(cls.NB_PATH.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def test_provenance_stamp_is_first_code_cell(self) -> None:
        """The D-21 stamp (key=value version lines) leads the code cells."""
        first = self._code_cells()[0]
        assert "transformers_version=" in first
        assert "torch_version=" in first
        assert "fla_version=" in first
        assert "megadna_commit=cb2f5ab4cc88dc0effe05c5f23358862c837014a" in first

    def test_genome_download_precedes_fasta_load(self) -> None:
        """The Ensembl !wget cell executes before the Fasta(...) load cell."""
        cells = self._code_cells()
        wget = next(i for i, s in enumerate(cells) if "!wget" in s and "ensemblgenomes" in s)
        # Match the executable call, not prose: the wget cell's own comment
        # mentions Fasta(...) too.
        fasta = next(i for i, s in enumerate(cells) if "= Fasta(" in s)
        assert wget < fasta

    def test_megadna_install_is_pinned_and_precedes_model_load(self) -> None:
        """The megaDNA install cell carries the FEASIBILITY pins and runs first."""
        cells = self._code_cells()
        install = next(
            i
            for i, s in enumerate(cells)
            if "git clone https://github.com/lingxusb/megaDNA.git" in s
        )
        assert "cb2f5ab4cc88dc0effe05c5f23358862c837014a" in cells[install]
        assert "MEGABYTE_pytorch==0.2.1" in cells[install]
        load = next(i for i, s in enumerate(cells) if "megaDNA_updated" in s and "load_model" in s)
        assert install < load

    def test_megadna_column_drop_is_stack_version_robust(self) -> None:
        """The MEGA-DNA column drop keeps only columns present on the stack.

        transformers 5.x fast tokenizers no longer emit token_type_ids, so
        the pre-repair hardcoded remove_columns list raised ValueError
        mid-notebook (repaired 08-04); the present-filter form runs on both
        sides of the transformers 4.49-5.x span.
        """
        cells = self._code_cells()
        drop = next(i for i, s in enumerate(cells) if "remove_columns(" in s)
        assert "if column in data.dataset.column_names" in cells[drop]
        assert "token_type_ids" in cells[drop]


class TestMegadnaSiblingContentContracts:
    """Fast JSON-level contracts for the repaired megaDNA siblings (08-05).

    Both notebooks executed end-to-end on the default project kernel with
    the FEASIBILITY-locked prerequisites installed into the project venv
    (reversible exact-version install); these pin the content invariants
    the executions proved, so an editorial revert fails in seconds on the
    fast lane instead of at the next real execution (REPAIR-01
    same-commit regression).
    """

    GEN_PATH = EXAMPLE_DIR / "notebooks" / "generation_megaDNA" / "inference.ipynb"
    HEAD_PATH = EXAMPLE_DIR / "notebooks" / "finetune_custom_head" / "finetune.ipynb"

    @staticmethod
    def _code_cells(path: Path) -> list[str]:
        nb = json.loads(path.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def test_siblings_carry_provenance_stamps(self) -> None:
        """Each sibling carries its own D-21 stamp (key=value lines + pins)."""
        for path in (self.GEN_PATH, self.HEAD_PATH):
            cells = self._code_cells(path)
            stamp = next(
                (s for s in cells if "transformers_version=" in s and "megadna_commit=" in s),
                None,
            )
            assert stamp is not None, f"no D-21 stamp in {path.name}"
            assert "torch_version=" in stamp
            assert "fla_version=not-used" in stamp
            assert "megabyte_version=0.2.1" in stamp

    def test_install_is_pinned_and_precedes_model_load(self) -> None:
        """The floating clone is gone; the FEASIBILITY pins precede the load."""
        for path in (self.GEN_PATH, self.HEAD_PATH):
            cells = self._code_cells(path)
            install = next(
                i
                for i, s in enumerate(cells)
                if "git clone https://github.com/lingxusb/megaDNA.git" in s
            )
            assert "cb2f5ab4cc88dc0effe05c5f23358862c837014a" in cells[install]
            assert "MEGABYTE_pytorch==0.2.1" in cells[install]
            load = next(
                i for i, s in enumerate(cells) if "megaDNA_updated" in s and "load_model" in s
            )
            assert install < load, f"install after load in {path.name}"

    @staticmethod
    def _active_lines(cell: str) -> str:
        """Drop comment lines: notebooks document the alternative source=
        routes as comments, and the contract pins the ACTIVE route."""
        return "\n".join(line for line in cell.splitlines() if not line.strip().startswith("#"))

    def test_source_routes_are_d15_aligned(self) -> None:
        """Active source= route matches the lock direction: ms-mirrored ids
        take modelscope, the HF-only lingxusb id takes huggingface."""
        head_cells = self._code_cells(self.HEAD_PATH)
        dnagpt = next(s for s in head_cells if "plant-dnagpt-BPE" in s and "load_model" in s)
        dnagpt_active = self._active_lines(dnagpt)
        assert 'source="modelscope"' in dnagpt_active
        assert 'source="huggingface"' not in dnagpt_active
        megadna = next(s for s in head_cells if "megaDNA_updated" in s and "load_model" in s)
        megadna_active = self._active_lines(megadna)
        assert 'source="huggingface"' in megadna_active
        assert 'source="modelscope"' not in megadna_active
        gen_cells = self._code_cells(self.GEN_PATH)
        gen_load = next(s for s in gen_cells if "megaDNA_updated" in s and "load_model" in s)
        gen_active = self._active_lines(gen_load)
        assert 'source="huggingface"' in gen_active
        assert 'source="modelscope"' not in gen_active


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

OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
MCP_ENDPOINT = "http://localhost:8000/mcp"

# D-07 stage contract (08-08, MCP-02): within one example-nightly job the
# three GPU/VRAM/port consumers run STAGED SERIAL, never concurrently --
#   stage 1 TORCH-HEAVY: pytest tests/examples with the two mcp gated ids
#     DESELECTED (Pitfall 7: `--deselect` both mcp_example entries, else the
#     pair runs once as an honest skip here and once for real in stage 3,
#     polluting the junit audit);
#   stage 2 MCP LIVE SERVER: start the dnallm MCP server on :8000 (yaml
#     host/port -- the CLI flags are dead), run dnallm/mcp/tests, stop it;
#   stage 3 OLLAMA BATCH: with ollama up (D-13 probe below), re-run just the
#     two gated mcp tests (server up + ollama probed).
# BETWEEN stages: explicit kernel pkill + VRAM settle (a few seconds' sleep
# after torch-heavy work lets the allocator release before ollama loads the
# ~3.3GB qwen3.5:4b weights -- owner swap 2026-10-06 15:27 CST). The ci.yml
# wiring lands in 08-09; these module comments are the contract it
# implements.


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


def _probe_http_with_retry(
    url: str,
    attempts: int = 30,
    interval_s: float = 2.0,
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """Retry-window wrapper over :func:`_probe_http` (D-13, 08-08).

    Retries *url* up to ``attempts`` times, ``interval_s`` apart (~60s total
    at the defaults) -- a runner-fresh ollama service may still be loading
    the ~3.3GB qwen3.5:4b weights when the test starts (the smaller model
    loads faster than the retired 17GB one, but the ~60s D-13 window
    contract is unchanged and the probe stays HTTP-reachability-only over
    /api/tags), and a single 2s probe
    would typed-skip on a service that is merely warming up. Returns as soon
    as any attempt succeeds; a persistent failure returns (False, evidence
    naming the url, the attempt count, and every attempt's verbatim result).
    Tests patch ``time.sleep`` so the window costs milliseconds in CI.
    """
    results: list[str] = []
    for attempt in range(1, attempts + 1):
        ok, evidence = _probe_http(url, timeout_s=timeout_s)
        if ok:
            return True, f"{evidence} (attempt {attempt}/{attempts})"
        results.append(f"attempt {attempt}: {evidence}")
        if attempt < attempts:
            time.sleep(interval_s)
    return False, f"{url} unreachable after {attempts} attempts: " + "; ".join(results)


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

    D-13 (08-08): the ollama leg probes through the ~60s retry window --
    a warming service (17GB model still loading) is distinguished from an
    absent one, and a persistent failure carries the full retry log in the
    skip message.  The D-13 fallback is infra-missing ONLY: a model that
    answers with wrong output is an assertion failure to be repaired,
    never a skip.
    """
    ollama_ok, ollama_ev = _probe_http_with_retry(OLLAMA_URL)
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
    """Isolated evo lane (08-06): probe the kernelspec venv, not this one.

    The evo notebook executes under the ``dnallm-evo-kernel`` kernelspec
    (``VIRTUAL_ENV`` pinned to ``.scratch/evo-venvs/evo``), so its
    prerequisites -- stripedhyena + evo2 + flash-attn -- live in THAT
    venv; a probe of the running interpreter would report absent forever
    (the project venv intentionally imports none of them).  Green -> the
    caller registers the kernelspec (idempotent, never installs) and
    executes for real; cold venv -> the honest optional-dep typed skip
    with the venv probe evidence embedded.
    """
    installed, evidence = evo_prerequisites_installed()
    if installed:
        return
    optional_dep_skip(
        f"execute {nb_name} (isolated evo venv prerequisites install-gated)",
        evidence=evidence,
    )


def _gate_megadna(nb_name: str) -> None:
    """megaDNA: pinned clone importable as ``megaDNA`` + MEGABYTE_pytorch."""
    _gate_optional_deps(nb_name, ("megaDNA", "MEGABYTE_pytorch"))


def _gate_megadna_isolated(nb_name: str) -> None:
    """Isolated megaDNA lane (08-04): probe the throwaway venv, not this one.

    finetune_generation runs under the ``dnallm-megadna`` kernelspec
    (VIRTUAL_ENV pinned to ``.scratch/megadna-venvs/megadna``), so its
    prerequisites live in THAT venv; a probe of the running interpreter
    would report absent forever.  Green -> the caller provisions the
    kernelspec (idempotent) and executes for real; cold venv -> the same
    honest optional-dep typed skip the project-venv family gate emits,
    with the venv probe evidence embedded.
    """
    installed, evidence = megadna_prerequisites_installed()
    if installed:
        return
    optional_dep_skip(
        f"execute {nb_name} (isolated megaDNA venv prerequisites install-gated)",
        evidence=evidence,
    )


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
        # Isolated lane (08-04): prerequisites probed in the throwaway
        # dnallm-megadna venv, not the project venv.
        _gate_megadna_isolated,
    ),
    ("notebooks/lora_finetune_inference/lora_finetune.ipynb", _gate_mamba),
    ("notebooks/lora_finetune_inference/lora_inference.ipynb", _gate_mamba),
]

# Gated entries whose per-cell budget (NOTEBOOK_EXEC_SPECS cell_timeout) is
# 3600s: the outer pytest-timeout mark must stay strictly ABOVE the cell
# timeout (run_notebook's contract -- an outer kill at the cell budget would
# preempt nbclient's clean CellTimeoutError handling and the partial-failure
# artifact capture), so these carry the 7200s override instead of the
# class-level 3600s mark. The mcp_example pair joined at cell-budget 3600
# per owner decision B (2026-10-06, 09-04): the num_ctx 8k cut is DEFERRED,
# so the agent brain serves the pair at its default context -- qwen3.5:4b
# since the owner swap decision 2026-10-06 15:27 CST (default context
# 262144, the same ~256k class as the previous qwen3.8; cell 3600s / outer
# 7200s budgets unchanged for the smaller model) -- and the un-cut latency
# tail crossed the old 1800s cell line live under the previous model (run
# 37406829738 stage 3 CellTimeoutError); revisit when/if the cut is
# un-deferred.
_TIMEOUT_7200_GATED: frozenset[str] = frozenset({
    "mcp_example/mcp_client_ollama_langchain_agents.ipynb",
    "mcp_example/mcp_client_ollama_pydantic_ai.ipynb",
    "notebooks/finetune_custom_head/finetune.ipynb",
    "notebooks/finetune_generation/finetune_generation.ipynb",
    "notebooks/lora_finetune_inference/lora_finetune.ipynb",
})

# Gated entries excluded from the example-nightly census by owner policy
# (D-01, directive 2026-10-05): giant-model (evo-class) execution tests.
# The runner environment is AVAILABLE, so a typed skip would fake
# "environment-unavailable" and pollute the skip audit -- the marker
# deselection is the only accepted mechanism and produces no skip message
# (CI-03).  The dispatch/manual lane runs these explicitly with -m giants.
# The fast evo contract tests (TestEvoIsolatedLane, TestSpecEnvOverrides
# evo rows) are deliberately NOT members: they are kernel-free fast-lane
# tests that must keep running on every fast leg.
_GIANTS_GATED: frozenset[str] = frozenset({
    "notebooks/generation_evo_models/inference.ipynb",
})


def _gated_test_param(nb_id: str) -> str | pytest.ParameterSet:
    """Build the parametrize entry for a gated notebook id, accumulating marks.

    Lane-exclusion marks are composable: the 7200s timeout override keeps
    the outer kill strictly above the 3600s cell budget, and the giants
    mark is the owner-policy census exclusion (D-01).

    Args:
        nb_id: Notebook id (POSIX relative to EXAMPLE_DIR).

    Returns:
        A bare id when no mark applies, else a :func:`pytest.param` carrying
        the accumulated marks.
    """
    marks: list[pytest.MarkDecorator] = []
    if nb_id in _TIMEOUT_7200_GATED:
        marks.append(pytest.mark.timeout(7200))
    if nb_id in _GIANTS_GATED:
        marks.append(pytest.mark.giants)
    if marks:
        return pytest.param(nb_id, marks=marks, id=nb_id)
    return nb_id


@pytest.mark.slow
@pytest.mark.timeout(3600)
class TestGatedNotebookExecution:
    """Probe-then-execute for environment-gated notebooks (D-05/D-06)."""

    @pytest.fixture
    def gated_sandbox(self, tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
        """Seed the gated notebook's directory and assert the tree stays clean."""
        nb_path = EXAMPLE_DIR / request.node.callspec.params["gated_id"]
        # D-05 (09-02): forward any spec-driven sandbox YAML patch the same
        # way the test body consumes spec keys (spec-env precedent) --
        # specs without a yaml_patch key seed unchanged (None).
        spec = NOTEBOOK_EXEC_SPECS[str(nb_path)]
        yield seed_sandbox(nb_path.parent, tmp_path, yaml_overrides=spec.get("yaml_patch"))
        assert_tree_clean()

    @pytest.mark.parametrize(
        "gated_id",
        [_gated_test_param(nb_id) for nb_id, _gate in GATED_NOTEBOOKS],
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
        elif kernel == MEGADNA_KERNEL_NAME:
            # Same contract for the isolated megaDNA lane (08-04): the gate
            # has already proven the pinned prerequisites live in the
            # throwaway venv, so provisioning only repairs/creates the
            # kernelspec -- a failure here is owner-visible, never a skip.
            ensure_megadna_kernel()
        elif kernel == EVO_KERNEL_NAME:
            # Isolated evo lane (08-06): register/repair the kernelspec
            # only -- the giants tier + evo stack are provisioned
            # out-of-band (dev box here; runner steps in 08-08/08-09), so
            # a failure is owner-visible, never a skip and never an
            # implicit install.
            ensure_evo_kernel()
        run_notebook(
            nb_path,
            gated_sandbox,
            cell_timeout=spec["cell_timeout"],
            artifact_dir=tmp_path / "artifacts",
            kernel_name=kernel,
            env=spec.get("env"),
        )
        assert_tree_clean()
