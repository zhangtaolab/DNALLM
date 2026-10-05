"""Runner-infrastructure contract tests (D-06/D-12, 09-02).

Pins the in-repo ollama systemd unit's TWO Environment lines and the
owner re-apply op documented in the runner README, so an editorial revert
of either pin fails on the fast lane in SECONDS instead of at the next
~25-minute real execution (D-07 same-change test contract):

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
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT_FILE = REPO_ROOT / "scripts" / "runner" / "ollama.service"
RUNNER_README = REPO_ROOT / "scripts" / "runner" / "README.md"


def _load_unit_text() -> str:
    """Return the in-repo ollama unit file text (the auditable source)."""
    return UNIT_FILE.read_text(encoding="utf-8")


def _load_runner_readme() -> str:
    """Return the runner README text (the owner re-apply op source)."""
    return RUNNER_README.read_text(encoding="utf-8")


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
