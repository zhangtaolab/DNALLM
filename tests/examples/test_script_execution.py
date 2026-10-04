"""Real end-to-end execution test for the example helper script.

Census lane of the v1.1 example rollout (D-08; dev-box leg of EXEC-04):
``example/notebooks/finetune_NER_task/generate_bpe_dataset.py`` runs
inside a tmp sandbox against its documented rice inputs -- downloaded
into the sandbox, never the repo tree -- and must freshly produce
``rice_gene_ner_BPE.pkl`` there.  Slow-marked so hosted fast legs never
download genomes.
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.examples._execution import (
    EXAMPLE_DIR,
    assert_tree_clean,
    environment_unavailable_skip,
    run_example_script,
    seed_sandbox,
)

SCRIPT_DIR = EXAMPLE_DIR / "notebooks" / "finetune_NER_task"
SCRIPT = SCRIPT_DIR / "generate_bpe_dataset.py"

# GAP-1 terminal signature (05-04): the zhangtaolab NT-family remote
# modeling_esm.py reads config.is_decoder / config.add_cross_attention --
# transformers-4.x PretrainedConfig defaults removed in 5.x and absent from
# the checkpoint's config.json.  plant-nucleotide-transformer-BPE (this
# script's tokenizer model) carries the same remote code (failing live at
# remote modeling_esm.py:335 after the 05-04 import shim let it load).
# Owner disposition pending (05-04 hand-off options); per D-08 the census
# terminal state for this item is an evidence-backed typed skip.  The skip
# is a live probe, not a dead skip: once the environment gap closes the
# script runs to completion and this test goes green through the real
# artifact assertions below.
_NT_STRUCTURAL_MARKER = "Failed to load model: 'EsmConfig' object has no attribute 'is_decoder'"

# Documented input-download URLs from the sibling notebook
# data_generation_and_inference.ipynb cell 9 (provenance pinned by the
# example tree).
RICE_INPUT_URLS: dict[str, str] = {
    "osa1_r7.asm.fa.gz": "https://rice.uga.edu/osa1r7_download/osa1_r7.asm.fa.gz",
    "osa1_r7.all_models.gff3.gz": (
        "https://rice.uga.edu/osa1r7_download/osa1_r7.all_models.gff3.gz"
    ),
}

# 05-06 census input cache (repo-root .scratch/, gitignored): rice.uga.edu
# flaps for hours at a time (2026-10-04: served the notebook lane at 18:11,
# SSL-EOF by 18:57, unreachable through a 48-minute probe window, then
# half-up at ~8KB/s). The cache is a self-refreshing mirror consulted
# FIRST; a cold cache downloads from the documented URLs and populates it
# (per-run provenance for CI/fresh checkouts). A cold cache plus an outage
# keeps the honest network-unavailable skip; 4xx never reaches the cache
# path at all (WR-04).
RICE_CACHE_DIR = EXAMPLE_DIR.parent / ".scratch" / "census-out" / "inputs"


def _download(url: str, dest: Path, timeout_s: int = 600) -> None:
    """Stream *url* to the sandbox-local *dest* (never the repo tree)."""
    # ruff: ignore[suspicious-url-open-usage]  # callers pass hardcoded https constants only
    with urllib.request.urlopen(url, timeout=timeout_s) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def _refresh_cache(name: str, sandbox: Path) -> None:
    """Copy a freshly downloaded input into the census cache (best-effort)."""
    try:
        RICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sandbox / name, RICE_CACHE_DIR / name)
    except OSError:
        return  # cache warming is best-effort; it must never fail the lane


def _seed_rice_input(name: str, url: str, sandbox: Path) -> None:
    """Seed a rice input into the sandbox: census cache first, else download.

    The cache is a self-refreshing mirror of the documented URLs (a cold
    cache downloads and then populates it), so a hit is byte-identical to
    what the URL serves; CI/fresh checkouts keep per-run download
    provenance, while the dev box never re-fetches 115MB from a host that
    flaps for hours and crawls at ~8KB/s when half-up (2026-10-04).

    Args:
        name: sandbox-local filename (also the cache key).
        url: documented upstream URL (the source of truth for cold caches).
        sandbox: the seeded tmp sandbox root.

    Raises:
        urllib.error.HTTPError: a permanent 4xx (reorganized URLs, gone
            dataset) -- never converted to a skip or a cache hit (WR-04).
    """
    cached = RICE_CACHE_DIR / name
    if cached.is_file() and cached.stat().st_size > 0:
        shutil.copy2(cached, sandbox / name)
        return
    try:
        _download(url, sandbox / name)
    except urllib.error.HTTPError as exc:
        # A permanent 4xx (reorganized URLs, gone dataset) is NOT
        # network unavailability: skipping would convert a broken input
        # contract into an ever-green typed skip (05 review WR-04).
        # Only server-side 5xx joins the skip path; 4xx re-raises.
        if exc.code < 500:
            raise
        outage = f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        outage = f"{type(exc).__name__}: {exc}"
    else:
        _refresh_cache(name, sandbox)
        return
    pytest.skip(
        f"network-unavailable: fetch {url} for generate_bpe_dataset.py "
        f"({outage}); census cache cold at {cached}"
    )


@pytest.mark.slow
@pytest.mark.timeout(3600)
class TestExampleScriptExecution:
    """Real example-script execution through the private harness."""

    def test_generate_bpe_dataset_produces_artifact(self, tmp_path: Path) -> None:
        """Run generate_bpe_dataset.py in-sandbox and prove a fresh pkl results."""
        sandbox = seed_sandbox(SCRIPT_DIR, tmp_path)
        for name, url in RICE_INPUT_URLS.items():
            _seed_rice_input(name, url, sandbox)
        out_pkl = sandbox / "rice_gene_ner_BPE.pkl"  # seeded copy must be rewritten
        run_start = time.time()
        try:
            run_example_script(SCRIPT, sandbox, artifact_dir=tmp_path / "artifacts")
        except AssertionError as exc:
            # Ladder terminal (D-06): only the documented structural marker
            # converts to the typed skip; every other failure stays loud.
            if _NT_STRUCTURAL_MARKER in str(exc):
                environment_unavailable_skip(
                    "execute generate_bpe_dataset.py (zhangtaolab/"
                    "plant-nucleotide-transformer-BPE remote modeling_esm.py needs "
                    "removed transformers-4.x PretrainedConfig defaults)",
                    evidence=str(exc).strip().splitlines()[-1],
                )
            raise
        assert out_pkl.is_file(), f"expected artifact missing: {out_pkl}"
        assert out_pkl.stat().st_size > 0, "artifact is empty"
        assert out_pkl.stat().st_mtime >= run_start, "artifact was not freshly rewritten"
        assert_tree_clean()


class TestRiceInputSeeding:
    """Fast (network-free) contract for :func:`_seed_rice_input` (08-02).

    Proves the three fallback semantics without touching the network:
    4xx re-raises even with a warm cache (WR-04); an outage with a warm
    cache seeds from the census cache instead of skipping; an outage with
    a cold cache typed-skips with the network-unavailable prefix.
    """

    URL = "https://rice.uga.edu/osa1r7_download/osa1_r7.asm.fa.gz"

    @staticmethod
    def _patch_module(
        monkeypatch: pytest.MonkeyPatch, cache_dir: Path, download_exc: Exception | None
    ) -> None:
        """Patch the RUNNING module object (dotted targets import a second copy)."""
        module = sys.modules[__name__]
        monkeypatch.setattr(module, "RICE_CACHE_DIR", cache_dir)
        if download_exc is not None:

            def fake_download(url: str, dest: Path, timeout_s: int = 600) -> None:
                raise download_exc

            monkeypatch.setattr(module, "_download", fake_download)

    def test_http_4xx_raises_on_cold_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A permanent 4xx on the download path re-raises loudly (WR-04).

        Cache-first means a warm cache never asks the server, so the 4xx
        path is reachable exactly when the cache is cold (CI/fresh
        checkouts) -- and there it must never convert to a skip.
        """
        self._patch_module(
            monkeypatch, tmp_path / "cold", urllib.error.HTTPError(self.URL, 404, "Gone", {}, None)
        )
        with pytest.raises(urllib.error.HTTPError):
            _seed_rice_input("osa1_r7.asm.fa.gz", self.URL, tmp_path)
        assert not (tmp_path / "osa1_r7.asm.fa.gz").exists()

    def test_outage_with_cold_cache_skips_typed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """URLError + no cached copy -> network-unavailable typed skip."""
        self._patch_module(monkeypatch, tmp_path / "cold", TimeoutError("timed out"))
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _seed_rice_input("osa1_r7.asm.fa.gz", self.URL, tmp_path)
        message = str(excinfo.value.args[0])
        assert message.startswith("network-unavailable:"), message
        assert "generate_bpe_dataset.py" in message, message

    def test_warm_cache_short_circuits_without_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A cached copy seeds the sandbox directly; the URL is never touched."""

        def forbidden_download(url: str, dest: Path, timeout_s: int = 600) -> None:
            raise AssertionError("download must not run when the cache is warm")

        cache = tmp_path / "inputs"
        cache.mkdir()
        (cache / "osa1_r7.asm.fa.gz").write_bytes(b"mirrored-bytes")
        self._patch_module(monkeypatch, cache, None)
        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_download", forbidden_download)
        _seed_rice_input("osa1_r7.asm.fa.gz", self.URL, tmp_path)
        assert (tmp_path / "osa1_r7.asm.fa.gz").read_bytes() == b"mirrored-bytes"
