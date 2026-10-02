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


def _download(url: str, dest: Path, timeout_s: int = 600) -> None:
    """Stream *url* to the sandbox-local *dest* (never the repo tree)."""
    # ruff: ignore[suspicious-url-open-usage]  # callers pass hardcoded https constants only
    with urllib.request.urlopen(url, timeout=timeout_s) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


@pytest.mark.slow
@pytest.mark.timeout(3600)
class TestExampleScriptExecution:
    """Real example-script execution through the private harness."""

    def test_generate_bpe_dataset_produces_artifact(self, tmp_path: Path) -> None:
        """Run generate_bpe_dataset.py in-sandbox and prove a fresh pkl results."""
        sandbox = seed_sandbox(SCRIPT_DIR, tmp_path)
        for name, url in RICE_INPUT_URLS.items():
            try:
                _download(url, sandbox / name)
            except (urllib.error.URLError, TimeoutError) as exc:
                pytest.skip(
                    f"network-unavailable: fetch {url} for generate_bpe_dataset.py "
                    f"({type(exc).__name__}: {exc})"
                )
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
