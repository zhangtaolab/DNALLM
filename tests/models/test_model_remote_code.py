"""Slow real-model smoke tests for trust_remote_code checkpoints on the installed transformers.

GAP-1 provenance (Phase 5 post-closure reopen, decision D-07): the registry checkpoint
``zhangtaolab/nucleotide-transformer-v2-100m-promoter`` (``dnallm/models/model_info.yaml``,
third model of ``example/notebooks/benchmark/benchmark_config.yaml``) ships a
transformers-4.x-era remote-code ``modeling_esm.py`` whose
``from transformers.modeling_utils import ...`` of the pruning helpers crashed on
transformers 5.x, where both symbols were removed.
``dnallm.utils.transformers_compat`` vendors and re-attaches the helpers; these tests
prove the repair end to end with a REAL load plus forward pass — import-only proof is
insufficient per D-07 (deeper 5.x breakage inside the remote module is a live risk
until a forward actually passes).
"""

import pytest
import torch

from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer


@pytest.mark.slow
@pytest.mark.timeout(1800)
class TestRemoteCodeCheckpointCompat:
    """Real-model load+forward smokes for remote-code checkpoints on the dev environment."""

    def test_nt_v2_promoter_loads_and_forwards(self) -> None:
        """The NT v2 100m promoter checkpoint loads and forwards with (1, 2) logits.

        Loads ``zhangtaolab/nucleotide-transformer-v2-100m-promoter`` through
        ``load_model_and_tokenizer`` with ``source="modelscope"`` (the exact route
        ``benchmark_config.yaml`` uses) for a binary promoter task, then runs a
        real forward pass on a tokenized 64 nt sequence.

        Raises:
            AssertionError: If loading fails, the forward logits are not shaped
                ``(1, 2)``, or the model config does not report 2 labels.
        """
        task_config = TaskConfig(
            task_type="binary",
            num_labels=2,
            label_names=["Not promoter", "Core promoter"],
        )

        model, tokenizer = load_model_and_tokenizer(
            "zhangtaolab/nucleotide-transformer-v2-100m-promoter",
            task_config,
            source="modelscope",
        )

        sequence = "ACGT" * 16  # 64 nt
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        assert tuple(logits.shape) == (1, 2), f"expected (1, 2) logits, got {tuple(logits.shape)}"
        assert model.config.num_labels == 2, (
            f"expected config.num_labels == 2, got {model.config.num_labels}"
        )
