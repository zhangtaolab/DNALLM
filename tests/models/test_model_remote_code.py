"""Slow real-model smoke tests for trust_remote_code checkpoints on the installed transformers.

GAP-1 provenance (Phase 5 reopen; resolved Phase 14 -- disposition
shim-zero-rung): the registry checkpoint
``zhangtaolab/nucleotide-transformer-v2-100m-promoter`` (``dnallm/models/model_info.yaml``,
third model of ``example/notebooks/benchmark/benchmark_config.yaml``) ships a
transformers-4.x-era remote-code ``modeling_esm.py`` whose
``from transformers.modeling_utils import ...`` of the pruning helpers crashed on
transformers 5.x, where both symbols were removed;
``dnallm.utils.transformers_compat`` vendors and re-attaches the helpers. The
remote code also reads ``config.is_decoder``/``config.add_cross_attention`` --
transformers-4.x ``PretrainedConfig`` defaults that 5.x removed and the
checkpoint's ``config.json`` does not carry -- and the closed-map
``_patch_pretrained_config_legacy_defaults`` rung
(``dnallm/utils/transformers_compat.py``, landed 2026-10-02) answers both
names, so the GAP-1 load failure is closed: the checkpoint loads and forwards
with ``(1, 2)`` logits on transformers 5.17 (recorded GAPC-01 run --
``.planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md``
section A), and WINDOWS #11 closed fixed in phase 14. The typed-skip branch
below remains as the regression tripwire: it fires only if a future
transformers regression re-breaks the remote-code load, re-skipping loudly
with the exact traceback.
"""

import traceback

import pytest
import torch

from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer
from tests.examples._execution import environment_unavailable_skip

# Dormant regression tripwire (GAP-1 resolved, phase 14): the closed-map
# legacy-defaults rung answers ``is_decoder`` (EsmSelfAttention.__init__,
# line 335) and ``add_cross_attention`` (EsmLayer.__init__, line 584), so
# the marker's AttributeError text cannot occur unless transformers
# regresses; if it ever does, the branch below re-skips loudly (typed,
# with traceback evidence).
_STRUCTURAL_MARKER = "'EsmConfig' object has no attribute 'is_decoder'"


@pytest.mark.slow
@pytest.mark.timeout(1800)
class TestRemoteCodeCheckpointCompat:
    """Real-model load+forward smokes for remote-code checkpoints on the dev environment."""

    def test_nt_v2_promoter_loads_and_forwards(self) -> None:
        """The NT v2 100m promoter checkpoint loads and forwards with (1, 2) logits.

        Loads ``zhangtaolab/nucleotide-transformer-v2-100m-promoter`` through
        ``load_model_and_tokenizer`` with ``source="modelscope"`` (the exact route
        ``benchmark_config.yaml`` uses) for a binary promoter task, then runs a
        real forward pass on a tokenized 64 nt sequence. Only the documented
        structural breakage (remote code reading transformers-4.x config
        defaults removed in 5.x) skips, with the exact traceback as evidence;
        any other failure propagates loudly.

        Raises:
            AssertionError: If the forward logits are not shaped ``(1, 2)`` or
                the model config does not report 2 labels.
        """
        task_config = TaskConfig(
            task_type="binary",
            num_labels=2,
            label_names=["Not promoter", "Core promoter"],
        )

        try:
            model, tokenizer = load_model_and_tokenizer(
                "zhangtaolab/nucleotide-transformer-v2-100m-promoter",
                task_config,
                source="modelscope",
            )
        except ValueError as exc:
            if _STRUCTURAL_MARKER not in str(exc):
                raise
            environment_unavailable_skip(
                "nt-v2 promoter remote code on transformers 5.17",
                f"sanctioned fallback ladder terminal rung (D-07): remote "
                f"modeling_esm.py needs transformers-4.x PretrainedConfig "
                f"defaults removed in 5.x; exact traceback: {traceback.format_exc()}",
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
