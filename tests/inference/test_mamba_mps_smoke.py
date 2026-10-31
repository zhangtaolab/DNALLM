"""Slow-leg plant-dnamamba MPS smoke: the Predictor keeps mps for mamba.

Loads the real ``zhangtaolab/plant-dnamamba-BPE`` checkpoint through
``load_model_and_tokenizer`` and constructs ``DNAInference`` (the Predictor
route) with device "mps" — the route whose mamba guard historically demoted
MPS to CPU (relaxed 2026-10-24 after measured parity: 512-token forward
349 ms MPS vs 2119 ms CPU, bitwise-identical, torch 2.14.1) — then runs one
real forward on MPS. The measured green load via the no-guard
``load_model_and_tokenizer`` route predates this test; this smoke pins the
Predictor route specifically.

Fallback contract (mirrors tests/models/test_plant_helixseek_smoke.py): on
ModelScope failure the load retries once with ``source="huggingface"``; if
both routes fail with environment-class errors (network/hub outage, missing
repo, absent optional dependency, classified by ``_is_environment_error``
imported from the HelixSeek smoke so the two cannot drift) the test skips
with the registered ``environment-unavailable:`` prefix carrying version +
exception evidence; a dnallm-side load regression propagates and fails the
test instead of surfacing as a whitelisted green skip. The static darwin
gate covers non-darwin collection (registered as a reason_like entry in
tests/expected_skips.yaml; the example-nightly full census collects it, CI
fast legs deselect slow); the runtime MPS gate covers darwin hosts without
a usable MPS device (same ``environment-unavailable:`` prefix).
"""

import sys

import pytest
import torch
import transformers

from dnallm.configuration.configs import TaskConfig
from dnallm.inference.inference import DNAInference
from dnallm.models.model import load_model_and_tokenizer
from tests.models.test_plant_helixseek_smoke import _is_environment_error

REPO_ID = "zhangtaolab/plant-dnamamba-BPE"


def _emit_env() -> None:
    """Emit the compat-gate evidence lines (transformers/torch versions)."""
    sys.stdout.write(f"transformers_version={transformers.__version__}\n")
    sys.stdout.write(f"torch_version={torch.__version__}\n")


def _load_with_fallback(repo_id: str, cfg: TaskConfig):
    """Load via ModelScope first, retry once on HuggingFace, else typed-skip.

    Only environment-class failures (per ``_is_environment_error``, imported
    from tests.models.test_plant_helixseek_smoke) record evidence and continue
    to the next source; a dnallm-side load regression propagates and fails the
    test instead of whitelisting as a green skip.

    Args:
        repo_id: Owner-org checkpoint repo id (never a floating third-party id).
        cfg: TaskConfig for the load (generation task type for this model).

    Returns:
        The ``(model, tokenizer)`` pair from ``load_model_and_tokenizer``.
    """
    errors = []
    for source in ("modelscope", "huggingface"):
        try:
            return load_model_and_tokenizer(repo_id, cfg, source=source)
        except Exception as exc:
            if not _is_environment_error(exc):
                raise
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    pytest.skip(
        "environment-unavailable: plant-dnamamba checkpoint load failed on both the "
        f"ModelScope and HuggingFace routes (transformers {transformers.__version__}, "
        f"torch {torch.__version__}; {' | '.join(errors)})"
    )


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MPS mamba smoke is darwin-only (Apple Silicon real device)",
)
def test_plant_dnamamba_mps_predictor_keeps_device(inference_config_factory):
    """plant-dnamamba-BPE through the Predictor keeps device mps and runs a forward."""
    _emit_env()
    if not torch.backends.mps.is_available():
        pytest.skip(
            "environment-unavailable: MPS not available on this darwin host "
            f"(transformers {transformers.__version__}, torch {torch.__version__})"
        )
    model, tokenizer = _load_with_fallback(REPO_ID, TaskConfig(task_type="generation"))
    # use_fp16=True deliberately: pins the conservative force-off on real hardware.
    config = inference_config_factory(
        task_type="generation", device="mps", use_fp16=True, max_length=512
    )
    engine = DNAInference(model=model, tokenizer=tokenizer, config=config)

    # The regression this smoke exists to pin: the relaxed mamba guard keeps mps...
    assert engine.device.type == "mps"
    # ...and the conservative fp16 force-off still fires on real hardware.
    assert engine.pred_config.use_fp16 is False
    # The constructor moved the real parameters to MPS before the guard ran.
    assert next(engine.model.parameters()).device.type == "mps"

    # One real forward on MPS. The generation task_type makes logits_to_preds
    # raise (unsupported task type), so do_pred=True is not viable; infer_seqs
    # with do_pred=False returns None. The non-empty forward evidence is
    # batch_infer's logits return — the forward half of infer_seqs. Logits are
    # moved to cpu by _process_batch_outputs, so the MPS pin lives on the
    # parameter assertion above.
    _, dataloader = engine.generate_dataset(["ACGT" * 32], batch_size=engine.pred_config.batch_size)
    logits, predictions, _ = engine.batch_infer(dataloader, do_pred=False)
    assert predictions is None
    assert logits is not None
    assert logits.numel() > 0
