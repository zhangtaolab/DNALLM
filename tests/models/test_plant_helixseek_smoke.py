"""Slow-leg PlantHelixSeek smoke tests: real loads through the generic dnallm route.

Both tests download the real checkpoints (~1.9 GB each) and load them through
``load_model_and_tokenizer`` with the ModelScope source first (the locked sourcing
decision), asserting the frozen label order post-load — the REG-02
silent-permutation guard: the assert target is the hard-coded upstream constant from
``tests.models.test_plant_helixseek_registry``, never the yaml list that was passed
into the load, so a permuted registry entry fails here.

Fallback contract (CONTEXT): on ModelScope failure each load retries once with
``source="huggingface"``; if both routes fail the test skips with the registered
``environment-unavailable:`` prefix carrying version + exception evidence. Before any
such skip is accepted, the documented response procedure is one manual
transformers-4.57 venv attempt (Phase-5 /tmp/feas-venv precedent) recorded as
evidence. Do not assert on stderr or captured warnings — the runs emit benign
HelixSeek cache and use_return_dict deprecation warnings (research Pattern 3).
"""

import sys
from pathlib import Path

import dnallm
import pytest
import torch
import transformers
import yaml

from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer
from tests.models.test_plant_helixseek_registry import ANNO_LABELS, CRE_LABELS

CRE_REPO_ID = "zhangtaolab/PlantHelixSeek-CRE"
ANNO_REPO_ID = "zhangtaolab/PlantHelixSeek-Anno"
REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml"


def _emit_env() -> None:
    """Emit the REG-03 compat-gate evidence lines (transformers/torch versions)."""
    sys.stdout.write(f"transformers_version={transformers.__version__}\n")
    sys.stdout.write(f"torch_version={torch.__version__}\n")


def _registry_task(repo_id: str) -> dict:
    """Read this model's finetuned entry from the packaged registry (frozen source)."""
    content = REGISTRY_PATH.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML error in {REGISTRY_PATH.name}: {e}")
    entries = [e for e in data["finetuned"] if e.get("model") == repo_id]
    assert len(entries) == 1, f"expected exactly one registry entry for {repo_id}"
    return entries[0]["task"]


def _load_with_fallback(repo_id: str, cfg: TaskConfig):
    """Load via ModelScope first, retry once on HuggingFace, else typed-skip.

    Args:
        repo_id: Owner-org checkpoint repo id (never a floating third-party id).
        cfg: TaskConfig built from the committed registry entry.

    Returns:
        The ``(model, tokenizer)`` pair from ``load_model_and_tokenizer``.
    """
    errors = []
    for source in ("modelscope", "huggingface"):
        try:
            return load_model_and_tokenizer(repo_id, cfg, source=source)
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    pytest.skip(
        "environment-unavailable: PlantHelixSeek checkpoint load failed on both the "
        f"ModelScope and HuggingFace routes (transformers {transformers.__version__}, "
        f"torch {torch.__version__}; {' | '.join(errors)})"
    )


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_planthelixseek_cre_smoke_load():
    """CRE loads via the generic route; id2label frozen order and (1, 2) forward."""
    # WR-01 guard: without flash-linear-attention the load runs the remote
    # code's silent pure-PyTorch non-KDA fallback (positionally dead outputs),
    # so a shape-only green here would validate the wrong kernel path.
    pytest.importorskip(
        "fla",
        reason="environment-unavailable: flash-linear-attention not installed — "
        "the PlantHelixSeek load would run the degraded non-KDA fallback "
        "(WR-01 typed skip; see tests/models/test_plant_helixseek_fla_kernels.py)",
    )
    _emit_env()
    task = _registry_task(CRE_REPO_ID)
    cfg = TaskConfig(
        task_type=task["task_type"],
        num_labels=task["num_labels"],
        label_names=task["label_names"],
        threshold=task["threshold"],
    )
    model, tokenizer = _load_with_fallback(CRE_REPO_ID, cfg)
    assert model.config.num_labels == 2
    # Assert against the frozen constant, not the yaml list fed into the load.
    assert model.config.id2label == {0: CRE_LABELS[0], 1: CRE_LABELS[1]}
    seq = "ACGT" * 125  # 500 bp, uppercase (the tokenizer vocab is ACGTN-only)
    enc = tokenizer([seq], return_tensors="pt", padding=True)
    # no_grad: the research memory budget (0.59 s / 3.83 GB peak) was measured on a
    # no-grad forward; an autograd graph over the 8192-token Anno window OOMs the box.
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(model.device),
            attention_mask=enc["attention_mask"].to(model.device),
        )
    assert tuple(out.logits.shape) == (1, 2)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_planthelixseek_anno_smoke_load():
    """Anno loads via the generic route; frozen 17-BILOU order and forward shape."""
    # WR-01 guard: same rationale as the CRE smoke — without fla the load
    # validates the degraded non-KDA fallback, not the checkpoint's real path.
    pytest.importorskip(
        "fla",
        reason="environment-unavailable: flash-linear-attention not installed — "
        "the PlantHelixSeek load would run the degraded non-KDA fallback "
        "(WR-01 typed skip; see tests/models/test_plant_helixseek_fla_kernels.py)",
    )
    _emit_env()
    task = _registry_task(ANNO_REPO_ID)
    cfg = TaskConfig(
        task_type=task["task_type"],
        num_labels=task["num_labels"],
        label_names=task["label_names"],
        threshold=task["threshold"],
    )
    model, tokenizer = _load_with_fallback(ANNO_REPO_ID, cfg)
    assert model.config.num_labels == 17
    # Assert against the frozen upstream constant, not the yaml list fed into the load.
    assert model.config.id2label == dict(enumerate(ANNO_LABELS))
    assert model.config.id2label[1] == "B-CDS"
    seq = "ACGT" * 2048  # 8192 bp inference window
    enc = tokenizer([seq], return_tensors="pt", padding=True)
    # no_grad: the research memory budget (7.9 s / 12.99 GB peak) was measured on a
    # no-grad forward; an autograd graph over this eager-attention window OOMs the box.
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(model.device),
            attention_mask=enc["attention_mask"].to(model.device),
        )
    # Logit width is window + 2: the checkpoint emits BOS/EOS positions (measured).
    assert tuple(out.logits.shape) == (1, 8194, 17)
