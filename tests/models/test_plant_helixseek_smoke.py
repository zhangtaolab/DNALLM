"""Slow-leg PlantHelixSeek smoke tests: real loads through the generic dnallm route.

Both tests download the real checkpoints (~1.9 GB each) and load them through
``load_model_and_tokenizer`` with the ModelScope source first (the locked sourcing
decision), asserting the frozen label order post-load — the REG-02
silent-permutation guard: the assert target is the hard-coded upstream constant from
``tests.models.test_plant_helixseek_registry``, never the yaml list that was passed
into the load, so a permuted registry entry fails here.

Fallback contract (CONTEXT): on ModelScope failure each load retries once with
``source="huggingface"``; if both routes fail with environment-class errors
(network/hub outage, missing repo, absent optional dependency, classified by
``_is_environment_error``) the test skips with the registered
``environment-unavailable:`` prefix carrying version + exception evidence; a
dnallm-side load regression propagates and fails the test instead of surfacing
as a whitelisted green skip (WR-02). Before any
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


def _is_environment_error(exc: BaseException) -> bool:
    """Classify a load exception as environment-caused or a dnallm regression.

    Rules, derived from the exception ladder in ``dnallm/models/model.py``:

    - ``ConnectionError`` / ``TimeoutError`` / ``OSError`` anywhere in the
      ``__cause__``/``__context__`` chain is environmental: requests'
      ``RequestException``, huggingface_hub HTTP errors, and socket errors all
      subclass ``OSError``, and the load block wraps everything as
      ``ValueError(f"Failed to load model: {e}") from e`` (model.py:887-888),
      so hub/network causes are visible only in the chain.
    - ``ImportError`` anywhere in the chain is environmental (the
      modelscope/transformers guards at model.py:444-448 and 476-480, or
      remote code importing an absent optional dependency such as fla).
    - The bare unchained ``ValueError(f"Model {name} download failed.")`` from
      ``download_model`` (model.py:375) is environmental: it is the single
      terminal signal covering network failures, hub outages, and missing
      repos on both sources, and it is raised outside the boundary wrap (the
      ``_get_model_path_and_imports`` call at model.py:834), so it arrives at
      the caller unchained and must be recognized by message shape.

    Anything else — a ``TypeError``/``AttributeError``/``KeyError`` from
    dnallm's dispatch or config plumbing, a boundary ``ValueError`` chained
    from a non-network cause, a CUDA-OOM ``RuntimeError`` — is NOT
    environmental and must propagate so the test fails with the real
    traceback.

    Args:
        exc: The exception raised by ``load_model_and_tokenizer``.

    Returns:
        bool: True when the failure is environment-class (the typed skip is
        legitimate); False when it is a dnallm-side regression.
    """
    environmental = (ConnectionError, TimeoutError, OSError, ImportError)
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, environmental):
            return True
        if (
            isinstance(node, ValueError)
            and str(node).startswith("Model ")
            and str(node).endswith(" download failed.")
        ):
            return True
        node = node.__cause__ or node.__context__
    return False


def _load_with_fallback(repo_id: str, cfg: TaskConfig):
    """Load via ModelScope first, retry once on HuggingFace, else typed-skip.

    Only environment-class failures (per :func:`_is_environment_error`) record
    evidence and continue to the next source; a dnallm-side load regression
    propagates and fails the test instead of whitelisting as a green skip.

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
            if not _is_environment_error(exc):
                raise
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    pytest.skip(
        "environment-unavailable: PlantHelixSeek checkpoint load failed on both the "
        f"ModelScope and HuggingFace routes (transformers {transformers.__version__}, "
        f"torch {torch.__version__}; {' | '.join(errors)})"
    )


class TestLoadWithFallbackClassification:
    """WR-02: only environment-class failures may produce the typed skip.

    A dnallm regression inside ``load_model_and_tokenizer`` (a TypeError from
    the dispatch chain, config plumbing KeyError, ...) must FAIL the test,
    never surface as a whitelisted ``environment-unavailable:`` skip. All
    tests here are fast: no network, no model downloads, not slow-marked.
    """

    def test_network_and_import_classes_are_environmental(self):
        for exc in (
            ConnectionError("connection refused"),
            TimeoutError("timed out"),
            OSError("socket error"),
            ImportError("fla kernels absent"),
        ):
            assert _is_environment_error(exc), type(exc).__name__

    def test_terminal_download_value_error_is_environmental(self):
        # download_model's sole terminal signal (model.py:375), raised unchained
        assert _is_environment_error(ValueError(f"Model {CRE_REPO_ID} download failed."))

    def test_wrapped_value_error_from_connection_error_is_environmental(self):
        # Boundary wrap (model.py:887-888): the network cause lives in the chain.
        # Setting __cause__ directly is exactly what ``raise ... from ...`` does.
        wrapped = ValueError("Failed to load model: hub unreachable")
        wrapped.__cause__ = ConnectionError("hub unreachable")
        assert _is_environment_error(wrapped)

    def test_bare_type_error_is_not_environmental(self):
        assert not _is_environment_error(TypeError("dispatch bug"))

    def test_wrapped_value_error_from_type_error_is_not_environmental(self):
        wrapped = ValueError("Failed to load model: unexpected keyword argument")
        wrapped.__cause__ = TypeError("unexpected keyword argument")
        assert not _is_environment_error(wrapped)

    def test_dnallm_regression_propagates_instead_of_skipping(self, monkeypatch):
        def _regression(repo_id, cfg, source):
            try:
                raise TypeError("unexpected keyword argument")
            except TypeError as cause:
                raise ValueError(f"Failed to load model: {cause}") from cause

        # Patch the module object that actually owns _load_with_fallback: the
        # repo's tests/ tree has no __init__.py, so under pytest's prepend
        # import mode a dotted-string target resolves to a second, lazily
        # created namespace-package module and the patch silently misses.
        smoke = sys.modules[_load_with_fallback.__module__]
        monkeypatch.setattr(smoke, "load_model_and_tokenizer", _regression)
        cfg = TaskConfig(task_type="binary", num_labels=2)
        with pytest.raises(ValueError, match="Failed to load model"):
            _load_with_fallback(CRE_REPO_ID, cfg)

    def test_environment_failure_on_both_routes_skips_typed(self, monkeypatch):
        def _network_down(repo_id, cfg, source):
            try:
                raise ConnectionError("hub unreachable")
            except ConnectionError as cause:
                raise ValueError(f"Failed to load model: {cause}") from cause

        smoke = sys.modules[_load_with_fallback.__module__]
        monkeypatch.setattr(smoke, "load_model_and_tokenizer", _network_down)
        cfg = TaskConfig(task_type="binary", num_labels=2)
        with pytest.raises(pytest.skip.Exception, match=r"environment-unavailable: PlantHelixSeek"):
            _load_with_fallback(CRE_REPO_ID, cfg)


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
