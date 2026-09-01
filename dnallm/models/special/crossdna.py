"""
CrossDNA support for DNALLM sequence classification.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import SequenceClassifierOutput

from ...utils import get_logger


logger = get_logger("dnallm.models.model")

_SEQUENCE_TASKS = {"binary", "multiclass", "multilabel", "regression"}
_PREFERRED_CHECKPOINT_DIRS = ("8.1M",)
_CLASS_CACHE: dict[tuple[type, type], type] = {}


def _read_crossdna_config(path: str) -> dict[str, Any] | None:
    """Return config.json when *path* looks like a CrossDNA checkpoint."""
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    model_type = str(config.get("model_type", "")).lower()
    auto_map = config.get("auto_map") or {}
    masked_lm_ref = str(auto_map.get("AutoModelForMaskedLM", "")).lower()

    if model_type == "crossdna" or "crossdna" in masked_lm_ref:
        return config
    return None


def _resolve_crossdna_checkpoint_dir(model_path: str) -> str | None:
    """Find the actual CrossDNA checkpoint directory inside *model_path*.

    Resolution order:
      1. ``model_path`` itself;
      2. known checkpoint subdirectories (currently ``8.1M``);
      3. exactly one immediate child directory containing a CrossDNA config.

    If multiple unknown CrossDNA checkpoint directories are found, fail loudly
    instead of silently selecting the wrong model size.
    """
    model_path = os.path.abspath(os.path.expanduser(model_path))

    if _read_crossdna_config(model_path) is not None:
        return model_path

    for subdir in _PREFERRED_CHECKPOINT_DIRS:
        candidate = os.path.join(model_path, subdir)
        if _read_crossdna_config(candidate) is not None:
            return candidate

    if not os.path.isdir(model_path):
        return None

    candidates: list[str] = []
    try:
        entries = os.listdir(model_path)
    except OSError:
        return None

    for entry in entries:
        candidate = os.path.join(model_path, entry)
        if os.path.isdir(candidate) and _read_crossdna_config(candidate) is not None:
            candidates.append(candidate)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    names = ", ".join(sorted(os.path.basename(path) for path in candidates))
    raise ValueError(
        "Multiple CrossDNA checkpoint directories were found: "
        f"{names}. Please pass a local path pointing directly to the desired "
        "checkpoint directory."
    )


def _build_crossdna_sequence_classification_class(
    base_mlm_class: type,
    config_class: type,
) -> type:
    """Build and cache a classifier subclass on the real upstream MLM class."""
    cache_key = (base_mlm_class, config_class)
    cached = _CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    class CrossDNAForSequenceClassification(base_mlm_class):
        """CrossDNA backbone with a sequence-level classification head.

        The class subclasses the *actual* upstream ``CrossDNAForMaskedLM`` class
        loaded from the checkpoint's remote code.  Therefore the original
        ``backbone.*`` parameter names are preserved and the pretrained weights
        can be loaded directly without modifying the upstream model files.
        """

        base_model_prefix = "backbone"

        # CrossDNA pretraining checkpoints may contain EMA-teacher copies.
        # Downstream classification intentionally does not keep these modules.
        _keys_to_ignore_on_load_unexpected = [  # noqa: RUF012
            r"backbone\.branchA_core_ema\..*",
            r"backbone\.branchB_core_ema\..*",
            r"backbone\.bridge_ema\..*",
        ]

        def __init__(self, config: Any):
            # Mutate only the in-memory config used by this downstream model.
            # The upstream config.json on disk / Hub remains untouched.
            config.pretrain = False
            config.for_representation = True
            config.architectures = ["CrossDNAForSequenceClassification"]

            # Build the exact original CrossDNA backbone/model implementation.
            super().__init__(config)

            self.config = config
            self.num_labels = int(getattr(config, "num_labels", 2))
            if self.num_labels < 1:
                raise ValueError(f"num_labels must be >= 1, got {self.num_labels}")

            # Downstream mode: consume the fused token representation directly.
            self.backbone.pretrain = False
            self.backbone.for_representation = True

            # Do NOT reuse CrossDNA's pretraining gate warm-up (5000 steps in
            # the 8.1M config).  Fine-tuning should use the pretrained gate from
            # the first optimization step unless explicitly overridden.
            classifier_gate_freeze_steps = int(getattr(config, "classifier_gate_freeze_steps", 0))
            classifier_detach_gate = bool(getattr(config, "classifier_detach_gate", False))
            self.backbone.gate_freeze_steps = classifier_gate_freeze_steps
            self.backbone.detach_gate = classifier_detach_gate

            config.classifier_gate_freeze_steps = classifier_gate_freeze_steps
            config.classifier_detach_gate = classifier_detach_gate

            # EMA teachers and all pretraining-only auxiliary losses are not
            # needed for sequence classification.
            self.backbone.use_ema_teacher = False
            self.backbone.auto_update_ema_in_forward = False
            self.backbone.use_rc_kl = False
            self.backbone.use_barlow = False
            self.backbone.use_tv = False

            for name in ("branchA_core_ema", "branchB_core_ema", "bridge_ema"):
                if hasattr(self.backbone, name):
                    delattr(self.backbone, name)

            classifier_dropout = getattr(config, "classifier_dropout", None)
            if classifier_dropout is None:
                classifier_dropout = getattr(config, "dropout", 0.1)
            classifier_dropout = float(classifier_dropout)  # type: ignore
            if not 0.0 <= classifier_dropout <= 1.0:
                raise ValueError(f"classifier_dropout must be in [0, 1], got {classifier_dropout}")

            self.classifier_dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Linear(int(config.d_model), self.num_labels)

            self.pooling = str(getattr(config, "classifier_pooling", "mean")).lower()
            if self.pooling not in {"mean", "max", "first", "last"}:
                raise ValueError(
                    "classifier_pooling must be one of: mean, max, first, last; "
                    f"got {self.pooling!r}"
                )

            # The currently distributed CrossDNA tokenizer encodes A/C/G/T/N
            # as 7/8/9/10/11, while the backbone expects 0/1/2/3/4.
            self.auto_remap_tokenizer_ids = bool(getattr(config, "auto_remap_tokenizer_ids", True))
            self.tokenizer_base_offset = int(getattr(config, "tokenizer_base_offset", 7))

            config.classifier_dropout = classifier_dropout
            config.classifier_pooling = self.pooling
            config.auto_remap_tokenizer_ids = self.auto_remap_tokenizer_ids
            config.tokenizer_base_offset = self.tokenizer_base_offset

            # The parent already called post_init() for the upstream modules.
            # Calling it again initializes only newly-added HF modules under
            # normal Transformers initialization semantics.
            self.post_init()

        def _prepare_input_ids(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if input_ids is None:
                raise ValueError("input_ids must be provided")
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must have shape [batch, length], got {tuple(input_ids.shape)}"
                )
            if input_ids.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError(f"input_ids must contain integer token IDs, got {input_ids.dtype}")

            if attention_mask is not None and attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same [batch, length] shape as "
                    f"input_ids; got {tuple(attention_mask.shape)} vs "
                    f"{tuple(input_ids.shape)}"
                )

            if not self.auto_remap_tokenizer_ids or input_ids.numel() == 0:
                return input_ids.long(), attention_mask

            offset = self.tokenizer_base_offset

            # Native CrossDNA IDs already occupy 0..alphabet_size-1.
            if int(input_ids.max().item()) < offset:
                return input_ids.long(), attention_mask

            upper = offset + int(self.config.alphabet_size)
            base_mask = (input_ids >= offset) & (input_ids < upper)
            n_id = int(self.config.alphabet_size) - 1

            normalized = torch.where(
                base_mask,
                input_ids - offset,
                torch.full_like(input_ids, n_id),
            ).long()

            # Exclude tokenizer special tokens from the sequence-level pool.
            if attention_mask is None:
                effective_mask = base_mask.to(dtype=torch.long)
            else:
                effective_mask = (attention_mask.to(dtype=torch.bool) & base_mask).to(
                    dtype=attention_mask.dtype
                )

            return normalized, effective_mask

        def _pool_sequence(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if hidden_states.ndim != 3:
                raise ValueError(
                    f"Expected hidden_states with shape [B, L, H], got {tuple(hidden_states.shape)}"
                )

            batch_size, seq_len, _ = hidden_states.shape
            if attention_mask is None:
                mask = torch.ones(
                    (batch_size, seq_len),
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            else:
                if attention_mask.shape != (batch_size, seq_len):
                    raise ValueError(
                        "attention_mask must match hidden states [B, L]; got "
                        f"{tuple(attention_mask.shape)}"
                    )
                mask = attention_mask.to(
                    device=hidden_states.device,
                    dtype=torch.bool,
                )

            all_masked = ~mask.any(dim=1)

            if self.pooling == "mean":
                weights = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                denom = weights.sum(dim=1).clamp_min(1.0)
                pooled = (hidden_states * weights).sum(dim=1) / denom
                return pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)

            if self.pooling == "max":
                fill_value = torch.finfo(hidden_states.dtype).min
                masked = hidden_states.masked_fill(
                    ~mask.unsqueeze(-1),
                    fill_value,
                )
                pooled = masked.max(dim=1).values
                return pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)

            if self.pooling == "first":
                first_idx = mask.to(torch.int64).argmax(dim=1)
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    first_idx,
                ]
                return pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)

            if self.pooling == "last":
                positions = (
                    torch
                    .arange(
                        seq_len,
                        device=hidden_states.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                last_idx = positions.masked_fill(~mask, -1).max(dim=1).values
                safe_idx = last_idx.clamp_min(0)
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    safe_idx,
                ]
                return pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)

            raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **kwargs: Any,
        ) -> SequenceClassifierOutput | tuple:
            del kwargs  # CrossDNA's current backbone does not consume HF kwargs.

            if return_dict is None:
                return_dict = bool(getattr(self.config, "use_return_dict", True))
            if output_hidden_states is None:
                output_hidden_states = bool(getattr(self.config, "output_hidden_states", False))

            input_ids, pooling_mask = self._prepare_input_ids(
                input_ids,
                attention_mask,
            )

            # With for_representation=True the upstream backbone returns the
            # fused token representation [B, L, d_model].
            hidden_states, _ = self.backbone(input_ids)
            pooled = self._pool_sequence(hidden_states, pooling_mask)
            logits = self.classifier(self.classifier_dropout(pooled))

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                problem_type = getattr(self.config, "problem_type", None)

                if problem_type is None:
                    if self.num_labels == 1:
                        problem_type = "regression"
                    elif labels.dtype in {
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    }:
                        problem_type = "single_label_classification"
                    else:
                        problem_type = "multi_label_classification"
                    self.config.problem_type = problem_type

                if problem_type == "regression":
                    if self.num_labels == 1:
                        loss = F.mse_loss(
                            logits.squeeze(-1),
                            labels.to(logits.dtype).squeeze(-1),
                        )
                    else:
                        loss = F.mse_loss(logits, labels.to(logits.dtype))
                elif problem_type == "single_label_classification":
                    loss = F.cross_entropy(
                        logits.view(-1, self.num_labels),
                        labels.long().view(-1),
                    )
                elif problem_type == "multi_label_classification":
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels.to(logits.dtype),
                    )
                else:
                    raise ValueError(
                        "Unsupported problem_type. Expected one of "
                        "'regression', 'single_label_classification', or "
                        f"'multi_label_classification'; got {problem_type!r}."
                    )

            hidden_output = (hidden_states,) if output_hidden_states else None

            if not return_dict:
                output: tuple[Any, ...] = (logits,)
                if output_hidden_states:
                    output += (hidden_output,)
                return ((loss,) + output) if loss is not None else output  # noqa: RUF005

            return SequenceClassifierOutput(
                loss=loss,  # type: ignore[arg-type]
                logits=logits,
                hidden_states=hidden_output,
                attentions=None,
            )

    # Bind the generated class to the exact remote CrossDNAConfig class object.
    CrossDNAForSequenceClassification.config_class = config_class
    CrossDNAForSequenceClassification.__name__ = "CrossDNAForSequenceClassification"
    CrossDNAForSequenceClassification.__qualname__ = "CrossDNAForSequenceClassification"
    CrossDNAForSequenceClassification.__module__ = __name__

    _CLASS_CACHE[cache_key] = CrossDNAForSequenceClassification
    return CrossDNAForSequenceClassification


def _register_crossdna_sequence_classification(
    config: Any,
    model_path: str,
    auto_model_for_sequence_classification: Any,
) -> type:
    """Register the local classifier against the exact loaded config class."""
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForMaskedLM")
    if not isinstance(class_ref, str) or not class_ref:
        raise ValueError(
            "CrossDNA config does not provide auto_map['AutoModelForMaskedLM']; "
            "cannot locate the upstream CrossDNA implementation."
        )

    base_mlm_class = get_class_from_dynamic_module(
        class_ref,
        model_path,
    )
    model_class = _build_crossdna_sequence_classification_class(
        base_mlm_class,
        config.__class__,
    )

    register = getattr(auto_model_for_sequence_classification, "register", None)
    if register is None:
        raise TypeError(
            "CrossDNA sequence classification requires the Hugging Face "
            "transformers AutoModelForSequenceClassification.register API."
        )

    register(
        config.__class__,
        model_class,
        exist_ok=True,
    )
    return model_class


def _problem_type_from_task(task_type: str) -> str:
    if task_type in {"binary", "multiclass"}:
        return "single_label_classification"
    if task_type == "multilabel":
        return "multi_label_classification"
    if task_type == "regression":
        return "regression"
    raise ValueError(f"Unsupported CrossDNA sequence task: {task_type}")


def _handle_crossdna_models(
    task_type: str,
    model_path: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    modules: dict[str, Any],
    head_config: Any = None,
    custom_tokenizer: Any = None,
    bnb_config: Any = None,
) -> tuple[Any, Any]:
    """Handle CrossDNA after DNALLM has downloaded/resolved the model path.

    Returns ``(None, None)`` when *model_path* is not a CrossDNA snapshot so the
    normal DNALLM loading path can continue unchanged.
    """
    checkpoint_dir = _resolve_crossdna_checkpoint_dir(model_path)
    if checkpoint_dir is None:
        return None, None

    logger.info(f"Detected CrossDNA checkpoint at {checkpoint_dir}")

    if head_config is not None:
        raise ValueError(
            "CrossDNA's special adapter currently uses its native sequence "
            "classification head. Remove task.head_config when fine-tuning "
            "CrossDNA with AutoModelForSequenceClassification."
        )

    auto_config = modules["AutoConfig"]
    auto_tokenizer = modules["AutoTokenizer"]

    if custom_tokenizer is None:
        tokenizer = auto_tokenizer.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
    else:
        tokenizer = custom_tokenizer()

    # Keep original MLM support working from the repo root by resolving 8.1M.
    if task_type == "mask":
        model_load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if bnb_config is not None:
            model_load_kwargs["quantization_config"] = bnb_config
            model_load_kwargs["device_map"] = "auto"
        model = modules["AutoModelForMaskedLM"].from_pretrained(
            checkpoint_dir,
            **model_load_kwargs,
        )
        return model, tokenizer

    if task_type not in _SEQUENCE_TASKS:
        raise ValueError(
            "CrossDNA DNALLM adapter currently supports task types: "
            "mask, binary, multiclass, multilabel, regression. "
            f"Got {task_type!r}."
        )

    config = auto_config.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
    )
    config.num_labels = int(num_labels)
    config.id2label = id2label
    config.label2id = label2id
    config.problem_type = _problem_type_from_task(task_type)

    # Downstream-only settings are added to the in-memory config; the upstream
    # config.json remains unchanged.
    if not hasattr(config, "classifier_pooling"):
        config.classifier_pooling = "mean"
    if not hasattr(config, "classifier_dropout"):
        config.classifier_dropout = getattr(config, "dropout", 0.1)
    if not hasattr(config, "classifier_gate_freeze_steps"):
        config.classifier_gate_freeze_steps = 0
    if not hasattr(config, "classifier_detach_gate"):
        config.classifier_detach_gate = False
    if not hasattr(config, "auto_remap_tokenizer_ids"):
        config.auto_remap_tokenizer_ids = True
    if not hasattr(config, "tokenizer_base_offset"):
        config.tokenizer_base_offset = 7

    auto_model_for_sequence_classification = modules["AutoModelForSequenceClassification"]
    _register_crossdna_sequence_classification(
        config=config,
        model_path=checkpoint_dir,
        auto_model_for_sequence_classification=auto_model_for_sequence_classification,
    )

    model_load_kwargs = {
        "config": config,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_load_kwargs["quantization_config"] = bnb_config
        model_load_kwargs["device_map"] = "auto"

    model = auto_model_for_sequence_classification.from_pretrained(
        checkpoint_dir,
        **model_load_kwargs,
    )
    model._crossdna_checkpoint_dir = checkpoint_dir

    return model, tokenizer


__all__ = [
    "_handle_crossdna_models",
    "_register_crossdna_sequence_classification",
    "_resolve_crossdna_checkpoint_dir",
]
