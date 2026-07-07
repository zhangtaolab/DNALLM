# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpeed-LLM inference entrypoint with optional Mamba2 sequence-classification support.

This file keeps the original text-generation path unchanged.  When
``--inference-task sequence_classification`` or ``--inference-task regression`` is
used, it builds the Mamba2 classification/regression model, loads the Megatron
checkpoint and runs one-pass prediction instead of token-by-token generation.

The classification path expects that ``mindspeed_llm.tasks.inference.module`` has
``MegatronModuleForSequenceClassification`` available.  If you used a custom
fine-tuning model class, pass it through ``--classification-model-class`` so the
checkpoint key names match training exactly.
"""

import csv
import importlib
import json
import os
from typing import Any
from collections.abc import Iterable
from tqdm import tqdm

import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.legacy.model import GPTModel
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from mindspeed_llm.tasks.inference.infer_base import task_factory
from mindspeed_llm.training.utils import auto_coverage
from mindspeed_llm.tasks.inference.module import (
    GPTModelInfer,
    MegatronModuleForCausalLM,
    MegatronModuleForSequenceClassification,
)


def disable_dynamo() -> None:
    """Disable torch dynamo/inductor, matching the standalone fine-tuning wrapper."""
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

    try:
        import torch._dynamo  # pylint: disable=import-outside-toplevel

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
    except Exception:  # noqa: S110 - best-effort compatibility guard
        pass


def setup_env() -> None:
    """Set single-node default runtime envs used by the provided fine-tune script."""
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("CPU_AFFINITY_CONF", "1")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "3600")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "2")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "6000")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _safe_add_argument(parser, *names: str, **kwargs) -> None:
    """Add an argparse argument only when none of its option strings exists."""
    existing_options = {opt for action in parser._actions for opt in action.option_strings}
    if any(name in existing_options for name in names):
        return
    parser.add_argument(*names, **kwargs)


def add_mamba2_inference_args(parser):
    """Additional args needed by Mamba2 classification/regression inference.

    Core Megatron/MindSpeed options such as ``--num-layers`` and
    ``--hidden-size`` are intentionally not re-registered here, because they are
    already parsed by Megatron.  This provider only adds downstream task and I/O
    options that are not available in the original ``inference.py``.
    """
    _safe_add_argument(
        parser,
        "--inference-task",
        type=str,
        default="generation",
        choices=("generation", "sequence_classification", "classification", "regression"),
        help="Use generation for the original path, or sequence_classification/regression for Mamba2 heads.",
    )
    _safe_add_argument(
        parser,
        "--num-labels",
        "--num_labels",
        dest="num_labels",
        type=int,
        default=2,
        help="Number of labels for sequence classification; use 1 for regression.",
    )
    _safe_add_argument(
        parser,
        "--problem-type",
        "--problem_type",
        dest="problem_type",
        type=str,
        default="single_label_classification",
        choices=(
            "regression",
            "single_label_classification",
            "multi_label_classification",
            "binary_classification",
        ),
        help="Prediction head type used during fine-tuning.",
    )
    _safe_add_argument(
        parser,
        "--input-text",
        type=str,
        default=None,
        help="One text/sequence to predict. For batch prediction, use --input-file.",
    )
    _safe_add_argument(
        parser,
        "--input-file",
        type=str,
        default=None,
        help="TXT/CSV/TSV/JSONL file for sequence classification or regression inference.",
    )
    _safe_add_argument(
        parser,
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL/CSV path. If omitted, predictions are printed on rank 0.",
    )
    _safe_add_argument(
        parser,
        "--text-column",
        type=str,
        default="sequence",
        help="Input text column name for CSV/TSV/JSONL files.",
    )
    _safe_add_argument(
        parser,
        "--id-column",
        type=str,
        default=None,
        help="Optional ID column copied to the output records.",
    )
    _safe_add_argument(
        parser,
        "--label-names",
        type=str,
        default=None,
        help="Comma-separated label names, e.g. negative,positive.",
    )
    _safe_add_argument(
        parser,
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary or multi-label prediction.",
    )
    _safe_add_argument(
        parser,
        "--classification-model-class",
        type=str,
        default=None,
        help=(
            "Optional custom model class used in fine-tuning, in module:Class format. "
            "Use this when checkpoint keys depend on your training model wrapper."
        ),
    )
    _safe_add_argument(
        parser,
        "--classification-head-bias",
        action="store_true",
        default=True,
        help="Use bias in the fallback local classifier head.",
    )
    _safe_add_argument(
        parser,
        "--no-classification-head-bias",
        dest="classification_head_bias",
        action="store_false",
        help="Disable bias in the fallback local classifier head.",
    )
    return parser


def is_sequence_prediction_task(args) -> bool:
    task = str(getattr(args, "inference_task", "generation")).lower()
    return task in {"sequence_classification", "classification", "regression"}


def _get_optional_arg(args, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


class Mamba2ForSequenceClassificationInfer(torch.nn.Module):
    """Fallback Mamba2 sequence-classification/regression model.

    Prefer using the exact model class from fine-tuning via
    ``--classification-model-class module:Class``.  This fallback is useful when
    your fine-tuning head is simply a linear layer on the last non-padding token.
    """

    def __init__(self, config, mamba_stack_spec, pre_process=True, post_process=True):
        super().__init__()
        from megatron.core.models.mamba import MambaModel  # pylint: disable=import-outside-toplevel

        args = get_args()
        self.num_labels = int(getattr(args, "num_labels", 2))
        self.problem_type = getattr(args, "problem_type", "single_label_classification")
        # post_process=False makes the Mamba backbone return hidden states rather
        # than LM logits, which can then be pooled for sequence prediction.
        self.backbone = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            hybrid_attention_ratio=_get_optional_arg(args, "hybrid_attention_ratio", 0.0),
            hybrid_mlp_ratio=_get_optional_arg(args, "hybrid_mlp_ratio", 0.0),
            hybrid_override_pattern=_get_optional_arg(args, "hybrid_override_pattern", None),
            post_process=False,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=_get_optional_arg(args, "rotary_base", 10000),
        )
        self.score = torch.nn.Linear(
            args.hidden_size,
            self.num_labels,
            bias=bool(getattr(args, "classification_head_bias", True)),
        )
        self.post_process = post_process

    def forward(
        self, input_ids, position_ids=None, attention_mask=None, labels=None, lengths=None, **kwargs
    ):
        del labels, kwargs
        hidden_states = self.backbone(input_ids, position_ids, attention_mask)
        hidden_states = self._unwrap_hidden_states(hidden_states)
        pooled = self._pool_last_non_padding_token(hidden_states, input_ids, lengths)
        return self.score(pooled)

    @staticmethod
    def _unwrap_hidden_states(hidden_states):
        if torch.is_tensor(hidden_states):
            return hidden_states
        if isinstance(hidden_states, dict):
            for key in ("hidden_states", "last_hidden_state", "output"):
                value = hidden_states.get(key)
                if torch.is_tensor(value):
                    return value
        if hasattr(hidden_states, "last_hidden_state") and torch.is_tensor(
            hidden_states.last_hidden_state
        ):
            return hidden_states.last_hidden_state
        if isinstance(hidden_states, (list, tuple)):
            for value in hidden_states:
                if torch.is_tensor(value):
                    return value
        raise RuntimeError("Could not extract hidden states from Mamba backbone output.")

    def _pool_last_non_padding_token(self, hidden_states, input_ids, lengths=None):
        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3-D hidden states, got shape {tuple(hidden_states.shape)}")

        if hidden_states.shape[0] == batch_size:
            batch_first_hidden = hidden_states
        elif hidden_states.shape[1] == batch_size:
            batch_first_hidden = hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError("Cannot infer batch dimension from Mamba hidden states.")

        if lengths is None:
            args = get_args()
            pad_id = getattr(args, "pad_id", None)
            if pad_id is None:
                pad_id = getattr(args, "eod_id", None)
            if pad_id is None:
                pad_id = getattr(args, "eos_id", None)
            if pad_id is None:
                lengths = torch.full(
                    (batch_size,), seq_length, dtype=torch.long, device=input_ids.device
                )
            else:
                lengths = (input_ids != pad_id).long().sum(dim=-1)

        indices = (lengths.to(batch_first_hidden.device) - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=batch_first_hidden.device)
        return batch_first_hidden[batch_indices, indices]


def import_object(import_path: str):
    """Import an object from 'module:Class' or 'module.Class'."""
    if ":" in import_path:
        module_name, object_name = import_path.split(":", 1)
    else:
        module_name, object_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _signature_accepts(signature, name: str) -> bool:
    """Whether a callable signature accepts a keyword argument."""
    import inspect  # pylint: disable=import-outside-toplevel

    if name in signature.parameters:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _filter_kwargs_for_signature(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by ``cls.__init__`` unless it has **kwargs."""
    import inspect  # pylint: disable=import-outside-toplevel

    signature = inspect.signature(cls.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _try_build_custom_class(cls, *, config, mamba_stack_spec, pre_process, post_process):
    """Instantiate the exact fine-tuning model class using common Mamba2 signatures.

    The user's training class usually follows Megatron Core MambaModel and may
    require ``config``, ``mamba_stack_spec``, ``vocab_size`` and
    ``max_sequence_length``.  The earlier implementation tried ``cls()`` as a
    fallback, which hides the useful constructor mismatch and caused:

        TypeError: Mamba2ForSequenceClassification.__init__() missing ...

    Here we first pass the required backbone dimensions, then progressively try
    compatible variants for older/local wrappers.
    """
    args = get_args()

    common_kwargs = {
        "args": args,
        "config": config,
        "mamba_stack_spec": mamba_stack_spec,
        "transformer_layer_spec": mamba_stack_spec,
        "vocab_size": args.padded_vocab_size,
        "max_sequence_length": args.max_position_embeddings,
        "pre_process": pre_process,
        "post_process": post_process,
        "fp16_lm_cross_entropy": getattr(args, "fp16_lm_cross_entropy", False),
        "parallel_output": False,
        "num_labels": getattr(args, "num_labels", 2),
        "problem_type": getattr(args, "problem_type", "single_label_classification"),
    }

    filtered_kwargs = _filter_kwargs_for_signature(cls, common_kwargs)

    # Most useful path: keyword instantiation matching the training class.
    call_patterns = [lambda: cls(**filtered_kwargs)]

    # Explicit common signatures.  These are kept after kwargs because some
    # classes do not expose all arguments as keywords.
    call_patterns.extend([
        lambda: cls(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
        ),
        lambda: cls(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
        ),
        lambda: cls(
            config,
            mamba_stack_spec,
            args.padded_vocab_size,
            args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
        ),
        lambda: cls(config, mamba_stack_spec, args.padded_vocab_size, args.max_position_embeddings),
        lambda: cls(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            pre_process=pre_process,
            post_process=post_process,
        ),
        lambda: cls(config, mamba_stack_spec, pre_process=pre_process, post_process=post_process),
        lambda: cls(config=config, pre_process=pre_process, post_process=post_process),
        lambda: cls(
            args=args,
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            pre_process=pre_process,
            post_process=post_process,
        ),
        lambda: cls(args),
    ])

    errors = []
    for builder in call_patterns:
        try:
            return builder()
        except TypeError as error:
            errors.append(str(error))

    joined_errors = "\n  - ".join(errors[-6:])
    raise TypeError(
        f"Failed to instantiate custom classification model class {cls}. "
        "Please make sure --classification-model-class points to the exact class used during fine-tuning. "
        f"Recent constructor errors:\n  - {joined_errors}"
    )


def build_mamba2_sequence_classification_model(pre_process=True, post_process=True):
    """Build the Mamba2 model used by classification/regression checkpoints."""
    from megatron.core.transformer import TransformerConfig  # pylint: disable=import-outside-toplevel

    args = get_args()
    print_rank_0("building Mamba2 sequence classification/regression model ...")
    config = core_transformer_config_from_args(args, TransformerConfig)

    if getattr(args, "use_legacy_models", False):
        raise AssertionError("Mamba2 is only supported with Megatron Core models.")

    if args.spec is None:
        raise ValueError(
            "Mamba2 inference requires --spec mindspeed_llm.tasks.models.spec.mamba_spec layer_spec"
        )
    mamba_stack_spec = import_module(args.spec)

    if getattr(args, "classification_model_class", None):
        cls = import_object(args.classification_model_class)
        return _try_build_custom_class(
            cls,
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    return Mamba2ForSequenceClassificationInfer(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        pre_process=pre_process,
        post_process=post_process,
    )


def model_provider(
    pre_process=True, post_process=True
) -> GPTModelInfer | GPTModel | torch.nn.Module:
    """Builds a generation model or a Mamba2 sequence prediction model."""
    args = get_args()

    if is_sequence_prediction_task(args):
        return build_mamba2_sequence_classification_model(
            pre_process=pre_process,
            post_process=post_process,
        )

    use_te = args.transformer_impl == "transformer_engine"

    if args.sequence_parallel and args.use_kv_cache:
        raise AssertionError("Use_kv_cache can not be true in sequence_parallel mode.")

    if args.num_layers_per_virtual_pipeline_stage is not None:
        raise AssertionError("VPP is not supported for inference.")

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm
                )

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = GPTModel(
            config,
            parallel_output=True if args.sequence_parallel else False,
            pre_process=pre_process,
            post_process=post_process,
        )

    return model


def parse_label_names(label_names: str | None):
    if not label_names:
        return None
    labels = [label.strip() for label in label_names.split(",") if label.strip()]
    return labels or None


def iter_batches(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def load_prediction_inputs(args) -> list[dict[str, Any]]:
    """Load rank-0 inference records from --input-text or --input-file."""
    if getattr(args, "input_text", None):
        return [{"id": 0, "text": args.input_text}]

    input_file = getattr(args, "input_file", None)
    if input_file is None:
        raise ValueError(
            "For classification/regression inference, provide --input-text or --input-file."
        )

    ext = os.path.splitext(input_file)[1].lower()
    records: list[dict[str, Any]] = []

    if ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        with open(input_file, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if args.text_column not in (reader.fieldnames or []):
                raise ValueError(f"Text column {args.text_column!r} is not found in {input_file}.")
            for row_idx, row in enumerate(reader):
                record = {
                    "id": row.get(args.id_column, row_idx) if args.id_column else row_idx,
                    "text": row[args.text_column],
                }
                for key, value in row.items():
                    if key not in {args.text_column, args.id_column}:
                        record[key] = value
                records.append(record)
    elif ext in {".jsonl", ".json"}:
        with open(input_file, encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if args.text_column not in row:
                    raise ValueError(
                        f"Text column {args.text_column!r} is not found in JSON record {row_idx}."
                    )
                record = {
                    "id": row.get(args.id_column, row_idx) if args.id_column else row_idx,
                    "text": row[args.text_column],
                }
                for key, value in row.items():
                    if key not in {args.text_column, args.id_column}:
                        record[key] = value
                records.append(record)
    else:
        with open(input_file, encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                text = line.strip()
                if text:
                    records.append({"id": row_idx, "text": text})

    if not records:
        raise ValueError("No valid inference records were loaded.")
    return records


def _select_batch_item(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if len(value) == 0:
            return value
        if len(value) > index:
            return value[index]
    return value


def merge_predictions(
    records: list[dict[str, Any]], prediction: dict[str, Any]
) -> list[dict[str, Any]]:
    merged = []
    for index, record in enumerate(records):
        output = dict(record)
        for key in ("probabilities", "predictions", "values", "labels"):
            if key in prediction:
                output[key] = _select_batch_item(prediction[key], index)
        merged.append(output)
    return merged


def write_prediction_outputs(outputs: list[dict[str, Any]], output_file: str | None) -> None:
    if not output_file:
        for output in outputs:
            print_rank_0(json.dumps(output, ensure_ascii=False))
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    ext = os.path.splitext(output_file)[1].lower()
    if ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        fieldnames = []
        for row in outputs:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(output_file, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for row in outputs:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                })
    else:
        with open(output_file, "w", encoding="utf-8") as handle:
            for row in outputs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_rank_0(f"Predictions have been written to {output_file}")


def run_sequence_prediction(args, model) -> None:
    """Run classification/regression inference and save/print outputs."""
    records = load_prediction_inputs(args)
    label_names = parse_label_names(getattr(args, "label_names", None))
    batch_size = max(1, int(getattr(args, "micro_batch_size", 1)))
    outputs: list[dict[str, Any]] = []

    for batch in tqdm(iter_batches(records, batch_size)):
        texts = [record["text"] for record in batch]
        prediction = model.predict(
            texts,
            problem_type=args.problem_type,
            num_labels=args.num_labels,
            label_names=label_names,
            threshold=args.threshold,
        )
        outputs.extend(merge_predictions(batch, prediction))

    write_prediction_outputs(outputs, getattr(args, "output_file", None))


@auto_coverage
def main():
    disable_dynamo()
    setup_env()

    initialize_megatron(
        extra_args_provider=add_mamba2_inference_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )

    args = get_args()

    if is_sequence_prediction_task(args):
        model = MegatronModuleForSequenceClassification.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=args.load,
        )
        run_sequence_prediction(args, model)
    else:
        model = MegatronModuleForCausalLM.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=args.load,
        )
        task_factory(args, model)


if __name__ == "__main__":
    main()
