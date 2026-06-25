#!/usr/bin/env python3
"""Small wrapper for Mamba2 sequence-classification/regression inference.

Put this file next to the modified MindSpeed-LLM ``inference.py`` and run this
wrapper instead of passing a long list of Megatron arguments every time.

Examples
--------
Single sequence:
    python infer_mamba2_megatron.py \
        --load /path/to/megatron_ckpt \
        --tokenizer-name-or-path /path/to/tokenizer \
        --input-text ACGTACGTACGT

Batch CSV/TSV/JSONL/TXT:
    python infer_mamba2_megatron.py \
        --load /path/to/megatron_ckpt \
        --tokenizer-name-or-path /path/to/tokenizer \
        --input-file test.csv \
        --text-column sequence \
        --output-file pred.jsonl
"""

import argparse
import importlib
import os
import sys


# ==========================
# 1) Default model settings
# ==========================
# Keep these values identical to the fine-tuning run, otherwise checkpoint keys
# and tensor shapes may not match.
DEFAULT_NUM_LAYERS = 48
DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_NUM_ATTENTION_HEADS = 32
DEFAULT_MAMBA_NUM_GROUPS = 1
DEFAULT_MAMBA_CHUNK_SIZE = 256
DEFAULT_MAMBA_STATE_DIM = 128
DEFAULT_MAMBA_D_CONV = 4
DEFAULT_MAMBA_EXPAND = 2
DEFAULT_MAMBA_HEAD_DIM = 64

# Classification/regression head. Keep consistent with training.
DEFAULT_NUM_LABELS = 2
DEFAULT_PROBLEM_TYPE = "single_label_classification"
# Examples:
#   "single_label_classification"
#   "binary_classification"
#   "multi_label_classification"
#   "regression"

# If your training class name/parameter names differ from the fallback head in
# inference.py, set this to the exact fine-tuning model class, e.g.
# DEFAULT_CLASSIFICATION_MODEL_CLASS = "dnallm.finetune.megatron:Mamba2ForSequenceClassification"
DEFAULT_CLASSIFICATION_MODEL_CLASS = (
    "dnallm.models.special.mamba_npu:Mamba2ForSequenceClassification"
)

# Optional names for readable outputs, e.g. "negative,positive".
DEFAULT_LABEL_NAMES = None


# ==========================
# 2) Default inference settings
# ==========================
DEFAULT_TOKENIZER_TYPE = "PretrainedFromHF"
DEFAULT_SEQ_LENGTH = 512
DEFAULT_MAX_POSITION_EMBEDDINGS = 512
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GLOBAL_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.5
DEFAULT_BF16 = False

# Parallelism defaults for a single-card/single-process inference run.
DEFAULT_TENSOR_MODEL_PARALLEL_SIZE = 1
DEFAULT_PIPELINE_MODEL_PARALLEL_SIZE = 1

# Import target for the modified inference.py. In most cases, keep it as
# "inference" and put this wrapper in the same directory as inference.py.
DEFAULT_INFERENCE_MODULE = "inference_mamba2_npu"


def parse_user_args():
    parser = argparse.ArgumentParser(
        description="Wrapper for Mamba2 Megatron classification/regression inference"
    )

    # Only frequently changed paths / data options are required at runtime.
    parser.add_argument(
        "--load", type=str, required=True, help="Megatron checkpoint directory to load."
    )
    parser.add_argument(
        "--tokenizer-name-or-path", type=str, required=True, help="HF tokenizer directory."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-text", type=str, default=None, help="One sequence/text to predict."
    )
    input_group.add_argument(
        "--input-file", type=str, default=None, help="TXT/CSV/TSV/JSONL input file."
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL/CSV/TSV file. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--text-column", type=str, default="sequence", help="Text column for CSV/TSV/JSONL inputs."
    )
    parser.add_argument(
        "--id-column", type=str, default=None, help="Optional ID column copied to outputs."
    )

    # Common overrides, so you do not need to edit the file for small changes.
    parser.add_argument(
        "--num_labels", "--num-labels", dest="num_labels", type=int, default=DEFAULT_NUM_LABELS
    )
    parser.add_argument(
        "--problem_type",
        "--problem-type",
        dest="problem_type",
        type=str,
        default=DEFAULT_PROBLEM_TYPE,
    )
    parser.add_argument("--label-names", type=str, default=DEFAULT_LABEL_NAMES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument(
        "--max-position-embeddings", type=int, default=DEFAULT_MAX_POSITION_EMBEDDINGS
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=DEFAULT_BF16)

    # Less frequent model-structure overrides.
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-attention-heads", type=int, default=DEFAULT_NUM_ATTENTION_HEADS)
    parser.add_argument("--mamba-num-groups", type=int, default=DEFAULT_MAMBA_NUM_GROUPS)
    parser.add_argument("--mamba-chunk-size", type=int, default=DEFAULT_MAMBA_CHUNK_SIZE)
    parser.add_argument("--mamba-state-dim", type=int, default=DEFAULT_MAMBA_STATE_DIM)
    parser.add_argument("--mamba-d-conv", type=int, default=DEFAULT_MAMBA_D_CONV)
    parser.add_argument("--mamba-expand", type=int, default=DEFAULT_MAMBA_EXPAND)
    parser.add_argument("--mamba-head-dim", type=int, default=DEFAULT_MAMBA_HEAD_DIM)
    parser.add_argument(
        "--classification-model-class", type=str, default=DEFAULT_CLASSIFICATION_MODEL_CLASS
    )
    parser.add_argument("--inference-module", type=str, default=DEFAULT_INFERENCE_MODULE)

    # Ignore launcher-injected args such as --local-rank=0. We set LOCAL_RANK by env below.
    args, _ = parser.parse_known_args()
    return args


def add_arg(args: list[str], name: str, value) -> None:
    if value is not None:
        args.extend([name, str(value)])


def build_default_args(u) -> list[str]:
    """Build the long Megatron/MindSpeed arg list for modified inference.py."""
    args: list[str] = [
        # Route to the new classification/regression path in modified inference.py.
        "--inference-task",
        "sequence_classification",
        # Mamba2 model spec: same family as fine-tuning.
        "--spec",
        "mindspeed_llm.tasks.models.spec.mamba_spec",
        "layer_spec",
        "--no-shared-storage",
        "--use-flash-attn",
        "--use-mcore-models",
        "--tensor-model-parallel-size",
        str(DEFAULT_TENSOR_MODEL_PARALLEL_SIZE),
        "--pipeline-model-parallel-size",
        str(DEFAULT_PIPELINE_MODEL_PARALLEL_SIZE),
        "--sequence-parallel",
        "--group-query-attention",
        "--num-query-groups",
        str(u.num_attention_heads),
        "--make-vocab-size-divisible-by",
        "1",
        # Architecture/runtime flags matching the fine-tuning wrapper.
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--init-method-std",
        "0.02",
        "--position-embedding-type",
        "none",
        "--normalization",
        "RMSNorm",
        "--use-fused-swiglu",
        "--use-fused-rmsnorm",
        "--swiglu",
        "--no-masked-softmax-fusion",
        "--attention-softmax-in-fp32",
        "--rotary-base",
        "10000",
        "--no-gradient-accumulation-fusion",
        "--norm-epsilon",
        "1e-6",
        "--model-type-hf",
        "mamba2",
        "--distributed-backend",
        "nccl",
        "--transformer-impl",
        "local",
    ]

    # Model shape parameters: must match fine-tuning.
    add_arg(args, "--num-layers", u.num_layers)
    add_arg(args, "--hidden-size", u.hidden_size)
    add_arg(args, "--num-attention-heads", u.num_attention_heads)
    add_arg(args, "--mamba-num-groups", u.mamba_num_groups)
    add_arg(args, "--mamba-chunk-size", u.mamba_chunk_size)
    add_arg(args, "--mamba-state-dim", u.mamba_state_dim)
    add_arg(args, "--mamba-d-conv", u.mamba_d_conv)
    add_arg(args, "--mamba-expand", u.mamba_expand)
    add_arg(args, "--mamba-head-dim", u.mamba_head_dim)

    # Task/inference parameters.
    add_arg(args, "--num_labels", u.num_labels)
    add_arg(args, "--problem_type", u.problem_type)
    add_arg(args, "--tokenizer-type", DEFAULT_TOKENIZER_TYPE)
    add_arg(args, "--tokenizer-name-or-path", u.tokenizer_name_or_path)
    add_arg(args, "--seq-length", u.seq_length)
    add_arg(args, "--max-position-embeddings", u.max_position_embeddings)
    add_arg(args, "--micro-batch-size", u.micro_batch_size)
    add_arg(args, "--global-batch-size", u.global_batch_size)
    add_arg(args, "--load", u.load)
    add_arg(args, "--threshold", u.threshold)
    add_arg(args, "--label-names", u.label_names)
    add_arg(args, "--classification-model-class", u.classification_model_class)

    # Input/output parameters consumed by the modified inference.py.
    add_arg(args, "--input-text", u.input_text)
    add_arg(args, "--input-file", u.input_file)
    add_arg(args, "--text-column", u.text_column)
    add_arg(args, "--id-column", u.id_column)
    add_arg(args, "--output-file", u.output_file)

    if u.bf16:
        args.append("--bf16")
        args.append("--reuse-fp32-param")

    return args


def disable_dynamo() -> None:
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    try:
        import torch._dynamo  # pylint: disable=import-outside-toplevel

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
    except Exception: # noqa: S110
        pass


def setup_env() -> None:
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


def import_main(module_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"{module_name!r} does not provide a main() function")
    return module.main


if __name__ == "__main__":
    disable_dynamo()
    setup_env()

    user_args = parse_user_args()
    megatron_args = build_default_args(user_args)

    # Replace the user-facing short CLI with the full Megatron CLI before
    # calling the modified inference.py main().
    sys.argv = ["inference_mamba2_npu.py", *megatron_args]

    start_inference = import_main(user_args.inference_module)
    start_inference()
