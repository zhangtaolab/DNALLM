import os
import sys
import argparse


def parse_user_args():
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-attention-heads", type=int, default=24)
    parser.add_argument("--mamba-num-groups", type=int, default=1)
    parser.add_argument("--mamba-chunk-size", type=int, default=256)
    parser.add_argument("--mamba-state-dim", type=int, default=128)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-head-dim", type=int, default=64)

    # 训练参数
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--problem_type", type=str, default="single_label_classification")
    parser.add_argument("--tokenizer-type", type=str, default="PretrainedFromHF")
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True)

    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-iters", type=int, default=2000)

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000)

    parser.add_argument("--lr", type=str, default="2e-5")
    parser.add_argument("--min-lr", type=str, default="2e-6")
    parser.add_argument("--lr-decay-style", type=str, default="cosine")
    parser.add_argument("--weight-decay", type=str, default="0.1")
    parser.add_argument("--adam-beta1", type=str, default="0.9")
    parser.add_argument("--adam-beta2", type=str, default="0.95")

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    # 输入输出参数
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--dev_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--tensorboard_dir", type=str, default=None)

    parser.add_argument(
        "--enable-hf2mg-convert",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args, _ = parser.parse_known_args()

    return args


def add_arg(args, name, value):
    args.extend([name, str(value)])


def build_default_args(u):
    tensorboard_dir = u.tensorboard_dir
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(u.save, "tf_logs")

    args = [
        "--spec",
        "mindspeed_llm.tasks.models.spec.mamba_spec",
        "layer_spec",
        "--reuse-fp32-param",
        "--no-shared-storage",
        "--use-distributed-optimizer",
        "--use-flash-attn",
        "--use-mcore-models",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--sequence-parallel",
        "--group-query-attention",
        "--num-query-groups",
        str(u.num_attention_heads),
        "--make-vocab-size-divisible-by",
        "1",
        "--clip-grad",
        "1.0",
        "--initial-loss-scale",
        "65536",
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
        # "--no-load-optim",
        # "--no-load-rng",
        # "--no-save-rng",
        # "--no-save-optim",
        "--model-type-hf",
        "mamba2",
        "--distributed-backend",
        "nccl",
        "--transformer-impl",
        "local",
    ]

    # 模型参数
    add_arg(args, "--num-layers", u.num_layers)
    add_arg(args, "--hidden-size", u.hidden_size)
    add_arg(args, "--num-attention-heads", u.num_attention_heads)
    add_arg(args, "--mamba-num-groups", u.mamba_num_groups)
    add_arg(args, "--mamba-chunk-size", u.mamba_chunk_size)
    add_arg(args, "--mamba-state-dim", u.mamba_state_dim)
    add_arg(args, "--mamba-d-conv", u.mamba_d_conv)
    add_arg(args, "--mamba-expand", u.mamba_expand)
    add_arg(args, "--mamba-head-dim", u.mamba_head_dim)

    # 训练参数
    add_arg(args, "--num_labels", u.num_labels)
    add_arg(args, "--problem_type", u.problem_type)
    add_arg(args, "--tokenizer-type", u.tokenizer_type)
    add_arg(args, "--tokenizer-name-or-path", u.tokenizer_name_or_path)
    add_arg(args, "--seq-length", u.seq_length)
    add_arg(args, "--max-position-embeddings", u.max_position_embeddings)
    add_arg(args, "--micro-batch-size", u.micro_batch_size)
    add_arg(args, "--global-batch-size", u.global_batch_size)
    add_arg(args, "--epochs", u.epochs)
    add_arg(args, "--train-iters", u.train_iters)
    add_arg(args, "--log-interval", u.log_interval)
    add_arg(args, "--save-interval", u.save_interval)
    add_arg(args, "--eval-interval", u.eval_interval)
    add_arg(args, "--lr", u.lr)
    add_arg(args, "--min-lr", u.min_lr)
    add_arg(args, "--lr-decay-style", u.lr_decay_style)
    add_arg(args, "--weight-decay", u.weight_decay)
    add_arg(args, "--adam-beta1", u.adam_beta1)
    add_arg(args, "--adam-beta2", u.adam_beta2)

    if u.bf16:
        args.append("--bf16")

    # 输入输出参数
    add_arg(args, "--train_csv", u.train_csv)
    add_arg(args, "--dev_csv", u.dev_csv)
    add_arg(args, "--test_csv", u.test_csv)
    add_arg(args, "--load", u.load)
    add_arg(args, "--save", u.save)
    add_arg(args, "--tensorboard_dir", tensorboard_dir)

    if u.enable_hf2mg_convert:
        args.append("--enable-hf2mg-convert")

    return args


def disable_dynamo():
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"

    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
    except Exception:  # noqa: S110
        pass


def setup_env():
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


if __name__ == "__main__":
    disable_dynamo()
    setup_env()

    user_args = parse_user_args()
    megatron_args = build_default_args(user_args)

    sys.argv = ["finetune_mamba2_megatron.py", *megatron_args]

    from dnallm.finetune.megatron import start_train

    start_train()
