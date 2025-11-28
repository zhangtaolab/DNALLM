import os
from ...utils import get_logger


logger = get_logger("dnallm.models.model")


def _handle_dnabert2_models(model_path: str, load_args: list) -> tuple:
    """Handle special case for DNABERT-2 models.
    If the installed Triton version not supports trans_b in tl.dot,
    we disable the triton flash attention by renaming the file
    flash_attn_triton.py to flash_attn_triton.py.disabled.
    This is a workaround for the issue in triton 2.0.0+ that
    causes flash attention to produce incorrect results.
    Args:
        model_path: Path to the model directory
    """
    if (
        "dnabert-2" not in os.path.basename(model_path.lower())
        and "dnabert-s" not in os.path.basename(model_path).lower()
    ):
        return None, None

    def triton_supports_trans_b():
        try:
            import triton.language as tl

            a = tl.zeros((1, 1), dtype=tl.float32)
            b = tl.zeros((1, 1), dtype=tl.float32)
            tl.dot(a, b, trans_b=True)
            return True
        except TypeError:
            return False
        except Exception:
            return True

    triton_file_path = None
    original_content = None
    should_disable_triton = triton_supports_trans_b()

    try:
        if should_disable_triton:
            triton_file_path = os.path.join(model_path, "flash_attn_triton.py")
            if os.path.exists(triton_file_path):
                # Step 1: Read and back up the original file content
                with open(triton_file_path, encoding="utf-8") as f:
                    original_content = f.read()
                # Step 2: Overwrite the file with code
                # that will raise an ImportError
                with open(triton_file_path, "w", encoding="utf-8") as f:
                    f.write(
                        'raise ImportError("Temporarily disabled '
                        'by script due to incompatible Triton version.")\n'
                    )
                logger.warning(
                    "Triton version does not support, "
                    "flash_attn_triton disabled."
                )
            else:
                triton_file_path = None

        from ..model import _load_model_by_task_type

        model, tokenizer = _load_model_by_task_type(*load_args)
        return model, tokenizer
    finally:
        # This block will execute no matter what,
        # ensuring the original file is restored.
        if original_content is not None and triton_file_path is not None:
            with open(triton_file_path, "w", encoding="utf-8") as f:
                f.write(original_content)
