from .logger import get_logger
from torch.cuda import get_device_capability


logger = get_logger("dnallm.utils.support")


def is_fp8_capable() -> bool:
    """Check if the current CUDA device supports FP8 precision.

    Returns:
                True if the device supports FP8 (
            compute capability >= 9.0),
            False otherwise
    """
    major, minor = get_device_capability()
    # Hopper (H100) has compute capability 9.0
    if (major, minor) >= (9, 0):
        return True
    else:
        logger.warning(
            f"Current device compute capability is {major}.{minor}, "
            "which does not support FP8."
        )
        return False


def is_flash_attention_capable():
    """Check if Flash Attention has been installed.
    Returns:
                True if Flash Attention is installed and the device supports it
            False otherwise
    """
    try:
        import flash_attn  # pyright: ignore[reportMissingImports]

        _ = flash_attn
        return True
    except Exception as e:
        logger.warning(f"Cannot find supported Flash Attention: {e}")
        return False
