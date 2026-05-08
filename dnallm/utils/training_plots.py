"""Standalone plotting utilities for training metrics visualization.

These functions read from a HuggingFace Trainer's state.log_history
and produce static plots as PNG files. They serve as a fallback when
TensorBoard or WandB are not available.
"""

from typing import Any
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_loss_curve(
    log_history: list[dict[str, Any]],
    output_path: str | Path = "training_loss.png",
    title: str = "Training Loss Curve",
) -> Path:
    """Plot training and evaluation loss curves from trainer log history.

    Args:
        log_history: List of log dictionaries from trainer.state.log_history.
        output_path: Path to save the PNG file.
        title: Plot title.

    Returns:
        Path to the saved PNG file.
    """
    output_path = Path(output_path)

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if train_losses:
        ax.plot(train_steps, train_losses, "b-", label="Train Loss", linewidth=2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, "r--", label="Eval Loss", linewidth=2)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_lr_schedule(
    log_history: list[dict[str, Any]],
    output_path: str | Path = "lr_schedule.png",
    title: str = "Learning Rate Schedule",
) -> Path:
    """Plot learning rate schedule from trainer log history.

    Args:
        log_history: List of log dictionaries from trainer.state.log_history.
        output_path: Path to save the PNG file.
        title: Plot title.

    Returns:
        Path to the saved PNG file.
    """
    output_path = Path(output_path)

    steps = []
    learning_rates = []

    for entry in log_history:
        step = entry.get("step")
        lr = entry.get("learning_rate")
        if step is not None and lr is not None:
            steps.append(step)
            learning_rates.append(lr)

    if not learning_rates:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No learning rate data found",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_title(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, learning_rates, "g-", linewidth=2)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Use log scale if LR spans multiple orders of magnitude
    min_lr = min(learning_rates)
    if min_lr > 0 and max(learning_rates) / min_lr > 100:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
