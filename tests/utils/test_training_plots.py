"""Tests for training plotting utilities."""

import tempfile
from pathlib import Path

import pytest

from dnallm.utils.training_plots import plot_loss_curve, plot_lr_schedule


class TestPlotLossCurve:
    """Test cases for plot_loss_curve function."""

    def test_plot_loss_curve_basic(self):
        """Test plotting with basic train/eval data."""
        log_history = [
            {"step": 10, "loss": 0.7, "learning_rate": 5e-5},
            {"step": 20, "loss": 0.5, "learning_rate": 4e-5},
            {"step": 30, "loss": 0.3, "learning_rate": 3e-5, "eval_loss": 0.45},
            {"step": 40, "loss": 0.25, "learning_rate": 2e-5, "eval_loss": 0.35},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "loss.png"
            result = plot_loss_curve(log_history, output_path=output_path)

            assert result.exists()
            assert result.stat().st_size > 0
            assert result.suffix == ".png"

    def test_plot_loss_curve_train_only(self):
        """Test plotting with only training loss (no eval)."""
        log_history = [
            {"step": 10, "loss": 0.7},
            {"step": 20, "loss": 0.5},
            {"step": 30, "loss": 0.3},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "train_only.png"
            result = plot_loss_curve(log_history, output_path=output_path)

            assert result.exists()
            assert result.stat().st_size > 0

    def test_plot_loss_curve_empty(self):
        """Test plotting with empty log history."""
        log_history = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.png"
            result = plot_loss_curve(log_history, output_path=output_path)

            assert result.exists()

    def test_plot_loss_curve_custom_title(self):
        """Test plotting with custom title."""
        log_history = [
            {"step": 10, "loss": 0.7},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom.png"
            result = plot_loss_curve(
                log_history, output_path=output_path, title="Custom Title"
            )

            assert result.exists()


class TestPlotLRSchedule:
    """Test cases for plot_lr_schedule function."""

    def test_plot_lr_schedule_basic(self):
        """Test plotting with basic LR data."""
        log_history = [
            {"step": 10, "learning_rate": 5e-5},
            {"step": 20, "learning_rate": 4e-5},
            {"step": 30, "learning_rate": 3e-5},
            {"step": 40, "learning_rate": 2e-5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lr.png"
            result = plot_lr_schedule(log_history, output_path=output_path)

            assert result.exists()
            assert result.stat().st_size > 0
            assert result.suffix == ".png"

    def test_plot_lr_schedule_log_scale(self):
        """Test that log scale is used for large LR range."""
        log_history = [
            {"step": 10, "learning_rate": 1e-2},
            {"step": 100, "learning_rate": 1e-5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lr_log.png"
            result = plot_lr_schedule(log_history, output_path=output_path)

            assert result.exists()
            assert result.stat().st_size > 0

    def test_plot_lr_schedule_empty(self):
        """Test plotting with empty log history."""
        log_history = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lr_empty.png"
            result = plot_lr_schedule(log_history, output_path=output_path)

            assert result.exists()

    def test_plot_lr_schedule_no_lr_key(self):
        """Test plotting when log entries lack learning_rate key."""
        log_history = [
            {"step": 10, "loss": 0.7},
            {"step": 20, "loss": 0.5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lr_missing.png"
            result = plot_lr_schedule(log_history, output_path=output_path)

            assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
