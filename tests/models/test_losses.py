"""Real-tensor tests for the loss functions in dnallm.models.losses."""

import pytest
import torch
from torch.nn import functional

from dnallm.models.losses import FocalLoss


def _bce_focal_terms(inputs, targets, alpha, gamma):
    """Independently recompute the focal-loss terms element-wise."""
    bce = functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-bce)
    return alpha * (1 - pt) ** gamma * bce


class TestFocalLossValues:
    """Hand-computed value assertions for every reduction mode."""

    @pytest.mark.parametrize(
        ("reduction", "expected"),
        [
            (
                "mean",
                _bce_focal_terms(
                    torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0]), 0.25, 2.0
                ).mean(),
            ),
            (
                "sum",
                _bce_focal_terms(
                    torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0]), 0.25, 2.0
                ).sum(),
            ),
        ],
    )
    def test_reduction_modes(self, reduction, expected):
        """mean and sum reduce the element-wise focal terms as specified."""
        inputs = torch.tensor([2.0, -1.0])
        targets = torch.tensor([1.0, 0.0])
        loss = FocalLoss(reduction=reduction)

        result = loss(inputs, targets)

        assert result == pytest.approx(expected.item(), rel=1e-6)

    def test_none_reduction_returns_elementwise(self):
        """reduction='none' returns the unreduced per-element tensor."""
        inputs = torch.tensor([2.0, -1.0, 0.5])
        targets = torch.tensor([1.0, 0.0, 1.0])
        loss = FocalLoss(reduction="none")

        result = loss(inputs, targets)

        expected = _bce_focal_terms(inputs, targets, 0.25, 2.0)
        assert result.shape == (3,)
        assert torch.allclose(result, expected)

    def test_easy_examples_downweighted(self):
        """Well-classified examples contribute near-zero focal weight."""
        loss = FocalLoss(alpha=1.0, gamma=2.0)

        easy = loss(torch.tensor([8.0]), torch.tensor([1.0]))
        hard = loss(torch.tensor([0.0]), torch.tensor([1.0]))

        assert easy.item() < hard.item()
        assert easy.item() == pytest.approx(
            _bce_focal_terms(torch.tensor([8.0]), torch.tensor([1.0]), 1.0, 2.0).item(),
            rel=1e-6,
        )

    def test_alpha_scales_the_loss(self):
        """alpha multiplies the focal loss linearly."""
        inputs = torch.tensor([0.5, -0.5])
        targets = torch.tensor([1.0, 0.0])

        loss_quarter = FocalLoss(alpha=0.25)(inputs, targets)
        loss_half = FocalLoss(alpha=0.5)(inputs, targets)

        assert loss_half.item() == pytest.approx(2.0 * loss_quarter.item(), rel=1e-6)

    def test_gamma_zero_reduces_to_alpha_bce(self):
        """gamma=0 collapses focal loss to alpha * BCEWithLogits."""
        inputs = torch.tensor([0.5, -0.5, 2.0])
        targets = torch.tensor([1.0, 0.0, 0.0])

        focal = FocalLoss(alpha=0.25, gamma=0.0)(inputs, targets)
        bce = 0.25 * functional.binary_cross_entropy_with_logits(inputs, targets)

        assert focal.item() == pytest.approx(bce.item(), rel=1e-6)

    def test_2d_inputs_supported(self):
        """Per-token 2D inputs reduce over every element."""
        inputs = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        result = FocalLoss()(inputs, targets)

        expected = _bce_focal_terms(inputs, targets, 0.25, 2.0).mean()
        assert result == pytest.approx(expected.item(), rel=1e-6)
        assert result.dim() == 0
        assert result.dtype == inputs.dtype


class TestFocalLossGradient:
    """Autograd behavior of the focal loss."""

    def test_loss_is_finite_and_non_negative(self):
        """The loss stays finite and non-negative across random inputs."""
        torch.manual_seed(0)
        inputs = torch.randn(16)
        targets = torch.randint(0, 2, (16,)).float()

        result = FocalLoss()(inputs, targets)

        assert torch.isfinite(result)
        assert result.item() >= 0

    def test_backward_produces_input_gradients(self):
        """loss.backward() yields non-None gradients on the logits."""
        inputs = torch.tensor([0.5, -0.5], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])

        loss = FocalLoss()(inputs, targets)
        loss.backward()

        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()

    def test_trainable_head_receives_gradients(self):
        """A linear head before the loss receives parameter gradients."""
        torch.manual_seed(0)
        head = torch.nn.Linear(4, 1)
        features = torch.randn(2, 4)
        targets = torch.tensor([1.0, 0.0])

        logits = head(features).squeeze(-1)
        FocalLoss()(logits, targets).backward()

        assert head.weight.grad is not None
        assert torch.any(head.weight.grad != 0)
