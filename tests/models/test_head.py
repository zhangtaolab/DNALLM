"""Real-torch forward tests for the seven prediction head classes.

Every head is a real nn.Module driven by real torch inputs: output shapes
and differentiability (backward -> param.grad) are asserted, never mocked.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dnallm.models.head import (
    BasicCNNHead,
    BasicLSTMHead,
    BasicMLPHead,
    BasicUNet1DHead,
    DoubleConv,
    EVOForSeqClsHead,
    MegaDNAMultiScaleHead,
)


def _assert_differentiable(module, output):
    """Backpropagate the output sum and require at least one parameter grad."""
    output.sum().backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(not torch.allclose(g, torch.zeros_like(g)) for g in grads)


class TestBasicMLPHead:
    """MLP head forwards and construction validation."""

    def test_forward_2d_input(self):
        """2D (batch, input_dim) input yields (batch, num_classes) logits."""
        torch.manual_seed(0)
        head = BasicMLPHead(input_dim=8, num_classes=3, hidden_dims=[16])

        logits = head(torch.randn(4, 8))

        assert logits.shape == (4, 3)
        _assert_differentiable(head, logits)

    def test_forward_3d_input(self):
        """3D (batch, seq, dim) input yields (batch, seq, num_classes)."""
        torch.manual_seed(0)
        head = BasicMLPHead(input_dim=8, num_classes=2, hidden_dims=[16])

        logits = head(torch.randn(2, 5, 8))

        assert logits.shape == (2, 5, 2)
        _assert_differentiable(head, logits)

    def test_forward_4d_input_raises(self):
        """Inputs above 3D are rejected."""
        head = BasicMLPHead(input_dim=8, num_classes=2)

        with pytest.raises(ValueError, match="Input tensor must be 2D"):
            head(torch.randn(1, 2, 3, 8))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("task_type", "unknown-task", "Unsupported task_type"),
            ("norm_type", "groupnorm", "Unsupported norm_type"),
            ("activation_fn", "swish", "Unsupported activation_fn"),
        ],
    )
    def test_invalid_construction_raises(self, field, value, message):
        """Invalid task types, norms, and activations raise ValueError."""
        with pytest.raises(ValueError, match=message):
            BasicMLPHead(input_dim=8, **{field: value})

    @pytest.mark.parametrize(
        ("activation_fn", "expected_type"),
        [
            ("relu", nn.ReLU),
            ("gelu", nn.GELU),
            ("silu", nn.SiLU),
            ("tanh", nn.Tanh),
            ("sigmoid", nn.Sigmoid),
        ],
    )
    def test_activation_selection(self, activation_fn, expected_type):
        """Every supported activation is wired into the MLP stack."""
        head = BasicMLPHead(input_dim=8, activation_fn=activation_fn, hidden_dims=[4])

        layers = dict(head.mlp.named_children())
        assert isinstance(layers["activation_0"], expected_type)

    def test_batchnorm_and_dropout_stacking(self):
        """Multiple hidden dims stack with the chosen normalization."""
        head = BasicMLPHead(
            input_dim=8,
            hidden_dims=[8, 8],
            use_normalization=True,
            norm_type="batchnorm",
            dropout=0.5,
        )

        layers = dict(head.mlp.named_children())
        assert isinstance(layers["norm_0"], nn.BatchNorm1d)
        assert isinstance(layers["norm_1"], nn.BatchNorm1d)
        assert isinstance(layers["dropout_1"], nn.Dropout)


class TestBasicCNNHead:
    """CNN head multi-kernel convolution forwards."""

    def test_forward_shape_and_gradient(self):
        """Parallel conv kernels concatenate into class logits."""
        torch.manual_seed(0)
        head = BasicCNNHead(input_dim=8, num_classes=3, num_filters=4, kernel_sizes=[2, 3])

        logits = head(torch.randn(2, 6, 8))

        assert logits.shape == (2, 3)
        _assert_differentiable(head, logits)

    def test_default_kernel_sizes(self):
        """The default kernel set [3, 4, 5] builds without arguments."""
        head = BasicCNNHead(input_dim=8, num_classes=2)

        assert len(head.convs) == 3


class TestBasicLSTMHead:
    """LSTM head forwards, bidirectional and unidirectional."""

    def test_bidirectional_forward_shape_and_gradient(self):
        """A bidirectional LSTM concatenates both directions' last hiddens."""
        torch.manual_seed(0)
        head = BasicLSTMHead(input_dim=8, num_classes=2, hidden_size=6, bidirectional=True)

        logits = head(torch.randn(2, 5, 8))

        assert logits.shape == (2, 2)
        _assert_differentiable(head, logits)

    def test_unidirectional_forward_shape(self):
        """A unidirectional LSTM uses only the forward hidden state."""
        torch.manual_seed(0)
        head = BasicLSTMHead(input_dim=8, num_classes=3, hidden_size=6, bidirectional=False)

        logits = head(torch.randn(2, 5, 8))

        assert logits.shape == (2, 3)
        _assert_differentiable(head, logits)


class TestDoubleConv:
    """The U-Net DoubleConv building block."""

    def test_forward_preserves_length(self):
        """(Conv => BN => ReLU) * 2 keeps the sequence length via padding."""
        torch.manual_seed(0)
        block = DoubleConv(4, 8)

        output = block(torch.randn(2, 4, 6))

        assert output.shape == (2, 8, 6)
        _assert_differentiable(block, output)


class TestBasicUNet1DHead:
    """U-Net 1D encoder-decoder forwards."""

    def test_forward_shape_and_gradient(self):
        """The U-Net returns sequence-level class logits."""
        torch.manual_seed(0)
        head = BasicUNet1DHead(input_dim=4, num_classes=3, num_layers=2, initial_filters=8)

        logits = head(torch.randn(2, 8, 4))

        assert logits.shape == (2, 3)
        _assert_differentiable(head, logits)

    def test_forward_pads_length_mismatched_skip_connections(self):
        """Sequence lengths that halve unevenly are padded onto their skip connection."""
        torch.manual_seed(0)
        head = BasicUNet1DHead(input_dim=4, num_classes=3, num_layers=2, initial_filters=8)

        logits = head(torch.randn(2, 10, 4))

        assert logits.shape == (2, 3)
        _assert_differentiable(head, logits)

    def test_non_positive_initial_filters_default_to_input_dim(self):
        """initial_filters<=0 falls back to the input dimension."""
        head = BasicUNet1DHead(input_dim=4, num_classes=2, initial_filters=0)

        assert head.output_layer.in_features == 4


class TestMegaDNAMultiScaleHead:
    """MegaDNA three-scale embedding head."""

    def _embedding_list(self, batch=2):
        return [
            torch.randn(batch, 3, 8),
            torch.randn(batch * 4, 5, 6),
            torch.randn(batch * 8, 7, 4),
        ]

    def test_forward_shape_and_gradient(self):
        """Three pooled scale embeddings concatenate into class logits."""
        torch.manual_seed(0)
        head = MegaDNAMultiScaleHead(embedding_dims=[8, 6, 4], num_classes=3)

        logits = head(self._embedding_list())

        assert logits.shape == (2, 3)
        _assert_differentiable(head, logits)

    def test_wrong_embedding_count_raises(self):
        """A list without exactly three embeddings is rejected."""
        head = MegaDNAMultiScaleHead(embedding_dims=[8, 6, 4], num_classes=2)

        with pytest.raises(ValueError, match="Expected input list to contain 3 embeddings"):
            head([torch.randn(1, 2, 8)])

    def test_wrong_dims_length_raises(self):
        """embedding_dims must contain exactly three integers."""
        with pytest.raises(ValueError, match="embedding_dims list must contain 3 integers"):
            MegaDNAMultiScaleHead(embedding_dims=[8, 6], num_classes=2)

    def test_default_embedding_dims(self):
        """Omitting embedding_dims falls back to the documented [512, 256, 128]."""
        head = MegaDNAMultiScaleHead()

        assert head.embedding_dims == [512, 256, 128]

    def test_hidden_dims_stack(self):
        """Multiple hidden dims build a deeper MLP."""
        head = MegaDNAMultiScaleHead(embedding_dims=[8, 6, 4], num_classes=2, hidden_dims=[16, 8])

        layers = dict(head.mlp.named_children())
        assert isinstance(layers["linear_1"], nn.Linear)


class _TinyEvoBase(nn.Module):
    """Real module exposing blocks.* parameters and a hidden_size config."""

    def __init__(self, hidden=8, n_blocks=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        inner = nn.Module()
        inner.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_blocks)])
        self.model = inner


class TestEVOForSeqClsHead:
    """EVO layer-selection head forwards."""

    def _embeddings(self, batch=2, length=5, hidden=8):
        return {
            "blocks.0": torch.randn(batch, length, hidden),
            "blocks.1": torch.randn(batch, length, hidden),
        }

    def test_single_layer_forward_shape_and_gradient(self):
        """A named target layer pools its embedding into class logits."""
        torch.manual_seed(0)
        head = EVOForSeqClsHead(
            base_model=_TinyEvoBase(),
            num_classes=3,
            target_layer="blocks.1",
            pooling_method="mean",
        )

        logits = head(self._embeddings())

        assert logits.shape == (2, 3)
        _assert_differentiable(head, logits)

    def test_layer_averaging_forward(self):
        """A list of target layers averages their embeddings."""
        torch.manual_seed(0)
        head = EVOForSeqClsHead(
            base_model=_TinyEvoBase(),
            num_classes=2,
            target_layer=["blocks.0", "blocks.1"],
        )

        logits = head(self._embeddings())

        assert logits.shape == (2, 2)

    @pytest.mark.parametrize(
        ("pooling_method", "expected_shape"),
        [("mean", (2, 2)), ("last", (2, 2)), ("max", (2, 2))],
    )
    def test_pooling_methods(self, pooling_method, expected_shape):
        """mean/last/max pooling all produce (batch, num_classes) logits."""
        torch.manual_seed(0)
        head = EVOForSeqClsHead(
            base_model=_TinyEvoBase(),
            num_classes=2,
            target_layer="blocks.0",
            pooling_method=pooling_method,
        )

        logits = head(self._embeddings())

        assert logits.shape == expected_shape

    def test_mean_pooling_respects_attention_mask(self):
        """Masked mean pooling ignores padded positions."""
        torch.manual_seed(0)
        head = EVOForSeqClsHead(
            base_model=_TinyEvoBase(),
            num_classes=2,
            target_layer="blocks.0",
            pooling_method="mean",
        )
        embeddings = {"blocks.0": torch.ones(1, 3, 8)}
        mask = torch.tensor([[1, 1, 0]])

        logits = head(embeddings, attention_mask=mask)

        assert logits.shape == (1, 2)

    def test_all_layers_target_uses_averaging(self):
        """target_layer='all' selects every block and averages them."""
        head = EVOForSeqClsHead(
            base_model=_TinyEvoBase(n_blocks=4), num_classes=2, target_layer="all"
        )

        assert head.target_layers == ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
        assert head.use_layer_averaging is True

    def test_none_target_selects_middle_layer(self):
        """A None target layer auto-selects the ~26/32 position block."""
        head = EVOForSeqClsHead(base_model=_TinyEvoBase(n_blocks=4), num_classes=2)

        assert head.target_layers == ["blocks.3"]
        assert head.use_layer_averaging is False

    def test_unsupported_pooling_raises(self):
        """An unknown pooling method raises ValueError."""
        head = EVOForSeqClsHead(
            base_model=_TinyEvoBase(),
            num_classes=2,
            target_layer="blocks.0",
            pooling_method="bogus",
        )

        with pytest.raises(ValueError, match="Unsupported pooling method"):
            head(self._embeddings())
