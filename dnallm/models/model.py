"""DNA Model loading and management utilities.

This module provides functions for downloading, loading, and
    managing DNA language models
from various sources including Hugging Face Hub, ModelScope, and local storage.
"""
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import os
import time
from glob import glob
from typing import Any
from collections import OrderedDict
from ..configuration.configs import TaskConfig
from ..utils import get_logger
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

logger = get_logger("dnallm.models.model")

# class BaseDNAModel(ABC):
#     @abstractmethod
#     def get_model(self) -> PreTrainedModel:
#         """Return the underlying transformer model"""
#         pass

#     @abstractmethod
#     def preprocess(self, sequences: list[str]) -> dict:
#         """Preprocess DNA sequences"""
#         pass


class BasicMLPHead(nn.Module):
    """
    A universal and customizable MLP model designed to be
    appended after the embedding output of models like Transformers
    to perform various downstream tasks such as classification and regression.

    Args:
        input_dim: Dimension of the input features
        num_classes: Number of output classes (for classification tasks)
        task_type: Type of task - 'binary', 'multiclass',
                   'multilabel', or 'regression'
        hidden_dims: List of hidden layer dimensions
        activation_fn: Activation function to use ('relu', 'gelu', 'silu',
                       'tanh', 'sigmoid')
        use_normalization: Whether to use normalization layers
        norm_type: Type of normalization - 'batchnorm' or 'layernorm'
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        task_type: str = "binary",
        hidden_dims: list | None = None,
        activation_fn: str = "relu",
        use_normalization: bool = True,
        norm_type: str = "layernorm",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512]
        if task_type not in [
            "binary",
            "multiclass",
            "multilabel",
            "regression",
        ]:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if norm_type not in ["batchnorm", "layernorm"]:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task_type = task_type
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        activation_layer = activations.get(activation_fn.lower())
        if activation_layer is None:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")
        layers = []
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append((f"linear_{i}", nn.Linear(current_dim, h_dim)))
            if use_normalization:
                layers.append((
                    f"norm_{i}",
                    nn.LayerNorm(h_dim)
                    if norm_type == "layernorm"
                    else nn.BatchNorm1d(h_dim),
                ))
            layers.append((f"activation_{i}", activation_layer))
            layers.append((f"dropout_{i}", nn.Dropout(p=dropout)))
            current_dim = h_dim
        self.mlp = nn.Sequential(OrderedDict(layers))
        self.output_layer = nn.Linear(current_dim, num_classes)

    def forward(self, x: torch.Tensor):
        if x.dim() > 3:
            raise ValueError(
                "Input tensor must be 2D (batch_size, input_dim) or "
                f"3D (batch_size, seq_len, input_dim), but got {x.shape}"
            )
        is_3d = x.dim() == 3
        if is_3d:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size * seq_len, -1)

        x = self.mlp(x)
        logits = self.output_layer(x)

        if is_3d:
            logits = logits.view(batch_size, seq_len, -1)

        return logits


class BasicCNNHead(nn.Module):
    """
    A CNN-based head for processing Transformer output sequences.
    This head applies multiple 1D convolutional layers with different
    kernel sizes to capture local patterns in the sequence data,
    followed by a fully connected layer for classification or regression tasks.

    Args:
        input_dim: Dimension of the input features
        num_classes: Number of output classes (for classification tasks)
        task_type: Type of task - 'binary', 'multiclass',
                   'multilabel', or 'regression'
        hidden_dims: List of hidden layer dimensions
        activation_fn: Activation function to use ('relu', 'gelu', 'silu',
                       'tanh', 'sigmoid')
        use_normalization: Whether to use normalization layers
        norm_type: Type of normalization - 'batchnorm' or 'layernorm'
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: str = "binary",
        num_filters: int = 128,
        kernel_sizes: list | None = None,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        # Define multiple parallel 1D convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim, out_channels=num_filters, kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # CNN feature dimension is the concatenation of all conv outputs
        cnn_output_dim = num_filters * len(kernel_sizes)

        # Define the final output layer
        self.output_layer = nn.Linear(cnn_output_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x = (batch_size, sequence_length, input_dim)

        # Conv1d expects input shape (batch, channels, length)
        x = x.permute(0, 2, 1)

        # Apply convolution, activation, and pooling for each kernel
        conv_outputs = []
        for conv in self.convs:
            conv_out = nn.functional.relu(conv(x))
            # Apply max pooling to reduce the sequence dimension to 1
            pooled_out = nn.functional.max_pool1d(
                conv_out, kernel_size=conv_out.shape[2]
            ).squeeze(2)
            conv_outputs.append(pooled_out)

        # Concatenate all convolution outputs
        concatenated = torch.cat(conv_outputs, dim=1)

        # Apply dropout
        dropped = self.dropout(concatenated)

        logits = self.output_layer(dropped)

        return logits


class BasicLSTMHead(nn.Module):
    """
    A LSTM-based head for processing Transformer output sequences.
    This head applies a multi-layer LSTM to capture sequential dependencies
    in the sequence data, followed by a fully connected layer for
    classification or regression tasks.

    Args:
        input_dim: Dimension of the input features
        num_classes: Number of output classes (for classification tasks)
        task_type: Type of task - 'binary', 'multiclass',
                   'multilabel', or 'regression'
        hidden_size: Number of features in the hidden state of the LSTM
        num_layers: Number of recurrent layers in the LSTM
        dropout: Dropout probability between LSTM layers
        bidirectional: Whether to use a bidirectional LSTM
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: str = "binary",
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,  # Accepts (batch, seq, feature) shaped inputs
        )

        # LSTM output feature dimension
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # Define the final output layer
        self.output_layer = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x = (batch_size, sequence_length, input_dim)

        # LSTM output includes all_outputs, (last_hidden, last_cell)
        # We typically use the last time step's hidden state
        # as the representation for the entire sequence
        _, (hidden, _) = self.lstm(x)

        # hidden = (num_layers * num_directions,
        #           batch_size, lstm_hidden_size)
        # If bidirectional, we need to concatenate
        # the last two layers' hidden states
        if self.lstm.bidirectional:
            # Concatenate the last hidden states of
            # the forward and backward layers
            # hidden[-2,:,:] is the forward, hidden[-1,:,:] is the backward
            hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_cat = hidden[-1, :, :]

        # hidden_cat = (batch_size, lstm_output_dim)
        logits = self.output_layer(hidden_cat)

        return logits


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class BasicUNet1DHead(nn.Module):
    """
    An U-net architecture adapted for 1D sequence data, suitable for
    classification and regression tasks.
    This model consists of an encoder-decoder structure with skip connections,
    allowing it to capture both local and global features in the inputs.

    Args:
        input_dim: The number of input features (channels) in the inputs.
        num_classes: The number of output classes for the classification task.
        task_type: The type of task (e.g., "binary" or "multi-class").
        num_layers: The number of downsampling/upsampling layers in the U-net.
        initial_filters: The number of filters in the first
                         convolutional layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task_type: str = "binary",
        num_layers: int = 2,
        initial_filters: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        if initial_filters is None or initial_filters <= 0:
            initial_filters = input_dim

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # --- Encoder (downsampling path) ---
        in_c = input_dim
        out_c = initial_filters
        for _ in range(num_layers):
            self.downs.append(DoubleConv(in_c, out_c))
            in_c = out_c
            out_c *= 2

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(in_c, out_c)

        # --- Decoder (upsampling path) ---
        in_c = out_c
        out_c //= 2
        for _ in range(num_layers):
            self.ups.append(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(in_c, out_c))
            in_c = out_c
            out_c //= 2

        # --- Final output layer ---
        # After U-Net processing,
        # the number of channels becomes initial_filters
        # We perform average pooling on the enhanced sequence
        # and then pass it to the linear layer
        self.output_layer = nn.Linear(initial_filters, num_classes)

    def forward(self, x: torch.Tensor):
        # x = (batch_size, sequence_length, input_dim)
        # Conv1d = (batch, channels, length)
        x = x.permute(0, 2, 1)

        skip_connections = []

        # Downsampling
        for down_conv in self.downs:
            x = down_conv(x)
            skip_connections.append(x)
            x = nn.functional.max_pool1d(x, 2)

        x = self.bottleneck(x)

        # Upsampling
        skip_connections = skip_connections[::-1]  # reverse to match order
        for i in range(0, len(self.ups), 2):
            up_conv = self.ups[i]
            double_conv = self.ups[i + 1]
            x = up_conv(x)

            # Get the corresponding skip connection
            skip = skip_connections[i // 2]

            # Handle potential size mismatch due to pooling/convolution
            if x.shape[2] != skip.shape[2]:
                diff = skip.shape[2] - x.shape[2]
                x = nn.functional.pad(x, [diff // 2, diff - diff // 2])

            # Concatenate skip connection
            x = torch.cat([skip, x], dim=1)
            x = double_conv(x)

        # At this point, x has shape (batch, initial_filters, sequence_length)
        # Perform average pooling
        # to obtain a representation of the entire sequence
        # permute to (batch, length, channels) then mean over length
        x = x.permute(0, 2, 1)
        pooled = torch.mean(x, dim=1)  # (batch, initial_filters)

        logits = self.output_layer(pooled)

        return logits


class MegaDNAMultiScaleHead(nn.Module):
    """
    A classification head tailored for the multi-scale embedding outputs
    of the MegaDNA model.
    It takes a list of embedding tensors, pools each tensor, and concatenates
    the results before passing them to an MLP for classification.

    Args:
        embedding_dims (list | None): A list of integers representing
            the dimensions of the input embeddings.
        num_classes (int): The number of output classes for classification.
        task_type (str): The type of task (e.g., "binary" or "multi-class").
        hidden_dims (list | None): A list of integers representing
            the sizes of hidden layers in the MLP.
        dropout (float): Dropout probability for regularization.
    """

    def __init__(
        self,
        embedding_dims: list | None = None,
        num_classes: int = 2,
        task_type: str = "binary",
        hidden_dims: list | None = None,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes
        self.task_type = task_type
        if hidden_dims is None:
            hidden_dims = [256]

        # Check that embedding_dims has exactly 3 elements
        if len(embedding_dims) != 3:
            raise ValueError(
                "embedding_dims list must contain 3 integers, "
                "corresponding to the outputs of 3 scales."
            )

        concatenated_dim = sum(embedding_dims)

        # --- Create MLP layers ---
        mlp_layers = []
        current_dim = concatenated_dim
        for i, h_dim in enumerate(hidden_dims):
            mlp_layers.append((f"linear_{i}", nn.Linear(current_dim, h_dim)))
            mlp_layers.append((f"norm_{i}", nn.LayerNorm(h_dim)))
            mlp_layers.append((f"activation_{i}", nn.GELU()))
            mlp_layers.append((f"dropout_{i}", nn.Dropout(p=dropout)))
            current_dim = h_dim

        self.mlp = nn.Sequential(OrderedDict(mlp_layers))

        # --- Final output layer ---
        self.output_layer = nn.Linear(current_dim, num_classes)

    def forward(self, embedding_list: list):
        # embedding_list contains 3 tensors from MegaDNA model
        # e.g., [ [1, 2, 512], [1, 65, 256], [64, 17, 196] ]

        if len(embedding_list) != 3:
            raise ValueError(
                "Expected input list to contain 3 embeddings, "
                f"but got {len(embedding_list)}."
            )

        # 1. Average pooling on the first scale's embedding
        # [batch, seq1, dim1] -> [batch, dim1]
        pooled_emb1 = torch.mean(embedding_list[0], dim=1)

        # 2. Average pooling on the second scale's embedding
        # [batch, seq2, dim2] -> [batch, dim2]
        pooled_emb2 = torch.mean(embedding_list[1], dim=1)

        # 3. Special processing for the third scale's embedding
        # [eff_batch, seq3, dim3] -> [batch, dim3]
        # We assume eff_batch is the original batch
        # expanded along the sequence dimension
        # Therefore, we perform average pooling across all dimensions
        # except the feature dimension
        emb3 = embedding_list[2]
        # First, flatten the eff_batch and seq3 dimensions
        emb3_flat = emb3.reshape(-1, emb3.shape[-1])
        # Then, take the mean while keeping the batch dimension as 1
        # for concatenation
        #
        # We assume the original batch size is 1,
        # if the batch size is greater than 1,
        # this logic needs to be adjusted.
        # For Hugging Face Trainer (batch size > 1),
        # a more robust approach is to reshape it back to the original
        # batch size, but this requires knowing the
        # specific parameters of Rearrange.
        # Currently, global average is a reasonable approximation.
        pooled_emb3 = torch.mean(emb3_flat, dim=0, keepdim=True)
        # If the original batch_size > 1, we need to repeat this vector
        # to match the batch size
        if pooled_emb1.shape[0] > 1 and pooled_emb3.shape[0] == 1:
            pooled_emb3 = pooled_emb3.repeat(pooled_emb1.shape[0], 1)

        # 4. Concatenate the three pooled vectors
        concatenated_vector = torch.cat(
            [pooled_emb1, pooled_emb2, pooled_emb3], dim=1
        )

        # 5. Pass through MLP and output layer to get logits
        hidden_output = self.mlp(concatenated_vector)
        logits = self.output_layer(hidden_output)

        return logits


class DNALLMforSequenceClassification(PreTrainedModel):
    """
    An automated wrapper that selects an appropriate pooling strategy
    based on the underlying model architecture and appends a customizable
    MLP head for sequence classification or regression tasks.

    Args:
        model: Pre-trained transformer model (e.g., BERT, GPT)
        tokenizer: Corresponding tokenizer for the model
        mlp_head_config: Configuration dictionary for the MLP head
    """

    config_class = AutoConfig

    def __init__(self, config, custom_model=None):
        super().__init__(config)
        from transformers import AutoModel

        if self.config.head_config.get("head", "").lower() == "megadna":
            self.backbone = custom_model
            self.score = MegaDNAMultiScaleHead(**self.config.head_config)
        elif "lucaone" in self.config.head_config.get("head", "").lower():
            from lucagplm import LucaGPLMModel

            self.backbone = LucaGPLMModel(config)
            transformer_output_dim = self.config.hidden_size
            classifier = self._determine_classifier()
            self.score = classifier(
                input_dim=transformer_output_dim, **self.config.head_config
            )
        else:
            import inspect

            self.backbone = AutoModel.from_config(
                config, trust_remote_code=True
            )
            forward_signature = inspect.signature(self.backbone.forward)
            self._backbone_supported_args = set(
                forward_signature.parameters.keys()
            )
            if hasattr(self.backbone.config, "hidden_size"):
                transformer_output_dim = self.backbone.config.hidden_size
            elif hasattr(self.backbone.config, "d_model"):
                transformer_output_dim = self.backbone.config.d_model
            else:
                raise ValueError(
                    "Cannot determine transformer output dimension. "
                    "Please specify 'input_dim' in head_config."
                )
            classifier = self._determine_classifier()
            self.score = classifier(
                input_dim=transformer_output_dim, **self.config.head_config
            )
        self.num_labels = self.config.num_labels
        # determine pooling strategy if not set
        self.pooling_strategy = self._determine_pooling_strategy()
        logger.info(f"Using {self.pooling_strategy} pooling strategy.")

        # self.post_init()

    @classmethod
    def from_base_model(cls, model_name_or_path: str, config, module=None):
        """
        Handles weights diffusion when loading a model from
        a pre-trained base model.
        """
        from transformers import AutoModel

        # 1. Use config to create an instance of our custom class.
        model = cls(config)
        # 2. Load the base pre-trained model separately.
        if module is not None:
            base_model = module.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        else:
            base_model = AutoModel.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        # 3. Assign the loaded weights to our backbone.
        model.backbone.load_state_dict(base_model.state_dict())

        return model

    def _determine_classifier(self):
        if (
            hasattr(self.config.head_config, "custom_head")
            and self.config.head_config["custom_head"] is not None
        ):
            # Use the custom head class provided in the config
            classifier = self.config.head_config["custom_head"]
        elif self.config.head_config.get("head", "").lower().endswith("mlp"):
            classifier = BasicMLPHead
        elif self.config.head_config.get("head", "").lower().endswith("cnn"):
            classifier = BasicCNNHead
        elif self.config.head_config.get("head", "").lower().endswith("lstm"):
            classifier = BasicLSTMHead
        elif self.config.head_config.get("head", "").lower().endswith("unet"):
            classifier = BasicUNet1DHead
        return classifier

    def _determine_pooling_strategy(self):
        if self.config.head_config.get("pooling_strategy") is not None:
            return self.config.head_config["pooling_strategy"]
        if getattr(self.backbone.config, "is_decoder", False):
            return "last"
        if hasattr(self.config, "cls_token_id"):
            if self.config.cls_token_id is not None:
                return "cls"
        if hasattr(self.config, "cls_idx"):
            if self.config.cls_idx is not None:
                return "cls"
        logger.warning(
            "Warning: Could not determine model type, "
            "falling back to 'mean' pooling."
        )
        return "mean"

    def _get_sentence_embedding(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            expanded_mask = attention_mask.unsqueeze(-1).expand(
                last_hidden_state.size()
            )
            masked_sum = torch.sum(last_hidden_state * expanded_mask, 1)
            actual_lengths = torch.clamp(expanded_mask.sum(1), min=1e-9)
            return masked_sum / actual_lengths
        elif self.pooling_strategy == "max":
            masked_hidden_state = last_hidden_state.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), -float("inf")
            )
            return masked_hidden_state.max(dim=1).values
        elif self.pooling_strategy == "last":
            batch_size = last_hidden_state.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                batch_size, device=last_hidden_state.device
            )
            return last_hidden_state[batch_indices, sequence_lengths, :]
        elif self.pooling_strategy == "first":
            return last_hidden_state[:, 0, :]
        else:
            raise ValueError(
                "Internal error: "
                f"Unsupported pooling strategy '{self.pooling_strategy}'"
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ):
        if kwargs.get("attention_mask") is not None:
            attention_mask = kwargs.get("attention_mask")
        else:
            pad_token_id = self.backbone.config.pad_token_id
            attention_mask = (input_ids != pad_token_id).long()
        if self.config.head_config.get("head", "").lower() == "megadna":
            # convert input_ids to torch.longtensor if not already
            if not isinstance(input_ids, torch.LongTensor):
                input_ids = input_ids.long()
            outputs = self.backbone(input_ids, return_value="embedding")
            last_hidden_state = outputs
        else:
            # Keep kwargs in the backbone's forward method
            backbone_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in self._backbone_supported_args
            }
            outputs = self.backbone(
                input_ids=input_ids,
                **backbone_kwargs,
            )
            if isinstance(outputs, dict) or hasattr(
                outputs, "last_hidden_state"
            ):
                last_hidden_state = outputs.last_hidden_state
            elif "last_hidden_state" in outputs:
                last_hidden_state = outputs["last_hidden_state"]
            else:
                last_hidden_state = outputs[0]
                if isinstance(last_hidden_state, tuple):
                    last_hidden_state = last_hidden_state[-1]
        # Get sentence embedding if needed
        if self.config.head_config.get("head", "").lower().endswith("mlp"):
            sentence_embedding = self._get_sentence_embedding(
                last_hidden_state, attention_mask
            )
        else:
            sentence_embedding = last_hidden_state
        logits = self.score(sentence_embedding)

        loss = None
        if labels is not None:
            loss_fct = None
            # Allow other loss functions that user selected or provided
            if self.config.head_config.get("loss_function") is not None:
                loss_fct = self.config.head_config["loss_function"]

                class FocalLoss(nn.Module):
                    def __init__(
                        self,
                        alpha=0.25,
                        gamma=2.0,
                        reduction="mean",
                    ):
                        super().__init__()
                        self.alpha = alpha  # controls class imbalance
                        self.gamma = gamma  # focuses on hard examples
                        self.reduction = reduction

                    def forward(self, inputs, targets):
                        # Calculate Binary Cross-Entropy Loss for each sample
                        bce_loss = (
                            nn.functional.binary_cross_entropy_with_logits(
                                inputs, targets, reduction="none"
                            )
                        )
                        # Compute pt (model confidence on true class)
                        pt = torch.exp(-bce_loss)
                        # Apply the focal adjustment
                        focal_loss = (
                            self.alpha * (1 - pt) ** self.gamma * bce_loss
                        )
                        # Apply reduction (mean, sum, or no reduction)
                        if self.reduction == "mean":
                            return focal_loss.mean()
                        elif self.reduction == "sum":
                            return focal_loss.sum()
                        else:
                            return focal_loss

                if isinstance(loss_fct, str):
                    loss_fn_kwargs = self.config.head_config.get(
                        "loss_function_kwargs", {}
                    )
                    if loss_fct.lower() == "mse":
                        loss_fct = nn.MSELoss()
                    elif loss_fct.lower() == "crossentropy":
                        loss_fct = nn.CrossEntropyLoss()
                    elif loss_fct.lower() == "bce":
                        loss_fct = nn.BCELoss()
                    elif loss_fct.lower() == "bcewithlogits":
                        loss_fct = nn.BCEWithLogitsLoss()
                    elif loss_fct.lower() == "focal":
                        loss_fct = FocalLoss(**loss_fn_kwargs)
                    elif loss_fct.lower() == "poisson":
                        loss_fct = nn.PoissonNLLLoss(**loss_fn_kwargs)
                    elif loss_fct.lower() == "cosine_similarity":
                        # Cosine Similarity Loss
                        loss_fct = nn.CosineEmbeddingLoss(**loss_fn_kwargs)
                    else:
                        raise ValueError(
                            f"Unsupported loss function: {loss_fct}"
                        )
                elif isinstance(loss_fct, nn.Module):
                    pass
                else:
                    raise ValueError(
                        "Loss function must be a string or "
                        "an nn.Module instance."
                    )
            if self.score.task_type == "regression":
                loss_fct = nn.MSELoss() if loss_fct is None else loss_fct
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.score.task_type in ["binary", "multiclass"]:
                loss_fct = (
                    nn.CrossEntropyLoss() if loss_fct is None else loss_fct
                )
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.score.task_type == "multilabel":
                loss_fct = (
                    nn.BCEWithLogitsLoss() if loss_fct is None else loss_fct
                )
                loss = loss_fct(logits, labels)

        # Expected output format for Trainer
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
            attentions=outputs.attentions
            if hasattr(outputs, "attentions")
            else None,
        )


def download_model(
    model_name: str,
    downloader: Any,
    revision: str | None = None,
    max_try: int = 10,
) -> str:
    """Download a model with retry mechanism for network issues.

    In case of network issues, this function will attempt to download the model
    multiple times before giving up.

    Args:
        model_name: Name of the model to download
        downloader: Download function to use (e.g., snapshot_download)
        max_try: Maximum number of download attempts, default 10

    Returns:
        Path where the model files are stored

    Raises:
        ValueError: If model download fails after all attempts
    """
    # In case network issue, try to download multi-times
    cnt = 0
    # init download status
    status = "incomplete"
    while True:
        if cnt >= max_try:
            break
        cnt += 1
        try:
            status = downloader(model_name, revision=revision)
            if status != "incomplete":
                logger.info(f"Model files are stored in {status}")
                break
        # track the error
        except Exception as e:
            # network issue
            if "connection" in str(e):
                reason = "unstable network connection."
            # model not found in HuggingFace
            elif "not found" in str(e).lower():
                reason = "repo is not found."
                logger.debug(e)
                break
            # model not exist in ModelScope
            elif "response [404]" in str(e).lower():
                reason = "repo is not existed."
                logger.debug(e)
                break
            else:
                reason = str(e)
                if "no revision" in reason.lower():
                    revision = None
            logger.warning(f"Retry: {cnt}, Status: {status}, Reason: {reason}")
            time.sleep(1)

    if status == "incomplete":
        raise ValueError(f"Model {model_name} download failed.")

    return status


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


def is_fp8_capable() -> bool:
    """Check if the current CUDA device supports FP8 precision.

    Returns:
                True if the device supports FP8 (
            compute capability >= 9.0),
            False otherwise
    """
    major, minor = torch.cuda.get_device_capability()
    # Hopper (H100) has compute capability 9.0
    if (major, minor) >= (9, 0):
        return True
    else:
        logger.warning(
            f"Current device compute capability is {major}.{minor}, "
            "which does not support FP8."
        )
        return False


def _handle_evo2_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO2 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO2 model, None otherwise
    """
    evo_models = {
        "evo2_1b_base": "evo2-1b-8k",
        "evo2_7b_base": "evo2-7b-8k",
        "evo2_7b_262k": "evo2-7b-262k",
        "evo2_7b": "evo2-7b-1m",
        "evo2_40b_base": "evo2-40b-8k",
        "evo2_40b": "evo2-40b-1m",
        "evo2_7b_microviridae": "evo2-7b-8k",
    }

    for m in evo_models:
        if m in model_name.lower():
            try:
                from evo2 import Evo2  # pyright: ignore[reportMissingImports]
                from vortex.model.tokenizer import CharLevelTokenizer

                # Overwrite Evo2 to avoid init errors
                class CustomEvo2(Evo2):
                    def __init__(self):
                        pass

                    load_evo2_model = Evo2.load_evo2_model

            except ImportError as e:
                raise ImportError(
                    f"EVO2 package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/ArcInstitute/evo2"
                ) from e

            model_path = (
                glob(model_name + "/*.pt")[0]
                if os.path.isdir(model_name)
                else model_name
            )
            # Check the dependencies and find the correct config file
            is_fp8 = is_fp8_capable()
            has_flash_attention = is_flash_attention_capable()
            config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            if is_fp8:
                suffix2 = ""
            else:
                suffix2 = "-noFP8"
            suffix = suffix1 + suffix2 + ".yml"
            config_path = os.path.join(config_dir, evo_models[m] + suffix)
            # Load the model with the built-in method
            evo2_model = CustomEvo2()
            if source.lower() == "local":
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=model_path,
                    config_path=config_path,
                )
            else:
                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                downloaded_model_path = os.path.join(
                    downloaded_model_path, m + ".pt"
                )
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=downloaded_model_path,
                    config_path=config_path,
                )
            tokenizer = CharLevelTokenizer(512)
            evo2_model.tokenizer = tokenizer
            evo2_model._model_path = downloaded_model_path
            return evo2_model, tokenizer

    return None


def _handle_evo1_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO1 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO1 model, None otherwise
    """
    evo_models = {
        "evo-1.5-8k-base": "evo-1-131k-base",
        "evo-1-8k-base": "evo-1-8k-base",
        "evo-1-131k-base": "evo-1-131k-base",
        "evo-1-8k-crispr": "evo-1-8k-base",
        "evo-1-8k-transposon": "evo-1-8k-base",
    }

    for m in evo_models:
        if m in model_name.lower():
            try:
                import yaml
                from evo import Evo  # pyright: ignore[reportMissingImports]
                from stripedhyena.utils import dotdict
                from stripedhyena.model import StripedHyena
                from stripedhyena.tokenizer import CharLevelTokenizer

                # Overwrite Evo2 to avoid init errors
                class CustomEvo1(Evo):
                    def __init__(self):
                        self.device = None
                        self.model = None

                    def load_checkpoint(
                        self,
                        model_name: str = "evo-1-8k-base",
                        revision: str = "main",
                        config_path: str | None = None,
                        modules: dict | None = None,
                    ) -> StripedHyena:
                        autoconfig = modules["AutoConfig"]
                        automodelforcausallm = modules["AutoModelForCausalLM"]
                        # Load the model configuration and weights
                        model_config = autoconfig.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            revision=revision,
                        )
                        model_config.use_cache = True
                        model = automodelforcausallm.from_pretrained(
                            model_name,
                            config=model_config,
                            trust_remote_code=True,
                            revision=revision,
                        )
                        state_dict = model.backbone.state_dict()
                        del model
                        del model_config
                        global_config = dotdict(
                            yaml.safe_load(open(config_path))
                        )
                        model = StripedHyena(global_config)
                        model.load_state_dict(state_dict, strict=True)
                        model.to_bfloat16_except_poles_residues()
                        return model

            except ImportError as e:
                raise ImportError(
                    f"EVO-1 package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/evo-design/evo"
                ) from e

            has_flash_attention = is_flash_attention_capable()
            config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            suffix = suffix1 + ".yml"
            config_path = os.path.join(config_dir, evo_models[m] + suffix)
            # Load the model with the built-in method
            evo_model = CustomEvo1()
            revision = (
                "1.1_fix"
                if "." in model_name and source == "huggingface"
                else "main"
            )
            _, modules = _get_model_path_and_imports(
                model_name, source, revision=revision
            )
            evo_model.model = evo_model.load_checkpoint(
                model_name=model_name,
                revision=revision,
                config_path=config_path,
                modules=modules,
            )
            tokenizer = CharLevelTokenizer(512)
            evo_model.tokenizer = tokenizer
            return evo_model, tokenizer

    return None


def _handle_gpn_models(
    model_name: str, extra: str | None = None
) -> str | None:
    gpn_models = [
        "gpn-brassicales",
        "gpn-animal-promoter",
        "gpn-msa-sapiens",
        "PhyloGPN",
        "gpn-brassicales-gxa-sorghum-v1",
    ]
    if extra:
        gpn_models.append(extra)
    for m in gpn_models:
        if m in model_name:
            try:
                import gpn.model

                _ = gpn.model
            except ImportError as e:
                raise ImportError(
                    f"gpn package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/songlab-cal/gpn"
                ) from e
            return m
    return None


def _handle_lucaone_models(
    model_name: str,
    source: str,
    head_config: dict | None,
    extra: str | None = None,
) -> tuple | None:
    lucaone_models = [
        "LucaOne-default-step5.6M",
        "LucaOne-default-step17.6M",
        "LucaOne-default-step36M",
        "LucaOne-gene-step36.8M",
    ]
    if extra:
        lucaone_models.append(extra)
    for m in lucaone_models:
        if m in model_name:
            try:
                from lucagplm import (
                    LucaGPLMModel,
                    LucaGPLMTokenizer,
                    LucaGPLMConfig,
                )

                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                lucaone_tokenizer = LucaGPLMTokenizer.from_pretrained(
                    downloaded_model_path
                )
                lucaone_model = LucaGPLMModel.from_pretrained(
                    downloaded_model_path
                )
                if head_config is not None:
                    head_config = head_config.__dict__
                    head_config["pooling_strategy"] = "cls"
                    model_config = LucaGPLMConfig.from_pretrained(
                        downloaded_model_path
                    )
                    model_config.head_config = head_config
                    lucaone_model.config = model_config
                    lucaone_model.config.pad_token_id = (
                        lucaone_tokenizer.pad_token_id
                    )
                    lucaone_model = (
                        DNALLMforSequenceClassification.from_base_model(
                            downloaded_model_path,
                            config=model_config,
                            module=LucaGPLMModel,
                        )
                    )
            except ImportError as e:
                raise ImportError(
                    f"lucagplm package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/LucaOne/LucaOne"
                ) from e
            return lucaone_model, lucaone_tokenizer
    return None


def _handle_megadna_models(
    model_name: str,
    source: str,
    head_config: dict | None,
    extra: str | None = None,
) -> tuple | None:
    """Handle special case for megaDNA models."""
    megadna_models = [
        "megaDNA_updated",
        "megaDNA_variants",
        "megaDNA_finetuned",
        "megaDNA_phage_145M",
        "megaDNA_phage_78M",
        "megaDNA_phage_277M",
        "megaDNA_phage_ecoli_finetuned",
    ]
    if extra:
        megadna_models.append(extra)
    for m in megadna_models:
        if m in model_name:
            from transformers import PretrainedConfig, PreTrainedTokenizer

            class MegaDNAConfig(PretrainedConfig):
                model_type = "megadna"

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            class DNATokenizer(PreTrainedTokenizer):
                """
                This tokenizer treats each nucleotide (A, T, C, G)
                as a separate token, along with special tokens for padding,
                end-of-sequence, and unknown tokens.
                """

                # Define the vocabulary file names for saving
                vocab_files_names = {  # noqa: RUF012
                    "vocab_file": "vocab.txt"
                }
                DEFAULT_TOKENS = ("**", "#")

                def __init__(
                    self,
                    vocab_file=None,  # from_pretrained will handle this
                    pad_token=DEFAULT_TOKENS[0],
                    eos_token=DEFAULT_TOKENS[1],
                    unk_token=None,
                    **kwargs,
                ):
                    # 1. Initialize your vocabulary and mappings
                    self.vocab = [pad_token, "A", "T", "C", "G", eos_token]
                    self.token_to_id = {
                        tok: idx for idx, tok in enumerate(self.vocab)
                    }
                    self.id_to_token = dict(enumerate(self.vocab))

                    # 2. Initialize the parent class
                    super().__init__(
                        pad_token=pad_token,
                        eos_token=eos_token,
                        unk_token=unk_token,
                        **kwargs,
                    )

                @property
                def vocab_size(self) -> int:
                    return len(self.vocab)

                def get_vocab(self) -> dict[str, int]:
                    return self.token_to_id.copy()

                def _tokenize(self, text: Any, **kwargs: Any) -> list[str]:
                    return list(text)

                def _convert_token_to_id(self, token: str) -> int:
                    return self.token_to_id.get(token, self.unk_token_id)

                def _convert_id_to_token(self, index: int):
                    return self.id_to_token.get(index, self.unk_token)

                def save_vocabulary(
                    self,
                    save_directory: str,
                    filename_prefix: str | None = None,
                ) -> tuple:
                    """
                    Implement how to save your vocabulary to a file.
                    The parent class's save_pretrained method will
                    automatically call this function.
                    """
                    vocab_file_path = os.path.join(
                        save_directory, self.vocab_files_names["vocab_file"]
                    )

                    with open(
                        vocab_file_path, "w", encoding="utf-8"
                    ) as writer:
                        writer.write("\n".join(self.vocab))

                    return (vocab_file_path,)

            try:
                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                if m in "megaDNA_updated":
                    full_model_name = "megaDNA_phage_145M.pt"
                elif m in "megaDNA_variants":
                    full_model_name = "megaDNA_phage_78M.pt"
                elif m in "megaDNA_finetuned":
                    full_model_name = "megaDNA_phage_ecoli_finetuned.pt"
                else:
                    full_model_name = "megaDNA_phage_145M.pt"
                downloaded_model_path = os.path.join(
                    downloaded_model_path, full_model_name
                )
                megadna_model = torch.load(
                    downloaded_model_path, weights_only=False
                )
                megadna_tokenizer = DNATokenizer()
                if head_config is not None:
                    head_config = head_config.__dict__
                    head_config["embedding_dims"] = [512, 256, 196]
                    model_config = MegaDNAConfig()
                    model_config.head_config = head_config
                    megadna_model.config = model_config
                    pad_token_id = megadna_tokenizer.pad_token_id
                    megadna_model.config.pad_token_id = pad_token_id
                    megadna_model = DNALLMforSequenceClassification(
                        config=model_config,
                        custom_model=megadna_model,
                    )
            except ImportError as e:
                raise ImportError(
                    f"megaDNA package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/lingxusb/megaDNA"
                ) from e

            return megadna_model, megadna_tokenizer

    return None


def _handle_omnidna_models(
    model_name: str, extra: str | None = None
) -> str | None:
    """Handle special case for Omni-DNA models."""
    omnidna_models = [
        "Omni-DNA-20M",
        "Omni-DNA-60M",
        "Omni-DNA-116M",
        "Omni-DNA-300M",
        "Omni-DNA-700M",
        "Omni-DNA-1B",
    ]
    if extra:
        omnidna_models.append(extra)
    for m in omnidna_models:
        if m in model_name:
            try:
                from olmo import version

                _ = version.VERSION
            except ImportError as e:
                raise ImportError(
                    f"ai2-olmo package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://huggingface.co/zehui127/Omni-DNA-20M"
                ) from e
            return m
    return None


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
    if "DNABERT-2" not in os.path.basename(
        model_path
    ) and "DNABERT-S" not in os.path.basename(model_path):
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

        model, tokenizer = _load_model_by_task_type(*load_args)
        return model, tokenizer
    finally:
        # This block will execute no matter what,
        # ensuring the original file is restored.
        if original_content is not None and triton_file_path is not None:
            with open(triton_file_path, "w", encoding="utf-8") as f:
                f.write(original_content)


def _setup_huggingface_mirror(use_mirror: bool) -> None:
    """Configure HuggingFace mirror settings.

    Args:
        use_mirror: Whether to use HuggingFace mirror
    """
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("Using HuggingFace mirror at hf-mirror.com")
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def _get_model_path_and_imports(
    model_name: str, source: str, revision: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Get model path and import the required libraries based on source.

    Args:
        model_name: Model name or path
        source: Source to load model from (
                'local',
                'huggingface',
                'modelscope')
        revision: Specific model revision (branch, tag, commit),
                  default None

    Returns:
        Tuple of (model_path, imported_modules_dict)

    Raises:
        ValueError: If local model not found or unsupported source
    """
    source_lower = source.lower()

    if source_lower == "local":
        if not os.path.exists(model_name):
            raise ValueError(f"Model {model_name} not found locally.")
        model_path = model_name

    elif source_lower == "huggingface":
        from huggingface_hub import snapshot_download as hf_snapshot_download

        model_path = download_model(
            model_name, downloader=hf_snapshot_download, revision=revision
        )

    elif source_lower == "modelscope":
        from modelscope.hub.snapshot_download import (
            snapshot_download as ms_snapshot_download,
        )

        model_path = download_model(
            model_name, downloader=ms_snapshot_download, revision=revision
        )

        # Import ModelScope modules
        try:
            from modelscope import (
                AutoConfig,
                AutoModel,
                AutoModelForMaskedLM,
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "ModelScope is required but not available. "
                "Please install it with 'pip install modelscope'."
            ) from e

        modules = {
            "AutoConfig": AutoConfig,
            "AutoModel": AutoModel,
            "AutoModelForMaskedLM": AutoModelForMaskedLM,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSequenceClassification": (
                AutoModelForSequenceClassification
            ),
            "AutoModelForTokenClassification": AutoModelForTokenClassification,
            "AutoTokenizer": AutoTokenizer,
        }

        return model_path, modules

    else:
        raise ValueError(f"Unsupported source: {source}")

    # Import transformers modules for local and huggingface sources
    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )
    except ImportError as e:
        raise ImportError(
            "Transformers is required but not available. "
            "Please install it with 'pip install transformers'."
        ) from e

    modules = {
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": (
            AutoModelForSequenceClassification
        ),
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
    }

    return model_path, modules


def _create_label_mappings(
    task_config: TaskConfig,
) -> tuple[dict[int, str], dict[str, int]]:
    """Create label mappings for classification tasks.

    Args:
        task_config: Task configuration object

    Returns:
        Tuple of (id2label, label2id) mappings
    """
    label_names = task_config.label_names
    if label_names is None:
        # Default empty mappings for tasks without labels
        return {}, {}
    id2label = dict(enumerate(label_names))
    label2id = {label: i for i, label in enumerate(label_names)}
    return id2label, label2id


def _load_model_by_task_type(
    task_type: str,
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    modules: dict[str, Any],
    head_config: dict | None = None,
) -> tuple[Any, Any]:
    """Load model and tokenizer based on task type.

    Args:
        task_type: Type of task (mask, generation, binary, etc.)
        model_name: Model name or path
        num_labels: Number of labels for classification tasks
        id2label: ID to label mapping
        label2id: Label to ID mapping
        modules: Dictionary of imported model classes
        head_config: Additional head configuration (if any)

    Returns:
        Tuple of (model, tokenizer)
    """
    auto_tokenizer = modules["AutoTokenizer"]

    # Common tokenizer loading
    if task_type == "token":
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, trust_remote_code=True, add_prefix_space=True
        )
    else:
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    # Custom model with specific head
    if head_config is not None:
        head_config = head_config.__dict__
        base_config = modules["AutoConfig"].from_pretrained(
            model_name, trust_remote_code=True
        )
        model_config = base_config
        model_config.head_config = head_config
        if hasattr(tokenizer, "cls_token_id"):
            model_config.cls_token_id = tokenizer.cls_token_id
        if hasattr(tokenizer, "cls_idx"):
            model_config.cls_idx = tokenizer.cls_idx
        model = DNALLMforSequenceClassification.from_base_model(
            model_name, config=model_config
        )
        return model, tokenizer

    # Model loading based on task type
    if task_type == "mask":
        model = modules["AutoModelForMaskedLM"].from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )
    elif task_type == "generation":
        model = modules["AutoModelForCausalLM"].from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )
    elif task_type in ["binary", "multiclass"]:
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif task_type == "multilabel":
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif task_type == "regression":
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif task_type == "token":
        model = modules["AutoModelForTokenClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        model = modules["AutoModel"].from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )

    return model, tokenizer


def _configure_model_padding(model, tokenizer) -> None:
    """Configure model padding token if not set.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _get_device() -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str,
    task_config: TaskConfig,
    source: str = "local",
    use_mirror: bool = False,
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Load model and tokenizer from either HuggingFace or ModelScope.

    This function handles loading of various model types based on the task
        configuration,
            including sequence classification, token classification,
            masked language modeling,
        and causal language modeling.

        Args:
            model_name: Model name or path
            task_config: Task configuration object containing task type and
                label information
                    source: Source to load model and tokenizer from (
                'local',
                'huggingface',
                'modelscope'),
                default 'local'
                    use_mirror: Whether to use HuggingFace mirror (
                hf-mirror.com),
                default False

        Returns:
            Tuple containing (model, tokenizer)

        Raises:
            ValueError: If model is not found locally or loading fails
    """
    # Setup HuggingFace mirror if needed
    _setup_huggingface_mirror(use_mirror)

    # Extract task configuration
    task_type = task_config.task_type
    if hasattr(task_config, "head_config"):
        head_config = task_config.head_config
    else:
        head_config = None
    num_labels = task_config.num_labels

    # Handle special case for EVO2 models
    evo2_result = _handle_evo2_models(model_name, source)
    if evo2_result is not None:
        return evo2_result

    # Handle special case for EVO1 models
    evo1_result = _handle_evo1_models(model_name, source)
    if evo1_result is not None:
        return evo1_result

    # Handle special case for GPN models
    _ = _handle_gpn_models(model_name)

    # Handle special case for megaDNA models
    megadna_result = _handle_megadna_models(model_name, source, head_config)
    if megadna_result is not None:
        return megadna_result

    # Handle special case for LucaOne models
    lucaone_result = _handle_lucaone_models(model_name, source, head_config)
    if lucaone_result is not None:
        return lucaone_result

    # Handle special case for Omni-DNA models
    _ = _handle_omnidna_models(model_name)

    # TODO: Add more special cases if needed

    # Get model path and import required modules
    downloaded_model_path, modules = _get_model_path_and_imports(
        model_name, source, revision=revision
    )
    if hasattr(task_config, "head_config"):
        model_name = downloaded_model_path

    # Create label mappings
    id2label, label2id = _create_label_mappings(task_config)

    # Load model and tokenizer based on task type
    try:
        # Ensure num_labels is not None for classification tasks
        if num_labels is None and task_type in [
            "binary",
            "multiclass",
            "multilabel",
            "regression",
            "token",
        ]:
            raise ValueError(
                f"num_labels is required for task type "
                f"'{task_type}' but is None"
            )

        # Use default value if num_labels is None for other tasks
        safe_num_labels = num_labels if num_labels is not None else 1
        # num_labels check for non-binary classification tasks
        if task_type == "regression" and safe_num_labels != 1:
            logger.warning(
                f"Regression task typically has num_labels=1, "
                f"but got {safe_num_labels}."
            )
            safe_num_labels = 1
        elif task_type == "generation" and safe_num_labels != 0:
            logger.warning(
                f"Generation task does not require num_labels, "
                f"but got {safe_num_labels}. Setting to 0."
            )
            safe_num_labels = 0
        elif task_type == "embedding" and safe_num_labels != 0:
            logger.warning(
                f"Embedding task does not require num_labels, "
                f"but got {safe_num_labels}. Setting to 0."
            )
            safe_num_labels = 0
        if task_type not in [
            "binary",
            "regression",
            "generation",
            "embedding",
        ]:
            if safe_num_labels < 2:
                raise ValueError(
                    f"num_labels should be at least 2 for task type "
                    f"'{task_type}', but got {safe_num_labels}."
                )

        load_args = [
            task_type,
            model_name,
            safe_num_labels,
            id2label,
            label2id,
            modules,
            head_config,
        ]
        model, tokenizer = _handle_dnabert2_models(
            downloaded_model_path, load_args
        )
        if model is None or tokenizer is None:
            model, tokenizer = _load_model_by_task_type(*load_args)
        # Set model path and source attributes
        model._model_path = downloaded_model_path
        model.source = source
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}") from e

    # Configure model padding
    _configure_model_padding(model, tokenizer)
    model = model.to(_get_device())

    return model, tokenizer


def peft_forward_compatiable(model: Any) -> Any:
    """Convert base model forward to be compatiable with HF

    Args:
        model: Base model

    Returns:
        model with changed forward function
    """
    import inspect

    sig = inspect.signature(model.forward)
    accepted_forward_args = set(sig.parameters.keys())
    original_forward = model.forward

    def forward_hf(*args, **kwargs):
        return original_forward(**{
            k: v for k, v in kwargs.items() if k in accepted_forward_args
        })

    model.forward = forward_hf
    return model


def clear_model_cache(source: str = "huggingface"):
    """Remove all the cached models

    Args:
        source: Source to clear model cache from (
                'huggingface',
                'modelscope'),
            default 'huggingface'
    """
    source_lower = source.lower()
    if source_lower == "huggingface":
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/hub"
        )
    elif source_lower == "modelscope":
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache/modelscope/hub"
        )
    else:
        logger.warning(f"Unsupported source: {source}. No action taken.")
        return

    if os.path.exists(cache_dir):
        files = glob(os.path.join(cache_dir, "*"))
        for f in files:
            try:
                if os.path.isdir(f):
                    import shutil

                    shutil.rmtree(f)
                else:
                    os.remove(f)
                logger.info(f"Removed cached file/directory: {f}")
            except Exception as e:
                logger.warning(f"Failed to remove {f}: {e}")
    else:
        logger.info(
            f"No cache directory found at {cache_dir}. Nothing to clear."
        )


def load_preset_model(
    model_name: str, task_config: TaskConfig
) -> tuple[Any, Any] | int:
    """Load a preset model and tokenizer based on the task configuration.

    This function loads models from the preset model registry, which contains
    pre-configured models for various DNA analysis tasks.

    Args:
        model_name: Name or path of the model
                task_config: Task configuration object containing task type and
            label information

    Returns:
        Tuple containing (model, tokenizer) if successful, 0 if model not found

    Note:
                If the model is not found in preset models,
            the function will print a warning
                and
            return 0. Use `load_model_and_tokenizer` function for custom model
            loading.
    """
    from .modeling_auto import MODEL_INFO

    source = "modelscope"
    use_mirror = False

    # Load model and tokenizer
    try:
        preset_models = [
            preset
            for model in MODEL_INFO
            for preset in MODEL_INFO[model].get("preset", [])
        ]
    except (KeyError, TypeError):
        preset_models = []
    if model_name in MODEL_INFO:
        model_info = MODEL_INFO[model_name]
        model_name = model_info["default"]
    elif model_name in preset_models:
        pass
    else:
        logger.debug(
            f"Model {model_name} not found in preset models. "
            "Please check the model name or use "
            "`load_model_and_tokenizer` function."
        )
        return 0
    return load_model_and_tokenizer(
        model_name, task_config, source, use_mirror
    )
