import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any


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
        **kwargs: Any,
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
        num_filters: Number of filters for each convolutional layer
        kernel_sizes: List of kernel sizes for the convolutional layers
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        # e.g., [ [24, 3, 512], [48, 65, 256], [3072, 17, 196] ]

        if len(embedding_list) != 3:
            raise ValueError(
                "Expected input list to contain 3 embeddings, "
                f"but got {len(embedding_list)}."
            )

        # 1. Average pooling on the first scale's embedding
        # [batch, seq1, dim1] -> [batch, dim1]
        emb1 = embedding_list[0]
        real_batch_size = emb1.shape[0]
        pooled_emb1 = torch.mean(emb1, dim=1)

        # 2. Average pooling on the second scale's embedding
        # [batch, seq2, dim2] -> [batch, dim2]
        emb2 = embedding_list[1]
        pool2_temp = torch.mean(emb2, dim=1)
        pool2_temp = pool2_temp.view(real_batch_size, -1, pool2_temp.shape[-1])
        pooled_emb2 = torch.mean(pool2_temp, dim=1)

        # 3. Special processing for the third scale's embedding
        # [eff_batch, seq3, dim3] -> [batch, dim3]
        # We assume eff_batch is the original batch
        # expanded along the sequence dimension
        # Therefore, we perform average pooling across all dimensions
        # except the feature dimension
        emb3 = embedding_list[2]
        # First, flatten the eff_batch and seq3 dimensions
        pool3_temp = torch.mean(emb3, dim=1)
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
        pool3_temp = pool3_temp.view(real_batch_size, -1, pool3_temp.shape[-1])
        pooled_emb3 = torch.mean(pool3_temp, dim=1)
        # If the original batch_size > 1, we need to repeat this vector
        # to match the batch size
        # if pooled_emb1.shape[0] > 1 and pooled_emb3.shape[0] == 1:
        #     pooled_emb3 = pooled_emb3.repeat(pooled_emb1.shape[0], 1)

        # 4. Concatenate the three pooled vectors
        concatenated_vector = torch.cat(
            [pooled_emb1, pooled_emb2, pooled_emb3], dim=1
        )

        # 5. Pass through MLP and output layer to get logits
        hidden_output = self.mlp(concatenated_vector)
        logits = self.output_layer(hidden_output)

        return logits


class EVOForSeqClsHead(nn.Module):
    """
    A classification head tailored for the embedding outputs
    of the EVO-series model.

    Args:
        base_model: The EVO model instance providing embeddings.
        num_classes: Number of output classes for classification.
        task_type: Type of task - 'binary', 'multiclass',
                   'multilabel', or 'regression'.
        target_layer: Specific layer(s) from which to extract embeddings.
                      Can be 'all' to average all layers,
                      a list of layer names, or a single layer name.
        pooling_method: Method to pool sequence embeddings.
        dropout_prob: Dropout probability for regularization
    """

    def __init__(
        self,
        base_model: any,
        num_classes: int = 2,
        task_type: str = "binary",
        target_layer: str | list[str] | None = None,
        pooling_method: str = "mean",
        dropout_prob: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.task_type = task_type
        self.pooling_method = pooling_method

        if target_layer == "all" or target_layer is None:
            self.target_layers = []
            for name, _ in base_model.model.named_parameters():
                if name.startswith("blocks"):
                    layer = "blocks." + name.split(".")[1]
                    if layer not in self.target_layers:
                        self.target_layers.append(layer)
            if target_layer is None:
                # Find middle layer which performs better than
                # the last layer
                mid_layer = round(len(self.target_layers) * 26 / 32)
                self.target_layers = [self.target_layers[mid_layer]]
                self.use_layer_averaging = False
            else:
                self.use_layer_averaging = True

        elif isinstance(target_layer, list):
            self.target_layers = target_layer
            self.use_layer_averaging = True

        else:
            self.target_layers = [target_layer]
            self.use_layer_averaging = False

        if target_layer != "all":
            print(f"Use layers: {self.target_layers} embeddings.")

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(
        self,
        embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if self.use_layer_averaging:
            all_layers_tensor = torch.stack(
                [embeddings[name] for name in self.target_layers], dim=0
            )
            sequence_output = torch.mean(all_layers_tensor, dim=0)
        else:
            sequence_output = embeddings[self.target_layers[0]]

        if self.pooling_method == "last":
            pooled_output = sequence_output[:, -1, :]

        elif self.pooling_method == "mean":
            if attention_mask is not None:
                mask_expanded = (
                    attention_mask
                    .unsqueeze(-1)
                    .expand(sequence_output.size())
                    .float()
                )
                sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(sequence_output, dim=1)

        elif self.pooling_method == "max":
            pooled_output, _ = torch.max(sequence_output, dim=1)

        else:
            raise ValueError(
                f"Unsupported pooling method: {self.pooling_method}"
            )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
