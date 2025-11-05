import numpy as np
import torch
import torch.nn as nn
from captum.attr import (
    LayerIntegratedGradients,
    Occlusion,
    FeatureAblation,
    LayerConductance,
    LayerDeepLift,
    NoiseTunnel,
    IntegratedGradients,
    DeepLift,
    GradientShap,
)
from transformers import PreTrainedModel, PreTrainedTokenizer
from .plot import (
    plot_attributions_token,
    plot_attributions_line,
    plot_attributions_multi,
)


# Used for LIG, DeepLIFT (Attribution to Embedding layer)
# Also used for Occlusion and FeatureAblation (perturbation-based methods)
class _CaptumWrapperInputIDs(nn.Module):
    """
    Captum wrapper, forward method accepts input_ids (LongTensor).
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Accepts input_ids.
        It assumes the model outputs an object with a .logits attribute.
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Supports ...ForSequenceClassification,
        # ...ForTokenClassification, ...ForCausalLM
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif "logits" in outputs:
            return outputs["logits"]
        else:
            raise TypeError(
                "Model output does not have a 'logits' in outputs. "
                "Ensure you are using a model with a head "
                "(e.g., AutoModelForSequenceClassification) "
                "and not a base model (e.g., AutoModel)."
            )


# Use for LayerConductance
# (when attributing to internal layers of Transformer/Mamba)
class _CaptumWrapperInputEmbeds(nn.Module):
    """
    Captum class, which forward method accepts inputs_embeds (FloatTensor).
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Accepts inputs_embeds instead of input_ids.
        """
        # Assume the model supports inputs_embeds
        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        if hasattr(outputs, "logits"):
            return outputs.logits
        elif "logits" in outputs:
            return outputs["logits"]
        else:
            raise TypeError(
                "Model output does not have a 'logits' in outputs. "
                "Ensure you are using a model with a head "
                "(e.g., AutoModelForSequenceClassification) "
                "and not a base model (e.g., AutoModel)."
            )


class DNAInterpret:
    """
    A class for interpreting DNA language models using Captum.

    Usage:
    >>> model, tokenizer = load_model_and_tokenizer(...)
    >>> interpreter = DNAInterpret(model, tokenizer)
    >>> tokens, scores = interpreter.run_lig(
    >>>     input_seq="ACGT...",
    >>>     target=1,
    >>>     task_type="seq_clf"
    >>> )
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: dict | None = None,
    ):
        """
        Initialize the interpreter.

        Args:
            model (PreTrainedModel): Model with a task head.
            tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.task_config = config["task"]
        self.pred_config = config["inference"]
        self.device = self.pred_config.device
        self.embedding_layer = None
        self.model.to(self.device)

        # Find suitable PAD token ID for baselines
        if hasattr(tokenizer, "pad_token_id"):
            if tokenizer.pad_token_id is not None:
                self.pad_token_id = tokenizer.pad_token_id
            else:
                if hasattr(tokenizer, "eos_token_id"):
                    if tokenizer.eos_token_id is not None:
                        print(
                            "Warning: tokenizer.pad_token_id is None. "
                            "Using tokenizer.eos_token_id as pad token "
                            "for baselines."
                        )
                        self.pad_token_id = tokenizer.eos_token_id
                    else:
                        print(
                            "Warning: No pad_token_id or eos_token_id "
                            "found. Using 0. This may be incorrect."
                        )
                        self.pad_token_id = 0
        else:
            if hasattr(tokenizer, "pad_token"):
                pad_token = tokenizer.pad_token
                if hasattr(tokenizer, "convert_tokens_to_ids"):
                    self.pad_token_id = tokenizer.convert_tokens_to_ids(
                        pad_token
                    )
                elif hasattr(tokenizer, "tokenize"):
                    self.pad_token_id = tokenizer.tokenize(pad_token)[0]
                elif hasattr(tokenizer, "encode"):
                    self.pad_token_id = tokenizer.encode(
                        pad_token, add_special_tokens=False
                    )[0]
                else:
                    print(
                        "Warning: Cannot determine pad_token_id. "
                        "Using 0. This may be incorrect."
                    )
                    self.pad_token_id = 0

    def _find_embedding_layer(self) -> nn.Module:
        """
        Heuristically find the model's main nn.Embedding layer.
        This is key for adapting to various architectures.
        """
        # Common paths for embedding layers in popular models
        # (BERT, DNABERT, ModernBert)
        common_paths = [
            "bert.embeddings.word_embeddings",
            "roberta.embeddings.word_embeddings",
            # (Llama, Mistral, Gemma)
            "model.embed_tokens",
            # (GPT-2)
            "transformer.wte",
            # (Mamba, Jamba)
            "backbone.embeddings",
            "embeddings",
            # (Hyena) - Assume it follows a similar pattern to 'embeddings'
        ]

        for path in common_paths:
            try:
                obj = self.model
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Embedding):
                    if self.embedding_layer != path:
                        print(f"Auto-detected embedding layer at: {path}")
                    self.embedding_layer = path
                    return obj
            except AttributeError:
                continue

        # General fallback: search all modules for nn.Embedding
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                # This is a risky guess since it might be a position embedding
                # But as a last resort
                if "position" not in name and "pos" not in name:
                    if self.embedding_layer != name:
                        print(
                            f"Warning: Fallback detection found "
                            f"an nn.Embedding: {name}. "
                            "This might be incorrect."
                        )
                    self.embedding_layer = name
                    return module

        raise RuntimeError(
            "Could not auto-detect nn.Embedding layer. "
            "Please manually pass the layer to `run_lig` "
            "via the `embedding_layer` argument."
        )

    def _get_input_tensors(
        self, sequence: str, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the DNA sequence and move to device."""
        try:
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
        except Exception:
            inputs = self.tokenizer.tokenize(sequence)
            input_ids = (
                torch.tensor(inputs, dtype=torch.long)
                .unsqueeze(0)
                .to(self.device)
            )
            attention_mask = torch.ones_like(input_ids).to(self.device)
        return input_ids, attention_mask

    def _ids_to_tokens(
        self, token_ids: int, input_seq: str | None = None
    ) -> list[str]:
        """Convert token IDs to string tokens."""
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        elif hasattr(self.tokenizer, "decode"):
            tokens = [self.tokenizer.decode(tid) for tid in token_ids]
        elif hasattr(self.tokenizer, "decode_token"):
            tokens = [self.tokenizer.decode_token(tid) for tid in token_ids]
        elif hasattr(self.tokenizer, "tokenize") and input_seq is not None:
            tokens = self.tokenizer.tokenize(input_seq)
        else:
            tokens = input_seq.split()
        return tokens

    def _get_pad_baseline(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a baseline tensor filled with PAD token IDs."""
        return torch.full_like(input_ids, self.pad_token_id).to(self.device)

    def _format_captum_target(
        self, target: int | str, token_index: int | None = None
    ) -> int | tuple[int, int]:
        """
        Formats the Captum 'target' parameter based on task type.
        """
        task_type = self.task_config.task_type
        if isinstance(target, str):
            target = self.task_config.label_names.index(target)
        if task_type in ["binary", "multiclass", "multilabel", "regression"]:
            # Sequence classification/regression: target is class index (int)
            # logits shape: [batch, num_classes]
            return target

        elif task_type == "token":
            # Token classification (NER): target is (token_index, class_index)
            # logits shape: [batch, seq_len, num_classes]
            if token_index is None:
                raise ValueError("`token_index` must be provided.")
            return (token_index, target)

        elif task_type == "generation":
            # CausalLM (Generation): target is (token_index, vocab_id)
            # logits shape: [batch, seq_len, vocab_size]
            if token_index is None:
                token_index = -1  # last token's prediction

            # At this point, 'target' is interpreted as *vocab_id*
            return (token_index, target)
        else:
            raise ValueError(f"Unknown task_type: {task_type}.")

    def run_lig(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        embedding_layer: nn.Module | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run Layer Integrated Gradients (LIG) for
        attribution to the Embedding layer.
        This is the most recommended method for k-mer importance analysis.

        Args:
            input_seq (str): input DNA sequence.
            target (int): Target for attribution.
                          - For 'seq_cls': target index (e.g., 1).
                          - for 'token_cls': target type index
                                (e.g., 2 for 'Promoter').
                          - For 'causal_lm': Target *Token ID*
                                 (e.g., 8 for 'G').
            task_type (str): Task type
            token_index (int, optional): for 'token_cls'/'causal_lm'
                Token position to explain.
            embedding_layer (nn.Module, optional): Specific Embedding layer.
                Auto-detected if None.
            max_length (int): Max token length for tokenizer.
            **kwargs: Extra args for
                captum.attr.LayerIntegratedGradients.attribute
                (for example: internal_batch_size=4).

        Returns:
            Tuple[List[str], np.ndarray]: tokens list, attribution scores array
        """
        # 1. Use wrapper that accepts input_ids
        wrapper = _CaptumWrapperInputIDs(self.model)

        # 2. Find or use specified Embedding layer
        if embedding_layer is None:
            embedding_layer = self._find_embedding_layer()

        lig = LayerIntegratedGradients(wrapper, embedding_layer)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 3. Prepare inputs and baselines
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )
        baseline_input_ids = self._get_pad_baseline(input_ids)

        # 4. Format Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. Compute attributions
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs,
        )

        # 6. Process results
        # Shape: (batch, seq_len, embed_dim) -> (seq_len)
        attr_scores = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attr_scores.cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def run_deeplift(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        embedding_layer: nn.Module | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run DeepLIFT for attribution to Embedding layer.
        """
        wrapper = _CaptumWrapperInputIDs(self.model)
        if embedding_layer is None:
            embedding_layer = self._find_embedding_layer()

        # Use LayerDeepLift
        ldl = LayerDeepLift(wrapper, embedding_layer)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )
        baseline_input_ids = self._get_pad_baseline(input_ids)
        captum_target = self._format_captum_target(target, token_index)

        attributions = ldl.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs,
        )

        attr_scores = (
            attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        )

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def run_gradshap(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        max_length: int | None = None,
        n_samples: int = 5,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run GradientSHAP at the Embedding layer.
        Attention: GradientSHAP can be slow.

        Args:
            ...
            n_samples (int): Number of samples to draw from the baseline.
        """
        # 1. Use wrapper that accepts inputs_embeds
        wrapper = _CaptumWrapperInputEmbeds(self.model)

        # 2. Use basic GradientShap (non-layer version)
        gs = GradientShap(wrapper)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 3. Prepare Embeddings (the same as LayerConductance)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )
        baseline_input_ids = self._get_pad_baseline(input_ids)
        embedding_layer = self._find_embedding_layer()
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_input_ids)

        # 4. Format Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. Call attribute at the embeds level
        attributions = gs.attribute(
            inputs=inputs_embeds,  # Pass in embeds
            baselines=baseline_embeds,  # Pass in baseline embeds
            n_samples=n_samples,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs,
        )

        # 6. Process results (shape: (batch, seq_len, embed_dim) -> (seq_len))
        attr_scores = (
            attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        )

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def run_occlusion(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        max_length: int | None = None,
        sliding_window_shapes: tuple[int] = (1,),
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run Occlusion (perturbation-based method).
        Attention: This method can be very slow.

        Args:
            ... (Args similar to run_lig) ...
            sliding_window_shapes (Tuple[int]):
                Size of the occlusion window,
                (1,) means occluding 1 token at a time.

        Returns:
            Tuple[List[str], np.ndarray]: tokens list, attribution scores array
        """
        # 1. Use wrapper that accepts input_ids
        wrapper = _CaptumWrapperInputIDs(self.model)
        occlusion = Occlusion(wrapper)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 2. Prepare inputs
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )

        # 3. Occlusion baseline is a single PAD token ID (scalar)
        baselines = self.pad_token_id

        # 4. Format Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. Compute attributions
        attributions = occlusion.attribute(
            inputs=input_ids,
            sliding_window_shapes=sliding_window_shapes,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            baselines=baselines,
            **kwargs,
        )

        # 6. Process results
        # Shape: (batch, seq_len) -> (seq_len)
        attr_scores = attributions.squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def run_feature_ablation(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run Feature Ablation (perturbation-based method).

        Args:
            ... (Args similar to run_occlusion) ...

        Returns:
            Tuple[List[str], np.ndarray]: tokens list, attribution scores array
        """
        # 1. Use wrapper that accepts input_ids
        wrapper = _CaptumWrapperInputIDs(self.model)
        ablation = FeatureAblation(wrapper)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 2. Prepare inputs and baselines
        # (FeatureAblation requires a tensor baseline)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )
        baselines = self._get_pad_baseline(input_ids)

        # 3. Format Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 4. FeatureAblation requires feature_mask to define features
        # (default one feature per token)
        # Shape: (batch_size, num_features) -> (1, seq_len)
        feature_mask = (
            torch.arange(input_ids.shape[1]).unsqueeze(0).to(self.device)
        )

        # 5. Compute attributions
        attributions = ablation.attribute(
            inputs=input_ids,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            baselines=baselines,
            feature_mask=feature_mask,
            **kwargs,
        )

        # 6. Process results
        # Shape: (batch, seq_len) -> (seq_len)
        attr_scores = attributions.squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def run_layer_conductance(
        self,
        input_seq: str,
        target: int | str,
        target_layer: nn.Module,
        token_index: int | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run Layer Conductance for attribution to an internal layer.

        Important:
        1. This method assumes self.model.forward supports 'inputs_embeds'.
        2. You must manually pass in 'target_layer'.

        Args:
            ... (Args similar to run_lig) ...
            target_layer (nn.Module): internal layer to analyze
                                     (such as: `model.bert.encoder.layer[-1]`).

        Returns:
            Tuple[List[str], np.ndarray]: tokens list, attribution scores array
        """
        # 1. Use wrapper that accepts inputs_embeds
        wrapper = _CaptumWrapperInputEmbeds(self.model)
        lc = LayerConductance(wrapper, target_layer)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 2. Prepare inputs (input_ids)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )
        baseline_input_ids = self._get_pad_baseline(input_ids)

        # 3. Manually convert IDs to Embeddings
        embedding_layer = self._find_embedding_layer()
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_input_ids)

        # 4. Format Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. Compute attributions (inputs are embeds)
        attributions_tuple = lc.attribute(
            inputs=inputs_embeds,  # <--- Pass in embeds
            baselines=baseline_embeds,  # <--- Pass in embeds
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs,
        )

        # 6. Process results
        if isinstance(attributions_tuple, tuple):
            attributions = attributions_tuple[0]
        else:
            attributions = attributions_tuple
        # Shape: (batch, seq_len, hidden_dim) -> (seq_len)
        attr_scores = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attr_scores.cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def run_noise_tunnel(
        self,
        input_seq: str,
        target: int | str,
        base_method: str,
        token_index: int | None = None,
        max_length: int | None = None,
        nt_type: str = "smoothgrad",
        nt_samples: int = 5,
        nt_stdevs: float = 0.1,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        Run NoiseTunnel (e.g., SmoothGrad) for smoother attributions.
        This will run at the 'inputs_embeds' level to avoid
        type conflicts with 'nn.Embedding'.

        Args:
            ...
            base_method (str): The base attribution method to use.
                               Support:
                                 'lig' (IntegratedGradients),
                                 'deeplift' (DeepLift),
                                 'gradshap' (GradientShap).
            nt_type (str): 'smoothgrad'(default), 'smoothgrad_sq' or 'vargrad'.
            nt_samples (int): The number of noise samples.
            nt_stdevs (float): The standard deviation of the noise.
        """
        print(
            f"Running NoiseTunnel ({nt_type}) "
            f"with base method: {base_method}..."
        )

        # 1. Use wrapper that accepts inputs_embeds
        wrapper = _CaptumWrapperInputEmbeds(self.model)

        # 2. Select base attribution method (non-Layer version)
        if base_method.lower() == "lig":
            attr_method = IntegratedGradients(wrapper)
        elif base_method.lower() == "deeplift":
            attr_method = DeepLift(wrapper)
        elif base_method.lower() == "gradshap":
            attr_method = GradientShap(wrapper)
        else:
            raise ValueError(
                f"Unknown base_method: {base_method}. "
                "Supported: 'lig', 'deeplift', 'gradshap'"
            )

        # 3. Wrap with NoiseTunnel
        nt = NoiseTunnel(attr_method)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 4. Prepare Embeddings (same as LayerConductance)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length
        )
        baseline_input_ids = self._get_pad_baseline(input_ids)
        embedding_layer = self._find_embedding_layer()
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_input_ids)

        # 5. Format Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 6. Prepare attribution parameters
        attr_kwargs = {
            "target": captum_target,
            "additional_forward_args": (attention_mask,),
            "nt_type": nt_type,
            "nt_samples": nt_samples,
            "stdevs": nt_stdevs,
            **kwargs,
        }

        # 7. Call attribute method
        # GradientShap need n_samples and baselines (distribution)
        if base_method.lower() == "gradshap":
            attributions = nt.attribute(
                inputs=inputs_embeds,
                baselines=baseline_embeds,
                n_samples=nt_samples,  # gradshap own n_samples
                **attr_kwargs,
            )
        else:  # LIG or DeepLIFT
            attributions = nt.attribute(
                inputs=inputs_embeds, baselines=baseline_embeds, **attr_kwargs
            )

        # 8. Process results
        # Shape: (batch, seq_len, hidden_dim) -> (seq_len)
        attr_scores = (
            attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        )

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0), input_seq=input_seq
        )
        return tokens, attr_scores

    def interpret(
        self,
        input_seq: str,
        method: str,
        target: int | str,
        token_index: int | None = None,
        target_layer: nn.Module | None = None,
        max_length: int | None = None,
        plot: bool = True,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """
        A unified interpretation interface that runs
        the specified attribution method.

        Args:
            input_seq (str): Input DNA sequence.
            method (str): Attribution method name.
                Support:
                    'lig', 'deeplift', 'gradshap',
                    'occlusion', 'feature_ablation',
                    'layer_conductance', 'noise_tunnel'.
            target (int): Target for attribution.
            token_index (int, optional): For 'token_cls'/'causal_lm'
                token position to explain.
            target_layer (nn.Module, optional): for 'layer_conductance',
                internal layer to analyze.
            max_length (int, optional): Max token length for tokenizer.
            plot (bool): Whether to store attributions for plotting.
            **kwargs: Extra args for specific attribution methods.

        Returns:
            Tuple[List[str], np.ndarray]: tokens list, attribution scores array
        """
        method = method.lower()
        if method == "lig":
            tokens, attr_scores = self.run_lig(
                input_seq, target, token_index, max_length=max_length, **kwargs
            )
        elif method == "deeplift":
            tokens, attr_scores = self.run_deeplift(
                input_seq, target, token_index, max_length=max_length, **kwargs
            )
        elif method == "gradshap":
            tokens, attr_scores = self.run_gradshap(
                input_seq, target, token_index, max_length=max_length, **kwargs
            )
        elif method == "occlusion":
            tokens, attr_scores = self.run_occlusion(
                input_seq, target, token_index, max_length=max_length, **kwargs
            )
        elif method == "feature_ablation":
            tokens, attr_scores = self.run_feature_ablation(
                input_seq, target, token_index, max_length=max_length, **kwargs
            )
        elif method == "layer_conductance":
            if target_layer is None:
                raise ValueError(
                    "`target_layer` must be provided for "
                    "`layer_conductance` method."
                )
            tokens, attr_scores = self.run_layer_conductance(
                input_seq,
                target,
                target_layer,
                token_index,
                max_length=max_length,
                **kwargs,
            )
        elif method == "noise_tunnel":
            tokens, attr_scores = self.run_noise_tunnel(
                input_seq, target, token_index, max_length=max_length, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Supported methods: 'lig', 'deeplift', "
                "'gradshap', 'occlusion', 'feature_ablation', "
                "'layer_conductance', 'noise_tunnel'."
            )
        if plot:
            self.attributions = (tokens, attr_scores)
        else:
            self.attributions = None

        return tokens, attr_scores

    def batch_interpret(
        self,
        input_seqs: list[str],
        method: str,
        targets: list[int],
        token_indices: list[int] | None = None,
        target_layers: list[nn.Module] | None = None,
        max_length: int | None = None,
        plot: bool = True,
        **kwargs,
    ) -> list[tuple[list[str], np.ndarray]]:
        """
        Batch interpret multiple sequences.

        Args:
            input_seqs (List[str]): List of input DNA sequences.
            method (str): Attribution method name.
            targets (List[int]):
                List of target class indices for each sequence.
            token_indices (List[int], optional):
                Each sequence's token_index list.
            target_layers (List[nn.Module], optional):
                Each sequence's target_layer list.
            max_length (int, optional): Max token length for tokenizer.
            plot (bool): Whether to store attributions for plotting.
            **kwargs: Extra args for specific attribution methods.

        Returns:
            List[Tuple[List[str], np.ndarray]]:
                List of (tokens list, attribution scores array) tuples.
        """
        results = []
        for i, seq in enumerate(input_seqs):
            token_index = None
            if token_indices is not None:
                token_index = token_indices[i]
            target_layer = None
            if target_layers is not None:
                target_layer = target_layers[i]
            tokens, scores = self.interpret(
                input_seq=seq,
                method=method,
                target=targets[i],
                token_index=token_index,
                target_layer=target_layer,
                max_length=max_length,
                **kwargs,
            )
            results.append((tokens, scores))
        if plot:
            self.attributions = results
        else:
            self.attributions = None

        return results

    def plot_attributions(self, plot_type: str = "token", **kwargs):
        """
        Plot the attributions using specified plot type.

        Args:
            plot_type (str): Plot type, 'token' (default), 'line', or 'multi'.
            **kwargs: Extra parameters for specific plotting functions.
        """
        if self.attributions is None:
            raise RuntimeError(
                "No attributions found. "
                "Please run `interpret` or `batch_interpret` "
                "with `plot=True` first."
            )

        plot_type = plot_type.lower()
        if isinstance(self.attributions, list):
            if plot_type != "multi":
                print(
                    "Warning: Multiple attributions found, "
                    "falling back to 'multi' plot."
                )
            # Multiple sequences' attributions
            plot = plot_attributions_multi(self.attributions, **kwargs)
        elif plot_type == "token":
            # Single sequence's token attribution plot
            tokens, scores = self.attributions
            plot = plot_attributions_token(tokens, scores, **kwargs)
        elif plot_type == "line":
            # Single sequence's line attribution plot
            tokens, scores = self.attributions
            plot = plot_attributions_line(tokens, scores, **kwargs)
        elif plot_type == "multi":
            # Multiple sequences' attributions
            plot = plot_attributions_multi(self.attributions, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot_type: {plot_type}. "
                "Supported: 'token', 'line', 'multi'."
            )
        return plot
