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
    GradientShap
)
from transformers import PreTrainedModel, PreTrainedTokenizer
from .plot import (plot_attributions_token, plot_attributions_line,
                   plot_attributions_multi)


# --- 内部包装器 ---
# 包装器 1: 接受 input_ids
# 用于 LayerIntegratedGradients (归因于Embedding层时)
# 用于 Occlusion 和 FeatureAblation (基于扰动的方法)
class _CaptumWrapperInputIDs(nn.Module):
    """
    Captum 包装器，forward 方法接受 input_ids (LongTensor)。
    """
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        这个 'forward' 接受 input_ids。
        它假定模型输出一个带有 .logits 属性的对象。
        """
        # 传递给Hugging Face模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 适用于 ...ForSequenceClassification,
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


# 包装器 2: 接受 inputs_embeds
# 用于 LayerConductance (归因于Transformer/Mamba内部层时)
class _CaptumWrapperInputEmbeds(nn.Module):
    """
    Captum 包装器，forward 方法接受 inputs_embeds (FloatTensor)。
    """
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self,
                inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        这个 'forward' 接受 inputs_embeds。
        """
        # 假定模型支持 inputs_embeds
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
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


# --- 主解释类 ---
class DNAInterpret:
    """
    DNA大语言模型可解释性工具包，集成了Captum。

    使用方法:
    >>> model, tokenizer = load_model_and_tokenizer(...)
    >>> interpreter = DNAInterpret(model, tokenizer)
    >>> tokens, scores = interpreter.run_lig(
    >>>     input_seq="ACGT...",
    >>>     target=1,
    >>>     task_type="seq_clf"
    >>> )
    """
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: dict | None = None):
        """
        初始化解释器。

        Args:
            model (PreTrainedModel): 已经加载的、带有任务头（task head）的Hugging Face模型。
            tokenizer (PreTrainedTokenizer): 已经加载的tokenizer。
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.task_config = config["task"]
        self.pred_config = config["inference"]
        self.device = self.pred_config.device
        self.embedding_layer = None
        self.model.to(self.device)

        # 寻找合适的 PAD token ID 作为基线
        if hasattr(tokenizer, "pad_token_id"):
            if tokenizer.pad_token_id is not None:
                self.pad_token_id = tokenizer.pad_token_id
            else:
                if hasattr(tokenizer, "eos_token_id"):
                    if tokenizer.eos_token_id is not None:
                        print("Warning: tokenizer.pad_token_id is None. "
                              "Using tokenizer.eos_token_id as pad token "
                              "for baselines.")
                        self.pad_token_id = tokenizer.eos_token_id
                    else:
                        print("Warning: No pad_token_id or eos_token_id "
                              "found. Using 0. This may be incorrect.")
                        self.pad_token_id = 0
        else:
            if hasattr(tokenizer, "pad_token"):
                pad_token = tokenizer.pad_token
                if hasattr(tokenizer, "convert_tokens_to_ids"):
                    self.pad_token_id = (
                        tokenizer.convert_tokens_to_ids(pad_token)
                    )
                elif hasattr(tokenizer, "tokenize"):
                    self.pad_token_id = (
                        tokenizer.tokenize(pad_token)[0]
                    )
                elif hasattr(tokenizer, "encode"):
                    self.pad_token_id = (
                        tokenizer.encode(
                            pad_token, add_special_tokens=False
                        )[0]
                    )
                else:
                    print("Warning: Cannot determine pad_token_id. "
                          "Using 0. This may be incorrect.")
                    self.pad_token_id = 0

    # --- 内部辅助函数 ---
    def _find_embedding_layer(self) -> nn.Module:
        """
        启发式地自动查找模型的主 nn.Embedding 层。
        这是适配多种架构的关键。
        """
        # 常见模型的 Embedding 层路径
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
            # (Hyena) - 假设它遵循类似 'embeddings' 的模式
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

        # 如果常见路径都失败了，进行一次通用搜索
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                # 这是一个有风险的猜测，因为它可能是位置嵌入
                # 但作为最后的手段
                if "position" not in name and "pos" not in name:
                    if self.embedding_layer != name:
                        print(f"Warning: Fallback detection found "
                              f"an nn.Embedding: {name}. "
                              "This might be incorrect.")
                    self.embedding_layer = name
                    return module

        raise RuntimeError(
            "Could not auto-detect nn.Embedding layer. "
            "Please manually pass the layer to `run_lig` "
            "via the `embedding_layer` argument."
        )

    def _get_input_tensors(self,
                           sequence: str,
                           max_length: int
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """将DNA序列tokenize并移动到设备。"""
        try:
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"  # 填充对于创建一致的基线很重要
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
        except Exception:
            inputs = self.tokenizer.tokenize(sequence)
            input_ids = torch.tensor(
                inputs, dtype=torch.long
            ).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(
                input_ids
            ).to(self.device)
        return input_ids, attention_mask

    def _ids_to_tokens(self,
                       token_ids: int,
                       input_seq: str | None = None) -> list[str]:
        """将 token IDs 转换为字符串 tokens。"""
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
        """创建全PAD的基线张量 (用于 LIG, FeatureAblation)。"""
        return torch.full_like(input_ids, self.pad_token_id).to(self.device)

    def _format_captum_target(
        self,
        target: int | str,
        token_index: int | None = None
    ) -> int | tuple[int, int]:
        """
        根据任务类型格式化 Captum 的 'target' 参数。
        """
        task_type = self.task_config.task_type
        if isinstance(target, str):
            target = self.task_config.label_names.index(target)
        if task_type in ["binary", "multiclass", "multilabel", "regression"]:
            # 序列分类 (或回归): target 是类别索引 (int)
            # logits 形状: [batch, num_classes]
            return target

        elif task_type == "token":
            # Token 分类 (NER): target 是 (token_index, class_index)
            # logits 形状: [batch, seq_len, num_classes]
            if token_index is None:
                raise ValueError("`token_index` must be provided.")
            return (token_index, target)

        elif task_type == "generation":
            # CausalLM (生成): target 是 (token_index, vocab_id)
            # logits 形状: [batch, seq_len, vocab_size]
            if token_index is None:
                token_index = -1  # 默认解释最后一个token的预测

            # 此时 'target' 被解释为 *词汇表ID*
            return (token_index, target)
        else:
            raise ValueError(f"Unknown task_type: {task_type}.")

    # --- 公共归因方法 ---

    def run_lig(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        embedding_layer: nn.Module | None = None,
        max_length: int | None = None ,
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 Layer Integrated Gradients (LIG)，归因于 Embedding 层。
        这是最推荐的 k-mer 重要性分析方法。

        Args:
            input_seq (str): 输入的DNA序列。
            target (int): 目标类别索引。
                          - 对于 'seq_clf': 目标类的索引 (e.g., 1)。
                          - 对于 'token_clf': 目标类的索引 (e.g., 2 for 'Promoter')。
                          - 对于 'causal_lm': 目标 *词汇表ID* (e.g., 8 for 'G')。
            task_type (str): 任务类型 ('seq_clf', 'token_clf', 'causal_lm')。
            token_index (int, optional): 对于 'token_clf'/'causal_lm'
                要解释的token位置。
            embedding_layer (nn.Module, optional): 手动指定Embedding层。
                如果为None，则自动查找。
            max_length (int): Tokenizer的最大长度。
            **kwargs: 传递给 captum.attr.LayerIntegratedGradients.attribute 的额外参数
                      (例如: internal_batch_size=4)。

        Returns:
            Tuple[List[str], np.ndarray]: (tokens 列表, 归因分数数组)
        """
        # 1. 使用接受 input_ids 的包装器
        wrapper = _CaptumWrapperInputIDs(self.model)

        # 2. 查找或使用指定的 Embedding 层
        if embedding_layer is None:
            embedding_layer = self._find_embedding_layer()

        lig = LayerIntegratedGradients(wrapper, embedding_layer)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 3. 准备输入和基线
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)
        baseline_input_ids = self._get_pad_baseline(input_ids)

        # 4. 格式化 Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. 计算归因
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs
        )

        # 6. 处理结果
        # 形状: (batch, seq_len, embed_dim) -> (seq_len)
        attr_scores = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attr_scores.cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def run_deeplift(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        embedding_layer: nn.Module | None = None,
        max_length: int | None = None ,
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 Layer DeepLIFT，归因于 Embedding 层。
        """
        wrapper = _CaptumWrapperInputIDs(self.model)
        if embedding_layer is None:
            embedding_layer = self._find_embedding_layer()

        # 使用 LayerDeepLift
        ldl = LayerDeepLift(wrapper, embedding_layer)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)
        baseline_input_ids = self._get_pad_baseline(input_ids)
        captum_target = self._format_captum_target(target, token_index)

        attributions = ldl.attribute(
            inputs=input_ids, baselines=baseline_input_ids,
            target=captum_target, additional_forward_args=(attention_mask,),
            **kwargs
        )

        attr_scores = attributions.sum(
            dim=-1
        ).squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def run_gradshap(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        max_length: int | None = None ,
        n_samples: int = 5,
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 GradientSHAP，归因于 Embedding 层。
        注意: GradientSHAP 速度较慢。

        Args:
            ...
            n_samples (int): 从基线中采样的数量。
        """
        # 1. 使用接受 inputs_embeds 的包装器
        wrapper = _CaptumWrapperInputEmbeds(self.model)

        # 2. 使用基础的 GradientShap (非 Layer 版本)
        gs = GradientShap(wrapper)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 3. 准备 Embeddings (与 LayerConductance 相同)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)
        baseline_input_ids = self._get_pad_baseline(input_ids)
        embedding_layer = self._find_embedding_layer()
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_input_ids)

        # 4. 格式化 Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. 在 embeds 层面调用 attribute
        attributions = gs.attribute(
            inputs=inputs_embeds,       # <--- 传入 embeds
            baselines=baseline_embeds,  # <--- 传入 embeds (作为基线分布)
            n_samples=n_samples,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs
        )

        # 6. 处理结果 (形状: (batch, seq_len, embed_dim) -> (seq_len))
        attr_scores = attributions.sum(
            dim=-1
        ).squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def run_occlusion(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        max_length: int | None = None ,
        sliding_window_shapes: tuple[int] = (1,),
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 Occlusion (遮挡)。
        注意：此方法可能非常慢。

        Args:
            ... (参数与 run_lig 类似) ...
            sliding_window_shapes (Tuple[int]): 遮挡窗口的大小，(1,) 表示一次遮挡1个token。

        Returns:
            Tuple[List[str], np.ndarray]: (tokens 列表, 归因分数数组)
        """
        # 1. 使用接受 input_ids 的包装器
        wrapper = _CaptumWrapperInputIDs(self.model)
        occlusion = Occlusion(wrapper)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 2. 准备输入
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)

        # 3. Occlusion 的基线是单个 PAD token ID (标量)
        baselines = self.pad_token_id

        # 4. 格式化 Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. 计算归因
        attributions = occlusion.attribute(
            inputs=input_ids,
            sliding_window_shapes=sliding_window_shapes,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            baselines=baselines,
            **kwargs
        )

        # 6. 处理结果
        # 形状: (batch, seq_len) -> (seq_len)
        attr_scores = attributions.squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def run_feature_ablation(
        self,
        input_seq: str,
        target: int | str,
        token_index: int | None = None,
        max_length: int | None = None ,
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 Feature Ablation (特征消融)。

        Args:
            ... (参数与 run_occlusion 类似) ...

        Returns:
            Tuple[List[str], np.ndarray]: (tokens 列表, 归因分数数组)
        """
        # 1. 使用接受 input_ids 的包装器
        wrapper = _CaptumWrapperInputIDs(self.model)
        ablation = FeatureAblation(wrapper)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 2. 准备输入和基线 (FeatureAblation 需要一个张量基线)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)
        baselines = self._get_pad_baseline(input_ids)

        # 3. 格式化 Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 4. FeatureAblation 需要 feature_mask 来定义特征（默认每个token一个特征）
        # 形状: (batch_size, num_features) -> (1, seq_len)
        feature_mask = torch.arange(
            input_ids.shape[1]
        ).unsqueeze(0).to(self.device)

        # 5. 计算归因
        attributions = ablation.attribute(
            inputs=input_ids,
            target=captum_target,
            additional_forward_args=(attention_mask,),
            baselines=baselines,
            feature_mask=feature_mask,
            **kwargs
        )

        # 6. 处理结果
        # 形状: (batch, seq_len) -> (seq_len)
        attr_scores = attributions.squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def run_layer_conductance(
        self,
        input_seq: str,
        target: int | str,
        target_layer: nn.Module,
        token_index: int | None = None,
        max_length: int | None = None ,
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 Layer Conductance (层电导)。

        重要: 
        1. 此方法假定 self.model.forward 支持 'inputs_embeds'。
        2. 您必须手动传入 'target_layer'。

        Args:
            ... (参数与 run_lig 类似) ...
            target_layer (nn.Module): 要分析的内部层 
                                     (例如: `model.bert.encoder.layer[-1]`)。

        Returns:
            Tuple[List[str], np.ndarray]: (tokens 列表, 归因分数数组)
        """
        # 1. 使用接受 inputs_embeds 的包装器
        wrapper = _CaptumWrapperInputEmbeds(self.model)
        lc = LayerConductance(wrapper, target_layer)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 2. 准备输入 (input_ids)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)
        baseline_input_ids = self._get_pad_baseline(input_ids)

        # 3. 【关键】手动将 IDs 转换为 Embeddings
        embedding_layer = self._find_embedding_layer()
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_input_ids)

        # 4. 格式化 Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 5. 计算归因 (输入是 embeds)
        attributions_tuple = lc.attribute(
            inputs=inputs_embeds,       # <--- 传入 embeds
            baselines=baseline_embeds,  # <--- 传入 embeds
            target=captum_target,
            additional_forward_args=(attention_mask,),
            **kwargs
        )

        # 6. 处理结果
        if isinstance(attributions_tuple, tuple):
            attributions = attributions_tuple[0]
        else:
            attributions = attributions_tuple
        # 形状: (batch, seq_len, hidden_dim) -> (seq_len)
        attr_scores = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attr_scores.cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def run_noise_tunnel(
        self,
        input_seq: str,
        target: int | str,
        base_method: str,
        token_index: int | None = None,
        max_length: int | None = None ,
        nt_type: str = "smoothgrad",
        nt_samples: int = 5,
        nt_stdevs: float = 0.1,
        **kwargs
    ) -> tuple[list[str], np.ndarray]:
        """
        运行 NoiseTunnel (例如 SmoothGrad) 以获得更平滑的归因。

        这将在 'inputs_embeds' 层面运行，以避免 'nn.Embedding' 的类型冲突。

        Args:
            ...
            base_method (str): 要使用的基础归因方法。
                               支持: 'lig' (IntegratedGradients), 
                                     'deeplift' (DeepLift),
                                     'gradshap' (GradientShap)。
            nt_type (str): 'smoothgrad' (默认), 'smoothgrad_sq', 或 'vargrad'.
            nt_samples (int): 噪声采样的数量。
            nt_stdevs (float): 噪声的标准差。
        """
        print(f"Running NoiseTunnel ({nt_type}) "
              f"with base method: {base_method}...")

        # 1. 使用接受 inputs_embeds 的包装器
        wrapper = _CaptumWrapperInputEmbeds(self.model)

        # 2. 选择基础归因方法 (非 Layer 版本)
        if base_method.lower() == "lig":
            attr_method = IntegratedGradients(wrapper)
        elif base_method.lower() == "deeplift":
            attr_method = DeepLift(wrapper)
        elif base_method.lower() == "gradshap":
            attr_method = GradientShap(wrapper)
        else:
            raise ValueError(f"Unknown base_method: {base_method}. "
                             "Supported: 'lig', 'deeplift', 'gradshap'")

        # 3. 用 NoiseTunnel 包装
        nt = NoiseTunnel(attr_method)

        # Get max token length from config if not provided
        if max_length is None:
            max_length = self.pred_config.max_length

        # 4. 准备 Embeddings (与 LayerConductance 相同)
        input_ids, attention_mask = self._get_input_tensors(
            input_seq, max_length)
        baseline_input_ids = self._get_pad_baseline(input_ids)
        embedding_layer = self._find_embedding_layer()
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_input_ids)

        # 5. 格式化 Captum target
        captum_target = self._format_captum_target(target, token_index)

        # 6. 准备归因参数
        attr_kwargs = {
            "target": captum_target,
            "additional_forward_args": (attention_mask,),
            "nt_type": nt_type,
            "nt_samples": nt_samples,
            "stdevs": nt_stdevs,  # 注意: NoiseTunnel 参数叫 'stdevs'
            **kwargs
        }

        # 7. 根据基础方法调用 attribute
        # GradientShap 需要 n_samples 和 baselines (分布)
        if base_method.lower() == "gradshap":
            attributions = nt.attribute(
                inputs=inputs_embeds,
                baselines=baseline_embeds,  # 使用 PAD embeds 作为基线分布
                n_samples=nt_samples,  # gradshap 自己的 n_samples
                **attr_kwargs
            )
        else:  # LIG 和 DeepLIFT
            attributions = nt.attribute(
                inputs=inputs_embeds,
                baselines=baseline_embeds,
                **attr_kwargs
            )

        # 8. 处理结果
        # 形状: (batch, seq_len, hidden_dim) -> (seq_len)
        attr_scores = attributions.sum(
            dim=-1
        ).squeeze(0).cpu().detach().numpy()

        tokens = self._ids_to_tokens(
            token_ids=input_ids.squeeze(0),
            input_seq=input_seq
        )
        return tokens, attr_scores

    def interpret(self, input_seq: str,
                  method: str,
                  target: int | str,
                  token_index: int | None = None,
                  target_layer: nn.Module | None = None,
                  max_length: int | None = None,
                  plot: bool = True,
                  **kwargs
                  ) -> tuple[list[str], np.ndarray]:
        """
        综合解释接口，根据指定方法运行归因。

        Args:
            input_seq (str): 输入的DNA序列。
            method (str): 归因方法名称。
                支持: 'lig', 'deeplift', 'gradshap',
                      'occlusion', 'feature_ablation',
                      'layer_conductance', 'noise_tunnel'。
            target (int): 目标类别索引。
            token_index (int, optional): 对于 'token_clf'/'causal_lm'
                要解释的token位置。
            target_layer (nn.Module, optional): 对于 'layer_conductance'，
                要分析的内部层。
            max_length (int, optional): Tokenizer的最大长度。
            plot (bool): 是否存储归因以供绘图使用。
            **kwargs: 传递给具体归因方法的额外参数。

        Returns:
            Tuple[List[str], np.ndarray]: (tokens 列表, 归因分数数组)
        """
        method = method.lower()
        if method == "lig":
            tokens, attr_scores = self.run_lig(
                input_seq, target, token_index,
                max_length=max_length, **kwargs
            )
        elif method == "deeplift":
            tokens, attr_scores = self.run_deeplift(
                input_seq, target, token_index,
                max_length=max_length, **kwargs
            )
        elif method == "gradshap":
            tokens, attr_scores = self.run_gradshap(
                input_seq, target, token_index,
                max_length=max_length, **kwargs
            )
        elif method == "occlusion":
            tokens, attr_scores = self.run_occlusion(
                input_seq, target, token_index,
                max_length=max_length, **kwargs
            )
        elif method == "feature_ablation":
            tokens, attr_scores = self.run_feature_ablation(
                input_seq, target, token_index,
                max_length=max_length, **kwargs
            )
        elif method == "layer_conductance":
            if target_layer is None:
                raise ValueError("`target_layer` must be provided for "
                                 "`layer_conductance` method.")
            tokens, attr_scores = self.run_layer_conductance(
                input_seq, target, target_layer, token_index,
                max_length=max_length, **kwargs
            )
        elif method == "noise_tunnel":
            tokens, attr_scores = self.run_noise_tunnel(
                input_seq, target, token_index,
                max_length=max_length, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}. "
                             "Supported methods: 'lig', 'deeplift', "
                             "'gradshap', 'occlusion', 'feature_ablation', "
                             "'layer_conductance', 'noise_tunnel'.")
        if plot:
            self.attributions = (tokens, attr_scores)
        else:
            self.attributions = None

        return tokens, attr_scores

    def batch_interpret(self, input_seqs: list[str],
                        method: str,
                        targets: list[int],
                        token_indices: list[int] | None = None,
                        target_layers: list[nn.Module] | None = None,
                        max_length: int | None = None,
                        plot: bool = True,
                        **kwargs
                        ) -> list[tuple[list[str], np.ndarray]]:
        """
        批量解释多个序列。

        Args:
            input_seqs (List[str]): 输入的DNA序列列表。
            method (str): 归因方法名称。
            targets (List[int]): 每个序列的目标类别索引列表。
            token_indices (List[int], optional): 每个序列的token位置列表。
            target_layers (List[nn.Module], optional): 每个序列的target_layer列表。
            max_length (int, optional): Tokenizer的最大长度。
            **kwargs: 传递给具体归因方法的额外参数。

        Returns:
            List[Tuple[List[str], np.ndarray]]: 每个序列的 (tokens 列表, 归因分数数组) 列表。
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
                **kwargs
            )
            results.append((tokens, scores))
        if plot:
            self.attributions = results
        else:
            self.attributions = None

        return results

    def plot_attributions(self,
                          plot_type: str = "token",
                          **kwargs
                          ):
        """
        绘制归因结果。

        Args:
            plot_type (str): 绘图类型，'token' (默认), 'line', 或 'multi'。
            **kwargs: 传递给具体绘图函数的额外参数。
        """
        if self.attributions is None:
            raise RuntimeError("No attributions found. "
                               "Please run `interpret` or `batch_interpret` "
                               "with `plot=True` first.")

        plot_type = plot_type.lower()
        if isinstance(self.attributions, list):
            if plot_type != "multi":
                print("Warning: Multiple attributions found, "
                      "falling back to 'multi' plot.")
            # 多个序列的归因图
            plot = plot_attributions_multi(self.attributions, **kwargs)
        elif plot_type == "token":
            # 单个序列的 token 归因图
            tokens, scores = self.attributions
            plot = plot_attributions_token(tokens, scores, **kwargs)
        elif plot_type == "line":
            # 单个序列的线性归因图
            tokens, scores = self.attributions
            plot = plot_attributions_line(tokens, scores, **kwargs)
        elif plot_type == "multi":
            # 多个序列的归因图
            plot = plot_attributions_multi(self.attributions, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. "
                             "Supported: 'token', 'line', 'multi'.")
        return plot
