import torch
from typing import List, Dict, Union
from pathlib import Path
from ..finetune.models import get_model
from .config import InferenceConfig
from .utils import batch_sequences, save_predictions

"""
DNA语言模型推理器模块

本模块实现了模型推理的核心功能，主要包括：

1. DNAPredictor类
   - 模型加载和初始化
   - 批量序列预测
   - 结果后处理
   - 设备管理
   - 半精度推理支持

2. 核心功能：
   - 模型状态管理
   - 批处理预测
   - 结果合并
   - 预测结果保存
   - 内存优化

3. 推理优化：
   - 批处理并行
   - GPU加速
   - 半精度计算
   - 内存效率优化

使用示例：
    predictor = DNAPredictor(
        model_type="plant_dna",
        model_path="models/checkpoint",
        config=inference_config
    )
    results = predictor.predict(sequences)
"""

class DNAPredictor:
    """DNA sequence predictor using fine-tuned models"""
    
    def __init__(self, model_type: str, model_path: str, config: InferenceConfig):
        """
        Initialize predictor
        
        Args:
            model_type: Type of model (plant_dna, dnabert, etc.)
            model_path: Path to fine-tuned model
            config: Inference configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        self.model = get_model(model_type, model_path)
        self.model.to(self.device)
        
        if config.use_fp16:
            self.model = self.model.half()
        
        self.model.eval()
    
    @torch.no_grad()
    def predict_batch(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Predict for a batch of sequences"""
        inputs = self.model.preprocess(sequences)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.config.use_fp16:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
            
        return outputs
    
    def predict(self, sequences: Union[str, List[str]], 
                save_to_file: bool = False) -> Dict[str, torch.Tensor]:
        """
        Predict for sequences
        
        Args:
            sequences: Single sequence or list of sequences
            save_to_file: Whether to save predictions to file
            
        Returns:
            Dictionary containing predictions
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        all_predictions = []
        for batch in batch_sequences(sequences, self.config.batch_size):
            predictions = self.predict_batch(batch)
            all_predictions.append(predictions)
            
        # Combine predictions
        combined = {
            k: torch.cat([p[k] for p in all_predictions])
            for k in all_predictions[0].keys()
        }
        
        if save_to_file and self.config.output_dir:
            save_predictions(combined, Path(self.config.output_dir))
            
        return combined 