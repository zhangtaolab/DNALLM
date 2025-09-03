"""
Model information loader for MCP server.
"""

import yaml
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelInfoLoader:
    """模型信息加载器"""
    
    def __init__(self, model_info_path: str = "dnallm/models/model_info.yaml"):
        self.model_info_path = model_info_path
        self.model_info: Optional[Dict[str, Any]] = None
        self._load_model_info()
    
    def _load_model_info(self) -> None:
        """加载模型信息"""
        try:
            with open(self.model_info_path, 'r', encoding='utf-8') as f:
                self.model_info = yaml.safe_load(f)
            logger.info(f"Loaded model info from {self.model_info_path}")
        except FileNotFoundError:
            logger.error(f"Model info file not found: {self.model_info_path}")
            self.model_info = {"pretrained": [], "finetuned": []}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format: {e}")
            self.model_info = {"pretrained": [], "finetuned": []}
    
    def get_finetuned_models(self) -> List[Dict[str, Any]]:
        """获取微调模型列表"""
        if not self.model_info:
            return []
        return self.model_info.get('finetuned', [])
    
    def get_pretrained_models(self) -> List[Dict[str, Any]]:
        """获取预训练模型列表"""
        if not self.model_info:
            return []
        return self.model_info.get('pretrained', [])
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取模型信息"""
        models = self.get_finetuned_models() + self.get_pretrained_models()
        for model in models:
            if model.get('name') == model_name:
                return model
        return None
    
    def get_models_by_task_type(self, task_type: str) -> List[Dict[str, Any]]:
        """根据任务类型获取模型列表"""
        models = self.get_finetuned_models()
        return [model for model in models if model.get('task', {}).get('task_type') == task_type]
    
    def get_all_task_types(self) -> List[str]:
        """获取所有任务类型"""
        models = self.get_finetuned_models()
        task_types = set()
        for model in models:
            task_type = model.get('task', {}).get('task_type')
            if task_type:
                task_types.add(task_type)
        return list(task_types)
