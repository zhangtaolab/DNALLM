"""
Model Configuration Generator for MCP Server

This module generates MCP server configurations based on model_info.yaml
and creates inference configurations for each model.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path


class MCPModelConfigGenerator:
    """基于 model_info.yaml 生成 MCP 服务器配置"""
    
    def __init__(self, model_info_path: str = "dnallm/models/model_info.yaml"):
        self.model_info_path = model_info_path
        self.model_info = self._load_model_info()
        self.finetuned_models = self.model_info.get('finetuned', [])
    
    def _load_model_info(self) -> Dict[str, Any]:
        """加载 model_info.yaml 文件"""
        try:
            with open(self.model_info_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model info file not found: {self.model_info_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {self.model_info_path}: {e}")
    
    def _find_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """根据模型名称查找模型信息"""
        for model in self.finetuned_models:
            if model['name'] == model_name:
                return model
        return None
    
    def _create_model_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """为单个模型创建 MCP 配置"""
        model_name = model_info['name']
        task_info = model_info['task']
        
        # 生成配置文件路径
        config_filename = f"{model_name.lower().replace(' ', '_')}_config.yaml"
        config_path = f"./configs/generated/{self._get_task_category(task_info['task_type'])}/{config_filename}"
        
        return {
            "name": model_name.lower().replace(' ', '_'),
            "model_name": model_name,
            "config_path": config_path,
            "enabled": True,
            "max_concurrent_requests": self._get_default_concurrent_requests(task_info['task_type']),
            "task_type": task_info['task_type'],
            "description": task_info['describe']
        }
    
    def _get_task_category(self, task_type: str) -> str:
        """根据任务类型获取配置目录"""
        if task_type == "binary":
            return "promoter_configs"  # 默认使用 promoter_configs，实际应该根据具体任务分类
        elif task_type == "multiclass":
            return "open_chromatin_configs"
        elif task_type == "regression":
            return "promoter_strength_configs"
        else:
            return "promoter_configs"
    
    def _get_default_concurrent_requests(self, task_type: str) -> int:
        """根据任务类型获取默认并发请求数"""
        if task_type == "binary":
            return 10
        elif task_type == "multiclass":
            return 6
        elif task_type == "regression":
            return 5
        else:
            return 8
    
    def _get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "log_level": "info",
            "cors_origins": ["*"]
        }
    
    def _get_mcp_config(self) -> Dict[str, Any]:
        """获取 MCP 配置"""
        return {
            "name": "DNALLM MCP Server",
            "version": "1.0.0",
            "description": "DNA sequence prediction server using MCP protocol"
        }
    
    def _get_sse_config(self) -> Dict[str, Any]:
        """获取 SSE 配置"""
        return {
            "heartbeat_interval": 30,
            "max_connections": 100,
            "buffer_size": 1000
        }
    
    def _get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "./logs/mcp_server.log"
        }
    
    def generate_mcp_server_config(self, selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """生成 MCP 服务器配置"""
        if selected_models is None:
            # 默认选择一些代表性的模型
            selected_models = [
                "Plant DNABERT BPE promoter",
                "Plant DNABERT BPE conservation", 
                "Plant DNABERT BPE open chromatin",
                "Plant DNABERT BPE promoter strength leaf"
            ]
        
        models_config = []
        for model_name in selected_models:
            model_info = self._find_model_info(model_name)
            if model_info:
                config = self._create_model_config(model_info)
                models_config.append(config)
        
        # 生成多模型并行预测配置
        multi_model_config = self._get_multi_model_config()
        
        return {
            "server": self._get_server_config(),
            "mcp": self._get_mcp_config(),
            "models": models_config,
            "multi_model": multi_model_config,
            "sse": self._get_sse_config(),
            "logging": self._get_logging_config()
        }
    
    def _get_multi_model_config(self) -> Dict[str, Any]:
        """获取多模型并行预测配置"""
        return {
            "enabled": True,
            "max_parallel_models": 8,
            "default_model_sets": {
                "comprehensive_analysis": {
                    "name": "Comprehensive DNA Analysis",
                    "description": "Analyze DNA sequence for multiple functional elements",
                    "models": [
                        "Plant DNABERT BPE open chromatin",
                        "Plant DNABERT BPE promoter",
                        "Plant DNABERT BPE H3K27me3",
                        "Plant DNABERT BPE H3K27ac",
                        "Plant DNABERT BPE H3K4me3",
                        "Plant DNABERT BPE conservation",
                        "Plant DNABERT BPE lncRNAs"
                    ]
                },
                "regulatory_analysis": {
                    "name": "Regulatory Element Analysis",
                    "description": "Focus on regulatory elements",
                    "models": [
                        "Plant DNABERT BPE promoter",
                        "Plant DNABERT BPE H3K27ac",
                        "Plant DNABERT BPE H3K4me3",
                        "Plant DNABERT BPE H3K27me3"
                    ]
                },
                "chromatin_analysis": {
                    "name": "Chromatin State Analysis",
                    "description": "Analyze chromatin accessibility and modifications",
                    "models": [
                        "Plant DNABERT BPE open chromatin",
                        "Plant DNABERT BPE H3K27ac",
                        "Plant DNABERT BPE H3K4me3",
                        "Plant DNABERT BPE H3K27me3"
                    ]
                }
            }
        }
    
    def _create_inference_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """为单个模型创建推理配置"""
        task_info = model_info['task']
        
        return {
            "task": {
                "task_type": task_info['task_type'],
                "num_labels": task_info['num_labels'],
                "label_names": task_info['label_names'],
                "threshold": task_info.get('threshold', 0.5)
            },
            "inference": {
                "batch_size": 16,
                "max_length": 512,
                "device": "auto",
                "num_workers": 4,
                "use_fp16": False,
                "output_dir": "./results"
            },
            "model": {
                "name": model_info['name'],
                "path": model_info['model'],
                "source": "huggingface",
                "trust_remote_code": True,
                "torch_dtype": "float32",
                "task_info": task_info
            }
        }
    
    def generate_inference_configs(self, output_dir: str = "./configs/generated") -> List[str]:
        """为每个模型生成独立的推理配置文件"""
        generated_files = []
        
        for model in self.finetuned_models:
            config = self._create_inference_config(model)
            
            # 确定输出目录
            task_type = model['task']['task_type']
            if task_type == "binary":
                category_dir = "promoter_configs"
            elif task_type == "multiclass":
                category_dir = "open_chromatin_configs"
            elif task_type == "regression":
                category_dir = "promoter_strength_configs"
            else:
                category_dir = "promoter_configs"
            
            # 创建目录
            category_path = os.path.join(output_dir, category_dir)
            os.makedirs(category_path, exist_ok=True)
            
            # 生成文件名
            filename = f"{model['name'].lower().replace(' ', '_')}_config.yaml"
            filepath = os.path.join(category_path, filename)
            
            # 保存配置
            self._save_config(config, filepath)
            generated_files.append(filepath)
        
        return generated_files
    
    def _save_config(self, config: Dict[str, Any], filepath: str) -> None:
        """保存配置到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def get_models_by_task_type(self, task_type: str) -> List[Dict[str, Any]]:
        """根据任务类型获取模型列表"""
        return [model for model in self.finetuned_models 
                if model['task']['task_type'] == task_type]
    
    def get_all_task_types(self) -> List[str]:
        """获取所有可用的任务类型"""
        task_types = set()
        for model in self.finetuned_models:
            task_types.add(model['task']['task_type'])
        return list(task_types)
    
    def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型能力信息"""
        model_info = self._find_model_info(model_name)
        if not model_info:
            return None
        
        task_info = model_info['task']
        return {
            "model_name": model_name,
            "task_type": task_info['task_type'],
            "num_labels": task_info['num_labels'],
            "label_names": task_info['label_names'],
            "description": task_info['describe'],
            "threshold": task_info.get('threshold', 0.5),
            "model_path": model_info['model']
        }


def main():
    """主函数，用于生成配置文件"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MCP server configurations")
    parser.add_argument("--model-info", default="dnallm/models/model_info.yaml",
                       help="Path to model_info.yaml file")
    parser.add_argument("--output-dir", default="./configs/generated",
                       help="Output directory for generated configs")
    parser.add_argument("--selected-models", nargs="+",
                       help="Selected models to include in MCP server config")
    parser.add_argument("--generate-inference", action="store_true",
                       help="Generate inference configs for all models")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = MCPModelConfigGenerator(args.model_info)
    
    # 生成 MCP 服务器配置
    mcp_config = generator.generate_mcp_server_config(args.selected_models)
    
    # 保存 MCP 服务器配置
    mcp_config_path = os.path.join(args.output_dir, "mcp_server_config.yaml")
    os.makedirs(os.path.dirname(mcp_config_path), exist_ok=True)
    
    with open(mcp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(mcp_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"MCP server config saved to: {mcp_config_path}")
    
    # 生成推理配置
    if args.generate_inference:
        generated_files = generator.generate_inference_configs(args.output_dir)
        print(f"Generated {len(generated_files)} inference config files")
        for file_path in generated_files:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()
