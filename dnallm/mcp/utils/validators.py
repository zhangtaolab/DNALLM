"""
Data validation utilities for MCP server.
"""

import re
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def validate_dna_sequence(sequence: str, max_length: int = 10000) -> Dict[str, Any]:
    """
    验证 DNA 序列
    
    Args:
        sequence: DNA 序列字符串
        max_length: 最大长度限制
    
    Returns:
        验证结果字典，包含 is_valid, errors, warnings
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "sequence_length": len(sequence),
        "cleaned_sequence": sequence
    }
    
    if not sequence:
        result["is_valid"] = False
        result["errors"].append("Sequence cannot be empty")
        return result
    
    # 转换为大写
    sequence = sequence.upper().strip()
    result["cleaned_sequence"] = sequence
    
    # 检查长度
    if len(sequence) > max_length:
        result["is_valid"] = False
        result["errors"].append(f"Sequence too long: {len(sequence)} > {max_length}")
    
    if len(sequence) < 10:
        result["warnings"].append("Sequence is very short (< 10 bp)")
    
    # 检查字符
    valid_chars = set('ATCGN')
    invalid_chars = set(sequence) - valid_chars
    
    if invalid_chars:
        result["is_valid"] = False
        result["errors"].append(f"Invalid characters found: {', '.join(invalid_chars)}")
    
    # 检查 N 的比例
    n_count = sequence.count('N')
    n_ratio = n_count / len(sequence) if sequence else 0
    
    if n_ratio > 0.1:
        result["warnings"].append(f"High N content: {n_ratio:.1%}")
    
    # 检查重复序列
    if len(sequence) > 100:
        # 检查简单的重复模式
        for pattern_length in [2, 3, 4, 5]:
            pattern = sequence[:pattern_length]
            if sequence.count(pattern) > len(sequence) // pattern_length * 0.8:
                result["warnings"].append(f"Possible repetitive sequence detected (pattern: {pattern})")
                break
    
    return result


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证模型配置
    
    Args:
        config: 模型配置字典
    
    Returns:
        验证结果字典
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    # 检查必需字段
    required_fields = ['name', 'model_name', 'config_path', 'task_type']
    for field in required_fields:
        if field not in config:
            result["is_valid"] = False
            result["errors"].append(f"Missing required field: {field}")
    
    # 验证任务类型
    if 'task_type' in config:
        valid_task_types = ['binary', 'multiclass', 'multilabel', 'regression']
        if config['task_type'] not in valid_task_types:
            result["is_valid"] = False
            result["errors"].append(f"Invalid task_type: {config['task_type']}. Must be one of {valid_task_types}")
    
    # 验证并发请求数
    if 'max_concurrent_requests' in config:
        try:
            max_req = int(config['max_concurrent_requests'])
            if max_req <= 0:
                result["is_valid"] = False
                result["errors"].append("max_concurrent_requests must be positive")
            elif max_req > 100:
                result["warnings"].append("max_concurrent_requests is very high (> 100)")
        except (ValueError, TypeError):
            result["is_valid"] = False
            result["errors"].append("max_concurrent_requests must be an integer")
    
    # 验证配置文件路径
    if 'config_path' in config:
        import os
        if not os.path.exists(config['config_path']):
            result["warnings"].append(f"Config file not found: {config['config_path']}")
    
    return result


def validate_prediction_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证预测请求
    
    Args:
        request: 预测请求字典
    
    Returns:
        验证结果字典
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "cleaned_request": request.copy()
    }
    
    # 检查必需字段
    if 'sequence' not in request:
        result["is_valid"] = False
        result["errors"].append("Missing required field: sequence")
    else:
        # 验证序列
        seq_validation = validate_dna_sequence(request['sequence'])
        if not seq_validation['is_valid']:
            result["is_valid"] = False
            result["errors"].extend(seq_validation['errors'])
        
        result["warnings"].extend(seq_validation['warnings'])
        result["cleaned_request"]["sequence"] = seq_validation['cleaned_sequence']
    
    # 验证模型名称
    if 'model_name' in request:
        model_name = request['model_name']
        if not isinstance(model_name, str) or not model_name.strip():
            result["is_valid"] = False
            result["errors"].append("model_name must be a non-empty string")
        else:
            result["cleaned_request"]["model_name"] = model_name.strip()
    
    # 验证任务类型
    if 'task_type' in request:
        task_type = request['task_type']
        valid_types = ['binary', 'multiclass', 'multilabel', 'regression']
        if task_type not in valid_types:
            result["is_valid"] = False
            result["errors"].append(f"Invalid task_type: {task_type}. Must be one of {valid_types}")
    
    # 验证批量请求
    if 'sequences' in request:
        sequences = request['sequences']
        if not isinstance(sequences, list):
            result["is_valid"] = False
            result["errors"].append("sequences must be a list")
        elif len(sequences) == 0:
            result["is_valid"] = False
            result["errors"].append("sequences list cannot be empty")
        elif len(sequences) > 100:
            result["warnings"].append(f"Large batch size: {len(sequences)} sequences")
        else:
            # 验证每个序列
            cleaned_sequences = []
            for i, seq in enumerate(sequences):
                seq_validation = validate_dna_sequence(seq)
                if not seq_validation['is_valid']:
                    result["errors"].append(f"Invalid sequence at index {i}: {', '.join(seq_validation['errors'])}")
                else:
                    cleaned_sequences.append(seq_validation['cleaned_sequence'])
                    result["warnings"].extend([f"Sequence {i}: {w}" for w in seq_validation['warnings']])
            
            result["cleaned_request"]["sequences"] = cleaned_sequences
    
    # 验证多模型请求
    if 'models' in request:
        models = request['models']
        if not isinstance(models, list):
            result["is_valid"] = False
            result["errors"].append("models must be a list")
        elif len(models) == 0:
            result["is_valid"] = False
            result["errors"].append("models list cannot be empty")
        elif len(models) > 10:
            result["warnings"].append(f"Many models requested: {len(models)}")
        else:
            # 验证每个模型名称
            cleaned_models = []
            for i, model in enumerate(models):
                if not isinstance(model, str) or not model.strip():
                    result["errors"].append(f"Invalid model name at index {i}")
                else:
                    cleaned_models.append(model.strip())
            
            result["cleaned_request"]["models"] = cleaned_models
    
    return result


def validate_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证服务器配置
    
    Args:
        config: 服务器配置字典
    
    Returns:
        验证结果字典
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    # 验证服务器配置
    if 'server' in config:
        server_config = config['server']
        
        # 验证端口
        if 'port' in server_config:
            try:
                port = int(server_config['port'])
                if port < 1 or port > 65535:
                    result["is_valid"] = False
                    result["errors"].append("Port must be between 1 and 65535")
            except (ValueError, TypeError):
                result["is_valid"] = False
                result["errors"].append("Port must be an integer")
        
        # 验证工作进程数
        if 'workers' in server_config:
            try:
                workers = int(server_config['workers'])
                if workers < 1:
                    result["is_valid"] = False
                    result["errors"].append("Workers must be positive")
                elif workers > 10:
                    result["warnings"].append("High number of workers (> 10)")
            except (ValueError, TypeError):
                result["is_valid"] = False
                result["errors"].append("Workers must be an integer")
    
    # 验证模型配置
    if 'models' in config:
        models = config['models']
        if not isinstance(models, list):
            result["is_valid"] = False
            result["errors"].append("models must be a list")
        else:
            for i, model_config in enumerate(models):
                model_validation = validate_model_config(model_config)
                if not model_validation['is_valid']:
                    result["errors"].extend([f"Model {i}: {error}" for error in model_validation['errors']])
                result["warnings"].extend([f"Model {i}: {warning}" for warning in model_validation['warnings']])
    
    return result


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    清理输入文本
    
    Args:
        text: 输入文本
        max_length: 最大长度
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 限制长度
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def validate_file_path(file_path: str, must_exist: bool = True) -> Dict[str, Any]:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        must_exist: 文件是否必须存在
    
    Returns:
        验证结果字典
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    if not file_path:
        result["is_valid"] = False
        result["errors"].append("File path cannot be empty")
        return result
    
    # 检查路径安全性
    if '..' in file_path or file_path.startswith('/'):
        result["warnings"].append("Potentially unsafe file path")
    
    # 检查文件是否存在
    if must_exist:
        import os
        if not os.path.exists(file_path):
            result["is_valid"] = False
            result["errors"].append(f"File does not exist: {file_path}")
        elif not os.path.isfile(file_path):
            result["is_valid"] = False
            result["errors"].append(f"Path is not a file: {file_path}")
    
    return result
