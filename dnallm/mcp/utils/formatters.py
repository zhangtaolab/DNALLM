"""
Result formatting utilities for MCP server.
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def format_prediction_result(
    result: Dict[str, Any], 
    model_name: str, 
    sequence: str,
    task_type: str,
    processing_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    格式化单个预测结果
    
    Args:
        result: 原始预测结果
        model_name: 模型名称
        sequence: 输入序列
        task_type: 任务类型
        processing_time: 处理时间（秒）
    
    Returns:
        格式化后的结果
    """
    formatted = {
        "model_name": model_name,
        "task_type": task_type,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "timestamp": datetime.now().isoformat(),
        "processing_time": processing_time,
        "prediction": result
    }
    
    # 根据任务类型格式化结果
    if task_type == "binary":
        formatted["prediction"] = _format_binary_result(result)
    elif task_type == "multiclass":
        formatted["prediction"] = _format_multiclass_result(result)
    elif task_type == "multilabel":
        formatted["prediction"] = _format_multilabel_result(result)
    elif task_type == "regression":
        formatted["prediction"] = _format_regression_result(result)
    
    return formatted


def _format_binary_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化二分类结果"""
    formatted = {
        "prediction": result.get("prediction", "Unknown"),
        "confidence": result.get("confidence", 0.0),
        "probabilities": result.get("probabilities", {}),
        "threshold": result.get("threshold", 0.5)
    }
    
    # 确保概率值在合理范围内
    if "probabilities" in formatted:
        for label, prob in formatted["probabilities"].items():
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                formatted["probabilities"][label] = 0.0
    
    return formatted


def _format_multiclass_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化多分类结果"""
    formatted = {
        "prediction": result.get("prediction", "Unknown"),
        "confidence": result.get("confidence", 0.0),
        "probabilities": result.get("probabilities", {}),
        "top_k": result.get("top_k", [])
    }
    
    # 确保概率值在合理范围内
    if "probabilities" in formatted:
        for label, prob in formatted["probabilities"].items():
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                formatted["probabilities"][label] = 0.0
    
    return formatted


def _format_multilabel_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化多标签结果"""
    formatted = {
        "predictions": result.get("predictions", []),
        "probabilities": result.get("probabilities", {}),
        "threshold": result.get("threshold", 0.5)
    }
    
    # 确保概率值在合理范围内
    if "probabilities" in formatted:
        for label, prob in formatted["probabilities"].items():
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                formatted["probabilities"][label] = 0.0
    
    return formatted


def _format_regression_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化回归结果"""
    formatted = {
        "prediction": result.get("prediction", 0.0),
        "confidence": result.get("confidence", 0.0),
        "uncertainty": result.get("uncertainty", 0.0),
        "range": result.get("range", [])
    }
    
    # 确保预测值是数字
    if not isinstance(formatted["prediction"], (int, float)):
        formatted["prediction"] = 0.0
    
    return formatted


def format_multi_model_result(
    results: Dict[str, Dict[str, Any]], 
    sequence: str,
    processing_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    格式化多模型预测结果
    
    Args:
        results: 多模型预测结果字典
        sequence: 输入序列
        processing_time: 总处理时间
    
    Returns:
        格式化后的多模型结果
    """
    formatted = {
        "sequence": sequence,
        "sequence_length": len(sequence),
        "timestamp": datetime.now().isoformat(),
        "processing_time": processing_time,
        "total_models": len(results),
        "predictions": {},
        "summary": {}
    }
    
    # 格式化每个模型的结果
    for model_name, result in results.items():
        if "error" in result:
            formatted["predictions"][model_name] = {
                "error": result["error"],
                "status": "failed"
            }
        else:
            # 提取任务类型（从模型名称推断或从结果中获取）
            task_type = _infer_task_type(model_name, result)
            formatted["predictions"][model_name] = format_prediction_result(
                result, model_name, sequence, task_type
            )
    
    # 生成摘要
    formatted["summary"] = _generate_summary(formatted["predictions"])
    
    return formatted


def _infer_task_type(model_name: str, result: Dict[str, Any]) -> str:
    """从模型名称或结果推断任务类型"""
    model_name_lower = model_name.lower()
    
    if "promoter" in model_name_lower and "strength" not in model_name_lower:
        return "binary"
    elif "conservation" in model_name_lower:
        return "binary"
    elif "lncrna" in model_name_lower:
        return "binary"
    elif "h3k" in model_name_lower:
        return "binary"
    elif "open_chromatin" in model_name_lower:
        return "multiclass"
    elif "strength" in model_name_lower:
        return "regression"
    else:
        # 从结果中推断
        if "probabilities" in result and isinstance(result["probabilities"], dict):
            if len(result["probabilities"]) == 2:
                return "binary"
            elif len(result["probabilities"]) > 2:
                return "multiclass"
        elif "prediction" in result and isinstance(result["prediction"], (int, float)):
            return "regression"
    
    return "binary"  # 默认


def _generate_summary(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """生成预测摘要"""
    summary = {
        "successful_predictions": 0,
        "failed_predictions": 0,
        "task_types": {},
        "confidence_scores": [],
        "top_predictions": []
    }
    
    for model_name, pred in predictions.items():
        if "error" in pred:
            summary["failed_predictions"] += 1
        else:
            summary["successful_predictions"] += 1
            
            # 统计任务类型
            task_type = pred.get("task_type", "unknown")
            summary["task_types"][task_type] = summary["task_types"].get(task_type, 0) + 1
            
            # 收集置信度分数
            if "prediction" in pred and "confidence" in pred["prediction"]:
                confidence = pred["prediction"]["confidence"]
                if isinstance(confidence, (int, float)):
                    summary["confidence_scores"].append(confidence)
            
            # 收集预测结果
            if "prediction" in pred and "prediction" in pred["prediction"]:
                prediction_value = pred["prediction"]["prediction"]
                summary["top_predictions"].append({
                    "model": model_name,
                    "prediction": prediction_value,
                    "confidence": pred["prediction"].get("confidence", 0.0)
                })
    
    # 计算平均置信度
    if summary["confidence_scores"]:
        summary["average_confidence"] = sum(summary["confidence_scores"]) / len(summary["confidence_scores"])
    else:
        summary["average_confidence"] = 0.0
    
    # 按置信度排序预测结果
    summary["top_predictions"].sort(key=lambda x: x["confidence"], reverse=True)
    summary["top_predictions"] = summary["top_predictions"][:5]  # 只保留前5个
    
    return summary


def format_error_response(
    error: str, 
    error_code: str = "PREDICTION_ERROR",
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    格式化错误响应
    
    Args:
        error: 错误消息
        error_code: 错误代码
        details: 错误详情
    
    Returns:
        格式化后的错误响应
    """
    response = {
        "error": True,
        "error_code": error_code,
        "message": error,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        response["details"] = details
    
    return response


def format_health_check_response(
    status: str = "healthy",
    models_loaded: int = 0,
    total_models: int = 0,
    uptime: Optional[float] = None
) -> Dict[str, Any]:
    """
    格式化健康检查响应
    
    Args:
        status: 服务状态
        models_loaded: 已加载模型数量
        total_models: 总模型数量
        uptime: 运行时间（秒）
    
    Returns:
        格式化后的健康检查响应
    """
    response = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models": {
            "loaded": models_loaded,
            "total": total_models,
            "loading_percentage": (models_loaded / total_models * 100) if total_models > 0 else 0
        }
    }
    
    if uptime is not None:
        response["uptime_seconds"] = uptime
        response["uptime_human"] = _format_uptime(uptime)
    
    return response


def _format_uptime(seconds: float) -> str:
    """格式化运行时间"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def format_model_list_response(
    models: List[Dict[str, Any]], 
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    格式化模型列表响应
    
    Args:
        models: 模型列表
        task_type: 任务类型过滤
    
    Returns:
        格式化后的模型列表响应
    """
    response = {
        "models": [],
        "total_count": len(models),
        "task_type": task_type,
        "timestamp": datetime.now().isoformat()
    }
    
    for model in models:
        formatted_model = {
            "name": model.get("name", ""),
            "model_name": model.get("model_name", ""),
            "task_type": model.get("task_type", ""),
            "description": model.get("description", ""),
            "enabled": model.get("enabled", True),
            "max_concurrent_requests": model.get("max_concurrent_requests", 10),
            "is_loaded": model.get("is_loaded", False),
            "usage_count": model.get("usage_count", 0),
            "last_used": model.get("last_used", None)
        }
        response["models"].append(formatted_model)
    
    return response


def format_model_info_response(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化模型信息响应
    
    Args:
        model_info: 模型信息字典
    
    Returns:
        格式化后的模型信息响应
    """
    response = {
        "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    }
    
    return response


def format_sse_event(
    event_type: str, 
    data: Any, 
    event_id: Optional[str] = None
) -> str:
    """
    格式化 SSE 事件
    
    Args:
        event_type: 事件类型
        data: 事件数据
        event_id: 事件ID
    
    Returns:
        格式化的 SSE 事件字符串
    """
    lines = []
    
    if event_id:
        lines.append(f"id: {event_id}")
    
    lines.append(f"event: {event_type}")
    
    # 将数据转换为 JSON
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, ensure_ascii=False)
    else:
        data_str = str(data)
    
    # 处理多行数据
    for line in data_str.split('\n'):
        lines.append(f"data: {line}")
    
    lines.append("")  # 空行表示事件结束
    
    return "\n".join(lines)
