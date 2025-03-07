from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import evaluate
import torch
import torch.nn as nn
from .tasks import TaskConfig, TaskType


def compute_metrics(task_config: TaskConfig, predictions: torch.Tensor, 
                   labels: torch.Tensor) -> dict:
    """Compute metrics based on task type"""
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    metrics = {}
    
    if task_config.task_type == TaskType.BINARY:
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        preds = (probs > task_config.threshold).astype(int)
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1"] = f1_score(labels, preds)
        
    elif task_config.task_type == TaskType.MULTICLASS:
        preds = predictions.argmax(axis=1)
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1_macro"] = f1_score(labels, preds, average="macro")
        metrics["f1_weighted"] = f1_score(labels, preds, average="weighted")
        
    else:  # Regression
        metrics["mse"] = mean_squared_error(labels, predictions)
        metrics["r2"] = r2_score(labels, predictions)
        
    return metrics 
