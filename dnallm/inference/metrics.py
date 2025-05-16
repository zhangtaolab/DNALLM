import os
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score,
                             average_precision_score, roc_curve, roc_auc_score, precision_recall_curve,
                             confusion_matrix, multilabel_confusion_matrix)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

metrics_path = os.path.join(os.path.dirname(__file__), "metrics") + "/"

import evaluate
from ..configuration.configs import TaskConfig


# Define evaluation metrics
def calculate_metric_with_sklearn(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    print(valid_labels.shape, valid_predictions.shape)
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

## Load evaluate metrics locally to avoid downloading from Hugging Face

def classification_metrics(plot=False):
    clf_metrics = evaluate.combine([metrics_path + "accuracy/accuracy.py",
                                    metrics_path + "f1/f1.py",
                                    metrics_path + "precision/precision.py",
                                    metrics_path + "recall/recall.py",
                                    metrics_path + "matthews_correlation/matthews_correlation.py"])
    auc_metric = evaluate.load(metrics_path + "roc_auc/roc_auc.py", "binary")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        predictions = np.argmax(logits, axis=-1)
        pred_probs = softmax(logits, axis=1)
        # metrics = clf_metrics.compute(predictions=predictions, references=labels)
        # roc_auc = auc_metric.compute(references=labels, prediction_scores=pred_probs[:, 1])
        # pr_auc = average_precision_score(y_true=labels, y_score=pred_probs[:, 1])
        # metrics["AUROC"] = roc_auc["roc_auc"]
        # metrics["AUPRC"] = pr_auc
        metrics = {}
        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision"] = precision_score(labels, predictions)
        metrics["recall"] = recall_score(labels, predictions)
        metrics["f1"] = f1_score(labels, predictions)
        metrics["mcc"] = matthews_corrcoef(labels, predictions)
        metrics["AUROC"] = roc_auc_score(labels, pred_probs[:, 1])
        metrics["AUPRC"] = average_precision_score(labels, pred_probs[:, 1])
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics["TPR"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["TNR"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        if plot:
            fpr, tpr, _ = roc_curve(labels, pred_probs[:, 1])
            precision, recall, _ = precision_recall_curve(labels, pred_probs[:, 1])
            metrics["curve"] = {
                "fpr": fpr,
                "tpr": tpr,
                "precision": precision,
                "recall": recall,
            }
        return metrics

    return compute_metrics


def regression_metrics(plot=False):

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        num_outputs = logits.shape[1]
        if num_outputs > 1:
            mse = mean_squared_error(labels, logits)
            mae = mean_absolute_error(labels, logits)
            r2  = r2_score(labels, logits, multioutput='uniform_average')
            metrics = {'mse': mse, 'mae': mae, "r2": r2}
        else:
            mse_metric = evaluate.load(metrics_path + "mse/mse.py")
            mae_metric = evaluate.load(metrics_path + "mae/mae.py")
            r2_metric = evaluate.load(metrics_path + "r_squared/r_squared.py")
            spm_metric = evaluate.load(metrics_path + "spearmanr/spearmanr.py")
            mse = mse_metric.compute(references=labels, predictions=logits)
            mae = mae_metric.compute(references=labels, predictions=logits)
            r2 = r2_metric.compute(references=labels, predictions=logits)
            spearmanr = spm_metric.compute(references=labels, predictions=logits)
            metrics = {**mse, **mae, "r2": r2, **spearmanr}
            if plot:
                metrics['scatter'] = {
                    'predicted': logits.numpy().flatten(),
                    'experiment': labels
                }
        return metrics

    return compute_metrics


def multi_classification_metrics(plot=False):
    # metric0 = evaluate.load(metrics_path + "accuracy/accuracy.py")
    # metric1 = evaluate.load(metrics_path + "precision/precision.py")
    # metric2 = evaluate.load(metrics_path + "recall/recall.py")
    # metric3 = evaluate.load(metrics_path + "f1/f1.py")
    # metric4 = evaluate.load(metrics_path + "matthews_correlation/matthews_correlation.py")
    # roc_metric = evaluate.load(metrics_path + "roc_auc/roc_auc.py", "multiclass")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        if logits.ndim == 3:
            logits = logits[:, 0, :]  # 调整此处以适应模型结构
        pred_probs = softmax(logits, axis=1)
        predictions = np.argmax(logits, axis=-1)

        metrics = {}
        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision"] = precision_score(labels, predictions, average="macro")
        metrics["recall"] = recall_score(labels, predictions, average="macro")
        metrics["f1"] = f1_score(labels, predictions, average="macro")
        metrics["precision_micro"] = precision_score(labels, predictions, average="micro")
        metrics["recall_micro"] = recall_score(labels, predictions, average="micro")
        metrics["f1_micro"] = f1_score(labels, predictions, average="micro")
        metrics["precision_weighted"] = precision_score(labels, predictions, average="weighted")
        metrics["recall_weighted"] = recall_score(labels, predictions, average="weighted")
        metrics["f1_weighted"] = f1_score(labels, predictions, average="weighted")
        metrics["mcc"] = matthews_corrcoef(labels, predictions)
        metrics["AUROC"] = roc_auc_score(labels, pred_probs, average="macro", multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(labels, pred_probs, average="macro")
        tpr_list, tnr_list, fpr_list, fnr_list = [], [], [], []
        for label_cnt in multilabel_confusion_matrix(labels, predictions):
            tn, fp, fn, tp = label_cnt.ravel()
            # 避免除 0
            tpr_list.append(tp/(tp+fn) if (tp+fn)>0 else 0)
            tnr_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)
            fpr_list.append(fp/(fp+tn) if (fp+tn)>0 else 0)
            fnr_list.append(fn/(fn+tp) if (fn+tp)>0 else 0)
        metrics["TPR"] = float(np.mean(tpr_list))
        metrics["TNR"] = float(np.mean(tnr_list))
        metrics["FPR"] = float(np.mean(fpr_list))
        metrics["FNR"] = float(np.mean(fnr_list))
        # accuracy = metric0.compute(predictions=predictions, references=labels)
        # precision = metric1.compute(predictions=predictions, references=labels, average="micro")
        # recall = metric2.compute(predictions=predictions, references=labels, average="micro")
        # f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
        # mcc = metric4.compute(predictions=predictions, references=labels)
        # roc_auc_ovr = roc_metric.compute(references=labels,
        #                                  prediction_scores=pred_probs,
        #                                  multi_class='ovr')
        # roc_auc_ovo = roc_metric.compute(references=labels,
        #                                  prediction_scores=pred_probs,
        #                                  multi_class='ovo')
        # metrics = {**accuracy, **precision, **recall, **f1, **mcc,
        #            "AUROC_ovr": roc_auc_ovr['roc_auc'], "AUROC_ovo": roc_auc_ovo['roc_auc']}
        # metrics["AUROC_ovr"] = roc_auc_ovr['roc_auc']
        # metrics["AUROC_ovo"] = roc_auc_ovo['roc_auc']
        if plot:
            fpr, tpr, _ = roc_curve(labels, pred_probs[:, 1])
            prec, rec, _ = precision_recall_curve(labels, pred_probs[:, 1])
            metrics["curve"] = {
                "fpr": fpr,
                "tpr": tpr,
                "precision": prec,
                "recall": rec,
            }
        return metrics

    return compute_metrics


def multi_labels_metrics(label_list, plot=False):
    # metric0 = evaluate.load(metrics_path + "accuracy/accuracy.py")
    # metric1 = evaluate.load(metrics_path + "precision/precision.py")
    # metric2 = evaluate.load(metrics_path + "recall/recall.py")
    # metric3 = evaluate.load(metrics_path + "f1/f1.py")

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits.numpy()
        labels = labels.numpy()
        pred_probs = sigmoid(logits)
        raw_pred = (pred_probs > 0.5).astype(int)
        predictions = raw_pred.reshape(-1)
        y_true = labels.astype(int).reshape(-1)

        # accuracy = metric0.compute(predictions=predictions, references=y_true)
        # precision = metric1.compute(predictions=predictions, references=y_true, average="macro")
        # recall = metric2.compute(predictions=predictions, references=y_true, average="macro")
        # f1 = metric3.compute(predictions=predictions, references=y_true, average="macro")
        # metrics = {**accuracy, **precision, **recall, **f1}
        metrics = {}
        metrics["accuracy"] = accuracy_score(labels, raw_pred)
        metrics["precision"] = precision_score(labels, raw_pred, average="macro")
        metrics["recall"] = recall_score(labels, raw_pred, average="macro")
        metrics["f1"] = f1_score(labels, raw_pred, average="macro")
        metrics["precision_micro"] = precision_score(labels, raw_pred, average="micro")
        metrics["recall_micro"] = recall_score(labels, raw_pred, average="micro")
        metrics["f1_micro"] = f1_score(labels, raw_pred, average="micro")
        metrics["precision_weighted"] = precision_score(labels, raw_pred, average="weighted")
        metrics["recall_weighted"] = recall_score(labels, raw_pred, average="weighted")
        metrics["f1_weighted"] = f1_score(labels, raw_pred, average="weighted")
        metrics["precision_samples"] = precision_score(labels, raw_pred, average="samples")
        metrics["recall_samples"] = recall_score(labels, raw_pred, average="samples")
        metrics["f1_samples"] = f1_score(labels, raw_pred, average="samples")
        mcc_per_label = {}
        roc_data, roc_auc = {}, {}
        pr_data, pr_auc = {}, {}
        for i in range(labels.shape[1]):
            # Compute matthews correlation coefficient for each class
            mcc_per_label[label_list[i]] = matthews_corrcoef(labels[:, i], raw_pred[:, i])
            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(labels[:, i], pred_probs[:, i])
            auc = roc_auc_score(labels[:, i], pred_probs[:, i])
            roc_data[label_list[i]] = (fpr, tpr)
            roc_auc[label_list[i]] = auc
            # Compute PR curve and PR area for each class
            prec, rec, _ = precision_recall_curve(labels[:, i], pred_probs[:, i])
            ap = average_precision_score(labels[:, i], pred_probs[:, i])
            pr_data[label_list[i]] = (prec, rec)
            pr_auc[label_list[i]] = ap
        metrics['mcc'] = np.mean(list(mcc_per_label.values()))
        metrics['AUROC'] = np.mean(list(roc_auc.values()))
        metrics['AUPRC'] = np.mean(list(pr_auc.values()))
        tpr_list, tnr_list, fpr_list, fnr_list = [], [], [], []
        for label_cnt in multilabel_confusion_matrix(labels, raw_pred):
            tn, fp, fn, tp = label_cnt.ravel()
            # 避免除 0
            tpr_list.append(tp/(tp+fn) if (tp+fn)>0 else 0)
            tnr_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)
            fpr_list.append(fp/(fp+tn) if (fp+tn)>0 else 0)
            fnr_list.append(fn/(fn+tp) if (fn+tp)>0 else 0)
        metrics["TPR"] = float(np.mean(tpr_list))
        metrics["TNR"] = float(np.mean(tnr_list))
        metrics["FPR"] = float(np.mean(fpr_list))
        metrics["FNR"] = float(np.mean(fnr_list))

        if plot:
            metrics["curve"] = {}
            for label in label_list:
                metrics["curve"][label] = {
                    "fpr": roc_data[label][0],
                    "tpr": roc_data[label][1],
                    "precision": pr_data[label][0],
                    "recall": pr_data[label][1],
                }
        return metrics

    return compute_metrics


def token_classification_metrics(label_list, plot=False, scheme="IOB2"):
    seqeval = evaluate.load(metrics_path + "seqeval/seqeval.py")

    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=-1)

        # 将id转换为原始的字符串类型的标签
        true_predictions = [
            [label_list[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels) 
        ]

        true_labels = [
            [label_list[l] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels) 
        ]

        result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme=scheme)

        return {
            "accuracy": result["overall_accuracy"],
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"]
        }

    return compute_metrics


def metrics_for_dnabert2(task):
    import torch

    r2_metric = evaluate.load("r_squared")
    spm_metric = evaluate.load("spearmanr")
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall", "matthews_correlation"])
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")
    metric4 = evaluate.load("matthews_correlation")
    roc_metric = evaluate.load("roc_auc", "multiclass")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task.lower() == "regression":
            r2 = r2_metric.compute(references=labels, predictions=logits[0])
            spearman = spm_metric.compute(references=labels, predictions=logits[0])
            return {"r2": r2, "spearmanr": spearman['spearmanr']}
        else:
            if task.lower() == "classification":
                predictions = torch.argmax(torch.from_numpy(logits[0]), dim=-1)
                return clf_metrics.compute(predictions=predictions, references=labels)
            else:
                pred_probs = softmax(logits[0], axis=1)
                predictions = [x.tolist().index(max(x)) for x in pred_probs]
                precision = metric1.compute(predictions=predictions, references=labels, average="micro")
                recall = metric2.compute(predictions=predictions, references=labels, average="micro")
                f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
                mcc = metric4.compute(predictions=predictions, references=labels)
                roc_auc_ovr = roc_metric.compute(references=labels,
                                                 prediction_scores=pred_probs,
                                                 multi_class='ovr')
                roc_auc_ovo = roc_metric.compute(references=labels,
                                                 prediction_scores=pred_probs,
                                                 multi_class='ovo')
                return {**precision, **recall, **f1, **mcc, "AUROC_ovr": roc_auc_ovr['roc_auc'], "AUROC_ovo": roc_auc_ovo['roc_auc']}

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        logits = logits[0] if isinstance(logits, tuple) else logits
        # pred_ids = torch.argmax(logits, dim=-1)
        return logits, labels
    
    return compute_metrics, preprocess_logits_for_metrics


def compute_metrics(task_config: TaskConfig, plot: bool=False) -> dict:
    """Compute metrics based on task type"""
    if task_config.task_type == "binary":
        return classification_metrics(plot=plot)
    elif task_config.task_type == "multiclass":
        return multi_classification_metrics(plot=plot)
    elif task_config.task_type == "multilabel":
        return multi_labels_metrics(task_config.label_names, plot=plot)
    elif task_config.task_type == "regression":
        return regression_metrics(plot=plot)
    elif task_config.task_type == "token":
        return token_classification_metrics(task_config.label_names, plot=plot)
    else:
        raise ValueError(f"Unsupported task type for evaluation: {task_config.task_type}")
