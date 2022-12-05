import torch

from datasets import (Dataset, DatasetDict, load_dataset)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import f1_score

import numpy as np


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    """Compute accuracy of predictions"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, value_classes, thresh=0.5, sigmoid=True):
    """Compute label-wise and averaged F1-scores"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred = (y_pred > thresh).numpy()

    f1_scores = {}
    for i, v in enumerate(value_classes):
        f1_scores[v] = round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 3)

    f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 3)

    return f1_scores


def compute_metrics(eval_pred, value_classes):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, value_classes)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels), 'f1-score': f1scores,
            'marco-avg-f1score': f1scores['avg-f1-score']}

def print_metrics(metrics):
    print("Accuracy threshold: " + metrics['accuracy_thresh'])
    print("F1 Score: " + metrics['f1-score'])
    print("Macro Avg F1: " + metrics['marco-avg-f1score'])
