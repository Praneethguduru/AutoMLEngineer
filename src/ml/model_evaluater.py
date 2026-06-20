import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(y_test, predictions):
    unique_classes = np.unique(y_test)
    is_multiclass = len(unique_classes) > 2
    avg = 'macro' if is_multiclass else 'binary'

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average=avg, zero_division=0),
        "recall": recall_score(y_test, predictions, average=avg, zero_division=0),
        "f1_score": f1_score(y_test, predictions, average=avg, zero_division=0)
    }