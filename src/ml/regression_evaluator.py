from sklearn.metrics import (mean_absolute_error,r2_score)


def evaluate_regression(y_test,predictions):

    return {
        "mae": mean_absolute_error(y_test,predictions),
        "r2": r2_score(y_test,predictions)
    }