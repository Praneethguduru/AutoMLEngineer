from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


def generate_confusion_report(
    y_test,
    predictions
):

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    return {
        "confusion_matrix": matrix,
        "classification_report": report
    }