from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_logistic_pipeline():

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    return pipeline