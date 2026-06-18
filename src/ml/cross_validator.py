from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from ml.pipeline_builder import build_logistic_pipeline

def cross_validate_model(df,target_column):

    X = df.drop(columns=[target_column])

    y = df[target_column]

    model = build_logistic_pipeline()

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="f1"
    )

    return {
        "mean_f1": scores.mean(),
        "std_f1": scores.std()
    }