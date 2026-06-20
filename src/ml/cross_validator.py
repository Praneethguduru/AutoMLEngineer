from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from ml.pipeline_builder import build_logistic_pipeline

def cross_validate_model(df, target_column, problem_type, best_pipeline):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if problem_type == "regression":
        scores = cross_val_score(
            best_pipeline,
            X,
            y,
            cv=5,
            scoring="r2"
        )
        return {
            "mean_r2": scores.mean(),
            "std_r2": scores.std()
        }
    else:
        is_multiclass = y.nunique() > 2
        scoring_metric = "f1_macro" if is_multiclass else "f1"
        scores = cross_val_score(
            best_pipeline,
            X,
            y,
            cv=5,
            scoring=scoring_metric
        )
        metric_name = "f1_macro" if is_multiclass else "f1"
        return {
            f"mean_{metric_name}": scores.mean(),
            f"std_{metric_name}": scores.std()
        }