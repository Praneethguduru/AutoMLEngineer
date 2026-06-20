import shap
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from ml.preprocessor_builder import build_preprocessor


def generate_shap_values(
    df,
    target_column,
    feature_report,
    problem_type="classification"
):

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    num_cols = feature_report["numerical_columns"]
    cat_cols = feature_report["categorical_columns"]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    if problem_type == "regression":
        model = RandomForestRegressor(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)

    X_preprocessed = pipeline.named_steps["preprocessor"].transform(X)
    if scipy.sparse.issparse(X_preprocessed):
        X_preprocessed = X_preprocessed.toarray()

    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"]
    )

    shap_values = explainer.shap_values(
        X_preprocessed
    )

    return {
        "model": pipeline.named_steps["model"],
        "explainer": explainer,
        "shap_values": shap_values,
        "preprocessed_features": X_preprocessed
    }