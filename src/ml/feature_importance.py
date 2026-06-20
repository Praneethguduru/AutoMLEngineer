from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from ml.preprocessor_builder import build_preprocessor


def get_feature_importance(
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

    importances = pipeline.named_steps["model"].feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    importance = {}
    for name, score in zip(feature_names, importances):
        if name.startswith("num__"):
            orig_name = name[5:]
        elif name.startswith("cat__"):
            col_name_part = name[5:]
            orig_name = col_name_part
            for cat_col in cat_cols:
                if col_name_part.startswith(cat_col):
                    orig_name = cat_col
                    break
        else:
            orig_name = name

        importance[orig_name] = importance.get(orig_name, 0.0) + score

    sorted_importance = dict(
        sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return sorted_importance