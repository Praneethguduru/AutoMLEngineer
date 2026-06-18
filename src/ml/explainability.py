import shap
from sklearn.ensemble import RandomForestClassifier


def generate_shap_values(
    df,
    target_column
):

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X, y)

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    return {
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values
    }