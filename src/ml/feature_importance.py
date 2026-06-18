from sklearn.ensemble import RandomForestClassifier


def get_feature_importance(
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

    importance = {}

    for feature, score in zip(
        X.columns,
        model.feature_importances_
    ):
        importance[feature] = score

    sorted_importance = dict(
        sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return sorted_importance