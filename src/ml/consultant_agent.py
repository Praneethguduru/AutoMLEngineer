def generate_insights(
    feature_importance
):

    top_features = list(
        feature_importance.keys()
    )[:3]

    insights = []

    insights.append(
        f"Top driver: {top_features[0]}"
    )

    insights.append(
        f"Second driver: {top_features[1]}"
    )

    insights.append(
        f"Third driver: {top_features[2]}"
    )

    return insights