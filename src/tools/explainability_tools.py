def generate_consultant_report_tool(
    feature_report
):

    numerical_cols = feature_report.get(
        "numerical_columns",
        []
    )

    top_features = numerical_cols[:3]

    report = []

    if len(top_features) > 0:
        report.append(
            f"Top driver: {top_features[0]}"
        )

    if len(top_features) > 1:
        report.append(
            f"Second driver: {top_features[1]}"
        )

    if len(top_features) > 2:
        report.append(
            f"Third driver: {top_features[2]}"
        )

    return report