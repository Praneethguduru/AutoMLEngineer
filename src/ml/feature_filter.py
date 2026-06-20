def remove_identifier_columns(
    df
):

    columns_to_drop = []

    for col in df.columns:

        col_lower = col.lower()

        if (
            "id" in col_lower
            or col_lower.endswith("id")
        ):
            columns_to_drop.append(col)
            continue

        # Drop other high-cardinality object/categorical columns that are likely identifiers (e.g. Ticket)
        # Keep 'name' to extract features (Title) later in the pipeline
        if str(df[col].dtype) in ["object", "str", "string"] and col_lower != "name":
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                columns_to_drop.append(col)

    df = df.drop(
        columns=columns_to_drop
    )

    return df, columns_to_drop