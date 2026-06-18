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

    df = df.drop(
        columns=columns_to_drop
    )

    return df, columns_to_drop