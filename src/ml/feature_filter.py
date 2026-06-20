import pandas as pd

def remove_identifier_columns(
    df
):

    columns_to_drop = []

    for col in df.columns:

        col_lower = col.lower()

        # Drop ID columns
        if (
            "id" in col_lower
            or col_lower.endswith("id")
        ):
            columns_to_drop.append(col)
            continue

        # Drop constant columns (0 or 1 unique value, including all-null columns)
        if df[col].nunique(dropna=True) <= 1:
            columns_to_drop.append(col)
            continue

        # Check if it is a datetime column
        is_datetime = False
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            is_datetime = True
        elif str(df[col].dtype) in ["object", "str", "string"]:
            sample_non_null = df[col].dropna()
            if not sample_non_null.empty and any(isinstance(x, str) and ('-' in x or '/' in x or ':' in x) for x in sample_non_null.head(10)):
                try:
                    converted_dt = pd.to_datetime(df[col], format='mixed', errors='coerce')
                    if converted_dt.notnull().sum() / len(df) > 0.8:
                        is_datetime = True
                except:
                    pass

        if is_datetime:
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