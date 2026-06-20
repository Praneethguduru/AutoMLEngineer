import pandas as pd


def classify_features(df, target_column):

    numerical_columns = []
    categorical_columns = []
    datetime_columns = []

    for col in df.columns:

        # Skip target column
        if col == target_column:
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
            datetime_columns.append(col)
            continue

        # Handle string/object columns
        if str(df[col].dtype) in ["object", "str", "string"]:

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            success_rate = (
                converted.notnull().sum()
                / len(df)
            )

            if success_rate > 0.90:
                numerical_columns.append(col)

            else:
                categorical_columns.append(col)

        # Handle actual numeric columns
        else:

            numerical_columns.append(col)

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "target": target_column
    }