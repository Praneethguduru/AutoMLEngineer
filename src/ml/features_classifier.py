import pandas as pd


def classify_features(df, target_column):

    numerical_columns = []
    categorical_columns = []

    for col in df.columns:

        # Skip target column
        if col == target_column:
            continue

        # Handle string/object columns
        if str(df[col].dtype) in ["object", "str"]:

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
        "target": target_column
    }