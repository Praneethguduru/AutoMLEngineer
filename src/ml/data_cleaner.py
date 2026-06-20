import pandas as pd


def clean_dataset(df, feature_report):

    cleaning_report = []

    for col in feature_report["numerical_columns"]:

        if pd.api.types.is_string_dtype(df[col]):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            cleaning_report.append(
                f"Converted {col} to numeric"
            )

        missing_count = df[col].isnull().sum()

        if missing_count > 0:

            median_value = df[col].median()

            df[col] = df[col].fillna(
                median_value
            )

            cleaning_report.append(
                f"Filled {missing_count} missing values in {col}"
            )

    for col in feature_report["categorical_columns"]:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            # Impute with mode
            mode_series = df[col].mode()
            if not mode_series.empty:
                fill_value = mode_series[0]
            else:
                fill_value = "Missing"
            df[col] = df[col].fillna(fill_value)
            cleaning_report.append(
                f"Filled {missing_count} missing values in categorical column {col} with '{fill_value}'"
            )

    return df, cleaning_report