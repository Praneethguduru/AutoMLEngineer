import pandas as pd


def classify_features(df, target_column):

    numerical_columns = []
    categorical_columns = []

    print("\n=== FEATURE CLASSIFICATION ===\n")

    for col in df.columns:

        # Skip target column
        if col == target_column:
            continue

        print(f"{col} -> {df[col].dtype}")

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

            print(
                f"   Numeric Conversion Success Rate: "
                f"{success_rate:.2f}"
            )

            if success_rate > 0.90:
                numerical_columns.append(col)
                print("   -> Classified as NUMERICAL")

            else:
                categorical_columns.append(col)
                print("   -> Classified as CATEGORICAL")

        # Handle actual numeric columns
        else:

            numerical_columns.append(col)

            #print(
            #    "   -> Classified as NUMERICAL "
            #    "(native numeric dtype)"
            #)

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target": target_column
    }