from sklearn.preprocessing import LabelEncoder


def engineer_features(df, feature_report, target_column):

    encoding_report = []

    for col in feature_report["categorical_columns"]:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col].astype(str))

        encoding_report.append(f"Encoded {col}")

    target_encoder = LabelEncoder()

    df[target_column] = (target_encoder.fit_transform(df[target_column]))

    return (df, encoding_report)