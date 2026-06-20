from sklearn.preprocessing import LabelEncoder


def engineer_features(df, feature_report, target_column, problem_type="classification"):

    encoding_report = []

    # Titanic-specific title extraction from 'Name'
    name_col = None
    for col in df.columns:
        if col.lower() == "name":
            name_col = col
            break

    if name_col:
        # Extract title (e.g. Mr, Mrs, Miss, Master)
        titles = df[name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
        titles = titles.fillna("Misc")
        title_mapping = {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Rare",
            "Countess": "Rare",
            "Capt": "Rare",
            "Col": "Rare",
            "Don": "Rare",
            "Dr": "Rare",
            "Major": "Rare",
            "Rev": "Rare",
            "Sir": "Rare",
            "Jonkheer": "Rare",
            "Dona": "Rare"
        }
        titles = titles.replace(title_mapping)
        
        df["Title"] = titles
        encoding_report.append("Extracted and grouped Title from Name column")
        
        df = df.drop(columns=[name_col])
        
        # Update feature_report
        if name_col in feature_report["categorical_columns"]:
            feature_report["categorical_columns"].remove(name_col)
        if "Title" not in feature_report["categorical_columns"]:
            feature_report["categorical_columns"].append("Title")

    if problem_type == "classification":
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column])
        encoding_report.append(f"Encoded target column {target_column} for classification")
    else:
        encoding_report.append(f"Left target column {target_column} as-is for regression")

    return (df, encoding_report)