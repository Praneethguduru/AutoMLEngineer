import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def engineer_features(df, feature_report, target_column, problem_type="classification"):

    encoding_report = []

    # 1. Titanic-specific title extraction from 'Name'
    name_col = next((c for c in df.columns if c.lower() == "name"), None)
    if name_col:
        titles = df[name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
        titles = titles.fillna("Misc")
        title_mapping = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
            "Col": "Rare", "Don": "Rare", "Dr": "Rare",
            "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
            "Jonkheer": "Rare", "Dona": "Rare"
        }
        titles = titles.replace(title_mapping)
        df["Title"] = titles
        df = df.drop(columns=[name_col])
        if name_col in feature_report["categorical_columns"]:
            feature_report["categorical_columns"].remove(name_col)
        if "Title" not in feature_report["categorical_columns"]:
            feature_report["categorical_columns"].append("Title")
        encoding_report.append("Extracted Title from Name")

    # 2. Titanic-specific deck extraction from 'Cabin'
    cabin_col = next((c for c in df.columns if c.lower() == "cabin"), None)
    if cabin_col:
        # Extract first letter as deck, nan becomes 'U'
        decks = df[cabin_col].astype(str).str[0]
        decks = decks.replace("n", "U")
        df["Deck"] = decks
        df = df.drop(columns=[cabin_col])
        if cabin_col in feature_report["categorical_columns"]:
            feature_report["categorical_columns"].remove(cabin_col)
        if "Deck" not in feature_report["categorical_columns"]:
            feature_report["categorical_columns"].append("Deck")
        encoding_report.append("Extracted Deck from Cabin")

    # 3. Titanic-specific FamilySize and IsAlone from SibSp/Parch
    sibsp_col = next((c for c in df.columns if c.lower() == "sibsp"), None)
    parch_col = next((c for c in df.columns if c.lower() == "parch"), None)
    if sibsp_col and parch_col:
        df["FamilySize"] = df[sibsp_col] + df[parch_col] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        for new_feat in ["FamilySize", "IsAlone"]:
            if new_feat not in feature_report["numerical_columns"]:
                feature_report["numerical_columns"].append(new_feat)
        encoding_report.append("Engineered FamilySize and IsAlone features")

    # 4. Text Length and Word Count for categorical features
    for col in list(feature_report["categorical_columns"]):
        # Only extract text features if it contains long strings
        # e.g., if max length > 10
        sample_lengths = df[col].astype(str).str.len()
        if sample_lengths.max() > 10:
            df[f"{col}_len"] = sample_lengths
            df[f"{col}_words"] = df[col].astype(str).str.split().str.len()
            for new_feat in [f"{col}_len", f"{col}_words"]:
                if new_feat not in feature_report["numerical_columns"]:
                    feature_report["numerical_columns"].append(new_feat)
            encoding_report.append(f"Extracted length and word count for {col}")

    # 5. Frequency encoding for categorical features
    for col in list(feature_report["categorical_columns"]):
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
        if f"{col}_freq" not in feature_report["numerical_columns"]:
            feature_report["numerical_columns"].append(f"{col}_freq")
        encoding_report.append(f"Applied frequency encoding to {col}")

    # 6. Datetime features extraction
    dt_cols = feature_report.get("datetime_columns", [])
    for col in list(dt_cols):
        dt_series = pd.to_datetime(df[col], format='mixed', errors='coerce')
        df[f"{col}_year"] = dt_series.dt.year.fillna(0).astype(int)
        df[f"{col}_month"] = dt_series.dt.month.fillna(0).astype(int)
        df[f"{col}_day"] = dt_series.dt.day.fillna(0).astype(int)
        df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
        df[f"{col}_hour"] = dt_series.dt.hour.fillna(0).astype(int)

        df = df.drop(columns=[col])
        if col in feature_report["datetime_columns"]:
            feature_report["datetime_columns"].remove(col)

        new_features = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek", f"{col}_hour"]
        for new_feat in new_features:
            if new_feat not in feature_report["numerical_columns"]:
                feature_report["numerical_columns"].append(new_feat)
        encoding_report.append(f"Parsed datetime column {col} into numeric components")

    # 7. Log transformation for skewed numerical columns
    for col in list(feature_report["numerical_columns"]):
        if df[col].nunique() > 1:
            try:
                skew = df[col].skew()
                if abs(skew) > 1.5:
                    shifted = df[col] - df[col].min()
                    df[f"{col}_log"] = np.log1p(shifted)
                    if f"{col}_log" not in feature_report["numerical_columns"]:
                        feature_report["numerical_columns"].append(f"{col}_log")
                    encoding_report.append(f"Applied log1p transformation to skewed column {col}")
            except:
                pass

    # 8. Encode Target Column
    if problem_type == "classification":
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column])
        encoding_report.append(f"Encoded target column {target_column} for classification")
    else:
        encoding_report.append(f"Left target column {target_column} as-is for regression")

    return (df, encoding_report)