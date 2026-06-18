from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ml.model_evaluater import evaluate_model
from ml.pipeline_builder import build_logistic_pipeline


def train_models(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = {}

    # Logistic Regression
    lr_pipeline = (build_logistic_pipeline())

    lr_pipeline.fit(X_train,y_train)

    lr_preds = lr_pipeline.predict(X_test)

    

    results["Logistic Regression"] = {"metrics": evaluate_model(y_test,lr_preds),
    "y_test": y_test,
    "predictions": lr_preds
}

    # Random Forest
    rf = RandomForestClassifier(
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_preds = rf.predict(X_test)

    results["Random Forest"] = {"metrics": evaluate_model(y_test,rf_preds),
    "y_test": y_test,
    "predictions": rf_preds
}

    return results