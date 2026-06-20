from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from ml.model_evaluater import evaluate_model
from ml.preprocessor_builder import build_preprocessor
from ml.regression_evaluator import evaluate_regression


def train_models(df, target_column, problem_type, feature_report):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    num_cols = feature_report["numerical_columns"]
    cat_cols = feature_report["categorical_columns"]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    results = {}

    if problem_type == "regression":
        # Linear Regression
        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])
        lr_pipeline.fit(X_train, y_train)
        predictions = lr_pipeline.predict(X_test)

        results["Linear Regression"] = {
            "metrics": evaluate_regression(y_test, predictions),
            "y_test": y_test,
            "predictions": predictions,
            "pipeline": lr_pipeline
        }

        # Decision Tree Regressor
        dt_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42))
        ])
        dt_pipeline.fit(X_train, y_train)
        dt_preds = dt_pipeline.predict(X_test)

        results["Decision Tree"] = {
            "metrics": evaluate_regression(y_test, dt_preds),
            "y_test": y_test,
            "predictions": dt_preds,
            "pipeline": dt_pipeline
        }

        # Random Forest Regressor
        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42))
        ])
        rf_pipeline.fit(X_train, y_train)
        rf_preds = rf_pipeline.predict(X_test)

        results["Random Forest"] = {
            "metrics": evaluate_regression(y_test, rf_preds),
            "y_test": y_test,
            "predictions": rf_preds,
            "pipeline": rf_pipeline
        }

    else:
        # Logistic Regression
        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ])
        lr_pipeline.fit(X_train, y_train)
        lr_preds = lr_pipeline.predict(X_test)

        results["Logistic Regression"] = {
            "metrics": evaluate_model(y_test, lr_preds),
            "y_test": y_test,
            "predictions": lr_preds,
            "pipeline": lr_pipeline
        }

        # Random Forest
        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42))
        ])
        rf_pipeline.fit(X_train, y_train)
        rf_preds = rf_pipeline.predict(X_test)

        results["Random Forest"] = {
            "metrics": evaluate_model(y_test, rf_preds),
            "y_test": y_test,
            "predictions": rf_preds,
            "pipeline": rf_pipeline
        }

        # Decision Tree
        dt_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42))
        ])
        dt_pipeline.fit(X_train, y_train)
        dt_preds = dt_pipeline.predict(X_test)

        results["Decision Tree"] = {
            "metrics": evaluate_model(y_test, dt_preds),
            "y_test": y_test,
            "predictions": dt_preds,
            "pipeline": dt_pipeline
        }

        # KNN
        knn_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
        knn_pipeline.fit(X_train, y_train)
        knn_preds = knn_pipeline.predict(X_test)

        results["KNN"] = {
            "metrics": evaluate_model(y_test, knn_preds),
            "y_test": y_test,
            "predictions": knn_preds,
            "pipeline": knn_pipeline
        }
        
        # SVM
        svm_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVC(kernel="rbf", random_state=42))
        ])
        svm_pipeline.fit(X_train, y_train)
        svm_preds = svm_pipeline.predict(X_test)

        results["SVM"] = {
            "metrics": evaluate_model(y_test, svm_preds),
            "y_test": y_test,
            "predictions": svm_preds,
            "pipeline": svm_pipeline
        }

    return results