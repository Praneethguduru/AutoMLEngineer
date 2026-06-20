from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
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

        # Ridge Regression
        ridge_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge())
        ])
        ridge_pipeline.fit(X_train, y_train)
        ridge_preds = ridge_pipeline.predict(X_test)
        results["Ridge"] = {
            "metrics": evaluate_regression(y_test, ridge_preds),
            "y_test": y_test,
            "predictions": ridge_preds,
            "pipeline": ridge_pipeline
        }

        # Lasso Regression
        lasso_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", Lasso())
        ])
        lasso_pipeline.fit(X_train, y_train)
        lasso_preds = lasso_pipeline.predict(X_test)
        results["Lasso"] = {
            "metrics": evaluate_regression(y_test, lasso_preds),
            "y_test": y_test,
            "predictions": lasso_preds,
            "pipeline": lasso_pipeline
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

        # Gradient Boosting Regressor
        gb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42))
        ])
        gb_pipeline.fit(X_train, y_train)
        gb_preds = gb_pipeline.predict(X_test)
        results["Gradient Boosting"] = {
            "metrics": evaluate_regression(y_test, gb_preds),
            "y_test": y_test,
            "predictions": gb_preds,
            "pipeline": gb_pipeline
        }

        # AdaBoost Regressor
        ada_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", AdaBoostRegressor(random_state=42))
        ])
        ada_pipeline.fit(X_train, y_train)
        ada_preds = ada_pipeline.predict(X_test)
        results["AdaBoost"] = {
            "metrics": evaluate_regression(y_test, ada_preds),
            "y_test": y_test,
            "predictions": ada_preds,
            "pipeline": ada_pipeline
        }

        # KNN Regressor
        knn_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", KNeighborsRegressor(n_neighbors=5))
        ])
        knn_pipeline.fit(X_train, y_train)
        knn_preds = knn_pipeline.predict(X_test)
        results["KNN Regressor"] = {
            "metrics": evaluate_regression(y_test, knn_preds),
            "y_test": y_test,
            "predictions": knn_preds,
            "pipeline": knn_pipeline
        }

        # SVM Regressor
        svm_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVR(kernel="rbf"))
        ])
        svm_pipeline.fit(X_train, y_train)
        svm_preds = svm_pipeline.predict(X_test)
        results["SVM Regressor"] = {
            "metrics": evaluate_regression(y_test, svm_preds),
            "y_test": y_test,
            "predictions": svm_preds,
            "pipeline": svm_pipeline
        }

        # XGBoost Regressor
        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(random_state=42))
        ])
        xgb_pipeline.fit(X_train, y_train)
        xgb_preds = xgb_pipeline.predict(X_test)
        results["XGBoost"] = {
            "metrics": evaluate_regression(y_test, xgb_preds),
            "y_test": y_test,
            "predictions": xgb_preds,
            "pipeline": xgb_pipeline
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

        # Gradient Boosting
        gb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42))
        ])
        gb_pipeline.fit(X_train, y_train)
        gb_preds = gb_pipeline.predict(X_test)
        results["Gradient Boosting"] = {
            "metrics": evaluate_model(y_test, gb_preds),
            "y_test": y_test,
            "predictions": gb_preds,
            "pipeline": gb_pipeline
        }

        # AdaBoost
        ada_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", AdaBoostClassifier(random_state=42))
        ])
        ada_pipeline.fit(X_train, y_train)
        ada_preds = ada_pipeline.predict(X_test)
        results["AdaBoost"] = {
            "metrics": evaluate_model(y_test, ada_preds),
            "y_test": y_test,
            "predictions": ada_preds,
            "pipeline": ada_pipeline
        }

        # XGBoost
        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
        ])
        xgb_pipeline.fit(X_train, y_train)
        xgb_preds = xgb_pipeline.predict(X_test)
        results["XGBoost"] = {
            "metrics": evaluate_model(y_test, xgb_preds),
            "y_test": y_test,
            "predictions": xgb_preds,
            "pipeline": xgb_pipeline
        }

    # Voting Ensemble of the top 3 models
    if problem_type == "regression":
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"].get("r2", -float("inf")),
            reverse=True
        )
    else:
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"].get("f1_score", -float("inf")),
            reverse=True
        )

    top_models = sorted_models[:3]
    if len(top_models) >= 2:
        estimators = [(name, res["pipeline"]) for name, res in top_models]
        if problem_type == "regression":
            voting_model = VotingRegressor(estimators=estimators)
            voting_model.fit(X_train, y_train)
            voting_preds = voting_model.predict(X_test)
            results["Voting Ensemble"] = {
                "metrics": evaluate_regression(y_test, voting_preds),
                "y_test": y_test,
                "predictions": voting_preds,
                "pipeline": voting_model
            }
        else:
            voting_model = VotingClassifier(estimators=estimators, voting="hard")
            voting_model.fit(X_train, y_train)
            voting_preds = voting_model.predict(X_test)
            results["Voting Ensemble"] = {
                "metrics": evaluate_model(y_test, voting_preds),
                "y_test": y_test,
                "predictions": voting_preds,
                "pipeline": voting_model
            }

    return results, (X_train, X_test, y_train, y_test)