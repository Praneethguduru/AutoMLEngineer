from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def tune_model(pipeline, model_name, X_train, y_train, problem_type):
    # If the model is Voting Ensemble, we do not tune it directly
    if "Voting" in model_name:
        print(f"Skipping tuning for Voting Ensemble.")
        return pipeline, {}

    # Define param grid based on model_name
    param_distributions = {}
    if "Random Forest" in model_name:
        param_distributions = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    elif "XGBoost" in model_name:
        param_distributions = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2]
        }
    elif "Gradient Boosting" in model_name:
        param_distributions = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7]
        }
    elif "Logistic Regression" in model_name:
        param_distributions = {
            "model__C": [0.01, 0.1, 1.0, 10.0]
        }
    elif "Ridge" in model_name or "Lasso" in model_name:
        param_distributions = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0]
        }
    elif "KNN" in model_name:
        param_distributions = {
            "model__n_neighbors": [3, 5, 11, 15],
            "model__weights": ["uniform", "distance"]
        }
    elif "SVM" in model_name:
        param_distributions = {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"]
        }
    elif "Decision Tree" in model_name:
        param_distributions = {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    elif "AdaBoost" in model_name:
        param_distributions = {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.1, 1.0]
        }

    if not param_distributions:
        print(f"No tuning grid defined for {model_name}. Returning original model.")
        return pipeline, {}

    # Scoring metric
    if problem_type == "regression":
        scoring = "r2"
    else:
        is_multiclass = y_train.nunique() > 2
        scoring = "f1_macro" if is_multiclass else "f1"

    print(f"Tuning {model_name} using RandomizedSearchCV (scoring='{scoring}')...")
    
    # We set n_iter=5 to make it fast but effective, cv=3
    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=5,
        cv=3,
        scoring=scoring,
        random_state=42
    )
    
    try:
        rs.fit(X_train, y_train)
        print(f"Tuning complete. Best params: {rs.best_params_}")
        return rs.best_estimator_, rs.best_params_
    except Exception as e:
        print(f"Error during tuning: {e}. Returning original model.")
        return pipeline, {}
