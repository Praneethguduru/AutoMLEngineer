import sys
import os
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")
from ml import feature_importance
from ml.data_loader import load_data
from ml.analyzer import analyze_dataset
from ml.features_classifier import classify_features
from ml.data_cleaner import clean_dataset
from ml.feature_engineer import engineer_features
from ml.model_trainer import train_models
from ml.model_selector import select_best_model
from ml.cross_validator import cross_validate_model
from ml.feature_importance import get_feature_importance
from ml.feature_filter import remove_identifier_columns
from ml.consultant_agent import generate_insights
from ml.confusion_matrix_report import generate_confusion_report
from ml.target_detector import detect_target_column
from ml.problem_detector import detect_problem_type
from ml.result_exporter import export_results
from ml.model_tuner import tune_model
import pickle
import shutil
from ml.report_generator import (
    generate_model_comparison_chart,
    generate_feature_importance_chart,
    generate_confusion_matrix_chart
)


def run_automl_pipeline(dataset_path, target_override=None):
    print(f"\n==========================================")
    print(f"Running AutoML Pipeline for: {dataset_path}")
    print(f"==========================================")
    
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} does not exist.")
        return

    # loading data
    df = load_data(dataset_path)

    # removing identifier columns
    df, removed_columns = remove_identifier_columns(df)

    # detecting target column
    if target_override and target_override in df.columns:
        target_column = target_override
        print(f"Using Specified Target Column: {target_column}")
    else:
        target_column = detect_target_column(df)
        print(f"Detected Target Column: {target_column}")

    # Drop rows where target column is null
    original_len = len(df)
    df = df.dropna(subset=[target_column])
    dropped_nulls = original_len - len(df)
    if dropped_nulls > 0:
        print(f"Dropped {dropped_nulls} rows with null targets.")

    if len(df) == 0:
        print(f"Error: All rows have null targets for column '{target_column}'")
        return

    # analyzing data
    report = analyze_dataset(df)

    # detecting problem type
    problem_type = detect_problem_type(df, target_column)
    print(f"Problem Type: {problem_type}")

    # classifying features
    feature_report = classify_features(df, target_column)
    
    # cleaning data
    df, cleaning_report = clean_dataset(df, feature_report)
    
    # encoding features
    df, encoding_report = engineer_features(df, feature_report, target_column, problem_type)

    # training models
    model_results, (X_train, X_test, y_train, y_test) = train_models(df, target_column, problem_type, feature_report)

    print("\nModel Results:")
    for model_name, res in model_results.items():
        print(f"  {model_name}:")
        for metric, value in res["metrics"].items():
            print(f"    {metric}: {value:.4f}")

    # selecting best model
    best_model = select_best_model(model_results)
    print(f"\nBest Model: {best_model['best_model']}")
    for metric, value in best_model["metrics"].items():
        print(f"  {metric}: {value:.4f}")
    
    best_model_name = best_model["best_model"]
    best_results = model_results[best_model_name]
    best_pipeline = best_results["pipeline"]

    # Hyperparameter Tuning
    model_to_tune_name = best_model_name
    model_to_tune_pipeline = best_pipeline
    if best_model_name == "Voting Ensemble":
        individual_results = {k: v for k, v in model_results.items() if k != "Voting Ensemble"}
        best_ind = select_best_model(individual_results)
        model_to_tune_name = best_ind["best_model"]
        model_to_tune_pipeline = individual_results[model_to_tune_name]["pipeline"]
        print(f"\nBest model is Voting Ensemble. Tuning best individual model instead: {model_to_tune_name}")
    
    tuned_pipeline, best_params = tune_model(model_to_tune_pipeline, model_to_tune_name, X_train, y_train, problem_type)
    
    # Evaluate tuned model
    if best_params:
        from ml.model_evaluater import evaluate_model
        from ml.regression_evaluator import evaluate_regression
        
        tuned_predictions = tuned_pipeline.predict(X_test)
        if problem_type == "regression":
            tuned_metrics = evaluate_regression(y_test, tuned_predictions)
        else:
            tuned_metrics = evaluate_model(y_test, tuned_predictions)
            
        print("\nTuned Model Metrics:")
        for metric, value in tuned_metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        # Compare original best and tuned model
        metric_to_compare = "r2" if problem_type == "regression" else "f1_score"
        original_metric_val = best_model["metrics"][metric_to_compare]
        tuned_metric_val = tuned_metrics.get(metric_to_compare, -float("inf"))
        
        if tuned_metric_val > original_metric_val:
            print(f"\nTuned model performed better ({tuned_metric_val:.4f} vs {original_metric_val:.4f}). Using tuned model as the final pipeline.")
            final_pipeline = tuned_pipeline
            final_model_name = f"Tuned {model_to_tune_name}"
            final_metrics = tuned_metrics
            
            # Update best_model and model_results
            best_model = {
                "best_model": final_model_name,
                "metrics": final_metrics
            }
            model_results[final_model_name] = {
                "metrics": final_metrics,
                "y_test": y_test,
                "predictions": tuned_predictions,
                "pipeline": final_pipeline
            }
        else:
            print(f"\nOriginal best model performed better or equal ({original_metric_val:.4f} vs {tuned_metric_val:.4f}). Using {best_model_name} as the final pipeline.")
            final_pipeline = best_pipeline
            final_model_name = best_model_name
            final_metrics = best_model["metrics"]
    else:
        final_pipeline = best_pipeline
        final_model_name = best_model_name
        final_metrics = best_model["metrics"]

    # Model Persistence
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    model_save_path = f"outputs/{dataset_name}_best_model.pkl"
    os.makedirs("outputs", exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(final_pipeline, f)
    print(f"Fitted best model pipeline saved to: {model_save_path}")
    try:
        shutil.copy2(model_save_path, "outputs/best_model.pkl")
    except Exception as e:
        print(f"Warning: Could not copy best_model.pkl: {e}")

    # Generate Model Comparison Chart
    try:
        generate_model_comparison_chart(model_results, problem_type, dataset_name)
    except Exception as e:
        print(f"Warning: Could not generate model comparison chart: {e}")

    # Bind variables back for cross-validation and confusion reporting
    best_results = model_results[final_model_name]
    best_pipeline = final_pipeline
    best_model_name = final_model_name


    cv_results = {}
    if problem_type == "classification":
        confusion_report = generate_confusion_report(
            best_results["y_test"],
            best_results["predictions"]
        )
        
        print("\nConfusion Matrix:")
        print(confusion_report["confusion_matrix"])

        print("\nClassification Report:")
        print(confusion_report["classification_report"])

        # Generate Confusion Matrix Chart
        try:
            generate_confusion_matrix_chart(best_results["y_test"], best_results["predictions"], dataset_name)
        except Exception as e:
            print(f"Warning: Could not generate confusion matrix heatmap: {e}")

        cv_results = cross_validate_model(df, target_column, problem_type, best_pipeline)

        print(f"Cross Validation ({best_model_name}):")
        for key, value in cv_results.items():
            print(f"  {key}: {value:.4f}")
    else:
        cv_results = cross_validate_model(df, target_column, problem_type, best_pipeline)

        print(f"Cross Validation ({best_model_name}):")
        for key, value in cv_results.items():
            print(f"  {key}: {value:.4f}")

    feature_importance = get_feature_importance(df, target_column, feature_report, problem_type)
    insights = generate_insights(feature_importance)

    print("\nConsultant Report / Key Drivers:")
    for insight in insights:
        print(f"  - {insight}")

    # Generate Feature Importance Chart
    try:
        generate_feature_importance_chart(feature_importance, dataset_name)
    except Exception as e:
        print(f"Warning: Could not generate feature importance chart: {e}")


    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_file = f"outputs/{dataset_name}_results.json"

    export_path = export_results(
        target_column=target_column,
        problem_type=problem_type,
        best_model=best_model,
        all_models=model_results,
        cv_results=cv_results,
        consultant_report=insights,
        output_file=output_file
    )
    print(f"\nResults exported to: {export_path}")
    try:
        shutil.copy2(export_path, "outputs/results.json")
    except Exception as e:
        print(f"Warning: Could not copy results.json: {e}")


def main():
    parser = argparse.ArgumentParser(description="AutoML Pipeline Runner")
    parser.add_argument("datasets", nargs="*", help="Optional dataset paths. If not specified, scans data/raw/")
    parser.add_argument("--target", default=None, help="Explicit target column name (optional)")
    args = parser.parse_args()

    dataset_paths = args.datasets
    if not dataset_paths:
        raw_dir = "data/raw"
        if os.path.exists(raw_dir):
            dataset_paths = glob.glob(os.path.join(raw_dir, "*.csv"))
        else:
            dataset_paths = []

    if not dataset_paths:
        print("No datasets found to process. Please place CSV files in 'data/raw' or pass paths as arguments.")
        return

    for path in dataset_paths:
        try:
            run_automl_pipeline(path, target_override=args.target)
        except Exception as e:
            print(f"\nError running AutoML pipeline on {path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()