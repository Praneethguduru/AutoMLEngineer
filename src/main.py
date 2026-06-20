import sys
import os
import glob
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


def run_automl_pipeline(dataset_path):
    print(f"\n==========================================")
    print(f"Running AutoML Pipeline for: {dataset_path}")
    print(f"==========================================")
    
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} does not exist.")
        return

    # loading data
    df = load_data(dataset_path)

    # analyzing data
    report = analyze_dataset(df)
    
    df, removed_columns = remove_identifier_columns(df)

    # detecting target column
    target_column = detect_target_column(df)
    print(f"Detected Target Column: {target_column}")

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
    model_results = train_models(df, target_column, problem_type, feature_report)

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


def main():
    if len(sys.argv) > 1:
        dataset_paths = sys.argv[1:]
    else:
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
            run_automl_pipeline(path)
        except Exception as e:
            print(f"\nError running AutoML pipeline on {path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()