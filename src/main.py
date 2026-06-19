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

def main():
    # loading data
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # analyzing data
    report = analyze_dataset(df)
    
    df, removed_columns = remove_identifier_columns(df)

    # detecting target column
    target_column = detect_target_column(df)
    print(f"Detected Target Column: {target_column}")

    # classifying features
    feature_report = classify_features(df, target_column)
    
    # cleaning data
    df, cleaning_report = clean_dataset(df, feature_report)
    
    # encoding features
    df, encoding_report = engineer_features(df, feature_report, target_column)

    # training models
    model_results = train_models(df, target_column)

    # selecting best model
    best_model = select_best_model(model_results)
    print(f"\nBest Model: {best_model['best_model']}")
    for metric, value in best_model["metrics"].items():
        print(f"  {metric}: {value:.4f}")
    
    best_model_name = best_model["best_model"]
    best_results = model_results[best_model_name]

    confusion_report = generate_confusion_report(
        best_results["y_test"],
        best_results["predictions"]
    )
    
    print("\nConfusion Matrix:")
    print(confusion_report["confusion_matrix"])

    print("\nClassification Report:")
    print(confusion_report["classification_report"])

    cv_results = cross_validate_model(df, target_column)

    print("Cross Validation (Logistic Regression):")
    for key, value in cv_results.items():
        print(f"  {key}: {value:.4f}")

    feature_importance = get_feature_importance(df, target_column)
    insights = generate_insights(feature_importance)

    print("\nConsultant Report / Key Drivers:")
    for insight in insights:
        print(f"  - {insight}")

if __name__ == "__main__":
    main()