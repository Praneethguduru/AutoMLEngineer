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

def main():
    # loading data
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # analyzing data
    report = analyze_dataset(df)
    
    for key, value in report.items():
        print(f"{key}: {value}")
    
    
    df, removed_columns = (remove_identifier_columns(df))

    print("\nREMOVED IDENTIFIERS")

    print(removed_columns)

    # classifying features
    feature_report = classify_features(df, 'Churn')
    print(feature_report)
    
    # cleaning data
    df, cleaning_report = clean_dataset(df, feature_report)
    print("\nCLEANING REPORT")

    for item in cleaning_report:
        print(item)
    
    # encoding features
    df, encoding_report = engineer_features(df, feature_report, "Churn")

    print("\nFEATURE ENGINEERING REPORT")

    for item in encoding_report:
        print(item)

    # training models
    model_results = train_models(df,"Churn")

    print("\nMODEL RESULTS")

    for model, result in model_results.items():

        print(f"\n{model}")

        for metric, value in result["metrics"].items():

            print(
                f"{metric}: {value:.4f}"
            )

    # selecting best model
    best_model = select_best_model(model_results)
    print("\nBEST MODEL")
    print(best_model)
    
    lr_results = model_results['Logistic Regression']

    confusion_report = (generate_confusion_report(
        lr_results["y_test"],
        lr_results["predictions"])
    )
    print("\nCONFUSION MATRIX")
    print(confusion_report["confusion_matrix"])

    print("\nCLASSIFICATION REPORT")
    print(confusion_report["classification_report"])








    cv_results = cross_validate_model(df,"Churn")

    print("\nCROSS VALIDATION")

    for key, value in cv_results.items():
        print(f"{key}: {value:.4f}")

    feature_importance = get_feature_importance(df, "Churn")

    print("\nTOP FEATURES")

    for feature, score in list(feature_importance.items())[:10]:

        print(f"{feature}: {score:.4f}")
    
    insights = generate_insights(feature_importance)

    print("\nCONSULTANT REPORT")

    for insight in insights:
        print(insight)

if __name__ == "__main__":
    main()