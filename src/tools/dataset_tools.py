from ml.analyzer import analyze_dataset
from ml.target_detector import detect_target_column
from ml.problem_detector import detect_problem_type


def analyze_dataset_tool(df):
    return analyze_dataset(df)


def detect_target_tool(df):
    return detect_target_column(df)


def detect_problem_type_tool(df, target):
    return detect_problem_type(df, target)