from typing import TypedDict


class AutoMLState(TypedDict):
    dataset_path: str

    target_column: str

    problem_type: str

    dataset_report: dict

    feature_report: dict

    model_results: dict

    best_model: dict

    consultant_report: list

    final_report: dict