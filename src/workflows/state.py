from typing import TypedDict


class AutoMLState(TypedDict):
    dataset_path: str

    df: object

    target_column: str

    problem_type: str

    dataset_report: dict

    model_results: dict

    best_model: dict

    consultant_report: list