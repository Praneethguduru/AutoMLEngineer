from ml.data_loader import load_data
from workflows.state import AutoMLState
from tools.training_tools import (
    train_models_tool,
    select_best_model_tool
)


def model_trainer_node(state: AutoMLState):

    # Train models
    model_results = train_models_tool(
        load_data(state["dataset_path"]),
        state["target_column"],
        state["problem_type"],
        state["feature_report"]
    )

    # Keep only metrics in LangGraph state
    clean_model_results = {}

    for model_name, result in model_results.items():
        clean_model_results[model_name] = {
            "metrics": result["metrics"]
        }

    # Select best model
    best_model = select_best_model_tool(
        clean_model_results
    )

    return {
        **state,
        "model_results": clean_model_results,
        "best_model": best_model
    }