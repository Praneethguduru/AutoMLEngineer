from workflows.state import AutoMLState

from tools.model_selection_tools import (
    select_best_model_tool
)


def model_selector_node(
    state: AutoMLState
):

    best_model = select_best_model_tool(
        state["model_results"]
    )

    return {
        **state,
        "best_model": best_model
    }