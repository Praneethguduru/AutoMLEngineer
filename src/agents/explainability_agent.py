from workflows.state import AutoMLState

from tools.explainability_tools import (
    generate_consultant_report_tool
)


def explainability_node(
    state: AutoMLState
):

    consultant_report = (
        generate_consultant_report_tool(
            state["feature_report"]
        )
    )

    return {
        **state,
        "consultant_report": consultant_report
    }