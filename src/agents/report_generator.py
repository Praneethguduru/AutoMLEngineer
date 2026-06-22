from workflows.state import AutoMLState


def report_generator_node(
    state: AutoMLState
):

    report = {
        "target": state["target_column"],
        "problem_type": state["problem_type"],
        "best_model": state["best_model"],
        "consultant_report": state["consultant_report"]
    }

    return {
        **state,
        "final_report": report
    }