def select_best_model_tool(model_results):
    best_name = max(
        model_results,
        key=lambda x: model_results[x]["metrics"]["f1_score"]
    )

    return {
        "best_model": best_name,
        "metrics": model_results[best_name]["metrics"]
    }