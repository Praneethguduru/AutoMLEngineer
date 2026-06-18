def select_best_model(model_results):

    best_model = max(
        model_results,
        key=lambda model: model_results[
            model
        ]["metrics"]["f1_score"]
    )

    return {
        "best_model": best_model,
        "metrics": model_results[
            best_model
        ]["metrics"]
    }