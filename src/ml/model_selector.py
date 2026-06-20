def select_best_model(model_results):

    sample_model = next(iter(model_results))
    metrics = model_results[sample_model]["metrics"]

    if "f1_score" in metrics:
        best_model = max(
            model_results,
            key=lambda model: model_results[model]["metrics"]["f1_score"]
        )
    elif "r2" in metrics:
        best_model = max(
            model_results,
            key=lambda model: model_results[model]["metrics"]["r2"]
        )
    else:
        best_model = sample_model

    return {
        "best_model": best_model,
        "metrics": model_results[best_model]["metrics"]
    }