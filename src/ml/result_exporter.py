import json
import os
import numpy as np


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def export_results(
    target_column,
    problem_type,
    best_model,
    all_models,
    cv_results,
    consultant_report,
    output_file="outputs/results.json"
):

    # Extract metrics for all models to avoid serializing arrays or pipelines
    model_metrics = {}
    for model_name, res in all_models.items():
        model_metrics[model_name] = res["metrics"]

    results = {
        "target": target_column,
        "problem_type": problem_type,
        "best_model": best_model,
        "all_models": model_metrics,
        "cross_validation": cv_results,
        "consultant_report": consultant_report
    }

    # Ensure output directory exists
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            cls=NumpyEncoder,
            indent=4
        )

    return output_file