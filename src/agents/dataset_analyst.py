import numpy as np
import pandas as pd 
from workflows.state import AutoMLState
from tools.dataset_tools import analyze_dataset_tool, detect_target_tool, detect_problem_type_tool
from ml.data_loader import load_data


def make_json_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return {k: make_json_serializable(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, pd.Series):
        return {k: make_json_serializable(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(v) for v in obj.tolist()]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                          np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)



def dataset_analyst_node(state: AutoMLState):

    df = load_data(state["dataset_path"])

    target = detect_target_tool(df)

    problem_type = detect_problem_type_tool(df,target)

    report = analyze_dataset_tool(df)

    report["missing values"] = report["missing values"].to_dict()

    report["statistical summary"] = (
        report["statistical summary"]
        .to_dict()
    )

    return make_json_serializable({
        **state,
        "df": df,
        "target_column": target,
        "problem_type": problem_type,
        "dataset_report": report
    })