def detect_problem_type(df,target_column):

    target = df[target_column]

    unique_values = target.nunique()

    if target.dtype == "object":
        return "classification"

    if unique_values <= 20:
        return "classification"

    return "regression"