from ml.model_trainer import train_models
from ml.model_selector import select_best_model


def train_models_tool(df,target_column):
    return train_models(df,target_column)


def select_best_model_tool(model_results):
    return select_best_model(model_results)