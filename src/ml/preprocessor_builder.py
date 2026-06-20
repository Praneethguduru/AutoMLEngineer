from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessor(numerical_columns, categorical_columns):

    transformers = []

    if len(numerical_columns) > 0:
        numerical_pipeline = Pipeline([("scaler", StandardScaler())])
        transformers.append(("num", numerical_pipeline, numerical_columns))

    if len(categorical_columns) > 0:
        categorical_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        # Pass-through if no numerical or categorical columns are defined
        return ColumnTransformer([("passthrough", "passthrough", [])])

    preprocessor = ColumnTransformer(transformers)

    return preprocessor