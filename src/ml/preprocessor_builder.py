from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessor(numerical_columns, categorical_columns):

    numerical_pipeline = Pipeline([("scaler",StandardScaler())])

    categorical_pipeline = Pipeline([("encoder",OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([("num", numerical_pipeline, numerical_columns),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor