import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Titanic-Survival")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "titanic-classifier")

TITANIC_DATA_URL = os.getenv(
    "TITANIC_DATA_URL",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
)

TITANIC_FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

NUMERIC_FEATURES = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]


def train():
    mlflow.sklearn.autolog()

    data = pd.read_csv(TITANIC_DATA_URL)
    data.columns = [column.lower() for column in data.columns]

    X = data[TITANIC_FEATURES].copy()
    y = data["survived"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train.iloc[:1],
        )

        client = MlflowClient()
        try:
            client.get_registered_model(REGISTERED_MODEL_NAME)
        except RestException:
            client.create_registered_model(REGISTERED_MODEL_NAME)

        model_version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model_info.model_uri,
            run_id=run.info.run_id,
        )

        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )


if __name__ == "__main__":
    train()