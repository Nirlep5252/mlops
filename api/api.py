import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow.pyfunc

app = FastAPI()

# Instrument once at import time so middleware registration happens before startup.
instrumentator = Instrumentator().instrument(app)

TARGET_MODEL_NAME = os.getenv("MODEL_NAME", "titanic-classifier")

TITANIC_FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

EMBARKED_DEFAULT = "S"


def fetch_latest_model():
    client = MlflowClient()
    try:
        model = client.get_registered_model(TARGET_MODEL_NAME)
    except MlflowException as exc:
        raise RuntimeError(f"Registered MLflow model '{TARGET_MODEL_NAME}' not found") from exc

    if not model:
        raise RuntimeError(f"Registered MLflow model '{TARGET_MODEL_NAME}' not found")

    return model.name


def fetch_latest_version(model_name):
    try:
        return mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}' from Production stage"
        ) from exc


@app.on_event("startup")
async def startup():
    instrumentator.expose(app)


@app.get("/predict/")
def model_output(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str | None = None,
):
    if embarked is None or not embarked.strip():
        embarked_value = EMBARKED_DEFAULT
    else:
        embarked_value = embarked.strip().upper()

    sex_value = sex.strip().lower()
    if sex_value not in ("male", "female"):
        raise HTTPException(status_code=400, detail="sex must be 'male' or 'female'")

    feature_values = {
        "pclass": pclass,
        "sex": sex_value,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked_value,
    }

    model_name = fetch_latest_model()
    model = fetch_latest_version(model_name)

    input_df = pd.DataFrame({key: [feature_values[key]] for key in TITANIC_FEATURES})
    prediction = model.predict(input_df)
    prediction_value = int(prediction[0])

    return {"survived": prediction_value}
