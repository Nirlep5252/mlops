import os

import requests as rs
import streamlit as st
from requests.exceptions import RequestException

# Allow overriding the API location so the app works both inside and outside docker-compose.
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8086")

st.title("MLOPs Assignment")

pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=0)
sex = st.selectbox("Sex", options=["male", "female"], index=0)
age = st.number_input("Age", value=30.0, min_value=0.0, step=1.0)
sibsp = st.number_input("Siblings/Spouses Aboard", value=0, min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard", value=0, min_value=0, step=1)
fare = st.number_input("Fare", value=32.0, min_value=0.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], index=0)


def get_api(params):
    targets = [API_BASE_URL.rstrip("/")]

    if API_BASE_URL == "http://api:8086":
        targets.append("http://localhost:8086")

    last_error = None
    for base_url in targets:
        try:
            response = rs.get(f"{base_url}/predict/", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            last_error = exc

    raise last_error if last_error else RuntimeError("API request failed")


if st.button("Get prediction"):
    params = {
        "pclass": int(pclass),
        "sex": sex,
        "age": float(age),
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": float(fare),
        "embarked": embarked,
    }

    try:
        data = get_api(params)
        if not isinstance(data, dict):
            st.success(f"Prediction: {data}")
            st.stop()

        survived = data.get("survived")
        confidence = data.get("confidence")
        status = "Survived" if survived == 1 else "Did not survive"

        if confidence is not None:
            st.success(f"{status} (confidence {confidence:.2%})")
        else:
            st.success(status)
    except Exception as exc:
        st.error(f"Failed to reach prediction API: {exc}")

