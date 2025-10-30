FROM python:3.11-slim

WORKDIR /training

# Upgrade pip first to ensure we get the latest manylinux wheels.
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["python", "model_training.py"]