import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover - optional dependency for the app runtime
    mlflow = None

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
model = None

if mlflow is not None:
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model("models:/MusicPopularityModel@champion")
    except Exception:
        model = None

app = FastAPI(title="Music Popularity Prediction API", version="1.0.0")
REQUESTS = Counter("music_api_requests_total", "Total requests for track popularity prediction")
LATENCY = Histogram("music_api_request_latency_seconds", "Request latency for predictions")


class TrackInput(BaseModel):
    danceability: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    loudness: float
    speechiness: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    instrumentalness: float = Field(ge=0.0, le=1.0)
    liveness: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    tempo: float


@app.get("/")
def home():
    return {"message": "Music Popularity Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(track: TrackInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Ensure MLflow is installed and the model is registered.",
        )

    REQUESTS.inc()
    payload = track.model_dump() if hasattr(track, "model_dump") else track.dict()
    row = pd.DataFrame([payload])
    with LATENCY.time():
        prediction = float(model.predict(row)[0])
    return {"predicted_popularity": prediction}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)