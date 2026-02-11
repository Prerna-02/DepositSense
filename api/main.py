"""
FastAPI application for Bank Marketing term-deposit predictions.
Endpoints: /health, /predict, /batch_predict
Includes SQLite logging of every prediction.
"""

import io
import csv
import sqlite3
import datetime
import json
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

from src.config import (
    MODEL_PATH,
    PIPELINE_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
)
from api.schemas import CustomerInput, PredictionResponse, HealthResponse


# ── Globals ──────────────────────────────────────────────────────────────
model = None
pipeline = None
DB_PATH = MODEL_DIR / "predictions.db"
MODEL_VERSION = "v1.0"


def _init_db():
    """Create predictions table if it doesn't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_json TEXT NOT NULL,
            probability REAL NOT NULL,
            prediction TEXT NOT NULL,
            model_version TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _log_prediction(input_data: dict, probability: float, prediction: str):
    """Log a prediction to SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO prediction_log (timestamp, input_json, probability, prediction, model_version) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            datetime.datetime.utcnow().isoformat(),
            json.dumps(input_data),
            probability,
            prediction,
            MODEL_VERSION,
        ),
    )
    conn.commit()
    conn.close()


# ── Lifespan (load model once) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, pipeline
    pipeline = joblib.load(PIPELINE_PATH)
    model = keras.models.load_model(MODEL_PATH)
    _init_db()
    print("✔ Model & pipeline loaded, DB ready")
    yield


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a customer will subscribe to a term deposit.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _customer_to_df(customer: CustomerInput) -> pd.DataFrame:
    """Convert a single CustomerInput to a 1-row DataFrame."""
    return pd.DataFrame([customer.model_dump()])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=(model is not None))


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput):
    """Predict for a single customer."""
    df = _customer_to_df(customer)
    X = pipeline.transform(df)
    prob = float(model.predict(X, verbose=0).ravel()[0])
    label = "yes" if prob >= 0.5 else "no"

    _log_prediction(customer.model_dump(), prob, label)

    return PredictionResponse(
        probability=round(prob, 4),
        prediction=label,
        model_version=MODEL_VERSION,
    )


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """Upload a CSV and get predictions for every row."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    X = pipeline.transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    probs = model.predict(X, verbose=0).ravel()
    labels = ["yes" if p >= 0.5 else "no" for p in probs]

    results = []
    for i, (prob, label) in enumerate(zip(probs, labels)):
        row_data = df.iloc[i].to_dict()
        _log_prediction(row_data, float(prob), label)
        results.append({
            "row": i,
            "probability": round(float(prob), 4),
            "prediction": label,
        })

    return {"predictions": results, "total": len(results)}
