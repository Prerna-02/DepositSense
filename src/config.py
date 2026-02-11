"""
Central configuration for the Bank Marketing ANN project.
All paths, feature definitions, and hyper-parameters live here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "bank.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

# ── Feature definitions ─────────────────────────────────────────────────
TARGET = "deposit"

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# ── Train / Val / Test split ────────────────────────────────────────────
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # fraction of the *remaining* data after test split
RANDOM_STATE = 42

# ── Model hyper-parameters ──────────────────────────────────────────────
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATES = [0.3, 0.2, 0.0]   # one per hidden layer (0 = no dropout)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 150
EARLY_STOP_PATIENCE = 10

# ── Model artifacts ─────────────────────────────────────────────────────
MODEL_PATH = MODEL_DIR / "ann_model.keras"
PIPELINE_PATH = MODEL_DIR / "preprocess_pipeline.pkl"
HISTORY_PATH = MODEL_DIR / "history.json"
