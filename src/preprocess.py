"""
Preprocessing pipeline for the Bank Marketing dataset.
- Loads raw CSV
- Encodes target (yes/no → 1/0)
- Applies ColumnTransformer (OneHotEncoder + StandardScaler)
- Splits into train / val / test
- Saves processed arrays and the fitted pipeline
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW,
    DATA_PROCESSED,
    MODEL_DIR,
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    PIPELINE_PATH,
)


def load_data() -> pd.DataFrame:
    """Load the raw bank-marketing CSV."""
    df = pd.read_csv(DATA_RAW)
    print(f"✔ Loaded {DATA_RAW.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map target column yes/no → 1/0."""
    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})
    counts = df[TARGET].value_counts()
    print(f"✔ Target encoded — class distribution:\n{counts.to_string()}")
    return df


def build_pipeline() -> ColumnTransformer:
    """Build a ColumnTransformer for numeric scaling + categorical OHE."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def preprocess_and_save():
    """End-to-end: load → encode → transform → split → save."""
    # 1. Load & encode
    df = load_data()
    df = encode_target(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    # 2. Train / temp split  →  temp / val split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"✔ Split sizes — Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 3. Fit pipeline on train, transform all splits
    pipeline = build_pipeline()
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)
    X_test = pipeline.transform(X_test)

    print(f"✔ Transformed feature dim: {X_train.shape[1]}")

    # 4. Save processed arrays
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val", X_val), ("y_val", y_val),
        ("X_test", X_test), ("y_test", y_test),
    ]:
        np.save(DATA_PROCESSED / f"{name}.npy", arr)

    # 5. Save feature names (useful for API & explainability)
    feature_names = (
        NUMERIC_FEATURES
        + list(pipeline.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    with open(DATA_PROCESSED / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 6. Save fitted pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"✔ Pipeline saved → {PIPELINE_PATH}")

    print("\n🎉 Preprocessing complete!")


if __name__ == "__main__":
    preprocess_and_save()
