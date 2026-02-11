"""
Train the MLP (ANN) model for bank marketing term-deposit prediction.
- Loads preprocessed arrays
- Builds a sequential Dense network
- Trains with early stopping and class weighting
- Saves model + training history
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    DATA_PROCESSED,
    HIDDEN_LAYERS,
    DROPOUT_RATES,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    MODEL_PATH,
    HISTORY_PATH,
    MODEL_DIR,
)


def load_processed_data():
    """Load the train/val numpy arrays."""
    X_train = np.load(DATA_PROCESSED / "X_train.npy")
    y_train = np.load(DATA_PROCESSED / "y_train.npy")
    X_val = np.load(DATA_PROCESSED / "X_val.npy")
    y_val = np.load(DATA_PROCESSED / "y_val.npy")
    print(f"✔ Loaded data — Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Compute balanced class weights to handle any imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes.astype(int), weights))
    print(f"✔ Class weights: {cw}")
    return cw


def build_model(input_dim: int) -> keras.Model:
    """Build the MLP model from config."""
    model = keras.Sequential(name="BankMarketing_MLP")
    model.add(layers.Input(shape=(input_dim,)))

    for units, drop in zip(HIDDEN_LAYERS, DROPOUT_RATES):
        model.add(layers.Dense(units, activation="relu"))
        if drop > 0:
            model.add(layers.Dropout(drop))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()
    return model


def train():
    """Full training routine."""
    X_train, y_train, X_val, y_val = load_processed_data()
    class_weights = compute_class_weights(y_train)
    model = build_model(X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✔ Model saved → {MODEL_PATH}")

    # Save history (convert numpy values to plain floats for JSON)
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"✔ History saved → {HISTORY_PATH}")

    # Quick summary
    val_metrics = {k: vals[-1] for k, vals in history.history.items() if k.startswith("val_")}
    print(f"\n📊 Best validation metrics: {val_metrics}")
    print("🎉 Training complete!")


if __name__ == "__main__":
    train()
