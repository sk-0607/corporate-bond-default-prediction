"""
src/models/train_lstm.py
-------------------------
LSTM model builder, Keras callbacks, and shared sequence-model training
function used by both the LSTM and GRU notebooks.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units: list[int] | None = None,
    dropout_rate: float = 0.25,
    learning_rate: float = 0.0001,
) -> Sequential:
    """
    Build a stacked LSTM model for binary classification.

    Parameters
    ----------
    input_shape   : (timesteps, n_features)
    lstm_units    : List of unit sizes, one LSTM layer per element.
                    Defaults to ``[32, 16]``.
    dropout_rate  : Dropout probability after each recurrent layer.
    learning_rate : Adam learning rate.

    Returns
    -------
    Compiled ``tf.keras.Sequential`` model.
    """
    if lstm_units is None:
        lstm_units = [32, 16]

    model = Sequential(name="LSTM_Bond_Default")

    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        kwargs: dict = dict(
            units=units,
            return_sequences=return_seq,
            recurrent_dropout=0.1,
            name=f"lstm_{i + 1}",
        )
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(LSTM(**kwargs))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu", name="dense_2"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["AUC", "Precision", "Recall"],
    )
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """
    Build the standard Keras callback list used by both LSTM and GRU training.

    Parameters
    ----------
    checkpoint_path : str
        File path where the best model weights will be saved
        (e.g. ``'../models/lstm_best.keras'``).

    Returns
    -------
    list of Keras callbacks: [EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        mode="min",
        min_lr=1e-6,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, reduce_lr, checkpoint]


def train_sequence_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict,
    checkpoint_path: str,
    batch_size: int = 32,
    epochs: int = 150,
):
    """
    Train an LSTM or GRU model with the standard callback set.

    Parameters
    ----------
    model           : Compiled Keras model.
    X_train, y_train: Training sequences and labels.
    X_val, y_val    : Validation sequences and labels.
    class_weights   : Dict ``{0: w0, 1: w1}`` from
                      :func:`src.data.preprocessing.compute_class_weights`.
    checkpoint_path : Path for ``ModelCheckpoint``.
    batch_size      : Mini-batch size.
    epochs          : Maximum number of training epochs.

    Returns
    -------
    keras.callbacks.History
    """
    callbacks = get_callbacks(checkpoint_path)

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )
    return history
