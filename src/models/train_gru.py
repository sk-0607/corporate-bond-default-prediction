"""
src/models/train_gru.py
------------------------
GRU model builder for the corporate bond default prediction project.

The training loop and callbacks are shared with the LSTM — import
:func:`src.models.train_lstm.train_sequence_model` and
:func:`src.models.train_lstm.get_callbacks` for those.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GRU,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_gru_model(
    input_shape: tuple[int, int],
    gru_units: list[int] | None = None,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.0001,
) -> Sequential:
    """
    Build a stacked GRU model for binary classification.

    Parameters
    ----------
    input_shape   : (timesteps, n_features)
    gru_units     : List of unit sizes, one GRU layer per element.
                    Defaults to ``[32, 16]``.
    dropout_rate  : Dropout probability after each recurrent layer.
    learning_rate : Adam learning rate.

    Returns
    -------
    Compiled ``tf.keras.Sequential`` model.
    """
    if gru_units is None:
        gru_units = [32, 16]

    model = Sequential(name="GRU_Bond_Default")

    for i, units in enumerate(gru_units):
        return_seq = i < len(gru_units) - 1
        kwargs: dict = dict(
            units=units,
            return_sequences=return_seq,
            recurrent_dropout=0.1,
            name=f"gru_{i + 1}",
        )
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(GRU(**kwargs))
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
