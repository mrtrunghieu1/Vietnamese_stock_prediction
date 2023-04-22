# Standard lib
import os

# Third party
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Local
from ..data_helper import EPOCHS, BATCH_SIZE
# Code
def build_model(X_train, y_train, X_val, y_val):
    """
    The build_model function builds and trains an LSTM model using the training and validation data.

    Parameters:
        X_train: Training data features
        y_train: Training data labels
        X_val: Validation data features
        y_val: Validation data labels

    Returns:
        model: Trained LSTM model
    """
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_absolute_error'])

    # Next, let's train the model:
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    return model
