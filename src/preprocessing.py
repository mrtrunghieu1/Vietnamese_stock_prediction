# Standard lib
import os

import numpy as np
# Third party
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Local
from data_helper import boundary_idx_test, train_size_ratio


# Code

def data_preprocessing(df):
    """
    Preprocesses a given DataFrame for use in an LSTM models. The function performs the following operations:

    1. Makes a copy of the input DataFrame and reorders the columns to match the expected order.
    2. Sets the date column as the index and converts all columns to float data type.
    3. Scales the data using MinMaxScaler to transform values to the range [0, 1].

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be preprocessed. The DataFrame should contain at least the columns
        "date", "open", "high", "low", and "close".

    Returns:
        scaler (sklearn.preprocessing.MinMaxScaler): The scaler object that was used to scale the data.
        scaled_data (numpy.ndarray): A 2D array containing the scaled data.
        df_copy (pandas.DataFrame): A copy of the input DataFrame with reordered columns, indexed by the "date" column,
        and with all columns converted to float data type.

    """
    df_copy = df.copy()
    df_copy = df_copy.reindex(columns=['date', 'open', 'high', 'low', 'close'])

    df_copy = df_copy.set_index("date")
    df_copy = df_copy.astype(float)
    # scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_X = scaler.fit_transform(df_copy[df_copy.columns[:-1]])
    scaler_y = scaler.fit_transform(np.asarray(df_copy[df_copy.columns[-1]]).reshape(-1, 1))

    return scaler, scaler_X, scaler_y, df_copy


def split_data(dataframe, scaler_X, scaler_y, df_low, df_low2):
    """
    This function splits the input dataset into training, validation, and test sets based on the specific day.
    and returns the corresponding data arrays.

    Parameters:
        scaler_y:
        scaler_X:
        dataframe: A pandas DataFrame containing the original dataset, with a "date" column and columns for "open",
                    "high", "low", and "close" prices.

    Returns:
    """
    # Find the index of the first timestamp that is greater than or equal to boundary_idx_test("24-02-2023")
    dates = dataframe.index
    boundary_idx = dataframe.index.searchsorted(pd.Timestamp(boundary_idx_test))

    # Split the data into training/validation and test sets
    train_val_dates, train_val_X, train_val_y = dates[:boundary_idx], scaler_X[:boundary_idx], scaler_y[:boundary_idx]
    test_dates, test_X, test_y = dates[boundary_idx:], scaler_X[boundary_idx:], scaler_y[boundary_idx:]
    df_min1, df_min2 = df_low[boundary_idx:], df_low2[boundary_idx:]

    # Further split the training/validation set into the training and validation sets
    train_size = int(len(train_val_X) * train_size_ratio)

    train_dates, X_train, y_train = train_val_dates[:train_size], train_val_X[:train_size, :], train_val_y[
                                                                                               :train_size, :]
    val_dates, X_val, y_val = train_val_dates[train_size:], train_val_X[train_size:], train_val_y[
                                                                                      train_size:, :]
    return train_dates, X_train, y_train, val_dates, X_val, y_val, test_dates, test_X, test_y, df_min1, df_min2
