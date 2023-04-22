# Standard lib
import os
# Third party
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Local
from data_helper import boundary_idx_test, train_size_ratio


# Code

def data_preprocessing(df):
    """
    Preprocesses a given DataFrame for use in an LSTM model. The function performs the following operations:

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
    scaled_data = scaler.fit_transform(df_copy)
    return scaler, scaled_data, df_copy


def split_data(dataframe, scaled_data):
    """
    This function splits the input dataset into training, validation, and test sets based on the specific day.
    and returns the corresponding data arrays.

    Parameters:
        dataframe: A pandas DataFrame containing the original dataset, with a "date" column and columns for "open",
                    "high", "low", and "close" prices.
        scaled_data: A NumPy array of the scaled data obtained from data_preprocessing() function.

    Returns:
        train_dates: A NumPy array containing the dates for the training set.
        X_train: A NumPy array containing the feature data for the training set.
        y_train: A NumPy array containing the target data for the training set.
        val_dates: A NumPy array containing the dates for the validation set.
        X_val: A NumPy array containing the feature data for the validation set.
        y_val: A NumPy array containing the target data for the validation set.
        test_dates: A NumPy array containing the dates for the test set.
        X_test: A NumPy array containing the feature data for the test set.
        y_test: A NumPy array containing the target data for the test set.
    """
    # Find the index of the first timestamp that is greater than or equal to boundary_idx_test("24-02-2023")
    dates = dataframe.index
    boundary_idx = dataframe.index.searchsorted(pd.Timestamp(boundary_idx_test))

    # Split the data into training/validation and test sets
    train_val_dates, train_val_data = dates[:boundary_idx], scaled_data[:boundary_idx]
    test_dates, test_data = dates[boundary_idx:], scaled_data[boundary_idx:]

    # Further split the training/validation set into the training and validation sets
    train_size = int(len(train_val_data) * train_size_ratio)

    train_dates, X_train, y_train = train_val_dates[:train_size], train_val_data[:train_size, :-1], train_val_data[
                                                                                                    :train_size, -1]
    val_dates, X_val, y_val = train_val_dates[train_size:], train_val_data[train_size:, :-1], train_val_data[
                                                                                              train_size:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    return train_dates, X_train, y_train, val_dates, X_val, y_val, test_dates, X_test, y_test
