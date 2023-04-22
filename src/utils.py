# Standard lib
import os
import sys
import datetime
import time

# Third party
import pandas as pd

# Local
from data_helper import price_names


# Code

def convert_datetime(df):
    """
    Convert a single 'intra_day' column into 'Date' and 'Time' columns

    Parameters:
        df: dataframe

    Returns:
        df_copy: new dataframe
    """
    df_copy = df.copy()
    df_copy['intra_day'] = pd.to_datetime(df_copy['Date'] + ' ' + df_copy['Time'])
    # Remove two columns name is 'Date' and 'Time'
    df_copy = df_copy.drop(['Date', 'Time'], axis=1)
    return df_copy


def split_time_interval(df):
    """
    Splits a DataFrame of intraday stock prices into hourly intervals and aggregates the prices within each interval
    to compute the open, close, high, and low prices for each interval.

    Parameters:
        df (pandas DataFrame): A DataFrame of intraday stock prices with columns 'Price' and 'intra_day'.

    Returns:
        df_copy: A pandas DataFrame containing the open, close, high, low, and date values for each hourly interval.
    """
    df_copy = df.copy()
    df_copy.set_index('intra_day', inplace=True)
    df_copy = df_copy.groupby([pd.Grouper(freq='D'), pd.Grouper(freq='60T')]).agg(
        {'Price': ['first', 'last', 'max', 'min']})
    df_copy['date'] = df_copy.index.get_level_values(-1)
    df_copy = df_copy.reset_index(drop=True)
    df_copy.columns = df_copy.columns.map('_'.join)
    df_copy.rename(columns=price_names, inplace=True)
    df_copy = df_copy.reindex(columns=['date', 'open', 'close', 'high', 'low'])
    return df_copy
