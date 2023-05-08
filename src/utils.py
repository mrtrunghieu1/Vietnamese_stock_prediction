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
    # df_copy = df.copy()
    # df_copy.set_index('intra_day', inplace=True)
    # df_copy = df_copy.groupby([pd.Grouper(freq='D'), pd.Grouper(freq='60T')]).agg(
    #     {'Price': ['first', 'last', 'max', 'min', lambda x: tuple(x.nsmallest(2))]})
    # df_copy['date'] = df_copy.index.get_level_values(-1)
    # # df_copy = df_copy.reset_index(drop=True)
    # # df_copy.columns = df_copy.columns.map('_'.join)
    # # df_copy.rename(columns=price_names, inplace=True)
    # # df_copy = df_copy.reindex(columns=['date', 'open', 'close', 'high', 'low'])

    df_copy = df.copy()
    df_copy.set_index('intra_day', inplace=True)
    df_copy = df_copy.groupby([pd.Grouper(freq='D'), pd.Grouper(freq='60T')]).agg(
        {'Price': ['first', 'last', 'max', 'min', lambda x: sorted(set(x))[1:2]]})
    df_copy['date'] = df_copy.index.get_level_values(-1)
    df_copy = df_copy.reset_index(drop=True)
    df_copy.columns = df_copy.columns.map('_'.join)
    df_copy.rename(columns=price_names, inplace=True)
    df_copy['low_2'] = [x[0] if len(x) > 0 else df_copy['low'][i] for i, x in enumerate(df_copy['low_2'])]
    df_copy = df_copy.reindex(columns=['date', 'open', 'close', 'high', 'low', 'low_2'])
    return df_copy


def check_stock_existence(company_symbol, intraday_companies):
    """
     Takes in a company symbol and a list of intraday companies and checks whether the given company symbol exists
     in the list

    Parameters:
        company_symbol: A string that represents the company symbol
        intraday_companies: A list of strings representing the company symbols

    Returns:
        Raises a ValueError exception with an appropriate error message if the company symbol does not exist in the list
        of intraday companies. If the company symbol exists in the list, the function does not return anything.
    """
    split_name_companies = [name.split('.')[0] for name in intraday_companies]
    if company_symbol not in split_name_companies:
        raise ValueError(f"The stock of company {company_symbol} doesn't exist. Please run again "
                         f"df_intraday_raw.parquet to find the stock.")


