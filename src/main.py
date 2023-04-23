# Standard lib
import os
import sys
import datetime
import time

# Third party
import pandas as pd

# Local
from preprocessing import data_preprocessing, split_data
from data_helper import intraday_data_path, processed_data_path
from data_helper import stock_companies, check_alldata_flag
from model.train_model import build_model
from utils import *

# Code
for stock_company in stock_companies:
    if check_alldata_flag:
        # Read intraday data file
        intraday_df = pd.read_parquet(intraday_data_path, engine='fastparquet')
        intraday_df = convert_datetime(intraday_df)
        print(datetime.datetime.now(), 'Processing the stock of {} company'.format(stock_company))
        df = intraday_df[intraday_df['Code'] == stock_company]
        df = split_time_interval(df)
        print(df.head())
        df.to_csv(os.path.join(processed_data_path, '{}.csv'.format(stock_company)))
    # Data Processing the stock market
    processed_companies = os.listdir(processed_data_path)
    try:
        check_stock_existence(stock_company, processed_companies)
        print(datetime.datetime.now(), 'Processing the stock of {} company'.format(stock_company))
        file_csv = os.path.join(processed_data_path, '{}.csv'.format(stock_company))
        df = pd.read_csv(file_csv)
        df['date'] = df['date'].astype("datetime64[ns]")
        scaler, scaled_data, dataframe = data_preprocessing(df)
        train_dates, X_train, y_train, val_dates, X_val, y_val, test_dates, X_test, y_test = split_data(dataframe,
                                                                                                        scaled_data)
        # Train LSTM model
        model = build_model(X_train, y_train, X_val, y_val)
        test_predictions = model.predict(X_test)
        print(test_predictions.shape)
    except ValueError as err:
        print(f"Error: {err}")
