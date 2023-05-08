# Standard lib

# Third party

# Local
from preprocessing import data_preprocessing, split_data
from data_helper import intraday_data_path, processed_data_path, results_path
from data_helper import stock_companies, check_alldata_flag
from model import build_model
from evaluation import OutputWriter
from utils import *

# Code
for stock_company in stock_companies:
    output_writer = OutputWriter(stock_company, os.path.join(results_path, stock_company))
    if check_alldata_flag:
        # Read intraday data file
        intraday_df = pd.read_parquet(intraday_data_path, engine='fastparquet')
        intraday_df = convert_datetime(intraday_df)
        print(datetime.datetime.now(), 'Processing the stock of {} company'.format(stock_company))
        df = intraday_df[intraday_df['Code'] == stock_company]
        df = split_time_interval(df)
        output_writer.write_data(processed_data_path, df)

    # Data Processing the stock market
    processed_companies = os.listdir(processed_data_path)
    check_stock_existence(stock_company, processed_companies)
    print(datetime.datetime.now(), 'Processing the stock of {} company'.format(stock_company))
    file_csv = os.path.join(processed_data_path, '{}.csv'.format(stock_company))
    df = pd.read_csv(file_csv)
    df['date'] = df['date'].astype("datetime64[ns]")
    scaler, scaler_X, scaler_y, dataframe = data_preprocessing(df)
    train_dates, X_train, y_train, val_dates, X_val, y_val, test_dates, \
        X_test, y_test, df_min1, df_min2 = split_data(dataframe, scaler_X, scaler_y, df['low'], df['low_2'])
    # Train LSTM models
    model = build_model(X_train, y_train, X_val, y_val)
    test_predictions = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predictions)
    test_ground_truth = scaler.inverse_transform(y_test)
    # Write the results
    output_writer.write_output_csv(test_dates, test_ground_truth, test_predict, df_min1, df_min2)
    output_writer.write_output_info(test_ground_truth, test_predict)
    # Visualization the results