# intraday data path
intraday_data_path = '../data/df_intraday_raw.parquet'
processed_data_path = '../data/processed'
results_path = '../../results'

# the code stocks of companies
stock_companies = ['CTG', 'MBB', 'SHB', 'TCB', 'TPB']

# change name columns
price_names = {
    'Price_first': 'open',
    'Price_last': 'close',
    'Price_max': 'high',
    'Price_min': 'low',
    'Price_<lambda_0>': 'low_2',
    'date_': 'date'
}

# Index boundary test set
boundary_idx_test = '2023-02-24'

# Train size ratio
train_size_ratio = 0.8

# The parameters of LSTM models
EPOCHS = 10
BATCH_SIZE = 16

# Flags
check_alldata_flag = True
