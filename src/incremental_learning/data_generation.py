import os
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm


class DataGenerator:
    def __init__(self, data_path, type_file="fastparquet"):
        self.data_path = data_path
        self.type_file = type_file

    @staticmethod
    def write_data(self, processed_data_path, company_name, df):
        check_folder_exists(processed_data_path)
        df.to_csv(os.path.join(processed_data_path, '{}.csv'.format(company_name)))

    def generate_data(self, processed_data_path):
        intraday_df = pd.read_parquet(self.data_path, engine=self.type_file)
        intraday_df = convert_datetime(intraday_df)
        for company_name in tqdm(np.unique(intraday_df["Code"])):
            df = intraday_df[intraday_df["Code"] == company_name]
            df = split_time_interval(df)
            self.write_data(processed_data_path, company_name, df)
