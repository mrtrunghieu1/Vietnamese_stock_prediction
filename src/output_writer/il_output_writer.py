# from output_writer.base import OutputWriter
from utils import check_folder_exists, get_stock_name
import os


class ILOutputWriter:
    """Write Output for the incremental learning"""

    def __init__(self, name_stock):
        self.name_stock = name_stock

    def write_data(self, result_path, df):
        check_folder_exists(result_path)
        df.to_csv(os.path.join(result_path, '{}.csv'.format(get_stock_name(self.name_stock))), index=False)
