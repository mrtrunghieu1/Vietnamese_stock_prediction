import os.path

# from output_writer.base import OutputWriter
from utils import check_folder_exists


# Code
class MLOutputWriter:
    """Write processed data for machine learning"""

    def __init__(self, name_stock):
        # self.output_path = output_path
        self.name_stock = name_stock

    def write_data(self, processed_data_path, df):
        check_folder_exists(processed_data_path)
        df.to_csv(os.path.join(processed_data_path, '{}.csv'.format(self.name_stock)))

