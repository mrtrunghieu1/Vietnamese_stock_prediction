# Library
import os
import pandas as pd
import numpy as np
from  import processed_folder_path
from river import stream
from src.output_writer.incremental_learning_output import ILOutputWriter
# Code
def main():
    data_generation = ILOutputWriter()
    for company_name in os.listdir(processed_folder_path):
        data_path = os.path.join(processed_folder_path, company_name)
        df = pd.read_csv(data_path, index_col=0)
        df = df.drop('date', axis=1)




if __name__ == "__main__":
    main()
