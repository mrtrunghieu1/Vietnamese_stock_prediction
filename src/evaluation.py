# Standard lib
import json
import os
# Third party
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Local
from data_helper import results_path


# Code
# !Scripts to evaluate the models and save the results
class OutputWriter:
    def __init__(self, name_stock, output_path):
        self.output_path = output_path
        self.name_stock = name_stock
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def write_data(self, processed_data_path, df):
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
        df.to_csv(os.path.join(processed_data_path, '{}.csv'.format(self.name_stock)))

    def write_output_csv(self, test_dates, ground_truth, predictions):
        df = pd.DataFrame({
            'date': test_dates,
            'code': ['{}'.format(self.name_stock)] * len(ground_truth),
            'close_ground_truth': np.squeeze(ground_truth),
            'close_predictions': np.squeeze(predictions)
        })
        df.to_csv(os.path.join(self.output_path, '{}.csv'.format(self.name_stock)))
        print("Save the {} stock successfully!".format(self.name_stock))

    def write_output_info(self, ground_truth, predictions, indent=None):
        data = {
            "code": self.name_stock,
            "mean_ground_truth": str(ground_truth.mean()),
            "mean_predictions": str(predictions.mean()),
            "MAE": str(mean_absolute_error(ground_truth, predictions)),
            "RMSE": str(mean_squared_error(ground_truth, predictions)),
            "R2": str(r2_score(ground_truth, predictions))
        }
        with open('{}/{}.json'.format(self.output_path, self.name_stock), 'w') as output_file:
            json.dump(data, output_file, indent=indent)
