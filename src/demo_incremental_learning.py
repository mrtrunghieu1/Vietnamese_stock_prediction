# Library
import os
import pandas as pd
import numpy as np
from configs.il_data_helper import *
from incremental_learning.visualization import ResultVisualization
from river import stream
from output_writer.il_output_writer import ILOutputWriter
from river import stream
from river import metrics
from river import neighbors
from tqdm import tqdm
from utils import get_stock_name
import datetime


# Code
def main():
    for company_name in tqdm(os.listdir(processed_folder_path)):
        output_writer = ILOutputWriter(company_name)
        stock_path = os.path.join(processed_folder_path, company_name)
        print(datetime.datetime.now(), 'Processing the stock of {} company'.format(get_stock_name(company_name)))

        dataset = stream.iter_csv(
            stock_path,
            converters={
                'open': float,
                'close': float,
                'high': float,
                'low': float,
                'low_2': float,
            },
            target='low',
        )

        # Initialize lists to store data
        data = []
        metric_history = []

        # Build Incremental Modeling
        model = neighbors.KNNRegressor()
        metric = metrics.MAE()

        for X, y in dataset:
            date = pd.to_datetime(X['date'])

            # Remove the '' and 'date' files from training data
            X.pop('')
            X.pop('date')

            # Obtain the prior prediction and update the model in one go
            y_pred = model.predict_one(X)
            model.learn_one(X, y)

            # Update the error metric
            metric.update(y, y_pred)
            metric_history.append(metric.get())

            # Store the data
            data.append({
                'date': date,
                'code': get_stock_name(company_name),
                'open': X['open'],
                'close': X['close'],
                'high': X['high'],
                '2nd_min_price': X['low_2'],
                'lowest_price': y,
                'lowest_prediction': y_pred
            })

        # visualization = ResultVisualization(company_name, y_trues, y_preds, metric_history)
        # visualization.plot_results()
        # visualization.plot_metric_history(metric="MAE")
        print(" ", metric)

        # Store the collected data
        df = pd.DataFrame(data)
        output_writer.write_data(il_result_path, df)



if __name__ == "__main__":
    main()
