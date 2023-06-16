import os
from abc import abstractmethod
import abc


class OutputWriter:
    def __int__(self, company_name):
        self.company_name = company_name
        # self.output_path = output_path

        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

    @abstractmethod
    def write_data(self, data, data_path):
        pass
