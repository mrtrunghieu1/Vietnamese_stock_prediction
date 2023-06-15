# Library
import os.path


class OutputWriter:
    def __int__(self, company_name, output_path):
        self.company_name = company_name
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

