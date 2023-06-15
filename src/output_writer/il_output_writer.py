from output_writer.base import OutputWriter


class ILOutputWriter(OutputWriter):
    """Write Output for the incremental learning"""
    def __int__(self):
        super().__int__("data_generation")

    def generate_data(self, data, path):
        data.to_csv(path, index=False)
