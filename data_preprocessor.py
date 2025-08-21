import pandas as pd


class DataPreprocessor:

    def __init__(self, samples_file, meth_ref_file):
        self.sample_file = samples_file
        self.meth_ref_file = meth_ref_file
        self.sample_info = pd.read_csv(self.sample_file, sep="\t")
        self.meth_ref = pd.read_csv(self.meth_ref_file, sep="\t")

    def load_data(self):
        pass
