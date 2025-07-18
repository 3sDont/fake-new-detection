# src/data_loader.py
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

class KaggleDataLoader:
    def __init__(self, dataset_id, true_filename, fake_filename):
        self.dataset_id = dataset_id
        self.true_filename = true_filename
        self.fake_filename = fake_filename

    def load(self):
        df_true = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, self.dataset_id, self.true_filename)
        df_fake = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, self.dataset_id, self.fake_filename)

        df_true['label'] = 1
        df_fake['label'] = 0
        return pd.concat([df_true, df_fake], ignore_index=True)
