class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        print("--- Preprocessing data: Dropping duplicates and nulls ---")
        df = self.df.drop_duplicates()
        df = df.dropna(subset=['text'])
        return df[['text', 'label']].reset_index(drop=True)
