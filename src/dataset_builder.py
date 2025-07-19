import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

class TextDatasetBuilder:
    def __init__(self, df, model_name):
        self.df = df
        self.model_name = model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize_function(self, examples):
        padding = 'max_length' if "xlnet" in self.model_name else False
        return self.tokenizer(examples['text'], padding=padding, truncation=True, max_length=512)

    def build(self):
        print(f"--- Building dataset for model: {self.model_name} ---")
        self.df['text'] = self.df['text'].astype(str)
        texts, labels = self.df['text'].values, self.df['label'].astype(np.int32).values

        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"Dataset split sizes:\n  Train: {len(X_train)}\n  Validation: {len(X_val)}\n  Test: {len(X_test)}")

        dataset = DatasetDict({
            'train': Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()}),
            'validation': Dataset.from_dict({'text': X_val.tolist(), 'label': y_val.tolist()}),
            'test': Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})
        })

        print("Tokenizing datasets...")
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        data_collator = None
        if "xlnet" not in self.model_name:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

        return tokenized_dataset, self.tokenizer, data_collator
