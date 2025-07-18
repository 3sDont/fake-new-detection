# model.py

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from datasets import Dataset, DatasetDict


class TextDatasetBuilder:
    def __init__(self, df, model_name):
        self.df = df
        self.model_name = model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(self, examples):
        padding = 'max_length' if "xlnet" in self.model_name else False
        return self.tokenizer(
            examples['text'], padding=padding, truncation=True, max_length=512
        )

    def build(self):
        self.df['text'] = self.df['text'].astype(str)
        texts = self.df['text'].values
        labels = self.df['label'].astype(np.int32).values

        # Step 1: Train (80%) vs Temp (20%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Step 2: Validation (10%) vs Test (10%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"Số lượng mẫu:")
        print(f"  Train      : {len(X_train)}")
        print(f"  Validation : {len(X_val)}")
        print(f"  Test       : {len(X_test)}")

        dataset = DatasetDict({
            'train': Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()}),
            'validation': Dataset.from_dict({'text': X_val.tolist(), 'label': y_val.tolist()}),
            'test': Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()}),
        })

        dataset = dataset.map(self.tokenize_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        data_collator = None
        if "xlnet" not in self.model_name:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

        return dataset, self.tokenizer, data_collator


class ModelBuilder:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        if torch.cuda.is_available():
            self.model.to('cuda')

    def get_model(self):
        return self.model


class ModelTrainer:
    def __init__(self, model, dataset, model_name, data_collator=None):
        self.model = model
        self.dataset = dataset
        self.data_collator = data_collator
        self.model_name = model_name

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds),
            'f1': f1_score(labels, preds)
        }

    def train(self):
        training_args = TrainingArguments(
            output_dir=f"./results/{self.model_name}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_steps=100,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir="./logs",
            fp16=torch.cuda.is_available(),
            report_to="wandb",
            save_total_limit=1
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=None,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        return trainer
