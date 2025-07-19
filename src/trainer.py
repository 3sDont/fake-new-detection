import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

class ModelTrainer:
    def __init__(self, model, dataset, model_name, data_collator=None):
        self.model = model
        self.dataset = dataset
        self.data_collator = data_collator
        self.model_name = model_name

    def train(self):
        training_args = TrainingArguments(
            output_dir=f"./results/{self.model_name}",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir="./logs",
            fp16=True,
            report_to="wandb",
            save_total_limit=1
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        print(f"--- Starting training for {self.model_name} ---")
        trainer.train()
        print("--- Training complete ---")
        return trainer
