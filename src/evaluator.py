import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, trainer, dataset):
        self.trainer = trainer
        self.dataset = dataset

    def evaluate(self):
        print("--- Evaluating model on the test set ---")
        predictions = self.trainer.predict(self.dataset['test'])
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=["FAKE", "REAL"]))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        # Lưu ma trận nhầm lẫn thay vì hiển thị
        plt.savefig(f"confusion_matrix_{self.trainer.args.output_dir.split('/')[-1]}.png")
        print(f"Confusion matrix saved to confusion_matrix_{self.trainer.args.output_dir.split('/')[-1]}.png")
        plt.close()
