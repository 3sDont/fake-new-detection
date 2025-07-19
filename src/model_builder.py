import torch
from transformers import AutoModelForSequenceClassification

class ModelBuilder:
    def __init__(self, model_name):
        print(f"--- Loading pre-trained model: {model_name} ---")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        if torch.cuda.is_available():
            self.model.to('cuda')

    def get_model(self):
        return self.model
