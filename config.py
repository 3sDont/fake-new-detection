# === DATA CONFIGURATION ===
DATASET_ID = "basdong/fake-news-dataset"
TRUE_FILENAME = "DataSet_Misinfo_TRUE.csv"
FAKE_FILENAME = "DataSet_Misinfo_FAKE.csv"

# === MODELS CONFIGURATION ===
# Danh sách các mô hình sẽ được huấn luyện và đánh giá
MODELS_TO_RUN = [
    "bert-base-uncased",
    "roberta-base",
    "xlnet-base-cased"
]

# === WANDB CONFIGURATION ===
# Tên project trên Weights & Biases
WANDB_PROJECT_NAME = "Fake-News-Detection-Transformers"
