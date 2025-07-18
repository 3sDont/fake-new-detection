# src/preprocess.py

import re
import string
import nltk
from nltk.corpus import stopwords

# Chỉ cần chạy lần đầu trong notebook Colab
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()                        # Viết thường
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Bỏ URL
    text = re.sub(r'<.*?>', '', text)               # Bỏ HTML
    text = text.translate(str.maketrans('', '', string.punctuation))  # Bỏ dấu câu
    text = re.sub(r'\d+', '', text)                 # Bỏ số
    text = re.sub(r'\s+', ' ', text).strip()        # Chuẩn hóa khoảng trắng
    return text

def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

def preprocess_dataframe(df):
    df = df.copy()
    df['clean_text'] = df['text'].apply(preprocess_text)
    return df
