# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_class_distribution(df):
    print("Class distribution:\n", df['label'].value_counts())
    sns.countplot(data=df, x='label')
    plt.title("Label Distribution")
    plt.xticks([0, 1], ['Fake', 'True'])
    plt.show()

def show_article_length_distribution(df):
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title("Article Length Distribution (in words)")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.show()

def show_top_words(df, label, top_n=20):
    from collections import Counter
    import string

    texts = df[df['label'] == label]['text'].dropna().str.lower()
    words = ' '.join(texts).translate(str.maketrans('', '', string.punctuation)).split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)

    words, counts = zip(*most_common)
    sns.barplot(x=counts, y=words)
    label_name = "True" if label == 1 else "Fake"
    plt.title(f"Top {top_n} Words in {label_name} Articles")
    plt.xlabel("Count")
    plt.show()
