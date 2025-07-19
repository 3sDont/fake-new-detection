import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Import modules and configurations
from src.data_loader import KaggleDataLoader
import config

class EDAVisualizer:
    def __init__(self, df):
        self.df = df.copy()
        # Xử lý các giá trị không phải chuỗi trong cột 'text' trước khi tính toán
        self.df['text'] = self.df['text'].astype(str)
        self.df['text_length'] = self.df['text'].apply(lambda x: len(x.split()))

    def profile(self):
        """Prints a profile of the dataset: nulls, duplicates, and descriptions."""
        print("--- DATA PROFILING ---")
        print("📌 Thông tin thiếu (Null Values):")
        print(self.df.isnull().sum(), '\n')

        print("📌 Số bản ghi trùng lặp (Duplicate Records):", self.df.duplicated().sum())

        print("\n📌 Thống kê mô tả (Descriptive Statistics):")
        print(self.df.describe(include='all'))
        print("------------------------\n")

    def visualize(self):
        """Generates and saves visualizations for the dataset."""
        print("--- GENERATING VISUALIZATIONS ---")
        
        # 1. Biểu đồ phân phối nhãn
        print("📊 1. Visualizing label distribution...")
        plt.figure(figsize=(8, 5))
        sns.countplot(x='label', data=self.df, palette='viridis')
        plt.title('Phân phối nhãn (0: FAKE, 1: REAL)', fontsize=14)
        plt.xlabel('Label')
        plt.ylabel('Số lượng')
        plt.savefig("label_distribution.png")
        plt.close()

        # 2. Biểu đồ độ dài văn bản
        print("📊 2. Visualizing text length distribution...")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='text_length', hue='label', kde=True, palette='magma', multiple='stack')
        plt.title('Phân phối độ dài văn bản theo Nhãn', fontsize=14)
        plt.xlabel('Số từ trong văn bản')
        plt.ylabel('Tần suất')
        plt.savefig("text_length_distribution.png")
        plt.close()

        # 3. WordCloud cho văn bản THẬT
        print("☁️ 3. Generating WordCloud for REAL news...")
        true_text = ' '.join(self.df[self.df['label'] == 1]['text'])
        wc_true = WordCloud(width=1000, height=500, background_color='white').generate(true_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc_true, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud - Tin THẬT', fontsize=16)
        plt.savefig("wordcloud_real.png")
        plt.close()

        # 4. WordCloud cho văn bản GIẢ
        print("☁️ 4. Generating WordCloud for FAKE news...")
        fake_text = ' '.join(self.df[self.df['label'] == 0]['text'])
        wc_fake = WordCloud(width=1000, height=500, background_color='black', colormap='autumn').generate(fake_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc_fake, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud - Tin GIẢ', fontsize=16)
        plt.savefig("wordcloud_fake.png")
        plt.close()

        # 5. Top 20 N-grams (Unigrams & Bigrams)
        print("📈 5. Visualizing top N-grams...")
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', max_features=20)
        X = vectorizer.fit_transform(self.df['text'])
        ngram_counts = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        freq_df = pd.DataFrame({'ngram': vocab, 'count': ngram_counts}).sort_values(by='count', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='count', y='ngram', data=freq_df, palette='coolwarm')
        plt.title("Top 20 Unigrams & Bigrams phổ biến nhất", fontsize=14)
        plt.xlabel("Tần suất")
        plt.ylabel("N-gram")
        plt.savefig("top_ngrams.png")
        plt.close()

        print("--- All visualizations saved as PNG files. ---")


def run_eda_pipeline():
    """Main function to run the data loading and EDA process."""
    print("Starting EDA Pipeline...")
    
    # Load data using the same loader as the main pipeline
    data_loader = KaggleDataLoader(config.DATASET_ID, config.TRUE_FILENAME, config.FAKE_FILENAME)
    df = data_loader.load()
    
    # Initialize and run the visualizer
    visualizer = EDAVisualizer(df)
    visualizer.profile()
    visualizer.visualize()
    
    print("EDA Pipeline Finished.")

if __name__ == "__main__":
    run_eda_pipeline()
