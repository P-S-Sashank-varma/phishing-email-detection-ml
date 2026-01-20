import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def preprocess_data(input_file, output_file):
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Ensure the dataset has 'text' and 'label' columns
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Input CSV must have 'text' and 'label' columns.")

    print(f"Total samples loaded: {len(data)}")
    print(f"Phishing samples: {(data['label'] == 1).sum()}")
    print(f"Legitimate samples: {(data['label'] == 0).sum()}")

    # Vectorize the text using TF-IDF with improved parameters
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit to top 1000 features
        min_df=1,           # Minimum document frequency
        max_df=0.8,         # Maximum document frequency
        ngram_range=(1, 2), # Use unigrams and bigrams
        stop_words='english'
    )
    X = vectorizer.fit_transform(data['text']).toarray()

    # Extract labels
    y = data['label'].values

    # Save preprocessed data and vectorizer
    with open(output_file, "wb") as f:
        pickle.dump((X, y, vectorizer), f)

    print(f"Preprocessed data saved to {output_file}")
    print(f"Feature vector shape: {X.shape}")

if __name__ == "__main__":
    preprocess_data("data/phishing_emails_expanded.csv", "data/preprocessed_data.pkl")
