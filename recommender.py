import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movies.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_and_clean_movies():
    movies = pd.read_csv(DATA_PATH)

    # Basic cleaning
    movies = movies.drop_duplicates(subset=["title"])
    movies = movies.dropna(subset=["title", "genres"])

    # Combined text for embeddings
    movies["text"] = movies["title"].astype(str) + " " + movies["genres"].astype(str)
    return movies

def build_embeddings(movies: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["text"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix

def prepare_and_save():
    movies = load_and_clean_movies()
    vectorizer, tfidf_matrix, similarity_matrix = build_embeddings(movies)

    # Save artifacts
    movies.to_csv(MODELS_DIR / "movies_clean.csv", index=False)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(tfidf_matrix, MODELS_DIR / "tfidf_matrix.pkl")
    joblib.dump(similarity_matrix, MODELS_DIR / "similarity_matrix.pkl")

    print("✅ Movies cleaned and embeddings saved in /models")

if __name__ == "__main__":
    prepare_and_save()
