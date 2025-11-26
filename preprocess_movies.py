import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def main():
    # 1. Load original MovieLens movies.csv
    movies_path = DATA_DIR / "movies.csv"
    movies = pd.read_csv(movies_path)

    # 2. Some basic cleaning
    movies = movies[["movieId", "title", "genres"]].copy()

    # Replace no-genre
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    # 3. Create one combined text column: title + genres
    movies["combined"] = (
        movies["title"].fillna("") + " " + movies["genres"].fillna("")
    )

    # 4. TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined"])

    # 5. Cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # 6. Save outputs into data/
    movies_clean_path = DATA_DIR / "movies_clean.csv"
    sim_matrix_path = DATA_DIR / "similarity_matrix.pkl"

    movies.to_csv(movies_clean_path, index=False)
    joblib.dump(cosine_sim, sim_matrix_path)

    print("Saved:")
    print(" -", movies_clean_path)
    print(" -", sim_matrix_path)

if __name__ == "__main__":
    main()
