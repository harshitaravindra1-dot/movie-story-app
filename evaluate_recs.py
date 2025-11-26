# src/evaluate_recs.py
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

movies = pd.read_csv(MODELS_DIR / "movies_clean.csv")
similarity_matrix = joblib.load(MODELS_DIR / "similarity_matrix.pkl")

movies["title_lower"] = movies["title"].str.lower()

def recommend_similar(title, top_k=5):
    title_lower = title.lower()
    if title_lower not in movies["title_lower"].values:
        return []
    idx = movies.index[movies["title_lower"] == title_lower][0]
    sims = list(enumerate(similarity_matrix[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_k+1]
    return [movies.iloc[i]["title"] for i, _ in sims]

def hit_rate_at_k(test_data, k=5):
    hits = 0
    total = 0
    for user, info in test_data.items():
        liked = info["liked"]
        for movie in liked:
            recs = recommend_similar(movie, top_k=k)
            total += 1
            if any(r in liked for r in recs):
                hits += 1
    return hits / total if total > 0 else 0

def main():
    test_users = {
        "user1": {"liked": ["Toy Story", "Jumanji"]},
        "user2": {"liked": ["The Matrix", "Inception"]},
    }

    hr = hit_rate_at_k(test_users, k=5)
    print("Hit Rate@5:", hr)

if __name__ == "__main__":
    main()
