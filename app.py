import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ==========================================================
#           3-WORD STORY GENERATOR (UNIQUE-ISH)
#  - Works for ANY 3 words
#  - Picks a random scenario each time
#  - Uses all three words in a natural way
# ==========================================================

def build_story_from_three_words(w1: str, w2: str, w3: str) -> str:
    w1 = w1.strip()
    w2 = w2.strip()
    w3 = w3.strip()

    words_str = f"\"{w1}\", \"{w2}\" and \"{w3}\""

    # Scenario 1 – Evening in the city
    def scenario_city():
        return f"""
It was a slow evening in the city when those three words first showed up: {words_str}.  
They were scribbled in small handwriting on the back of a bus ticket someone had dropped.

On the way home, they kept turning the words over in their mind.  
Maybe "{w1}" was a place they hadn’t visited yet, "{w2}" sounded like a secret someone hadn’t told them,  
and "{w3}" felt like something warm and familiar.

Later, sitting by the window with the lights of traffic below, they opened their notebook and wrote a tiny story idea.  
In it, "{w1}" became the setting, "{w2}" became the problem, and "{w3}" became the quiet solution at the end.

The story was short and simple, but it made them smile.  
Those three words, {words_str}, had turned a normal city evening into the start of something new.
""".strip()

    # Scenario 2 – College / classroom
    def scenario_classroom():
        return f"""
In class, the lecturer decided to make things a little more interesting.  
They turned to the board and wrote three random words in big letters: {words_str}.

Everyone in the room laughed at the combination.  
“What kind of story can you possibly write with those?” someone joked.

But as the minutes passed, ideas slowly formed.  
"{w1}" became the first scene, "{w2}" turned into an unexpected character,  
and "{w3}" ended up as the last line that tied everything together.

By the time the bell rang, they had filled an entire page.  
The assignment wasn’t graded, it wasn’t even collected,  
but they walked out of the room feeling strangely proud that {words_str}  
had become a story that only they had written in exactly that way.
""".strip()

    # Scenario 3 – Train / travel
    def scenario_train():
        return f"""
On a long train journey, boredom crept in slowly like the evening light outside the window.  
To stay awake, they opened their phone and typed three words into their notes app: {words_str}.

At first it was just a little game.  
They looked out at the stations sliding past and tried to match what they saw with each word.  
A lonely sign reminded them of "{w1}", a stranger’s conversation echoed something like "{w2}",  
and a small stall selling tea and snacks made them think of "{w3}".

The journey that had felt endless suddenly felt like a story unfolding in small moments.  
By the time the train finally reached their stop, the three words had turned into a private memory.  
Whenever they thought of {words_str}, they no longer pictured simple words –  
they remembered that quiet train, the soft shaking of the compartment, and a day that became special for no big reason at all.
""".strip()

    # Scenario 4 – Night + journal
    def scenario_night():
        return f"""
Late at night, when the house was finally quiet, they opened their journal to write about the day.  
For no clear reason, three words appeared at the top of the page: {words_str}.

They stared at them for a moment, wondering why these three.  
"{w1}" reminded them of something they wanted to reach, "{w2}" felt like something that had been taken or lost,  
and "{w3}" sounded like comfort after a long, tiring day.

Line by line, a small story began to form.  
A character chased "{w1}", protected "{w2}", and finally found peace in "{w3}".  
It wasn’t perfect or dramatic, but it was honest, and it felt real.

When they closed the journal, they felt a little lighter.  
Those three simple words – {words_str} – had helped them understand their own thoughts more clearly than any long speech could.
""".strip()

    # Pick a random scenario each time → uniqueness
    scenario_fn = random.choice([scenario_city, scenario_classroom, scenario_train, scenario_night])
    return scenario_fn()


# ==========================================================
#                MOVIE RECOMMENDER FUNCTIONS
# ==========================================================

@st.cache_resource
def load_recommender_artifacts():
    """
    Load movie dataset and similarity matrix for recommendations.
    """
    movies = pd.read_csv(MODELS_DIR / "movies_clean.csv")
    similarity_matrix = joblib.load(MODELS_DIR / "similarity_matrix.pkl")
    movies["title_lower"] = movies["title"].str.lower()
    return movies, similarity_matrix


def recommend_similar(movies, similarity_matrix, title, top_k=5):
    """
    Recommend similar movies based on cosine similarity matrix.
    """
    title_lower = title.lower()
    if title_lower not in movies["title_lower"].values:
        return []
    idx = movies.index[movies["title_lower"] == title_lower][0]
    sims = list(enumerate(similarity_matrix[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_k+1]
    return [movies.iloc[i]["title"] for i, _ in sims]


# ==========================================================
#                      STREAMLIT APP
# ==========================================================

def main():
    st.title("🎬 AI-Powered Movie Recommendation & 3-Word Story Generator")

    tab1, tab2 = st.tabs(["🎥 Movie Recommendation", "📖 Story From 3 Words"])

    # ---------------------------------------------------------
    #                 TAB 1 — MOVIE RECOMMENDATION
    # ---------------------------------------------------------
    with tab1:
        st.header("Movie Recommendation")

        mood = st.selectbox("Your current mood", ["happy", "sad", "excited", "romantic", "bored"])
        fav_genre = st.text_input("Your favourite genre", "Romance")
        fav_movie = st.text_input("One movie you like", "")

        movies, sim_matrix = load_recommender_artifacts()

        if st.button("Get Movie Recommendations"):
            st.subheader("Recommended Movies For You")

            if fav_movie.strip():
                recs = recommend_similar(movies, sim_matrix, fav_movie, top_k=5)
                if not recs:
                    st.write("Couldn't find similar movies, recommending by genre instead.")
                    genre_matches = movies[movies["genres"].str.contains(fav_genre, case=False, na=False)]
                    recs = genre_matches["title"].head(5).tolist()
            else:
                genre_matches = movies[movies["genres"].str.contains(fav_genre, case=False, na=False)]
                recs = genre_matches["title"].head(5).tolist()

            if not recs:
                st.write("No recommendations found. Try a different genre or movie name.")
            else:
                for i, r in enumerate(recs, start=1):
                    st.write(f"{i}. {r}")

    # ---------------------------------------------------------
    #                 TAB 2 — STORY GENERATION
    # ---------------------------------------------------------
    with tab2:
        st.header("Story from 3 Words")

        w1 = st.text_input("Word 1", key="w1")
        w2 = st.text_input("Word 2", key="w2")
        w3 = st.text_input("Word 3", key="w3")

        if st.button("Generate Story", key="story_btn"):
            if not (w1.strip() and w2.strip() and w3.strip()):
                st.error("Please enter all three words.")
            else:
                story = build_story_from_three_words(w1, w2, w3)
                st.subheader("✨ Your Generated Story")
                st.write(story)


if __name__ == "__main__":
    main()
