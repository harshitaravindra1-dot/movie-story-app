import pandas as pd
from pathlib import Path

# BLEU and ROUGE from sacrebleu + rouge-score
import sacrebleu
from rouge_score import rouge_scorer

from src.story_model import load_story_model, generate_story

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "story_tests.csv"

def template_story(name, mood, genre):
    """Simple baseline story template."""
    return (
        f"{name} was feeling very {mood}. "
        f"To cheer up, they went on a {genre}-style journey "
        f"and learned something important about themselves."
    )

def main():
    print(f"✅ Looking for test file at: {DATA_PATH}")

    if not DATA_PATH.exists():
        print("❌ story_tests.csv not found. Make sure it is in the 'data' folder.")
        raise SystemExit

    # 1. Load test data
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} test rows from story_tests.csv.")

    # 2. Load story model
    print("⏳ Loading story model (this may take a moment the first time)...")
    gen = load_story_model()
    print("✅ Story model loaded.")

    model_outputs = []
    references = []

    print("⏳ Generating stories...")
    for _, row in df.iterrows():
        name, mood, genre = row["name"], row["mood"], row["genre"]

        # reference (baseline template)
        ref = template_story(name, mood, genre)
        references.append(ref)

        # model-generated story
        story = generate_story(gen, name, mood, genre, max_len=160, temp=0.9)
        model_outputs.append(story)

    # 3. Compute BLEU with sacrebleu
    print("⏳ Computing BLEU with sacrebleu...")
    # sacrebleu expects list of system outputs + list of list of references
    bleu = sacrebleu.corpus_bleu(model_outputs, [references])

    # 4. Compute ROUGE with rouge-score
    print("⏳ Computing ROUGE with rouge-score...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_f = []
    rougel_f = []

    for ref, pred in zip(references, model_outputs):
        score = scorer.score(ref, pred)
        rouge1_f.append(score["rouge1"].fmeasure)
        rougel_f.append(score["rougeL"].fmeasure)

    avg_rouge1 = sum(rouge1_f) / len(rouge1_f)
    avg_rougel = sum(rougel_f) / len(rougel_f)

    print("\n==================== RESULTS ====================")
    print(f"BLEU score: {bleu.score:.2f}")
    print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-L F1: {avg_rougel:.4f}")
    print("=================================================\n")

if __name__ == "__main__":
    main()
