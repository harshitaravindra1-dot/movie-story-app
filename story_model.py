from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DEFAULT_MODEL_NAME = "distilgpt2"  # small GPT-2 variant

def load_story_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Loads a small causal language model and returns a text-generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return gen

def generate_story(generator, name: str, mood: str, genre: str, max_len: int = 180, temp: float = 0.9):
    """
    (Used in evaluation script) – story based on name, mood, genre.
    You can keep this for BLEU/ROUGE evaluation.
    """
    prompt = (
        f"Write a short {genre} story for a person named {name} who is feeling {mood}. "
        f"The story should be simple, engaging, and positive."
    )
    out = generator(
        prompt,
        max_length=max_len,
        num_return_sequences=1,
        do_sample=True,
        temperature=temp,
    )
    return out[0]["generated_text"]