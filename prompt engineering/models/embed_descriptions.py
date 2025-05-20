import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def embed_file(input_path, output_path, model_name="all-MiniLM-L6-v2", batch_size=64):
    model = SentenceTransformer(model_name)

    with open(input_path, "r") as f:
        entries = [json.loads(line) for line in f]

    texts = [entry["description"] for entry in entries]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    with open(output_path, "w") as f:
        for entry, emb in zip(entries, embeddings):
            minimal_entry = {
                "location": entry["location"],  # or entry["location_name"]
                "embedding": emb.tolist()
            }
            f.write(json.dumps(minimal_entry) + "\n")

    print(f"Done: {output_path}")

# Embed each split
embed_file("descriptions_train.jsonl", "embeddings_train.jsonl")
embed_file("descriptions_val.jsonl", "embeddings_val.jsonl")
embed_file("descriptions_test.jsonl", "embeddings_test.jsonl")

