# embed descriptions
from sentence_transformers import SentenceTransformer
import json

def embed_and_save_descriptions(input_path, output_path, model_name="all-MiniLM-L6-v2", batch_size=64):
    # Load the pre-trained model
    model = SentenceTransformer(model_name)

    # Load data from the JSONL file
    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    # Extract descriptions and encode them
    texts = [entry["description"] for entry in entries]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Save only location and embedding to the output JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry, emb in zip(entries, embeddings):
            minimal_entry = {
                "location": entry["location"],
                "embedding": emb.tolist()
            }
            f.write(json.dumps(minimal_entry) + "\n")

    print(f"[DONE] Saved embeddings to {output_path}")


# make (image, emb) pairs, subclass of base class
class BigEarthNetS2EmbeddingDataset(BigEarthNetS2Custom):
    def __init__(self, root, embedding_file, selected_bands=None, target_size=(120, 120), transform=None):
        super().__init__(root, selected_bands, target_size, transform)

        # Load embedding dict: location -> embedding
        with open(embedding_file, "r") as f:
            self.embedding_dict = {
                json.loads(line)["location"]: json.loads(line)["embedding"]
                for line in f
            }

        # Filter only folders with an embedding
        self.patch_folders = [f for f in self.patch_folders if f in self.embedding_dict]

    def __getitem__(self, idx):
        image, metadata = super().__getitem__(idx)
        loc_name = metadata["location_name"]
        embedding = torch.tensor(self.embedding_dict[loc_name], dtype=torch.float)

        return {
            "image": image,
            "embedding": embedding
        }

# example use...
dataset = BigEarthNetS2EmbeddingDataset(root="/BigEarthNet-v1.0", embedding_file="embeddings_only.jsonl")

sample = dataset[0]
print(sample["image"].shape)
print(sample["embedding"].shape)


# dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,     # or 64??
    shuffle=True,
    num_workers=4,    
    pin_memory=True
)
