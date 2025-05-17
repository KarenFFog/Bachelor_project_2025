import json
from torchgeo.datasets import BigEarthNet

def extract_split_names(split, limit=None):
    print(f"Loading BigEarthNet split: {split}...", flush=True)

    dataset = BigEarthNet(
        root="BigEarthNet_data_train_s2",
        split=split,
        bands="s2",
        num_classes=43,
        download=False
    )

    folder_names = [sample["image"].parent.name for sample in dataset]

    if limit:
        folder_names = folder_names[:limit]
        print(f"[DEBUG] Limiting to first {limit} samples for {split}.", flush=True)

    return folder_names

splits = {}
limit_per_split = 1000  # Set to an integer like 1000 for test runs

for split in ["train", "val", "test"]:
    print(f"\n--- Extracting {split} split ---", flush=True)
    folders = extract_split_names(split, limit=limit_per_split)
    splits[split] = folders
    print(f"Found {len(folders)} folders in {split}.", flush=True)

with open("ben_splits.json", "w") as f:
    json.dump(splits, f)

print("\nSaved ben_splits.json successfully!", flush=True)


