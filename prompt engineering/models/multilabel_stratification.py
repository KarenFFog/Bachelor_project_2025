import json
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# load training data
metadata_path = "metadata_train.jsonl"

X_data, y_labels, all_entries = [], [], []

with open(metadata_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        all_entries.append(entry)
        X_data.append(entry["location_name"])
        y_labels.append(entry["labels"])


X_data = np.array(X_data).reshape(-1, 1)  # Needs to be 2D
mlb = MultiLabelBinarizer()
y_data = mlb.fit_transform(y_labels)
print(f"loading done", flush=True)

# Create fast lookup dict: location_name -> entry
entry_dict = {e["location_name"]: e for e in all_entries}


def get_few_shot_subset(X, y, percent, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    test_size = 1 - (percent / 100)
    X_small, y_small, _, _ = iterative_train_test_split(X_shuffled, y_shuffled, test_size=test_size)
    return X_small.flatten(), y_small


# Save function 
def save_subset(X_subset, output_path):
    with open(output_path, "w") as f_out:
        for name in X_subset:
            entry = entry_dict[name]
            json.dump(entry, f_out)
            f_out.write("\n")


# Generate and save subsets with 5 seeds
seeds = [42, 43, 44, 45, 46]
percents = [1, 5, 10, 50]

for pct in percents:
    for seed in seeds:
        X_sub, _ = get_few_shot_subset(X_data, y_data, pct, seed)
        save_subset(X_sub, f"Subsets/metadata_train_{pct}pct_seed{seed}.jsonl")

save_subset(X_data.flatten(), "metadata_train_100pct.jsonl")

print("All subsets saved")

