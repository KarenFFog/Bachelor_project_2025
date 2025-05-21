import json
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load your JSONL file
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

# Subset function
def get_few_shot_subset(X, y, percent):
    test_size = 1 - (percent / 100)
    X_small, y_small, _, _ = iterative_train_test_split(X, y, test_size=test_size)
    return X_small.flatten(), y_small

# Save function 
def save_subset(X_subset, output_path):
    with open(output_path, "w") as f_out:
        for name in X_subset:
            entry = entry_dict[name]
            json.dump(entry, f_out)
            f_out.write("\n")

# Generate and save subsets
for pct in [10, 5, 1]:
    X_sub, _ = get_few_shot_subset(X_data, y_data, pct)
    save_subset(X_sub, f"metadata_train_{pct}pct.jsonl")

print(f"all subsets saved", flush=True)
