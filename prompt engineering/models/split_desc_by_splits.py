import json
from glob import glob


# === Load split lists from earlier ===
def load_split(file):
    with open(file, "r") as f:
        return set(line.strip().split(",")[0] for line in f if line.strip())

split_files = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv"
}
split_sets = {k: load_split(v) for k, v in split_files.items()}


# === Collect all descriptions into one dict ===
all_descriptions = {}

for file in sorted(glob("descriptions_part_*.jsonl")):
    print(f"Loading {file}...")
    with open(file, "r") as f:
        for line in f:
            entry = json.loads(line)
            loc = entry["location"]
            all_descriptions[loc] = entry["description"]


# === Initialize output files ===
writers = {
    "train": open("descriptions_train.jsonl", "w"),
    "val": open("descriptions_val.jsonl", "w"),
    "test": open("descriptions_test.jsonl", "w")
}

# === Track counters and unmatched ===
counters = {"train": 0, "val": 0, "test": 0}
unmatched = []

# === Assign each location to the correct split ===
for loc, desc in all_descriptions.items():
    matched = False
    for split, names in split_sets.items():
        if loc in names:
            writers[split].write(json.dumps({"location": loc, "description": desc}) + "\n")
            counters[split] += 1
            matched = True
            break
    if not matched:
        unmatched.append(loc)

# === Close files ===
for f in writers.values():
    f.close()

# === Report ===
print("Done!")
for split, count in counters.items():
    print(f"{split}: {count} descriptions written")
if unmatched:
    print(f"[WARN] {len(unmatched)} descriptions not found in any split (e.g. {unmatched[:5]})")


