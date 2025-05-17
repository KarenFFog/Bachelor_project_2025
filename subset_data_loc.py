import numpy as np
import matplotlib.pyplot as plt

import torchgeo
from torchgeo.datasets import MMEarth
import json


# Load the dataset 
dataset = MMEarth(root="", subset="MMEarth100k")

# Define JSONL file path
jsonl_file_path = "prompt engineering\mmearth_coordinates.jsonl"

# Open the file and write data in JSONL format
with open(jsonl_file_path, "w", encoding="utf-8") as file:
    for i in range(100):  # Loop through first 100 samples
        sample = dataset[i]  # Get sample as a dictionary
        data = {
            "latitude": sample["lat"],
            "longitude": sample["lon"]
        }
        file.write(json.dumps(data) + "\n")  # Write as JSONL (one line per sample)

print(f"Coordinates saved successfully: {jsonl_file_path}")
