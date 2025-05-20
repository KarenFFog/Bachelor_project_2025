import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from baseline import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ===== CONFIG =====
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
model_path = "best_model.pth"
metadata_path = "metadata_test.jsonl"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LABEL SETUP =====
all_classes = [
    "Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
    "Road and rail networks and associated land", "Port areas", "Airports", "Mineral extraction sites",
    "Dump sites", "Construction sites", "Green urban areas", "Sport and leisure facilities",
    "Non-irrigated arable land", "Permanently irrigated land", "Rice fields", "Vineyards",
    "Fruit trees and berry plantations", "Olive groves", "Pastures",
    "Annual crops associated with permanent crops", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland", "Moors and heathland", "Sclerophyllous vegetation",
    "Transitional woodland/shrub", "Beaches, dunes, sands", "Bare rock", "Sparsely vegetated areas",
    "Burnt areas", "Inland marshes", "Peatbogs", "Salt marshes", "Salines", "Intertidal flats",
    "Water courses", "Water bodies", "Coastal lagoons", "Estuaries", "Sea and ocean"
]
class_to_idx = {label: i for i, label in enumerate(all_classes)}

# ===== LOAD TEST LOCATIONS =====
with open(metadata_path, "r") as f:
    test_locations = [json.loads(line)["location_name"] for line in f]

# ===== INIT TEST DATASET =====
test_dataset = BigEarthNetS2ClassifierDataset(
    root=root,
    class_to_idx=class_to_idx,
    folder_list=test_locations,
    selected_bands=[
        'B02', 'B03', 'B04', 'B05', 'B06',
        'B07', 'B08', 'B8A', 'B11', 'B12'
    ]
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ===== LOAD MODEL =====
model = BigEarthNetResNet50(in_channels=10, num_classes=43, pretrained=False).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ===== EVALUATION LOOP =====
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()

        outputs = model(images).sigmoid().cpu().numpy()
        all_preds.append(outputs)
        all_targets.append(labels)


# ===== AGGREGATE RESULTS =====
y_pred = np.vstack(all_preds)
y_true = np.vstack(all_targets)
y_pred_bin = (y_pred > 0.5).astype(int)


# ===== METRICS =====
print("\n--- Evaluation on Test Set ---")
print("F1 Score (macro):", f1_score(y_true, y_pred_bin, average="macro"))
print("F1 Score (micro):", f1_score(y_true, y_pred_bin, average="micro"))
print("Precision (macro):", precision_score(y_true, y_pred_bin, average="macro"))
print("Recall (macro):", recall_score(y_true, y_pred_bin, average="macro"))

