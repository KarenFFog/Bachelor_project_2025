import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from baseline import *
from linear_prob import LinearProbeModel

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys

if len(sys.argv) < 2 or sys.argv[1] not in ["1", "5", "10", "100"]:
    print("Usage: python script.py [1|5|10|100]")
    sys.exit(1)

# Get the subset percentage from command-line argument
subset = sys.argv[1]  # expects "1", "5", "10", or "100"


# ===== CONFIG =====
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
model_path = f"best_lb_model_{subset}.pth"
metadata_path = "metadata_test.jsonl"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === LABEL SETUP ===
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
    folder_list=test_locations
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ===== LOAD MODEL =====
model = LinearProbeModel(
    pretrained_path=None,  # no need to re-load encoder weights
    in_channels=12,
    num_classes=43
).to(device)
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
f1_macro = f1_score(y_true, y_pred_bin, average="macro")
f1_micro = f1_score(y_true, y_pred_bin, average="micro")
precision_macro = precision_score(y_true, y_pred_bin, average="macro")
recall_macro = recall_score(y_true, y_pred_bin, average="macro")

print("\n--- Evaluation on Test Set ---")
print("F1 Score (macro):", f1_macro)
print("F1 Score (micro):", f1_micro)
print("Precision (macro):", precision_macro)
print("Recall (macro):", recall_macro)


# ===== ANALYZE UNPREDICTED CLASSES =====
import numpy as np

predicted_any = np.any(y_pred_bin, axis=0)
unpredicted_classes = np.where(predicted_any == 0)[0]
print(f"Number of unpredicted classes: {len(unpredicted_classes)}")
print("Classes with no predictions:", unpredicted_classes)


# ===== LOG RESULTS TO FILE =====
log_file = "lin_probe_results_log.csv"
with open(log_file, "a") as f:
    f.write(f"{subset}%,{f1_macro:.4f},{f1_micro:.4f},{precision_macro:.4f},{recall_macro:.4f}\n")
