import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from baseline import *

from torchvision.models import resnet50
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys

# === LINEAR PROBE MODEL ===
class LinearProbeModel(nn.Module):
    def __init__(self, pretrained_path, in_channels=12, num_classes=43):
        super().__init__()
        self.backbone = resnet50()
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()  # remove SBERT projection
        self.classifier = nn.Linear(2048, num_classes)

        # Load weights
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded backbone weights (missing: {missing}, unexpected: {unexpected})")

        # Freeze all layers except classifier
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


#if len(sys.argv) < 3:
    #print("Usage: python script.py [1|5|10|100] [seed]")
    #sys.exit(1)

# Get the subset percentage from command-line argument
#subset = sys.argv[1]  # expects "1", "5", "10", or "100"
#seed = sys.argv[2] # 42, 43, 44, 45, 46

# ===== CONFIG =====
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
#model_path = f"Early_stopping/best_lin_model_{subset}pct_seed{seed}.pth"
model_path = f"best_lb_model_100.pth"
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
precision_micro = precision_score(y_true, y_pred_bin, average="micro")
recall_macro = recall_score(y_true, y_pred_bin, average="macro")
recall_micro = recall_score(y_true, y_pred_bin, average="micro")

# ===== UNPREDICTED CLASSES =====
predicted_any = np.any(y_pred_bin, axis=0)
unpredicted_classes = np.where(predicted_any == 0)[0]
unpredicted_names = [all_classes[i] for i in unpredicted_classes]

# ===== SAVE TO FILE =====
results = {
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "precision_macro": precision_macro,
    "precision_micro": precision_micro,
    "recall_macro": recall_macro,
    "recall_micro": recall_micro,
    "unpredicted_class_count": len(unpredicted_classes),
    "unpredicted_classes": unpredicted_names
}

#output_path = f"Early_stopping/Eval_results_lin/lin_eval_results_{subset}pct_seed{seed}.json"
output_path = "Early_stopping/Eval_results_lin/lin_eval_results_100pct.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

