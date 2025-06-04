import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import os
from torchvision.models import resnet50
from generate import *
from baseline import *

import sys

class FineTuneModel(nn.Module):
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

        # DO NOT freeze any layers — full fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
batch_size = 32
start_epoch = 0
epochs = 50
patience = 3
best_val_loss = float("inf")
patience_counter = 0


# === LABEL SETUP ===
all_classes = [
    "Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
    "Road and rail networks and associated land", "Port areas", "Airports", "Mineral extraction sites",
    "Dump sites", "Construction sites", "Green urban areas", "Sport and leisure facilities",
    "Non-irrigated arable land", "Permanently irrigated land", "Rice fields", "Vineyards", 
    "Fruit trees and berry plantations", "Olive groves", "Pastures",
    "Annual crops associated with permanent crops", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland",
    "Moors and heathland", "Sclerophyllous vegetation", "Transitional woodland/shrub", "Beaches, dunes, sands",
    "Bare rock", "Sparsely vegetated areas", "Burnt areas", "Inland marshes", "Peatbogs", "Salt marshes",
    "Salines", "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons", "Estuaries",
    "Sea and ocean"
]

class_to_idx = {label: i for i, label in enumerate(all_classes)}


# === LOAD METADATA ===
# Get the subset percentage from command-line argument
subset = sys.argv[1]  # expects "1", "5", "10", or "100"
seed = sys.argv[2] 

# Determine the correct metadata file
train_file = "metadata_train.jsonl" if subset == "100" else f"Subsets/metadata_train_{subset}pct_seed{seed}.jsonl"

# Load training locations
with open(train_file, "r") as f:
    train_locations = [json.loads(line)["location_name"] for line in f]

with open("metadata_val.jsonl", "r") as f:
    val_locations = [json.loads(line)["location_name"] for line in f]


# === INIT DATASETS ===
train_dataset = BigEarthNetS2ClassifierDataset(
    root=root,
    class_to_idx=class_to_idx,
    folder_list=train_locations
)

val_dataset = BigEarthNetS2ClassifierDataset(
    root=root,
    class_to_idx=class_to_idx,
    folder_list=val_locations
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# === INIT MODEL ===
model = FineTuneModel(pretrained_path="best_pretrain_model.pth").to(device) ######
print(f"Model loaded", flush=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

# === TRAINING LOOP ===
loss_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}", flush=True)
    print(f"{len(train_loader)}", flush=True)


    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            val_loss += criterion(logits, labels).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}", flush=True)

    loss_history.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_loss": avg_val_loss
    })

    best_val_loss, patience_counter, should_stop = maybe_save_checkpoint(
        model=model,
        val_loss=avg_val_loss,
        best_loss=best_val_loss,
        patience_counter=patience_counter,
        patience=patience,
        epoch=epoch,
	check_point_path = f"Early_stopping/ft_checkpoint_{subset}pct_seed{seed}.json",
        path=f"Early_stopping/best_ft_model_{subset}pct_seed{seed}.pth"
    )

    if should_stop:
        print("Early stopping triggered.", flush=True)
        break

with open(f"Early_stopping/ft_loss_history_{subset}pct_seed{seed}.json", "w") as f:
    json.dump(loss_history, f, indent=2)

