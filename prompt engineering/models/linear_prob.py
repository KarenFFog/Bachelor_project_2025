import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import os
from torchvision.models import resnet50
from generate import *
from baseline import *
from run_baseline import class_to_idx

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


# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
batch_size = 32
epochs = 50
patience = 3


# === LOAD METADATA ===
# Get the subset percentage from command-line argument
subset = sys.argv[1]  # expects "1", "5", "10", or "100"

# Determine the correct metadata file
train_file = "metadata_train.jsonl" if subset == "100" else f"metadata_train_{subset}pct.jsonl"

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
model = LinearProbeModel(pretrained_path="best_pretrain_model.pth").to(device) ######
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)


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
    
    print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")
    print(f"{len(train_loader)}")


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
    print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
	
    best_val_loss, patience_counter, should_stop = maybe_save_checkpoint(
        model=model, 
        val_loss=avg_val_loss, 
        best_loss=best_val_loss, 
        patience_counter=patience_counter, 
        patience=patience, 
        epoch=epoch,
	check_point_path = "lin_prob_checkinfo_{subset}.json",
        path="best_lb_model_{subset}.pth"
    )

    if should_stop:
        print("Early stopping triggered.", flush=True)
        break

# evaluate on test set

# plot results (% of train used, F1 score)


