from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.data import Subset

from baseline import *

import json
import time

start = time.time()

# === LOAD METADATA ===
with open("metadata_train.jsonl", "r") as f:
    train_locations = [json.loads(line)["location_name"] for line in f]

with open("metadata_val.jsonl", "r") as f:
    val_locations = [json.loads(line)["location_name"] for line in f]


# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"


# === LABEL SETUP ===
all_classes = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Olive groves",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Burnt areas",
    "Inland marshes",
    "Peatbogs",
    "Salt marshes",
    "Salines",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean"
]

class_to_idx = {label: i for i, label in enumerate(all_classes)}


# === INIT TRAIN SET ===
start_time = time.time()

dataset = BigEarthNetS2ClassifierDataset(
    root=root,
    class_to_idx=class_to_idx,
    folder_list=train_locations
)

end_time = time.time()
print(f"train dataset initialized in {(end_time - start_time):.2f} seconds", flush=True)

# start with subset - works! 
# small_dataset = Subset(dataset, range(1000))  # First 1000 samples
# dataloader = DataLoader(small_dataset, batch_size=32, shuffle=True, num_workers=4)

# dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)


# === INIT VAL SET ===
val_dataset = BigEarthNetS2ClassifierDataset(
    root=root,
    class_to_idx=class_to_idx,
    folder_list=val_locations
)
print("val dataset initialized", flush=True)

# val dataloader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)



# === INIT MODEL ===
model = BigEarthNetResNet50(in_channels=12, num_classes=43, pretrained=False).to(device)


# Resume training if checkpoint exists
start_epoch = 0
best_val_loss = float("inf")

'''
if os.path.exists("checkpoint_info.json"):
    with open("checkpoint_info.json", "r") as f:
        checkpoint = json.load(f)
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]

    model.load_state_dict(torch.load("best_model.pth"))
    print(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}", flush=True)
'''

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("model initialized", flush=True)


# === TRAINING LOOP ===
epochs = 50
patience_counter = 0
patience = 3

for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # just a check
        if epoch == start_epoch and i == 0:
            print(f"Image shape: {images.shape}", flush=True)
            print(f"Label shape: {labels.shape}", flush=True)

        outputs = model(images)  # shape: [B, 43]
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch {epoch+1} | Batch {i+1} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}", flush=True)
    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}", flush=True)

    best_val_loss, patience_counter, should_stop = maybe_save_checkpoint(
            model=model, 
            val_loss=avg_val_loss, 
            best_loss=best_val_loss, 
            patience_counter=patience_counter, 
            patience=patience, 
            epoch=epoch,
            check_point_path="baseline_check_point.json",
            path="best_baseline_model_12.pth"
    )

    if should_stop:
        print("Early stopping triggered.", flush=True)
        break

print(f"Total training time: {(time.time() - start) / 60:.2f} minutes", flush=True)

