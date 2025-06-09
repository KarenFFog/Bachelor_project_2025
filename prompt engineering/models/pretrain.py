from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.data import Subset

from emb_image_model import *

import json
import time
from tqdm import tqdm


# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
train_json = "embeddings_train.jsonl"
val_json = "embeddings_val.jsonl"
epochs = 12
batch_size = 32
embedding_dim = 384
patience = 7


# === INIT DATASETS ===
with open("metadata_train.jsonl") as f:
    train_folders = [json.loads(line)["location_name"] for line in f]

train_dataset = ImageTextEmbeddingDataset(
    root=root,
    embedding_jsonl="embeddings_train.jsonl",
    folder_list=train_folders
)

with open("metadata_val.jsonl") as f:
    val_folders = [json.loads(line)["location_name"] for line in f]

val_dataset = ImageTextEmbeddingDataset(
    root=root,
    embedding_jsonl="embeddings_val.jsonl",
    folder_list=val_folders
)

#train_subset = Subset(train_dataset, range(min(1000, len(train_dataset))))
#val_subset = Subset(val_dataset, range(min(500, len(val_dataset))))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# === LOAD MODEL ===
model = ResNetImageEmbedder(in_channels=12, embedding_dim=embedding_dim).to(device)
loss_fn = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# === TRAINING LOOP ===
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images = batch["image"].to(device)
        targets = batch["embedding"].to(device)
        labels = torch.ones(len(images)).to(device)  # cosine loss requires +1 label for similarity

        preds = model(images)
        loss = loss_fn(preds, targets, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f}", flush=True)

    # === VALIDATION ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            images = batch["image"].to(device)
            targets = batch["embedding"].to(device)
            labels = torch.ones(len(images)).to(device)

            preds = model(images)
            loss = loss_fn(preds, targets, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}", flush=True)

    # === Early stopping + checkpoint ===
    best_val_loss, patience_counter, should_stop = maybe_save_checkpoint(
        model=model,
        val_loss=avg_val_loss,
        best_loss=best_val_loss,
        patience_counter=patience_counter,
        patience=patience,
        epoch=epoch,
        check_point_path="pretrain_check_point.json",
        path="best_pretrain_model.pth"
    )

    if should_stop:
        print("Early stopping triggered.", flush=True)
        break
