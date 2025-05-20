from generate import *
from baseline import maybe_save_checkpoint

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights



# === Data class ===
class ImageTextEmbeddingDataset(BigEarthNetS2Custom):
    def __init__(self, root, embedding_jsonl, selected_bands=None, target_size=(120, 120), transform=None, folder_list=None):
        super().__init__(root, selected_bands, target_size, transform)

        # Load location → embedding
        with open(embedding_jsonl, "r") as f:
            self.embedding_map = {
                json.loads(line)["location"]: json.loads(line)["embedding"]
                for line in f
            }

        # Intersect patch folders with valid locations
        valid_locations = set(self.embedding_map.keys())
        if folder_list is not None:
            valid_locations = valid_locations.intersection(folder_list)

        self.patch_folders = [f for f in self.patch_folders if f in valid_locations]

    def __getitem__(self, idx):
        try:
            image, metadata = super().__getitem__(idx)
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata is not a dict: {type(metadata)}")

            location = metadata["location_name"]
            embedding = torch.tensor(self.embedding_map[location], dtype=torch.float)

            return {"image": image, "embedding": embedding}

        except Exception as e:
            print(f"[WARN] Skipping sample at index {idx}: {e}", flush=True)
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

    

# == ResNet-50 model ===
class ResNetImageEmbedder(nn.Module):
    def __init__(self, in_channels=10, embedding_dim=384, pretrained=False):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Change first conv to accept 10-band Sentinel-2
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier head with projection to SBERT space
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.backbone(x)


