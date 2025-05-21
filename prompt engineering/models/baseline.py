from generate import *

class BigEarthNetS2ClassifierDataset(BigEarthNetS2Custom):
    def __init__(self, root, class_to_idx, selected_bands=None, target_size=(120, 120), transform=None, folder_list=None):
        super().__init__(root, selected_bands, target_size, transform, folder_list)
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        try: 
            image, metadata = super().__getitem__(idx)
            assert isinstance(metadata, dict), f"metadata is {type(metadata)}: {metadata}"
        
            labels = metadata["labels"]

            # Multi-label binary vector
            label_vec = torch.zeros(len(self.class_to_idx), dtype=torch.float)
            for label in labels:
                if label in self.class_to_idx:
                    label_vec[self.class_to_idx[label]] = 1.0

            return {
                "image": image,
                "label": label_vec
            }
            
        except Exception as e:
            print(f"[WARN] Failed at index {idx}: {e}", flush=True)
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

# ResNet50 model definition
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class BigEarthNetResNet50(nn.Module):
    def __init__(self, in_channels=12, num_classes=43, pretrained=False):
        super().__init__()
        
        # Load base model
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        #self.model = models.resnet50(pretrained=pretrained)
        
        # Change first conv layer to accept 10 bands instead of 3
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final classification layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def maybe_save_checkpoint(model, val_loss, best_loss, patience_counter, patience, epoch, check_point_path, path="best_model.pth"):
    """
    Saves the model if val_loss improves.
    Returns updated (best_loss, patience_counter, should_stop).
    """
    if val_loss < best_loss:
        print(f"Val loss improved ({val_loss:.4f} < {best_loss:.4f}). Saving model...", flush=True)
        torch.save(model.state_dict(), path)

        with open(check_point_path, "w") as f:
            json.dump({"epoch": epoch + 1, "val_loss": val_loss}, f)

        return val_loss, 0, False
    else:
        patience_counter += 1
        print(f"⚠️ No improvement. ({patience_counter}/{patience})", flush=True)
        return best_loss, patience_counter, patience_counter >= patience

