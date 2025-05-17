from torchgeo.datasets import BigEarthNet

dataset_val = BigEarthNet(
    root='BigEarthNet_data_val_s2',
    split='val',
    bands='s2',
    num_classes=43,
    transforms=None,
    download=True,
    checksum=False
)

