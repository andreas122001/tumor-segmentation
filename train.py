import gzip
from monai.data import Dataset
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
from monai.data import PILReader
from monai.config import print_config
import monai
import matplotlib.pyplot as plt
import monai.transforms
import monai.visualize

from ignite.engine import (
    Engine,
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from monai.data import DataLoader
from tqdm import tqdm
from data import create_dataset_dicts
from monai.data import Dataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    Compose, 
    Resized, 
    Spacingd,
    ToTensord,
    SpatialPadd,
    ToDeviced,
    SelectItemsd,
    NormalizeIntensityd,
    AsDiscreted,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
    AsDiscreted
)

import torch
import time
import numpy as np

device = "cuda"

# Train data
data_dicts = create_dataset_dicts("data/train")
train_transforms = Compose(
    [
        # Load data
        LoadImaged(keys=["image", "mask"]),
        SelectItemsd(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Normalization and resizing
        NormalizeIntensityd(keys="image"),
        Resized(
            keys=["image", "mask"],
            spatial_size=(256, 256, 80),
            mode=("bilinear", "nearest"),
        ),
        RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=(0, 1)),
        ToDeviced(keys=["image", "mask"], device=device),
    ]
)
train_dataset = Dataset(data_dicts, train_transforms)


# Test data
test_transforms = Compose(
    [      
        # Load data
        LoadImaged(keys=["image", "mask"]),
        SelectItemsd(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),

        # Ensure correct size
        Resized(
            keys=["image", "mask"],
            spatial_size=(256, 256, 80),
            mode=("bilinear", "nearest"),
        ),
    ]
)
data_dicts_test = create_dataset_dicts("data/test")
test_dataset = Dataset(data_dicts_test, test_transforms)

# Create model
model = monai.networks.nets.UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(32, 64, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

batch_size = 8

loader = DataLoader(train_dataset, batch_size=batch_size)


# Training loop
losses = []
epochs = 30
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
loss_function = monai.losses.DiceCELoss(include_background=False, softmax=True, to_onehot_y=True)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for e in range(epochs):
    print(f"Epoch: {e}")
    time.sleep(0.1)
    model.train()
    for batch in tqdm(loader, desc="Training step"):
        inputs, labels = batch['image'].to(device), batch['mask'].to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_function(preds, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    lr_scheduler.step()
    print(f"Loss: {np.mean(losses[-130:])}")


from scipy.signal import savgol_filter

plt.plot(savgol_filter(losses, 250, 3), color="red", alpha=1.0)
plt.plot(losses, color="green", alpha=0.3)
plt.title("Training loss")
plt.xlabel("Training step")
plt.ylabel("Dice loss")
plt.savefig("loss.png")


# Test metric
from monai.metrics import DiceMetric 

dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=3)
test_loader = DataLoader(test_dataset, batch_size=8)
for batch in tqdm(test_loader, desc="Testing steps"):
    inputs, labels = batch['image'], batch['mask']

    logits = model(inputs)
    # preds = logits#.softmax(1).argmax(1)

    dice_metric(y_pred=preds, y=labels)
    
dice_score = dice_metric.aggregate().item()
print(f"Dice score: {dice_score:.3f}")
dice_metric.reset()


sample = test_dataset[15]
pred = model(sample['image'].unsqueeze(0))
pred = pred.softmax(1).argmax(1)

monai.visualize.matshow3d(monai.transforms.Orientation("SPL")(pred), every_n=9, figsize=(6,6))
plt.savefig("pred.png")

monai.visualize.matshow3d(monai.transforms.Orientation("SPL")(sample['mask']), every_n=9, figsize=(6,6))
plt.savefig("target.png")