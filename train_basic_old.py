import os.path
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import monai.visualize

import matplotlib.pyplot as plt

from monai.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data import create_dataset_dicts
from monai.data import Dataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    RandRotate90d,
    Compose,
    Resized,
    ToDeviced,
    SelectItemsd,
    NormalizeIntensityd,
)

import torch
import time
import numpy as np
from scipy.signal import savgol_filter
from monai.metrics import DiceMetric
from monai.handlers import CheckpointSaver

device = "cuda"

# image: (1, 256, 256, 80)
# mask: (1, 256, 256, 80)

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
    channels=(32, 64, 256, 128),
    strides=(2, 2, 2),
    num_res_units=2,
)
if os.path.isfile("logs/model.pth"):
    model.load_state_dict(torch.load("logs/model.pth"))
model = model.to(device)

batch_size = 4
epochs = 100

loader = DataLoader(train_dataset, batch_size=batch_size)

# Training loop
losses = []
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
loss_function = monai.losses.DiceCELoss(
    include_background=True, to_onehot_y=True,
    weight=torch.tensor([3.3394375e-01, 4.3348691e+02, 3.1475632e+02]).to(device)
)
lr_scheduler = monai.optimizers.lr_scheduler.WarmupCosineSchedule(optimizer, warmup_steps=5, t_total=epochs)
writer = SummaryWriter("logs/writer")
# checkpoint = CheckpointSaver("logs",
#                              save_dict={
#                                  'network': model,
#                                  'optimizer': optimizer,
#                                  'lr_scheduler': lr_scheduler
#                              },
#                              name="model1"
#                              )
step = 0
try:
    for e in range(epochs):
        print(f"Epoch: {e + 1}/{epochs}")
        time.sleep(0.1)
        model.train()
        for batch in tqdm(loader, desc="Training step"):
            inputs, labels = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_function(preds, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), step)
            losses.append(loss.item())
            step += 1
        # writer.add_scalar("learning_rate", lr_scheduler.get_lr())
        lr_scheduler.step()
        print(f"Loss: {np.mean(losses[-130:]):.3f}")
except KeyboardInterrupt:
    print("Stopped training.")

# writer.add_hparams(
#     hparam_dict=Config.__dict__,
#     metric_dict={}
# )

if epochs > 0:
    plt.plot(savgol_filter(losses, len(losses) // 2, 3), color="red", alpha=1.0)
    plt.plot(losses, color="green", alpha=0.3)
    plt.title("Training loss")
    plt.xlabel("Training step")
    plt.ylabel("Dice loss")
    plt.savefig("logs/loss.png")

# Test metric
dice_metric = DiceMetric(include_background=True, reduction="mean_batch", num_classes=3)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
for batch in tqdm(test_loader, desc="Testing steps"):
    # inputs.shape: [1, 3, 256, 256, 80]
    # labels.shape: [1, 1, 256, 256, 80]
    inputs, labels = batch["image"].to(device), batch["mask"].to(device)

    pred = model(inputs)
    dice_metric(y_pred=pred.argmax(1), y=labels)

dice_score = dice_metric.aggregate()
for i, score in enumerate(dice_score):
    print(f"Dice score {i}: {score:.4f}")
dice_metric.reset()

# Visualization
sample = test_dataset[15]
pred = model(sample["image"].to(device).unsqueeze(0))
pred = pred.softmax(1).argmax(1)
pred = pred.cpu()

monai.visualize.matshow3d(
    monai.transforms.Orientation("SPL")(pred), every_n=9, figsize=(6, 6)
)
plt.savefig("logs/pred.png")

monai.visualize.matshow3d(
    monai.transforms.Orientation("SPL")(sample["mask"]), every_n=9, figsize=(6, 6)
)
plt.savefig("logs/target.png")

torch.save(model.state_dict(), "logs/model.pth")
