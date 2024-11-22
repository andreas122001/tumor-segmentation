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
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    Compose,
    Resized,
    Spacingd,
    ToTensord,
    SpatialPadd,
    SpatialCropd,
    ToDeviced,
    SelectItemsd,
    NormalizeIntensityd,
    AsDiscreted,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
    AsDiscreted,
    Activations,
    AsDiscrete,
    Orientationd,
)

import torch
import time
import numpy as np
from scipy.signal import savgol_filter
from monai.metrics import DiceMetric
from monai.handlers import CheckpointSaver

device = "cuda"

class ConvertLabelIdToChannel(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


# Train data
data_dicts = create_dataset_dicts("data/train")
train_transforms = Compose(
    [
        # Load data
        LoadImaged(keys=["image", "mask"]),
        SelectItemsd(keys=["image", "mask"]),
        ConvertLabelIdToChannel(keys="mask"),
        EnsureChannelFirstd(keys=["image"]),

        # Normalization and cropping
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "mask"], roi_size=[256, 256, 96], random_size=False),
        # SpatialCropd(keys=["image", "mask"], roi_center=[256,256,40], roi_size=[196, 196, 80]),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),

        NormalizeIntensityd(keys="image", nonzero=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

        # Ensure correct size
        Resized(
            keys=["image", "mask"],
            spatial_size=(256, 256, 96),
            mode=("bilinear", "nearest"),
        ),
    ]
)
train_dataset = Dataset(data_dicts, train_transforms)

# Test data
test_transforms = Compose(
    [
        # Load data
        LoadImaged(keys=["image", "mask"]),
        SelectItemsd(keys=["image", "mask"]),
        ConvertLabelIdToChannel(keys="mask"),
        EnsureChannelFirstd(keys=["image"]),

        # Normalization and resizing
        SpatialCropd(keys=["image", "mask"], roi_center=[256,256,40], roi_size=[256, 256, 96]),
        NormalizeIntensityd(keys="image"),
        Resized(keys=["image", "mask"], spatial_size=(256, 256, 96), mode=("bilinear", "nearest")),
        ToDeviced(keys=["image", "mask"], device=device),
    ]
)
data_dicts_test = create_dataset_dicts("data/test")
test_dataset = Dataset(data_dicts_test, test_transforms)

# Create model
model = monai.networks.nets.SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=2,
    dropout_prob=0.2,
).to(device)
if os.path.isfile("logs/model.pth"):
    model.load_state_dict(torch.load("logs/model.pth"))
model = model.to(device)

batch_size = 1
epochs = 50

loader = DataLoader(train_dataset, batch_size=batch_size)

# Training loop
losses = []
optimizer = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=1e-5)
loss_function = monai.losses.DiceLoss(
    smooth_nr=0,
    smooth_dr=1e-5,
    squared_pred=True,
    to_onehot_y=False, # False
    sigmoid=True,
    # weight=torch.tensor([1.1886071e+00, 8.6305177e-01]).to(device)
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
writer = SummaryWriter("logs/writer")

step = 0
try:
    for e in range(epochs):
        print(f"Epoch: {e+1}/{epochs}")
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
        lr_scheduler.step(e)
        print(f"Loss: {np.mean(losses[-130:])}")
        step += 1
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
from monai.metrics import DiceMetric
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
test_loader = DataLoader(test_dataset, batch_size=1)
for batch in tqdm(test_loader, desc="Testing steps"):
    inputs, labels = batch['image'], batch['mask']

    logits = model(inputs)
    preds = post_trans(logits)
    dice_metric(y_pred=preds, y=labels)

dice_score = dice_metric.aggregate()
print(f"Dice score 0: {dice_score[0]:.3f}")
print(f"Dice score 1: {dice_score[1]:.3f}")
dice_metric.reset()

# Visualization

idx = 0
sample = test_dataset[idx]

pred = post_trans(model(sample['image'].unsqueeze(0))).cpu()
pred = torch.stack([torch.zeros(pred.shape[2:]), pred[0,0], pred[0,1]], axis=0)[:,:,:,:].sum(-1).transpose(0,2).transpose(0,1)
mask = sample['mask'].cpu()
mask = torch.stack([torch.zeros(mask.shape[1:]), mask[0], mask[1]], axis=0)[:,:,:,:].sum(-1).transpose(0,2).transpose(0,1)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
[[a.axis('off') for a in ax] for ax in axs]
for row in range(len(axs)):
    axs[row][0].imshow(pred[:,:,row], cmap='Reds')
    axs[row][1].imshow(mask[:,:,row], cmap='Reds')
fig.suptitle("Pred. vs mask")
fig.tight_layout()
plt.savefig("logs/pred_vs_target.png")

torch.save(model.state_dict(), "logs/model.pth")
