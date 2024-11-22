import os.path
import warnings

import monai.visualize

import matplotlib.pyplot as plt

from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.transforms import (
    Compose,
    Resized,
    Spacingd,
    SpatialCropd,
    ToDeviced,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Activations,
    AsDiscrete,
    Orientationd, CropForegroundd,
)

import torch
import time
import numpy as np
from scipy.signal import savgol_filter
from monai.metrics import DiceMetric
from dataset import HNTSDataset

experiment_id = "exp2"
device = "cuda"
batch_size = 1
epochs = 0
lr_init = 1e-2
lr_min = 1e-5
weight_decay = 1e-5
os.makedirs(f"logs/{experiment_id}", exist_ok=True)

# Train data
train_transforms = Compose(
    [
        # Normalization and cropping
        RandSpatialCropd(
            keys=["image", "mask"], roi_size=[256, 256, 96], random_size=False
        ),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True),
        RandScaleIntensityd(keys="image", factors=0.05, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.05, prob=1.0),
    ]
)
train_dataset = HNTSDataset("data/train", transform=train_transforms)

# Test data
test_transforms = Compose(
    [
        NormalizeIntensityd(keys="image"),
    ]
)
test_dataset = HNTSDataset("data/test", transform=test_transforms)

# Create model
model = monai.networks.nets.SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=2,
    dropout_prob=0.2,
).to(device)

if os.path.isfile(f"logs/{experiment_id}/model.pth"):
    model.load_state_dict(torch.load(f"logs/{experiment_id}/model.pth"))
model = model.to(device)

loader = DataLoader(train_dataset, batch_size=batch_size)

# Training loop
losses = []
optimizer = torch.optim.Adam(model.parameters(), lr_init, weight_decay=weight_decay)
loss_function = monai.losses.DiceLoss(
    smooth_nr=0,
    smooth_dr=1e-5,
    squared_pred=True,
    to_onehot_y=False,  # False
    sigmoid=True,
    # weight=torch.tensor([1.1886071e+00, 8.6305177e-01]).to(device)
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=5e-5
)
writer = SummaryWriter(f"logs/writer/{experiment_id}")

step = 0
try:
    for e in range(epochs):
        print(f"Epoch: {e + 1}/{epochs}")
        time.sleep(0.1)
        model.train()
        for batch in tqdm(loader, desc="Training step"):
            inputs, labels = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(inputs)
                loss = loss_function(preds, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        lr_scheduler.step()
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
    plt.savefig(f"logs/{experiment_id}/loss.png")


# Test metric
def inference(model, input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(256, 256, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.amp.autocast('cuda'):
        return _compute(input)


post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
test_loader = DataLoader(test_dataset, batch_size=1)
for batch in tqdm(test_loader, desc="Testing steps"):
    inputs, labels = batch["image"].to(device), batch["mask"].to(device)
    preds = inference(model, inputs)
    preds = [post_trans(i) for i in decollate_batch(preds)]

    dice_metric(y_pred=preds, y=labels)

dice_score = dice_metric.aggregate()
print(f"Dice score 0: {dice_score[0]:.3f}")
print(f"Dice score 1: {dice_score[1]:.3f}")
dice_metric.reset()

# Visualization
idx = 0
sample = test_dataset[idx]

pred = inference(model, sample['image'].unsqueeze(0).to(device)).cpu()

pred = post_trans(pred)
pred = (
    torch.stack([torch.zeros(pred.shape[2:]), pred[0, 0], pred[0, 1]], axis=0)[
    :, :, :, :
    ]
    .sum(-1)
    .transpose(0, 2)
    .transpose(0, 1)
)
mask = sample["mask"]
mask = (
    torch.stack([torch.zeros(mask.shape[1:]), mask[0], mask[1]], axis=0)[:, :, :, :]
    .sum(-1)
    .transpose(0, 2)
    .transpose(0, 1)
)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
[[a.axis("off") for a in ax] for ax in axs]
for row in range(len(axs)):
    axs[row][0].imshow(pred[:, :, row], cmap="Reds")
    axs[row][1].imshow(mask[:, :, row], cmap="Reds")
fig.suptitle("Pred. vs mask")
fig.tight_layout()
plt.savefig(f"logs/{experiment_id}/pred_vs_target.png")

torch.save(model.state_dict(), f"logs/{experiment_id}/model.pth")
