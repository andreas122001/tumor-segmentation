import os.path

import monai.visualize

import matplotlib.pyplot as plt

from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Activations,
    AsDiscrete,
)

import torch
from scipy.signal import savgol_filter
from monai.metrics import DiceMetric
from dataset import HNTSDataset
from trainer import MedSegTrainer

experiment_id = "exp1-SegResNet-lower-lr"
device = "cuda"
batch_size = 2
epochs = 150
lr_init = 1e-4  # 5e-4
lr_min = 1e-8  # 1e-6
weight_decay = 1e-5
smooth_nr = 0
smooth_dr = 1e-5
os.makedirs(f"logs/{experiment_id}", exist_ok=True)

# Train data
train_transforms = Compose(
    [
        # Normalization and cropping
        RandSpatialCropd(
            keys=["image", "mask"], roi_size=[224, 224, 96], random_size=False
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
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
)


# Create model
model = monai.networks.nets.SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=2,
    dropout_prob=0.2,
).to(device)
# model = monai.networks.nets.UNETR(
#     in_channels=1,
#     out_channels=2,
#     img_size=(224, 224, 96),
#     feature_size=16,
#     hidden_size=768,
#     num_heads=12,
#     proj_type="conv",
#     norm_name="instance",
# ).to(device)

if os.path.isfile(f"logs/{experiment_id}/model.pth"):
    model.load_state_dict(torch.load(f"logs/{experiment_id}/model.pth"))
model = model.to(device)


# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr_init, weight_decay=weight_decay)
loss_function = monai.losses.DiceCELoss(
    smooth_nr=smooth_nr,
    smooth_dr=smooth_dr,
    squared_pred=True,
    to_onehot_y=False,  # labels are already separated by channel
    sigmoid=True,  # 0 is background, 1 is label
    weight=torch.tensor([1.1698134, 0.8732383]).to(device),
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr_min
)
writer = SummaryWriter(f"logs/writer/{experiment_id}")

trainer = MedSegTrainer(
    experiment_name=experiment_id,
    model=model,
    epochs=epochs,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    loss_f=loss_function,
    logger=writer,
    save_every=20,
)
trainer.fit(train_loader=train_loader)

# step = 0
# try:
#     model.train()
#     for e in range(epochs):
#         print(f"Epoch: {e + 1}/{epochs}")
#         time.sleep(0.1)
#         for batch in tqdm(loader, desc="Training step"):
#             inputs, labels = batch["image"].to(device), batch["mask"].to(device)
#             optimizer.zero_grad()
#             preds = model(inputs)
#             loss = loss_function(preds, labels)
#             loss.backward()
#             optimizer.step()

#             losses.append(loss.item())
#         lr_scheduler.step()
#         print(f"Loss: {np.mean(losses[-130:])}")
#         step += 1
#         torch.save(model.state_dict(), f"logs/{experiment_id}/model.pth")

# except KeyboardInterrupt:
#     print("Stopped training.")

torch.save(model.state_dict(), f"logs/{experiment_id}/model.pth")
losses = trainer.losses
if len(losses) > 0:
    fig = plt.figure()
    plt.plot(savgol_filter(losses, len(losses) // 2, 3), color="red", alpha=1.0)
    plt.plot(losses, color="green", alpha=0.3)
    plt.title("Training loss")
    plt.xlabel("Training step")
    plt.ylabel("Dice loss")
    plt.savefig(f"logs/{experiment_id}/loss.png")
    writer.add_figure("loss", fig)

model.eval()

# Test data
torch.clear_autocast_cache()
del train_dataset
test_transforms = Compose(
    [
        NormalizeIntensityd(keys="image"),
    ]
)
test_dataset = HNTSDataset("data/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1)


# Test metric
def inference(model, input_):
    def _compute(input_):
        return sliding_window_inference(
            inputs=input_,
            roi_size=(224, 224, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.amp.autocast("cuda"):
        return _compute(input_)


post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
with torch.no_grad():
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
with torch.no_grad():
    pred = inference(model, sample["image"].unsqueeze(0).to(device)).cpu()
preds = [post_trans(i) for i in decollate_batch(preds)]
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
fig.suptitle("Pred. vs target")
fig.tight_layout()
plt.savefig(f"logs/{experiment_id}/pred_vs_target.png")
writer.add_figure("pred_v_target", fig)

writer.close()
