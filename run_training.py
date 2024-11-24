import argparse

import monai
import monai.networks
import monai.networks.nets
import monai.transforms
import torch

from dataset import HNTSDataset
from definitions.models import models
from trainer import MedSegTrainer
from torch.utils.tensorboard import SummaryWriter


def get_model(option):
    model_class, model_args = models[option]
    return model_class(**model_args)


class DefaultConfig:
    device = "cuda"
    batch_size = 4
    epochs = 400
    lr_init = 1e-4
    lr_min = 1e-10
    cpu_cores = 8
    weight_decay = 1e-5
    smooth_nr = 0
    smooth_dr = 1e-5


config = DefaultConfig()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--experiment-id", type=str, default="debug")
parser.add_argument("--model-type", type=str, default="UNet", choices=models.keys())
parser.add_argument("--batch_size", type=int, default=config.batch_size)
parser.add_argument("--epochs", type=int, default=config.epochs)
parser.add_argument("--lr-init", type=float, default=config.lr_init)
parser.add_argument("--lr-min", type=float, default=config.lr_min)
parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
parser.add_argument("--smooth-nr", type=float, default=config.smooth_nr)
parser.add_argument("--smooth-dr", type=float, default=config.smooth_dr)
parser.add_argument("--cpu-cores", type=int, default=config.cpu_cores)
parser.add_argument(
    "--device", type=str, default=config.device, choices=["cpu", "cuda"]
)
parser.add_argument("--save-every", type=int, default=50)

args = parser.parse_args()

print("Arguments:\n", args)
train_transforms = monai.transforms.Compose(
    [
        # Normalization and cropping
        monai.transforms.RandSpatialCropd(
            keys=["image", "mask"], roi_size=[224, 224, 96], random_size=False
        ),
        monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        monai.transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True),
        monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
train_dataset = HNTSDataset("data/train", transform=train_transforms)
train_loader = monai.data.DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.cpu_cores,
)

model = get_model(args.model_type)
model.to(args.device)


optimizer = torch.optim.Adam(
    model.parameters(), config.lr_init, weight_decay=config.weight_decay
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.epochs, eta_min=config.lr_min
)
loss_function = monai.losses.DiceCELoss(
    smooth_nr=config.smooth_nr,
    smooth_dr=config.smooth_dr,
    squared_pred=True,
    to_onehot_y=False,  # labels are already separated by channel
    sigmoid=True,  # 0 is background, 1 is label
    weight=torch.tensor([1.1698134, 0.8732383]).to(config.device),
)
writer = SummaryWriter(f"logs/writer/{args.experiment_id}")

# Run training
trainer = MedSegTrainer(
    model=model,
    epochs=args.epochs,
    loss_f=loss_function,
    optimizer=optimizer,
    experiment_name=args.experiment_id,
    device=args.device,
    writer=...,
    save_every=args.save_every,
    config=args.__dict__,
)
trainer.fit(train_loader=train_loader)
trainer.save_checkpoint()


# Load test dataset
del train_dataset
del train_loader
torch.clear_autocast_cache()
test_dataset = HNTSDataset(
    "data/test", transform=monai.transforms.NormalizeIntensityd(keys="image")
)
test_loader = monai.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.cpu_cores,
)

# Create visualizations
trainer.create_prediction_slider_and_video(test_dataset[9])

metrics_dict = {
    "IoU": monai.metrics.MeanIoU(include_background=True, reduction="mean"),
    "Dice": monai.metrics.DiceMetric(include_background=True, reduction="mean"),
    "DiceBatched": monai.metrics.DiceMetric(
        include_background=True, reduction="mean_batch"
    ),
}

# Run tests
score = trainer.test(
    test_loader=test_loader,
    metrics=metrics_dict,
)
print(f"Metrics: {score}")
