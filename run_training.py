import argparse
from typing import Optional

import monai
import monai.networks
import monai.networks.nets
import monai.transforms
import torch

from dataset import HNTSDataset
from definitions.models import get_model, models
from trainer import MedSegTrainer
from torch.utils.tensorboard import SummaryWriter


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
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment-id", type=str, default="debug")
parser.add_argument("--model-type", type=str, default="UNet", choices=models.keys())
parser.add_argument("--batch-size", type=int, default=config.batch_size)
parser.add_argument("--epochs", type=int, default=config.epochs)
parser.add_argument("--lr-init", type=float, default=config.lr_init)
parser.add_argument("--lr-min", type=float, default=config.lr_min)
parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
parser.add_argument("--smooth-nr", type=float, default=config.smooth_nr)
parser.add_argument("--smooth-dr", type=float, default=config.smooth_dr)
parser.add_argument("--cpu-cores", type=int, default=config.cpu_cores)
parser.add_argument("--no-training", action="store_true")
parser.add_argument("--no-testing", action="store_true")
parser.add_argument(
    "--device", type=str, default=config.device, choices=["cpu", "cuda"]
)
parser.add_argument("--save-every", type=int, default=1000)

args = parser.parse_args()

print("Arguments:\n", args)

if not args.no_training:
    # Load training dataset
    train_transforms = monai.transforms.Compose(
        [
            # Normalization and cropping
        monai.transforms.RandSpatialCropd(
            keys=["image", "mask"], roi_size=[224, 224, 96], random_size=False
        ),
            monai.transforms.RandFlipd(
                keys=["image", "mask"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "mask"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "mask"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(keys="image", nonzero=True),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    train_dataset = HNTSDataset("data/train", transform=train_transforms)
    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.cpu_cores,
    )

model = get_model(args.model_type)
model.to(args.device)


optimizer = torch.optim.Adam(
    model.parameters(), args.lr_init, weight_decay=args.weight_decay
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.lr_min
)
loss_function = monai.losses.DiceLoss(
    smooth_nr=args.smooth_nr,
    smooth_dr=args.smooth_dr,
    squared_pred=True,
    to_onehot_y=False,  # labels are already separated by channel
    sigmoid=True,  # 0 is background, 1 is label
)
writer = SummaryWriter(f"logs/writer/{args.experiment_id}")

# Run training
trainer = MedSegTrainer(
    model=model,
    epochs=args.epochs,
    loss_f=loss_function,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    experiment_name=args.experiment_id,
    device=args.device,
    writer=writer,
    save_every=args.save_every,
    config=args.__dict__,
)
if args.checkpoint:
    trainer.load_checkpoint(checkpoint_path=args.checkpoint)

if not args.no_training:
    trainer.fit(train_loader=train_loader)
    trainer.save_checkpoint()
    del train_dataset
    del train_loader
else:
    print("Skipping training...")

# Not sure if this has an effect or not
torch.clear_autocast_cache()


if not args.no_testing:
    # Load test dataset
    test_dataset = HNTSDataset(
        "data/test", transform=monai.transforms.NormalizeIntensityd(keys="image")
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.cpu_cores,
    )

    # Run tests
    metrics_dict = {
        "IoU": monai.metrics.MeanIoU(include_background=True, reduction="mean"),
        "Dice": monai.metrics.DiceMetric(include_background=True, reduction="mean"),
        "DiceBatched": monai.metrics.DiceMetric(
            include_background=True, reduction="mean_batch"
        ),
    }

    score = trainer.test(
        test_loader=test_loader,
        metrics=metrics_dict,
    )
    print(f"Metrics: {score}")

    del test_loader
    torch.clear_autocast_cache()

    # Create visualizations
    trainer.create_prediction_slider_and_video(
        test_dataset[9]
    )  # use random sample like no. 9
else:
    print("Skipping testing...")

# Log param count so it can be tracked
total_params = torch.tensor(0)
for params in model.parameters():
    n_params = torch.prod(torch.tensor(params.shape))
    total_params += n_params
writer.add_scalar("param_count", total_params.item())

trainer.writer.close()

print("Done!")
