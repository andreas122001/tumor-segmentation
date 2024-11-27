import os
import time
from matplotlib import pyplot as plt
import monai
import torch
from tqdm import tqdm
from scipy.signal import savgol_filter
import imageio
from matplotlib import colors
from definitions.colors import GTVp_col, GTVn_col

from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete


class VoidWriter(SummaryWriter):
    "Dummy writer that doesn't log anything"

    def __init__(self, *args, **kwargs):
        pass


class MedSegTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        loss_f: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        experiment_name: str = "test",
        device="cuda",
        writer: SummaryWriter = VoidWriter(),
        save_every=-1,
        config: dict = {},
    ) -> None:
        super().__init__()
        self.model = model
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.lr_scheduler = lr_scheduler
        self.step = 0
        self.epoch = 0
        self.writer = writer
        self.device = device
        self.save_every = save_every
        self.scaler = torch.GradScaler()
        self.losses = []
        self.log_path = f"logs/{experiment_name}"
        self.save_path = f"{self.log_path}/checkpoints"
        self.config = config

        os.makedirs(self.save_path, exist_ok=True)

    def fit(self, train_loader: torch.utils.data.DataLoader):
        print("=== Starting training! ===")
        self.model.to(self.device)
        self.model.train()
        for e in range(self.epoch, self.epochs):
            epoch_loss = 0
            print(f"Epoch {e+1}/{self.epochs}")
            for batch in tqdm(train_loader, desc="Training step", leave=True):
                loss = self.training_step(batch)
                self.writer.add_scalar("training_loss", loss, self.step, new_style=True)
                self.losses.append(loss)
                epoch_loss += loss
            # After epoch
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.writer.add_scalar(
                "learning_rate",
                (
                    self.lr_scheduler.get_last_lr()[0]
                    if self.lr_scheduler
                    else self.optimizer.param_groups[0]["lr"]
                ),
                self.step,
                new_style=True,
            )
            if self.save_every != -1 and (e + 1) % self.save_every == 0:
                self.save_checkpoint()
            print(f"Loss epoch: {epoch_loss:.3f}")
            self.epoch += 1
        self.log_loss_plot()

    def training_step(self, batch: dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        inputs, targets = self._unpack_batch(batch)

        # with torch.amp.autocast("cuda"):  # mixed precision to save the precious memory
        logits = self.model(inputs)
        loss = self.loss_f(logits, targets)
        loss.backward()
        # self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )  # all the cool kids use grad norm clipping
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.optimizer.step()

        self.step += 1
        return loss.item()

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        metrics: monai.metrics.CumulativeIterationMetric,
    ):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            metric_results = {}
            for batch in tqdm(test_loader, desc="Testing steps", leave=True):
                inputs, targets = self._unpack_batch(batch)

                preds = self.inference(inputs)
                # Update metric
                for name, metric in metrics.items():
                    metric(y_pred=preds, y=targets)

        # Aggregate, gather, and reset
        for name, metric in metrics.items():
            aggregate = metric.aggregate()
            if aggregate.shape[0] > 1:
                for i, value in enumerate(aggregate):
                    # This is ugly, but it works
                    metric_results[f"{name}[{test_loader.dataset.id_to_label[i]}]"] = (
                        value.item()
                    )
            else:
                metric_results[name] = aggregate.item()
            metric.reset()

        # Write to tensorboard
        for name, val in metric_results.items():
            self.writer.add_scalar(
                f"test/{name}",
                val,
                global_step=self.step,
                new_style=True,
            )

        self.writer.add_hparams(
            run_name=".",
            hparam_dict=self.config,
            metric_dict=metric_results,
            global_step=self.step,
        )
        return metric_results

    def inference(self, input_, raw=False):
        self.model.eval()
        self.model.to(self.device)

        def _compute(input_):
            return sliding_window_inference(
                inputs=input_,
                roi_size=(224, 224, 96),
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.2,
            )

        with torch.no_grad():
            if raw:
                return _compute(input_.to(self.device))

            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            return post_trans(_compute(input_.to(self.device)))

    def create_prediction_slider_and_video(self, sample):
        image = sample["image"]
        mask_target = sample["mask"]
        mask_pred = self.inference(image.unsqueeze(0))[0].cpu()

        to_spl = monai.transforms.Orientation(axcodes="SPL")
        image = to_spl(image)
        mask_target = to_spl(mask_target)
        mask_pred = to_spl(mask_pred)

        write_dir = f"{self.log_path}/prediction_slices"
        os.makedirs(f"{write_dir}", exist_ok=True)
        title = ["Prediction", "Target"]
        for slice_idx in tqdm(
            range(mask_target.shape[1]), desc="Creatig prediction slices"
        ):
            image_slice = image[0, slice_idx]
            label0 = [mask_pred[0, slice_idx], mask_target[0, slice_idx]]
            label1 = [mask_pred[1, slice_idx], mask_target[1, slice_idx]]
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            for i in range(2):
                axs[i].imshow(image_slice, cmap="gray", alpha=1.0)
                axs[i].imshow(
                    label0[i],
                    cmap=colors.ListedColormap(["white", GTVp_col]),
                    alpha=0.5,
                )
                axs[i].imshow(
                    label1[i],
                    cmap=colors.ListedColormap(["white", GTVn_col]),
                    alpha=0.5,
                )
                axs[i].set_title(title[i])
                axs[i].axis("off")
            fig.tight_layout()
            fig.savefig(f"{write_dir}/{slice_idx:04d}.png")
            self.writer.add_figure("prediction_vs_target", fig, global_step=slice_idx)
        frames = [
            imageio.imread(os.path.join(write_dir, f))
            for f in sorted(os.listdir(write_dir))
        ]
        print("Generating GIF...")
        imageio.mimsave(f"{self.log_path}/slices.gif", frames, format="GIF", fps=10)

    def log_loss_plot(self):
        if len(self.losses) > 0:
            fig = plt.figure()
            plt.plot(
                savgol_filter(self.losses, len(self.losses) // 2, 3),
                color="red",
                alpha=1.0,
            )
            plt.plot(self.losses, color="green", alpha=0.3)
            plt.title("Training loss")
            plt.xlabel("Training step")
            plt.ylabel("Dice loss")
            plt.savefig(f"logs/{self.experiment_name}/loss.png")
            self.writer.add_figure("loss", fig)

    def save_checkpoint(self, checkpoint_name=None):
        checkpoint_name = checkpoint_name if checkpoint_name else self.step
        save_path = f"{self.save_path}/{checkpoint_name}"

        os.makedirs(f"{save_path}", exist_ok=True)
        torch.save(self.model.state_dict(), f"{save_path}/model.pth")
        torch.save(self.optimizer.state_dict(), f"{save_path}/optimizer.pth")
        if self.lr_scheduler:
            torch.save(
                self.lr_scheduler.state_dict(),
                f"{save_path}/lr_scheduler.pth",
            )
        torch.save(
            {"step": self.step, "losses": self.losses, "epoch": self.epoch},
            f"{save_path}/trainer_state.pth",
        )

    def load_checkpoint(self, checkpoint_path):
        trainer_state_dict: dict = torch.load(f"{checkpoint_path}/trainer_state.pth")
        self.step = trainer_state_dict.get("step", 0)
        self.losses = trainer_state_dict.get("losses", [])
        self.epoch = trainer_state_dict.get("epoch", 0)

        self.model.load_state_dict(torch.load(f"{checkpoint_path}/model.pth"))
        self.optimizer.load_state_dict(torch.load(f"{checkpoint_path}/optimizer.pth"))
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(
                torch.load(f"{checkpoint_path}/lr_scheduler.pth")
            )

            self.lr_scheduler.T_max = self.epochs
            self.lr_scheduler.last_epoch = self.epoch

    def _unpack_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["image"].to(self.device), batch["mask"].to(self.device)


def main():
    trainer = MedSegTrainer()


if __name__ == "__main__":
    main()
