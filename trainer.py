import monai
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


class VoidWriter(SummaryWriter):
    def __init__(self):
        super().__init__()


class MedSegTrainer:

    def __init__(
            self,
            epochs: int,
            loss_f: torch.nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            device="cuda",
            logger: SummaryWriter = VoidWriter()
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.lr_scheduler = lr_scheduler
        self.step = 0
        self.logger = logger
        self.device = device

    def _unpack_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["image"].to(self.device), batch["mask"].to(self.device)

    def fit(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader):
        model.train()
        for e in range(self.epochs):
            print(f"Epoch {e}/{self.epochs}")
            for batch in tqdm(train_loader, desc="Training step", leave=False):
                loss = self.training_step(model, batch)
                self.logger.add_scalar("training_loss", loss, self.step)
            self.lr_scheduler.step()
            self.logger.add_scalar("learning_rate", self.lr_scheduler.get_last_lr(), self.step)

    def training_step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        inputs, targets = self._unpack_batch(batch)

        logits = model(inputs)
        loss = self.loss_f(logits, targets)
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return loss.item()

    def test(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
             metric: monai.metrics.CumulativeIterationMetric,
             post_trans: monai.transforms.compose.Compose):
        model.to(self.device)
        for batch in tqdm(test_loader, desc="Testing steps"):
            inputs, targets = self._unpack_batch(batch)

            logits = model(inputs)
            preds = post_trans(logits)
            metric(y_pred=preds, y=targets)

        score = metric.aggregate()
        metric.reset()
        return score


def main():
    trainer = MedSegTrainer()


if __name__ == "__main__":
    main()
