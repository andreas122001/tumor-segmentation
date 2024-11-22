from typing import Any
from monai.data import DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
from torch.nn.modules.module import T
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class VoidWriter(SummaryWriter):
    def __init__():
        ...


class MedSegTrainer:

    def __init__(
        self,
        epochs,
        loss_f,
        optimizer,
        lr_scheduler=None,
        logger=VoidWriter()
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.lr_scheduler = lr_scheduler
        self.step = 0

    @staticmethod
    def _unpack_batch(batch):
        return batch["image"], batch["mask"]

    def fit(self, model, train_loader):
        for e in range(self.epochs):
            print(f"Epoch {e}/{self.epochs}")
            for batch in tqdm(train_loader, desc="Training step", leave=False):
                loss = self.training_step(model, batch)
            self.lr_scheduler.step()

    def training_step(self, model, batch):
        self.optimizer.zero_grad()
        inputs, targets = self._unpack_batch(batch)

        logits = model(inputs)
        loss = self.loss_f(logits, targets)
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return loss.item()


def main():
    trainer = MedSegTrainer()


if __name__ == "__main__":
    main()
