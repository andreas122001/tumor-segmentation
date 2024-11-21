from typing import Any
from monai.data import DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
from torch.nn.modules.module import T


class MedSegTrainer:

    def __init__(self,
                 epochs,
                 loss_f,
                 optimizer,
                 lr_scheduler=None,
                 ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.lr_scheduler = lr_scheduler
        self.step = 0

    @staticmethod
    def _unpack_batch(batch):
        return batch['image'], batch['mask']

    def fit(self, model, train_loader):
        ...

    def training_step(self, batch, batch_idx, dataloader_idx):
        inputs, targets = self._unpack_batch(batch)

        logits = self.forward(inputs)
        loss = self.loss_f(logits, targets)

        # Logging
        self.log("loss", loss.item())
        self.step += 1

        return loss.item()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def validation_step(
            self, batch, batch_idx, dataloader_idx
    ) -> torch.Tensor | np.Mapping[str, Any] | None:
        return super().validation_step(batch, batch_idx, dataloader_idx)


def main():
    trainer = MedSegTrainer()

    trainer.fit


if __name__ == '__main__':
    main()
