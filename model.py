import torch


class DummySegmentation(torch.Module):
    def forward(x):
        return x
