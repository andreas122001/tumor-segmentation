from cProfile import label

import ipywidgets as widgets
import monai
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from dataset import HNTSDataset

dataset = HNTSDataset("data/train", monai.transforms.Compose([
    monai.transforms.ScaleIntensityd(keys='image')
]))

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.tight_layout()

sample_idx = 10
sample = dataset[sample_idx]
image = monai.transforms.Orientation(axcodes="SPL")(sample['image'])
mask = monai.transforms.Orientation(axcodes="SPL")(sample['mask'])


def update_slice(slice_idx):
    slice_idx = int(slice_idx)
    image_slice = image[0, slice_idx]
    label0 = mask[0, slice_idx]
    label1 = mask[1, slice_idx]

    ax.clear()
    ax.imshow(image_slice, cmap="gray", alpha=1.0)
    ax.imshow(label0, cmap="Reds", alpha=0.3)
    ax.imshow(label1, cmap="plasma", alpha=0.3)
    ax.axis("off")


slice_ax = plt.axes((0.25, 0.9, 0.65, 0.03))
slice_slider = Slider(ax=slice_ax, label="Slice index", valmin=0, valmax=image.shape[1] - 1, valstep=1)
slice_slider.on_changed(update_slice)
update_slice(0)
plt.show()
