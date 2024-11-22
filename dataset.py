import monai.data
import os
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    LoadImaged,
    SelectItemsd,
    EnsureChannelFirstd,
)

from monai.data import Dataset


def create_dataset_dicts(root_data) -> list[dict[str, str]]:
    data_dicts = []
    for root, _, files in sorted(os.walk(root_data)):

        # This is a folder with files (duh)
        if "preRT" in root and len(files) > 0:
            mask = [f for f in files if "_preRT_mask.nii.gz" in f][0]
            imag = [f for f in files if "_preRT_T2.nii.gz" in f][0]
            data_dicts.append(
                {
                    "image": os.path.join(root, imag),
                    "mask": os.path.join(root, mask),
                }
            )

    return data_dicts


class ConvertLabelIdToChannel(MapTransform):
    """
    Transform to separate the labels into each of their own channel
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


class HNTSDataset(Dataset):
    """
    Loads samples from the HNTSMRG24 dataset.
    The samples are converted such that each label becomes its own channel, except from the background label (0) which is ignored.
    GTVp is at channel 0, GTVn at channel 1.
    """

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.index = create_dataset_dicts(root)
        self.root = root
        self.id_to_label = {
            0: "GTVp",
            1: "GTVn",
        }
        self.transform = transform
        self.load_transform = Compose(
            [
                LoadImaged(keys=["image", "mask"]),
                SelectItemsd(keys=["image", "mask"]),
                ConvertLabelIdToChannel(keys="mask"),
                EnsureChannelFirstd(keys=["image"]),
            ]
        )

    def __getitem__(self, i):
        data = self.load_transform(self.index[i])
        if self.transform:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    dataset = HNTSDataset("data/train")
    print(dataset[0])
