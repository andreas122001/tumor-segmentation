import os

import torch
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    MapTransform,
    LoadImaged,
    SelectItemsd,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Identity,
)


def create_dataset_dicts(root_data, include_midRT=False) -> list[dict[str, str]]:
    """Construct the dataset dicts for the MONAI dataset"""
    data_dicts = []
    for root, _, files in sorted(os.walk(root_data)):

        # This is a folder with files (duh)
        if len(files) > 0 and ("preRT" in root or (include_midRT and "midRT" in root)):
            mask = [f for f in files if "_mask.nii.gz" in f][0]
            imag = [f for f in files if "_T2.nii.gz" in f][0]
            data_dicts.append(
                {
                    "image": str(os.path.join(root, imag)),
                    "mask": str(os.path.join(root, mask)),
                    "id": str(root.split("/")[-2]),
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
            result = [d[key] == 0, d[key] == 1, d[key] == 2]
            d[key] = torch.stack(result, dim=0).float()
        return d


class HNTSDataset(CacheDataset):
    """
    Loads samples from the HNTSMRG24 dataset. The samples are converted such that each label becomes its own channel,
    except from the background label (0) which is ignored. GTVp is at channel 0, GTVn at channel 1.
    """

    def __init__(self, root, transform=None, include_midRT=False):
        self.index = create_dataset_dicts(root, include_midRT)
        full_transform = Compose(
            [
                LoadImaged(keys=["image", "mask"]),
                ConvertLabelIdToChannel(keys="mask"),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image", "mask"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "mask"],
                    pixdim=(1.0, 1.0, 1.0),  # from image metadata, isotropic spacing
                    mode=("bilinear", "nearest"),
                ),
                Compose(
                    transform if transform else Identity()
                ),  # add the optional transform (if exists)
            ]
        )
        self.root = root  # root path
        self.id_to_label = {  # dataset labels in text
            0: "GTVp",
            1: "GTVn",
        }

        # Apply the super-class functionality
        super().__init__(self.index, full_transform, num_workers=None)

    def __len__(self):
        return self.index.__len__()


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = HNTSDataset("data/test")
    for i in tqdm(range(len(dataset))):
        _ = dataset[i]
