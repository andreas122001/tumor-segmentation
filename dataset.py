import monai.data
import os

from monai.data import Dataset


# TODO: very hardcoded and not generic, but works for now (fix later tho)
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
                    "id": root.split("\\")[2],
                }
            )

    return data_dicts


class HNTSDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        index = create_dataset_dicts(root)
        self.data = Dataset(index, )
        self.load_transform = Compose([
        # Load data
        LoadImaged(keys=["image", "mask"]),
        SelectItemsd(keys=["image", "mask"]),
        ConvertLabelIdToChannel(keys="mask"),
        EnsureChannelFirstd(keys=["image"]),
    ])

    def __getitem__(self, item):
        ...