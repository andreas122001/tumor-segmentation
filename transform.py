from monai.transforms import (
    Compose,
    LoadImaged,
    SelectItemsd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Resized,
    RandRotate90d,
    ToDeviced
)


def default_train_transform():
    return Compose(
        [
            # Load data
            LoadImaged(keys=["image", "mask"]),
            SelectItemsd(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Normalization and resizing
            NormalizeIntensityd(keys="image"),
            Resized(
                keys=["image", "mask"],
                spatial_size=(256, 256, 80),
                mode=("bilinear", "nearest"),
            ),
            RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=(0, 1)),
            ToDeviced(keys=["image", "mask"], device="cuda:0"),
        ]
    )


def default_transform():
    return Compose(
        [
            # Load data
            LoadImaged(keys=["image", "mask"]),
            SelectItemsd(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Normalization
            NormalizeIntensityd(keys="image"),
            # Ensure correct size
            Resized(
                keys=["image", "mask"],
                spatial_size=(256, 256, 80),
                mode=("bilinear", "nearest"),
            ),
        ]
    )
