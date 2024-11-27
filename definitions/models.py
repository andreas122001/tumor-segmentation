import monai


models = {
    "UNet": (
        monai.networks.nets.UNet,
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,
            "channels": (32, 64, 128, 256),
            "strides": (2, 2, 2),
            "num_res_units": 2,
        },
    ),
    "SegResNet": (
        monai.networks.nets.SegResNet,
        {
            "blocks_down": [2, 4, 4, 4],
            "blocks_up": [2, 4, 4],
            "init_filters": 16,
            "in_channels": 1,
            "out_channels": 2,
            "dropout_prob": 0.2,
        },
    ),
    "UNETR": (
        monai.networks.nets.UNETR,
        {
            "in_channels": 1,
            "out_channels": 2,
            "img_size": (224, 224, 96),
            "feature_size": 16,
            "hidden_size": 768,
            "num_heads": 16,
            "proj_type": "conv",
            "norm_name": "instance",
        },
    ),
    "SwinUNETR": (
        monai.networks.nets.SwinUNETR,
        {
            "in_channels": 1,
            "out_channels": 2,
            "img_size": (224, 224, 96),
            "spatial_dims": 3,
            "use_checkpoint": True,
            "use_v2": False,
            "num_heads": (3, 6, 12, 24),
            "drop_rate": 0.2
        },
    ),
}

def get_model(option):
    model_class, model_args = models[option]
    return model_class(**model_args)
