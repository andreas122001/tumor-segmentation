import gzip
from monai.data import Dataset
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
from monai.data import PILReader
from monai.config import print_config
import monai
import matplotlib.pyplot as plt
import monai.transforms
import monai.visualize

filename = "data/test/29/preRT/29_preRT_T2.nii.gz"
filename_mask = "data/test/29/preRT/29_preRT_mask.nii.gz"

data = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(filename)
mask = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
    filename_mask
)
print(f"image data shape: {data.shape}")
print(f"meta data: {data.meta.keys()}")
mask = mask > 0

masked_data = data + mask * 255

monai.visualize.matshow3d(monai.transforms.Orientation("SPL")(masked_data), every_n=5)
plt.show()
