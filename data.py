import os


# TODO: very hardcoded and not generic, but works for now (fix later tho)
def create_dataset_dicts(root_data) -> dict[str, str]:
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


if __name__ == "__main__":
    root_data = "data/train"
    data_dicts = create_dataset_dicts(root_data)
    print(data_dicts)
