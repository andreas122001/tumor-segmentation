**Training Dataset for HNTSMRG 2024 Challenge**

&nbsp;

**Overview**

This repository houses the publicly available training dataset for the [Head and Neck Tumor Segmentation for MR-Guided Applications (HNTSMRG) 2024 Challenge](https://hntsmrg24.grand-challenge.org/).

Patient cohorts correspond to patients with histologically proven head and neck cancer who underwent radiotherapy (RT) at The University of Texas MD Anderson Cancer Center. The cancer types are predominately oropharyngeal cancer or cancer of unknown primary. Images include a pre-RT T2w MRI scan (1-3 weeks before start of RT) and a mid-RT T2w MRI scan (2-4 weeks intra-RT) for each patient. Segmentation masks of primary gross tumor volumes (_abbreviated GTVp_) and involved metastatic lymph nodes (_abbreviated GTVn_) are provided for each image ([derived from multi-observer STAPLE consensus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1283110/)).

HNTSMRG 2024 is split into 2 tasks:

- **Task 1**: Segmentation of tumor volumes (GTVp and GTVn) on **pre-RT MRI**.
- **Task 2**: Segmentation of tumor volumes (GTVp and GTVn) on **mid-RT MRI**.

The same patient cases will be used for the training and test sets of both tasks of this challenge. Therefore, we are releasing a single training dataset that can be used to construct solutions for either segmentation task. The test data provided (via Docker containers), however, will be different for the two tasks. Please consult the [challenge website for more details](https://hntsmrg24.grand-challenge.org/dataset/).

&nbsp;

**Data Details**

- DICOM files (images and structure files) have been converted to NIfTI format (_.nii.gz_) for ease of use by participants via [DICOMRTTool](https://pubmed.ncbi.nlm.nih.gov/33607331/) v. 1.0.
- Images are a mix of fat-suppressed and non-fat-suppressed MRI sequences. Pre-RT and mid-RT image pairs for a given patient are consistently either fat-suppressed or non-fat-suppressed.
- Though some sequences may appear to be contrast enhancing, _no exogenous contrast is used_.
- All images have been manually cropped from the top of the clavicles to the bottom of the nasal septum (~ oropharynx region to shoulders), allowing for more consistent image field of views and removal of identifiable facial structures.
- The mask files have one of three possible values: **_background = 0, GTVp = 1, GTVn = 2_** (in the case of multiple lymph nodes, they are concatenated into one single label). This labeling convention is [similar to the 2022 HECKTOR Challenge](https://hecktor.grand-challenge.org/Data/).
- **150 unique patients** are included in this dataset. Anonymized patient numeric identifiers are utilized.
- The entire training dataset is **~15 GB**.

&nbsp;

**Dataset Folder/File Structure**

The dataset is uploaded as a ZIP archive. Please unzip before use. NIfTI files conform to the following standardized nomenclature: _ID_timepoint_image/mask.nii.gz_. For mid-RT files, a "registered" suffix (_ID_timepoint_image/mask_registered.nii.gz_) indicates the image or mask has been registered to the mid-RT image space (_see more details in Additional Notes below_).

The data is provided with the following folder hierarchy:

- **_Top-level folder_** _(named "HNTSMRG24_train")_
  - **_Patient-level folder_** _(anonymized patient ID, example: "2")_
    - **_Pre-radiotherapy data folder_** _("preRT")_
      - Original pre-RT T2w MRI volume (_example: "2_preRT_T2.nii.gz"_).
      - Original pre-RT tumor segmentation mask (_example: "2_preRT_mask.nii.gz"_).
    - **_Mid-radiotherapy data folder_** _("midRT")_
      - Original mid-RT T2w MRI volume (_example: "2_midRT_T2.nii.gz"_).
      - Original mid-RT tumor segmentation mask (_example: "2_midRT_mask.nii.gz"_).
      - Registered pre-RT T2w MRI volume (_example: "2_preRT_T2_registered.nii.gz"_).
      - Registered pre-RT tumor segmentation mask (_example: "2_preRT_mask_registered.nii.gz"_).

&nbsp;

_Note_: Cases will exhibit variable presentation of ground truth mask structures. For example, a case could have only a GTVp label present, only a GTVn label present, both GTVp and GTVn labels present, or a completely empty mask (i.e., complete tumor response at mid-RT). **The following case IDs have empty masks at mid-RT (indicating a complete response): 21, 25, 29, 42.** These empty masks are _not errors_. There will similarly be some cases in the test set for Task 2 that have empty masks.

&nbsp;

**Details Relevant for Algorithm Building**

- The goal of Task 1 is to generate a pre-RT tumor segmentation mask (e.g., "_2_preRT_mask.nii.gz"_ is the relevant label). During blind testing for Task 1, only the pre-RT MRI (e.g., "_2_preRT_T2.nii.gz_") will be provided to the participants algorithms.
- The goal of Task 2 is to generate a mid-RT segmentation mask (e.g., "_2_midRT_mask.nii.gz"_ is the relevant label). During blind testing for Task 2, the mid-RT MRI (e.g., "_2_midRT_T2.nii.gz_"), registered pre-RT MRI (e.g., "_2_preRT_T2_registered.nii.gz_"), and registered pre-RT tumor segmentation mask (e.g., "_2_preRT_mask_registered.nii.gz_"), will be provided to the participants algorithms.
- When building models, the resolution of the generated prediction masks should be the same as the corresponding MRI for the given task. In other words, the generated masks should be in the correct pixel spacing and origin with respect to the original reference frame (i.e., pre-RT image for Task 1, mid-RT image for Task 2). More details on the submission of models will be located on the [challenge website](https://hntsmrg24.grand-challenge.org/submission-instructions/).

&nbsp;

**Additional Notes**

1. _General notes._
    1. NIfTI format images and segmentations may be easily visualized in any NIfTI viewing software such as [3D Slicer](https://www.slicer.org/).
    2. **Test data will not be made public until the completion of the challenge**. The complete training and test data will be published together (along with all original multi-observer annotations and relevant clinical data) at a later date via The Cancer Imaging Archive. Expected date ~ Spring 2024.
2. _Task 1 related notes._
    1. When training their algorithms for Task 1, participants can choose to use only pre-RT data or add in mid-RT data as well. Initially, our plan was to limit participants to utilizing only pre-RT data for training their algorithms in Task 1. However, upon reflection, we recognized that in a practical setting, individuals aiming to develop auto-segmentation algorithms could theoretically train models using any accessible data at their disposal. Based on current literature, we actually don't know what the best solution would be! Would the incorporation of mid-RT data for training a pre-RT segmentation model actually be helpful, or would it merely introduce harmful noise? The answer remains unclear. Therefore, we leave this choice to the participants.
    2. _Remember, though, during testing, you will ONLY have the pre-RT image as an input to your model_ (naturally, since Task 1 is a pre-RT segmentation task and you won't know what mid-RT data for a patient will look like).
3. _Task 2 related notes._
    1. In addition to the mid-RT MRI and segmentation mask, we have also provided a registered pre-RT MRI and the corresponding registered pre-RT segmentation mask for each patient. We offer this data for participants who opt not to integrate any image registration techniques into their algorithms for Task 2 but still wish to use the two images as a joint input to their model. Moreover, in a real-world adaptive RT context, such registered scans are typically readily accessible. Naturally, participants are also free to incorporate their own image registration processes into their pipelines if they wish (or ignore the pre-RT images/masks altogether).
        1. Registrations were generated using [SimpleITK](https://pypi.org/project/SimpleITK/), where the mid-RT image serves as the fixed image and the pre-RT image serves as the moving image. Specifically, we utilized the following steps: 1. Apply a centered transformation, 2. Apply a rigid transformation, 3. Apply a deformable transformation with Elastix using a preset parameter map ([Parameter map 23 in the Elastix Zoo](https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/Par0023)). This particular deformable transformation was selected as it is open-source and was benchmarked in a previous similar application ([https://doi.org/10.1002/mp.16128](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16128)). For cases where excessive warping was noted during deformable registration (a small minority of cases), only the rigid transformation was applied.

&nbsp;

**Contact**

We have set up a general email address that you can message to notify all organizers at: [hntsmrg2024@gmail.com](mailto:hntsmrg2024@gmail.com).  Additional specific organizer contacts:

- Kareem A. Wahid, PhD (_kawahid@mdanderson.org_)
- Cem Dede, MD (_cdede@mdanderson.org_)
- Mohamed A. Naser, PhD (_manaser@mdanderson.org_)