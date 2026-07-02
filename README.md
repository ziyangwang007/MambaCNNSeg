# When Mamba Meets with CNN

PyTorch implementation of **When Mamba Meets with CNN: Hybrid Dual-Encoding Network with Adaptive Graph-aware and Cross-Reverse Fusion for Medical Image Segmentation**.

This repository implements a hybrid U-shaped medical image segmentation network that combines a Visual Mamba encoder for long-range contextual modeling with a ConvNeXt encoder for local detail extraction. The model further introduces adaptive graph-aware feature perception and cross-reverse fusion to reduce redundancy, improve feature alignment, and preserve complementary global-local information.

## Highlights

- Hybrid dual-encoder architecture with Visual Mamba and ConvNeXt branches.
- Adaptive graph-aware perception module for high-confidence feature-node selection and graph-space correlation modeling.
- Cross-reverse scaling fusion for complementary global-local feature aggregation.
- Supports experiments on GlaS, ISIC-2017, and ACDC medical image segmentation datasets.
- Includes multiple baseline models for comparison, including UNet, Attention UNet, H2Former, TransUNet, UDTransNet, CSWin-UNet, VMUNet, Mamba-UNet, U-KAN, CMUNeXt, AMSUNet, and MBSNet.

## Method Overview

The proposed model contains four main components:

1. **ConvNeXt-based local encoder**: extracts fine-grained spatial and boundary features.
2. **Visual Mamba-based global encoder and decoder**: models long-range dependencies using state-space visual modeling while keeping linear computational complexity.
3. **Adaptive Graph-Aware Perception Module**: selects informative spatial feature nodes from CNN and Mamba feature streams and learns their correlations in graph space.
4. **Cross-Reverse Scaling Fusion Module**: fuses heterogeneous features by emphasizing information that is underrepresented in the other stream.

In the current codebase, the proposed model is instantiated mainly through:

```text
models/Mamba_CNN.py        # MBUnet wrapper
models/cnn_mamba_parts.py  # Visual Mamba backbone loader
models/mamba_sys_new.py    # VSSM encoder-decoder with fusion blocks
models/mokuai.py           # fusion and graph-related modules
```

The training scripts currently use `MCFusionNet_wo_reverse_scale_fusion` as the model name for the proposed network. In the 2D training script, `MCFusionNet` and `MCFusionNet_wo_reverse_scale_fusion` both instantiate `MBUnet`.


## Repository Structure

```text
.
├── Config.py
├── Load_Dataset.py
├── train_2Dkfold.py
├── test_2Dkfold.py
├── train_ACDC.py
├── test_ACDC.py
├── train_one_epoch.py
├── utils.py
├── models/
│   ├── Mamba_CNN.py
│   ├── cnn_mamba_parts.py
│   ├── mamba_sys_new.py
│   ├── mokuai.py
│   └── ...
├── nets/
├── datasets/          # not tracked; download from Google Drive
└── pretrained_ckpt/   # not tracked; download from Google Drive
```

Note: `train_ACDC.py` and `test_ACDC.py` import `utils_acdc`. Please make sure the `utils_acdc/` folder is included in the repository if you want to run ACDC experiments.

## Installation

### Option 1: Conda environment

```bash
conda env create -f environment.yml
conda activate mamba
```

The provided environment is based on Python 3.9, CUDA 11.8, PyTorch 2.0.1, `mamba-ssm`, and `causal-conv1d`.

### Option 2: pip requirements

```bash
pip install -r requirements.txt
```

For Mamba-based models, installing `mamba-ssm` and `causal-conv1d` on a Linux machine with an NVIDIA GPU is recommended.

## Data and Checkpoints

The `datasets/` and `pretrained_ckpt/` folders are not included in the GitHub repository because of file size. Download them from Google Drive and place them at the repository root.

[Download datasets and pretrained model](https://drive.google.com/file/d/12bSen6fOsaQZDX5_g46rJhCCu7eH9Ium/view?usp=sharing) |

Expected placement:

```text
.
├── datasets/
└── pretrained_ckpt/
    └── vmamba_small_e238_ema.pth
```

The Visual Mamba backbone loader expects:

```text
pretrained_ckpt/vmamba_small_e238_ema.pth
```

If your downloaded package also contains trained segmentation checkpoints, keep the extracted relative paths unchanged or update the corresponding paths in the test scripts.

## Dataset Preparation

### GlaS

The 2D data loader expects the following structure:

```text
datasets/GlaS/
├── Train_Folder/
│   ├── img/
│   └── labelcol/
└── Test_Folder/
    ├── img/
    └── labelcol/
```

For GlaS, the mask file should share the image stem and use `.png` format.

### ISIC-2017

The code expects images and masks in `img/` and `labelcol/` folders. ISIC-2017 masks are expected to follow the naming format:

```text
<image_name>_segmentation.png
```

A suggested structure is:

```text
datasets/ISIC17/
├── train/
│   ├── img/
│   └── labelcol/
└── val/
    ├── img/
    └── labelcol/
```

The current `train_2Dkfold.py` contains local absolute paths for ISIC-2017. Before running ISIC experiments, update these paths in `train_2Dkfold.py` and `Config.py`, or create symlinks to match your local environment.

### ACDC

For ACDC, keep the directory structure provided in the Google Drive package. The training and testing scripts receive the dataset paths through:

```text
--root_dir
--volume_path
--list_dir
```

Example:

```bash
python train_ACDC.py \
  --model_name MCFusionNet_wo_reverse_scale_fusion \
  --root_dir datasets/ACDC/ \
  --volume_path datasets/ACDC/test \
  --list_dir datasets/ACDC/lists_ACDC
```

## Training

### 2D segmentation: GlaS

```bash
CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py \
  --task_name GlaS \
  --model_name MCFusionNet_wo_reverse_scale_fusion
```

Training outputs are saved to:

```text
GlaS_kfold/<model_name>/<session_name>/
```

Each fold checkpoint is saved under:

```text
GlaS_kfold/<model_name>/<session_name>/models/fold_<id>/best_model-<model_name>.pth.tar
```

### 2D segmentation: ISIC-2017

After updating the ISIC paths:

```bash
CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py \
  --task_name ISIC17 \
  --model_name MCFusionNet_wo_reverse_scale_fusion
```

### 3D segmentation: ACDC

```bash
CUDA_VISIBLE_DEVICES=0 python train_ACDC.py \
  --model_name MCFusionNet_wo_reverse_scale_fusion \
  --root_dir datasets/ACDC/ \
  --volume_path datasets/ACDC/test \
  --list_dir datasets/ACDC/lists_ACDC
```

ACDC checkpoints are saved under:

```text
model_pth/ACDC/<experiment_name>/
```

## Testing

### 2D testing

Before testing, set the correct values in `Config.py`:

```python
task_name = 'GlaS'
model_name = 'MCFusionNet_wo_reverse_scale_fusion'
test_session = 'Test_session_MM.DD_HHhMM'
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python test_2Dkfold.py
```

The script loads five fold checkpoints and reports Dice, IoU, accuracy, sensitivity, and MAE. Visualization results are saved to:

```text
<task_name>_visualize_test/
```

### ACDC testing

```bash
CUDA_VISIBLE_DEVICES=0 python test_ACDC.py \
  --model_name MCFusionNet_wo_reverse_scale_fusion \
  --root_dir datasets/ACDC/ \
  --volume_path datasets/ACDC/test \
  --list_dir datasets/ACDC/lists_ACDC
```

The script reports mean Dice, HD95, Jaccard, and ASD.

## Supported Model Names

The training scripts include several model options:

```text
UNet
R34_UNet
MCFusionNet
MCFusionNet_wo_reverse_scale_fusion
MBSNet
CMUNeXt
AMSUnet
AttUNet
H2Former
TransUNet
UKAN
UDTransNet
vmunet
cswin_unet
MambaUnet
```

For the proposed method, use:

```text
MCFusionNet_wo_reverse_scale_fusion
```

## Reproducibility Notes

- Default image size: 224 × 224.
- Default seed: 666 for 2D experiments and 2024 for ACDC experiments.
- GlaS uses learning rate `1e-3` and batch size `4`.
- ISIC-2017 uses learning rate `1e-4` and batch size `20`.
- ACDC uses learning rate `1e-4` and batch size `12`.
- 2D experiments use five-fold cross-validation in `train_2Dkfold.py`.
- ACDC uses a 70/10/20 train/validation/test split as described in the manuscript.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@misc{wang2025mamba_cnn_fusion,
  title  = {When Mamba Meets with CNN: Hybrid Dual-Encoding Network with Adaptive Graph-aware and Cross-Reverse Fusion for Medical Image Segmentation},
  author = {Wang, Ziyang and Zhang, Zhuo and Ma, Siyuan},
  year   = {2025}
}
```

Please update the BibTeX entry with the final venue, volume, pages, and DOI once available.

## Acknowledgements

This repository includes implementations or adapted components from several open-source medical image segmentation projects and baseline models. Please also cite the corresponding original papers and repositories when using those components.

## Contact

For questions, please contact:

```text
Ziyang Wang
School of Computer Science and Digital Technologies, Aston University
Email: z.wang47@aston.ac.uk
Website: https://www.zywang.site/
```
