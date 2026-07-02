#! bin/bash
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name UNet && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name R34_UNet && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name CMUNeXt && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name AMSUnet && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name H2Former && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name TransUNet && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name UDTransNet && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name AttUNet && \
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name vmunet && \ 
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name UKAN && \ 
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name cswin_unet  
# CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name ISIC17 --model_name MambaUnet 
CUDA_VISIBLE_DEVICES=0 python train_2Dkfold.py --task_name GlaS --model_name MCFusionNet_wo_reverse_scale_fusion

